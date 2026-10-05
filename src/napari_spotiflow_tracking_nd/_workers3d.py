from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from qtpy.QtCore import QThread, Signal

from napari_spotiflow_tracking_nd._fitting3d import (
    DEFAULT_N_SIGMA,
    DEFAULT_PATCH_RADIUS_ZYX,
    fit_spots_3d,
    paint_label_mask,
)
from napari_spotiflow_tracking_nd._segmentation3d import detect_spots_3d
from napari_spotiflow_tracking_nd._track_analysis3d import (
    MSD_COLUMNS,
    PROPERTY_COLUMNS,
    STEP_COLUMNS,
    AnalysisParams,
    analyse_track,
    lifetime_distribution,
)
from napari_spotiflow_tracking_nd._track_analysis3d import (
    missing_columns as missing_track_columns,
)
from napari_spotiflow_tracking_nd._tracking3d import (
    add_physical_columns,
    choose_link_coordinates,
    filter_short_tracks,
    flag_censored,
    iter_link,
    missing_columns,
)


class VolumeOpWorker3D(QThread):
    """Stream a pre-processing op over a folder of per-timepoint TIFFs,
    one volume at a time: read the input file, apply `op`, write the
    result to `output_dir` under the same filename.

    Cancellation (`cancel()`) is checked between files — the file
    currently being written always finishes, so output is never a
    half-written TIFF.
    """

    progress = Signal(int, int)  # (current, total), 1-indexed current
    # Named `run_finished`, not `finished`: `QThread` already defines a
    # no-argument `finished` signal, and overriding it with a different
    # signature would collide the moment anything calls `start()` instead
    # of `run()`. (`progress`/`errored` are not `QThread` built-ins.)
    run_finished = Signal(list)  # output paths, as str, in input order
    errored = Signal(str)

    def __init__(self, files: list[Path], output_dir: Path, op, op_kwargs: dict, parent=None):
        super().__init__(parent)
        self._files = list(files)
        self._output_dir = Path(output_dir)
        self._op = op
        self._op_kwargs = dict(op_kwargs)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            written = []
            total = len(self._files)
            for i, path in enumerate(self._files):
                if self._cancelled:
                    break
                volume = tifffile.imread(str(path))
                result = self._op(volume, **self._op_kwargs)
                out_path = self._output_dir / path.name
                tifffile.imwrite(str(out_path), result)
                written.append(str(out_path))
                self.progress.emit(i + 1, total)
            self.run_finished.emit(written)
        except Exception as exc:
            self.errored.emit(str(exc))


class DetectionWorker3D(QThread):
    """Stream Spotiflow 3D detection over a folder of per-timepoint TIFFs,
    appending each timepoint's spots to a CSV as they're found so a long
    run's progress survives a cancel or a crash.
    """

    progress = Signal(int, int)  # (current, total), 1-indexed current
    # `run_finished`, not `finished` — see `VolumeOpWorker3D` for why.
    run_finished = Signal(str)   # the CSV path written
    errored = Signal(str)

    def __init__(
        self,
        files: list[Path],
        model,  # a loaded model, or a zero-arg callable returning one
        csv_path: Path,
        prob_thresh: float | None = 0.5,
        min_distance: int = 2,
        fit: bool = False,
        patch_radius_zyx=DEFAULT_PATCH_RADIUS_ZYX,
        mask_output_dir: Path | None = None,
        max_workers: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._files = list(files)
        self._model = model
        self._csv_path = Path(csv_path)
        self._prob_thresh = prob_thresh
        self._min_distance = min_distance
        self._fit = fit
        self._patch_radius_zyx = tuple(patch_radius_zyx)
        self._mask_output_dir = Path(mask_output_dir) if mask_output_dir else None
        self._max_workers = max_workers
        self._mask_dtype = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            # `model` may be a zero-arg callable so that a slow first
            # weight load happens here, on the worker thread, rather than
            # freezing the UI before this thread even starts. Resolved
            # once, and a failure lands on `errored` like any other.
            if callable(self._model) and not hasattr(self._model, "predict"):
                self._model = self._model()
            if self._mask_output_dir is not None and not self._fit:
                # The mask is painted from the fits: without them every
                # volume would be written empty, one full-size file per
                # timepoint, with nothing to show it had gone wrong.
                raise ValueError(
                    "Label-mask export needs Gaussian fitting turned on — "
                    "the masks are painted from the fits."
                )
            if self._csv_path.exists():
                self._csv_path.unlink()
            if self._mask_output_dir is not None:
                self._mask_output_dir.mkdir(parents=True, exist_ok=True)

            # One pool for the whole run: Windows spawns rather than forks,
            # so a pool per timepoint would spend ~0.5-1 s per process
            # starting up, 349 times over.
            pool = None
            if self._fit and self._max_workers != 1:
                pool = ProcessPoolExecutor(max_workers=self._max_workers)
            try:
                self._detect_all(pool)
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)
            self.run_finished.emit(str(self._csv_path))
        except Exception as exc:
            self.errored.emit(str(exc))

    def _detect_all(self, pool):
        header_written = False
        written_masks = []
        total = len(self._files)
        for t, path in enumerate(self._files):
            if self._cancelled:
                break
            volume = tifffile.imread(str(path))
            points, details = detect_spots_3d(
                volume, self._model,
                prob_thresh=self._prob_thresh,
                min_distance=self._min_distance,
            )

            fits = None
            if self._fit:
                fits = fit_spots_3d(
                    volume, points,
                    patch_radius_zyx=self._patch_radius_zyx,
                    pool=pool,
                )

            if len(points) > 0:
                df = self._rows_for(t, points, details, fits)
                df.to_csv(self._csv_path, mode="a", header=not header_written, index=False)
                header_written = True

            if self._mask_output_dir is not None:
                # Written even for a timepoint with no spots, so the mask
                # series stays aligned 1:1 with the input series.
                mask = paint_label_mask(volume.shape, fits or [], n_sigma=DEFAULT_N_SIGMA)
                mask = self._to_run_dtype(mask, written_masks)
                out = self._mask_output_dir / path.name
                tifffile.imwrite(str(out), mask)
                written_masks.append(out)

            self.progress.emit(t + 1, total)

    def _to_run_dtype(self, mask, written_masks):
        """Keep one dtype across the whole mask series.

        `paint_label_mask` widens to uint32 only when a timepoint has more
        than 65,535 spots, so a series that crosses that line partway
        through would be ragged — and `_io3d.build_lazy_stack_from_files`
        refuses a ragged series, making the plugin unable to open its own
        output. When a later timepoint forces the wider dtype, the masks
        already written are re-read and rewritten to match. That is slow,
        but it happens at most once in a run and only past 65,535 spots.
        """
        if self._mask_dtype is None or mask.dtype.itemsize > np.dtype(self._mask_dtype).itemsize:
            if self._mask_dtype is not None:
                for earlier in written_masks:
                    tifffile.imwrite(
                        str(earlier), tifffile.imread(str(earlier)).astype(mask.dtype),
                    )
            self._mask_dtype = mask.dtype
        return mask.astype(self._mask_dtype, copy=False)

    @staticmethod
    def _rows_for(t, points, details, fits):
        coords = np.asarray(points)
        # Built column-by-column, not as one `np.hstack` block: hstacking
        # the integer frame index with float coords and probabilities
        # promotes everything to float64, and the CSV's `frame` column
        # comes out as `0.0, 1.0, ...`.
        columns = {
            "frame": np.full(len(coords), t, dtype=np.int64),
            "z": coords[:, 0],
            "y": coords[:, 1],
            "x": coords[:, 2],
            "prob": np.asarray(details.prob).reshape(-1),
        }
        if fits is not None:
            # `z, y, x` stay as detected; the refined centre is separate,
            # so nothing downstream shifts when fitting is turned on.
            columns.update(
                fit_z=np.array([f.z0 for f in fits]),
                fit_y=np.array([f.y0 for f in fits]),
                fit_x=np.array([f.x0 for f in fits]),
                sigma_z=np.array([f.sigma_z for f in fits]),
                sigma_y=np.array([f.sigma_y for f in fits]),
                sigma_x=np.array([f.sigma_x for f in fits]),
                amplitude=np.array([f.amplitude for f in fits]),
                background=np.array([f.background for f in fits]),
                integrated_intensity=np.array([f.integrated_intensity for f in fits]),
                converged=np.array([f.converged for f in fits]),
            )
        return pd.DataFrame(columns)


class TrackingWorker3D(QThread):
    """Link a detection CSV into tracks, one frame at a time, and write
    the tracks CSV once linking is done.

    Written once at the end, not appended per frame like detection: the
    min-length filter and the `censored` flag both need each track whole.
    It goes to a `.partial` file first and is renamed into place, so a
    failed run never leaves a valid-looking half table — nor destroys a
    previous run's output under the same name.

    Cancellation (`cancel()`) is checked between frames. The frames linked
    so far are still filtered, flagged and written, with the last frame
    linked treated as the end of the series.
    """

    progress = Signal(int, int)  # (current, total), 1-indexed current
    # `run_finished`, not `finished` — see `VolumeOpWorker3D` for why.
    run_finished = Signal(str)   # the tracks CSV written, or "" if none
    errored = Signal(str)

    def __init__(
        self,
        csv_path: Path,
        output_path: Path,
        search_range_zyx: tuple[float, float, float],
        memory: int = 3,
        min_length: int = 1,
        use_fit: bool = True,
        adaptive: bool = False,
        calibration: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._csv_path = Path(csv_path)
        self._output_path = Path(output_path)
        self._search_range_zyx = tuple(search_range_zyx)
        self._memory = memory
        self._min_length = min_length
        self._use_fit = use_fit
        self._adaptive = adaptive
        self._calibration = dict(calibration or {})
        self._cancelled = False
        # Set by `run()` before `run_finished`: the series' first frame and
        # the last frame actually linked (earlier than the series' end when
        # cancelled), so the caller can record how far the run got.
        self.first_frame: int | None = None
        self.last_frame: int | None = None

    def cancel(self):
        self._cancelled = True

    def run(self):
        partial = self._output_path.with_name(self._output_path.name + ".partial")
        try:
            spots = pd.read_csv(self._csv_path)
            missing = missing_columns(spots)
            if missing:
                raise ValueError(
                    f"{self._csv_path.name} is not a detection CSV — missing "
                    f"column(s): {', '.join(missing)}."
                )
            if spots["frame"].nunique() < 2:
                raise ValueError(
                    "Spots are needed in at least two timepoints to link "
                    "anything into tracks."
                )
            spots = choose_link_coordinates(spots, use_fit=self._use_fit)

            first = int(spots["frame"].min())
            total = int(spots["frame"].max()) - first + 1
            linked, last_linked = [], None
            for t, frame_df in iter_link(
                spots, self._search_range_zyx, self._memory,
                adaptive=self._adaptive,
                should_stop=lambda: self._cancelled,
            ):
                if len(frame_df):
                    linked.append(frame_df)
                last_linked = t
                self.progress.emit(t - first + 1, total)

            self.first_frame, self.last_frame = first, last_linked

            if not linked:
                # Cancelled before a single spot was linked: nothing to
                # write, and the caller must not mistake an older file of
                # the same name for this run's output.
                self.run_finished.emit("")
                return

            tracks = pd.concat(linked, ignore_index=True)
            tracks = filter_short_tracks(tracks, self._min_length)
            tracks = flag_censored(tracks, first, last_linked, self._memory)
            tracks = add_physical_columns(tracks, self._calibration)
            tracks.to_csv(partial, index=False)
            os.replace(partial, self._output_path)
            self.run_finished.emit(str(self._output_path))
        except Exception as exc:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass  # best-effort cleanup must not swallow `errored`
            self.errored.emit(str(exc))


def replace_dir(src: Path, dst: Path) -> None:
    """Move the finished folder `src` to `dst`, replacing any earlier `dst`
    only once `src` is complete — so a failed run can never leave `dst`
    half old, half new."""
    old = dst.with_name(dst.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if dst.exists():
        os.replace(dst, old)
    os.replace(src, dst)
    shutil.rmtree(old, ignore_errors=True)


class TrackAnalysisWorker3D(QThread):
    """Analyse a tracks CSV one track at a time and write the results
    folder once every track is done.

    Cancellation (`cancel()`) is checked between tracks and writes
    *nothing*: unlike a cancelled tracking run, a partial analysis is a
    biased sample (the first tracks by ID), not a longer series. Output
    is assembled in `<folder>.partial` and swapped into place by
    `replace_dir`, so an error never touches an earlier results folder.
    """

    progress = Signal(int, int)  # (current, total), 1-indexed current
    # `run_finished`, not `finished` — see `VolumeOpWorker3D` for why.
    run_finished = Signal(str)   # the results folder, or "" if cancelled
    errored = Signal(str)

    def __init__(
        self,
        csv_path: Path,
        output_dir: Path,
        params: AnalysisParams,
        record: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._csv_path = Path(csv_path)
        self._output_dir = Path(output_dir)
        self._params = params
        self._record = dict(record or {})
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        partial = self._output_dir.with_name(self._output_dir.name + ".partial")
        try:
            tracks = pd.read_csv(self._csv_path)
            missing = missing_track_columns(tracks)
            if missing:
                raise ValueError(
                    f"{self._csv_path.name} is not a tracks CSV — missing "
                    f"column(s): {', '.join(missing)}."
                )
            if tracks.empty:
                raise ValueError(f"{self._csv_path.name} has no tracks.")

            groups = [group for _, group in tracks.groupby("particle", sort=True)]
            properties, msd_rows, step_rows = [], [], []
            for i, track in enumerate(groups):
                if self._cancelled:
                    self.run_finished.emit("")
                    return
                row, msd, steps = analyse_track(track, self._params)
                properties.append(row)
                msd_rows.extend(msd)
                step_rows.extend(steps)
                self.progress.emit(i + 1, len(groups))

            table = pd.DataFrame(properties, columns=PROPERTY_COLUMNS)
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(parents=True)
            table.to_csv(partial / "track_properties.csv", index=False)
            lifetime_distribution(table, self._params.frame_interval_s).to_csv(
                partial / "lifetime_distribution.csv", index=False
            )
            pd.DataFrame(msd_rows, columns=MSD_COLUMNS).to_csv(
                partial / "msd_curves.csv", index=False
            )
            pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(
                partial / "steps.csv", index=False
            )
            (partial / "analysis.json").write_text(
                json.dumps(self._record, indent=2), encoding="utf-8"
            )
            replace_dir(partial, self._output_dir)
            self.run_finished.emit(str(self._output_dir))
        except Exception as exc:
            shutil.rmtree(partial, ignore_errors=True)
            self.errored.emit(str(exc))


class FunctionWorker3D(QThread):
    """Run one callable on a background thread and hand back its result.

    The two streaming workers above own a loop and can stop between
    iterations. This one carries a single blocking call â€” a preview of one
    volume â€” so `cancel()` cannot interrupt it. Cancelling only suppresses
    the result; the thread still runs to completion in the background.
    """

    run_finished = Signal(object)  # whatever `fn` returned
    # `run_finished`, not `finished` â€” see `VolumeOpWorker3D` for why.
    errored = Signal(str)

    def __init__(self, fn, args=(), kwargs=None, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:
            if not self._cancelled:
                self.errored.emit(str(exc))
            return
        if not self._cancelled:
            self.run_finished.emit(result)
