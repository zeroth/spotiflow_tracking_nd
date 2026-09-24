from __future__ import annotations

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
                tifffile.imwrite(str(self._mask_output_dir / path.name), mask)

            self.progress.emit(t + 1, total)

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
