from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from qtpy.QtCore import QThread, Signal

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
    finished = Signal(list)      # output paths, as str, in input order
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
            self.finished.emit(written)
        except Exception as exc:
            self.errored.emit(str(exc))


class DetectionWorker3D(QThread):
    """Stream Spotiflow 3D detection over a folder of per-timepoint TIFFs,
    appending each timepoint's spots to a CSV as they're found so a long
    run's progress survives a cancel or a crash.
    """

    progress = Signal(int, int)  # (current, total), 1-indexed current
    finished = Signal(str)       # the CSV path written
    errored = Signal(str)

    def __init__(
        self,
        files: list[Path],
        model,
        csv_path: Path,
        prob_thresh: float | None = 0.5,
        min_distance: int = 2,
        parent=None,
    ):
        super().__init__(parent)
        self._files = list(files)
        self._model = model
        self._csv_path = Path(csv_path)
        self._prob_thresh = prob_thresh
        self._min_distance = min_distance
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._csv_path.exists():
                self._csv_path.unlink()
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
                if len(points) > 0:
                    frame_col = np.full((len(points), 1), t)
                    prob_col = np.asarray(details.prob).reshape(-1, 1)
                    rows = np.hstack([frame_col, points, prob_col])
                    df = pd.DataFrame(rows, columns=["frame", "z", "y", "x", "prob"])
                    df.to_csv(self._csv_path, mode="a", header=not header_written, index=False)
                    header_written = True
                self.progress.emit(t + 1, total)
            self.finished.emit(str(self._csv_path))
        except Exception as exc:
            self.errored.emit(str(exc))
