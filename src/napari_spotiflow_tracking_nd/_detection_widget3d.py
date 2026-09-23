from __future__ import annotations

import json
from pathlib import Path

import napari
import pandas as pd
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_spotiflow_tracking_nd._io3d import read_volume
from napari_spotiflow_tracking_nd._preprocessing3d import _get_device
from napari_spotiflow_tracking_nd._segmentation3d import (
    PRETRAINED_3D_MODELS,
    detect_spots_3d,
    load_model,
)
from napari_spotiflow_tracking_nd._run_state import RunStateMixin
from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D, FunctionWorker3D

# Calibration keys copied from the source image layer onto the Points
# layer, and written to the CSV's sidecar, so a later phase can turn
# pixel/frame track properties into physical units.
CALIBRATION_KEYS = ("pixel_size_xy_um", "voxel_size_z_um", "frame_interval_s")


def calibration_sidecar_path(csv_path: str | Path) -> Path:
    """The JSON sidecar recording the calibration a CSV was detected under:
    the CSV path with its extension replaced by `.json`
    (`spots.csv` -> `spots.json`)."""
    return Path(csv_path).with_suffix(".json")


class DetectionWidget3D(RunStateMixin, QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        # Last model loaded, keyed by (name, device). Preview is meant to
        # be clicked repeatedly while tuning, and re-loading the weights
        # every click is by far the slowest part of it.
        self._model_cache = None
        self._model_cache_key: tuple[str, str] | None = None
        self._setup_ui()
        self._connect_events()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        io_group = QGroupBox("Input")
        io_layout = QVBoxLayout()
        io_group.setLayout(io_layout)
        self._image_combo = QComboBox()
        io_layout.addWidget(QLabel("Input layer:"))
        io_layout.addWidget(self._image_combo)
        layout.addWidget(io_group)

        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()
        model_group.setLayout(model_layout)
        # Editable: the drop-down offers Spotiflow's built-in 3D models,
        # but a custom checkpoint folder (typed, pasted, or picked with
        # Browse…) is still a valid entry — `load_model` treats anything
        # that isn't a registered name as a path.
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.addItems(PRETRAINED_3D_MODELS)
        self._model_combo.setCurrentText(PRETRAINED_3D_MODELS[0])
        model_layout.addWidget(self._model_combo)
        self._browse_model_btn = QPushButton("Browse…")
        self._browse_model_btn.clicked.connect(self._browse_custom_model)
        model_layout.addWidget(self._browse_model_btn)
        self._use_gpu_check = QCheckBox("Use GPU")
        # On by default: `_get_device` quietly falls back to "cpu"
        # when CUDA is unavailable, so this costs nothing on a
        # CPU-only machine and saves ticking it every session on a
        # GPU one.
        self._use_gpu_check.setChecked(True)
        model_layout.addWidget(self._use_gpu_check)
        layout.addWidget(model_group)

        params_group = QGroupBox("Detection parameters")
        params_layout = QHBoxLayout()
        params_group.setLayout(params_layout)
        params_layout.addWidget(QLabel("prob_thresh:"))
        self._prob_thresh_spin = QDoubleSpinBox()
        self._prob_thresh_spin.setRange(0.0, 1.0)
        self._prob_thresh_spin.setSingleStep(0.05)
        self._prob_thresh_spin.setValue(0.5)
        params_layout.addWidget(self._prob_thresh_spin)
        params_layout.addWidget(QLabel("min_distance:"))
        self._min_distance_spin = QSpinBox()
        self._min_distance_spin.setRange(1, 100)
        self._min_distance_spin.setValue(2)
        params_layout.addWidget(self._min_distance_spin)
        layout.addWidget(params_group)

        out_group = QGroupBox("Output")
        out_layout = QHBoxLayout()
        out_group.setLayout(out_layout)
        self._csv_path_edit = QLineEdit()
        out_layout.addWidget(self._csv_path_edit)
        self._browse_csv_btn = QPushButton("Browse…")
        self._browse_csv_btn.clicked.connect(self._browse_csv_path)
        out_layout.addWidget(self._browse_csv_btn)
        layout.addWidget(out_group)

        run_row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview (current timepoint)")
        self._preview_btn.clicked.connect(self._run_preview)
        run_row.addWidget(self._preview_btn)
        self._run_btn = QPushButton("Run Detection")
        self._run_btn.clicked.connect(self._run_detection)
        run_row.addWidget(self._run_btn)
        layout.addLayout(run_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("%v / %m")
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip(
            "Stops after the timepoint currently being detected. A preview is "
            "a single call and cannot be interrupted — cancelling discards its "
            "result."
        )
        layout.addWidget(self._cancel_btn)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

        self._init_run_state(
            run_buttons=[self._run_btn, self._preview_btn],
            cancel_button=self._cancel_btn,
            progress_bar=self._progress_bar,
            status_label=self._status_label,
        )

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_image_combo)
        self.viewer.layers.events.removed.connect(self._refresh_image_combo)
        # Populate once up front: in the normal workflow the folder is
        # loaded in the Pre-Processing widget *before* this one is opened,
        # so waiting on `layers.events.inserted` leaves the combo empty.
        self._refresh_image_combo()

    def _refresh_image_combo(self, event=None):
        current = self._image_combo.currentText()
        self._image_combo.clear()
        names = [l.name for l in self.viewer.layers if isinstance(l, napari.layers.Image)]
        self._image_combo.addItems(names)
        if current in names:
            self._image_combo.setCurrentText(current)

    def _browse_custom_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select model checkpoint folder")
        if folder:
            self._model_combo.setCurrentText(folder)

    def _browse_csv_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save spots CSV", "", "CSV files (*.csv)")
        if path:
            self._csv_path_edit.setText(path)

    def _load_model_cached(self, model_name: str, device: str):
        """`load_model`, reusing the previous result when the (name,
        device) pair is unchanged."""
        key = (model_name, device)
        if self._model_cache_key != key:
            self._model_cache = load_model(model_name, device=device)
            self._model_cache_key = key
        return self._model_cache

    def _model_loader(self, model_name: str, device: str):
        """A zero-arg loader for the workers to call on their own thread.

        Wraps the failure so the message still names the model: the
        worker's `errored` channel emits `str(exc)` verbatim, so context
        added here is the only context the user gets.
        """

        def _load():
            try:
                return self._load_model_cached(model_name, device)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load model {model_name!r}: {exc}"
                ) from exc

        return _load

    def _run_preview(self):
        """Detect on the currently-displayed timepoint only, in memory.

        Writes no CSV and needs no output path — the point is to tune
        `prob_thresh` / `min_distance` against one volume before
        committing to the whole series.
        """
        name = self._image_combo.currentText()
        if not name:
            show_error("No input layer selected.")
            return
        layer = self.viewer.layers[name]
        files = layer.metadata.get("source_files")
        if not files:
            show_error(
                "Selected layer has no source files (open the folder via "
                "File > Open, drag-and-drop, or the napari CLI)."
            )
            return

        step = self.viewer.dims.current_step
        t = int(step[0]) if len(step) else 0
        # The viewer's dims can outrun the layer (another, longer layer
        # selected, or a stale step after a reload) — clamp, don't IndexError.
        t = max(0, min(t, len(files) - 1))

        device = _get_device(use_gpu=self._use_gpu_check.isChecked())
        model_name = self._model_combo.currentText()

        def _detect():
            # Both the model load and the detection happen on the worker
            # thread; the cache is safe unlocked because only one job runs
            # at a time.
            model = self._model_loader(model_name, device)()
            volume = read_volume(files[t])
            points, _details = detect_spots_3d(
                volume,
                model,
                prob_thresh=self._prob_thresh_spin.value(),
                min_distance=self._min_distance_spin.value(),
            )
            return points

        worker = FunctionWorker3D(_detect)

        def _on_finished(points):
            # One preview layer, updated in place: tuning means clicking
            # this repeatedly, and a layer per click would bury the viewer.
            preview_name = f"{layer.name} (spot preview)"
            if preview_name in self.viewer.layers:
                self.viewer.layers[preview_name].data = points
            else:
                # Points here are (z, y, x) for a single volume, so the
                # source layer's scale minus its leading T entry.
                self.viewer.add_points(
                    points,
                    name=preview_name,
                    scale=layer.scale[1:],
                    face_color="transparent",
                    border_color="red",
                )
            show_info(
                f"Preview: {len(points)} spots on timepoint {t} "
                "(nothing written to disk)."
            )

        self._begin_run(
            worker, total=1, description=f"Previewing detection on timepoint {t}",
            on_finished=_on_finished,
        )

    def _run_detection(self):
        name = self._image_combo.currentText()
        if not name:
            show_error("No input layer selected.")
            return
        layer = self.viewer.layers[name]
        files = layer.metadata.get("source_files")
        if not files:
            show_error(
                "Selected layer has no source files (open the folder via "
                "File > Open, drag-and-drop, or the napari CLI)."
            )
            return
        csv_path = self._csv_path_edit.text()
        if not csv_path:
            show_error("No output CSV path selected.")
            return

        # `_get_device` falls back to "cpu" when CUDA isn't actually
        # available, instead of erroring deep inside Spotiflow.
        device = _get_device(use_gpu=self._use_gpu_check.isChecked())
        model_name = self._model_combo.currentText()

        # Passed as a callable, not a loaded model: resolving it is slow
        # enough to freeze the UI on its own, so `DetectionWorker3D` calls
        # this on its own thread. The cache needs no lock because only one
        # job runs at a time.
        worker = DetectionWorker3D(
            files=files,
            model=self._model_loader(model_name, device),
            csv_path=Path(csv_path),
            prob_thresh=self._prob_thresh_spin.value(),
            min_distance=self._min_distance_spin.value(),
        )

        def _on_finished(written_csv_path):
            # `DetectionWorker3D` only writes the CSV once at least one
            # frame has a detection — if every frame in the run has zero
            # spots, the file is never created even though `finished`
            # still fires with its path.
            if not Path(written_csv_path).exists():
                show_info("Detection finished — no spots found.")
                return

            calibration = {k: layer.metadata.get(k) for k in CALIBRATION_KEYS}
            # The CSV's coordinates are in pixels/frames; the calibration
            # that turns them into µm/seconds lives only in the layer's
            # metadata, which doesn't survive to disk. Record it beside the
            # CSV so a later phase (and anyone re-opening the CSV) has it.
            sidecar = calibration_sidecar_path(written_csv_path)
            try:
                sidecar.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            except OSError as exc:
                show_error(f"Could not write the calibration sidecar {sidecar}: {exc}")

            df = pd.read_csv(written_csv_path)
            if len(df) == 0:
                show_info("Detection finished — no spots found.")
                return
            points = df[["frame", "z", "y", "x"]].to_numpy()
            # layer.scale is (T, Z, Y, X) — one entry per point column above,
            # since points here are (frame, z, y, x), not just (z, y, x).
            self.viewer.add_points(
                points,
                name=f"{layer.name} (spots){self.cancelled_suffix()}",
                scale=layer.scale,
                metadata={"csv_path": written_csv_path, **calibration},
                properties={"prob": df["prob"].to_numpy()},
            )
            show_info(f"Detected {len(df)} spots across {df['frame'].nunique()} timepoints.")

        self._begin_run(
            worker, total=len(files), description="Detecting spots",
            on_finished=_on_finished,
        )
