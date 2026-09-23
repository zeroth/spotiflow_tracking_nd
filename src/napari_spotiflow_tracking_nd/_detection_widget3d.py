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

from napari_spotiflow_tracking_nd._preprocessing3d import _get_device
from napari_spotiflow_tracking_nd._segmentation3d import PRETRAINED_3D_MODELS, load_model
from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

# Calibration keys copied from the source image layer onto the Points
# layer, and written to the CSV's sidecar, so a later phase can turn
# pixel/frame track properties into physical units.
CALIBRATION_KEYS = ("pixel_size_xy_um", "voxel_size_z_um", "frame_interval_s")


def calibration_sidecar_path(csv_path: str | Path) -> Path:
    """The JSON sidecar recording the calibration a CSV was detected under:
    the CSV path with its extension replaced by `.json`
    (`spots.csv` -> `spots.json`)."""
    return Path(csv_path).with_suffix(".json")


class DetectionWidget3D(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker: DetectionWorker3D | None = None
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

        self._run_btn = QPushButton("Run Detection")
        self._run_btn.clicked.connect(self._run_detection)
        layout.addWidget(self._run_btn)

        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)
        self._blocking_note = QLabel(
            "Note: a run blocks the napari window until it finishes (Phase 1 "
            "runs synchronously). It is working, not hung."
        )
        self._blocking_note.setWordWrap(True)
        layout.addWidget(self._blocking_note)
        layout.addStretch()

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
        try:
            model = load_model(self._model_combo.currentText(), device=device)
        except Exception as exc:
            # A bad model name, a missing/corrupt checkpoint folder, or a
            # failed weight download must not raise out of a Qt slot.
            show_error(f"Could not load model "
                       f"{self._model_combo.currentText()!r}: {exc}")
            return

        self._progress_bar.setMaximum(len(files))
        self._progress_bar.setValue(0)

        worker = DetectionWorker3D(
            files=files,
            model=model,
            csv_path=Path(csv_path),
            prob_thresh=self._prob_thresh_spin.value(),
            min_distance=self._min_distance_spin.value(),
        )
        worker.progress.connect(lambda cur, total: self._progress_bar.setValue(cur))
        worker.errored.connect(show_error)

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
                name=f"{layer.name} (spots)",
                scale=layer.scale,
                metadata={"csv_path": written_csv_path, **calibration},
                properties={"prob": df["prob"].to_numpy()},
            )
            show_info(f"Detected {len(df)} spots across {df['frame'].nunique()} timepoints.")

        worker.run_finished.connect(_on_finished)
        self._worker = worker
        worker.run()  # Phase 1: synchronous, same rationale as PreProcessingWidget3D._run_op.
