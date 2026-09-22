from __future__ import annotations

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

from napari_spotiflow_tracking_nd._segmentation3d import load_model
from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D


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
        self._model_name_edit = QLineEdit("synth_3d")
        model_layout.addWidget(self._model_name_edit)
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
        layout.addStretch()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_image_combo)
        self.viewer.layers.events.removed.connect(self._refresh_image_combo)

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
            self._model_name_edit.setText(folder)

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
            show_error("Selected layer has no source files (load it via Pre-Processing's 'Load Folder').")
            return
        csv_path = self._csv_path_edit.text()
        if not csv_path:
            show_error("No output CSV path selected.")
            return

        device = "cuda" if self._use_gpu_check.isChecked() else "cpu"
        model = load_model(self._model_name_edit.text(), device=device)

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
                metadata={"csv_path": written_csv_path},
                properties={"prob": df["prob"].to_numpy()},
            )
            show_info(f"Detected {len(df)} spots across {df['frame'].nunique()} timepoints.")

        worker.finished.connect(_on_finished)
        self._worker = worker
        worker.run()  # Phase 1: synchronous, same rationale as PreProcessingWidget3D._run_op.
