from __future__ import annotations

from pathlib import Path

import napari
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
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

from napari_spotiflow_tracking_nd._io3d import compute_layer_scale, load_folder_as_lazy_stack
from napari_spotiflow_tracking_nd._preprocessing3d import (
    apply_calculation,
    gaussian_blur_3d,
    remove_background_high_pass_3d,
)
from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D


class PreProcessingWidget3D(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker: VolumeOpWorker3D | None = None
        self._setup_ui()
        self._connect_events()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── Load folder ──────────────────────────────────────────────
        load_group = QGroupBox("Load timeseries folder")
        load_layout = QVBoxLayout()
        load_group.setLayout(load_layout)

        folder_row = QHBoxLayout()
        self._input_folder_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_input_folder)
        folder_row.addWidget(self._input_folder_edit)
        folder_row.addWidget(browse_btn)
        load_layout.addLayout(folder_row)

        calib_row = QHBoxLayout()
        calib_row.addWidget(QLabel("xy (µm):"))
        self._pixel_size_xy_spin = QDoubleSpinBox()
        self._pixel_size_xy_spin.setRange(0.0, 100.0)
        self._pixel_size_xy_spin.setDecimals(4)
        calib_row.addWidget(self._pixel_size_xy_spin)
        calib_row.addWidget(QLabel("z (µm):"))
        self._voxel_size_z_spin = QDoubleSpinBox()
        self._voxel_size_z_spin.setRange(0.0, 100.0)
        self._voxel_size_z_spin.setDecimals(4)
        calib_row.addWidget(self._voxel_size_z_spin)
        calib_row.addWidget(QLabel("dt (s):"))
        self._frame_interval_spin = QDoubleSpinBox()
        self._frame_interval_spin.setRange(0.0, 10000.0)
        self._frame_interval_spin.setDecimals(3)
        calib_row.addWidget(self._frame_interval_spin)
        load_layout.addLayout(calib_row)

        self._load_btn = QPushButton("Load Folder")
        self._load_btn.clicked.connect(self._load_folder)
        load_layout.addWidget(self._load_btn)
        layout.addWidget(load_group)

        # ── Input / output selection ─────────────────────────────────
        io_group = QGroupBox("Input / Output")
        io_layout = QVBoxLayout()
        io_group.setLayout(io_layout)

        self._image_combo = QComboBox()
        io_layout.addWidget(QLabel("Input layer:"))
        io_layout.addWidget(self._image_combo)

        out_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        out_browse_btn = QPushButton("Browse…")
        out_browse_btn.clicked.connect(self._browse_output_folder)
        out_row.addWidget(self._output_folder_edit)
        out_row.addWidget(out_browse_btn)
        io_layout.addWidget(QLabel("Output folder:"))
        io_layout.addLayout(out_row)
        layout.addWidget(io_group)

        # ── Background removal ───────────────────────────────────────
        bg_group = QGroupBox("Background Removal (3D high-pass)")
        bg_layout = QHBoxLayout()
        bg_group.setLayout(bg_layout)
        bg_layout.addWidget(QLabel("sigma:"))
        self._bg_sigma_spin = QDoubleSpinBox()
        self._bg_sigma_spin.setRange(0.1, 200.0)
        self._bg_sigma_spin.setValue(10.0)
        bg_layout.addWidget(self._bg_sigma_spin)
        self._bg_run_btn = QPushButton("Run")
        bg_layout.addWidget(self._bg_run_btn)
        layout.addWidget(bg_group)

        # ── Gaussian blur ────────────────────────────────────────────
        blur_group = QGroupBox("Gaussian Blur")
        blur_layout = QHBoxLayout()
        blur_group.setLayout(blur_layout)
        blur_layout.addWidget(QLabel("sigma:"))
        self._blur_sigma_spin = QDoubleSpinBox()
        self._blur_sigma_spin.setRange(0.0, 200.0)
        self._blur_sigma_spin.setValue(1.0)
        blur_layout.addWidget(self._blur_sigma_spin)
        self._blur_run_btn = QPushButton("Run")
        blur_layout.addWidget(self._blur_run_btn)
        layout.addWidget(blur_group)

        # ── Arithmetic ───────────────────────────────────────────────
        calc_group = QGroupBox("Arithmetic")
        calc_layout = QHBoxLayout()
        calc_group.setLayout(calc_layout)
        self._calc_op_combo = QComboBox()
        self._calc_op_combo.addItems(["add", "subtract"])
        calc_layout.addWidget(self._calc_op_combo)
        self._calc_value_spin = QSpinBox()
        self._calc_value_spin.setRange(0, 65535)
        calc_layout.addWidget(self._calc_value_spin)
        self._calc_run_btn = QPushButton("Run")
        calc_layout.addWidget(self._calc_run_btn)
        layout.addWidget(calc_group)

        # ── Progress ─────────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)
        layout.addStretch()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_image_combo)
        self.viewer.layers.events.removed.connect(self._refresh_image_combo)
        self._bg_run_btn.clicked.connect(self._run_remove_background)
        self._blur_run_btn.clicked.connect(self._run_gaussian_blur)
        self._calc_run_btn.clicked.connect(self._run_arithmetic)

    def _refresh_image_combo(self, event=None):
        current = self._image_combo.currentText()
        self._image_combo.clear()
        names = [l.name for l in self.viewer.layers if isinstance(l, napari.layers.Image)]
        self._image_combo.addItems(names)
        if current in names:
            self._image_combo.setCurrentText(current)

    # ── Folder browsing ──────────────────────────────────────────────

    def _browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self._input_folder_edit.setText(folder)

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._output_folder_edit.setText(folder)

    # ── Load ──────────────────────────────────────────────────────────

    def _load_folder(self):
        folder = self._input_folder_edit.text()
        if not folder:
            show_error("No input folder selected.")
            return
        try:
            stack, files = load_folder_as_lazy_stack(folder)
        except (FileNotFoundError, ValueError) as exc:
            show_error(str(exc))
            return

        pixel_size_xy_um = self._pixel_size_xy_spin.value() or None
        voxel_size_z_um = self._voxel_size_z_spin.value() or None
        frame_interval_s = self._frame_interval_spin.value() or None
        scale = compute_layer_scale(pixel_size_xy_um, voxel_size_z_um)

        layer = self.viewer.add_image(
            stack,
            name=Path(folder).name,
            scale=scale,
            metadata={
                "source_files": files,
                "pixel_size_xy_um": pixel_size_xy_um,
                "voxel_size_z_um": voxel_size_z_um,
                "frame_interval_s": frame_interval_s,
            },
        )
        show_info(f"Loaded {len(files)} timepoints from {folder}")
        return layer

    # ── Shared run machinery ─────────────────────────────────────────

    def _get_input_layer(self):
        name = self._image_combo.currentText()
        if not name:
            show_error("No input layer selected.")
            return None
        return self.viewer.layers[name]

    def _get_output_dir(self) -> Path | None:
        folder = self._output_folder_edit.text()
        if not folder:
            show_error("No output folder selected.")
            return None
        return Path(folder)

    def _run_op(self, op, op_kwargs: dict, suffix: str):
        layer = self._get_input_layer()
        if layer is None:
            return
        output_dir = self._get_output_dir()
        if output_dir is None:
            return
        files = layer.metadata.get("source_files")
        if not files:
            show_error("Selected layer has no source files (load it via 'Load Folder').")
            return

        self._progress_bar.setMaximum(len(files))
        self._progress_bar.setValue(0)

        worker = VolumeOpWorker3D(files=files, output_dir=output_dir, op=op, op_kwargs=op_kwargs)
        worker.progress.connect(lambda cur, total: self._progress_bar.setValue(cur))
        worker.errored.connect(show_error)

        def _on_finished(written_paths):
            if not written_paths:
                return
            stack, out_files = load_folder_as_lazy_stack(output_dir)
            self.viewer.add_image(
                stack,
                name=f"{layer.name} ({suffix})",
                scale=layer.scale,
                metadata={
                    "source_files": out_files,
                    "pixel_size_xy_um": layer.metadata.get("pixel_size_xy_um"),
                    "voxel_size_z_um": layer.metadata.get("voxel_size_z_um"),
                    "frame_interval_s": layer.metadata.get("frame_interval_s"),
                },
            )
            show_info(f"Wrote {len(written_paths)} volumes to {output_dir}")

        worker.finished.connect(_on_finished)
        self._worker = worker
        worker.run()  # Phase 1: run synchronously on the calling thread; the
        # `QThread` API is kept so a future phase can switch to `worker.start()`
        # without changing this call site's signal wiring.

    def _run_remove_background(self):
        self._run_op(
            remove_background_high_pass_3d,
            {"sigma": self._bg_sigma_spin.value()},
            f"bg s={self._bg_sigma_spin.value()}",
        )

    def _run_gaussian_blur(self):
        self._run_op(
            gaussian_blur_3d,
            {"sigma": self._blur_sigma_spin.value()},
            f"blur s={self._blur_sigma_spin.value()}",
        )

    def _run_arithmetic(self):
        op = self._calc_op_combo.currentText()
        value = self._calc_value_spin.value()
        self._run_op(apply_calculation, {"op": op, "value": value}, f"{op} {value}")
