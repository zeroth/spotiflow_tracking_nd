from __future__ import annotations

from pathlib import Path

import napari
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

from napari_spotiflow_tracking_nd._io3d import (
    build_lazy_stack_from_files,
    compute_layer_scale,
    read_volume,
)
from napari_spotiflow_tracking_nd._preprocessing3d import (
    _get_device,
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

        # ── Input / output selection ─────────────────────────────────
        # Loading a folder is the napari reader's job (File > Open,
        # drag-and-drop, or the CLI) — this widget only operates on
        # whatever image layer is already open.
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

        # ── Calibration ───────────────────────────────────────────────
        # The reader can't prompt for these, so they're entered here and
        # applied to whichever layer is selected above.
        calib_group = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()
        calib_group.setLayout(calib_layout)

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
        calib_layout.addLayout(calib_row)

        self._apply_calibration_btn = QPushButton("Apply Calibration")
        self._apply_calibration_btn.clicked.connect(self._apply_calibration)
        calib_layout.addWidget(self._apply_calibration_btn)
        layout.addWidget(calib_group)

        # ── Background removal ───────────────────────────────────────
        bg_group = QGroupBox("Background Removal (3D high-pass)")
        bg_layout = QHBoxLayout()
        bg_group.setLayout(bg_layout)
        bg_layout.addWidget(QLabel("sigma:"))
        self._bg_sigma_spin = QDoubleSpinBox()
        self._bg_sigma_spin.setRange(0.1, 200.0)
        self._bg_sigma_spin.setValue(10.0)
        bg_layout.addWidget(self._bg_sigma_spin)
        self._bg_use_gpu_check = QCheckBox("Use GPU")
        bg_layout.addWidget(self._bg_use_gpu_check)
        self._bg_preview_btn = QPushButton("Preview")
        self._bg_run_btn = QPushButton("Run")
        bg_layout.addWidget(self._bg_preview_btn)
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
        self._blur_preview_btn = QPushButton("Preview")
        self._blur_run_btn = QPushButton("Run")
        blur_layout.addWidget(self._blur_preview_btn)
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
        self._calc_preview_btn = QPushButton("Preview")
        self._calc_run_btn = QPushButton("Run")
        calc_layout.addWidget(self._calc_preview_btn)
        calc_layout.addWidget(self._calc_run_btn)
        layout.addWidget(calc_group)

        # ── Progress ─────────────────────────────────────────────────
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
        self._bg_run_btn.clicked.connect(self._run_remove_background)
        self._blur_run_btn.clicked.connect(self._run_gaussian_blur)
        self._calc_run_btn.clicked.connect(self._run_arithmetic)
        self._bg_preview_btn.clicked.connect(self._preview_remove_background)
        self._blur_preview_btn.clicked.connect(self._preview_gaussian_blur)
        self._calc_preview_btn.clicked.connect(self._preview_arithmetic)
        # Populate once up front: `layers.events.inserted` only fires for
        # layers added *after* this point, so a widget opened on a viewer
        # that already has images would otherwise start with an empty combo.
        self._refresh_image_combo()

    def _refresh_image_combo(self, event=None):
        current = self._image_combo.currentText()
        self._image_combo.clear()
        names = [l.name for l in self.viewer.layers if isinstance(l, napari.layers.Image)]
        self._image_combo.addItems(names)
        if current in names:
            self._image_combo.setCurrentText(current)

    # ── Folder browsing ──────────────────────────────────────────────

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._output_folder_edit.setText(folder)

    # ── Calibration ───────────────────────────────────────────────────

    def _apply_calibration(self):
        """Set scale + calibration metadata on the selected input layer.

        Loading is the reader's job now (File > Open, drag-and-drop, the
        CLI); this is how calibration reaches a layer that opened that way,
        since a reader has no UI to prompt through. Works on any image
        layer with the expected metadata keys, not just one the reader
        produced.
        """
        layer = self._get_input_layer()
        if layer is None:
            return
        pixel_size_xy_um = self._pixel_size_xy_spin.value() or None
        voxel_size_z_um = self._voxel_size_z_spin.value() or None
        frame_interval_s = self._frame_interval_spin.value() or None

        layer.scale = compute_layer_scale(pixel_size_xy_um, voxel_size_z_um)
        layer.metadata["pixel_size_xy_um"] = pixel_size_xy_um
        layer.metadata["voxel_size_z_um"] = voxel_size_z_um
        layer.metadata["frame_interval_s"] = frame_interval_s
        show_info(f"Calibration applied to {layer.name!r}.")

    # ── Shared run machinery ─────────────────────────────────────────

    def _get_input_layer(self):
        name = self._image_combo.currentText()
        if not name:
            show_error("No input layer selected.")
            return None
        return self.viewer.layers[name]

    def _get_output_dir(self, layer=None) -> Path | None:
        """The validated output directory, or None (after `show_error`).

        `VolumeOpWorker3D` writes `output_dir / path.name`, mirroring the
        input filenames exactly, so an output folder that *is* the input
        folder would overwrite the source acquisition in place, file by
        file, with no confirmation and no undo. Both folder pickers are
        adjacent `Browse…` dialogs, so that slip is a plausible one; it is
        refused here, before any worker is constructed.
        """
        folder = self._output_folder_edit.text()
        if not folder:
            show_error("No output folder selected.")
            return None

        try:
            out = Path(folder).resolve()
        except OSError as exc:
            show_error(str(exc))
            return None

        src_files = layer.metadata.get("source_files") if layer is not None else None
        if src_files:
            try:
                src_dir = Path(src_files[0]).resolve().parent
            except OSError:
                src_dir = None
            if src_dir is not None and out == src_dir:
                show_error(
                    "Output folder is the input folder — pick a different one "
                    "to avoid overwriting your source data."
                )
                return None
        return out

    def _current_timepoint_volume(self, layer):
        """Read just the currently-displayed timepoint off disk.

        Returns `(volume, t)`, or `(None, None)` after `show_error`. This
        is what makes preview quick: one file instead of the whole series.
        """
        files = layer.metadata.get("source_files")
        if not files:
            show_error(
                "Selected layer has no source files (open the folder via "
                "File > Open, drag-and-drop, or the napari CLI)."
            )
            return None, None

        step = self.viewer.dims.current_step
        t = int(step[0]) if len(step) else 0
        # The viewer's dims can outrun the layer (another, longer layer
        # selected, or a stale step after a reload) — clamp rather than
        # IndexError.
        t = max(0, min(t, len(files) - 1))

        try:
            return read_volume(files[t]), t
        except (OSError, ValueError) as exc:
            show_error(f"Could not read timepoint {t}: {exc}")
            return None, None

    def _show_preview_layer(self, name: str, data, source_layer):
        """Add the preview layer, or update it in place if it already
        exists — tuning means clicking Preview repeatedly, and a new layer
        per click would bury the viewer."""
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            # A single (Z, Y, X) volume, so the source layer's scale minus
            # its leading T entry.
            self.viewer.add_image(data, name=name, scale=source_layer.scale[1:])

    def _run_preview(self, op, op_kwargs: dict, kind: str):
        """Apply `op` to the current timepoint only, in memory.

        Writes nothing, and needs no output folder — the point is to tune
        parameters before committing to a whole-series run.
        """
        layer = self._get_input_layer()
        if layer is None:
            return
        volume, t = self._current_timepoint_volume(layer)
        if volume is None:
            return

        try:
            result = op(volume, **op_kwargs)
        except Exception as exc:
            show_error(f"Preview failed: {exc}")
            return

        self._show_preview_layer(f"{layer.name} ({kind} preview)", result, layer)
        show_info(f"Preview: {kind} on timepoint {t} (nothing written to disk).")

    def _run_op(self, op, op_kwargs: dict, suffix: str):
        layer = self._get_input_layer()
        if layer is None:
            return
        files = layer.metadata.get("source_files")
        if not files:
            show_error(
                "Selected layer has no source files (open the folder via "
                "File > Open, drag-and-drop, or the napari CLI)."
            )
            return
        output_dir = self._get_output_dir(layer)
        if output_dir is None:
            return

        self._progress_bar.setMaximum(len(files))
        self._progress_bar.setValue(0)

        worker = VolumeOpWorker3D(files=files, output_dir=output_dir, op=op, op_kwargs=op_kwargs)
        worker.progress.connect(lambda cur, total: self._progress_bar.setValue(cur))
        worker.errored.connect(show_error)

        def _on_finished(written_paths):
            if not written_paths:
                return
            # Built from the exact list this run wrote, not from a fresh
            # glob of `output_dir`: an unrelated `.tif` left in that folder
            # would otherwise silently join the new layer, or raise a
            # shape-mismatch ValueError out of this signal handler.
            try:
                stack, out_files = build_lazy_stack_from_files(written_paths)
            except (OSError, ValueError) as exc:
                show_error(f"Wrote {len(written_paths)} volumes, but could not "
                           f"load them back as a layer: {exc}")
                return
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

        worker.run_finished.connect(_on_finished)
        self._worker = worker
        worker.run()  # Phase 1: run synchronously on the calling thread; the
        # `QThread` API is kept so a future phase can switch to `worker.start()`
        # without changing this call site's signal wiring.

    # ── Op builders ──────────────────────────────────────────────────
    # Each returns (op, kwargs, run_suffix, preview_kind). `run_suffix`
    # carries the parameters, so every run makes a distinctly-named output
    # layer; `preview_kind` deliberately does not, so repeated previews
    # replace one layer instead of stacking one per parameter value.

    def _bg_op(self):
        # `_get_device` falls back to "cpu" when CUDA isn't actually
        # available, instead of erroring deep inside the torch call.
        device = _get_device(use_gpu=self._bg_use_gpu_check.isChecked())
        sigma = self._bg_sigma_spin.value()
        return (
            remove_background_high_pass_3d,
            {"sigma": sigma, "device": device},
            f"bg s={sigma}",
            "bg",
        )

    def _blur_op(self):
        sigma = self._blur_sigma_spin.value()
        return gaussian_blur_3d, {"sigma": sigma}, f"blur s={sigma}", "blur"

    def _calc_op(self):
        op = self._calc_op_combo.currentText()
        value = self._calc_value_spin.value()
        return apply_calculation, {"op": op, "value": value}, f"{op} {value}", "calc"

    # ── Run (whole series) ───────────────────────────────────────────

    def _run_remove_background(self):
        op, kwargs, suffix, _kind = self._bg_op()
        self._run_op(op, kwargs, suffix)

    def _run_gaussian_blur(self):
        op, kwargs, suffix, _kind = self._blur_op()
        self._run_op(op, kwargs, suffix)

    def _run_arithmetic(self):
        op, kwargs, suffix, _kind = self._calc_op()
        self._run_op(op, kwargs, suffix)

    # ── Preview (current timepoint only) ─────────────────────────────

    def _preview_remove_background(self):
        op, kwargs, _suffix, kind = self._bg_op()
        self._run_preview(op, kwargs, kind)

    def _preview_gaussian_blur(self):
        op, kwargs, _suffix, kind = self._blur_op()
        self._run_preview(op, kwargs, kind)

    def _preview_arithmetic(self):
        op, kwargs, _suffix, kind = self._calc_op()
        self._run_preview(op, kwargs, kind)
