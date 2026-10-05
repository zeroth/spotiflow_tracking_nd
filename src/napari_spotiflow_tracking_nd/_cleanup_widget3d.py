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
    isotropic_zyx,
    read_volume,
    validate_output_dir,
)
from napari_spotiflow_tracking_nd._preprocessing3d import (
    _get_device,
    apply_calculation,
    gaussian_blur_3d,
    remove_background_high_pass_3d,
)
from napari_spotiflow_tracking_nd._run_state import RunStateMixin
from napari_spotiflow_tracking_nd._workers3d import FunctionWorker3D, VolumeOpWorker3D


class PreProcessingWidget3D(RunStateMixin, QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
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

        self._link_sigma_check = QCheckBox("Link σ to calibration")
        self._link_sigma_check.setToolTip(
            "σ is in voxels, and voxels are rarely cubes. With this on, "
            "the σ below is the lateral value and the axial one is derived "
            "from the calibration, so the filter covers the same physical "
            "distance on every axis."
        )
        calib_layout.addWidget(self._link_sigma_check)
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
        # On by default: `_get_device` quietly falls back to "cpu"
        # when CUDA is unavailable (see the detection widget).
        self._bg_use_gpu_check.setChecked(True)
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
        self._progress_bar.setFormat("%v / %m")
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip(
            "Stops after the volume currently being written. A preview is a "
            "single call and cannot be interrupted — cancelling discards its "
            "result."
        )
        layout.addWidget(self._cancel_btn)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

        self._init_run_state(
            run_buttons=[
                self._bg_run_btn, self._blur_run_btn, self._calc_run_btn,
                self._bg_preview_btn, self._blur_preview_btn, self._calc_preview_btn,
            ],
            cancel_button=self._cancel_btn,
            progress_bar=self._progress_bar,
            status_label=self._status_label,
        )

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

        The rule — and why it exists — lives in `_io3d.validate_output_dir`,
        shared with the detection widget's mask folder.
        """
        src_files = layer.metadata.get("source_files") if layer is not None else None
        out, error = validate_output_dir(self._output_folder_edit.text(), src_files)
        if error is not None:
            show_error(error)
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

        worker = FunctionWorker3D(op, args=(volume,), kwargs=op_kwargs)

        def _on_finished(result):
            self._show_preview_layer(f"{layer.name} ({kind} preview)", result, layer)
            show_info(f"Preview: {kind} on timepoint {t} (nothing written to disk).")

        self._begin_run(
            worker, total=1, description=f"Previewing {kind} on timepoint {t}",
            on_finished=_on_finished,
        )

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

        worker = VolumeOpWorker3D(files=files, output_dir=output_dir, op=op, op_kwargs=op_kwargs)

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
                # A cancelled run is still loaded — with its partial state
                # in the name, so it is never mistaken for a full one.
                name=f"{layer.name} ({suffix}){self.cancelled_suffix()}",
                scale=layer.scale,
                metadata={
                    "source_files": out_files,
                    "pixel_size_xy_um": layer.metadata.get("pixel_size_xy_um"),
                    "voxel_size_z_um": layer.metadata.get("voxel_size_z_um"),
                    "frame_interval_s": layer.metadata.get("frame_interval_s"),
                },
            )
            show_info(f"Wrote {len(written_paths)} volumes to {output_dir}")

        self._begin_run(
            worker, total=len(files), description=f"Running {suffix}", on_finished=_on_finished,
        )

    @staticmethod
    def _sigma_label(sigma) -> str:
        """`sigma` for a layer name, without float noise."""
        if isinstance(sigma, tuple):
            return "(" + ", ".join(f"{v:.3g}" for v in sigma) + ")"
        return f"{sigma:.3g}"

    def _sigma_for(self, sigma_xy: float, layer):
        """σ as `(z, y, x)` when linked to calibration, else the scalar.

        A σ of 4 means four *voxels*, and these voxels are not cubes — at
        0.104 µm laterally against 0.210 µm axially, the same number covers
        twice the physical distance along z. Linking derives the axial
        value so the filter is isotropic in µm rather than in array
        indices.

        Returns None (after `show_error`) when linking is on but the layer
        carries no calibration: a filter quietly doing something other than
        what the ticked box promises is worse than a refusal.
        """
        if not self._link_sigma_check.isChecked():
            return sigma_xy

        pixel_size_xy_um = layer.metadata.get("pixel_size_xy_um")
        voxel_size_z_um = layer.metadata.get("voxel_size_z_um")
        if not pixel_size_xy_um or not voxel_size_z_um:
            show_error(
                "σ is linked to calibration, but this layer has none — set "
                "the pixel and voxel size above and press Apply Calibration "
                "first."
            )
            return None

        return isotropic_zyx(sigma_xy, pixel_size_xy_um, voxel_size_z_um)

    # ── Op builders ──────────────────────────────────────────────────
    # Each returns (op, kwargs, run_suffix, preview_kind). `run_suffix`
    # carries the parameters, so every run makes a distinctly-named output
    # layer; `preview_kind` deliberately does not, so repeated previews
    # replace one layer instead of stacking one per parameter value.

    def _bg_op(self, layer):
        # `_get_device` falls back to "cpu" when CUDA isn't actually
        # available, instead of erroring deep inside the torch call.
        device = _get_device(use_gpu=self._bg_use_gpu_check.isChecked())
        sigma = self._sigma_for(self._bg_sigma_spin.value(), layer)
        if sigma is None:
            return None
        return (
            remove_background_high_pass_3d,
            {"sigma": sigma, "device": device},
            f"bg s={self._sigma_label(sigma)}",
            "bg",
        )

    def _blur_op(self, layer):
        sigma = self._sigma_for(self._blur_sigma_spin.value(), layer)
        if sigma is None:
            return None
        return (
            gaussian_blur_3d,
            {"sigma": sigma},
            f"blur s={self._sigma_label(sigma)}",
            "blur",
        )

    def _calc_op(self, layer):  # `layer` unused: arithmetic has no sigma
        op = self._calc_op_combo.currentText()
        value = self._calc_value_spin.value()
        return apply_calculation, {"op": op, "value": value}, f"{op} {value}", "calc"

    # ── Dispatch ─────────────────────────────────────────────────────

    def _dispatch(self, build, *, preview: bool):
        """Resolve the input layer, build the op against it, and start.

        The builders need the layer because a σ linked to calibration is
        derived from that layer's metadata, and because they may refuse
        (returning None) when the calibration it promises is missing.
        """
        layer = self._get_input_layer()
        if layer is None:
            return
        built = build(layer)
        if built is None:
            return
        op, kwargs, suffix, kind = built
        if preview:
            self._run_preview(op, kwargs, kind)
        else:
            self._run_op(op, kwargs, suffix)

    # ── Run (whole series) ───────────────────────────────────────────

    def _run_remove_background(self):
        self._dispatch(self._bg_op, preview=False)

    def _run_gaussian_blur(self):
        self._dispatch(self._blur_op, preview=False)

    def _run_arithmetic(self):
        self._dispatch(self._calc_op, preview=False)

    # ── Preview (current timepoint only) ─────────────────────────────

    def _preview_remove_background(self):
        self._dispatch(self._bg_op, preview=True)

    def _preview_gaussian_blur(self):
        self._dispatch(self._blur_op, preview=True)

    def _preview_arithmetic(self):
        self._dispatch(self._calc_op, preview=True)
