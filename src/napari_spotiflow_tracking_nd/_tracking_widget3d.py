"""The Spot Tracking widget: link a detection CSV into 3D tracks."""
from __future__ import annotations

import json
import os
from pathlib import Path

import napari
import pandas as pd
import trackpy
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

from napari_spotiflow_tracking_nd._detection_widget3d import (
    CALIBRATION_KEYS,
    calibration_sidecar_path,
)
from napari_spotiflow_tracking_nd._io3d import compute_layer_scale, isotropic_zyx
from napari_spotiflow_tracking_nd._layer_utils import refresh_layer_combo
from napari_spotiflow_tracking_nd._run_state import RunStateMixin
from napari_spotiflow_tracking_nd._tracking3d import LINK_COLUMNS
from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

#: Per-spot columns carried onto the Tracks layer as `features`, when the
#: tracks CSV has them — enough to colour tracks by fit quality or flag.
FEATURE_COLUMNS = (
    "coord_source", "prob", "censored",
    "amplitude", "integrated_intensity", "background",
)


def default_tracks_path(csv_path) -> Path:
    """`spots.csv` -> `spots_tracks.csv`, beside it."""
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_tracks.csv")


def same_file(a, b) -> bool:
    """True when `a` and `b` name the same file — after resolving `..`
    and, on Windows, ignoring case — whether or not either exists."""
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def read_sidecar_calibration(csv_path) -> dict:
    """The calibration recorded in `csv_path`'s JSON sidecar; `{}` when
    there is none or it cannot be read."""
    try:
        data = json.loads(calibration_sidecar_path(csv_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: data.get(k) for k in CALIBRATION_KEYS}


def _calibration_spin(maximum: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, maximum)
    spin.setDecimals(decimals)
    # 0 reads as "unknown", the Pre-Processing widget's convention.
    spin.setSpecialValueText("unknown")
    return spin


class TrackingWidget3D(RunStateMixin, QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        # The spots layer the current input CSV came from, if any — its
        # scale is the fallback for an uncalibrated run.
        self._source_layer = None
        # The input last set through `_set_input_csv`, to tell a typed or
        # pasted path from the one already adopted.
        self._last_input_csv = ""
        self._summary = ""
        self._setup_ui()
        self._connect_events()

    # ── UI ───────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        in_group = QGroupBox("Input")
        in_layout = QVBoxLayout()
        in_group.setLayout(in_layout)
        in_layout.addWidget(QLabel("Spots layer:"))
        self._points_combo = QComboBox()
        in_layout.addWidget(self._points_combo)
        csv_row = QHBoxLayout()
        self._csv_path_edit = QLineEdit()
        self._csv_path_edit.setPlaceholderText("Detection CSV")
        csv_row.addWidget(self._csv_path_edit)
        self._browse_csv_btn = QPushButton("Browse CSV…")
        self._browse_csv_btn.setToolTip(
            "Track a detection CSV straight from disk, without loading its "
            "spots into napari first."
        )
        self._browse_csv_btn.clicked.connect(self._browse_input_csv)
        csv_row.addWidget(self._browse_csv_btn)
        in_layout.addLayout(csv_row)
        self._csv_info_label = QLabel()
        in_layout.addWidget(self._csv_info_label)
        layout.addWidget(in_group)

        calib_group = QGroupBox("Calibration")
        calib_layout = QHBoxLayout()
        calib_group.setLayout(calib_layout)
        calib_layout.addWidget(QLabel("xy (µm):"))
        self._pixel_size_xy_spin = _calibration_spin(100.0, 4)
        calib_layout.addWidget(self._pixel_size_xy_spin)
        calib_layout.addWidget(QLabel("z (µm):"))
        self._voxel_size_z_spin = _calibration_spin(100.0, 4)
        calib_layout.addWidget(self._voxel_size_z_spin)
        calib_layout.addWidget(QLabel("dt (s):"))
        self._frame_interval_spin = _calibration_spin(10000.0, 3)
        calib_layout.addWidget(self._frame_interval_spin)
        layout.addWidget(calib_group)

        link_group = QGroupBox("Linking")
        link_layout = QVBoxLayout()
        link_group.setLayout(link_layout)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Search range xy (px):"))
        self._search_range_xy_spin = QDoubleSpinBox()
        self._search_range_xy_spin.setRange(0.1, 100.0)
        self._search_range_xy_spin.setSingleStep(0.5)
        self._search_range_xy_spin.setValue(4.0)
        range_row.addWidget(self._search_range_xy_spin)
        range_row.addWidget(QLabel("z (px):"))
        self._search_range_z_spin = QDoubleSpinBox()
        self._search_range_z_spin.setRange(0.1, 100.0)
        self._search_range_z_spin.setSingleStep(0.5)
        self._search_range_z_spin.setValue(4.0)
        range_row.addWidget(self._search_range_z_spin)
        link_layout.addLayout(range_row)

        self._link_calibration_check = QCheckBox("Link to calibration")
        self._link_calibration_check.setToolTip(
            "Derive the z search range from the xy one so both span the same "
            "distance in µm: z = xy × (pixel size xy / voxel size z)."
        )
        link_layout.addWidget(self._link_calibration_check)

        mem_row = QHBoxLayout()
        mem_row.addWidget(QLabel("Memory (frames):"))
        self._memory_spin = QSpinBox()
        self._memory_spin.setRange(0, 50)
        self._memory_spin.setValue(3)
        mem_row.addWidget(self._memory_spin)
        mem_row.addWidget(QLabel("Min track length:"))
        self._min_length_spin = QSpinBox()
        self._min_length_spin.setRange(1, 100000)
        self._min_length_spin.setValue(1)
        mem_row.addWidget(self._min_length_spin)
        link_layout.addLayout(mem_row)

        self._use_fit_check = QCheckBox("Use fitted centres")
        self._use_fit_check.setChecked(True)
        self._use_fit_check.setToolTip(
            "Link on the Gaussian-fit centre where the fit converged, the "
            "detected centre elsewhere. No effect on a CSV without fits."
        )
        link_layout.addWidget(self._use_fit_check)
        self._adaptive_check = QCheckBox("Adaptive search")
        self._adaptive_check.setToolTip(
            "In regions too dense to link, shrink the search range step by "
            "step instead of failing."
        )
        link_layout.addWidget(self._adaptive_check)
        layout.addWidget(link_group)

        out_group = QGroupBox("Output")
        out_layout = QHBoxLayout()
        out_group.setLayout(out_layout)
        self._output_path_edit = QLineEdit()
        out_layout.addWidget(self._output_path_edit)
        browse_out_btn = QPushButton("Browse…")
        browse_out_btn.clicked.connect(self._browse_output_csv)
        out_layout.addWidget(browse_out_btn)
        layout.addWidget(out_group)

        self._run_btn = QPushButton("Run Tracking")
        self._run_btn.clicked.connect(self._run_tracking)
        layout.addWidget(self._run_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("%v / %m")
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip(
            "Stops after the current frame. The tracks linked so far are kept."
        )
        layout.addWidget(self._cancel_btn)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

        self._init_run_state(
            run_buttons=[self._run_btn],
            cancel_button=self._cancel_btn,
            progress_bar=self._progress_bar,
            status_label=self._status_label,
        )
        self._update_z_range()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_points_combo)
        self.viewer.layers.events.removed.connect(self._refresh_points_combo)
        self._points_combo.currentTextChanged.connect(self._on_points_layer_chosen)
        self._csv_path_edit.editingFinished.connect(self._on_csv_path_typed)
        for spin in (self._search_range_xy_spin, self._pixel_size_xy_spin, self._voxel_size_z_spin):
            spin.valueChanged.connect(self._update_z_range)
        self._link_calibration_check.toggled.connect(self._update_z_range)
        self._refresh_points_combo()

    # ── input ────────────────────────────────────────────────────────

    def _refresh_points_combo(self, event=None):
        refresh_layer_combo(
            self._points_combo, self.viewer, napari.layers.Points,
            predicate=lambda layer: bool(layer.metadata.get("csv_path")),
        )
        # The refresh is silent (see `refresh_layer_combo`), so the first
        # spots layer is adopted here — but only into an empty input: a
        # layer added elsewhere must not replace a CSV the user browsed to.
        if not self._csv_path_edit.text() and self._points_combo.currentText():
            self._on_points_layer_chosen(self._points_combo.currentText())

    def _on_points_layer_chosen(self, name: str):
        if not name or name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        self._set_input_csv(layer.metadata["csv_path"], layer)

    def _on_csv_path_typed(self):
        text = self._csv_path_edit.text().strip()
        if text and text != self._last_input_csv:
            self._set_input_csv(text)

    def _browse_input_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open detection CSV", "", "CSV files (*.csv)")
        if path:
            self._set_input_csv(path)

    def _browse_output_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save tracks CSV", self._output_path_edit.text(), "CSV files (*.csv)"
        )
        if path:
            self._output_path_edit.setText(path)

    def _set_input_csv(self, csv_path, layer=None):
        """Make `csv_path` the input: default the output beside it and
        prefill calibration — the layer's values first, then the CSV's
        sidecar, else blank."""
        self._source_layer = layer
        self._last_input_csv = str(csv_path)
        self._csv_path_edit.setText(str(csv_path))
        self._output_path_edit.setText(str(default_tracks_path(csv_path)))

        from_sidecar = read_sidecar_calibration(csv_path)
        from_layer = layer.metadata if layer is not None else {}
        calibration = {k: from_layer.get(k) or from_sidecar.get(k) for k in CALIBRATION_KEYS}
        self._pixel_size_xy_spin.setValue(calibration["pixel_size_xy_um"] or 0.0)
        self._voxel_size_z_spin.setValue(calibration["voxel_size_z_um"] or 0.0)
        self._frame_interval_spin.setValue(calibration["frame_interval_s"] or 0.0)

        try:
            frames = pd.read_csv(csv_path, usecols=["frame"])["frame"]
        except (OSError, ValueError) as exc:
            self._csv_info_label.setText(f"Cannot read this CSV: {exc}")
        else:
            self._csv_info_label.setText(
                f"{len(frames)} spots across {frames.nunique()} timepoints."
            )

    # ── parameters ───────────────────────────────────────────────────

    def _calibration(self) -> dict:
        return {
            "pixel_size_xy_um": self._pixel_size_xy_spin.value() or None,
            "voxel_size_z_um": self._voxel_size_z_spin.value() or None,
            "frame_interval_s": self._frame_interval_spin.value() or None,
        }

    def _update_z_range(self, *_):
        linked = self._link_calibration_check.isChecked()
        self._search_range_z_spin.setEnabled(not linked)
        calibration = self._calibration()
        if linked and calibration["pixel_size_xy_um"] and calibration["voxel_size_z_um"]:
            self._search_range_z_spin.setValue(isotropic_zyx(
                self._search_range_xy_spin.value(),
                calibration["pixel_size_xy_um"],
                calibration["voxel_size_z_um"],
            )[0])

    def _search_range_zyx(self) -> tuple[float, float, float] | None:
        """The `(z, y, x)` search range in voxels, or None (after
        `show_error`) when linked to a calibration that is blank.

        Computed here, not read back from the z box: the box rounds to two
        decimals, and linking should use the exact ratio.
        """
        xy = self._search_range_xy_spin.value()
        if not self._link_calibration_check.isChecked():
            return (self._search_range_z_spin.value(), xy, xy)
        calibration = self._calibration()
        if not calibration["pixel_size_xy_um"] or not calibration["voxel_size_z_um"]:
            show_error(
                "The search range is linked to calibration, but the pixel and "
                "voxel sizes are blank — fill them in, or untick Link to "
                "calibration."
            )
            return None
        return isotropic_zyx(xy, calibration["pixel_size_xy_um"], calibration["voxel_size_z_um"])

    # ── run ──────────────────────────────────────────────────────────

    def _run_tracking(self):
        csv_text = self._csv_path_edit.text().strip()
        if not csv_text:
            show_error("No detection CSV selected.")
            return
        csv_path = Path(csv_text)
        if not csv_path.is_file():
            show_error(f"Detection CSV not found: {csv_path}")
            return
        out_text = self._output_path_edit.text().strip()
        if not out_text:
            show_error("No output path for the tracks CSV.")
            return
        out_path = Path(out_text)
        if same_file(csv_path, out_path):
            show_error(
                "The tracks CSV would overwrite the detection CSV — pick a "
                "different output path."
            )
            return
        if same_file(calibration_sidecar_path(out_path), calibration_sidecar_path(csv_path)):
            show_error(
                "The tracks sidecar would overwrite the detection CSV's "
                "sidecar — give the output path a different name, ending "
                "in .csv."
            )
            return
        search_range = self._search_range_zyx()
        if search_range is None:
            return

        calibration = self._calibration()
        record = {
            **calibration,
            "search_range_zyx": [float(v) for v in search_range],
            "link_to_calibration": self._link_calibration_check.isChecked(),
            "memory": self._memory_spin.value(),
            "min_length": self._min_length_spin.value(),
            "use_fit": self._use_fit_check.isChecked(),
            "adaptive": self._adaptive_check.isChecked(),
            "source_csv": str(csv_path),
            "trackpy_version": trackpy.__version__,
        }
        source_layer = self._source_layer
        if source_layer is not None and (
            source_layer not in self.viewer.layers
            or not same_file(source_layer.metadata.get("csv_path", ""), csv_path)
        ):
            source_layer = None
        name_base = source_layer.name if source_layer is not None else csv_path.stem

        worker = TrackingWorker3D(
            csv_path=csv_path,
            output_path=out_path,
            search_range_zyx=search_range,
            memory=record["memory"],
            min_length=record["min_length"],
            use_fit=record["use_fit"],
            adaptive=record["adaptive"],
            calibration=calibration,
        )

        def _on_finished(written: str):
            if not written:
                show_info("Tracking stopped before any spot was linked — nothing written.")
                return
            sidecar = calibration_sidecar_path(written)
            # A cancelled run's CSV is a prefix of the series; the sidecar is
            # what says so (the layer name does not outlive the session).
            run_record = {
                **record,
                "cancelled": self._cancel_requested,
                "first_frame": worker.first_frame,
                "last_frame": worker.last_frame,
            }
            try:
                sidecar.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
            except OSError as exc:
                show_error(f"Could not write the tracking sidecar {sidecar}: {exc}")
            tracks = pd.read_csv(written)
            if tracks.empty:
                show_info("No tracks left after the minimum-length filter.")
                return
            self._add_tracks_layer(
                tracks,
                name=f"{name_base} (tracks){self.cancelled_suffix()}",
                calibration=calibration,
                source_layer=source_layer,
                csv_path=written,
            )

        self._summary = ""
        # The frame count is only known once the worker has read the CSV;
        # its first `progress` sets the bar's maximum.
        self._begin_run(worker, total=0, description="Linking tracks", on_finished=_on_finished)

    def _add_tracks_layer(self, tracks, *, name, calibration, source_layer, csv_path):
        # napari reorders Tracks rows by (track, time) internally; sorting
        # first keeps `features` aligned with the rows they describe.
        tracks = tracks.sort_values(["particle", "frame"], kind="stable").reset_index(drop=True)
        data = tracks[["particle", "frame", *LINK_COLUMNS]].to_numpy(dtype=float)
        # The spots layer is what the tracks must overlay, so its scale wins
        # over typed calibration (which may describe an uncalibrated image).
        if source_layer is not None:
            scale = tuple(source_layer.scale)
        elif calibration["pixel_size_xy_um"] and calibration["voxel_size_z_um"]:
            scale = compute_layer_scale(
                calibration["pixel_size_xy_um"], calibration["voxel_size_z_um"]
            )
        else:
            scale = (1.0, 1.0, 1.0, 1.0)
        features = tracks[[c for c in FEATURE_COLUMNS if c in tracks.columns]]
        self.viewer.add_tracks(
            data,
            name=name,
            scale=scale,
            features=features,
            metadata={"tracks_csv_path": csv_path, **calibration},
        )
        n_censored = tracks.loc[tracks["censored"], "particle"].nunique()
        self._summary = (
            f"{tracks['particle'].nunique()} tracks, {len(tracks)} spots, "
            f"{n_censored} censored, across {tracks['frame'].nunique()} frames."
        )
        show_info(self._summary)

    def _on_run_finished(self, payload=None):
        # The mixin writes "Done." / "Cancelled at …" after the host's
        # handler; the summary goes after that so it is what stays visible.
        super()._on_run_finished(payload)
        if self._summary:
            self._set_status(f"{self._status_label.text()} {self._summary}")
