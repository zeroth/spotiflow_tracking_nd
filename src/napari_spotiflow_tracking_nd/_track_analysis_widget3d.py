"""The Track Analysis widget: per-track lifetimes, motion and intensity
steps from a tracks CSV, written to a results folder."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import scipy
from napari.utils.notifications import show_error, show_info, show_warning
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

from napari_spotiflow_tracking_nd import __version__ as PLUGIN_VERSION
from napari_spotiflow_tracking_nd._detection_widget3d import CALIBRATION_KEYS
from napari_spotiflow_tracking_nd._layer_utils import calibration_spin, refresh_layer_combo
from napari_spotiflow_tracking_nd._run_state import RunStateMixin
from napari_spotiflow_tracking_nd._track_analysis3d import AnalysisParams
from napari_spotiflow_tracking_nd._tracking_widget3d import (
    read_sidecar,
    read_sidecar_calibration,
    same_file,
)
from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

#: The motility combo's labels and the `AnalysisParams.motility_from` they mean.
MOTILITY_FROM = {"3D": "3d", "xy": "xy"}


def default_analysis_dir(csv_path) -> Path:
    """`spots_tracks.csv` -> `spots_tracks_analysis/`, beside it."""
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_analysis")


def summarise(properties: pd.DataFrame) -> str:
    """One status line: track count, uncensored lifetimes, motility."""
    n = len(properties)
    done = properties.loc[~properties["censored"].astype(bool), "lifetime_s"]
    text = f"{n} tracks — {len(done)} uncensored"
    if len(done):
        text += f", median lifetime {done.median():.3g} s"
    fitted = properties["motility_class"] != "unclassified"
    n_fitted = int(fitted.sum())
    text += f"; {n_fitted} with D/α"
    if n_fitted:
        diffusive = (properties.loc[fitted, "motility_class"] == "diffusive").mean()
        text += f" ({diffusive * 100:.0f}% diffusive)"
    return text + "."


class TrackAnalysisWidget3D(RunStateMixin, QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        # The Tracks layer the input CSV came from, if any — the per-track
        # results are joined onto its `features`.
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
        in_layout.addWidget(QLabel("Tracks layer:"))
        self._tracks_combo = QComboBox()
        in_layout.addWidget(self._tracks_combo)
        csv_row = QHBoxLayout()
        self._csv_path_edit = QLineEdit()
        self._csv_path_edit.setPlaceholderText("Tracks CSV")
        csv_row.addWidget(self._csv_path_edit)
        self._browse_csv_btn = QPushButton("Browse CSV…")
        self._browse_csv_btn.clicked.connect(self._browse_input_csv)
        csv_row.addWidget(self._browse_csv_btn)
        in_layout.addLayout(csv_row)
        self._csv_info_label = QLabel()
        in_layout.addWidget(self._csv_info_label)
        layout.addWidget(in_group)

        calib_group = QGroupBox("Calibration (all required)")
        calib_layout = QHBoxLayout()
        calib_group.setLayout(calib_layout)
        calib_layout.addWidget(QLabel("xy (µm):"))
        self._pixel_size_xy_spin = calibration_spin(100.0, 4)
        calib_layout.addWidget(self._pixel_size_xy_spin)
        calib_layout.addWidget(QLabel("z (µm):"))
        self._voxel_size_z_spin = calibration_spin(100.0, 4)
        calib_layout.addWidget(self._voxel_size_z_spin)
        calib_layout.addWidget(QLabel("dt (s):"))
        self._frame_interval_spin = calibration_spin(10000.0, 3)
        calib_layout.addWidget(self._frame_interval_spin)
        layout.addWidget(calib_group)

        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        params_group.setLayout(params_layout)

        msd_row = QHBoxLayout()
        msd_row.addWidget(QLabel("Max lag (frames):"))
        self._max_lag_spin = QSpinBox()
        self._max_lag_spin.setRange(1, 1000)
        self._max_lag_spin.setValue(25)
        msd_row.addWidget(self._max_lag_spin)
        msd_row.addWidget(QLabel("Min spots for MSD:"))
        self._min_msd_spots_spin = QSpinBox()
        self._min_msd_spots_spin.setRange(2, 100000)
        self._min_msd_spots_spin.setValue(10)
        msd_row.addWidget(self._min_msd_spots_spin)
        params_layout.addLayout(msd_row)

        motility_row = QHBoxLayout()
        motility_row.addWidget(QLabel("Motility from:"))
        self._motility_combo = QComboBox()
        self._motility_combo.addItems(list(MOTILITY_FROM))
        self._motility_combo.setToolTip(
            "Which α sets the motility class. Axial localisation is usually "
            "2–3× worse than lateral; xy ignores it."
        )
        motility_row.addWidget(self._motility_combo)
        motility_row.addStretch()
        params_layout.addLayout(motility_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step window:"))
        self._step_window_spin = QSpinBox()
        self._step_window_spin.setRange(1, 200)
        self._step_window_spin.setValue(3)
        step_row.addWidget(self._step_window_spin)
        step_row.addWidget(QLabel("Step threshold:"))
        self._step_threshold_spin = QDoubleSpinBox()
        self._step_threshold_spin.setRange(0.0, 1.0)
        self._step_threshold_spin.setSingleStep(0.05)
        self._step_threshold_spin.setDecimals(2)
        self._step_threshold_spin.setValue(0.5)
        step_row.addWidget(self._step_threshold_spin)
        params_layout.addLayout(step_row)
        layout.addWidget(params_group)

        out_group = QGroupBox("Results folder")
        out_layout = QHBoxLayout()
        out_group.setLayout(out_layout)
        self._output_dir_edit = QLineEdit()
        out_layout.addWidget(self._output_dir_edit)
        browse_out_btn = QPushButton("Browse…")
        browse_out_btn.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(browse_out_btn)
        layout.addWidget(out_group)

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.clicked.connect(self._run_analysis)
        layout.addWidget(self._run_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("%v / %m")
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip(
            "Stops after the current track and writes nothing — a partial "
            "analysis would be a biased sample of the tracks."
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

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_tracks_combo)
        self.viewer.layers.events.removed.connect(self._refresh_tracks_combo)
        self._tracks_combo.currentTextChanged.connect(self._on_tracks_layer_chosen)
        self._csv_path_edit.editingFinished.connect(self._on_csv_path_typed)
        self._refresh_tracks_combo()

    # ── input ────────────────────────────────────────────────────────

    def _refresh_tracks_combo(self, event=None):
        refresh_layer_combo(
            self._tracks_combo, self.viewer, napari.layers.Tracks,
            predicate=lambda layer: bool(layer.metadata.get("tracks_csv_path")),
        )
        # The refresh is silent, so the first tracks layer is adopted here
        # — but only into an empty input, never over a browsed CSV.
        if not self._csv_path_edit.text() and self._tracks_combo.currentText():
            self._on_tracks_layer_chosen(self._tracks_combo.currentText())

    def _on_tracks_layer_chosen(self, name: str):
        if not name or name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        self._set_input_csv(layer.metadata["tracks_csv_path"], layer)

    def _on_csv_path_typed(self):
        text = self._csv_path_edit.text().strip()
        if text and text != self._last_input_csv:
            self._set_input_csv(text)

    def _browse_input_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open tracks CSV", "", "CSV files (*.csv)")
        if path:
            self._set_input_csv(path)

    def _browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select results folder")
        if folder:
            self._output_dir_edit.setText(folder)

    def _set_input_csv(self, csv_path, layer=None):
        """Make `csv_path` the input: default the results folder beside it
        and prefill calibration — the layer's values first, then the
        tracks sidecar, else blank."""
        self._source_layer = layer
        self._last_input_csv = str(csv_path)
        self._csv_path_edit.setText(str(csv_path))
        self._output_dir_edit.setText(str(default_analysis_dir(csv_path)))

        from_sidecar = read_sidecar_calibration(csv_path)
        from_layer = layer.metadata if layer is not None else {}
        calibration = {k: from_layer.get(k) or from_sidecar.get(k) for k in CALIBRATION_KEYS}
        self._pixel_size_xy_spin.setValue(calibration["pixel_size_xy_um"] or 0.0)
        self._voxel_size_z_spin.setValue(calibration["voxel_size_z_um"] or 0.0)
        self._frame_interval_spin.setValue(calibration["frame_interval_s"] or 0.0)

        try:
            ids = pd.read_csv(csv_path, usecols=["particle"])["particle"]
        except (OSError, ValueError) as exc:
            self._csv_info_label.setText(f"Cannot read this CSV: {exc}")
        else:
            self._csv_info_label.setText(f"{ids.nunique()} tracks, {len(ids)} spots.")

    # ── parameters ───────────────────────────────────────────────────

    def _params(self) -> AnalysisParams | None:
        """The run's parameters, or None (after `show_error`) when any
        calibration value is blank — a 3D distance in raw voxels mixes
        lateral and axial steps of different sizes."""
        pixel_xy = self._pixel_size_xy_spin.value()
        voxel_z = self._voxel_size_z_spin.value()
        frame_s = self._frame_interval_spin.value()
        if not (pixel_xy and voxel_z and frame_s):
            show_error(
                "Track analysis needs the pixel size (xy), voxel size (z) and "
                "frame interval — fill in all three."
            )
            return None
        return AnalysisParams(
            pixel_size_xy_um=pixel_xy,
            voxel_size_z_um=voxel_z,
            frame_interval_s=frame_s,
            max_lag=self._max_lag_spin.value(),
            min_msd_spots=self._min_msd_spots_spin.value(),
            motility_from=MOTILITY_FROM[self._motility_combo.currentText()],
            step_window=self._step_window_spin.value(),
            step_threshold=self._step_threshold_spin.value(),
        )

    # ── run ──────────────────────────────────────────────────────────

    def _run_analysis(self):
        csv_text = self._csv_path_edit.text().strip()
        if not csv_text:
            show_error("No tracks CSV selected.")
            return
        csv_path = Path(csv_text)
        if not csv_path.is_file():
            show_error(f"Tracks CSV not found: {csv_path}")
            return
        out_text = self._output_dir_edit.text().strip()
        if not out_text:
            show_error("No results folder selected.")
            return
        out_dir = Path(out_text)
        if same_file(out_dir, csv_path.parent):
            show_error(
                "The results folder would be the tracks CSV's own folder — "
                "pick a different one."
            )
            return
        if out_dir.exists() and not out_dir.is_dir():
            show_error(f"{out_dir} is a file, not a folder — pick a different results folder.")
            return
        params = self._params()
        if params is None:
            return

        sidecar = read_sidecar(csv_path)
        if sidecar.get("cancelled"):
            show_warning(
                "These tracks come from a cancelled tracking run (frames "
                f"{sidecar.get('first_frame')}–{sidecar.get('last_frame')}) — "
                "lifetimes near the end of that range are cut short."
            )
        record = {
            **asdict(params),
            "source_csv": str(csv_path),
            "tracking_cancelled": sidecar.get("cancelled"),
            "tracking_first_frame": sidecar.get("first_frame"),
            "tracking_last_frame": sidecar.get("last_frame"),
            "versions": {
                "plugin": PLUGIN_VERSION,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
        }
        source_layer = self._source_layer
        if source_layer is not None and (
            source_layer not in self.viewer.layers
            or not same_file(source_layer.metadata.get("tracks_csv_path", ""), csv_path)
        ):
            source_layer = None

        worker = TrackAnalysisWorker3D(csv_path, out_dir, params, record)

        def _on_finished(written: str):
            if not written:
                show_info("Analysis cancelled — nothing written.")
                return
            properties = pd.read_csv(Path(written) / "track_properties.csv")
            if source_layer is not None:
                self._attach_features(source_layer, properties)
            self._summary = summarise(properties)
            show_info(self._summary)

        self._summary = ""
        # The track count is only known once the worker has read the CSV;
        # its first `progress` sets the bar's maximum.
        self._begin_run(worker, total=0, description="Analysing tracks", on_finished=_on_finished)

    @staticmethod
    def _attach_features(layer, properties: pd.DataFrame):
        """Join the per-track columns onto `layer`'s per-vertex `features`
        by `track_id`, so napari's colour-by-feature can use them. Every
        analysis column is overwritten, so a re-run replaces the previous
        run's values; only `track_id` and `censored` (owned by tracking,
        identical per track) are left as the layer has them."""
        features = layer.features.copy()
        table = properties.set_index("track_id")
        for column in table.columns:
            if column != "censored":
                features[column] = features["track_id"].map(table[column]).to_numpy()
        layer.features = features

    def _on_run_finished(self, payload=None):
        # The mixin writes "Done." / "Cancelled at …" after the host's
        # handler; the summary goes after that so it is what stays visible.
        super()._on_run_finished(payload)
        if self._summary:
            self._set_status(f"{self._status_label.text()} {self._summary}")
