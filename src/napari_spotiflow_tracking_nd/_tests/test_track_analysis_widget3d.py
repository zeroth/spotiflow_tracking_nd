from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CALIBRATION = {"pixel_size_xy_um": 0.104, "voxel_size_z_um": 0.210, "frame_interval_s": 3.0}


@pytest.fixture
def analysis_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._track_analysis_widget3d import TrackAnalysisWidget3D

    viewer = make_napari_viewer()
    widget = TrackAnalysisWidget3D(viewer)
    # Synchronous through the seam; real threading is covered by
    # test_threading.py.
    widget._launch_worker = lambda worker: worker.run()
    return widget, viewer


def _write_tracks(folder: Path, name="spots_tracks.csv", sidecar=CALIBRATION, n_frames=12):
    """Three tracks — two jittering in place (censored: they span the
    series), one living frames 3..7 — plus the tracking sidecar."""
    rng = np.random.default_rng(0)
    rows = []
    for track_id, frames in ((0, range(n_frames)), (1, range(n_frames)), (2, range(3, 8))):
        for t in frames:
            rows.append({
                "particle": track_id, "frame": t,
                "link_z": 5 + 0.1 * rng.normal(), "link_y": 20.0 * track_id + 0.1 * rng.normal(),
                "link_x": 10.0, "censored": track_id != 2,
            })
    folder.mkdir(parents=True, exist_ok=True)
    csv = folder / name
    pd.DataFrame(rows).to_csv(csv, index=False)
    if sidecar is not None:
        csv.with_suffix(".json").write_text(json.dumps(sidecar))
    return csv


def _add_tracks_layer(viewer, csv, name="vol (tracks)", **metadata):
    df = pd.read_csv(csv).sort_values(["particle", "frame"])
    return viewer.add_tracks(
        df[["particle", "frame", "link_z", "link_y", "link_x"]].to_numpy(dtype=float),
        name=name,
        metadata={"tracks_csv_path": str(csv), **metadata},
    )


def _output_folder(tmp_path):
    return tmp_path / "out" / "spots_tracks_analysis"


class TestCreation:
    def test_has_the_expected_controls(self, analysis_widget):
        widget, _ = analysis_widget
        for attr in (
            "_tracks_combo", "_csv_path_edit", "_browse_csv_btn", "_csv_info_label",
            "_pixel_size_xy_spin", "_voxel_size_z_spin", "_frame_interval_spin",
            "_max_lag_spin", "_min_msd_spots_spin", "_motility_combo",
            "_step_window_spin", "_step_threshold_spin", "_output_dir_edit",
            "_run_btn", "_cancel_btn", "_progress_bar", "_status_label",
        ):
            assert hasattr(widget, attr), f"missing {attr}"

    def test_defaults(self, analysis_widget):
        widget, _ = analysis_widget
        assert widget._max_lag_spin.value() == 25
        assert widget._min_msd_spots_spin.value() == 10
        assert widget._motility_combo.currentText() == "3D"
        assert widget._step_window_spin.value() == 3
        assert widget._step_threshold_spin.value() == 0.5

    def test_registered_in_the_manifest(self):
        from importlib.resources import files

        manifest = files("napari_spotiflow_tracking_nd").joinpath("napari.yaml").read_text()
        assert "_track_analysis_widget3d:TrackAnalysisWidget3D" in manifest
        assert "Track Analysis" in manifest


class TestInput:
    def test_only_tracks_layers_from_tracking_are_listed(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        viewer.add_tracks(np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1.0]]), name="other")
        _add_tracks_layer(viewer, _write_tracks(tmp_path))

        listed = [widget._tracks_combo.itemText(i) for i in range(widget._tracks_combo.count())]

        assert listed == ["vol (tracks)"]

    def test_a_tracks_layer_fills_input_output_and_counts(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        csv = _write_tracks(tmp_path)

        _add_tracks_layer(viewer, csv)

        assert widget._csv_path_edit.text() == str(csv)
        assert widget._output_dir_edit.text() == str(tmp_path / "spots_tracks_analysis")
        assert "3 tracks" in widget._csv_info_label.text()

    def test_a_typed_path_is_adopted(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        csv = _write_tracks(tmp_path / "b")

        widget._csv_path_edit.setText(str(csv))
        widget._csv_path_edit.editingFinished.emit()

        assert widget._output_dir_edit.text() == str(tmp_path / "b" / "spots_tracks_analysis")
        assert widget._pixel_size_xy_spin.value() == pytest.approx(0.104)


class TestCalibrationPrefill:
    def test_layer_values_win_then_the_sidecar_fills_the_gaps(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        csv = _write_tracks(tmp_path, sidecar={**CALIBRATION, "pixel_size_xy_um": 0.1})

        _add_tracks_layer(viewer, csv, pixel_size_xy_um=0.2)

        assert widget._pixel_size_xy_spin.value() == pytest.approx(0.2)
        assert widget._voxel_size_z_spin.value() == pytest.approx(0.210)
        assert widget._frame_interval_spin.value() == pytest.approx(3.0)


class TestRefusals:
    def test_incomplete_calibration_is_refused(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path, sidecar=None))
        widget._output_dir_edit.setText(str(_output_folder(tmp_path)))
        widget._pixel_size_xy_spin.setValue(0.104)
        widget._voxel_size_z_spin.setValue(0.210)  # frame interval left blank

        widget._run_analysis()

        assert not _output_folder(tmp_path).exists()
        assert not widget._is_running

    def test_the_tracks_csvs_own_folder_is_refused(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path))
        widget._output_dir_edit.setText(str(tmp_path / "sub" / ".."))
        before = sorted(p.name for p in tmp_path.iterdir())

        widget._run_analysis()

        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_an_output_path_that_is_a_file_is_refused(self, analysis_widget, tmp_path):
        # Review focus: replace_dir would otherwise move the file aside.
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path))
        target = tmp_path / "out" / "results.csv"
        target.parent.mkdir()
        target.write_text("precious")
        widget._output_dir_edit.setText(str(target))

        widget._run_analysis()

        assert target.read_text() == "precious"
        assert sorted(p.name for p in target.parent.iterdir()) == ["results.csv"]

    def test_a_missing_tracks_csv_is_refused(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        csv = _write_tracks(tmp_path)
        widget._set_input_csv(csv)
        widget._output_dir_edit.setText(str(_output_folder(tmp_path)))
        csv.unlink()

        widget._run_analysis()

        assert not _output_folder(tmp_path).exists()
        assert not widget._is_running


class TestRun:
    def test_writes_the_results_folder_with_its_record(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        csv = _write_tracks(tmp_path)
        widget._set_input_csv(csv)
        out = _output_folder(tmp_path)
        widget._output_dir_edit.setText(str(out))
        widget._motility_combo.setCurrentText("xy")

        widget._run_analysis()

        assert sorted(p.name for p in out.iterdir()) == [
            "analysis.json", "lifetime_distribution.csv", "msd_curves.csv",
            "steps.csv", "track_properties.csv",
        ]
        record = json.loads((out / "analysis.json").read_text())
        assert record["pixel_size_xy_um"] == pytest.approx(0.104)
        assert record["frame_interval_s"] == 3.0
        assert record["max_lag"] == 25 and record["min_msd_spots"] == 10
        assert record["motility_from"] == "xy"
        assert record["step_window"] == 3 and record["step_threshold"] == 0.5
        assert record["source_csv"] == str(csv)
        assert record["tracking_cancelled"] is None
        assert {"numpy", "scipy", "pandas", "plugin"} <= set(record["versions"])

    def test_per_track_results_are_joined_onto_the_tracks_layer(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        layer = _add_tracks_layer(viewer, _write_tracks(tmp_path))
        widget._output_dir_edit.setText(str(_output_folder(tmp_path)))
        n_vertices = len(layer.data)

        widget._run_analysis()

        features = layer.features
        assert len(features) == n_vertices
        for name in ("lifetime_s", "D_3d", "alpha_3d", "motility_class", "n_steps"):
            assert name in features.columns, f"missing feature {name}"
        short = features.loc[features["track_id"] == 2, "lifetime_frames"]
        assert (short == 5).all()

    def test_a_rerun_overwrites_the_previous_runs_features(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        layer = _add_tracks_layer(viewer, _write_tracks(tmp_path))
        out = _output_folder(tmp_path)
        widget._output_dir_edit.setText(str(out))
        n_vertices = len(layer.data)
        widget._run_analysis()

        widget._motility_combo.setCurrentText("xy")
        widget._max_lag_spin.setValue(10)
        widget._run_analysis()

        features = layer.features
        assert len(features) == n_vertices
        second = pd.read_csv(out / "track_properties.csv").set_index("track_id")
        for column in ("motility_class", "alpha_xy"):
            expected = features["track_id"].map(second[column])
            pd.testing.assert_series_equal(
                features[column].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
            )

    def test_the_status_reports_the_summary(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path))
        widget._output_dir_edit.setText(str(_output_folder(tmp_path)))

        widget._run_analysis()

        text = widget._status_label.text()
        assert "3 tracks" in text
        assert "1 uncensored" in text
        assert "median lifetime 15 s" in text

    def test_tracks_from_a_cancelled_tracking_run_warn_and_are_recorded(
        self, analysis_widget, tmp_path, monkeypatch
    ):
        # Review focus: lifetimes near the end of a cancelled run are cut short.
        from napari_spotiflow_tracking_nd import _track_analysis_widget3d

        warnings = []
        monkeypatch.setattr(_track_analysis_widget3d, "show_warning", warnings.append)
        widget, _ = analysis_widget
        csv = _write_tracks(
            tmp_path, sidecar={**CALIBRATION, "cancelled": True, "first_frame": 0, "last_frame": 11},
        )
        widget._set_input_csv(csv)
        out = _output_folder(tmp_path)
        widget._output_dir_edit.setText(str(out))

        widget._run_analysis()

        assert len(warnings) == 1 and "cancelled" in warnings[0]
        record = json.loads((out / "analysis.json").read_text())
        assert record["tracking_cancelled"] is True
        assert record["tracking_last_frame"] == 11

    def test_a_cancelled_analysis_writes_nothing(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path))
        out = _output_folder(tmp_path)
        widget._output_dir_edit.setText(str(out))

        def _launch(worker):
            # Through the widget's own cancel path, as the Cancel button
            # does — so the status reports the cancel.
            widget._request_cancel()
            worker.run()

        widget._launch_worker = _launch
        widget._run_analysis()

        assert not out.exists()
        assert "Cancelled" in widget._status_label.text()


class TestSummarise:
    def test_counts_and_percentages(self):
        from napari_spotiflow_tracking_nd._track_analysis_widget3d import summarise

        props = pd.DataFrame({
            "censored": [False, False, True, False],
            "lifetime_s": [9.0, 15.0, 30.0, 21.0],
            "motility_class": ["diffusive", "directed", "diffusive", "unclassified"],
        })

        assert summarise(props) == (
            "4 tracks — 3 uncensored, median lifetime 15 s; "
            "3 with D/α (67% diffusive)."
        )

    def test_nothing_uncensored_or_fitted(self):
        from napari_spotiflow_tracking_nd._track_analysis_widget3d import summarise

        props = pd.DataFrame({
            "censored": [True], "lifetime_s": [9.0], "motility_class": ["unclassified"],
        })

        assert summarise(props) == "1 tracks — 0 uncensored; 0 with D/α."
