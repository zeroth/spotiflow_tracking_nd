from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tracking_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._tracking_widget3d import TrackingWidget3D

    viewer = make_napari_viewer()
    widget = TrackingWidget3D(viewer)
    # Synchronous through the seam; real threading is covered by
    # test_threading.py.
    widget._launch_worker = lambda worker: worker.run()
    return widget, viewer


def _write_spots_csv(path, n_frames=4, fitted=False):
    rows = []
    for t in range(n_frames):
        rows.append({"frame": t, "z": 5.0, "y": 10.0, "x": 10.0, "prob": 0.9})
        rows.append({"frame": t, "z": 20.0, "y": 40.0, "x": 40.0, "prob": 0.8})
    df = pd.DataFrame(rows)
    if fitted:
        df["fit_z"], df["fit_y"], df["fit_x"] = df["z"] + 0.1, df["y"] + 0.1, df["x"] + 0.1
        df["amplitude"], df["integrated_intensity"], df["background"] = 100.0, 1000.0, 10.0
        df["converged"] = True
    df.to_csv(path, index=False)
    return Path(path)


def _add_spots_layer(viewer, csv_path, name="vol (spots)", **calibration):
    df = pd.read_csv(csv_path)
    return viewer.add_points(
        df[["frame", "z", "y", "x"]].to_numpy(),
        name=name,
        scale=(1.0, 0.210, 0.104, 0.104),
        metadata={
            "csv_path": str(csv_path),
            "pixel_size_xy_um": calibration.get("pixel_size_xy_um"),
            "voxel_size_z_um": calibration.get("voxel_size_z_um"),
            "frame_interval_s": calibration.get("frame_interval_s"),
        },
    )


def _tracks_layers(viewer):
    return [l for l in viewer.layers if l.__class__.__name__ == "Tracks"]


class TestCreation:
    def test_has_the_expected_controls(self, tracking_widget):
        widget, _ = tracking_widget
        for attr in (
            "_points_combo", "_csv_path_edit", "_browse_csv_btn", "_csv_info_label",
            "_pixel_size_xy_spin", "_voxel_size_z_spin", "_frame_interval_spin",
            "_search_range_xy_spin", "_link_calibration_check", "_search_range_z_spin",
            "_memory_spin", "_min_length_spin", "_use_fit_check", "_adaptive_check",
            "_output_path_edit", "_run_btn", "_cancel_btn", "_progress_bar", "_status_label",
        ):
            assert hasattr(widget, attr), f"missing {attr}"

    def test_defaults(self, tracking_widget):
        widget, _ = tracking_widget
        assert widget._search_range_xy_spin.value() == 4.0
        assert widget._memory_spin.value() == 3
        assert widget._min_length_spin.value() == 1
        assert widget._use_fit_check.isChecked()
        assert not widget._adaptive_check.isChecked()
        assert not widget._link_calibration_check.isChecked()

    def test_registered_in_the_manifest(self):
        from importlib.resources import files

        manifest = files("napari_spotiflow_tracking_nd").joinpath("napari.yaml").read_text()
        assert "_tracking_widget3d:TrackingWidget3D" in manifest
        assert "Spot Tracking" in manifest


class TestInput:
    def test_only_points_layers_from_detection_are_listed(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        viewer.add_points([[0, 1, 1, 1]], name="hand-drawn")
        _add_spots_layer(viewer, csv)

        listed = [widget._points_combo.itemText(i) for i in range(widget._points_combo.count())]

        assert listed == ["vol (spots)"]

    def test_a_spots_layer_fills_the_input_and_a_default_output(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")

        _add_spots_layer(viewer, csv)

        assert widget._csv_path_edit.text() == str(csv)
        assert widget._output_path_edit.text() == str(tmp_path / "spots_tracks.csv")
        assert "8 spots" in widget._csv_info_label.text()
        assert "4 timepoints" in widget._csv_info_label.text()

    def test_a_layer_added_later_does_not_replace_a_browsed_csv(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        browsed = _write_spots_csv(tmp_path / "browsed.csv")
        widget._set_input_csv(browsed)

        _add_spots_layer(viewer, _write_spots_csv(tmp_path / "other.csv"))

        assert widget._csv_path_edit.text() == str(browsed)


class TestCalibrationPrefill:
    def test_layer_values_win_then_the_sidecar_fills_the_gaps(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        (tmp_path / "spots.json").write_text(json.dumps(
            {"pixel_size_xy_um": 0.1, "voxel_size_z_um": 0.2, "frame_interval_s": 3.0}
        ))

        _add_spots_layer(viewer, csv, pixel_size_xy_um=0.104)

        assert widget._pixel_size_xy_spin.value() == pytest.approx(0.104)
        assert widget._voxel_size_z_spin.value() == pytest.approx(0.2)
        assert widget._frame_interval_spin.value() == pytest.approx(3.0)

    def test_null_everywhere_leaves_the_fields_blank(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        (tmp_path / "spots.json").write_text(json.dumps(
            {"pixel_size_xy_um": None, "voxel_size_z_um": None, "frame_interval_s": None}
        ))

        widget._set_input_csv(csv)

        assert widget._pixel_size_xy_spin.value() == 0
        assert widget._voxel_size_z_spin.value() == 0
        assert widget._frame_interval_spin.value() == 0

    def test_an_unreadable_sidecar_is_treated_as_absent(self, tracking_widget, tmp_path):
        widget, _ = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        (tmp_path / "spots.json").write_text("{not json")

        widget._set_input_csv(csv)

        assert widget._pixel_size_xy_spin.value() == 0

    def test_a_typed_path_re_adopts_output_and_calibration(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        a = _write_spots_csv(tmp_path / "a.csv")
        b = _write_spots_csv(tmp_path / "b.csv")
        (tmp_path / "a.json").write_text(json.dumps({"pixel_size_xy_um": 0.1}))
        (tmp_path / "b.json").write_text(json.dumps({"pixel_size_xy_um": 0.2}))
        _add_spots_layer(viewer, a)
        assert widget._pixel_size_xy_spin.value() == pytest.approx(0.1)

        widget._csv_path_edit.setText(str(b))
        widget._csv_path_edit.editingFinished.emit()

        assert widget._output_path_edit.text() == str(tmp_path / "b_tracks.csv")
        assert widget._pixel_size_xy_spin.value() == pytest.approx(0.2)
        assert widget._source_layer is None

    def test_finishing_an_unchanged_path_keeps_a_hand_edited_output(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        _add_spots_layer(viewer, _write_spots_csv(tmp_path / "a.csv"))
        widget._output_path_edit.setText(str(tmp_path / "mine.csv"))

        widget._csv_path_edit.editingFinished.emit()

        assert widget._output_path_edit.text() == str(tmp_path / "mine.csv")


class TestSearchRange:
    def test_unlinked_uses_the_z_box_as_typed(self, tracking_widget):
        widget, _ = tracking_widget
        widget._search_range_xy_spin.setValue(4.0)
        widget._search_range_z_spin.setValue(3.0)

        assert widget._search_range_zyx() == (3.0, 4.0, 4.0)

    def test_linked_derives_z_from_the_calibration(self, tracking_widget):
        widget, _ = tracking_widget
        widget._pixel_size_xy_spin.setValue(0.104)
        widget._voxel_size_z_spin.setValue(0.210)
        widget._search_range_xy_spin.setValue(4.0)
        widget._link_calibration_check.setChecked(True)

        assert widget._search_range_zyx() == pytest.approx((4.0 * 0.104 / 0.210, 4.0, 4.0))
        # Shown, read-only, in the z box.
        assert not widget._search_range_z_spin.isEnabled()
        assert widget._search_range_z_spin.value() == pytest.approx(1.98, abs=0.01)

    def test_linked_without_calibration_refuses_the_run(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        widget._set_input_csv(csv)
        widget._link_calibration_check.setChecked(True)

        widget._run_tracking()

        assert _tracks_layers(viewer) == []
        assert not (tmp_path / "spots_tracks.csv").exists()


class TestRun:
    def test_writes_the_csv_and_a_scaled_tracks_layer(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv", fitted=True)
        _add_spots_layer(viewer, csv)

        widget._run_tracking()

        out = tmp_path / "spots_tracks.csv"
        assert pd.read_csv(out)["particle"].nunique() == 2
        (layer,) = _tracks_layers(viewer)
        assert layer.name == "vol (spots) (tracks)"
        # Uncalibrated fields: the scale falls back to the spots layer's.
        assert tuple(layer.scale) == pytest.approx((1.0, 0.210, 0.104, 0.104))
        # Data is [particle, frame, link_z, link_y, link_x] in voxels —
        # the fitted centres here, since the fits converged.
        assert layer.data[0, 2:].tolist() == pytest.approx([5.1, 10.1, 10.1])
        for name in ("coord_source", "prob", "censored", "amplitude",
                     "integrated_intensity", "background"):
            assert name in layer.features.columns, f"missing feature {name}"
        assert layer.metadata["tracks_csv_path"] == str(out)

    def test_calibration_fields_set_the_scale(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        widget._set_input_csv(_write_spots_csv(tmp_path / "spots.csv"))
        widget._pixel_size_xy_spin.setValue(0.1)
        widget._voxel_size_z_spin.setValue(0.3)

        widget._run_tracking()

        (layer,) = _tracks_layers(viewer)
        assert tuple(layer.scale) == pytest.approx((1.0, 0.3, 0.1, 0.1))
        assert layer.name == "spots (tracks)"

    def test_typed_calibration_does_not_move_tracks_off_their_spots_layer(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        _add_spots_layer(viewer, _write_spots_csv(tmp_path / "spots.csv"))
        widget._pixel_size_xy_spin.setValue(0.5)
        widget._voxel_size_z_spin.setValue(0.5)

        widget._run_tracking()

        (layer,) = _tracks_layers(viewer)
        assert tuple(layer.scale) == pytest.approx((1.0, 0.210, 0.104, 0.104))

    def test_the_sidecar_records_calibration_and_parameters(self, tracking_widget, tmp_path):
        import trackpy

        widget, _ = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        widget._set_input_csv(csv)
        widget._frame_interval_spin.setValue(3.0)
        widget._memory_spin.setValue(2)

        widget._run_tracking()

        record = json.loads((tmp_path / "spots_tracks.json").read_text())
        assert record["frame_interval_s"] == 3.0
        assert record["pixel_size_xy_um"] is None
        assert record["search_range_zyx"] == [4.0, 4.0, 4.0]
        assert record["link_to_calibration"] is False
        assert record["memory"] == 2
        assert record["min_length"] == 1
        assert record["use_fit"] is True
        assert record["adaptive"] is False
        assert record["source_csv"] == str(csv)
        assert record["trackpy_version"] == trackpy.__version__

    def test_the_sidecar_marks_a_complete_run(self, tracking_widget, tmp_path):
        widget, _ = tracking_widget
        widget._set_input_csv(_write_spots_csv(tmp_path / "spots.csv", n_frames=4))

        widget._run_tracking()

        record = json.loads((tmp_path / "spots_tracks.json").read_text())
        frames = pd.read_csv(tmp_path / "spots_tracks.csv")["frame"]
        assert record["cancelled"] is False
        assert (record["first_frame"], record["last_frame"]) == (frames.min(), frames.max())

    def test_the_sidecar_marks_a_cancelled_run(self, tracking_widget, tmp_path):
        widget, _ = tracking_widget
        widget._set_input_csv(_write_spots_csv(tmp_path / "spots.csv", n_frames=10))
        widget._progress_bar.valueChanged.connect(
            lambda value: widget._cancel_btn.click() if value == 3 else None
        )

        widget._run_tracking()

        record = json.loads((tmp_path / "spots_tracks.json").read_text())
        assert record["cancelled"] is True
        assert (record["first_frame"], record["last_frame"]) == (0, 2)

    def test_an_output_whose_sidecar_would_clobber_the_detection_sidecar_is_refused(
        self, tracking_widget, tmp_path,
    ):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        (tmp_path / "spots.json").write_text('{"pixel_size_xy_um": 0.1}')
        widget._set_input_csv(csv)
        widget._output_path_edit.setText(str(tmp_path / "spots.txt"))

        widget._run_tracking()

        assert _tracks_layers(viewer) == []
        assert not (tmp_path / "spots.txt").exists()
        assert (tmp_path / "spots.json").read_text() == '{"pixel_size_xy_um": 0.1}'

    def test_the_status_reports_the_summary(self, tracking_widget, tmp_path):
        widget, _ = tracking_widget
        widget._set_input_csv(_write_spots_csv(tmp_path / "spots.csv"))

        widget._run_tracking()

        text = widget._status_label.text()
        assert "2 tracks" in text and "8 spots" in text and "2 censored" in text

    def test_the_input_csv_is_never_the_output(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        widget._set_input_csv(csv)
        widget._output_path_edit.setText(str(csv))
        before = csv.read_text()

        widget._run_tracking()

        assert csv.read_text() == before
        assert _tracks_layers(viewer) == []

    def test_the_input_csv_spelled_differently_is_still_refused(self, tracking_widget, tmp_path):
        # Review focus: `..` segments and (on Windows) letter case must not
        # sneak an overwrite past the guard.
        widget, viewer = tracking_widget
        (tmp_path / "sub").mkdir()
        csv = _write_spots_csv(tmp_path / "spots.csv")
        widget._set_input_csv(csv)
        widget._output_path_edit.setText(str(tmp_path / "sub" / ".." / "spots.csv"))
        before = csv.read_text()

        widget._run_tracking()

        assert csv.read_text() == before
        assert _tracks_layers(viewer) == []

    def test_a_missing_input_csv_shows_an_error_and_does_not_run(self, tracking_widget, tmp_path):
        # Review focus: a spots layer whose CSV has since been moved.
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv")
        _add_spots_layer(viewer, csv)
        csv.unlink()

        widget._run_tracking()

        assert _tracks_layers(viewer) == []
        assert not widget._is_running

    def test_a_worker_error_adds_no_layer_and_returns_to_idle(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        csv = _write_spots_csv(tmp_path / "spots.csv", n_frames=1)
        widget._set_input_csv(csv)

        widget._run_tracking()

        assert _tracks_layers(viewer) == []
        assert not widget._is_running
        assert "two timepoints" in widget._status_label.text()

    def test_a_cancelled_run_labels_its_layer(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        widget._set_input_csv(_write_spots_csv(tmp_path / "spots.csv", n_frames=10))
        widget._progress_bar.valueChanged.connect(
            lambda value: widget._cancel_btn.click() if value == 3 else None
        )

        widget._run_tracking()

        (layer,) = _tracks_layers(viewer)
        assert "[cancelled 3/10]" in layer.name

    def test_a_run_cancelled_before_any_frame_adds_no_layer(self, tracking_widget, tmp_path):
        widget, viewer = tracking_widget
        widget._set_input_csv(_write_spots_csv(tmp_path / "spots.csv"))

        def _launch(worker):
            worker.cancel()
            worker.run()

        widget._launch_worker = _launch
        widget._run_tracking()

        assert _tracks_layers(viewer) == []
        assert not (tmp_path / "spots_tracks.csv").exists()
