from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


@pytest.fixture
def detection_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._detection_widget3d import DetectionWidget3D

    viewer = make_napari_viewer()
    widget = DetectionWidget3D(viewer)
    return widget, viewer


class TestDetectionWidget3DCreation:
    def test_creates_without_error(self, detection_widget):
        widget, _ = detection_widget
        assert widget is not None

    def test_use_gpu_is_on_by_default(self, detection_widget):
        # `_get_device` falls back to "cpu" when CUDA isn't available, so
        # defaulting this on is safe on a CPU-only machine and saves
        # ticking it every session on a GPU one.
        widget, _ = detection_widget
        assert widget._use_gpu_check.isChecked()

    def test_has_the_expected_controls(self, detection_widget):
        widget, _ = detection_widget
        for attr in (
            "_image_combo", "_model_combo", "_browse_model_btn",
            "_prob_thresh_spin", "_min_distance_spin",
            "_csv_path_edit", "_browse_csv_btn", "_run_btn", "_preview_btn",
            "_progress_bar",
        ):
            assert hasattr(widget, attr), f"missing {attr}"


class TestModelDropdown:
    def test_lists_every_built_in_3d_model(self, detection_widget):
        from napari_spotiflow_tracking_nd._segmentation3d import PRETRAINED_3D_MODELS

        widget, _ = detection_widget
        listed = [
            widget._model_combo.itemText(i) for i in range(widget._model_combo.count())
        ]

        assert listed == list(PRETRAINED_3D_MODELS)

    def test_defaults_to_the_first_built_in_model(self, detection_widget):
        from napari_spotiflow_tracking_nd._segmentation3d import PRETRAINED_3D_MODELS

        widget, _ = detection_widget

        assert widget._model_combo.currentText() == PRETRAINED_3D_MODELS[0]

    def test_a_custom_checkpoint_path_is_still_accepted(self, detection_widget, tmp_path):
        # The combo is editable so a local checkpoint folder — the thing
        # `load_model` falls through to `Spotiflow.from_folder` for — can
        # still be typed, pasted, or filled in by Browse….
        widget, _ = detection_widget

        widget._model_combo.setCurrentText(str(tmp_path / "my_checkpoint"))

        assert widget._model_combo.currentText() == str(tmp_path / "my_checkpoint")

    def test_browse_puts_the_chosen_folder_into_the_combo(self, detection_widget, tmp_path):
        widget, _ = detection_widget
        chosen = str(tmp_path / "checkpoint_dir")

        with patch(
            "napari_spotiflow_tracking_nd._detection_widget3d.QFileDialog.getExistingDirectory",
            return_value=chosen,
        ):
            widget._browse_custom_model()

        assert widget._model_combo.currentText() == chosen

    def test_combo_is_populated_for_layers_that_existed_before_construction(
        self, make_napari_viewer
    ):
        # The normal workflow loads the folder in the Pre-Processing widget
        # *before* this one is opened, so `layers.events.inserted` has
        # already fired and the combo would otherwise start empty.
        from napari_spotiflow_tracking_nd._detection_widget3d import DetectionWidget3D

        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((2, 4, 8, 8), dtype=np.uint16), name="already_here")

        widget = DetectionWidget3D(viewer)

        assert widget._image_combo.currentText() == "already_here"


class TestRunDetection:
    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_writes_csv_and_adds_a_scaled_points_layer(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        mock_model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        mock_model.predict.return_value = (np.array([[1, 2, 3]]), details)
        mock_load_model.return_value = mock_model

        viewer.add_image(
            np.zeros((2, 4, 16, 16), dtype=np.uint16),
            name="vol",
            scale=(1.0, 0.210, 0.104, 0.104),
            metadata={"source_files": files},
        )
        widget._image_combo.clear()
        widget._image_combo.addItem("vol")
        widget._image_combo.setCurrentText("vol")
        widget._model_combo.setCurrentText("synth_3d")
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))

        widget._run_detection()

        df = pd.read_csv(tmp_path / "spots.csv")
        assert list(df.columns) == ["frame", "z", "y", "x", "prob"]
        assert len(df) == 2  # one row per timepoint (2 timepoints, 1 spot each)

        points_layers = [l for l in viewer.layers if l.__class__.__name__ == "Points"]
        assert len(points_layers) == 1
        # Points are (frame, z, y, x) — 4 columns — so scale carries all 4
        # entries from the source image layer, including the unscaled T axis.
        assert tuple(points_layers[0].scale) == (1.0, 0.210, 0.104, 0.104)

    def test_missing_input_layer_shows_error_not_exception(self, detection_widget, tmp_path):
        widget, viewer = detection_widget
        widget._image_combo.clear()
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))

        widget._run_detection()  # must not raise

        assert not (tmp_path / "spots.csv").exists()

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_zero_spots_in_every_frame_does_not_raise(
        self, mock_load_model, detection_widget, tmp_path
    ):
        # DetectionWorker3D only writes the CSV once at least one frame has
        # a detection, so an all-empty run means `finished` fires with a
        # path to a CSV that was never created. `_on_finished` must handle
        # that gracefully instead of letting `pd.read_csv` raise
        # FileNotFoundError out of the synchronous `worker.run()` call.
        widget, viewer = detection_widget
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        mock_model = MagicMock()
        details_empty = MagicMock()
        details_empty.prob = np.empty((0,))
        mock_model.predict.side_effect = [
            (np.empty((0, 3)), details_empty),
            (np.empty((0, 3)), details_empty),
        ]
        mock_load_model.return_value = mock_model

        viewer.add_image(
            np.zeros((2, 4, 16, 16), dtype=np.uint16),
            name="vol",
            scale=(1.0, 0.210, 0.104, 0.104),
            metadata={"source_files": files},
        )
        widget._image_combo.clear()
        widget._image_combo.addItem("vol")
        widget._image_combo.setCurrentText("vol")
        widget._model_combo.setCurrentText("synth_3d")
        csv_path = tmp_path / "spots.csv"
        widget._csv_path_edit.setText(str(csv_path))

        widget._run_detection()  # must not raise

        assert not csv_path.exists()
        points_layers = [l for l in viewer.layers if l.__class__.__name__ == "Points"]
        assert len(points_layers) == 0

    def test_a_failing_load_model_shows_an_error_instead_of_raising(
        self, detection_widget, tmp_path
    ):
        # A bad model name, a missing checkpoint folder or a failed weight
        # download must not raise straight out of a Qt slot.
        widget, viewer = detection_widget
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=1, shape=(4, 16, 16))

        viewer.add_image(
            np.zeros((1, 4, 16, 16), dtype=np.uint16),
            name="vol",
            metadata={"source_files": files},
        )
        widget._image_combo.setCurrentText("vol")
        widget._model_combo.setCurrentText("no_such_model")
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))

        with patch(
            "napari_spotiflow_tracking_nd._detection_widget3d.load_model",
            side_effect=RuntimeError("no weights for you"),
        ), patch(
            "napari_spotiflow_tracking_nd._detection_widget3d.show_error"
        ) as mock_show_error:
            widget._run_detection()  # must not raise

        assert mock_show_error.called
        assert "no weights for you" in " ".join(
            str(c) for c in mock_show_error.call_args_list
        )
        assert not (tmp_path / "spots.csv").exists()


class TestCalibrationIsThreadedThrough:
    """`frame_interval_s` and the two spatial calibrations have to reach the
    run's outputs — later phases turn pixel/frame track properties into
    physical units, and the CSV alone carries no calibration at all."""

    @staticmethod
    def _run(widget, viewer, tmp_path, calibration):
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))
        viewer.add_image(
            np.zeros((2, 4, 16, 16), dtype=np.uint16),
            name="vol",
            scale=(1.0, 0.210, 0.104, 0.104),
            metadata={"source_files": files, **calibration},
        )
        widget._image_combo.setCurrentText("vol")
        widget._model_combo.setCurrentText("synth_3d")
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))
        widget._run_detection()

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_points_layer_metadata_carries_the_calibration(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        mock_model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        mock_model.predict.return_value = (np.array([[1, 2, 3]]), details)
        mock_load_model.return_value = mock_model

        self._run(
            widget, viewer, tmp_path,
            {"pixel_size_xy_um": 0.104, "voxel_size_z_um": 0.210,
             "frame_interval_s": 2.5},
        )

        points = [l for l in viewer.layers if l.__class__.__name__ == "Points"][0]
        assert points.metadata["csv_path"] == str(tmp_path / "spots.csv")
        assert points.metadata["pixel_size_xy_um"] == 0.104
        assert points.metadata["voxel_size_z_um"] == 0.210
        assert points.metadata["frame_interval_s"] == 2.5

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_a_json_sidecar_records_the_calibration_next_to_the_csv(
        self, mock_load_model, detection_widget, tmp_path
    ):
        from napari_spotiflow_tracking_nd._detection_widget3d import (
            calibration_sidecar_path,
        )

        widget, viewer = detection_widget
        mock_model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        mock_model.predict.return_value = (np.array([[1, 2, 3]]), details)
        mock_load_model.return_value = mock_model

        self._run(
            widget, viewer, tmp_path,
            {"pixel_size_xy_um": 0.104, "voxel_size_z_um": 0.210,
             "frame_interval_s": 2.5},
        )

        sidecar = calibration_sidecar_path(tmp_path / "spots.csv")
        assert sidecar == tmp_path / "spots.json"
        assert json.loads(sidecar.read_text()) == {
            "pixel_size_xy_um": 0.104,
            "voxel_size_z_um": 0.210,
            "frame_interval_s": 2.5,
        }

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_uncalibrated_run_records_nulls_rather_than_omitting_keys(
        self, mock_load_model, detection_widget, tmp_path
    ):
        from napari_spotiflow_tracking_nd._detection_widget3d import (
            calibration_sidecar_path,
        )

        widget, viewer = detection_widget
        mock_model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        mock_model.predict.return_value = (np.array([[1, 2, 3]]), details)
        mock_load_model.return_value = mock_model

        self._run(widget, viewer, tmp_path, {})

        assert json.loads(calibration_sidecar_path(tmp_path / "spots.csv").read_text()) == {
            "pixel_size_xy_um": None,
            "voxel_size_z_um": None,
            "frame_interval_s": None,
        }


class TestPreview:
    """Quick mode: detect on the currently-displayed timepoint only, so
    prob_thresh / min_distance can be tuned before a whole-series run."""

    @staticmethod
    def _setup(widget, viewer, tmp_path, n_timepoints=3):
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=n_timepoints,
                                       shape=(4, 16, 16))
        viewer.add_image(
            np.zeros((n_timepoints, 4, 16, 16), dtype=np.uint16),
            name="vol",
            scale=(1.0, 0.210, 0.104, 0.104),
            metadata={"source_files": files},
        )
        widget._image_combo.setCurrentText("vol")
        return files

    @staticmethod
    def _mock_model(n_spots=2):
        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.9, 0.7][:n_spots])
        model.predict.return_value = (
            np.array([[1, 2, 3], [2, 4, 6]][:n_spots]), details,
        )
        return model

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_previews_the_current_timepoint_without_writing_a_csv(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        files = self._setup(widget, viewer, tmp_path)
        seen = {}

        def _predict(volume, **kwargs):
            seen["volume"] = volume
            details = MagicMock()
            details.prob = np.array([0.9])
            return np.array([[1, 2, 3]]), details

        model = MagicMock()
        model.predict.side_effect = _predict
        mock_load_model.return_value = model

        viewer.dims.set_current_step(0, 2)  # look at timepoint 2
        # No CSV path set at all — preview must still work.
        assert widget._csv_path_edit.text() == ""
        widget._run_preview()

        # It read timepoint 2 off disk, not the whole series.
        assert np.array_equal(seen["volume"], tifffile.imread(str(files[2])))
        assert model.predict.call_count == 1
        assert not list(tmp_path.glob("*.csv"))

        points = viewer.layers["vol (spot preview)"]
        assert len(points.data) == 1
        # A single volume's points are (z, y, x) — the source layer's
        # scale without its leading T entry.
        assert tuple(points.scale) == (0.210, 0.104, 0.104)

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_repeated_previews_reuse_the_loaded_model(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        self._setup(widget, viewer, tmp_path)
        mock_load_model.return_value = self._mock_model()

        widget._run_preview()
        widget._prob_thresh_spin.setValue(0.9)  # tune, preview again
        widget._run_preview()

        # Re-loading the weights on every click is what makes preview slow.
        assert mock_load_model.call_count == 1

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_changing_the_model_reloads_it(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        self._setup(widget, viewer, tmp_path)
        mock_load_model.return_value = self._mock_model()

        widget._model_combo.setCurrentText("synth_3d")
        widget._run_preview()
        widget._model_combo.setCurrentText("smfish_3d")
        widget._run_preview()

        assert mock_load_model.call_count == 2

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_repeated_previews_replace_one_layer_rather_than_stacking(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        self._setup(widget, viewer, tmp_path)
        mock_load_model.return_value = self._mock_model()

        widget._run_preview()
        widget._run_preview()

        points_layers = [l for l in viewer.layers if l.__class__.__name__ == "Points"]
        assert len(points_layers) == 1

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_a_stale_timepoint_beyond_the_series_is_clamped(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        viewer.add_image(np.zeros((10, 4, 16, 16), dtype=np.uint16), name="longer")
        self._setup(widget, viewer, tmp_path, n_timepoints=2)
        mock_load_model.return_value = self._mock_model()
        viewer.dims.set_current_step(0, 9)

        widget._run_preview()  # must not IndexError

        assert "vol (spot preview)" in viewer.layers

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_a_failing_model_load_shows_an_error_instead_of_raising(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        self._setup(widget, viewer, tmp_path)
        mock_load_model.side_effect = RuntimeError("no such checkpoint")

        with patch(
            "napari_spotiflow_tracking_nd._detection_widget3d.show_error"
        ) as mock_show_error:
            widget._run_preview()  # must not raise

        assert mock_show_error.called

    def test_preview_without_source_files_shows_an_error(self, detection_widget):
        widget, viewer = detection_widget
        viewer.add_image(np.zeros((2, 4, 8, 8), dtype=np.uint16), name="plain")
        widget._image_combo.setCurrentText("plain")

        with patch(
            "napari_spotiflow_tracking_nd._detection_widget3d.show_error"
        ) as mock_show_error:
            widget._run_preview()  # must not raise

        assert mock_show_error.called
