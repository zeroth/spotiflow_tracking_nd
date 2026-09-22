from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

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

    def test_has_the_expected_controls(self, detection_widget):
        widget, _ = detection_widget
        for attr in (
            "_image_combo", "_model_name_edit", "_browse_model_btn",
            "_prob_thresh_spin", "_min_distance_spin",
            "_csv_path_edit", "_browse_csv_btn", "_run_btn", "_progress_bar",
        ):
            assert hasattr(widget, attr), f"missing {attr}"


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
        widget._model_name_edit.setText("synth_3d")
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
