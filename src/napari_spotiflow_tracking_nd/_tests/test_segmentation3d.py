from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def _make_mock_model():
    model = MagicMock()
    details = MagicMock()
    details.prob = np.array([0.9, 0.7])
    model.predict.return_value = (
        np.array([[3, 10, 20], [5, 30, 40]]),  # 2 spots at (z, y, x)
        details,
    )
    return model


class TestDetectSpots3D:
    def test_returns_points_and_details(self):
        from napari_spotiflow_tracking_nd._segmentation3d import detect_spots_3d

        model = _make_mock_model()
        volume = np.random.rand(16, 64, 64).astype(np.float32)

        points, details = detect_spots_3d(volume, model, prob_thresh=0.5, min_distance=2)

        assert points.shape == (2, 3)
        assert list(points[0]) == [3, 10, 20]
        assert list(details.prob) == [0.9, 0.7]

    def test_calls_model_predict_with_params(self):
        from napari_spotiflow_tracking_nd._segmentation3d import detect_spots_3d

        model = _make_mock_model()
        volume = np.random.rand(16, 64, 64).astype(np.float32)

        detect_spots_3d(volume, model, prob_thresh=0.3, min_distance=5)

        model.predict.assert_called_once_with(
            volume, prob_thresh=0.3, min_distance=5, verbose=False,
        )

    def test_returns_empty_array_when_no_spots(self):
        from napari_spotiflow_tracking_nd._segmentation3d import detect_spots_3d

        model = MagicMock()
        details = MagicMock()
        details.prob = np.empty((0,))
        model.predict.return_value = (np.empty((0, 3)), details)
        volume = np.random.rand(16, 64, 64).astype(np.float32)

        points, details = detect_spots_3d(volume, model)

        assert points.shape == (0, 3)


class TestLoadModel:
    @patch("spotiflow.model.Spotiflow")
    def test_loads_a_registered_3d_pretrained_model(self, MockSpotiflow):
        from napari_spotiflow_tracking_nd._segmentation3d import load_model

        mock_model = MagicMock()
        MockSpotiflow.from_pretrained.return_value = mock_model

        result = load_model("synth_3d", device="cpu")

        MockSpotiflow.from_pretrained.assert_called_once_with("synth_3d", map_location="cpu")
        assert result is mock_model

    @patch("spotiflow.model.Spotiflow")
    def test_loads_a_custom_model_from_path(self, MockSpotiflow):
        from napari_spotiflow_tracking_nd._segmentation3d import load_model

        mock_model = MagicMock()
        MockSpotiflow.from_folder.return_value = mock_model

        result = load_model("/path/to/custom_3d_model", device="cuda")

        MockSpotiflow.from_folder.assert_called_once_with(
            "/path/to/custom_3d_model", map_location="cuda"
        )
        assert result is mock_model
