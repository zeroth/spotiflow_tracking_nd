from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


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


def _blob_volume(centres, shape=(24, 64, 64), sigma_zyx=(2.0, 1.5, 1.5),
                 amplitude=100.0, noise=0.0, seed=0):
    """A volume of anisotropic Gaussian spots on a flat background."""
    zz, yy, xx = np.indices(shape, dtype=np.float64)
    vol = np.full(shape, 50.0)
    sz, sy, sx = sigma_zyx
    for z, y, x in centres:
        vol += amplitude * np.exp(
            -((zz - z) ** 2 / (2 * sz ** 2) + (yy - y) ** 2 / (2 * sy ** 2)
              + (xx - x) ** 2 / (2 * sx ** 2))
        )
    if noise:
        vol += np.random.default_rng(seed).normal(0, noise, shape)
    return np.clip(vol, 0, None).astype(np.uint16)


CENTRES = [(6, 12, 15), (12, 40, 20), (17, 30, 50), (8, 52, 48)]


class TestFilterResponse:
    def test_response_is_about_the_spot_amplitude_for_a_matched_sigma(self):
        # Calibrated so a threshold or a `response` value reads as roughly
        # "peak brightness above background", for LoG and DoG alike.
        from napari_spotiflow_tracking_nd._segmentation3d import filter_response_3d

        vol = _blob_volume([(12, 32, 32)], amplitude=200.0)
        for method in ("log", "dog"):
            resp = filter_response_3d(vol, method, sigma_zyx=(2.0, 1.5, 1.5))
            assert resp.dtype == np.float32
            assert resp.shape == vol.shape
            assert np.unravel_index(np.argmax(resp), resp.shape) == (12, 32, 32)
            assert resp[12, 32, 32] == pytest.approx(200.0, rel=0.15), method

    def test_flat_background_gives_no_response(self):
        from napari_spotiflow_tracking_nd._segmentation3d import filter_response_3d

        vol = np.full((10, 20, 20), 500, dtype=np.uint16)
        for method in ("log", "dog"):
            assert np.abs(filter_response_3d(vol, method, (1.5, 1.5, 1.5))).max() < 1e-2

    def test_unknown_method_is_refused(self):
        from napari_spotiflow_tracking_nd._segmentation3d import filter_response_3d

        with pytest.raises(ValueError, match="method"):
            filter_response_3d(np.zeros((4, 8, 8)), "wavelet", (1.0, 1.0, 1.0))


class TestClassicalDetector3D:
    @pytest.mark.parametrize("method", ["log", "dog"])
    def test_finds_every_spot_and_nothing_else(self, method):
        from napari_spotiflow_tracking_nd._segmentation3d import ClassicalDetector3D

        vol = _blob_volume(CENTRES, amplitude=100.0, noise=5.0)
        det = ClassicalDetector3D(method=method, sigma_zyx=(2.0, 1.5, 1.5), threshold=5.0)

        points, details = det.predict(vol, min_distance=2)

        assert points.shape == (len(CENTRES), 3)
        found = sorted(map(tuple, np.round(points).astype(int)))
        for got, want in zip(found, sorted(CENTRES)):
            assert np.abs(np.subtract(got, want)).max() <= 1
        # Same (points, details) shape as a Spotiflow model, so the worker
        # treats both alike; no probability exists, so it is NaN (blank).
        assert np.isnan(details.prob).all() and len(details.prob) == len(points)
        assert len(details.response) == len(points)
        assert (details.snr >= 5.0).all()
        assert details.response == pytest.approx(100.0, rel=0.3)

    def test_a_higher_threshold_drops_the_faint_spot(self):
        from napari_spotiflow_tracking_nd._segmentation3d import ClassicalDetector3D

        vol = _blob_volume(CENTRES[:1], amplitude=100.0, noise=5.0) \
            + _blob_volume(CENTRES[1:2], amplitude=25.0) - 50
        det = ClassicalDetector3D(method="log", sigma_zyx=(2.0, 1.5, 1.5), threshold=5.0)
        assert len(det.predict(vol)[0]) == 2
        det.threshold = 40.0
        points, _ = det.predict(vol)
        assert len(points) == 1
        assert tuple(np.round(points[0]).astype(int)) == CENTRES[0]

    def test_min_distance_merges_close_peaks(self):
        from napari_spotiflow_tracking_nd._segmentation3d import ClassicalDetector3D

        vol = _blob_volume([(12, 30, 28), (12, 30, 34)], amplitude=100.0, noise=2.0)
        det = ClassicalDetector3D(method="log", sigma_zyx=(2.0, 1.5, 1.5), threshold=5.0)
        assert len(det.predict(vol, min_distance=2)[0]) == 2
        assert len(det.predict(vol, min_distance=7)[0]) == 1

    def test_spots_touching_the_volume_edge_are_kept(self):
        # Volumes are often only a few dozen planes deep; dropping peaks
        # within `min_distance` of the border would lose real spots.
        from napari_spotiflow_tracking_nd._segmentation3d import ClassicalDetector3D

        vol = _blob_volume([(1, 32, 32)], amplitude=100.0, noise=2.0)
        det = ClassicalDetector3D(method="dog", sigma_zyx=(2.0, 1.5, 1.5), threshold=5.0)
        assert len(det.predict(vol, min_distance=3)[0]) == 1

    def test_noise_free_volume_without_spots_finds_nothing(self):
        from napari_spotiflow_tracking_nd._segmentation3d import ClassicalDetector3D

        det = ClassicalDetector3D(method="log", sigma_zyx=(1.5, 1.5, 1.5), threshold=5.0)
        points, details = det.predict(np.full((8, 16, 16), 100, dtype=np.uint16))
        assert points.shape == (0, 3)
        assert len(details.prob) == 0

    def test_params_are_recorded_for_the_sidecar(self):
        from napari_spotiflow_tracking_nd._segmentation3d import ClassicalDetector3D

        det = ClassicalDetector3D(method="dog", sigma_zyx=(2.0, 1.5, 1.5), threshold=7.0)
        assert det.params() == {
            "method": "dog", "sigma_zyx": [2.0, 1.5, 1.5], "threshold_snr": 7.0,
            "dog_sigma_ratio": 1.6,
        }


class TestLocalMaxima:
    @pytest.mark.parametrize("min_distance", [1, 2, 4])
    def test_matches_skimage_peak_local_max(self, min_distance):
        from skimage.feature import peak_local_max

        from napari_spotiflow_tracking_nd._segmentation3d import (
            _local_maxima,
            filter_response_3d,
        )

        rng = np.random.default_rng(1)
        resp = filter_response_3d(rng.normal(100, 10, (16, 48, 48)), "log", (1.0, 1.0, 1.0))
        ours = _local_maxima(resp, min_distance, threshold=1.0)
        ref = peak_local_max(resp, min_distance=min_distance, threshold_abs=1.0,
                             exclude_border=False)
        assert set(map(tuple, ours)) == set(map(tuple, ref))

    def test_a_flat_plateau_yields_one_peak(self):
        from napari_spotiflow_tracking_nd._segmentation3d import _local_maxima

        resp = np.zeros((5, 9, 9), dtype=np.float32)
        resp[2, 3:5, 3:5] = 7.0
        assert len(_local_maxima(resp, 2, threshold=1.0)) == 1
