from __future__ import annotations

import sys

import numpy as np
import pytest


@pytest.fixture(params=["torch", "scipy"])
def backend(request, monkeypatch):
    """Run background-removal tests against both the torch path and the
    scipy fallback (forced by making `import torch` raise ImportError)."""
    if request.param == "scipy":
        monkeypatch.setitem(sys.modules, "torch", None)
    return request.param


def _blob_volume(dtype=np.uint16):
    vol = np.full((16, 32, 32), 100, dtype=np.float64)
    vol[6:10, 12:20, 12:20] = 3000.0
    return vol.astype(dtype)


class TestRemoveBackgroundHighPass3D:
    def test_preserves_input_dtype(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.uint16)
        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.dtype == np.uint16
        assert out.shape == vol.shape

    def test_restores_the_original_mean(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.uint16)
        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.mean() == pytest.approx(vol.mean(), rel=0.2)

    def test_integer_result_does_not_wrap_around(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.uint16)
        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.min() >= 0
        assert out.max() <= int(vol.max()) + int(round(vol.mean())) + 1

    def test_accepts_anisotropic_sigma_tuple(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.uint16)
        out = remove_background_high_pass_3d(vol, sigma=(2.0, 4.0, 4.0))
        assert out.shape == vol.shape

    def test_float_input_is_not_clipped(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.float32)
        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.dtype == np.float32
        assert out.min() < 0


class TestGaussianBlur3D:
    def test_blurs_and_preserves_dtype(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        vol = _blob_volume(np.uint16)
        out = gaussian_blur_3d(vol, sigma=2.0)
        assert out.dtype == np.uint16
        assert out.shape == vol.shape
        # Blurring spreads the peak: the max drops, the total mass is kept.
        assert out.max() < vol.max()
        assert out.sum() == pytest.approx(vol.sum(), rel=0.05)

    def test_non_positive_scalar_sigma_returns_input_unchanged(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        vol = _blob_volume(np.uint16)
        out = gaussian_blur_3d(vol, sigma=0.0)
        assert out is vol

    def test_all_non_positive_tuple_sigma_returns_input_unchanged(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        vol = _blob_volume(np.uint16)
        out = gaussian_blur_3d(vol, sigma=(0.0, 0.0, 0.0))
        assert out is vol

    def test_anisotropic_sigma_blurs_axes_independently(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        vol = _blob_volume(np.uint16)
        blurred_z_only = gaussian_blur_3d(vol, sigma=(4.0, 0.001, 0.001))
        blurred_isotropic = gaussian_blur_3d(vol, sigma=(4.0, 4.0, 4.0))
        # Blurring only along Z spreads less total variance than blurring
        # all three axes, so the isotropic peak is lower.
        assert blurred_isotropic.max() <= blurred_z_only.max()

    def test_wrong_ndim_raises(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        with pytest.raises(ValueError, match="3D"):
            gaussian_blur_3d(np.zeros((8, 8), dtype=np.uint16), sigma=2.0)


class TestApplyCalculation:
    def test_add_clips_at_dtype_max(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        vol = np.full((4, 8, 8), 65530, dtype=np.uint16)
        out = apply_calculation(vol, "add", 100)
        assert out.dtype == np.uint16
        assert out.max() == 65535

    def test_subtract_clips_at_zero(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        vol = np.full((4, 8, 8), 5, dtype=np.uint16)
        out = apply_calculation(vol, "subtract", 100)
        assert out.min() == 0

    def test_unknown_op_raises(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        with pytest.raises(ValueError):
            apply_calculation(np.zeros((2, 4, 4), dtype=np.uint16), "multiply", 2)
