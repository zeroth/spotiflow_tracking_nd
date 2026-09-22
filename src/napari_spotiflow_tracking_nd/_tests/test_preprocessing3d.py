from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest


def _require_torch():
    """Skip unless torch really imports here.

    `pytest.importorskip` only catches ImportError; a torch whose native
    libraries fail to load raises OSError (e.g. WinError 1114), which would
    error the test rather than skip it.
    """
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"torch is not importable here: {exc}")


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

    def test_thin_volume_does_not_crash_on_padding(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        # Regression test for torch padding failure on thin volumes.
        # A (4, 32, 32) volume with sigma=3.0 would produce radius=6 in the
        # torch path, which is > dim_size for Z. The fix clamps radius to
        # dim_size-1 to ensure F.pad() doesn't crash.
        vol = np.full((4, 32, 32), 100, dtype=np.uint16)
        vol[1:3, 12:20, 12:20] = 3000

        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.shape == vol.shape
        assert out.dtype == vol.dtype


class TestBackendsAgree:
    """The torch path and the scipy fallback must apply the *same* filter.

    The old torch kernel spanned a fixed [-2, 2] regardless of `sigma`, so
    its effective sigma was ~0.70x the requested one while scipy's was
    exactly `sigma` — the same `sigma` meant two different filters, chosen
    only by which backend happened to import.
    """

    @pytest.mark.parametrize("sigma", [2.0, 3.0, 5.0])
    def test_torch_kernel_has_the_requested_standard_deviation(self, sigma):
        _require_torch()
        from scipy.ndimage import gaussian_filter

        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        # The low-pass background the filter removes *is* the Gaussian the
        # backend applied, so recovering it from an impulse and measuring
        # its second moment measures the backend's effective sigma. The
        # volume is wide enough (64³) that no radius gets clamped.
        impulse = np.zeros((64, 64, 64), dtype=np.float32)
        impulse[32, 32, 32] = 1000.0

        residual = remove_background_high_pass_3d(impulse, sigma=sigma).astype(
            np.float64
        ) - float(np.mean(impulse))
        background_torch = impulse.astype(np.float64) - residual
        background_scipy = gaussian_filter(impulse.astype(np.float64), sigma=sigma)

        def _effective_sigma(background, axis):
            profile = background.sum(
                axis=tuple(a for a in range(3) if a != axis)
            )
            profile = np.clip(profile, 0.0, None)
            coords = np.arange(len(profile)) - 32
            return float(
                np.sqrt((profile * coords ** 2).sum() / profile.sum())
            )

        for axis in range(3):
            measured = _effective_sigma(background_torch, axis)
            # The old fixed-[-2, 2] kernel measured ~0.70 * sigma here.
            assert measured == pytest.approx(sigma, rel=0.05), f"axis {axis}"
            assert measured == pytest.approx(
                _effective_sigma(background_scipy, axis), rel=0.05
            ), f"axis {axis}"

    @pytest.mark.parametrize("sigma", [2.0, 3.0])
    def test_torch_and_scipy_outputs_agree_numerically(self, sigma, monkeypatch):
        _require_torch()
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.float32)
        out_torch = remove_background_high_pass_3d(vol, sigma=sigma).astype(np.float64)

        monkeypatch.setitem(sys.modules, "torch", None)  # force the scipy path
        out_scipy = remove_background_high_pass_3d(vol, sigma=sigma).astype(np.float64)

        span = float(np.abs(out_scipy - out_scipy.mean()).max())
        max_diff = float(np.abs(out_torch - out_scipy).max())
        # Not bit-identical (different boundary handling), but the same
        # filter: well under 5% of the residual's own dynamic range. The
        # old fixed-[-2,2] kernel was ~29% off here.
        assert max_diff < 0.05 * span, f"max_diff={max_diff}, span={span}"

    def test_clamped_radius_warns_that_the_effective_sigma_shrank(self):
        _require_torch()
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        # Z is only 4 planes, so a sigma-10 kernel's radius cannot fit and
        # gets clamped — the effective sigma along Z silently shrinks, so
        # the user has to be told.
        vol = np.full((4, 64, 64), 100, dtype=np.uint16)
        with pytest.warns(RuntimeWarning, match="effective sigma"):
            out = remove_background_high_pass_3d(vol, sigma=10.0)
        assert out.shape == vol.shape

    def test_no_warning_when_the_radius_fits(self):
        _require_torch()
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = np.full((64, 64, 64), 100, dtype=np.uint16)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            remove_background_high_pass_3d(vol, sigma=2.0)


class TestTorchErrorsAreNotSwallowed:
    def test_an_oserror_inside_the_torch_computation_propagates(self, monkeypatch):
        # The try/except must guard only the *imports*. A real CUDA driver
        # fault raised mid-computation used to be silently rerouted to the
        # scipy fallback, masking a hardware problem as a missing torch.
        _require_torch()
        from napari_spotiflow_tracking_nd import _preprocessing3d

        def _boom(*args, **kwargs):
            raise OSError("CUDA driver is broken")

        monkeypatch.setattr(_preprocessing3d, "_remove_background_torch_3d", _boom)

        with pytest.raises(OSError, match="CUDA driver is broken"):
            _preprocessing3d.remove_background_high_pass_3d(
                _blob_volume(np.uint16), sigma=2.0
            )


class TestGetDevice:
    def test_returns_cpu_when_gpu_is_not_requested(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import _get_device

        assert _get_device(use_gpu=False) == "cpu"

    def test_falls_back_to_cpu_when_cuda_is_unavailable(self, monkeypatch):
        import types

        from napari_spotiflow_tracking_nd._preprocessing3d import _get_device

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _get_device(use_gpu=True) == "cpu"

    def test_falls_back_to_cpu_when_torch_will_not_import(self, monkeypatch):
        from napari_spotiflow_tracking_nd._preprocessing3d import _get_device

        monkeypatch.setitem(sys.modules, "torch", None)  # -> ImportError

        assert _get_device(use_gpu=True) == "cpu"


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
