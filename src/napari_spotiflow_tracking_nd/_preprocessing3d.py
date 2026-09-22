from __future__ import annotations

import warnings

import numpy as np

# Kernels are truncated at `_TRUNCATE * sigma`, matching the default of
# `scipy.ndimage.gaussian_filter` (truncate=4.0) so the torch path and the
# scipy fallback apply the *same* filter for the same `sigma`.
_TRUNCATE = 4.0


def _get_device(use_gpu: bool = False) -> str:
    """Return 'cuda' if requested and available, else 'cpu'."""
    if use_gpu:
        try:
            import torch
        except (ImportError, OSError):
            # No torch, or a torch whose native libraries won't load.
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def _rebaseline(residual: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Shift a zero-centred residual onto the original's mean, in its dtype.

    See the 2D plugin's `_preprocessing._rebaseline` for the rationale —
    ported unchanged: integer dtypes are clipped to their representable
    range before the cast; floats are cast without clipping.
    """
    out = residual + float(np.mean(original))
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        out = np.clip(out, info.min, info.max)
    return out.astype(original.dtype)


def _sigma_zyx(sigma: float | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(sigma, (int, float)):
        return (float(sigma), float(sigma), float(sigma))
    sz, sy, sx = sigma
    return (float(sz), float(sy), float(sx))


def remove_background_high_pass_3d(
    volume: np.ndarray,
    sigma: float | tuple[float, float, float] = 10.0,
    device: str = "cpu",
) -> np.ndarray:
    """Remove background from a single (Z, Y, X) volume by subtracting a
    Gaussian low-pass filtered copy.

    Uses a separable PyTorch conv3d for GPU acceleration when torch is
    importable; falls back to `scipy.ndimage.gaussian_filter` on CPU
    otherwise. The subtraction is re-baselined onto the volume's own mean
    (see `_rebaseline`) exactly as the 2D plugin does per-frame.

    Args:
        volume: 3D array (Z, Y, X).
        sigma: one float (isotropic) or (sigma_z, sigma_y, sigma_x) — z and
            xy sampling often differ, so anisotropic sigma is supported.
        device: 'cuda' for GPU, 'cpu' for CPU.

    Returns:
        Background-subtracted volume, re-baselined and cast back to the
        input dtype.
    """
    sigma_z, sigma_y, sigma_x = _sigma_zyx(sigma)

    # Only the *imports* are guarded: a torch that isn't installed, or one
    # whose native libraries won't load (Windows `OSError`), means "use
    # scipy". An error raised once we're inside `_remove_background_torch_3d`
    # — a genuine CUDA driver fault, say, or a bug here — must propagate,
    # not be silently rerouted to a different (CPU) code path.
    try:
        import torch
        import torch.nn.functional as F
    except (ImportError, OSError):
        from scipy.ndimage import gaussian_filter
        background = gaussian_filter(
            volume.astype(np.float64), sigma=(sigma_z, sigma_y, sigma_x)
        )
        return _rebaseline(volume.astype(np.float64) - background, volume)

    return _remove_background_torch_3d(
        volume, sigma_z, sigma_y, sigma_x, device, torch, F
    )


def _remove_background_torch_3d(volume, sigma_z, sigma_y, sigma_x, device, torch, F):
    """Separable 3D Gaussian background subtraction via torch conv3d."""
    dim_z, dim_y, dim_x = volume.shape

    def _kernel(sigma, dim_size, axis):
        """A normalized 1D Gaussian kernel of standard deviation `sigma`,
        sampled on the voxel grid and truncated at `_TRUNCATE * sigma` —
        i.e. the same discrete kernel `scipy.ndimage.gaussian_filter`
        builds, so both backends apply the same filter.

        The radius is clamped to `dim_size - 1` because `F.pad(...,
        mode="reflect")` cannot pad by more than the dimension it pads.
        Clamping shrinks the *effective* sigma along that axis, so it
        warns rather than degrading the filter silently.
        """
        if sigma <= 0:
            return torch.ones(1, dtype=torch.float32), 0
        requested = max(int(np.ceil(sigma * _TRUNCATE)), 1)
        radius = min(requested, max(dim_size - 1, 0))
        t = torch.arange(-radius, radius + 1, dtype=torch.float32)
        h = torch.exp(-(t ** 2) / (2.0 * sigma ** 2))
        h = h / h.sum()
        if radius < requested:
            effective = float(torch.sqrt((h * t ** 2).sum()))
            warnings.warn(
                f"sigma={sigma:g} needs a kernel radius of {requested} along "
                f"{axis}, but that axis is only {dim_size} voxels deep; the "
                f"radius was clamped to {radius}, so the effective sigma "
                f"along {axis} is about {effective:.2f}, not {sigma:g}.",
                RuntimeWarning,
                stacklevel=4,  # report against the caller of the public fn
            )
        return h, radius

    kz, rz = _kernel(sigma_z, dim_z, "Z")
    ky, ry = _kernel(sigma_y, dim_y, "Y")
    kx, rx = _kernel(sigma_x, dim_x, "X")

    wz = kz.reshape(1, 1, -1, 1, 1)
    wy = ky.reshape(1, 1, 1, -1, 1)
    wx = kx.reshape(1, 1, 1, 1, -1)

    x = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

    if device == "cuda":
        x = x.cuda()
        wz, wy, wx = wz.cuda(), wy.cuda(), wx.cuda()

    y = F.pad(x, pad=(0, 0, 0, 0, rz, rz), mode="reflect")
    y = F.conv3d(y, weight=wz, padding="valid")
    y = F.pad(y, pad=(0, 0, ry, ry, 0, 0), mode="reflect")
    y = F.conv3d(y, weight=wy, padding="valid")
    y = F.pad(y, pad=(rx, rx, 0, 0, 0, 0), mode="reflect")
    y = F.conv3d(y, weight=wx, padding="valid")

    residual = (x - y).squeeze().cpu().numpy().astype(np.float64)
    return _rebaseline(residual, volume)


def gaussian_blur_3d(
    volume: np.ndarray, sigma: float | tuple[float, float, float]
) -> np.ndarray:
    """3D Gaussian blur of a single (Z, Y, X) volume.

    `sigma` is one float (isotropic) or (sigma_z, sigma_y, sigma_x). A
    non-positive scalar, or an all non-positive tuple, returns the input
    unchanged — extending the 2D plugin's `sigma <= 0` convention to 3D's
    anisotropic sigma. Computed in float64, cast back to the input dtype.

    Raises:
        ValueError: if `volume` is not 3D.
    """
    from scipy.ndimage import gaussian_filter

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X) volume, got {volume.ndim}D")

    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            return volume
        sigma_arg = float(sigma)
    else:
        sigma_arg = tuple(float(s) for s in sigma)
        if all(s <= 0 for s in sigma_arg):
            return volume

    return gaussian_filter(volume.astype(np.float64), sigma=sigma_arg).astype(volume.dtype)


def apply_calculation(image: np.ndarray, op: str, value: int) -> np.ndarray:
    """Add or subtract an integer scalar from an image or volume.

    Ported unchanged from the 2D plugin — already shape-agnostic. Returns
    a new array with the same dtype as the input; integer dtypes clip to
    their representable range, floats do not.

    Raises:
        ValueError: if `op` is not one of "add" or "subtract".
    """
    if op not in ("add", "subtract"):
        raise ValueError(f"Unknown op: {op!r}")

    sign = 1 if op == "add" else -1

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        wide = image.astype(np.int64) + sign * int(value)
        clipped = np.clip(wide, info.min, info.max)
        return clipped.astype(image.dtype)

    return (image + sign * value).astype(image.dtype, copy=True)
