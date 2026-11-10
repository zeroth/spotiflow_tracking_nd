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


# Rolling ball is computed on a shrunken copy (ImageJ's trick): a 3D ball is
# not separable, so its cost grows with the cube of the radius — radius 10 on
# a 20×256×256 volume already takes ~12 s at full resolution. Each axis is
# shrunk by block-minimum so the ball rolled there is at most about this
# many voxels in radius, then the background is interpolated back up. The
# background is smooth by construction, so the loss is negligible.
_ROLLING_BALL_MAX_SHRUNK_RADIUS = 5.0


def _require_3d(volume: np.ndarray) -> None:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X) volume, got {volume.ndim}D")


def _positive_radius_zyx(radius) -> tuple[float, float, float]:
    radii = _sigma_zyx(radius)
    if any(r <= 0 for r in radii):
        raise ValueError(f"radius must be positive on every axis, got {radius!r}")
    return radii


def _subtract_background(volume: np.ndarray, background: np.ndarray) -> np.ndarray:
    """`volume - background`, floored at 0 and cast back to the input dtype.

    Unlike the Gaussian high-pass, the residual of a rolling ball or an
    opening is non-negative by construction (the background never rises
    above the signal), so it is *not* re-baselined: background sits at 0
    and spots keep their height above it. The floor only absorbs
    interpolation overshoot; integer dtypes are then clipped to range.
    """
    residual = np.maximum(volume.astype(np.float64) - background, 0.0)
    if np.issubdtype(volume.dtype, np.integer):
        info = np.iinfo(volume.dtype)
        residual = np.clip(np.rint(residual), info.min, info.max)
    return residual.astype(volume.dtype)


def _block_min(volume: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    """Downsample by taking the minimum of each `factors`-sized block.

    The edge is padded by replication up to a whole number of blocks.
    Minimum, not mean, so a spot can't lift the shrunken background.
    """
    pads = [(0, (-n) % f) for n, f in zip(volume.shape, factors)]
    v = np.pad(volume, pads, mode="edge")
    (nz, ny, nx), (fz, fy, fx) = v.shape, factors
    return v.reshape(nz // fz, fz, ny // fy, fy, nx // fx, fx).min(axis=(1, 3, 5))


def remove_background_rolling_ball_3d(
    volume: np.ndarray,
    radius: float | tuple[float, float, float] = 10.0,
) -> np.ndarray:
    """Remove background from a single (Z, Y, X) volume with a rolling ball.

    The background is what a ball rolled underneath the intensity surface
    can reach (skimage's `rolling_ball`); anything narrower than the ball —
    the spots — is left in the residual. Use a radius larger than the
    biggest spot.

    Args:
        volume: 3D array (Z, Y, X).
        radius: ball radius in voxels — one float, or (r_z, r_y, r_x) for an
            ellipsoid on anisotropic voxels. The ball's height in intensity
            units equals its lateral radius, as in ImageJ.

    Returns:
        `volume - background`, non-negative, in the input dtype (see
        `_subtract_background`).

    Raises:
        ValueError: if `volume` is not 3D or any radius is non-positive.
    """
    from skimage.restoration import ellipsoid_kernel, rolling_ball
    from skimage.transform import resize

    _require_3d(volume)
    radii = _positive_radius_zyx(radius)

    factors = tuple(
        max(1, int(np.ceil(r / _ROLLING_BALL_MAX_SHRUNK_RADIUS))) for r in radii
    )
    shrunk_radii = [r / f for r, f in zip(radii, factors)]
    semi_axes = [max(1, int(round(r))) for r in shrunk_radii]
    # `ellipsoid_kernel` takes full extents and uses `shape // 2` as the
    # semi-axes; its `intensity` is the ball's height (lateral radius).
    kernel = ellipsoid_kernel(
        tuple(2 * a + 1 for a in semi_axes), intensity=shrunk_radii[2]
    )

    small = _block_min(volume.astype(np.float64), factors)
    background = rolling_ball(small, kernel=kernel)
    if any(f > 1 for f in factors):
        padded_shape = tuple(s * f for s, f in zip(small.shape, factors))
        background = resize(
            background, padded_shape, order=1, mode="edge",
            anti_aliasing=False, preserve_range=True,
        )
        background = background[tuple(slice(0, n) for n in volume.shape)]
    return _subtract_background(volume, background)


def remove_background_erosion_3d(
    volume: np.ndarray,
    radius: float | tuple[float, float, float] = 10.0,
) -> np.ndarray:
    """Remove background from a single (Z, Y, X) volume by grey-scale
    erosion followed by dilation (a morphological opening; the result is a
    white top-hat).

    Erosion alone (a local minimum) is a background estimate, but it is
    biased low everywhere the background varies, and subtracting it leaves
    a halo around every spot. Dilating it back with the same box restores
    the background's level while still leaving out anything narrower than
    the box. Both filters use a flat cuboid, which is separable — this is
    fast at any radius.

    Args:
        volume: 3D array (Z, Y, X).
        radius: half-width of the box in voxels — one float, or
            (r_z, r_y, r_x). The box is `2 * round(r) + 1` voxels per axis;
            use a radius larger than the biggest spot.

    Returns:
        `volume - opening`, non-negative, in the input dtype.

    Raises:
        ValueError: if `volume` is not 3D or any radius is non-positive.
    """
    from scipy.ndimage import grey_dilation, grey_erosion

    _require_3d(volume)
    radii = _positive_radius_zyx(radius)
    size = tuple(2 * int(round(r)) + 1 for r in radii)
    eroded = grey_erosion(volume, size=size, mode="nearest")
    background = grey_dilation(eroded, size=size, mode="nearest")
    return _subtract_background(volume, background)


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
