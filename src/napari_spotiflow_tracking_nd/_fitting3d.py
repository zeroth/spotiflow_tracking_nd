"""Fit a 3D Gaussian to each detected spot, and paint labelled masks.

Pure logic: no Qt, no napari. `DetectionWorker3D` calls this once per
volume, inside the same pass that reads the volume for detection.

Fitting is `scipy.optimize.curve_fit` per spot, which measured 12.5 ms on
a 7x11x11 patch of the reference deconvolved data — 28 minutes for that
series' ~1.08M spots across 8 processes, which is why there is no GPU
backend here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

# Voxels, (z, y, x). Anisotropic because z is sampled coarsely: 0.210 µm
# against 0.104 µm laterally on the reference dataset. This is also the
# geometry the 12.5 ms/fit figure was measured against.
DEFAULT_PATCH_RADIUS_ZYX = (3, 5, 5)
# What a spot gets when its fit fails or was never attempted.
DEFAULT_FALLBACK_RADIUS_ZYX = (1.5, 2.5, 2.5)

_TWO_PI_POW_1_5 = (2.0 * np.pi) ** 1.5


@dataclass
class SpotFit3D:
    """One spot's fitted Gaussian, in voxel units.

    `converged` is False when `curve_fit` failed, when it returned
    something implausible, or when the spot sat too close to a volume edge
    to extract a full patch. In all three cases the other fields carry
    fallback values rather than nothing, so a detection is never dropped
    for want of a fit.
    """

    z0: float
    y0: float
    x0: float
    sigma_z: float
    sigma_y: float
    sigma_x: float
    amplitude: float
    background: float
    converged: bool

    @property
    def integrated_intensity(self) -> float:
        """The Gaussian's volume integral, analytically.

        No mask has to be painted to measure a spot — which is what keeps
        a 349-timepoint run from having to build 113 GB of label volumes.
        """
        return float(
            self.amplitude * _TWO_PI_POW_1_5
            * self.sigma_z * self.sigma_y * self.sigma_x
        )


def _gaussian_3d(coords, z0, y0, x0, sigma_z, sigma_y, sigma_x, amplitude, background):
    z, y, x = coords
    return (
        background
        + amplitude
        * np.exp(
            -(
                ((z - z0) ** 2) / (2.0 * sigma_z ** 2)
                + ((y - y0) ** 2) / (2.0 * sigma_y ** 2)
                + ((x - x0) ** 2) / (2.0 * sigma_x ** 2)
            )
        )
    ).ravel()


def _patch_grid(radius_zyx):
    """Patch-local coordinates, centred on 0 — so a fitted offset is
    relative to the rounded voxel the patch was cut around."""
    rz, ry, rx = radius_zyx
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return (
        zz.ravel().astype(float),
        yy.ravel().astype(float),
        xx.ravel().astype(float),
    )


def extract_patch(volume, centre_zyx, radius_zyx):
    """The cube of `volume` around `centre_zyx`, or None if it doesn't fit.

    Returning None rather than a clipped patch keeps the fit honest: a
    partial patch biases the centre and the background toward whichever
    side was cut off.
    """
    rz, ry, rx = radius_zyx
    z, y, x = (int(round(float(c))) for c in centre_zyx)
    if not (
        rz <= z < volume.shape[0] - rz
        and ry <= y < volume.shape[1] - ry
        and rx <= x < volume.shape[2] - rx
    ):
        return None
    return volume[z - rz:z + rz + 1, y - ry:y + ry + 1, x - rx:x + rx + 1].astype(np.float64)


# How far the fitted centre may move from the detected one, in voxels.
# `curve_fit` is unbounded, so on a crowded patch it will happily slide
# off this spot and onto a brighter neighbour several voxels away, then
# report *that* spot's amplitude under this spot's identity. Measured on
# synthetic spots with the +/-1 voxel jitter a rounded detection carries,
# a genuine fit moves at most ~1.5 voxels, so 2.0 keeps every real spot
# and rejects the migration.
_MAX_CENTRE_SHIFT = 2.0


def _max_shift(radius: float) -> float:
    """The shift cap, never wider than the patch it is measured in."""
    return min(float(radius), _MAX_CENTRE_SHIFT)


def _fallback(centre_zyx, fallback_radius_zyx) -> SpotFit3D:
    """The spot's row when it was never successfully measured.

    Sigma is a real fallback, because `paint_label_mask` still has to give
    the spot an ellipsoid. The intensities are NaN, not patch statistics:
    on a crowded patch `max - min` is a *neighbour's* brightness, which
    would put failed rows at the top of an `integrated_intensity`
    distribution instead of out of it. NaN drops out of a mean; zero does
    not.
    """
    z, y, x = (float(c) for c in centre_zyx)
    sz, sy, sx = (float(r) for r in fallback_radius_zyx)
    return SpotFit3D(z, y, x, sz, sy, sx, np.nan, np.nan, converged=False)


def fit_patch(patch, centre_zyx, radius_zyx, fallback_radius_zyx) -> SpotFit3D:
    """Fit one extracted patch. Never raises — a bad fit is a fallback."""
    coords = _patch_grid(radius_zyx)
    background0 = float(np.min(patch))
    amplitude0 = float(np.max(patch) - background0)
    p0 = (
        0.0, 0.0, 0.0,
        max(radius_zyx[0] / 2.0, 0.5),
        max(radius_zyx[1] / 2.5, 0.5),
        max(radius_zyx[2] / 2.5, 0.5),
        amplitude0 if amplitude0 > 0 else 1.0,
        background0,
    )
    try:
        popt, _ = curve_fit(_gaussian_3d, coords, patch.ravel(), p0=p0, maxfev=2000)
    except Exception:
        return _fallback(centre_zyx, fallback_radius_zyx)

    dz, dy, dx, sigma_z, sigma_y, sigma_x, amplitude, background = popt
    # `curve_fit` is unbounded here (bounds would cost the speed the whole
    # design rests on), so the result needs checking rather than trusting:
    # the model is even in sigma, so a converged fit may report sigma = -2.1,
    # and noise can "converge" on a sigma wider than the patch.
    sigma_z, sigma_y, sigma_x = abs(float(sigma_z)), abs(float(sigma_y)), abs(float(sigma_x))
    # `amplitude > 0` is not enough: on a flat patch `curve_fit` converges
    # happily with an amplitude of ~1e-07 on a background of 100, which is
    # arithmetically positive and physically nothing. The amplitude has to
    # be a real fraction of the patch's own dynamic range.
    patch_range = float(np.max(patch) - np.min(patch))
    plausible = (
        np.all(np.isfinite(popt))
        and 0.0 < sigma_z <= 2 * radius_zyx[0] + 1
        and 0.0 < sigma_y <= 2 * radius_zyx[1] + 1
        and 0.0 < sigma_x <= 2 * radius_zyx[2] + 1
        and abs(dz) <= _max_shift(radius_zyx[0])
        and abs(dy) <= _max_shift(radius_zyx[1])
        and abs(dx) <= _max_shift(radius_zyx[2])
        and patch_range > 0.0
        and amplitude >= 0.2 * patch_range
    )
    if not plausible:
        return _fallback(centre_zyx, fallback_radius_zyx)

    # The patch was cut around the rounded voxel, so the offset is relative
    # to that, not to the sub-voxel detection coordinate.
    cz, cy, cx = (float(round(float(c))) for c in centre_zyx)
    return SpotFit3D(
        cz + float(dz), cy + float(dy), cx + float(dx),
        sigma_z, sigma_y, sigma_x,
        float(amplitude), float(background),
        converged=True,
    )


def _fit_patch(args):
    """Module-level, single-tuple entry point so it can cross a process
    boundary on Windows (spawn pickles the callable and its arguments)."""
    patch, centre_zyx, radius_zyx, fallback_radius_zyx = args
    return fit_patch(patch, centre_zyx, radius_zyx, fallback_radius_zyx)


def fit_spots_3d(
    volume,
    points,
    patch_radius_zyx=DEFAULT_PATCH_RADIUS_ZYX,
    fallback_radius_zyx=DEFAULT_FALLBACK_RADIUS_ZYX,
    pool=None,
) -> list[SpotFit3D]:
    """Fit every spot in `points` against `volume`, in `points` order.

    `pool` is an optional `concurrent.futures.Executor` owned by the
    caller. It is injected rather than created here because a pool per
    volume would spend ~0.5-1 s per process starting up, 349 times over.

    What crosses the process boundary is a *patch* per spot (~7 KB), never
    the volume (~323 MB).
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return []
    points = points.reshape(-1, 3)

    fits: list[SpotFit3D | None] = [None] * len(points)
    jobs: list[tuple[int, tuple]] = []
    for i, centre in enumerate(points):
        patch = extract_patch(volume, centre, patch_radius_zyx)
        if patch is None:
            fits[i] = _fallback(centre, fallback_radius_zyx)
            continue
        jobs.append(
            (i, (patch, tuple(float(c) for c in centre),
                 tuple(patch_radius_zyx), tuple(fallback_radius_zyx)))
        )

    if pool is not None and len(jobs) > 1:
        for (i, _args), fit in zip(jobs, pool.map(_fit_patch, [a for _, a in jobs])):
            fits[i] = fit
    else:
        for i, args in jobs:
            fits[i] = _fit_patch(args)

    return [f for f in fits if f is not None]


DEFAULT_N_SIGMA = 2.0
# A uint16 label of 65536 wraps to 0 — background — which would silently
# erase spots from the mask rather than fail. Above this many spots the
# mask is uint32 instead.
_UINT16_MAX_LABEL = 65535


def paint_label_mask(shape, fits, n_sigma: float = DEFAULT_N_SIGMA) -> np.ndarray:
    """A labelled volume: label `i + 1` for `fits[i]`.

    Each spot is the ellipsoid within `n_sigma` standard deviations of its
    fitted centre, so an anisotropic fit paints an anisotropic blob. Spots
    are painted in order, so where two overlap the later label wins.

    Every fit gets at least its centre voxel, including failed ones and
    ones whose ellipsoid is narrower than a voxel: a spot present in the
    CSV is present in the mask.
    """
    dtype = np.uint16 if len(fits) <= _UINT16_MAX_LABEL else np.uint32
    mask = np.zeros(shape, dtype=dtype)

    for label, fit in enumerate(fits, start=1):
        radius_z = max(n_sigma * fit.sigma_z, 0.0)
        radius_y = max(n_sigma * fit.sigma_y, 0.0)
        radius_x = max(n_sigma * fit.sigma_x, 0.0)

        z_lo = max(int(np.floor(fit.z0 - radius_z)), 0)
        z_hi = min(int(np.ceil(fit.z0 + radius_z)) + 1, shape[0])
        y_lo = max(int(np.floor(fit.y0 - radius_y)), 0)
        y_hi = min(int(np.ceil(fit.y0 + radius_y)) + 1, shape[1])
        x_lo = max(int(np.floor(fit.x0 - radius_x)), 0)
        x_hi = min(int(np.ceil(fit.x0 + radius_x)) + 1, shape[2])

        if z_lo < z_hi and y_lo < y_hi and x_lo < x_hi and min(radius_z, radius_y, radius_x) > 0:
            zz, yy, xx = np.ogrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
            inside = (
                ((zz - fit.z0) / radius_z) ** 2
                + ((yy - fit.y0) / radius_y) ** 2
                + ((xx - fit.x0) / radius_x) ** 2
            ) <= 1.0
            mask[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][inside] = label

        # The ellipsoid can round away to nothing for a very sharp spot;
        # the centre voxel is painted unconditionally so the spot is never
        # in the CSV but absent from the mask.
        cz, cy, cx = (int(round(fit.z0)), int(round(fit.y0)), int(round(fit.x0)))
        if 0 <= cz < shape[0] and 0 <= cy < shape[1] and 0 <= cx < shape[2]:
            mask[cz, cy, cx] = label

    return mask
