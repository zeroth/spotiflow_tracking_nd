# 3D Gaussian Spot Fitting and Labelled Masks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a 3D Gaussian to each detected spot during the detection pass, add its size/brightness/background to the spots CSV, and paint a labelled mask for overlay or on-demand TIFF export.

**Architecture:** A new pure-Python `_fitting3d.py` does the fitting (scipy `curve_fit` per spot, patches sent to a `ProcessPoolExecutor` owned by the caller) and the mask painting (analytic ellipsoid per spot). `DetectionWorker3D` gains three optional arguments and keeps its single pass over the data. The widget gains a checkbox, two patch-radius spin boxes, and a mask-export folder.

**Tech Stack:** numpy, scipy.optimize.curve_fit, concurrent.futures.ProcessPoolExecutor, tifffile, qtpy, napari, pytest.

**Spec:** [docs/superpowers/specs/2026-09-24-gaussian-fit-and-mask-design.md](../specs/2026-09-24-gaussian-fit-and-mask-design.md)

## Global Constraints

- `_fitting3d.py` imports no Qt and no napari. It is pure logic, unit-tested directly.
- **Worker processes receive patches, never the volume.** One 323 MB volume copied per spot is fatal at 3,101 spots/timepoint; a patch is ~6.8 KB.
- **The process pool is created once per run, not per timepoint.** Windows spawns (~0.5–1 s/process); a pool per volume would burn ~6 minutes across 349 timepoints. The pool is therefore *injected* into `fit_spots_3d`, not created by it.
- A detection never disappears because its fit failed or its patch ran off an edge: it keeps its CSV row with `converged=False` and the fallback σ, and it is still painted in the mask.
- CSV values stay in **voxel units**. The JSON sidecar already records calibration; physical units are Phase 3's job.
- `z, y, x` remain exactly as Spotiflow detected them. The refined centre is added as separate `fit_z, fit_y, fit_x` columns, so turning fitting on never shifts coordinates under Phase 2 tracking.
- Existing behaviour with fitting off is byte-identical: same CSV columns, same performance.
- Tests pass `pool=None` so no processes spawn under pytest.

## Review Focus

Input classes the spec implies that no happy path exercises, most likely to bite first. Each has a test pinned to the task that owns the code.

1. **`curve_fit` returns a negative or absurd σ.** The model is even in σ, so a converged fit can legitimately come back with σ = −2.1; painting `n_sigma × −2.1` gives an empty mask and `integrated_intensity` goes negative. Must be sanitised, not trusted. (Task 1)
2. **More than 65,535 spots in one timepoint.** A uint16 label of 65,536 wraps to 0 — background — silently erasing spots from the mask. Plausible on deconvolved data at a lower threshold. (Task 2)
3. **A timepoint with zero spots while fitting is on.** `fit_spots_3d` gets an empty array; must return `[]` rather than raise, and write no rows. (Task 1)
4. **Every fit in a timepoint fails** (a flat or saturated volume). All rows `converged=False`, mask painted entirely from fallbacks, no division by zero in `integrated_intensity`. (Tasks 1, 2)
5. **A spot whose 2σ ellipsoid is smaller than one voxel.** A sharp spot with σ_z ≈ 0.3 rounds to an empty ellipsoid, so a spot in the CSV has no mask voxel at all. (Task 2)

---

### Task 1: `_fitting3d.py` — fitting one volume's spots

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_fitting3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SpotFit3D` dataclass with fields `z0 y0 x0 sigma_z sigma_y sigma_x amplitude background converged` and property `integrated_intensity`; `DEFAULT_PATCH_RADIUS_ZYX = (3, 5, 5)`; `DEFAULT_FALLBACK_RADIUS_ZYX = (1.5, 2.5, 2.5)`; `extract_patch(volume, centre_zyx, radius_zyx) -> np.ndarray | None`; `fit_patch(patch, centre_zyx, radius_zyx, fallback_radius_zyx) -> SpotFit3D`; `_fit_patch(args_tuple) -> SpotFit3D` (module-level, picklable); `fit_spots_3d(volume, points, patch_radius_zyx=..., fallback_radius_zyx=..., pool=None) -> list[SpotFit3D]`. Tasks 2 and 4 consume these.

Note on the signature: the spec wrote `max_workers=None`; it is `pool=None` here, because "one pool per run" requires the caller to own the pool. Task 4 creates it.

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


def synth_volume(shape=(24, 64, 64), spots=(), background=100.0, dtype=np.uint16):
    """A volume containing Gaussians at known centres with known sigmas.

    `spots` is a sequence of (z0, y0, x0, sigma_z, sigma_y, sigma_x, amplitude).
    """
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
    volume = np.full(shape, background, dtype=float)
    for z0, y0, x0, sz, sy, sx, amp in spots:
        volume += amp * np.exp(
            -(((zz - z0) ** 2) / (2 * sz ** 2)
              + ((yy - y0) ** 2) / (2 * sy ** 2)
              + ((xx - x0) ** 2) / (2 * sx ** 2))
        )
    return volume.astype(dtype)


class TestFitRecoversKnownGaussians:
    """The test that matters: a wrong model function, a transposed axis or a
    bad initial guess all survive a smoke test and all fail this."""

    def test_recovers_centre_and_anisotropic_sigmas(self):
        from napari_spotiflow_tracking_nd._fitting3d import fit_spots_3d

        truth = (11.4, 31.6, 30.2, 1.4, 2.6, 2.1, 4000.0)
        volume = synth_volume(spots=[truth])
        # Detection gives integer-ish coordinates; the fit refines them.
        points = np.array([[11.0, 32.0, 30.0]])

        fits = fit_spots_3d(volume, points)

        assert len(fits) == 1
        fit = fits[0]
        assert fit.converged
        assert fit.z0 == pytest.approx(truth[0], abs=0.3)
        assert fit.y0 == pytest.approx(truth[1], abs=0.3)
        assert fit.x0 == pytest.approx(truth[2], abs=0.3)
        assert fit.sigma_z == pytest.approx(truth[3], rel=0.15)
        assert fit.sigma_y == pytest.approx(truth[4], rel=0.15)
        assert fit.sigma_x == pytest.approx(truth[5], rel=0.15)
        assert fit.amplitude == pytest.approx(truth[6], rel=0.15)
        assert fit.background == pytest.approx(100.0, rel=0.3)

    def test_does_not_confuse_the_axes(self):
        # A spot much wider in y than x: swapping the axes anywhere in the
        # model or the grid makes this fail while the symmetric case passes.
        from napari_spotiflow_tracking_nd._fitting3d import fit_spots_3d

        volume = synth_volume(spots=[(12.0, 32.0, 32.0, 1.2, 4.0, 1.5, 3000.0)])
        fits = fit_spots_3d(volume, np.array([[12.0, 32.0, 32.0]]))

        assert fits[0].sigma_y > 2.0 * fits[0].sigma_x

    def test_fits_several_spots_independently(self):
        from napari_spotiflow_tracking_nd._fitting3d import fit_spots_3d

        volume = synth_volume(spots=[
            (12.0, 20.0, 20.0, 1.3, 2.0, 2.0, 3000.0),
            (12.0, 44.0, 44.0, 1.3, 3.5, 3.5, 2000.0),
        ])
        fits = fit_spots_3d(volume, np.array([[12.0, 20.0, 20.0], [12.0, 44.0, 44.0]]))

        assert all(f.converged for f in fits)
        assert fits[1].sigma_y > fits[0].sigma_y


class TestIntegratedIntensity:
    def test_matches_a_numeric_sum_over_a_synthesised_spot(self):
        from napari_spotiflow_tracking_nd._fitting3d import fit_spots_3d

        amp, sz, sy, sx = 5000.0, 1.6, 2.4, 2.4
        volume = synth_volume(shape=(32, 64, 64), spots=[(16.0, 32.0, 32.0, sz, sy, sx, amp)],
                              background=0.0, dtype=np.float64)
        fits = fit_spots_3d(volume, np.array([[16.0, 32.0, 32.0]]),
                            patch_radius_zyx=(6, 10, 10))

        numeric = float(volume.sum())
        assert fits[0].integrated_intensity == pytest.approx(numeric, rel=0.1)

    def test_is_analytic_not_dependent_on_the_volume(self):
        from napari_spotiflow_tracking_nd._fitting3d import SpotFit3D

        fit = SpotFit3D(0, 0, 0, 1.0, 2.0, 3.0, 10.0, 0.0, True)
        expected = 10.0 * (2 * np.pi) ** 1.5 * 1.0 * 2.0 * 3.0

        assert fit.integrated_intensity == pytest.approx(expected)


class TestFailureAndFallback:
    def test_a_flat_patch_does_not_converge_and_keeps_the_spot(self):
        # ~7% of real spots take this path, so it is a main path.
        from napari_spotiflow_tracking_nd._fitting3d import fit_spots_3d

        volume = np.full((24, 64, 64), 100, dtype=np.uint16)
        fits = fit_spots_3d(volume, np.array([[12.0, 32.0, 32.0]]))

        assert len(fits) == 1, "a failed fit must not drop the spot"
        assert not fits[0].converged
        assert fits[0].sigma_z > 0 and fits[0].sigma_y > 0 and fits[0].sigma_x > 0
        assert np.isfinite(fits[0].integrated_intensity)

    def test_a_spot_too_close_to_an_edge_is_not_fitted_but_is_kept(self):
        from napari_spotiflow_tracking_nd._fitting3d import (
            DEFAULT_FALLBACK_RADIUS_ZYX, fit_spots_3d,
        )

        volume = synth_volume(spots=[(12.0, 32.0, 32.0, 1.3, 2.0, 2.0, 3000.0)])
        fits = fit_spots_3d(volume, np.array([[0.0, 1.0, 1.0]]))  # in the corner

        assert len(fits) == 1
        assert not fits[0].converged
        assert fits[0].sigma_z == pytest.approx(DEFAULT_FALLBACK_RADIUS_ZYX[0])
        assert fits[0].z0 == 0.0 and fits[0].y0 == 1.0 and fits[0].x0 == 1.0

    def test_a_negative_sigma_from_curve_fit_is_never_returned(self):
        # The model is even in sigma, so curve_fit may converge on a
        # negative one. Painting n_sigma * -2.1 gives an empty mask and a
        # negative integrated intensity.
        from napari_spotiflow_tracking_nd._fitting3d import (
            DEFAULT_PATCH_RADIUS_ZYX, fit_patch,
        )

        volume = synth_volume(spots=[(12.0, 32.0, 32.0, 1.4, 2.2, 2.2, 3000.0)])
        patch = volume[9:16, 27:38, 27:38].astype(float)

        fit = fit_patch(patch, (12.0, 32.0, 32.0), DEFAULT_PATCH_RADIUS_ZYX, (1.5, 2.5, 2.5))

        assert fit.sigma_z > 0 and fit.sigma_y > 0 and fit.sigma_x > 0
        assert fit.integrated_intensity > 0

    def test_an_absurd_sigma_is_rejected_as_not_converged(self):
        # A fit that "succeeds" with sigma wider than the patch describes
        # nothing; trusting it would paint a mask over the whole volume.
        from napari_spotiflow_tracking_nd._fitting3d import fit_patch

        rng = np.random.default_rng(0)
        patch = rng.normal(100, 1, size=(7, 11, 11))  # noise, no spot

        fit = fit_patch(patch, (12.0, 32.0, 32.0), (3, 5, 5), (1.5, 2.5, 2.5))

        if fit.converged:
            assert fit.sigma_z <= 7 and fit.sigma_y <= 11 and fit.sigma_x <= 11
        else:
            assert fit.sigma_z == pytest.approx(1.5)


class TestProcessPool:
    """The one test that crosses a real process boundary.

    Everything else passes `pool=None`. This is what would catch a
    `_fit_patch` that stopped being module-level, a `SpotFit3D` that
    stopped being picklable, or a patch argument that accidentally became
    the whole volume — all of which fail only under `spawn`.
    """

    def test_a_real_pool_agrees_with_serial_fitting(self):
        from concurrent.futures import ProcessPoolExecutor

        from napari_spotiflow_tracking_nd._fitting3d import fit_spots_3d

        volume = synth_volume(spots=[
            (12.0, 20.0, 20.0, 1.3, 2.2, 2.2, 3000.0),
            (12.0, 44.0, 44.0, 1.5, 2.8, 2.8, 2500.0),
        ])
        points = np.array([[12.0, 20.0, 20.0], [12.0, 44.0, 44.0]])

        serial = fit_spots_3d(volume, points)
        with ProcessPoolExecutor(max_workers=2) as pool:
            parallel = fit_spots_3d(volume, points, pool=pool)

        assert len(parallel) == len(serial) == 2
        for got, want in zip(parallel, serial):
            assert got.converged == want.converged
            assert got.sigma_z == pytest.approx(want.sigma_z, rel=1e-6)
            assert got.sigma_y == pytest.approx(want.sigma_y, rel=1e-6)
            assert got.z0 == pytest.approx(want.z0, rel=1e-6)

    def test_results_stay_in_the_order_of_the_input_points(self):
        # `pool.map` preserves order, but an edge spot is filled in before
        # the pool runs — so the two paths have to be stitched back
        # together correctly.
        from concurrent.futures import ProcessPoolExecutor

        from napari_spotiflow_tracking_nd._fitting3d import fit_spots_3d

        volume = synth_volume(spots=[(12.0, 32.0, 32.0, 1.3, 2.2, 2.2, 3000.0)])
        # First point is in the corner (no fit), second is the real spot.
        points = np.array([[0.0, 0.0, 0.0], [12.0, 32.0, 32.0]])

        with ProcessPoolExecutor(max_workers=2) as pool:
            fits = fit_spots_3d(volume, points, pool=pool)

        assert len(fits) == 2
        assert not fits[0].converged, "the corner spot must stay first"
        assert fits[1].converged, "the real spot must stay second"


class TestEmptyAndShape:
    def test_no_spots_returns_no_fits(self):
        from napari_spotiflow_tracking_nd._fitting3d import fit_spots_3d

        volume = np.zeros((8, 16, 16), dtype=np.uint16)

        assert fit_spots_3d(volume, np.empty((0, 3))) == []

    def test_extract_patch_returns_none_off_the_edge(self):
        from napari_spotiflow_tracking_nd._fitting3d import extract_patch

        volume = np.zeros((8, 16, 16), dtype=np.uint16)

        assert extract_patch(volume, (0, 8, 8), (3, 5, 5)) is None
        assert extract_patch(volume, (4, 8, 8), (3, 5, 5)).shape == (7, 11, 11)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._fitting3d'`

- [ ] **Step 3: Implement it**

Create `src/napari_spotiflow_tracking_nd/_fitting3d.py`:

```python
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


def _fallback(centre_zyx, fallback_radius_zyx, patch=None) -> SpotFit3D:
    z, y, x = (float(c) for c in centre_zyx)
    sz, sy, sx = (float(r) for r in fallback_radius_zyx)
    if patch is None:
        background = amplitude = 0.0
    else:
        background = float(np.min(patch))
        amplitude = float(np.max(patch) - background)
    return SpotFit3D(z, y, x, sz, sy, sx, amplitude, background, converged=False)


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
        return _fallback(centre_zyx, fallback_radius_zyx, patch)

    dz, dy, dx, sigma_z, sigma_y, sigma_x, amplitude, background = popt
    # `curve_fit` is unbounded here (bounds would cost the speed the whole
    # design rests on), so the result needs checking rather than trusting:
    # the model is even in sigma, so a converged fit may report sigma = -2.1,
    # and noise can "converge" on a sigma wider than the patch.
    sigma_z, sigma_y, sigma_x = abs(float(sigma_z)), abs(float(sigma_y)), abs(float(sigma_x))
    plausible = (
        np.all(np.isfinite(popt))
        and 0.0 < sigma_z <= 2 * radius_zyx[0] + 1
        and 0.0 < sigma_y <= 2 * radius_zyx[1] + 1
        and 0.0 < sigma_x <= 2 * radius_zyx[2] + 1
        and abs(dz) <= radius_zyx[0]
        and abs(dy) <= radius_zyx[1]
        and abs(dx) <= radius_zyx[2]
        and amplitude > 0.0
    )
    if not plausible:
        return _fallback(centre_zyx, fallback_radius_zyx, patch)

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run the whole suite**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/ -q`
Expected: PASS — nothing else touched yet.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_fitting3d.py src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py
git commit -m "feat(nd): 3D Gaussian spot fitting"
```

---

### Task 2: Labelled mask painting

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_fitting3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py`

**Interfaces:**
- Consumes: `SpotFit3D` (Task 1).
- Produces: `paint_label_mask(shape, fits, n_sigma=2.0) -> np.ndarray` and `DEFAULT_N_SIGMA = 2.0`. Tasks 4 and 5 consume it.

Note: the spec's signature took `points` and `fallback_radius_zyx` too. Both are redundant — each `SpotFit3D` already carries its centre and (for a failed fit) its fallback σ. Dropped rather than left as parameters that can disagree with the fits.

- [ ] **Step 1: Write the failing tests**

Append to `src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py`:

```python
class TestPaintLabelMask:
    def test_one_label_per_spot_in_order(self):
        from napari_spotiflow_tracking_nd._fitting3d import SpotFit3D, paint_label_mask

        fits = [
            SpotFit3D(6.0, 10.0, 10.0, 1.5, 2.0, 2.0, 100.0, 0.0, True),
            SpotFit3D(6.0, 30.0, 30.0, 1.5, 2.0, 2.0, 100.0, 0.0, True),
        ]

        mask = paint_label_mask((16, 48, 48), fits)

        assert mask[6, 10, 10] == 1
        assert mask[6, 30, 30] == 2
        assert set(np.unique(mask)) == {0, 1, 2}

    def test_the_ellipsoid_is_anisotropic_as_the_sigmas_say(self):
        from napari_spotiflow_tracking_nd._fitting3d import SpotFit3D, paint_label_mask

        # Wide in y, narrow in z.
        fits = [SpotFit3D(10.0, 24.0, 24.0, 1.0, 4.0, 1.0, 100.0, 0.0, True)]
        mask = paint_label_mask((24, 48, 48), fits, n_sigma=2.0)

        extent_z = (mask[:, 24, 24] == 1).sum()
        extent_y = (mask[10, :, 24] == 1).sum()
        assert extent_y > 2 * extent_z

    def test_a_failed_fit_is_still_painted(self):
        from napari_spotiflow_tracking_nd._fitting3d import SpotFit3D, paint_label_mask

        fits = [SpotFit3D(8.0, 16.0, 16.0, 1.5, 2.5, 2.5, 0.0, 0.0, False)]

        mask = paint_label_mask((16, 32, 32), fits)

        assert mask[8, 16, 16] == 1, "a spot in the CSV must be in the mask"

    def test_a_sub_voxel_spot_still_occupies_its_centre(self):
        # 2 * 0.2 = 0.4 voxels: the ellipsoid rounds to nothing, and the
        # spot would vanish from the mask while sitting in the CSV.
        from napari_spotiflow_tracking_nd._fitting3d import SpotFit3D, paint_label_mask

        fits = [SpotFit3D(8.0, 16.0, 16.0, 0.2, 0.2, 0.2, 100.0, 0.0, True)]

        mask = paint_label_mask((16, 32, 32), fits)

        assert mask[8, 16, 16] == 1
        assert (mask == 1).sum() >= 1

    def test_overlapping_spots_let_the_later_label_win(self):
        from napari_spotiflow_tracking_nd._fitting3d import SpotFit3D, paint_label_mask

        fits = [
            SpotFit3D(8.0, 16.0, 16.0, 2.0, 3.0, 3.0, 100.0, 0.0, True),
            SpotFit3D(8.0, 16.0, 16.0, 2.0, 3.0, 3.0, 100.0, 0.0, True),
        ]

        mask = paint_label_mask((16, 32, 32), fits)

        assert mask[8, 16, 16] == 2

    def test_a_spot_at_the_volume_edge_is_clipped_not_wrapped(self):
        from napari_spotiflow_tracking_nd._fitting3d import SpotFit3D, paint_label_mask

        fits = [SpotFit3D(0.0, 0.0, 0.0, 2.0, 3.0, 3.0, 100.0, 0.0, True)]

        mask = paint_label_mask((16, 32, 32), fits)

        assert mask[0, 0, 0] == 1
        assert mask[-1, -1, -1] == 0, "the ellipsoid must not wrap around"

    def test_no_fits_gives_an_empty_mask(self):
        from napari_spotiflow_tracking_nd._fitting3d import paint_label_mask

        mask = paint_label_mask((8, 16, 16), [])

        assert mask.shape == (8, 16, 16)
        assert mask.max() == 0

    def test_more_than_65535_spots_do_not_wrap_to_background(self):
        # uint16 label 65536 wraps to 0, silently erasing spots from the
        # mask. Plausible on deconvolved data at a low threshold.
        from napari_spotiflow_tracking_nd._fitting3d import SpotFit3D, paint_label_mask

        n = 70000
        fits = [
            SpotFit3D(1.0, float(i % 300), float(i // 300), 0.4, 0.4, 0.4, 10.0, 0.0, True)
            for i in range(n)
        ]

        mask = paint_label_mask((3, 300, 300), fits)

        assert mask.dtype == np.uint32
        assert mask.max() == n
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py -k PaintLabelMask -v`
Expected: FAIL — `ImportError: cannot import name 'paint_label_mask'`

- [ ] **Step 3: Implement it**

Append to `src/napari_spotiflow_tracking_nd/_fitting3d.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py -v`
Expected: PASS — Task 1's tests plus the eight new ones.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_fitting3d.py src/napari_spotiflow_tracking_nd/_tests/test_fitting3d.py
git commit -m "feat(nd): paint labelled ellipsoid masks from spot fits"
```

---

### Task 3: Wire fitting into `DetectionWorker3D`

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_workers3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py`

**Interfaces:**
- Consumes: `fit_spots_3d`, `paint_label_mask`, `DEFAULT_PATCH_RADIUS_ZYX`, `DEFAULT_N_SIGMA` (Tasks 1–2).
- Produces: `DetectionWorker3D.__init__` gains `fit: bool = False`, `patch_radius_zyx=DEFAULT_PATCH_RADIUS_ZYX`, `mask_output_dir: Path | None = None`, `max_workers: int | None = None`. With `fit=False` the CSV and behaviour are unchanged. Task 4's widget passes these.

- [ ] **Step 1: Write the failing tests**

Append to `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py`:

```python
class TestDetectionWorkerFitting:
    @staticmethod
    def _model(n_spots=2):
        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.9, 0.7][:n_spots])
        model.predict.return_value = (
            np.array([[4.0, 8.0, 8.0], [4.0, 20.0, 20.0]][:n_spots]), details,
        )
        return model

    @staticmethod
    def _folder(tmp_path, n_timepoints=2):
        return write_timepoint_folder(
            tmp_path / "in", n_timepoints=n_timepoints, shape=(12, 32, 32),
        )

    def test_fitting_off_leaves_the_csv_columns_unchanged(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=self._model(), csv_path=tmp_path / "spots.csv",
        )
        worker.run()

        df = pd.read_csv(tmp_path / "spots.csv")
        assert list(df.columns) == ["frame", "z", "y", "x", "prob"]

    def test_fitting_on_adds_exactly_the_named_columns(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=self._model(), csv_path=tmp_path / "spots.csv",
            fit=True, max_workers=1,
        )
        worker.run()

        df = pd.read_csv(tmp_path / "spots.csv")
        assert list(df.columns) == [
            "frame", "z", "y", "x", "prob",
            "fit_z", "fit_y", "fit_x",
            "sigma_z", "sigma_y", "sigma_x",
            "amplitude", "background", "integrated_intensity", "converged",
        ]

    def test_the_detected_coordinates_are_not_replaced_by_the_fitted_ones(self, tmp_path):
        # Phase 2 tracking reads z/y/x; turning fitting on must not shift
        # them underneath it.
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=self._model(n_spots=1), csv_path=tmp_path / "spots.csv",
            fit=True, max_workers=1,
        )
        worker.run()

        df = pd.read_csv(tmp_path / "spots.csv")
        assert df["z"].iloc[0] == 4.0
        assert df["y"].iloc[0] == 8.0
        assert df["x"].iloc[0] == 8.0

    def test_no_mask_is_written_unless_a_folder_is_given(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=self._model(), csv_path=tmp_path / "spots.csv",
            fit=True, max_workers=1,
        )
        worker.run()

        assert not list(tmp_path.glob("**/*mask*"))

    def test_a_mask_folder_gets_one_label_tiff_per_timepoint(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path, n_timepoints=3)
        mask_dir = tmp_path / "masks"
        worker = DetectionWorker3D(
            files=files, model=self._model(), csv_path=tmp_path / "spots.csv",
            fit=True, mask_output_dir=mask_dir, max_workers=1,
        )
        worker.run()

        written = sorted(p.name for p in mask_dir.iterdir())
        assert written == [f.name for f in files]
        mask = tifffile.imread(str(mask_dir / files[0].name))
        assert mask.shape == (12, 32, 32)
        assert mask.max() == 2  # two spots, labels 1 and 2

    def test_a_timepoint_with_no_spots_writes_no_rows_and_still_masks(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path, n_timepoints=1)
        model = MagicMock()
        empty = MagicMock()
        empty.prob = np.empty((0,))
        model.predict.return_value = (np.empty((0, 3)), empty)
        mask_dir = tmp_path / "masks"

        worker = DetectionWorker3D(
            files=files, model=model, csv_path=tmp_path / "spots.csv",
            fit=True, mask_output_dir=mask_dir, max_workers=1,
        )
        worker.run()  # must not raise

        assert not (tmp_path / "spots.csv").exists()
        assert tifffile.imread(str(mask_dir / files[0].name)).max() == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py -k DetectionWorkerFitting -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'fit'`

- [ ] **Step 3: Implement it**

In `src/napari_spotiflow_tracking_nd/_workers3d.py`, add to the imports:

```python
from concurrent.futures import ProcessPoolExecutor

from napari_spotiflow_tracking_nd._fitting3d import (
    DEFAULT_N_SIGMA,
    DEFAULT_PATCH_RADIUS_ZYX,
    fit_spots_3d,
    paint_label_mask,
)
```

Replace `DetectionWorker3D.__init__`'s signature and body with:

```python
    def __init__(
        self,
        files: list[Path],
        model,  # a loaded model, or a zero-arg callable returning one
        csv_path: Path,
        prob_thresh: float | None = 0.5,
        min_distance: int = 2,
        fit: bool = False,
        patch_radius_zyx=DEFAULT_PATCH_RADIUS_ZYX,
        mask_output_dir: Path | None = None,
        max_workers: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._files = list(files)
        self._model = model
        self._csv_path = Path(csv_path)
        self._prob_thresh = prob_thresh
        self._min_distance = min_distance
        self._fit = fit
        self._patch_radius_zyx = tuple(patch_radius_zyx)
        self._mask_output_dir = Path(mask_output_dir) if mask_output_dir else None
        self._max_workers = max_workers
        self._cancelled = False
```

Then replace the body of `run()` with:

```python
    def run(self):
        try:
            # `model` may be a zero-arg callable so that a slow first
            # weight load happens here, on the worker thread, rather than
            # freezing the UI before this thread even starts. Resolved
            # once, and a failure lands on `errored` like any other.
            if callable(self._model) and not hasattr(self._model, "predict"):
                self._model = self._model()
            if self._csv_path.exists():
                self._csv_path.unlink()
            if self._mask_output_dir is not None:
                self._mask_output_dir.mkdir(parents=True, exist_ok=True)

            # One pool for the whole run: Windows spawns rather than forks,
            # so a pool per timepoint would spend ~0.5-1 s per process
            # starting up, 349 times over.
            pool = None
            if self._fit and self._max_workers != 1:
                pool = ProcessPoolExecutor(max_workers=self._max_workers)
            try:
                self._detect_all(pool)
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)
            self.run_finished.emit(str(self._csv_path))
        except Exception as exc:
            self.errored.emit(str(exc))

    def _detect_all(self, pool):
        header_written = False
        total = len(self._files)
        for t, path in enumerate(self._files):
            if self._cancelled:
                break
            volume = tifffile.imread(str(path))
            points, details = detect_spots_3d(
                volume, self._model,
                prob_thresh=self._prob_thresh,
                min_distance=self._min_distance,
            )

            fits = None
            if self._fit:
                fits = fit_spots_3d(
                    volume, points,
                    patch_radius_zyx=self._patch_radius_zyx,
                    pool=pool,
                )

            if len(points) > 0:
                df = self._rows_for(t, points, details, fits)
                df.to_csv(self._csv_path, mode="a", header=not header_written, index=False)
                header_written = True

            if self._mask_output_dir is not None:
                # Written even for a timepoint with no spots, so the mask
                # series stays aligned 1:1 with the input series.
                mask = paint_label_mask(volume.shape, fits or [], n_sigma=DEFAULT_N_SIGMA)
                tifffile.imwrite(str(self._mask_output_dir / path.name), mask)

            self.progress.emit(t + 1, total)

    @staticmethod
    def _rows_for(t, points, details, fits):
        coords = np.asarray(points)
        # Built column-by-column, not as one `np.hstack` block: hstacking
        # the integer frame index with float coords and probabilities
        # promotes everything to float64, and the CSV's `frame` column
        # comes out as `0.0, 1.0, ...`.
        columns = {
            "frame": np.full(len(coords), t, dtype=np.int64),
            "z": coords[:, 0],
            "y": coords[:, 1],
            "x": coords[:, 2],
            "prob": np.asarray(details.prob).reshape(-1),
        }
        if fits is not None:
            # `z, y, x` stay as detected; the refined centre is separate,
            # so nothing downstream shifts when fitting is turned on.
            columns.update(
                fit_z=np.array([f.z0 for f in fits]),
                fit_y=np.array([f.y0 for f in fits]),
                fit_x=np.array([f.x0 for f in fits]),
                sigma_z=np.array([f.sigma_z for f in fits]),
                sigma_y=np.array([f.sigma_y for f in fits]),
                sigma_x=np.array([f.sigma_x for f in fits]),
                amplitude=np.array([f.amplitude for f in fits]),
                background=np.array([f.background for f in fits]),
                integrated_intensity=np.array([f.integrated_intensity for f in fits]),
                converged=np.array([f.converged for f in fits]),
            )
        return pd.DataFrame(columns)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py -v`
Expected: PASS — the new tests plus every existing worker test (fitting off must be unchanged).

- [ ] **Step 5: Run the whole suite**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_workers3d.py src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py
git commit -m "feat(nd): fit spots and write label masks in the detection pass"
```

---

### Task 4: Detection widget controls

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_detection_widget3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py`

**Interfaces:**
- Consumes: `DetectionWorker3D`'s new arguments (Task 3); `fit_spots_3d`, `paint_label_mask` (Tasks 1–2).
- Produces: new widget attributes `_fit_check`, `_patch_radius_z_spin`, `_patch_radius_xy_spin`, `_export_masks_check`, `_mask_folder_edit`.

- [ ] **Step 1: Write the failing tests**

Append to `src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py`:

```python
class TestFitControls:
    def test_has_the_fitting_controls(self, detection_widget):
        widget, _ = detection_widget
        for attr in (
            "_fit_check", "_patch_radius_z_spin", "_patch_radius_xy_spin",
            "_export_masks_check", "_mask_folder_edit",
        ):
            assert hasattr(widget, attr), f"missing {attr}"

    def test_fitting_is_off_by_default(self, detection_widget):
        # It costs ~28 min on the reference deconvolved series; tuning runs
        # should not pay that unasked.
        widget, _ = detection_widget
        assert not widget._fit_check.isChecked()
        assert not widget._export_masks_check.isChecked()

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_the_run_passes_the_fit_settings_to_the_worker(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        TestPreview._setup(widget, viewer, tmp_path)
        mock_load_model.return_value = TestPreview._mock_model()
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))
        widget._fit_check.setChecked(True)
        widget._patch_radius_z_spin.setValue(4)
        widget._patch_radius_xy_spin.setValue(6)
        widget._export_masks_check.setChecked(True)
        widget._mask_folder_edit.setText(str(tmp_path / "masks"))

        captured = {}

        def _launch(worker):
            captured["fit"] = worker._fit
            captured["radius"] = worker._patch_radius_zyx
            captured["mask_dir"] = worker._mask_output_dir
            worker.run()

        widget._launch_worker = _launch
        widget._run_detection()

        assert captured["fit"] is True
        assert captured["radius"] == (4, 6, 6)
        assert captured["mask_dir"] == tmp_path / "masks"

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_exporting_masks_without_a_folder_is_refused(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        TestPreview._setup(widget, viewer, tmp_path)
        mock_load_model.return_value = TestPreview._mock_model()
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))
        widget._fit_check.setChecked(True)
        widget._export_masks_check.setChecked(True)  # but no folder

        with patch(
            "napari_spotiflow_tracking_nd._detection_widget3d.show_error"
        ) as mock_show_error:
            widget._run_detection()

        assert mock_show_error.called
        assert not (tmp_path / "spots.csv").exists()

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_preview_with_fitting_adds_a_labels_layer(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        TestPreview._setup(widget, viewer, tmp_path)
        mock_load_model.return_value = TestPreview._mock_model()
        widget._fit_check.setChecked(True)

        widget._run_preview()

        labels = [l for l in viewer.layers if l.__class__.__name__ == "Labels"]
        assert len(labels) == 1
        assert labels[0].name == "vol (spot mask preview)"

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_preview_without_fitting_adds_no_labels_layer(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        TestPreview._setup(widget, viewer, tmp_path)
        mock_load_model.return_value = TestPreview._mock_model()

        widget._run_preview()

        assert [l for l in viewer.layers if l.__class__.__name__ == "Labels"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py -k FitControls -v`
Expected: FAIL — `AssertionError: missing _fit_check`

- [ ] **Step 3: Add the controls**

In `_setup_ui`, immediately after the `params_group` block, insert:

```python
        fit_group = QGroupBox("Gaussian fit")
        fit_layout = QVBoxLayout()
        fit_group.setLayout(fit_layout)

        fit_row = QHBoxLayout()
        self._fit_check = QCheckBox("Fit Gaussians to spots")
        self._fit_check.setToolTip(
            "Adds each spot's size, brightness and local background to the "
            "CSV. Costs roughly 12 ms per spot, so a dense series takes tens "
            "of minutes — leave it off while tuning the threshold."
        )
        fit_row.addWidget(self._fit_check)
        fit_row.addWidget(QLabel("patch radius z:"))
        self._patch_radius_z_spin = QSpinBox()
        self._patch_radius_z_spin.setRange(1, 30)
        self._patch_radius_z_spin.setValue(DEFAULT_PATCH_RADIUS_ZYX[0])
        fit_row.addWidget(self._patch_radius_z_spin)
        fit_row.addWidget(QLabel("xy:"))
        self._patch_radius_xy_spin = QSpinBox()
        self._patch_radius_xy_spin.setRange(1, 30)
        self._patch_radius_xy_spin.setValue(DEFAULT_PATCH_RADIUS_ZYX[1])
        fit_row.addWidget(self._patch_radius_xy_spin)
        fit_layout.addLayout(fit_row)

        mask_row = QHBoxLayout()
        self._export_masks_check = QCheckBox("Export label masks to:")
        self._export_masks_check.setToolTip(
            "One uint16 label volume per timepoint — the same size as the "
            "input, so a long series is large. The masks are reconstructible "
            "from the CSV's fit columns, so this is off by default."
        )
        mask_row.addWidget(self._export_masks_check)
        self._mask_folder_edit = QLineEdit()
        mask_row.addWidget(self._mask_folder_edit)
        mask_browse_btn = QPushButton("Browse…")
        mask_browse_btn.clicked.connect(self._browse_mask_folder)
        mask_row.addWidget(mask_browse_btn)
        fit_layout.addLayout(mask_row)
        layout.addWidget(fit_group)
```

Add the import and the browse handler:

```python
from napari_spotiflow_tracking_nd._fitting3d import (
    DEFAULT_PATCH_RADIUS_ZYX,
    fit_spots_3d,
    paint_label_mask,
)
```

```python
    def _browse_mask_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select mask output folder")
        if folder:
            self._mask_folder_edit.setText(folder)
```

- [ ] **Step 4: Pass the settings to the worker**

In `_run_detection`, after the existing `csv_path` guard, add:

```python
        patch_radius_zyx = (
            self._patch_radius_z_spin.value(),
            self._patch_radius_xy_spin.value(),
            self._patch_radius_xy_spin.value(),
        )
        mask_output_dir = None
        if self._export_masks_check.isChecked():
            folder = self._mask_folder_edit.text()
            if not folder:
                show_error("Mask export is on but no mask folder is selected.")
                return
            mask_output_dir = Path(folder)
```

and extend the `DetectionWorker3D(...)` construction with:

```python
            fit=self._fit_check.isChecked(),
            patch_radius_zyx=patch_radius_zyx,
            mask_output_dir=mask_output_dir,
```

- [ ] **Step 5: Add the preview mask**

In `_run_preview`, capture the settings on the main thread alongside the existing ones:

```python
        do_fit = self._fit_check.isChecked()
        patch_radius_zyx = (
            self._patch_radius_z_spin.value(),
            self._patch_radius_xy_spin.value(),
            self._patch_radius_xy_spin.value(),
        )
```

Change `_detect` to return the mask too, fitting serially (one volume, one
click — no pool):

```python
        def _detect():
            model = load_the_model()
            volume = read_volume(files[t])
            points, _details = detect_spots_3d(
                volume, model,
                prob_thresh=prob_thresh,
                min_distance=min_distance,
            )
            if not do_fit:
                return points, None
            fits = fit_spots_3d(volume, points, patch_radius_zyx=patch_radius_zyx)
            return points, paint_label_mask(volume.shape, fits)
```

and in `_on_finished`, unpack and add the Labels layer:

```python
        def _on_finished(result):
            points, mask = result
            ...existing points-layer handling, unchanged...
            if mask is not None:
                mask_name = f"{layer.name} (spot mask preview)"
                if mask_name in self.viewer.layers:
                    self.viewer.layers[mask_name].data = mask
                else:
                    self.viewer.add_labels(mask, name=mask_name, scale=layer.scale[1:])
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py -v`
Expected: PASS — the new tests plus all existing detection-widget tests.

- [ ] **Step 7: Run the whole suite**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/ -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_detection_widget3d.py src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py
git commit -m "feat(nd): Gaussian-fit controls and mask preview in the Detection widget"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the finished behaviour. Produces nothing code depends on.

- [ ] **Step 1: Document the feature**

In `README.md`, after the **Spot detection** bullet, add:

```markdown
- **Gaussian fitting (optional)** — fits a 3D Gaussian to each detected
  spot and adds its refined centre, per-axis σ, amplitude, local
  background and analytic integrated intensity to the spots CSV. Roughly
  12 ms per spot, parallelised across processes: about 28 minutes for a
  dense 349-timepoint series, so it is off by default and the threshold
  tuning you do with **Preview** stays fast. `z, y, x` remain exactly as
  detected — the fit's centre is added as `fit_z, fit_y, fit_x`, so
  turning fitting on never moves coordinates under anything downstream.
  A spot whose fit fails, or which sits too close to a volume edge to cut
  a full patch, keeps its row with `converged=False` and a fallback σ
  rather than disappearing.
- **Label masks (optional)** — each fitted spot painted as an ellipsoid
  out to 2σ, one label per spot. Shown as a Labels layer when you
  **Preview** with fitting on; written to disk as one label TIFF per
  timepoint only when you ask, because a label volume is the same size as
  the data (a 349-timepoint series is ~113 GB). The masks are
  reconstructible from the CSV's fit columns, which is why the parameters
  are what's always saved.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(nd): Gaussian fitting and label masks"
```

---

## Self-Review Notes

- **Spec coverage:** `_fitting3d.py` fitting → Task 1; mask painting → Task 2; analytic measurements and the CSV columns → Tasks 1 and 3; worker integration, one-pool-per-run, patches-not-volumes, mask TIFF on demand → Task 3; widget controls and the preview Labels layer → Task 4; README → Task 5.
- **Placeholder scan:** none — every step carries real code or an exact command with expected output.
- **Type consistency:** `SpotFit3D`'s field names are used identically in Tasks 2, 3 and 4. `fit_spots_3d(volume, points, patch_radius_zyx=..., fallback_radius_zyx=..., pool=None)` and `paint_label_mask(shape, fits, n_sigma=...)` are defined in Tasks 1–2 and called with those exact signatures in Tasks 3–4. `DEFAULT_PATCH_RADIUS_ZYX` / `DEFAULT_N_SIGMA` are the single source for defaults.
- **Two deviations from the spec, both deliberate:** `fit_spots_3d` takes `pool` rather than `max_workers`, because "one pool per run" requires the caller to own it (the worker does, and exposes `max_workers` itself); and `paint_label_mask` drops the spec's `points` and `fallback_radius_zyx` parameters, because each `SpotFit3D` already carries its centre and fallback σ — keeping them would allow the arguments to disagree with the fits.
- **Review Focus coverage:** negative/absurd σ → Task 1 `test_a_negative_sigma_from_curve_fit_is_never_returned`, `test_an_absurd_sigma_is_rejected_as_not_converged`; >65,535 labels → Task 2 `test_more_than_65535_spots_do_not_wrap_to_background`; zero spots → Task 1 `test_no_spots_returns_no_fits` and Task 3 `test_a_timepoint_with_no_spots_writes_no_rows_and_still_masks`; all fits failing → Task 1 `test_a_flat_patch_does_not_converge_and_keeps_the_spot` and Task 2 `test_a_failed_fit_is_still_painted`; sub-voxel ellipsoid → Task 2 `test_a_sub_voxel_spot_still_occupies_its_centre`.
- **Multiprocessing coverage, added during self-review:** the first draft left the process boundary entirely untested, which is the riskiest thing here — pickling failures under Windows `spawn` surface nowhere else. Task 1 now has `TestProcessPool`: one test agreeing a real two-process pool with serial fitting, one pinning result order when an edge spot bypasses the pool. Still not covered: that the pool survives 349 timepoints inside a live QThread, and that what crosses the boundary is a patch rather than the volume (the latter is visible in `fit_spots_3d` but not asserted). A reviewer should weigh whether that residue matters.
