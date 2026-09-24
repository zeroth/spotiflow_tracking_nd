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
