from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking_nd._track_analysis3d import AnalysisParams

# The reference dataset's calibration: anisotropic voxels, 3 s per volume.
PARAMS = AnalysisParams(pixel_size_xy_um=0.104, voxel_size_z_um=0.210, frame_interval_s=3.0)


def _track(pos_um, track_id=0, frames=None, censored=False, **extra):
    """A tracks-CSV-shaped frame for one track, from `(z, y, x)` µm
    positions — converted to the voxel `link_*` columns the tracker writes."""
    pos_um = np.asarray(pos_um, dtype=float)
    frames = np.arange(len(pos_um)) if frames is None else np.asarray(frames)
    df = pd.DataFrame({
        "particle": track_id,
        "frame": frames,
        "link_z": pos_um[:, 0] / 0.210,
        "link_y": pos_um[:, 1] / 0.104,
        "link_x": pos_um[:, 2] / 0.104,
        "censored": censored,
    })
    for name, value in extra.items():
        df[name] = value
    return df


class TestMissingColumns:
    def test_a_tracks_table_is_complete(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import missing_columns

        assert missing_columns(_track([[0, 0, 0]])) == []

    def test_names_what_is_missing_in_order(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import missing_columns

        df = pd.DataFrame({"particle": [0], "frame": [0], "link_y": [0.0]})
        assert missing_columns(df) == ["link_z", "link_x", "censored"]


class TestClassifyMotility:
    @pytest.mark.parametrize("alpha, expected", [
        (0.1, "constrained"), (0.4, "diffusive"), (1.0, "diffusive"),
        (1.2, "diffusive"), (1.9, "directed"), (np.nan, "unclassified"),
        (None, "unclassified"),
    ])
    def test_thresholds(self, alpha, expected):
        from napari_spotiflow_tracking_nd._track_analysis3d import classify_motility

        assert classify_motility(alpha) == expected


class TestPositionsUm:
    def test_scales_each_axis_by_its_own_voxel_size(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import positions_um

        track = pd.DataFrame({
            "particle": 0, "frame": [0], "link_z": [1.0], "link_y": [1.0],
            "link_x": [1.0], "censored": False,
        })

        _, pos = positions_um(track, PARAMS)

        assert pos[0].tolist() == pytest.approx([0.210, 0.104, 0.104])

    def test_gap_frames_are_nan_rows(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import positions_um

        frames, pos = positions_um(_track(np.zeros((3, 3)), frames=[4, 5, 7]), PARAMS)

        assert frames.tolist() == [4, 5, 6, 7]
        assert np.isnan(pos[2]).all()
        assert np.isfinite(pos[[0, 1, 3]]).all()


class TestMsdTable:
    def test_a_constant_step_gives_lag_squared_growth(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import msd_table

        pos = np.column_stack([np.zeros(5), np.zeros(5), np.arange(5.0)])

        table = msd_table(pos, max_lag=25)

        assert table.index.tolist() == [1, 2, 3, 4]
        assert table["msd"].tolist() == pytest.approx([1.0, 4.0, 9.0, 16.0])
        assert table["n_pairs"].tolist() == [4, 3, 2, 1]
        assert np.isnan(table["sem"].iloc[-1])  # one pair: no spread

    def test_gaps_drop_out_and_empty_lags_are_omitted(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import msd_table

        # Present at frames 0, 1, 3 only (frame 2 a gap).
        pos = np.array([[0, 0, 0], [0, 0, 1.0], [np.nan] * 3, [0, 0, 3.0]])

        table = msd_table(pos, max_lag=25)

        assert table.index.tolist() == [1, 2, 3]
        assert table["n_pairs"].tolist() == [1, 1, 1]
        assert table["msd"].tolist() == pytest.approx([1.0, 4.0, 9.0])

    def test_max_lag_caps_the_lags(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import msd_table

        pos = np.column_stack([np.zeros(10), np.zeros(10), np.arange(10.0)])

        assert msd_table(pos, max_lag=3).index.tolist() == [1, 2, 3]


class TestFitMsd:
    def test_recovers_d_and_alpha_from_an_exact_curve(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import fit_msd

        t = np.arange(1, 11) * 3.0
        d, alpha = fit_msd(t, 6 * 0.02 * t ** 0.8, k=6.0)

        assert d == pytest.approx(0.02, rel=1e-3)
        assert alpha == pytest.approx(0.8, rel=1e-3)

    def test_a_hopeless_curve_returns_nan_instead_of_raising(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import fit_msd

        d, alpha = fit_msd([3.0, 6.0], [np.nan, np.inf], k=6.0)

        assert np.isnan(d) and np.isnan(alpha)


class TestAnalyseTrackMotion:
    def test_a_brownian_walk_recovers_d_and_alpha_in_3d_and_xy(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        rng = np.random.default_rng(1)
        true_d, dt = 0.01, 3.0
        d3, a3, dxy, axy = [], [], [], []
        for track_id in range(150):
            steps = rng.normal(0, np.sqrt(2 * true_d * dt), size=(100, 3))
            row, _, _ = analyse_track(_track(np.cumsum(steps, axis=0), track_id), PARAMS)
            d3.append(row["D_3d"])
            a3.append(row["alpha_3d"])
            dxy.append(row["D_xy"])
            axy.append(row["alpha_xy"])

        assert np.median(a3) == pytest.approx(1.0, abs=0.15)
        assert np.median(d3) == pytest.approx(true_d, rel=0.25)
        assert np.median(axy) == pytest.approx(1.0, abs=0.15)
        assert np.median(dxy) == pytest.approx(true_d, rel=0.25)

    def test_constant_velocity_is_directed(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        rng = np.random.default_rng(2)
        t = np.arange(40)
        pos = np.column_stack([0 * t, 0 * t, 0.2 * t]) + rng.normal(0, 0.01, (40, 3))

        row, _, _ = analyse_track(_track(pos), PARAMS)

        assert row["alpha_3d"] > 1.2
        assert row["motility_class"] == "directed"

    def test_jitter_in_place_is_constrained(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        rng = np.random.default_rng(3)

        row, _, _ = analyse_track(_track(rng.normal(0, 0.02, (40, 3))), PARAMS)

        assert row["motility_class"] == "constrained"

    def test_motility_from_xy_uses_alpha_xy(self):
        from dataclasses import replace

        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        # Directed along z only: xy is stationary jitter.
        rng = np.random.default_rng(4)
        t = np.arange(40)
        pos = np.column_stack([0.2 * t, 0 * t, 0 * t]) + rng.normal(0, 0.01, (40, 3))

        row_3d, _, _ = analyse_track(_track(pos), PARAMS)
        row_xy, _, _ = analyse_track(_track(pos), replace(PARAMS, motility_from="xy"))

        assert row_3d["motility_class"] == "directed"
        assert row_xy["motility_class"] == "constrained"

    def test_the_motility_code_follows_the_class(self):
        # An integer twin of `motility_class` that napari can colour by.
        from napari_spotiflow_tracking_nd._track_analysis3d import (
            MOTILITY_CODES,
            PROPERTY_COLUMNS,
            analyse_track,
        )

        assert MOTILITY_CODES == {"unclassified": 0, "constrained": 1, "diffusive": 2, "directed": 3}
        assert PROPERTY_COLUMNS.index("motility_code") == PROPERTY_COLUMNS.index("motility_class") + 1
        rng = np.random.default_rng(3)
        constrained, _, _ = analyse_track(_track(rng.normal(0, 0.02, (40, 3))), PARAMS)
        short, _, _ = analyse_track(_track(rng.normal(0, 0.1, (5, 3))), PARAMS)

        assert constrained["motility_code"] == 1
        assert short["motility_code"] == 0
        assert isinstance(short["motility_code"], int)

    def test_a_short_track_is_not_fitted(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, msd_rows, _ = analyse_track(_track(np.random.default_rng(5).normal(0, 0.1, (9, 3))), PARAMS)

        assert np.isnan(row["D_3d"]) and np.isnan(row["alpha_xy"])
        assert row["msd_lags_fitted"] == 0
        assert row["motility_class"] == "unclassified"
        # The curve is still reported, unfitted.
        assert msd_rows and all(np.isnan(r["msd_fit"]) for r in msd_rows)

    def test_a_fitted_track_reports_its_curve_and_fit(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import MSD_COLUMNS, analyse_track

        rng = np.random.default_rng(6)
        row, msd_rows, _ = analyse_track(_track(np.cumsum(rng.normal(0, 0.2, (30, 3)), axis=0)), PARAMS)

        assert row["msd_lags_fitted"] == 25
        assert {r["dims"] for r in msd_rows} == {"3d", "xy"}
        assert all(set(r) == set(MSD_COLUMNS) for r in msd_rows)
        first = next(r for r in msd_rows if r["dims"] == "3d" and r["lag_frames"] == 1)
        assert first["lag_s"] == 3.0
        assert np.isfinite(first["msd_fit"])


class TestAnalyseTrackGeometryAndLifetime:
    def test_anisotropic_steps_and_gap_bookkeeping(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        # One z-voxel step (0.210 µm), then a gap, then one x-pixel step (0.104 µm).
        row, _, _ = analyse_track(
            _track([[0, 0, 0], [0.210, 0, 0], [0.210, 0, 0.104]], frames=[0, 1, 3]), PARAMS,
        )

        assert row["path_length_um"] == pytest.approx(0.314)
        assert row["net_displacement_um"] == pytest.approx(np.hypot(0.210, 0.104))
        assert row["n_spots"] == 3
        assert row["n_gap_frames"] == 1
        assert row["lifetime_frames"] == 4
        assert row["lifetime_s"] == 12.0
        assert row["mean_speed_um_s"] == pytest.approx(0.314 / 9.0)

    def test_censored_is_carried_through(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, _, _ = analyse_track(_track(np.zeros((3, 3)), censored=True), PARAMS)

        assert row["censored"] is True

    def test_a_single_spot_track_is_all_nan_motion_and_no_crash(self):
        # Review focus: tracking's min length defaults to 1.
        from napari_spotiflow_tracking_nd._track_analysis3d import PROPERTY_COLUMNS, analyse_track

        row, msd_rows, step_rows = analyse_track(_track([[1.0, 2.0, 3.0]], frames=[7]), PARAMS)

        assert list(row) == PROPERTY_COLUMNS
        assert row["lifetime_frames"] == 1 and row["lifetime_s"] == 3.0
        assert row["path_length_um"] == 0.0 and row["net_displacement_um"] == 0.0
        assert np.isnan(row["mean_speed_um_s"])
        assert np.isnan(row["D_3d"]) and row["motility_class"] == "unclassified"
        assert msd_rows == [] and step_rows == []


class TestAnalyseTrackIntensity:
    def test_a_two_level_trace_with_failed_fits_has_one_step(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import STEP_COLUMNS, analyse_track

        frames = np.arange(30)
        values = np.where(frames < 15, 1000.0, 400.0)
        converged = np.ones(30, dtype=bool)
        converged[[3, 20]] = False

        row, _, steps = analyse_track(
            _track(np.zeros((30, 3)), integrated_intensity=values, converged=converged), PARAMS,
        )

        assert row["n_intensity_points"] == 28
        assert row["mean_intensity"] == pytest.approx((14 * 1000 + 14 * 400) / 28)
        assert row["max_intensity"] == 1000.0
        assert row["n_steps"] == 1
        assert set(steps[0]) == set(STEP_COLUMNS)
        # The ported 2D detector lands on the last high point, one early.
        assert steps[0]["frame"] in (14, 15)
        assert steps[0]["step_height"] == pytest.approx(-600, rel=0.1)
        assert row["mean_step_height"] == steps[0]["step_height"]

    def test_a_trace_too_short_for_the_window_is_not_assessed(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, _, steps = analyse_track(
            _track(np.zeros((6, 3)), integrated_intensity=1000.0, converged=True), PARAMS,
        )

        # 6 points < 2 × 3 + 1: NaN, not zero steps.
        assert np.isnan(row["n_steps"]) and np.isnan(row["mean_step_height"])
        assert steps == []

    def test_a_flat_trace_has_zero_steps(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, _, steps = analyse_track(
            _track(np.zeros((20, 3)), integrated_intensity=1000.0, converged=True), PARAMS,
        )

        assert row["n_steps"] == 0 and np.isnan(row["mean_step_height"])
        assert steps == []

    def test_no_fit_columns_means_no_intensity(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, _, _ = analyse_track(_track(np.zeros((20, 3))), PARAMS)

        assert row["n_intensity_points"] == 0
        assert np.isnan(row["mean_intensity"]) and np.isnan(row["max_intensity"])
        assert np.isnan(row["n_steps"])

    def test_a_mostly_failed_fit_trace_gives_nan_steps(self):
        # Review focus: on decon_10 ~75 % of fits fail.
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        converged = np.zeros(20, dtype=bool)
        converged[::4] = True  # 5 of 20
        values = np.where(converged, 1000.0, np.nan)

        row, _, steps = analyse_track(
            _track(np.zeros((20, 3)), integrated_intensity=values, converged=converged), PARAMS,
        )

        assert row["n_intensity_points"] == 5
        assert np.isnan(row["n_steps"])
        assert steps == []


class TestIsResultsFolder:
    def _folder(self, tmp_path, *names):
        folder = tmp_path / "res"
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_text("x")
        return folder

    def test_a_full_or_partial_set_of_result_files_with_the_record(self, tmp_path):
        from napari_spotiflow_tracking_nd._track_analysis3d import RESULT_FILES, is_results_folder

        assert is_results_folder(self._folder(tmp_path, *RESULT_FILES))
        assert is_results_folder(self._folder(tmp_path / "a", "analysis.json"))

    @pytest.mark.parametrize("names", [
        (),                                          # empty
        ("track_properties.csv", "steps.csv"),       # no analysis.json
        ("analysis.json", "notes.txt"),              # something else beside it
    ])
    def test_anything_else_is_not(self, tmp_path, names):
        from napari_spotiflow_tracking_nd._track_analysis3d import is_results_folder

        assert not is_results_folder(self._folder(tmp_path, *names))

    def test_a_subfolder_a_file_or_nothing_is_not(self, tmp_path):
        from napari_spotiflow_tracking_nd._track_analysis3d import is_results_folder

        folder = self._folder(tmp_path, "analysis.json")
        (folder / "raw").mkdir()
        (tmp_path / "f.txt").write_text("x")

        assert not is_results_folder(folder)
        assert not is_results_folder(tmp_path / "f.txt")
        assert not is_results_folder(tmp_path / "missing")


class TestLifetimeDistribution:
    def test_counts_uncensored_tracks_only(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import (
            LIFETIME_COLUMNS,
            lifetime_distribution,
        )

        props = pd.DataFrame({
            "lifetime_frames": [3, 3, 5, 9],
            "censored": [False, False, False, True],
        })

        dist = lifetime_distribution(props, 3.0)

        assert list(dist.columns) == LIFETIME_COLUMNS
        assert dist["lifetime_frames"].tolist() == [3, 5]
        assert dist["lifetime_s"].tolist() == [9.0, 15.0]
        assert dist["count"].tolist() == [2, 1]
        assert dist["fraction"].tolist() == pytest.approx([2 / 3, 1 / 3])

    def test_all_censored_gives_an_empty_table_with_columns(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import (
            LIFETIME_COLUMNS,
            lifetime_distribution,
        )

        dist = lifetime_distribution(
            pd.DataFrame({"lifetime_frames": [4], "censored": [True]}), 3.0,
        )

        assert dist.empty and list(dist.columns) == LIFETIME_COLUMNS
