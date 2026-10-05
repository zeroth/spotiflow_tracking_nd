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
