from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _spots(rows):
    """A detection-CSV-shaped frame from `[frame, z, y, x]` rows."""
    return pd.DataFrame(rows, columns=["frame", "z", "y", "x"]).astype({"frame": int})


def _tracked(rows):
    """`[particle, frame, link_z, link_y, link_x]` rows, as linking returns them."""
    return pd.DataFrame(
        rows, columns=["particle", "frame", "link_z", "link_y", "link_x"]
    ).astype({"particle": int, "frame": int})


class TestMissingColumns:
    def test_a_detection_csv_is_complete(self):
        from napari_spotiflow_tracking_nd._tracking3d import missing_columns

        assert missing_columns(_spots([[0, 1, 2, 3]])) == []

    def test_names_what_is_missing_in_order(self):
        from napari_spotiflow_tracking_nd._tracking3d import missing_columns

        df = pd.DataFrame({"frame": [0], "x": [1.0]})
        assert missing_columns(df) == ["z", "y"]


class TestChooseLinkCoordinates:
    @staticmethod
    def _fitted():
        return pd.DataFrame({
            "frame": [0, 0, 0, 0],
            "z": [1.0] * 4, "y": [2.0] * 4, "x": [3.0] * 4,
            "fit_z": [1.2, 1.2, np.nan, 1.2],
            "fit_y": [2.2] * 4, "fit_x": [3.2] * 4,
            "converged": [True, False, True, np.nan],
        })

    def test_only_converged_finite_fits_are_used(self):
        from napari_spotiflow_tracking_nd._tracking3d import choose_link_coordinates

        out = choose_link_coordinates(self._fitted())

        assert list(out["coord_source"]) == ["fit", "detected", "detected", "detected"]
        assert out["link_z"].tolist() == [1.2, 1.0, 1.0, 1.0]
        assert out["link_y"].tolist() == [2.2, 2.0, 2.0, 2.0]

    def test_the_detected_and_fitted_columns_are_untouched(self):
        from napari_spotiflow_tracking_nd._tracking3d import choose_link_coordinates

        df = self._fitted()
        out = choose_link_coordinates(df)

        pd.testing.assert_frame_equal(out[df.columns], df)

    def test_use_fit_false_links_on_the_detected_centre(self):
        from napari_spotiflow_tracking_nd._tracking3d import choose_link_coordinates

        out = choose_link_coordinates(self._fitted(), use_fit=False)

        assert (out["coord_source"] == "detected").all()
        assert out["link_z"].tolist() == [1.0] * 4

    def test_a_csv_without_fit_columns_is_all_detected(self):
        from napari_spotiflow_tracking_nd._tracking3d import choose_link_coordinates

        out = choose_link_coordinates(_spots([[0, 1, 2, 3], [1, 4, 5, 6]]))

        assert (out["coord_source"] == "detected").all()
        assert out[["link_z", "link_y", "link_x"]].values.tolist() == [[1, 2, 3], [4, 5, 6]]

    def test_a_blank_converged_cell_read_from_csv_is_not_a_fit(self, tmp_path):
        # Review focus: a blank `converged` reads back as NaN in an object
        # column, and `bool(nan)` is True — it must not count as converged.
        from napari_spotiflow_tracking_nd._tracking3d import choose_link_coordinates

        pd.DataFrame({
            "frame": [0, 0], "z": [1.0, 1.0], "y": [2.0, 2.0], "x": [3.0, 3.0],
            "fit_z": [1.2, 1.2], "fit_y": [2.2, 2.2], "fit_x": [3.2, 3.2],
            "converged": [True, np.nan],
        }).to_csv(tmp_path / "spots.csv", index=False)

        out = choose_link_coordinates(pd.read_csv(tmp_path / "spots.csv"))

        assert list(out["coord_source"]) == ["fit", "detected"]


class TestFilterShortTracks:
    def test_drops_tracks_shorter_than_the_minimum(self):
        from napari_spotiflow_tracking_nd._tracking3d import filter_short_tracks

        df = _tracked([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 2, 1, 1, 1],
                       [1, 1, 9, 9, 9], [1, 2, 9, 9, 9], [2, 1, 50, 50, 50]])

        assert sorted(filter_short_tracks(df, 2)["particle"].unique()) == [0, 1]
        assert sorted(filter_short_tracks(df, 3)["particle"].unique()) == [0]

    def test_a_minimum_of_one_keeps_everything(self):
        from napari_spotiflow_tracking_nd._tracking3d import filter_short_tracks

        df = _tracked([[0, 0, 1, 1, 1], [1, 1, 9, 9, 9]])

        assert len(filter_short_tracks(df, 1)) == 2


class TestFlagCensored:
    def test_tracks_touching_either_end_are_censored(self):
        from napari_spotiflow_tracking_nd._tracking3d import flag_censored

        df = _tracked([
            [0, 0, 1, 1, 1], [0, 1, 1, 1, 1],     # starts at the first frame
            [1, 1, 5, 5, 5], [1, 2, 5, 5, 5],     # wholly inside
            [2, 2, 9, 9, 9], [2, 3, 9, 9, 9],     # runs into the last frame
        ])

        out = flag_censored(df, first_frame=0, last_frame=3)

        assert out.groupby("particle")["censored"].all().to_dict() == {0: True, 1: False, 2: True}
        # Every row of a track carries its track's flag.
        assert out.groupby("particle")["censored"].nunique().eq(1).all()

    def test_an_earlier_last_frame_censors_tracks_reaching_it(self):
        # A cancelled run treats the last frame actually linked as the end.
        from napari_spotiflow_tracking_nd._tracking3d import flag_censored

        df = _tracked([[0, 1, 5, 5, 5], [0, 2, 5, 5, 5]])

        assert flag_censored(df, first_frame=0, last_frame=2)["censored"].all()
        assert not flag_censored(df, first_frame=0, last_frame=9)["censored"].any()


class TestAddPhysicalColumns:
    def test_adds_each_column_only_for_a_known_value(self):
        from napari_spotiflow_tracking_nd._tracking3d import add_physical_columns

        df = _tracked([[0, 2, 10.0, 20.0, 30.0]])

        out = add_physical_columns(
            df, {"pixel_size_xy_um": 0.104, "voxel_size_z_um": None, "frame_interval_s": 3.0}
        )

        assert "z_um" not in out
        assert out["y_um"].iloc[0] == pytest.approx(20.0 * 0.104)
        assert out["x_um"].iloc[0] == pytest.approx(30.0 * 0.104)
        assert out["time_s"].iloc[0] == pytest.approx(6.0)

    def test_full_calibration_converts_from_the_link_coordinates(self):
        from napari_spotiflow_tracking_nd._tracking3d import add_physical_columns

        df = _tracked([[0, 0, 10.0, 20.0, 30.0]])

        out = add_physical_columns(
            df, {"pixel_size_xy_um": 0.104, "voxel_size_z_um": 0.210, "frame_interval_s": 3.0}
        )

        assert out["z_um"].iloc[0] == pytest.approx(10.0 * 0.210)

    def test_no_calibration_adds_nothing(self):
        from napari_spotiflow_tracking_nd._tracking3d import add_physical_columns

        df = _tracked([[0, 0, 1.0, 2.0, 3.0]])
        out = add_physical_columns(
            df, {"pixel_size_xy_um": None, "voxel_size_z_um": 0, "frame_interval_s": None}
        )

        assert list(out.columns) == list(df.columns)


def _link_all(df, search_range_zyx, memory=0, **kwargs):
    from napari_spotiflow_tracking_nd._tracking3d import choose_link_coordinates, iter_link

    linked = [
        frame for _, frame in iter_link(
            choose_link_coordinates(df), search_range_zyx, memory, **kwargs
        ) if len(frame)
    ]
    return pd.concat(linked, ignore_index=True)


class TestIterLink:
    def test_links_stationary_spots_into_one_track_each(self):
        df = _spots([[t, 5, 5, 5] for t in range(4)] + [[t, 30, 30, 30] for t in range(4)])

        out = _link_all(df, (2, 2, 2))

        assert out["particle"].nunique() == 2
        assert len(out) == 8

    def test_an_anisotropic_range_rejects_a_z_jump_an_isotropic_one_accepts(self):
        # 3 z-voxels per frame is ~0.63 µm at 0.210 µm/z-voxel: within an
        # isotropic 4-voxel range, but beyond the (1.98, 4, 4) range that
        # matches a 4-pixel xy range in µm.
        df = _spots([[t, 10 + 3 * t, 20, 20] for t in range(4)])

        assert _link_all(df, (4, 4, 4))["particle"].nunique() == 1
        assert _link_all(df, (1.98, 4, 4))["particle"].nunique() == 4

    def test_memory_counts_frames_not_iterations(self):
        # Review focus: frame 2 has no spots at all. trackpy counts memory
        # in frames it is fed, so skipping frame 2 would let memory=0
        # bridge the gap.
        df = _spots([[t, 5, 5, 5] for t in (0, 1, 3, 4)])

        assert _link_all(df, (2, 2, 2), memory=0)["particle"].nunique() == 2
        assert _link_all(df, (2, 2, 2), memory=1)["particle"].nunique() == 1

    def test_yields_every_frame_including_empty_ones(self):
        from napari_spotiflow_tracking_nd._tracking3d import choose_link_coordinates, iter_link

        df = choose_link_coordinates(_spots([[t, 5, 5, 5] for t in (2, 3, 5)]))

        assert [t for t, _ in iter_link(df, (2, 2, 2), 1)] == [2, 3, 4, 5]

    def test_should_stop_ends_between_frames(self):
        from napari_spotiflow_tracking_nd._tracking3d import choose_link_coordinates, iter_link

        df = choose_link_coordinates(_spots([[t, 5, 5, 5] for t in range(10)]))
        seen = []

        for t, _ in iter_link(df, (2, 2, 2), 0, should_stop=lambda: len(seen) >= 3):
            seen.append(t)

        assert seen == [0, 1, 2]

    def test_a_dense_region_raises_a_plain_message(self):
        rng = np.random.default_rng(0)
        df = _spots(np.column_stack([
            np.repeat([0, 1], 100),
            rng.uniform(0, 5, 200), rng.uniform(0, 10, 200), rng.uniform(0, 10, 200),
        ]))

        with pytest.raises(ValueError, match="Adaptive search"):
            _link_all(df, (8, 8, 8))

    def test_adaptive_search_resolves_the_dense_region(self):
        rng = np.random.default_rng(0)
        df = _spots(np.column_stack([
            np.repeat([0, 1], 100),
            rng.uniform(0, 5, 200), rng.uniform(0, 10, 200), rng.uniform(0, 10, 200),
        ]))

        assert len(_link_all(df, (8, 8, 8), adaptive=True)) == 200

    def test_float_frame_numbers_from_an_older_csv_still_link(self):
        df = _spots([[t, 5, 5, 5] for t in range(3)]).astype({"frame": float})

        assert _link_all(df, (2, 2, 2))["particle"].nunique() == 1
