from __future__ import annotations

import numpy as np
import pytest
import tifffile

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


class TestListTimepointFiles:
    def test_orders_numerically_not_lexicographically(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import list_timepoint_files

        # Write T_9 and T_10 so a lexicographic sort ("T_10" < "T_9") would
        # misorder them; the real dataset goes past 99 timepoints too.
        for t in (10, 2, 9):
            tifffile.imwrite(str(tmp_path / f"stack_T_{t:03d}_C_00.tif"),
                              np.zeros((2, 4, 4), dtype=np.uint16))

        files = list_timepoint_files(tmp_path)

        assert [f.name for f in files] == [
            "stack_T_002_C_00.tif", "stack_T_009_C_00.tif", "stack_T_010_C_00.tif",
        ]

    def test_no_matching_files_raises(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import list_timepoint_files

        with pytest.raises(FileNotFoundError):
            list_timepoint_files(tmp_path)

    def test_file_without_timepoint_token_raises(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import list_timepoint_files

        tifffile.imwrite(str(tmp_path / "no_token_here.tif"),
                          np.zeros((2, 4, 4), dtype=np.uint16))

        with pytest.raises(ValueError, match="no_token_here.tif"):
            list_timepoint_files(tmp_path)


class TestProbeVolumeShape:
    def test_returns_shape_and_dtype_without_reading_pixels(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import probe_volume_shape

        path = tmp_path / "stack_T_000_C_00.tif"
        tifffile.imwrite(str(path), np.zeros((5, 10, 12), dtype=np.uint16))

        shape, dtype = probe_volume_shape(path)

        assert shape == (5, 10, 12)
        assert dtype == np.uint16

    def test_a_2d_tiff_raises_instead_of_silently_becoming_a_timepoint(self, tmp_path):
        # A folder of 2D planes would otherwise build a (T, Y, X) stack that
        # then gets a 4-element `scale` handed to `add_image` on 3-dim data —
        # an opaque napari error instead of a clear message here.
        from napari_spotiflow_tracking_nd._io3d import probe_volume_shape

        path = tmp_path / "plane_T_000_C_00.tif"
        tifffile.imwrite(str(path), np.zeros((10, 12), dtype=np.uint16))

        with pytest.raises(ValueError, match="3D"):
            probe_volume_shape(path)


class TestBuildLazyStackFromFiles:
    def test_uses_only_the_given_files_not_the_whole_folder(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import build_lazy_stack_from_files

        paths = write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        # An unrelated volume of a *different* shape in the same folder: a
        # folder glob would pick it up and raise; an explicit list must not.
        tifffile.imwrite(str(tmp_path / "other_T_099_C_00.tif"),
                         np.zeros((8, 16, 16), dtype=np.uint16))

        stack, files = build_lazy_stack_from_files(paths)

        assert stack.shape == (2, 4, 8, 8)
        assert files == paths

    def test_accepts_string_paths_and_returns_path_objects(self, tmp_path):
        from pathlib import Path

        from napari_spotiflow_tracking_nd._io3d import build_lazy_stack_from_files

        paths = write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))

        stack, files = build_lazy_stack_from_files([str(p) for p in paths])

        assert stack.shape == (2, 4, 8, 8)
        assert all(isinstance(f, Path) for f in files)

    def test_empty_file_list_raises(self):
        from napari_spotiflow_tracking_nd._io3d import build_lazy_stack_from_files

        with pytest.raises(ValueError):
            build_lazy_stack_from_files([])


class TestLoadFolderAsLazyStack:
    def test_builds_a_lazy_t_z_y_x_stack(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

        write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))

        stack, files = load_folder_as_lazy_stack(tmp_path)

        assert stack.shape == (3, 4, 8, 8)
        assert stack.dtype == np.uint16
        assert len(files) == 3

    def test_stack_is_lazy_until_computed(self, tmp_path):
        import dask.array as da
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))

        stack, _ = load_folder_as_lazy_stack(tmp_path)

        assert isinstance(stack, da.Array)

    def test_values_match_the_source_files(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

        paths = write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        expected_t0 = tifffile.imread(str(paths[0]))

        stack, files = load_folder_as_lazy_stack(tmp_path)

        assert np.array_equal(np.asarray(stack[0]), expected_t0)
        assert files[0] == paths[0]

    def test_mismatched_volume_raises_with_offending_filename(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

        write_timepoint_folder(tmp_path, n_timepoints=1, shape=(4, 8, 8))
        tifffile.imwrite(str(tmp_path / "stack_T_001_C_00.tif"),
                          np.zeros((4, 8, 9), dtype=np.uint16))  # X differs

        with pytest.raises(ValueError, match="stack_T_001_C_00.tif"):
            load_folder_as_lazy_stack(tmp_path)


class TestComputeLayerScale:
    def test_full_calibration_gives_z_y_x_scale(self):
        from napari_spotiflow_tracking_nd._io3d import compute_layer_scale

        assert compute_layer_scale(pixel_size_xy_um=0.104, voxel_size_z_um=0.210) == (
            1.0, 0.210, 0.104, 0.104,
        )

    def test_missing_calibration_gives_unit_scale(self):
        from napari_spotiflow_tracking_nd._io3d import compute_layer_scale

        assert compute_layer_scale(None, None) == (1.0, 1.0, 1.0, 1.0)
        assert compute_layer_scale(0.104, None) == (1.0, 1.0, 1.0, 1.0)
        assert compute_layer_scale(None, 0.210) == (1.0, 1.0, 1.0, 1.0)


class TestValidateOutputDir:
    """Shared by both widgets: every writer in this plugin writes
    `output_dir / input_file.name`, so an output folder that is the input
    folder overwrites the source acquisition in place.
    """

    def test_an_empty_folder_is_refused(self):
        from napari_spotiflow_tracking_nd._io3d import validate_output_dir

        out, error = validate_output_dir("", None)

        assert out is None
        assert error

    def test_the_input_folder_is_refused(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import validate_output_dir

        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2)

        out, error = validate_output_dir(str(tmp_path / "in"), files)

        assert out is None
        assert "input folder" in error

    def test_a_different_folder_is_accepted(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import validate_output_dir

        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2)

        out, error = validate_output_dir(str(tmp_path / "out"), files)

        assert error is None
        assert out == (tmp_path / "out").resolve()

    def test_the_input_folder_spelled_differently_is_still_refused(self, tmp_path):
        # `Browse…` and a typed path can disagree on separators and on
        # `.`; resolving both is what makes the guard hold.
        from napari_spotiflow_tracking_nd._io3d import validate_output_dir

        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2)

        out, error = validate_output_dir(str(tmp_path / "in" / "."), files)

        assert out is None
        assert "input folder" in error


class TestIsotropicZyx:
    def test_derives_the_axial_value_from_the_calibration(self):
        from napari_spotiflow_tracking_nd._io3d import isotropic_zyx

        # Same physical distance on every axis: a 0.210 µm z-step is about
        # twice a 0.104 µm pixel, so z gets about half the voxels.
        assert isotropic_zyx(4.0, 0.104, 0.210) == pytest.approx((1.981, 4.0, 4.0), abs=1e-3)

    def test_cubic_voxels_leave_the_value_unchanged(self):
        from napari_spotiflow_tracking_nd._io3d import isotropic_zyx

        assert isotropic_zyx(3.0, 0.1, 0.1) == pytest.approx((3.0, 3.0, 3.0))

    def test_returns_plain_floats(self):
        from napari_spotiflow_tracking_nd._io3d import isotropic_zyx

        assert all(type(v) is float for v in isotropic_zyx(4, 0.104, 0.210))
