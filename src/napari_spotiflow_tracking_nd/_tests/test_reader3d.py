from __future__ import annotations

import numpy as np
import pytest
import tifffile

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


class TestGetReaderAccepts:
    def test_accepts_a_folder_of_per_timepoint_tiffs(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))

        reader = get_reader(str(tmp_path))

        assert reader is not None

    def test_accepts_a_single_file_inside_the_series(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        files = write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))

        reader = get_reader(str(files[1]))  # any one file in the series

        assert reader is not None

    def test_single_file_reader_loads_the_whole_parent_folder(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        files = write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))

        reader = get_reader(str(files[1]))
        layer_data = reader(str(files[1]))

        data, attrs, layer_type = layer_data[0]
        assert data.shape == (3, 4, 8, 8)
        assert len(attrs["metadata"]["source_files"]) == 3


class TestGetReaderDeclines:
    def test_declines_a_list_of_paths(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        files = write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))

        assert get_reader([str(f) for f in files]) is None

    def test_declines_a_nonexistent_path(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        assert get_reader(str(tmp_path / "does_not_exist")) is None

    def test_declines_an_empty_folder(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        assert get_reader(str(tmp_path)) is None

    def test_declines_a_folder_with_tiffs_that_have_no_timepoint_token(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        tifffile.imwrite(str(tmp_path / "no_token_here.tif"), np.zeros((4, 8, 8), dtype=np.uint16))

        assert get_reader(str(tmp_path)) is None

    def test_declines_a_single_file_with_no_timepoint_token(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        path = tmp_path / "no_token_here.tif"
        tifffile.imwrite(str(path), np.zeros((4, 8, 8), dtype=np.uint16))

        assert get_reader(str(path)) is None


class TestReadVolumeTimeseries:
    def test_returns_a_lazy_image_layer_at_unit_scale_with_null_calibration(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        files = write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))

        reader = get_reader(str(tmp_path))
        layer_data = reader(str(tmp_path))

        assert len(layer_data) == 1
        data, attrs, layer_type = layer_data[0]
        assert layer_type == "image"
        assert data.shape == (3, 4, 8, 8)
        assert attrs["scale"] == (1.0, 1.0, 1.0, 1.0)
        assert attrs["metadata"]["source_files"] == files
        assert attrs["metadata"]["pixel_size_xy_um"] is None
        assert attrs["metadata"]["voxel_size_z_um"] is None
        assert attrs["metadata"]["frame_interval_s"] is None
        assert attrs["name"] == tmp_path.name

    def test_a_2d_tiff_folder_raises_a_clear_error(self, tmp_path):
        # `get_reader` only checks for a `_T_<n>_` token, not the volume's
        # shape, so it accepts here; the shape mismatch surfaces when napari
        # actually calls the reader function, same as any other reader
        # exception napari shows to the user.
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        tifffile.imwrite(
            str(tmp_path / "plane_T_000_C_00.tif"), np.zeros((16, 16), dtype=np.uint16)
        )

        reader = get_reader(str(tmp_path))
        assert reader is not None

        with pytest.raises(ValueError, match="3D"):
            reader(str(tmp_path))

    def test_a_corrupt_tiff_raises_a_clear_error(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        (tmp_path / "broken_T_000_C_00.tif").write_bytes(b"not a tiff at all")

        reader = get_reader(str(tmp_path))
        assert reader is not None

        with pytest.raises(Exception):  # tifffile's own TiffFileError
            reader(str(tmp_path))
