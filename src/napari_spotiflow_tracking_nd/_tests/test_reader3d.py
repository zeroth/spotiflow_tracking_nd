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


def _write_hyperstack(path, shape=(3, 4, 8, 8), calibrated=True):
    """A single (T, Z, Y, X) ImageJ hyperstack, like a SlideBook/Fiji
    export, with values that encode their own timepoint so a mis-ordered
    split is caught."""
    data = np.zeros(shape, dtype=np.uint16)
    for t in range(shape[0]):
        data[t] = np.arange(np.prod(shape[1:]), dtype=np.uint16).reshape(shape[1:]) + 100 * t
    kwargs = {}
    metadata = {"axes": "TZYX"}
    if calibrated:
        kwargs["resolution"] = (1 / 0.104, 1 / 0.104)
        metadata.update(unit="micron", spacing=0.21, finterval=3.0)
    tifffile.imwrite(str(path), data, imagej=True, metadata=metadata, **kwargs)
    return data


class TestHyperstack:
    def test_accepts_a_single_tzyx_hyperstack(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        path = tmp_path / "Capture 2 - Green.tif"  # no `_T_<n>_` token
        _write_hyperstack(path)

        assert get_reader(str(path)) is not None

    def test_declines_a_folder_holding_only_a_hyperstack(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        _write_hyperstack(tmp_path / "Capture 2 - Green.tif")

        assert get_reader(str(tmp_path)) is None

    def test_splits_into_per_timepoint_files_beside_the_source(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        path = tmp_path / "Capture 2 - Green.tif"
        data = _write_hyperstack(path)

        (stack, attrs, layer_type), = get_reader(str(path))(str(path))

        files = attrs["metadata"]["source_files"]
        assert layer_type == "image"
        assert stack.shape == data.shape
        assert len(files) == data.shape[0]
        assert all(f.parent == tmp_path / "Capture 2 - Green_timepoints" for f in files)
        for t, f in enumerate(files):
            np.testing.assert_array_equal(tifffile.imread(str(f)), data[t])
        np.testing.assert_array_equal(np.asarray(stack), data)
        assert attrs["name"] == "Capture 2 - Green"

    def test_reads_calibration_from_the_imagej_metadata(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        path = tmp_path / "stack.tif"
        _write_hyperstack(path)

        (_, attrs, _), = get_reader(str(path))(str(path))

        meta = attrs["metadata"]
        assert meta["pixel_size_xy_um"] == pytest.approx(0.104, rel=1e-4)
        assert meta["voxel_size_z_um"] == pytest.approx(0.21)
        assert meta["frame_interval_s"] == pytest.approx(3.0)
        assert attrs["scale"] == pytest.approx((1.0, 0.21, 0.104, 0.104), rel=1e-4)

    def test_uncalibrated_hyperstack_opens_at_unit_scale(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        path = tmp_path / "stack.tif"
        _write_hyperstack(path, calibrated=False)

        (_, attrs, _), = get_reader(str(path))(str(path))

        assert attrs["scale"] == (1.0, 1.0, 1.0, 1.0)
        assert attrs["metadata"]["pixel_size_xy_um"] is None
        assert attrs["metadata"]["voxel_size_z_um"] is None

    def test_reopening_reuses_the_split_instead_of_rewriting(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        path = tmp_path / "stack.tif"
        _write_hyperstack(path)
        (_, attrs, _), = get_reader(str(path))(str(path))
        mtimes = [f.stat().st_mtime_ns for f in attrs["metadata"]["source_files"]]

        (_, attrs, _), = get_reader(str(path))(str(path))

        assert [f.stat().st_mtime_ns for f in attrs["metadata"]["source_files"]] == mtimes

    def test_a_stale_split_from_a_different_stack_is_rewritten(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        path = tmp_path / "stack.tif"
        _write_hyperstack(path, shape=(2, 4, 8, 8))
        get_reader(str(path))(str(path))
        data = _write_hyperstack(path, shape=(3, 5, 8, 8))

        (stack, attrs, _), = get_reader(str(path))(str(path))

        assert stack.shape == data.shape
        np.testing.assert_array_equal(np.asarray(stack), data)

    def test_a_multichannel_hyperstack_is_declined(self, tmp_path):
        from napari_spotiflow_tracking_nd._reader3d import get_reader

        path = tmp_path / "two_channel.tif"
        tifffile.imwrite(
            str(path), np.zeros((2, 4, 2, 8, 8), dtype=np.uint16),
            imagej=True, metadata={"axes": "TZCYX"},
        )

        assert get_reader(str(path)) is None
