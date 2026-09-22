from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from napari_spotiflow_tracking_nd._tests.conftest import (
    add_layer_from_folder,
    write_timepoint_folder,
)


@pytest.fixture
def preproc_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._cleanup_widget3d import PreProcessingWidget3D

    viewer = make_napari_viewer()
    widget = PreProcessingWidget3D(viewer)
    return widget, viewer


def _add_and_select(widget, viewer, folder):
    """Add `folder` as a layer (as the reader would) and select it in the
    widget's input combo."""
    add_layer_from_folder(viewer, folder)
    widget._image_combo.setCurrentIndex(widget._image_combo.count() - 1)


class TestPreProcessingWidget3DCreation:
    def test_creates_without_error(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget is not None

    def test_has_the_expected_controls(self, preproc_widget):
        widget, _ = preproc_widget
        for attr in (
            "_pixel_size_xy_spin", "_voxel_size_z_spin", "_frame_interval_spin",
            "_apply_calibration_btn",
            "_image_combo", "_output_folder_edit",
            "_bg_sigma_spin", "_bg_use_gpu_check", "_bg_run_btn",
            "_blur_sigma_spin", "_blur_run_btn",
            "_calc_op_combo", "_calc_value_spin", "_calc_run_btn",
        ):
            assert hasattr(widget, attr), f"missing {attr}"

    def test_combo_is_populated_for_layers_that_existed_before_construction(
        self, make_napari_viewer
    ):
        # `layers.events.inserted` only fires for layers added *after* the
        # widget connects to it, so a widget opened on a viewer that
        # already holds images must populate its combo once at startup.
        from napari_spotiflow_tracking_nd._cleanup_widget3d import PreProcessingWidget3D

        viewer = make_napari_viewer()
        viewer.add_image(np.zeros((2, 4, 8, 8), dtype=np.uint16), name="already_here")

        widget = PreProcessingWidget3D(viewer)

        assert widget._image_combo.currentText() == "already_here"


class TestApplyCalibration:
    def test_sets_scale_and_metadata_on_the_selected_layer(
        self, preproc_widget, tmp_path
    ):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))
        _add_and_select(widget, viewer, tmp_path)

        widget._pixel_size_xy_spin.setValue(0.104)
        widget._voxel_size_z_spin.setValue(0.210)
        widget._frame_interval_spin.setValue(2.5)
        widget._apply_calibration()

        layer = viewer.layers[0]
        # napari's layer.scale is a numpy array — compare via tuple(), never
        # bare `==` (a multi-element array in a bare assert raises
        # "truth value ambiguous", not a clean test failure).
        assert tuple(layer.scale) == (1.0, 0.210, 0.104, 0.104)
        assert layer.metadata["pixel_size_xy_um"] == 0.104
        assert layer.metadata["voxel_size_z_um"] == 0.210
        assert layer.metadata["frame_interval_s"] == 2.5

    def test_without_calibration_values_leaves_unit_scale(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        _add_and_select(widget, viewer, tmp_path)

        widget._apply_calibration()

        layer = viewer.layers[0]
        assert tuple(layer.scale) == (1.0, 1.0, 1.0, 1.0)
        assert layer.metadata["pixel_size_xy_um"] is None
        assert layer.metadata["voxel_size_z_um"] is None
        assert layer.metadata["frame_interval_s"] is None

    def test_no_input_layer_selected_shows_an_error_instead_of_raising(self, preproc_widget):
        widget, viewer = preproc_widget

        with patch(
            "napari_spotiflow_tracking_nd._cleanup_widget3d.show_error"
        ) as mock_show_error:
            widget._apply_calibration()  # must not raise

        assert mock_show_error.called


class TestRunRemoveBackground:
    def test_writes_output_folder_and_loads_it_as_a_new_layer(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        _add_and_select(widget, viewer, tmp_path / "in")
        widget._output_folder_edit.setText(str(tmp_path / "out_bg"))
        widget._bg_sigma_spin.setValue(2.0)

        widget._run_remove_background()  # calls worker.run() synchronously (see widget code)

        assert (tmp_path / "out_bg").exists()
        assert len(list((tmp_path / "out_bg").iterdir())) == 2
        assert len(viewer.layers) == 2


class TestRunGaussianBlur:
    def test_writes_output_folder(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        _add_and_select(widget, viewer, tmp_path / "in")
        widget._output_folder_edit.setText(str(tmp_path / "out_blur"))
        widget._blur_sigma_spin.setValue(1.5)

        widget._run_gaussian_blur()

        assert len(list((tmp_path / "out_blur").iterdir())) == 2


class TestRunArithmetic:
    def test_writes_output_folder(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        _add_and_select(widget, viewer, tmp_path / "in")
        widget._output_folder_edit.setText(str(tmp_path / "out_calc"))
        widget._calc_op_combo.setCurrentText("add")
        widget._calc_value_spin.setValue(10)

        widget._run_arithmetic()

        assert len(list((tmp_path / "out_calc").iterdir())) == 2


class TestOutputFolderGuard:
    """`VolumeOpWorker3D` writes `output_dir / path.name`, so an output
    folder that *is* the input folder overwrites the source acquisition in
    place, file by file, with no confirmation and no undo."""

    @pytest.mark.parametrize(
        "run_method", ["_run_remove_background", "_run_gaussian_blur", "_run_arithmetic"]
    )
    def test_output_equal_to_input_folder_is_refused_for_every_run_button(
        self, preproc_widget, tmp_path, run_method
    ):
        widget, viewer = preproc_widget
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 16, 16))
        before = {p.name: p.read_bytes() for p in sorted(in_dir.iterdir())}

        _add_and_select(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(in_dir))
        widget._bg_sigma_spin.setValue(2.0)
        widget._calc_value_spin.setValue(10)

        with patch(
            "napari_spotiflow_tracking_nd._cleanup_widget3d.show_error"
        ) as mock_show_error:
            getattr(widget, run_method)()

        assert mock_show_error.called
        message = " ".join(str(c) for c in mock_show_error.call_args_list)
        assert "input folder" in message

        # Nothing written, nothing overwritten, no new layer.
        after = {p.name: p.read_bytes() for p in sorted(in_dir.iterdir())}
        assert after == before
        assert len(viewer.layers) == 1

    def test_guard_survives_a_non_normalised_path_spelling(
        self, preproc_widget, tmp_path
    ):
        widget, viewer = preproc_widget
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 16, 16))
        before = {p.name: p.read_bytes() for p in sorted(in_dir.iterdir())}

        _add_and_select(widget, viewer, in_dir)
        # The same folder, spelled with a trailing "/." — resolve() must
        # see through it.
        widget._output_folder_edit.setText(str(in_dir) + os.sep + ".")

        with patch(
            "napari_spotiflow_tracking_nd._cleanup_widget3d.show_error"
        ) as mock_show_error:
            widget._run_arithmetic()

        assert mock_show_error.called
        after = {p.name: p.read_bytes() for p in sorted(in_dir.iterdir())}
        assert after == before

    def test_a_different_output_folder_still_runs(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 16, 16))

        _add_and_select(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(tmp_path / "out"))
        widget._calc_value_spin.setValue(10)

        widget._run_arithmetic()

        assert len(list((tmp_path / "out").iterdir())) == 2
        assert len(viewer.layers) == 2


class TestOutputReloadUsesOnlyWhatWasWritten:
    def test_unrelated_tif_already_in_the_output_folder_is_not_picked_up(
        self, preproc_widget, tmp_path
    ):
        # Reloading the whole output folder would either fold a leftover
        # file into the new layer or raise a shape-mismatch ValueError out
        # of the worker's finished handler.
        widget, viewer = preproc_widget
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 16, 16))
        out_dir.mkdir()
        tifffile.imwrite(
            str(out_dir / "leftover_T_099_C_00.tif"),
            np.zeros((8, 32, 32), dtype=np.uint16),
        )

        _add_and_select(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(out_dir))
        widget._calc_value_spin.setValue(10)

        widget._run_arithmetic()  # must not raise

        assert len(viewer.layers) == 2
        new_layer = viewer.layers[1]
        assert new_layer.data.shape == (2, 4, 16, 16)
        assert len(new_layer.metadata["source_files"]) == 2
        assert all(
            "leftover" not in p.name for p in new_layer.metadata["source_files"]
        )


class TestRunWithoutSourceFiles:
    def test_a_layer_with_no_source_files_shows_an_error_instead_of_raising(
        self, preproc_widget
    ):
        widget, viewer = preproc_widget
        viewer.add_image(np.zeros((2, 4, 8, 8), dtype=np.uint16), name="plain")
        widget._image_combo.setCurrentText("plain")

        with patch(
            "napari_spotiflow_tracking_nd._cleanup_widget3d.show_error"
        ) as mock_show_error:
            widget._run_arithmetic()  # must not raise

        assert mock_show_error.called
        assert len(viewer.layers) == 1
