from __future__ import annotations

import numpy as np
import pytest

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


@pytest.fixture
def preproc_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._cleanup_widget3d import PreProcessingWidget3D

    viewer = make_napari_viewer()
    widget = PreProcessingWidget3D(viewer)
    return widget, viewer


class TestPreProcessingWidget3DCreation:
    def test_creates_without_error(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget is not None

    def test_has_the_expected_controls(self, preproc_widget):
        widget, _ = preproc_widget
        for attr in (
            "_input_folder_edit", "_load_btn",
            "_pixel_size_xy_spin", "_voxel_size_z_spin", "_frame_interval_spin",
            "_image_combo", "_output_folder_edit",
            "_bg_sigma_spin", "_bg_run_btn",
            "_blur_sigma_spin", "_blur_run_btn",
            "_calc_op_combo", "_calc_value_spin", "_calc_run_btn",
        ):
            assert hasattr(widget, attr), f"missing {attr}"


class TestLoadFolder:
    def test_load_folder_adds_a_scaled_image_layer(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))

        widget._input_folder_edit.setText(str(tmp_path))
        widget._pixel_size_xy_spin.setValue(0.104)
        widget._voxel_size_z_spin.setValue(0.210)
        widget._load_folder()

        assert len(viewer.layers) == 1
        layer = viewer.layers[0]
        assert layer.data.shape == (3, 4, 8, 8)
        # napari's layer.scale is a numpy array — compare via tuple(), never
        # bare `==` (a multi-element array in a bare assert raises
        # "truth value ambiguous", not a clean test failure).
        assert tuple(layer.scale) == (1.0, 0.210, 0.104, 0.104)
        assert len(layer.metadata["source_files"]) == 3

    def test_load_folder_without_calibration_uses_unit_scale(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))

        widget._input_folder_edit.setText(str(tmp_path))
        widget._load_folder()

        assert tuple(viewer.layers[0].scale) == (1.0, 1.0, 1.0, 1.0)


class TestRunRemoveBackground:
    def test_writes_output_folder_and_loads_it_as_a_new_layer(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        widget._input_folder_edit.setText(str(tmp_path / "in"))
        widget._load_folder()
        widget._image_combo.setCurrentIndex(0)
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

        widget._input_folder_edit.setText(str(tmp_path / "in"))
        widget._load_folder()
        widget._image_combo.setCurrentIndex(0)
        widget._output_folder_edit.setText(str(tmp_path / "out_blur"))
        widget._blur_sigma_spin.setValue(1.5)

        widget._run_gaussian_blur()

        assert len(list((tmp_path / "out_blur").iterdir())) == 2


class TestRunArithmetic:
    def test_writes_output_folder(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        widget._input_folder_edit.setText(str(tmp_path / "in"))
        widget._load_folder()
        widget._image_combo.setCurrentIndex(0)
        widget._output_folder_edit.setText(str(tmp_path / "out_calc"))
        widget._calc_op_combo.setCurrentText("add")
        widget._calc_value_spin.setValue(10)

        widget._run_arithmetic()

        assert len(list((tmp_path / "out_calc").iterdir())) == 2
