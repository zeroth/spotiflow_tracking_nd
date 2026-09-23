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
    # Run the worker synchronously: these tests assert immediately after
    # the call, and the real threading is covered by test_threading.py.
    widget._launch_worker = lambda worker: worker.run()
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

    def test_use_gpu_is_on_by_default(self, preproc_widget):
        # See the detection widget: `_get_device` degrades to "cpu"
        # gracefully, so on-by-default costs nothing without CUDA.
        widget, _ = preproc_widget
        assert widget._bg_use_gpu_check.isChecked()

    def test_has_the_expected_controls(self, preproc_widget):
        widget, _ = preproc_widget
        for attr in (
            "_pixel_size_xy_spin", "_voxel_size_z_spin", "_frame_interval_spin",
            "_apply_calibration_btn",
            "_image_combo", "_output_folder_edit",
            "_bg_sigma_spin", "_bg_use_gpu_check", "_bg_run_btn",
            "_blur_sigma_spin", "_blur_run_btn",
            "_calc_op_combo", "_calc_value_spin", "_calc_run_btn",
            "_cancel_btn", "_status_label", "_link_sigma_check",
            "_bg_preview_btn", "_blur_preview_btn", "_calc_preview_btn",
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


class TestPreview:
    """Quick mode: apply an op to the currently-displayed timepoint only,
    in memory, so parameters can be tuned before a whole-series run."""

    def test_previews_the_currently_displayed_timepoint(self, preproc_widget, tmp_path):
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        widget, viewer = preproc_widget
        files = write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))
        _add_and_select(widget, viewer, tmp_path)

        viewer.dims.set_current_step(0, 1)  # look at timepoint 1
        widget._calc_op_combo.setCurrentText("add")
        widget._calc_value_spin.setValue(10)
        widget._preview_arithmetic()

        preview = viewer.layers[f"{tmp_path.name} (calc preview)"]
        expected = apply_calculation(tifffile.imread(str(files[1])), "add", 10)
        assert np.array_equal(np.asarray(preview.data), expected)

    def test_preview_writes_nothing_and_needs_no_output_folder(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 8, 8))
        _add_and_select(widget, viewer, in_dir)
        before = sorted(p.name for p in tmp_path.rglob("*"))

        # No output folder set at all — preview must still work.
        assert widget._output_folder_edit.text() == ""
        widget._preview_gaussian_blur()

        assert sorted(p.name for p in tmp_path.rglob("*")) == before
        assert f"{in_dir.name} (blur preview)" in viewer.layers

    def test_repeated_previews_replace_one_layer_rather_than_stacking(
        self, preproc_widget, tmp_path
    ):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        _add_and_select(widget, viewer, tmp_path)

        widget._blur_sigma_spin.setValue(1.0)
        widget._preview_gaussian_blur()
        widget._blur_sigma_spin.setValue(3.0)  # a different parameter value
        widget._preview_gaussian_blur()

        # The source layer plus exactly one preview layer.
        assert len(viewer.layers) == 2

    def test_preview_uses_the_source_layers_spatial_scale(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        _add_and_select(widget, viewer, tmp_path)
        widget._pixel_size_xy_spin.setValue(0.104)
        widget._voxel_size_z_spin.setValue(0.210)
        widget._apply_calibration()

        widget._preview_gaussian_blur()

        preview = viewer.layers[f"{tmp_path.name} (blur preview)"]
        # A single (Z, Y, X) volume: the source's scale without its T entry.
        assert tuple(preview.scale) == (0.210, 0.104, 0.104)

    def test_a_stale_timepoint_beyond_the_series_is_clamped(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        # A longer layer drags the viewer's dims past the short one's range.
        viewer.add_image(np.zeros((10, 4, 8, 8), dtype=np.uint16), name="longer")
        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        _add_and_select(widget, viewer, tmp_path)
        viewer.dims.set_current_step(0, 9)

        widget._preview_gaussian_blur()  # must not IndexError

        assert f"{tmp_path.name} (blur preview)" in viewer.layers

    def test_preview_without_source_files_shows_an_error(self, preproc_widget):
        widget, viewer = preproc_widget
        viewer.add_image(np.zeros((2, 4, 8, 8), dtype=np.uint16), name="plain")
        widget._image_combo.setCurrentText("plain")

        with patch(
            "napari_spotiflow_tracking_nd._cleanup_widget3d.show_error"
        ) as mock_show_error:
            widget._preview_gaussian_blur()  # must not raise

        assert mock_show_error.called


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


class TestThreadedRunWiring:
    def test_run_uses_the_run_state_machine(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))
        _add_and_select(widget, viewer, tmp_path / "in")
        widget._output_folder_edit.setText(str(tmp_path / "out"))
        widget._calc_value_spin.setValue(5)

        widget._run_arithmetic()

        # Back to idle once the (synchronous) worker finished.
        assert not widget._is_running
        assert widget._bg_run_btn.isEnabled()
        assert not widget._cancel_btn.isEnabled()
        assert widget._status_label.text() == "Done."

    def test_a_cancelled_run_keeps_and_labels_the_partial_output(
        self, preproc_widget, tmp_path
    ):
        widget, viewer = preproc_widget
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        write_timepoint_folder(in_dir, n_timepoints=4, shape=(4, 16, 16))
        _add_and_select(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(out_dir))
        widget._calc_value_spin.setValue(5)

        # Cancel as soon as the first volume is reported done.
        def _launch(worker):
            worker.progress.connect(lambda cur, total: widget._request_cancel())
            worker.run()

        widget._launch_worker = _launch
        widget._run_arithmetic()

        # Nothing deleted, and the layer says how far it got.
        assert len(list(out_dir.iterdir())) == 1
        partial = viewer.layers[-1]
        assert "cancelled 1/4" in partial.name
        assert len(partial.metadata["source_files"]) == 1

    def test_preview_runs_through_a_worker(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        _add_and_select(widget, viewer, tmp_path)

        widget._preview_gaussian_blur()

        assert f"{tmp_path.name} (blur preview)" in viewer.layers
        assert not widget._is_running
        assert widget._status_label.text() == "Done."

    def test_the_blocking_note_is_gone(self, preproc_widget):
        # It apologised for a freeze that no longer happens.
        widget, _ = preproc_widget
        assert not hasattr(widget, "_blocking_note")


class TestLinkSigmaToCalibration:
    """σ is in voxels, and these voxels are not cubes: 0.104 µm laterally
    against 0.210 µm axially. A single σ therefore covers twice the
    physical distance along z as it does across x/y."""

    @staticmethod
    def _calibrated(widget, viewer, folder, xy=0.104, z=0.210):
        write_timepoint_folder(folder, n_timepoints=2, shape=(8, 16, 16))
        _add_and_select(widget, viewer, folder)
        widget._pixel_size_xy_spin.setValue(xy)
        widget._voxel_size_z_spin.setValue(z)
        widget._apply_calibration()
        return viewer.layers[0]

    def test_unlinked_sigma_stays_a_plain_scalar(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        layer = self._calibrated(widget, viewer, tmp_path)
        widget._blur_sigma_spin.setValue(4.0)

        assert widget._blur_op(layer)[1]["sigma"] == 4.0

    def test_linked_sigma_derives_the_axial_value_from_calibration(
        self, preproc_widget, tmp_path
    ):
        widget, viewer = preproc_widget
        layer = self._calibrated(widget, viewer, tmp_path)
        widget._blur_sigma_spin.setValue(4.0)
        widget._link_sigma_check.setChecked(True)

        sigma = widget._blur_op(layer)[1]["sigma"]

        # Same physical width on every axis: ~half the voxels along z.
        assert sigma == pytest.approx((4.0 * 0.104 / 0.210, 4.0, 4.0))

    def test_background_removal_links_as_well(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        layer = self._calibrated(widget, viewer, tmp_path)
        widget._bg_sigma_spin.setValue(10.0)
        widget._link_sigma_check.setChecked(True)

        sigma = widget._bg_op(layer)[1]["sigma"]

        assert sigma == pytest.approx((10.0 * 0.104 / 0.210, 10.0, 10.0))

    def test_isotropic_voxels_leave_sigma_effectively_unchanged(
        self, preproc_widget, tmp_path
    ):
        # A sanity check on the direction of the ratio: with cubic voxels
        # linking must be a no-op, not a doubling or a halving.
        widget, viewer = preproc_widget
        layer = self._calibrated(widget, viewer, tmp_path, xy=0.1, z=0.1)
        widget._blur_sigma_spin.setValue(4.0)
        widget._link_sigma_check.setChecked(True)

        assert widget._blur_op(layer)[1]["sigma"] == pytest.approx((4.0, 4.0, 4.0))

    def test_linking_without_calibration_refuses_the_run(self, preproc_widget, tmp_path):
        # Silently falling back to isotropic would make the filter do
        # something other than what the ticked box promises.
        widget, viewer = preproc_widget
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=2, shape=(8, 16, 16))
        _add_and_select(widget, viewer, in_dir)  # no calibration applied
        widget._link_sigma_check.setChecked(True)
        widget._output_folder_edit.setText(str(tmp_path / "out"))

        with patch(
            "napari_spotiflow_tracking_nd._cleanup_widget3d.show_error"
        ) as mock_show_error:
            widget._run_gaussian_blur()  # must not raise

        assert mock_show_error.called
        assert "Apply Calibration" in " ".join(
            str(c) for c in mock_show_error.call_args_list
        )
        assert not (tmp_path / "out").exists()

    def test_the_applied_sigma_is_visible_in_the_layer_name(
        self, preproc_widget, tmp_path
    ):
        widget, viewer = preproc_widget
        in_dir = tmp_path / "in"
        layer = self._calibrated(widget, viewer, in_dir)
        widget._blur_sigma_spin.setValue(4.0)
        widget._link_sigma_check.setChecked(True)
        widget._output_folder_edit.setText(str(tmp_path / "out"))

        widget._run_gaussian_blur()

        # What was actually applied, readable from the layer list.
        assert "1.98" in viewer.layers[-1].name

    def test_a_scalar_sigma_is_not_printed_with_float_noise(
        self, preproc_widget, tmp_path
    ):
        widget, viewer = preproc_widget
        layer = self._calibrated(widget, viewer, tmp_path)
        widget._blur_sigma_spin.setValue(1.5)

        assert widget._blur_op(layer)[2] == "blur s=1.5"

