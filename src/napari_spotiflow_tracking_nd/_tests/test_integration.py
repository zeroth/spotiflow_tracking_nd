"""End-to-end trace across the Phase 1 widgets.

Every other test file is task-scoped: `test_detection_widget3d.py` hand-builds
an image layer with `metadata={"source_files": files}` rather than obtaining
one from `PreProcessingWidget3D`, so the `source_files` contract between the
two widgets is never actually exercised — rename that key in one file and the
rest of the suite still passes.

This walks the real workflow with only the Spotiflow model mocked: open a
folder through the real reader -> apply calibration -> run a pre-processing
op -> hand the *resulting* layer to the detection widget -> run detection ->
check the CSV, the sidecar and the Points layer.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


@patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
def test_preprocessing_output_layer_is_consumable_by_the_detection_widget(
    mock_load_model, make_napari_viewer, tmp_path
):
    from napari_spotiflow_tracking_nd._cleanup_widget3d import PreProcessingWidget3D
    from napari_spotiflow_tracking_nd._detection_widget3d import (
        DetectionWidget3D,
        calibration_sidecar_path,
    )

    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "blurred"
    source_files = write_timepoint_folder(in_dir, n_timepoints=3, shape=(4, 16, 16))

    viewer = make_napari_viewer()

    # ── Reader: open the folder exactly as File > Open / drag-and-drop
    # would, with no widget involved ───────────────────────────────────
    from napari_spotiflow_tracking_nd._reader3d import get_reader

    reader = get_reader(str(in_dir))
    assert reader is not None
    data, attrs, layer_type = reader(str(in_dir))[0]
    raw_layer = viewer.add_image(data, **attrs)

    assert len(viewer.layers) == 1
    assert raw_layer.metadata["source_files"] == source_files

    # ── Task 5: apply calibration, then run a real pre-processing op ─────
    preproc = PreProcessingWidget3D(viewer)
    # Runs are threaded now; this test asserts immediately after each call,
    # so drive the workers synchronously through the seam. Real threading
    # is covered by test_threading.py.
    preproc._launch_worker = lambda worker: worker.run()
    preproc._image_combo.setCurrentText(raw_layer.name)
    preproc._pixel_size_xy_spin.setValue(0.104)
    preproc._voxel_size_z_spin.setValue(0.210)
    preproc._frame_interval_spin.setValue(2.5)
    preproc._apply_calibration()

    preproc._output_folder_edit.setText(str(out_dir))
    preproc._blur_sigma_spin.setValue(1.5)
    preproc._run_gaussian_blur()

    assert len(viewer.layers) == 2
    processed = viewer.layers[1]
    # The output layer must itself satisfy the contract the detection
    # widget depends on: source files on disk plus the calibration.
    assert len(processed.metadata["source_files"]) == 3
    assert all(p.exists() for p in processed.metadata["source_files"])
    assert all(p.parent == out_dir for p in processed.metadata["source_files"])
    assert processed.metadata["pixel_size_xy_um"] == 0.104
    assert processed.metadata["voxel_size_z_um"] == 0.210
    assert processed.metadata["frame_interval_s"] == 2.5

    # ── Task 6: detect on that output layer, mocking only the model ──────
    mock_model = MagicMock()

    def _predict(volume, **kwargs):
        assert volume.shape == (4, 16, 16)  # a real volume read back off disk
        details = MagicMock()
        details.prob = np.array([0.9, 0.7])
        return np.array([[1, 2, 3], [2, 4, 6]]), details

    mock_model.predict.side_effect = _predict
    mock_load_model.return_value = mock_model

    detection = DetectionWidget3D(viewer)
    detection._launch_worker = lambda worker: worker.run()  # see above
    # The combo must already list the pre-processing output, without any
    # further layer event.
    assert processed.name in [
        detection._image_combo.itemText(i)
        for i in range(detection._image_combo.count())
    ]
    detection._image_combo.setCurrentText(processed.name)
    csv_path = tmp_path / "spots.csv"
    detection._csv_path_edit.setText(str(csv_path))
    detection._run_detection()

    # ── The run's outputs ────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["frame", "z", "y", "x", "prob"]
    assert list(df["frame"]) == [0, 0, 1, 1, 2, 2]
    assert df["frame"].dtype.kind == "i"

    assert json.loads(calibration_sidecar_path(csv_path).read_text()) == {
        "pixel_size_xy_um": 0.104,
        "voxel_size_z_um": 0.210,
        "frame_interval_s": 2.5,
    }

    points_layers = [l for l in viewer.layers if l.__class__.__name__ == "Points"]
    assert len(points_layers) == 1
    points = points_layers[0]
    assert len(points.data) == 6
    assert tuple(points.scale) == (1.0, 0.210, 0.104, 0.104)
    assert points.metadata["frame_interval_s"] == 2.5
    assert points.metadata["csv_path"] == str(csv_path)
