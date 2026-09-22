from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


class TestVolumeOpWorker3D:
    def test_writes_one_output_file_per_input_with_matching_names(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        files = write_timepoint_folder(in_dir, n_timepoints=3, shape=(4, 8, 8))

        progress_calls = []
        finished = {}
        worker = VolumeOpWorker3D(
            files=files, output_dir=out_dir, op=apply_calculation,
            op_kwargs={"op": "add", "value": 5},
        )
        worker.progress.connect(lambda cur, total: progress_calls.append((cur, total)))
        worker.run_finished.connect(lambda paths: finished.update(paths=paths))

        worker.run()  # call directly — no real threading in the test

        written = sorted(p.name for p in out_dir.iterdir())
        assert written == [f.name for f in files]
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]
        assert len(finished["paths"]) == 3

    def test_output_matches_the_op_applied_to_the_input(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        files = write_timepoint_folder(in_dir, n_timepoints=1, shape=(4, 8, 8))
        original = tifffile.imread(str(files[0]))

        worker = VolumeOpWorker3D(
            files=files, output_dir=out_dir, op=apply_calculation,
            op_kwargs={"op": "add", "value": 7},
        )
        worker.run()

        out = tifffile.imread(str(out_dir / files[0].name))
        expected = apply_calculation(original, "add", 7)
        assert np.array_equal(out, expected)

    def test_cancel_stops_before_the_next_file(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        files = write_timepoint_folder(in_dir, n_timepoints=5, shape=(4, 8, 8))

        worker = VolumeOpWorker3D(
            files=files, output_dir=out_dir, op=apply_calculation,
            op_kwargs={"op": "add", "value": 1},
        )
        # Cancel as soon as the first file's progress is reported.
        worker.progress.connect(lambda cur, total: worker.cancel())
        worker.run()

        assert len(list(out_dir.iterdir())) == 1

    def test_errored_signal_carries_the_exception_message(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        files = write_timepoint_folder(in_dir, n_timepoints=1, shape=(4, 8, 8))

        def _broken_op(volume, **kwargs):
            raise RuntimeError("boom")

        errors = []
        worker = VolumeOpWorker3D(files=files, output_dir=out_dir, op=_broken_op, op_kwargs={})
        worker.errored.connect(errors.append)
        worker.run()

        assert errors == ["boom"]


class TestDetectionWorker3D:
    def test_writes_a_csv_with_frame_z_y_x_prob_columns(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        in_dir = tmp_path / "in"
        files = write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 8, 8))
        csv_path = tmp_path / "spots.csv"

        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        model.predict.return_value = (np.array([[1, 2, 3]]), details)

        finished = {}
        worker = DetectionWorker3D(files=files, model=model, csv_path=csv_path)
        worker.run_finished.connect(lambda path: finished.update(path=path))
        worker.run()

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["frame", "z", "y", "x", "prob"]
        assert list(df["frame"]) == [0, 1]
        assert finished["path"] == str(csv_path)

    def test_frame_column_is_written_as_integers_not_floats(self, tmp_path):
        # Building the rows with a single np.hstack promotes the integer
        # frame index to float64 alongside the coords/probabilities, so the
        # CSV reads "0.0, 1.0, ..." instead of "0, 1, ...".
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        in_dir = tmp_path / "in"
        files = write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 8, 8))
        csv_path = tmp_path / "spots.csv"

        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        model.predict.return_value = (np.array([[1, 2, 3]]), details)

        worker = DetectionWorker3D(files=files, model=model, csv_path=csv_path)
        worker.run()

        text = csv_path.read_text().splitlines()
        assert text[1].split(",")[0] == "0"
        assert text[2].split(",")[0] == "1"
        df = pd.read_csv(csv_path)
        assert df["frame"].dtype.kind == "i"

    def test_run_finished_does_not_shadow_qthreads_own_finished_signal(self, tmp_path):
        # QThread defines a no-argument `finished` signal; overriding it
        # with a differently-shaped one breaks the moment anything calls
        # `start()` instead of `run()`.
        from qtpy.QtCore import QThread

        from napari_spotiflow_tracking_nd._workers3d import (
            DetectionWorker3D,
            VolumeOpWorker3D,
        )

        for cls in (VolumeOpWorker3D, DetectionWorker3D):
            assert "finished" not in vars(cls)
            assert "run_finished" in vars(cls)
            # …and the two signals we deliberately did not rename really
            # aren't QThread built-ins.
            assert not hasattr(QThread, "progress")
            assert not hasattr(QThread, "errored")

    def test_frame_with_no_spots_writes_no_rows_for_that_frame(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        in_dir = tmp_path / "in"
        files = write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 8, 8))
        csv_path = tmp_path / "spots.csv"

        model = MagicMock()
        details_empty = MagicMock()
        details_empty.prob = np.empty((0,))
        details_full = MagicMock()
        details_full.prob = np.array([0.9])
        model.predict.side_effect = [
            (np.empty((0, 3)), details_empty),
            (np.array([[1, 2, 3]]), details_full),
        ]

        worker = DetectionWorker3D(files=files, model=model, csv_path=csv_path)
        worker.run()

        df = pd.read_csv(csv_path)
        assert list(df["frame"]) == [1]

    def test_errored_signal_carries_the_exception_message(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        in_dir = tmp_path / "in"
        files = write_timepoint_folder(in_dir, n_timepoints=1, shape=(4, 8, 8))
        csv_path = tmp_path / "spots.csv"

        model = MagicMock()
        model.predict.side_effect = RuntimeError("model exploded")

        errors = []
        worker = DetectionWorker3D(files=files, model=model, csv_path=csv_path)
        worker.errored.connect(errors.append)
        worker.run()

        assert errors == ["model exploded"]
