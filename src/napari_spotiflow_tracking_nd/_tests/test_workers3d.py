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


class TestFunctionWorker3D:
    def test_emits_the_callables_return_value(self):
        from napari_spotiflow_tracking_nd._workers3d import FunctionWorker3D

        results = []
        worker = FunctionWorker3D(lambda: {"spots": 3})
        worker.run_finished.connect(results.append)

        worker.run()

        assert results == [{"spots": 3}]

    def test_passes_args_and_kwargs_through(self):
        from napari_spotiflow_tracking_nd._workers3d import FunctionWorker3D

        results = []
        worker = FunctionWorker3D(lambda a, b, c=0: (a, b, c), args=(1, 2), kwargs={"c": 3})
        worker.run_finished.connect(results.append)

        worker.run()

        assert results == [(1, 2, 3)]

    def test_errored_carries_the_exception_message(self):
        from napari_spotiflow_tracking_nd._workers3d import FunctionWorker3D

        def _boom():
            raise RuntimeError("preview blew up")

        errors = []
        worker = FunctionWorker3D(_boom)
        worker.errored.connect(errors.append)

        worker.run()

        assert errors == ["preview blew up"]

    def test_cancel_suppresses_the_result(self):
        # A single blocking call can't be interrupted, so cancel means
        # "abandon the result", not "stop the work".
        from napari_spotiflow_tracking_nd._workers3d import FunctionWorker3D

        results = []
        worker = FunctionWorker3D(lambda: "too late")
        worker.run_finished.connect(results.append)
        worker.cancel()

        worker.run()

        assert results == []

    def test_cancel_also_suppresses_an_error(self):
        # An abandoned preview that happens to fail should not pop an error
        # dialog for a result nobody is waiting for.
        from napari_spotiflow_tracking_nd._workers3d import FunctionWorker3D

        def _boom():
            raise RuntimeError("nobody cares")

        errors = []
        worker = FunctionWorker3D(_boom)
        worker.errored.connect(errors.append)
        worker.cancel()

        worker.run()

        assert errors == []


class TestDetectionWorkerLazyModel:
    """`model` may be a zero-arg callable, so the slow first weight load
    happens on the worker thread instead of freezing the UI first."""

    @staticmethod
    def _one_spot_model():
        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.9])
        model.predict.return_value = (np.array([[1, 2, 3]]), details)
        return model

    def test_a_callable_model_is_resolved_inside_run(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        model = self._one_spot_model()
        calls = []

        def _factory():
            calls.append(len(calls))
            return model

        worker = DetectionWorker3D(
            files=files, model=_factory, csv_path=tmp_path / "spots.csv",
        )

        assert calls == [], "the factory must not run at construction time"

        worker.run()

        assert calls == [0], "resolved exactly once, inside run()"
        assert (tmp_path / "spots.csv").exists()

    def test_a_ready_model_still_works_unchanged(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = write_timepoint_folder(tmp_path, n_timepoints=1, shape=(4, 8, 8))
        worker = DetectionWorker3D(
            files=files, model=self._one_spot_model(), csv_path=tmp_path / "spots.csv",
        )

        worker.run()

        assert (tmp_path / "spots.csv").exists()

    def test_a_failing_model_factory_goes_to_errored(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = write_timepoint_folder(tmp_path, n_timepoints=1, shape=(4, 8, 8))

        def _factory():
            raise RuntimeError("no such checkpoint")

        errors = []
        worker = DetectionWorker3D(
            files=files, model=_factory, csv_path=tmp_path / "spots.csv",
        )
        worker.errored.connect(errors.append)

        worker.run()

        assert errors == ["no such checkpoint"]
        assert not (tmp_path / "spots.csv").exists()


class TestDetectionWorkerFitting:
    @staticmethod
    def _model(n_spots=2):
        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.9, 0.7][:n_spots])
        model.predict.return_value = (
            np.array([[4.0, 8.0, 8.0], [4.0, 20.0, 20.0]][:n_spots]), details,
        )
        return model

    @staticmethod
    def _folder(tmp_path, n_timepoints=2):
        return write_timepoint_folder(
            tmp_path / "in", n_timepoints=n_timepoints, shape=(12, 32, 32),
        )

    def test_fitting_off_leaves_the_csv_columns_unchanged(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=self._model(), csv_path=tmp_path / "spots.csv",
        )
        worker.run()

        df = pd.read_csv(tmp_path / "spots.csv")
        assert list(df.columns) == ["frame", "z", "y", "x", "prob"]

    def test_fitting_on_adds_exactly_the_named_columns(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=self._model(), csv_path=tmp_path / "spots.csv",
            fit=True, max_workers=1,
        )
        worker.run()

        df = pd.read_csv(tmp_path / "spots.csv")
        assert list(df.columns) == [
            "frame", "z", "y", "x", "prob",
            "fit_z", "fit_y", "fit_x",
            "sigma_z", "sigma_y", "sigma_x",
            "amplitude", "background", "integrated_intensity", "converged",
        ]

    def test_the_detected_coordinates_are_not_replaced_by_the_fitted_ones(self, tmp_path):
        # Phase 2 tracking reads z/y/x; turning fitting on must not shift
        # them underneath it.
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=self._model(n_spots=1), csv_path=tmp_path / "spots.csv",
            fit=True, max_workers=1,
        )
        worker.run()

        df = pd.read_csv(tmp_path / "spots.csv")
        assert df["z"].iloc[0] == 4.0
        assert df["y"].iloc[0] == 8.0
        assert df["x"].iloc[0] == 8.0

    def test_no_mask_is_written_unless_a_folder_is_given(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=self._model(), csv_path=tmp_path / "spots.csv",
            fit=True, max_workers=1,
        )
        worker.run()

        assert not list(tmp_path.glob("**/*mask*"))

    def test_a_mask_folder_gets_one_label_tiff_per_timepoint(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path, n_timepoints=3)
        mask_dir = tmp_path / "masks"
        worker = DetectionWorker3D(
            files=files, model=self._model(), csv_path=tmp_path / "spots.csv",
            fit=True, mask_output_dir=mask_dir, max_workers=1,
        )
        worker.run()

        written = sorted(p.name for p in mask_dir.iterdir())
        assert written == [f.name for f in files]
        mask = tifffile.imread(str(mask_dir / files[0].name))
        assert mask.shape == (12, 32, 32)
        assert mask.max() == 2  # two spots, labels 1 and 2

    def test_a_timepoint_with_no_spots_writes_no_rows_and_still_masks(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = self._folder(tmp_path, n_timepoints=1)
        model = MagicMock()
        empty = MagicMock()
        empty.prob = np.empty((0,))
        model.predict.return_value = (np.empty((0, 3)), empty)
        mask_dir = tmp_path / "masks"

        worker = DetectionWorker3D(
            files=files, model=model, csv_path=tmp_path / "spots.csv",
            fit=True, mask_output_dir=mask_dir, max_workers=1,
        )
        worker.run()  # must not raise

        assert not (tmp_path / "spots.csv").exists()
        assert tifffile.imread(str(mask_dir / files[0].name)).max() == 0


class TestMaskExportRequiresFitting:
    """A label mask is painted from the fits. With fitting off there are
    no fits, so the old code wrote a full-size series of all-zero volumes:
    113 GB of nothing for the reference dataset, with no error.
    """

    def test_a_mask_folder_without_fitting_is_refused(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        files = TestDetectionWorkerFitting._folder(tmp_path)
        worker = DetectionWorker3D(
            files=files, model=TestDetectionWorkerFitting._model(),
            csv_path=tmp_path / "spots.csv",
            fit=False, mask_output_dir=tmp_path / "masks",
        )
        errors = []
        worker.errored.connect(errors.append)
        worker.run()

        assert errors, "the run must not silently write empty masks"
        assert "fit" in errors[0].lower()
        assert not list((tmp_path / "masks").glob("*.tif"))


class TestMaskSeriesDtypeIsRunWide:
    """`_io3d` refuses a series whose dtype changes partway through, so a
    per-timepoint dtype choice can produce a mask folder this plugin's own
    reader cannot open.
    """

    def test_the_whole_series_keeps_one_dtype_when_a_later_frame_needs_wider(
        self, tmp_path, monkeypatch
    ):
        import napari_spotiflow_tracking_nd._fitting3d as F
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        # Exercise the real widening path at a threshold small enough to
        # reach with two spots instead of 65,536.
        monkeypatch.setattr(F, "_UINT16_MAX_LABEL", 1)

        files = TestDetectionWorkerFitting._folder(tmp_path, n_timepoints=2)
        model = MagicMock()
        details_one, details_two = MagicMock(), MagicMock()
        details_one.prob = np.array([0.9])
        details_two.prob = np.array([0.9, 0.7])
        model.predict.side_effect = [
            (np.array([[4.0, 8.0, 8.0]]), details_one),        # 1 spot  -> uint16
            (np.array([[4.0, 8.0, 8.0], [4.0, 20.0, 20.0]]), details_two),  # 2 -> uint32
        ]
        worker = DetectionWorker3D(
            files=files, model=model, csv_path=tmp_path / "spots.csv",
            fit=True, max_workers=1, mask_output_dir=tmp_path / "masks",
        )
        worker.run()

        written = sorted((tmp_path / "masks").glob("*.tif"))
        assert len(written) == 2
        dtypes = {tifffile.imread(str(p)).dtype for p in written}
        assert len(dtypes) == 1, f"ragged mask series: {dtypes}"
        assert dtypes == {np.dtype(np.uint32)}

    def test_the_mask_series_loads_back_through_the_reader(self, tmp_path, monkeypatch):
        import napari_spotiflow_tracking_nd._fitting3d as F
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        monkeypatch.setattr(F, "_UINT16_MAX_LABEL", 1)

        files = TestDetectionWorkerFitting._folder(tmp_path, n_timepoints=2)
        model = MagicMock()
        d1, d2 = MagicMock(), MagicMock()
        d1.prob, d2.prob = np.array([0.9]), np.array([0.9, 0.7])
        model.predict.side_effect = [
            (np.array([[4.0, 8.0, 8.0]]), d1),
            (np.array([[4.0, 8.0, 8.0], [4.0, 20.0, 20.0]]), d2),
        ]
        worker = DetectionWorker3D(
            files=files, model=model, csv_path=tmp_path / "spots.csv",
            fit=True, max_workers=1, mask_output_dir=tmp_path / "masks",
        )
        worker.run()

        stack, _ = load_folder_as_lazy_stack(tmp_path / "masks")  # must not raise
        assert stack.shape[0] == 2


def _write_spots_csv(path, n_frames=5):
    """Two stationary spots per frame, far apart."""
    rows = []
    for t in range(n_frames):
        rows.append({"frame": t, "z": 5.0, "y": 10.0, "x": 10.0, "prob": 0.9})
        rows.append({"frame": t, "z": 20.0, "y": 40.0, "x": 40.0, "prob": 0.8})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run_tracking_worker(worker):
    seen = {"progress": [], "finished": [], "errored": []}
    worker.progress.connect(lambda c, t: seen["progress"].append((c, t)))
    worker.run_finished.connect(seen["finished"].append)
    worker.errored.connect(seen["errored"].append)
    worker.run()
    return seen


class TestTrackingWorker3D:
    def test_writes_a_tracks_csv_and_reports_progress_per_frame(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = _write_spots_csv(tmp_path / "spots.csv")
        out = tmp_path / "spots_tracks.csv"

        seen = _run_tracking_worker(TrackingWorker3D(src, out, (2.0, 4.0, 4.0)))

        assert seen["errored"] == []
        assert seen["progress"] == [(i, 5) for i in range(1, 6)]
        assert seen["finished"] == [str(out)]
        df = pd.read_csv(out)
        assert df["particle"].nunique() == 2
        for column in ("frame", "z", "y", "x", "prob", "link_z", "link_y", "link_x",
                       "coord_source", "particle", "censored"):
            assert column in df.columns, f"missing {column}"

    def test_no_partial_file_is_left_behind(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = _write_spots_csv(tmp_path / "spots.csv")
        out = tmp_path / "t.csv"

        _run_tracking_worker(TrackingWorker3D(src, out, (2.0, 4.0, 4.0)))

        assert sorted(p.name for p in tmp_path.iterdir()) == ["spots.csv", "t.csv"]

    def test_calibration_adds_physical_columns(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = _write_spots_csv(tmp_path / "spots.csv")
        out = tmp_path / "t.csv"
        calibration = {"pixel_size_xy_um": 0.104, "voxel_size_z_um": 0.210, "frame_interval_s": 3.0}

        _run_tracking_worker(TrackingWorker3D(src, out, (2.0, 4.0, 4.0), calibration=calibration))

        df = pd.read_csv(out)
        assert {"z_um", "y_um", "x_um", "time_s"} <= set(df.columns)
        assert df["time_s"].max() == pytest.approx(12.0)

    def test_min_length_drops_short_tracks(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = tmp_path / "spots.csv"
        pd.DataFrame({
            "frame": [0, 1, 2, 1],
            "z": [5.0, 5.0, 5.0, 30.0], "y": [5.0, 5.0, 5.0, 30.0], "x": [5.0, 5.0, 5.0, 30.0],
        }).to_csv(src, index=False)
        out = tmp_path / "t.csv"

        _run_tracking_worker(TrackingWorker3D(src, out, (2.0, 2.0, 2.0), min_length=2))

        assert len(pd.read_csv(out)) == 3

    def test_cancel_writes_the_frames_linked_so_far_flagged_censored(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = _write_spots_csv(tmp_path / "spots.csv", n_frames=10)
        out = tmp_path / "t.csv"
        worker = TrackingWorker3D(src, out, (2.0, 4.0, 4.0))
        worker.progress.connect(lambda current, total: worker.cancel() if current == 3 else None)

        seen = _run_tracking_worker(worker)

        assert seen["finished"] == [str(out)]
        df = pd.read_csv(out)
        assert df["frame"].max() == 2
        # Both tracks run into frame 2, the last frame linked: their true
        # end is unknown.
        assert df["censored"].all()

    def test_cancel_before_any_frame_writes_nothing(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = _write_spots_csv(tmp_path / "spots.csv")
        out = tmp_path / "t.csv"
        worker = TrackingWorker3D(src, out, (2.0, 4.0, 4.0))
        worker.cancel()

        seen = _run_tracking_worker(worker)

        assert seen["finished"] == [""]
        assert not out.exists()

    def test_a_csv_missing_columns_is_refused_by_name(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = tmp_path / "bad.csv"
        pd.DataFrame({"frame": [0, 1], "y": [1.0, 1.0], "x": [1.0, 1.0]}).to_csv(src, index=False)

        seen = _run_tracking_worker(TrackingWorker3D(src, tmp_path / "t.csv", (2.0, 4.0, 4.0)))

        assert len(seen["errored"]) == 1
        assert "z" in seen["errored"][0]

    def test_a_single_frame_is_refused(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = _write_spots_csv(tmp_path / "spots.csv", n_frames=1)

        seen = _run_tracking_worker(TrackingWorker3D(src, tmp_path / "t.csv", (2.0, 4.0, 4.0)))

        assert "two timepoints" in seen["errored"][0]

    def test_an_error_keeps_an_earlier_output_file(self, tmp_path):
        # Review focus: re-running onto an existing tracks CSV must not
        # destroy it when the new run fails.
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        src = tmp_path / "bad.csv"
        pd.DataFrame({"frame": [0, 1], "y": [1.0, 1.0], "x": [1.0, 1.0]}).to_csv(src, index=False)
        out = tmp_path / "t.csv"
        out.write_text("previous run")

        seen = _run_tracking_worker(TrackingWorker3D(src, out, (2.0, 4.0, 4.0)))

        assert seen["errored"]
        assert out.read_text() == "previous run"
        assert not (tmp_path / "t.csv.partial").exists()

    def test_a_dense_region_reports_the_adaptive_hint(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackingWorker3D

        rng = np.random.default_rng(0)
        src = tmp_path / "dense.csv"
        pd.DataFrame({
            "frame": np.repeat([0, 1], 100),
            "z": rng.uniform(0, 5, 200), "y": rng.uniform(0, 10, 200), "x": rng.uniform(0, 10, 200),
        }).to_csv(src, index=False)

        seen = _run_tracking_worker(TrackingWorker3D(src, tmp_path / "t.csv", (8.0, 8.0, 8.0)))

        assert "Adaptive search" in seen["errored"][0]
