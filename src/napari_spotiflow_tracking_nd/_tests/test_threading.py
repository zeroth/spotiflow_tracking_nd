"""The tests that actually prove the threading works.

Every other widget test overrides `_launch_worker` to run the worker
synchronously, which exercises the wiring but none of the concurrency.
These use the real `.start()` and wait on signals, so they are what stands
between this feature and a UI that freezes, crashes on close, or reports a
run as finished while it is still going.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

from napari_spotiflow_tracking_nd._tests.conftest import (
    add_layer_from_folder,
    write_timepoint_folder,
)


@pytest.fixture
def preproc(make_napari_viewer):
    from napari_spotiflow_tracking_nd._cleanup_widget3d import PreProcessingWidget3D

    viewer = make_napari_viewer()
    return PreProcessingWidget3D(viewer), viewer


def _select_folder(widget, viewer, folder):
    add_layer_from_folder(viewer, folder)
    widget._image_combo.setCurrentIndex(widget._image_combo.count() - 1)


class TestRealThreading:
    def test_the_work_happens_off_the_main_thread(self, preproc, tmp_path, qtbot):
        widget, viewer = preproc
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=3, shape=(4, 16, 16))
        _select_folder(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(tmp_path / "out"))
        widget._calc_value_spin.setValue(5)

        main_thread = threading.get_ident()
        worker_threads = []

        def _spy(volume, **kwargs):
            worker_threads.append(threading.get_ident())
            return apply_calculation(volume, **kwargs)

        with qtbot.waitSignal(widget._progress_bar.valueChanged, timeout=10000):
            widget._run_op(_spy, {"op": "add", "value": 5}, "calc")

        qtbot.waitUntil(lambda: not widget._is_running, timeout=10000)

        assert worker_threads, "the op never ran"
        assert all(t != main_thread for t in worker_threads), (
            "the op ran on the main thread — the UI would freeze"
        )

    def test_the_gui_is_updated_from_the_main_thread(self, preproc, tmp_path, qtbot):
        # Signals are delivered on the thread that connected them, so the
        # finished handler — which calls viewer.add_image — must land on
        # main. Touching Qt from a worker thread is undefined behaviour.
        widget, viewer = preproc
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 16, 16))
        _select_folder(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(tmp_path / "out"))
        widget._calc_value_spin.setValue(5)

        main_thread = threading.get_ident()
        handler_threads = []
        # napari's Viewer is a pydantic evented model and refuses attribute
        # assignment, so observe the layer event instead of patching
        # `add_image`: it fires on whichever thread did the adding.
        viewer.layers.events.inserted.connect(
            lambda _e: handler_threads.append(threading.get_ident())
        )

        widget._run_arithmetic()
        qtbot.waitUntil(lambda: not widget._is_running, timeout=10000)

        assert handler_threads == [main_thread]

    def test_buttons_are_disabled_while_a_real_run_is_in_flight(
        self, preproc, tmp_path, qtbot
    ):
        widget, viewer = preproc
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=6, shape=(4, 16, 16))
        _select_folder(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(tmp_path / "out"))
        widget._calc_value_spin.setValue(5)

        states = []
        widget._progress_bar.valueChanged.connect(
            lambda _v: states.append(
                (widget._calc_run_btn.isEnabled(), widget._cancel_btn.isEnabled())
            )
        )

        widget._run_arithmetic()
        qtbot.waitUntil(lambda: not widget._is_running, timeout=15000)

        assert states, "no progress was reported"
        assert all(state == (False, True) for state in states), (
            "run button should be disabled and cancel enabled mid-run"
        )
        # ...and restored afterwards.
        assert widget._calc_run_btn.isEnabled()
        assert not widget._cancel_btn.isEnabled()

    def test_cancelling_a_real_run_stops_it_and_keeps_partial_output(
        self, preproc, tmp_path, qtbot
    ):
        widget, viewer = preproc
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        write_timepoint_folder(in_dir, n_timepoints=8, shape=(4, 16, 16))
        _select_folder(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(out_dir))

        # A deliberately slow op. `progress` reaches the main thread through
        # a queued connection, so with an instant op the worker finishes
        # every volume before the event loop delivers the first one and the
        # Cancel click lands too late to prove anything. Sleeping per volume
        # makes the race the user's (a human clicking Cancel), not the
        # scheduler's.
        def _slow_add(volume, **kwargs):
            time.sleep(0.05)
            return apply_calculation(volume, "add", 5)

        # Click Cancel as soon as the first volume lands.
        widget._progress_bar.valueChanged.connect(
            lambda value: widget._cancel_btn.click() if value == 1 else None
        )

        widget._run_op(_slow_add, {}, "calc")
        qtbot.waitUntil(lambda: not widget._is_running, timeout=15000)

        written = list(out_dir.iterdir())
        assert 0 < len(written) < 8, "cancel should have stopped it early"
        assert "cancelled" in viewer.layers[-1].name
        assert "Cancelled" in widget._status_label.text()

    def test_closing_the_widget_mid_run_does_not_crash(self, preproc, tmp_path, qtbot):
        # Qt destroying a running QThread is a hard crash, not an exception.
        widget, viewer = preproc
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=8, shape=(4, 16, 16))
        _select_folder(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(tmp_path / "out"))
        widget._calc_value_spin.setValue(5)

        widget._progress_bar.valueChanged.connect(
            lambda value: widget.close() if value == 1 else None
        )

        widget._run_arithmetic()
        qtbot.waitUntil(lambda: not widget._is_running, timeout=15000)

        assert widget._worker is None

    def test_cancelling_a_real_preview_returns_the_widget_to_idle(
        self, preproc, tmp_path, qtbot
    ):
        # A cancelled FunctionWorker3D deliberately emits neither
        # run_finished nor errored, so teardown cannot hang off those:
        # without a finished-driven path the widget stays busy forever and
        # every later click is refused.
        widget, viewer = preproc
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 16, 16))
        _select_folder(widget, viewer, in_dir)

        def _slow_blur(volume, **kwargs):
            time.sleep(0.3)
            return volume

        widget._run_preview(_slow_blur, {}, "blur")
        widget._cancel_btn.click()

        qtbot.waitUntil(lambda: not widget._is_running, timeout=10000)
        assert widget._blur_preview_btn.isEnabled()
        assert widget._calc_run_btn.isEnabled()
        assert not widget._cancel_btn.isEnabled()
        assert widget._worker is None

    def test_a_real_run_that_errors_returns_the_widget_to_idle(
        self, preproc, tmp_path, qtbot
    ):
        widget, viewer = preproc
        in_dir = tmp_path / "in"
        write_timepoint_folder(in_dir, n_timepoints=3, shape=(4, 16, 16))
        _select_folder(widget, viewer, in_dir)
        widget._output_folder_edit.setText(str(tmp_path / "out"))

        def _boom(volume, **kwargs):
            raise RuntimeError("op exploded")

        widget._run_op(_boom, {}, "calc")

        qtbot.waitUntil(lambda: not widget._is_running, timeout=10000)
        assert widget._calc_run_btn.isEnabled()
        assert not widget._cancel_btn.isEnabled()


def _write_tracking_csv(path, n_frames):
    import pandas as pd

    rows = [{"frame": t, "z": 5.0, "y": 10.0, "x": 10.0} for t in range(n_frames)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def tracking(make_napari_viewer, monkeypatch):
    from napari_spotiflow_tracking_nd import _workers3d
    from napari_spotiflow_tracking_nd._tracking_widget3d import TrackingWidget3D

    # Linking is fast; slow each frame so a human-speed Cancel/close can
    # land mid-run (see the pre-processing cancel test for why).
    real_iter_link = _workers3d.iter_link

    def _slow_iter_link(*args, **kwargs):
        for item in real_iter_link(*args, **kwargs):
            time.sleep(0.05)
            yield item

    monkeypatch.setattr(_workers3d, "iter_link", _slow_iter_link)
    viewer = make_napari_viewer()
    return TrackingWidget3D(viewer), viewer


class TestRealThreadingTracking:
    def test_cancelling_a_real_tracking_run_keeps_the_frames_linked(
        self, tracking, tmp_path, qtbot
    ):
        widget, viewer = tracking
        widget._set_input_csv(_write_tracking_csv(tmp_path / "spots.csv", n_frames=20))
        widget._progress_bar.valueChanged.connect(
            lambda value: widget._cancel_btn.click() if value == 1 else None
        )

        widget._run_tracking()
        qtbot.waitUntil(lambda: not widget._is_running, timeout=15000)

        import pandas as pd

        written = pd.read_csv(tmp_path / "spots_tracks.csv")
        assert 0 < written["frame"].nunique() < 20, "cancel should have stopped it early"
        assert "cancelled" in viewer.layers[-1].name

    def test_closing_the_tracking_widget_mid_run_does_not_crash(
        self, tracking, tmp_path, qtbot
    ):
        widget, _ = tracking
        widget._set_input_csv(_write_tracking_csv(tmp_path / "spots.csv", n_frames=20))
        widget._progress_bar.valueChanged.connect(
            lambda value: widget.close() if value == 1 else None
        )

        widget._run_tracking()
        qtbot.waitUntil(lambda: not widget._is_running, timeout=15000)

        assert widget._worker is None


@pytest.fixture
def track_analysis(make_napari_viewer, monkeypatch):
    from napari_spotiflow_tracking_nd import _workers3d
    from napari_spotiflow_tracking_nd._track_analysis_widget3d import TrackAnalysisWidget3D

    # Analysis is fast; slow each track so a human-speed Cancel can land
    # mid-run (see the pre-processing cancel test for why).
    real_analyse_track = _workers3d.analyse_track

    def _slow_analyse_track(*args, **kwargs):
        time.sleep(0.05)
        return real_analyse_track(*args, **kwargs)

    monkeypatch.setattr(_workers3d, "analyse_track", _slow_analyse_track)
    viewer = make_napari_viewer()
    return TrackAnalysisWidget3D(viewer), viewer


class TestRealThreadingTrackAnalysis:
    def test_cancelling_a_real_analysis_writes_nothing(self, track_analysis, tmp_path, qtbot):
        import json

        import pandas as pd

        widget, _ = track_analysis
        rows = [
            {"particle": p, "frame": t, "link_z": 5.0, "link_y": 10.0 * p,
             "link_x": 10.0, "censored": False}
            for p in range(20) for t in range(5)
        ]
        csv = tmp_path / "spots_tracks.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        csv.with_suffix(".json").write_text(json.dumps(
            {"pixel_size_xy_um": 0.104, "voxel_size_z_um": 0.210, "frame_interval_s": 3.0}
        ))
        widget._set_input_csv(csv)
        out = tmp_path / "results"
        widget._output_dir_edit.setText(str(out))
        widget._progress_bar.valueChanged.connect(
            lambda value: widget._cancel_btn.click() if value == 1 else None
        )

        widget._run_analysis()
        qtbot.waitUntil(lambda: not widget._is_running, timeout=15000)

        assert not out.exists()
        assert not (tmp_path / "results.partial").exists()
        assert "Cancelled" in widget._status_label.text()

