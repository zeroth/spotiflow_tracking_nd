from __future__ import annotations

import pytest
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import QLabel, QProgressBar, QPushButton, QWidget

from napari_spotiflow_tracking_nd._run_state import RunStateMixin


class FakeWorker(QThread):
    """A worker with the same signal surface as the real ones, whose body
    the test drives explicitly."""

    progress = Signal(int, int)
    run_finished = Signal(object)
    errored = Signal(str)

    def __init__(self, emit_total=3, fail_with=None):
        super().__init__()
        self._emit_total = emit_total
        self._fail_with = fail_with
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        if self._fail_with is not None:
            self.errored.emit(self._fail_with)
            return
        for i in range(self._emit_total):
            if self.cancelled:
                break
            self.progress.emit(i + 1, self._emit_total)
        self.run_finished.emit("payload")


class Host(RunStateMixin, QWidget):
    """Minimal host: the mixin needs controls, not a real plugin widget."""

    def __init__(self):
        super().__init__()
        self.run_btn = QPushButton("Run")
        self.preview_btn = QPushButton("Preview")
        self.cancel_btn = QPushButton("Cancel")
        self.bar = QProgressBar()
        self.status = QLabel()
        self.launched = []
        self.finished_payloads = []
        self._init_run_state(
            run_buttons=[self.run_btn, self.preview_btn],
            cancel_button=self.cancel_btn,
            progress_bar=self.bar,
            status_label=self.status,
        )

    def _launch_worker(self, worker):
        # The seam: run synchronously so the test can assert immediately.
        self.launched.append(worker)
        worker.run()

    def start(self, worker, total=3):
        return self._begin_run(
            worker, total=total, description="Working",
            on_finished=self.finished_payloads.append,
        )


@pytest.fixture
def host(qtbot):
    widget = Host()
    qtbot.addWidget(widget)
    return widget


class TestRunLifecycle:
    def test_buttons_are_idle_before_a_run(self, host):
        assert host.run_btn.isEnabled()
        assert host.preview_btn.isEnabled()
        assert not host.cancel_btn.isEnabled()

    def test_a_completed_run_returns_the_buttons_to_idle(self, host):
        host.start(FakeWorker())

        assert host.run_btn.isEnabled()
        assert host.preview_btn.isEnabled()
        assert not host.cancel_btn.isEnabled()
        assert not host._is_running

    def test_the_hosts_finished_handler_receives_the_payload(self, host):
        host.start(FakeWorker())

        assert host.finished_payloads == ["payload"]

    def test_progress_drives_the_bar(self, host):
        host.start(FakeWorker(emit_total=4), total=4)

        assert host.bar.maximum() == 4
        assert host.bar.value() == 4


class TestSecondJobRefused:
    def test_a_second_job_is_refused_while_one_is_in_flight(self, host):
        # Two concurrent workers would write the same output folder.
        started = []

        class Reentrant(FakeWorker):
            def run(self):
                # Re-enter while this job is still "running".
                started.append(host.start(FakeWorker()))
                self.run_finished.emit("payload")

        host.start(Reentrant())

        assert started == [False], "the nested start should have been refused"
        assert len(host.launched) == 1


class TestCancel:
    def test_cancel_asks_the_worker_to_stop_and_marks_the_run(self, host):
        class CancellingWorker(FakeWorker):
            def run(self):
                self.progress.emit(2, 5)
                host._request_cancel()  # user clicks Cancel mid-run
                self.run_finished.emit("payload")

        host.start(CancellingWorker(), total=5)

        assert host.cancelled_suffix() == " [cancelled 2/5]"
        assert "Cancelled" in host.status.text()

    def test_cancel_before_any_progress_does_not_produce_a_nonsense_suffix(self, host):
        # `_last_progress` is still (0, total) here — the suffix must read
        # sensibly rather than "[cancelled 0/0]" or blowing up.
        class ImmediateCancel(FakeWorker):
            def run(self):
                host._request_cancel()
                self.run_finished.emit("payload")

        host.start(ImmediateCancel(), total=349)

        assert host.cancelled_suffix() == " [cancelled 0/349]"

    def test_a_clean_run_has_no_cancelled_suffix(self, host):
        host.start(FakeWorker())

        assert host.cancelled_suffix() == ""

    def test_cancel_when_nothing_is_running_is_a_no_op(self, host):
        host._request_cancel()  # must not raise

        assert not host._is_running


class TestErrorPath:
    def test_an_errored_run_returns_the_buttons_to_idle(self, host):
        # Otherwise the widget is stuck with everything greyed out forever.
        host.start(FakeWorker(fail_with="disk on fire"))

        assert host.run_btn.isEnabled()
        assert not host.cancel_btn.isEnabled()
        assert not host._is_running
        assert "disk on fire" in host.status.text()

    def test_an_errored_run_does_not_call_the_finished_handler(self, host):
        host.start(FakeWorker(fail_with="disk on fire"))

        assert host.finished_payloads == []
