"""Shared run-state machine for widgets that drive a worker thread.

Both plugin widgets need the same thing: exactly one job in flight, Run
and Preview disabled while it runs, a Cancel button that is only live
then, a progress bar, a status line, and a teardown that never lets Qt
destroy a running `QThread`. That is enough stateful behaviour that two
copies would drift, so it lives here.
"""
from __future__ import annotations

from napari.utils.notifications import show_error
from qtpy.QtWidgets import QApplication

# The widget currently running a job, process-wide. The spec's rule is
# "exactly one job in flight at a time" — not one per widget: both widgets
# drive the same GPU, so a pre-processing run and a detection run together
# is how a long job on a real dataset turns into a CUDA OOM that looks like
# a plugin bug.
_ACTIVE_RUN = None


def _claim_process_run(owner) -> bool:
    global _ACTIVE_RUN
    if _ACTIVE_RUN is not None and _ACTIVE_RUN is not owner:
        return False
    _ACTIVE_RUN = owner
    return True


def _release_process_run(owner) -> None:
    global _ACTIVE_RUN
    if _ACTIVE_RUN is owner:
        _ACTIVE_RUN = None


class RunStateMixin:
    """Mixed into a `QWidget` alongside the widget's own UI.

    The host creates its own controls and registers them with
    `_init_run_state`, then starts jobs with `_begin_run`.
    """

    def _init_run_state(self, *, run_buttons, cancel_button, progress_bar, status_label):
        self._run_buttons = list(run_buttons)
        self._cancel_button = cancel_button
        self._progress_bar = progress_bar
        self._status_label = status_label
        self._worker = None
        self._busy = False
        self._cancel_requested = False
        self._last_progress = (0, 0)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._request_cancel)
        self._set_status("Idle.")

    # ── state ────────────────────────────────────────────────────────

    @property
    def _is_running(self) -> bool:
        # Busy, not "holds a worker": the worker reference outlives the
        # result when a real thread is still unwinding (see `_end_run`).
        return self._busy

    def _set_status(self, text: str):
        self._status_label.setText(text)

    def _set_busy(self, busy: bool):
        for button in self._run_buttons:
            button.setEnabled(not busy)
        self._cancel_button.setEnabled(busy)

    # ── the test seam ────────────────────────────────────────────────

    def _launch_worker(self, worker):
        """Start `worker` on its own thread.

        The one seam tests override — replacing it with `worker.run()`
        keeps the work synchronous while the guard, the wiring, the button
        states and the progress handling above and below still run.
        """
        worker.start()

    # ── orchestration ────────────────────────────────────────────────

    def _begin_run(self, worker, *, total: int, description: str, on_finished) -> bool:
        """Wire `worker` up, flip the UI to busy and launch it.

        Returns False without starting anything when a job is already in
        flight — here or in the other widget.
        """
        if self._busy:
            show_error("A job is already running — cancel it first.")
            return False
        if not _claim_process_run(self):
            show_error(
                "Another job is already running in this plugin — wait for it "
                "to finish or cancel it first."
            )
            return False

        self._worker = worker
        self._busy = True
        self._cancel_requested = False
        self._last_progress = (0, total)
        # Busy first, then the bar: resetting the bar emits `valueChanged`,
        # and every such event during a run should see the buttons already
        # in their busy state.
        self._set_busy(True)
        self._progress_bar.setMaximum(max(total, 0))
        self._progress_bar.setValue(0)
        self._set_status(f"{description}…")

        # `FunctionWorker3D` has no `progress` — a single call has nothing
        # to report until it is done.
        if hasattr(worker, "progress"):
            worker.progress.connect(self._on_progress)
        worker.errored.connect(self._on_errored)
        # The host's handler is connected first so its layer exists before
        # the UI reports idle; Qt calls slots in connection order.
        worker.run_finished.connect(on_finished)
        worker.run_finished.connect(self._on_run_finished)
        # `QThread.finished` is the only signal that fires on *every* exit
        # path, including a cancelled `FunctionWorker3D`, which emits
        # neither `run_finished` nor `errored` by design. Without this a
        # cancelled preview would leave the widget busy forever.
        worker.finished.connect(self._on_worker_thread_finished)

        self._launch_worker(worker)
        return True

    def _on_progress(self, current: int, total: int):
        self._last_progress = (current, total)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

    def _on_errored(self, message: str):
        if not self._cancel_requested:
            show_error(message)
            self._set_status(f"Failed: {message}")
        else:
            self._set_status(f"Cancelled at {self._progress_text()}.")
        self._end_run()

    def _on_run_finished(self, _payload=None):
        if self._cancel_requested:
            self._set_status(f"Cancelled at {self._progress_text()}.")
        else:
            self._set_status("Done.")
        self._end_run()

    def _on_worker_thread_finished(self):
        """The thread has exited — whatever it did or didn't report.

        A cancelled `FunctionWorker3D` suppresses both `run_finished` and
        `errored`, so this is the only thing that can return the widget to
        idle in that case.
        """
        if self._busy:
            self._set_status("Cancelled — the result was discarded.")
            self._end_run()
        # Safe to release only now: the thread is done with itself.
        self._worker = None

    def _end_run(self):
        self._busy = False
        _release_process_run(self)
        self._set_busy(False)
        worker = self._worker
        # Dropping the last reference to a *running* `QThread` risks Qt
        # collecting it mid-run, which is a hard crash rather than an
        # exception. When a real thread is still unwinding,
        # `_on_worker_thread_finished` releases it instead.
        if worker is None or not worker.isRunning():
            self._worker = None

    def _request_cancel(self):
        if not self._busy or self._worker is None:
            return
        self._cancel_requested = True
        self._worker.cancel()
        self._set_status("Cancelling — finishing the current volume…")

    # ── helpers for the host's finished handlers ─────────────────────

    def _progress_text(self) -> str:
        current, total = self._last_progress
        return f"{current}/{total}" if total else f"{current}"

    def cancelled_suffix(self) -> str:
        """`" [cancelled 120/349]"` after a cancel, else `""`.

        Appended to the name of a layer built from partial output, so a
        half-finished result is never mistaken for a complete one. Stays
        set until the next `_begin_run`; callers use it from inside a
        finished handler.
        """
        if not self._cancel_requested:
            return ""
        return f" [cancelled {self._progress_text()}]"

    # ── teardown ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Never let Qt destroy a running `QThread` — that is a hard crash,
        not an exception. A series run returns within one volume; a preview
        is a single uninterruptible call, so this wait can be long."""
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.isRunning():
            self._cancel_requested = True
            worker.cancel()
            # Paint the explanation before blocking, or the wait is
            # indistinguishable from a hang.
            self._set_status("Waiting for the running job to finish…")
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
            worker.wait()
        self._worker = None
        self._busy = False
        _release_process_run(self)
        super().closeEvent(event)
