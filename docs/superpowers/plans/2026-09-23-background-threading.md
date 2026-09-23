# Background Threading, Cancellation and Live Progress — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move both widgets' runs and previews onto a background thread, with a Cancel button that works, a progress bar that animates, and partial results kept and loaded when a run is cancelled.

**Architecture:** The existing `QThread` subclasses stay as they are and are started with `.start()` instead of `.run()`. A new `FunctionWorker3D` carries single-volume preview work onto a thread. A `RunStateMixin` holds the run-state machine — one job in flight, button enable/disable, cancel flag, progress tracking, safe teardown — so both widgets share one implementation instead of two copies.

**Tech Stack:** qtpy (PyQt6), QThread, pytest-qt (`qtbot.waitSignal`), napari.

**Spec:** [docs/superpowers/specs/2026-09-23-background-threading-design.md](../specs/2026-09-23-background-threading-design.md)

## Global Constraints

- Exactly one job runs per widget at a time. This invariant is load-bearing: it is what makes the detection widget's model cache safe without a lock. If it is ever relaxed, that cache needs a lock.
- `VolumeOpWorker3D` is **not** modified. Both streaming workers already break their loop on cancel and still emit `run_finished` with whatever was written; that behaviour is relied on and must not change.
- `DetectionWorker3D` gets exactly **one** additive, backward-compatible change (Task 1): its `model` argument may also be a zero-argument callable, resolved inside `run()`. This is what moves the slow first weight-load onto the worker thread. Passing a ready model keeps working unchanged.
- Signal names stay `progress(int,int)` / `run_finished(...)` / `errored(str)`. `run_finished` is deliberately not `finished` — `QThread` defines its own no-argument `finished`, and the collision is exactly what `.start()` would trigger.
- Cancellation granularity, stated in the UI rather than implied: series runs stop after the **current volume** finishes; a preview is one blocking call and **cannot** be interrupted — cancel abandons its result while the thread runs to completion.
- Nothing already written to disk is deleted on cancel, ever.
- No GPU, no real Spotiflow weights, and no real dataset required by any test.

## Review Focus

Input classes the spec implies but that no task's happy path exercises, most likely to bite first. Each has a test pinned to the task that owns the code.

1. **Run clicked twice, or Run then Preview, before the first finishes** — the second must be refused, or two workers write the same output folder concurrently. (Task 2)
2. **Cancel clicked before any `progress` has been emitted** — `_last_progress` is still `(0, total)`; the status and layer suffix must not read `[cancelled 0/0]` or divide by zero. (Task 2)
3. **Worker emits `errored` instead of `run_finished`** — the UI must return to idle and re-enable its buttons, or the widget is permanently stuck with everything greyed out. (Task 2)
4. **The widget is closed while a run is in flight** — must cancel and `wait()`; letting Qt destroy a running `QThread` is a hard crash, not an exception. (Task 5)
5. **A detection run cancelled before any spot was found** — the partial CSV was never created, so the finished handler must not add an empty Points layer or raise on a missing file. (Task 4)

---

### Task 1: `FunctionWorker3D`, and a lazily-resolved detection model

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_workers3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FunctionWorker3D(fn, args=(), kwargs=None, parent=None)` with signals `run_finished(object)` (the callable's return value) and `errored(str)`, plus `cancel()`. Tasks 3 and 4 use it for preview. Also: `DetectionWorker3D`'s `model` argument now accepts a zero-argument callable, resolved once inside `run()` — Task 4 passes one so the weight load happens on the worker thread.

- [ ] **Step 1: Write the failing tests**

Append to `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py -k "FunctionWorker or LazyModel" -v`
Expected: FAIL — `ImportError: cannot import name 'FunctionWorker3D'`, and the lazy-model tests fail because a callable is passed straight to `detect_spots_3d` as if it were a model.

- [ ] **Step 3: Implement it**

In `src/napari_spotiflow_tracking_nd/_workers3d.py`, delete the unused `import traceback` on line 3 (it has never been referenced; `errored.emit(str(exc))` does not use it), and append:

```python
class FunctionWorker3D(QThread):
    """Run one callable on a background thread and hand back its result.

    The two streaming workers above own a loop and can stop between
    iterations. This one carries a single blocking call — a preview of one
    volume — so `cancel()` cannot interrupt it. Cancelling only suppresses
    the result; the thread still runs to completion in the background.
    """

    run_finished = Signal(object)  # whatever `fn` returned
    # `run_finished`, not `finished` — see `VolumeOpWorker3D` for why.
    errored = Signal(str)

    def __init__(self, fn, args=(), kwargs=None, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:
            if not self._cancelled:
                self.errored.emit(str(exc))
            return
        if not self._cancelled:
            self.run_finished.emit(result)
```

Then make `DetectionWorker3D` accept a callable model. Update its
docstring and `__init__` signature line to:

```python
    def __init__(
        self,
        files: list[Path],
        model,  # a loaded model, or a zero-arg callable returning one
        csv_path: Path,
        prob_thresh: float | None = 0.5,
        min_distance: int = 2,
        parent=None,
    ):
```

and insert this as the first statement inside `run()`'s `try:` block,
before `if self._csv_path.exists():`:

```python
            # `model` may be a zero-arg callable so that a slow first
            # weight load happens here, on the worker thread, rather than
            # freezing the UI before this thread even starts. Resolved
            # once, and a failure lands on `errored` like any other.
            if callable(self._model) and not hasattr(self._model, "predict"):
                self._model = self._model()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py -v`
Expected: PASS — the eight new tests plus the existing ones.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_workers3d.py src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py
git commit -m "feat(nd): FunctionWorker3D for single-call background work"
```

---

### Task 2: `RunStateMixin` — the run-state machine

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_run_state.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_run_state.py`

**Interfaces:**
- Consumes: nothing from Task 1 (tests use their own fake worker).
- Produces: `RunStateMixin`, mixed into a `QWidget`. Public surface used by Tasks 3 and 4:
  - `_init_run_state(*, run_buttons, cancel_button, progress_bar, status_label)` — call at the end of the host's `_setup_ui`.
  - `_begin_run(worker, *, total: int, description: str, on_finished) -> bool` — guards, wires, flips the UI to busy, launches. Returns `False` and does nothing if a job is already running.
  - `_launch_worker(worker)` — **the test seam**; defaults to `worker.start()`.
  - `cancelled_suffix() -> str` — `" [cancelled 120/349]"` after a cancel, else `""`.
  - `_is_running` property, `closeEvent`.

Note on the seam: the spec called it `_start_worker`; it is split here so that overriding it in a test still exercises the guard, the signal wiring, the button states and the progress handling. Only the launch primitive is replaced.

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_run_state.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_run_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._run_state'`

- [ ] **Step 3: Implement the mixin**

Create `src/napari_spotiflow_tracking_nd/_run_state.py`:

```python
"""Shared run-state machine for widgets that drive a worker thread.

Both plugin widgets need the same thing: exactly one job in flight, Run
and Preview disabled while it runs, a Cancel button that is only live
then, a progress bar, a status line, and a teardown that never lets Qt
destroy a running `QThread`. That is enough stateful behaviour that two
copies would drift, so it lives here.
"""
from __future__ import annotations

from napari.utils.notifications import show_error


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
        self._cancel_requested = False
        self._last_progress = (0, 0)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._request_cancel)
        self._set_status("Idle.")

    # ── state ────────────────────────────────────────────────────────

    @property
    def _is_running(self) -> bool:
        return self._worker is not None

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
        flight — the button states should already prevent that, but two
        workers writing the same output folder is bad enough to guard
        twice.
        """
        if self._is_running:
            show_error("A job is already running — cancel it first.")
            return False

        self._worker = worker
        self._cancel_requested = False
        self._last_progress = (0, total)
        self._progress_bar.setMaximum(max(total, 0))
        self._progress_bar.setValue(0)
        self._set_busy(True)
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

        self._launch_worker(worker)
        return True

    def _on_progress(self, current: int, total: int):
        self._last_progress = (current, total)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

    def _on_errored(self, message: str):
        show_error(message)
        self._set_status(f"Failed: {message}")
        self._end_run()

    def _on_run_finished(self, _payload=None):
        if self._cancel_requested:
            self._set_status(f"Cancelled at {self._progress_text()}.")
        else:
            self._set_status("Done.")
        self._end_run()

    def _end_run(self):
        self._worker = None
        self._set_busy(False)

    def _request_cancel(self):
        if not self._is_running:
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
        half-finished result is never mistaken for a complete one.
        """
        if not self._cancel_requested:
            return ""
        return f" [cancelled {self._progress_text()}]"

    # ── teardown ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Never let Qt destroy a running `QThread` — that is a hard crash,
        not an exception. A series run returns within one volume; a preview
        may take its full duration, which is the cost of a single
        uninterruptible call."""
        worker = self._worker
        if worker is not None:
            self._cancel_requested = True
            worker.cancel()
            worker.wait()
        super().closeEvent(event)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_run_state.py -v`
Expected: PASS — all 12 tests.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_run_state.py src/napari_spotiflow_tracking_nd/_tests/test_run_state.py
git commit -m "feat(nd): shared run-state machine for threaded jobs"
```

---

### Task 3: Thread the Pre-Processing widget

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_cleanup_widget3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py`

**Interfaces:**
- Consumes: `FunctionWorker3D` (Task 1); `RunStateMixin` with `_init_run_state`, `_begin_run`, `_launch_worker`, `cancelled_suffix`, `_is_running` (Task 2).
- Produces: `PreProcessingWidget3D` now mixes in `RunStateMixin`; new attributes `_cancel_btn`, `_status_label`. `_blocking_note` is removed. Task 5's threaded tests drive this widget.

- [ ] **Step 1: Write the failing tests**

In `src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py`, change the fixture so widget tests run synchronously through the seam, and add the new behaviour tests.

Replace the `preproc_widget` fixture with:

```python
@pytest.fixture
def preproc_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._cleanup_widget3d import PreProcessingWidget3D

    viewer = make_napari_viewer()
    widget = PreProcessingWidget3D(viewer)
    # Run the worker synchronously: these tests assert immediately after
    # the call, and the real threading is covered by test_threading.py.
    widget._launch_worker = lambda worker: worker.run()
    return widget, viewer
```

Append:

```python
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
```

Also add `"_cancel_btn", "_status_label"` to the tuple in `test_has_the_expected_controls`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py -k "ThreadedRunWiring or expected_controls" -v`
Expected: FAIL — `AttributeError: 'PreProcessingWidget3D' object has no attribute '_launch_worker'`

- [ ] **Step 3: Mix in the state machine and add the controls**

In `src/napari_spotiflow_tracking_nd/_cleanup_widget3d.py`:

Add the import beside the others:

```python
from napari_spotiflow_tracking_nd._run_state import RunStateMixin
from napari_spotiflow_tracking_nd._workers3d import FunctionWorker3D, VolumeOpWorker3D
```

Change the class statement (the mixin goes first so its `closeEvent` wins):

```python
class PreProcessingWidget3D(RunStateMixin, QWidget):
```

In `__init__`, drop the `self._worker: VolumeOpWorker3D | None = None` line — `_init_run_state` owns `_worker` now.

Replace the progress block at the end of `_setup_ui` (the `_progress_bar` + `_blocking_note` + `addStretch` section) with:

```python
        # ── Progress ─────────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("%v / %m")
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip(
            "Stops after the volume currently being written. A preview is a "
            "single call and cannot be interrupted — cancelling discards its "
            "result."
        )
        layout.addWidget(self._cancel_btn)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

        self._init_run_state(
            run_buttons=[
                self._bg_run_btn, self._blur_run_btn, self._calc_run_btn,
                self._bg_preview_btn, self._blur_preview_btn, self._calc_preview_btn,
            ],
            cancel_button=self._cancel_btn,
            progress_bar=self._progress_bar,
            status_label=self._status_label,
        )
```

- [ ] **Step 4: Route the series run through `_begin_run`**

Replace the body of `_run_op` from `self._progress_bar.setMaximum(...)` to the end with:

```python
        worker = VolumeOpWorker3D(files=files, output_dir=output_dir, op=op, op_kwargs=op_kwargs)

        def _on_finished(written_paths):
            if not written_paths:
                return
            # Built from the exact list this run wrote, not from a fresh
            # glob of `output_dir`: an unrelated `.tif` left in that folder
            # would otherwise silently join the new layer, or raise a
            # shape-mismatch ValueError out of this signal handler.
            try:
                stack, out_files = build_lazy_stack_from_files(written_paths)
            except (OSError, ValueError) as exc:
                show_error(f"Wrote {len(written_paths)} volumes, but could not "
                           f"load them back as a layer: {exc}")
                return
            self.viewer.add_image(
                stack,
                # A cancelled run is still loaded — with its partial state
                # in the name, so it is never mistaken for a full one.
                name=f"{layer.name} ({suffix}){self.cancelled_suffix()}",
                scale=layer.scale,
                metadata={
                    "source_files": out_files,
                    "pixel_size_xy_um": layer.metadata.get("pixel_size_xy_um"),
                    "voxel_size_z_um": layer.metadata.get("voxel_size_z_um"),
                    "frame_interval_s": layer.metadata.get("frame_interval_s"),
                },
            )
            show_info(f"Wrote {len(written_paths)} volumes to {output_dir}")

        self._begin_run(
            worker, total=len(files), description=f"Running {suffix}", on_finished=_on_finished,
        )
```

- [ ] **Step 5: Route preview through `FunctionWorker3D`**

Replace the body of `_run_preview` after the `volume, t = ...` guard with:

```python
        worker = FunctionWorker3D(op, args=(volume,), kwargs=op_kwargs)

        def _on_finished(result):
            self._show_preview_layer(f"{layer.name} ({kind} preview)", result, layer)
            show_info(f"Preview: {kind} on timepoint {t} (nothing written to disk).")

        self._begin_run(
            worker, total=1, description=f"Previewing {kind} on timepoint {t}",
            on_finished=_on_finished,
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py -v`
Expected: PASS — the existing tests (now running through the seam) plus the four new ones.

- [ ] **Step 7: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_cleanup_widget3d.py src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py
git commit -m "feat(nd): thread the Pre-Processing widget's runs and previews"
```

---

### Task 4: Thread the Detection widget, and move the model load onto the thread

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_detection_widget3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py`

**Interfaces:**
- Consumes: `FunctionWorker3D` (Task 1); `RunStateMixin` (Task 2) — same surface as Task 3.
- Produces: `DetectionWidget3D` mixes in `RunStateMixin`; new `_cancel_btn`, `_status_label`; `_blocking_note` removed. `_load_model_cached` is now called from inside worker bodies.

- [ ] **Step 1: Write the failing tests**

In `src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py`, change the fixture to use the seam:

```python
@pytest.fixture
def detection_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._detection_widget3d import DetectionWidget3D

    viewer = make_napari_viewer()
    widget = DetectionWidget3D(viewer)
    # Synchronous through the seam; real threading is covered by
    # test_threading.py.
    widget._launch_worker = lambda worker: worker.run()
    return widget, viewer
```

Append:

```python
class TestThreadedDetectionWiring:
    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_run_returns_to_idle_via_the_state_machine(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))
        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        model.predict.return_value = (np.array([[1, 2, 3]]), details)
        mock_load_model.return_value = model

        viewer.add_image(
            np.zeros((2, 4, 16, 16), dtype=np.uint16), name="vol",
            metadata={"source_files": files},
        )
        widget._image_combo.setCurrentText("vol")
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))

        widget._run_detection()

        assert not widget._is_running
        assert widget._run_btn.isEnabled()
        assert not widget._cancel_btn.isEnabled()
        assert widget._status_label.text() == "Done."

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_the_model_is_loaded_inside_the_worker_not_before_it(
        self, mock_load_model, detection_widget, tmp_path
    ):
        # A first weight load is slow; doing it on the main thread would
        # freeze the UI before the worker even started.
        widget, viewer = detection_widget
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=1, shape=(4, 16, 16))
        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        model.predict.return_value = (np.array([[1, 2, 3]]), details)
        mock_load_model.return_value = model

        viewer.add_image(
            np.zeros((1, 4, 16, 16), dtype=np.uint16), name="vol",
            metadata={"source_files": files},
        )
        widget._image_combo.setCurrentText("vol")
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))

        loaded_before_launch = {}

        def _launch(worker):
            loaded_before_launch["value"] = mock_load_model.call_count
            worker.run()

        widget._launch_worker = _launch
        widget._run_detection()

        assert loaded_before_launch["value"] == 0, "model was loaded on the main thread"
        assert mock_load_model.call_count == 1

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_a_cancelled_run_with_no_spots_yet_adds_no_layer(
        self, mock_load_model, detection_widget, tmp_path
    ):
        # The partial CSV was never created, so the finished handler must
        # not raise on the missing file or add an empty Points layer.
        widget, viewer = detection_widget
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=4, shape=(4, 16, 16))
        model = MagicMock()
        details = MagicMock()
        details.prob = np.empty((0,))
        model.predict.return_value = (np.empty((0, 3)), details)
        mock_load_model.return_value = model

        viewer.add_image(
            np.zeros((4, 4, 16, 16), dtype=np.uint16), name="vol",
            metadata={"source_files": files},
        )
        widget._image_combo.setCurrentText("vol")
        csv_path = tmp_path / "spots.csv"
        widget._csv_path_edit.setText(str(csv_path))

        def _launch(worker):
            worker.progress.connect(lambda cur, total: widget._request_cancel())
            worker.run()

        widget._launch_worker = _launch
        widget._run_detection()  # must not raise

        assert not csv_path.exists()
        assert [l for l in viewer.layers if l.__class__.__name__ == "Points"] == []
        assert not widget._is_running

    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_preview_runs_through_a_worker(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))
        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.9])
        model.predict.return_value = (np.array([[1, 2, 3]]), details)
        mock_load_model.return_value = model

        viewer.add_image(
            np.zeros((2, 4, 16, 16), dtype=np.uint16), name="vol",
            scale=(1.0, 0.21, 0.104, 0.104),
            metadata={"source_files": files},
        )
        widget._image_combo.setCurrentText("vol")

        widget._run_preview()

        assert "vol (spot preview)" in viewer.layers
        assert not widget._is_running

    def test_the_blocking_note_is_gone(self, detection_widget):
        widget, _ = detection_widget
        assert not hasattr(widget, "_blocking_note")
```

Also add `"_cancel_btn", "_status_label"` to the tuple in `test_has_the_expected_controls`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py -k "ThreadedDetectionWiring or expected_controls" -v`
Expected: FAIL — `AttributeError: 'DetectionWidget3D' object has no attribute '_launch_worker'`

- [ ] **Step 3: Mix in the state machine and add the controls**

In `src/napari_spotiflow_tracking_nd/_detection_widget3d.py`, add the imports:

```python
from napari_spotiflow_tracking_nd._run_state import RunStateMixin
from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D, FunctionWorker3D
```

Change the class statement:

```python
class DetectionWidget3D(RunStateMixin, QWidget):
```

In `__init__`, drop `self._worker: DetectionWorker3D | None = None` (the mixin owns `_worker`); keep the two model-cache lines.

Replace the progress block at the end of `_setup_ui` (`_progress_bar` + `_blocking_note` + `addStretch`) with:

```python
        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("%v / %m")
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip(
            "Stops after the timepoint currently being detected. A preview is "
            "a single call and cannot be interrupted — cancelling discards its "
            "result."
        )
        layout.addWidget(self._cancel_btn)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

        self._init_run_state(
            run_buttons=[self._run_btn, self._preview_btn],
            cancel_button=self._cancel_btn,
            progress_bar=self._progress_bar,
            status_label=self._status_label,
        )
```

- [ ] **Step 4: Move the model load into the worker body, and route the run through `_begin_run`**

In `_run_detection`, delete the `device = ...` / `try: model = self._load_model_cached(...)` block (the model must not be loaded on the main thread) and replace everything from there to the end of the method with:

```python
        device = _get_device(use_gpu=self._use_gpu_check.isChecked())
        model_name = self._model_combo.currentText()

        # Passed as a callable, not a loaded model: resolving it is slow
        # enough to freeze the UI on its own, so `DetectionWorker3D` calls
        # this on its own thread (Task 1). The cache needs no lock because
        # only one job runs at a time.
        worker = DetectionWorker3D(
            files=files,
            model=lambda: self._load_model_cached(model_name, device),
            csv_path=Path(csv_path),
            prob_thresh=self._prob_thresh_spin.value(),
            min_distance=self._min_distance_spin.value(),
        )

        def _on_finished(written_csv_path):
            # `DetectionWorker3D` only writes the CSV once at least one
            # frame has a detection — if every frame in the run has zero
            # spots (or it was cancelled before the first), the file is
            # never created even though `run_finished` still fires.
            if not Path(written_csv_path).exists():
                show_info("Detection finished — no spots found.")
                return

            calibration = {k: layer.metadata.get(k) for k in CALIBRATION_KEYS}
            sidecar = calibration_sidecar_path(written_csv_path)
            try:
                sidecar.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            except OSError as exc:
                show_error(f"Could not write the calibration sidecar {sidecar}: {exc}")

            df = pd.read_csv(written_csv_path)
            if len(df) == 0:
                show_info("Detection finished — no spots found.")
                return
            points = df[["frame", "z", "y", "x"]].to_numpy()
            self.viewer.add_points(
                points,
                name=f"{layer.name} (spots){self.cancelled_suffix()}",
                scale=layer.scale,
                metadata={"csv_path": written_csv_path, **calibration},
                properties={"prob": df["prob"].to_numpy()},
            )
            show_info(f"Detected {len(df)} spots across {df['frame'].nunique()} timepoints.")

        self._begin_run(
            worker, total=len(files), description="Detecting spots",
            on_finished=_on_finished,
        )
```

- [ ] **Step 5: Route preview through `FunctionWorker3D`, loading the model inside it**

Replace everything in `_run_preview` from the `device = ...` line to the end with:

```python
        device = _get_device(use_gpu=self._use_gpu_check.isChecked())
        model_name = self._model_combo.currentText()

        def _detect():
            # Both the model load and the detection happen on the worker
            # thread; the cache is safe unlocked because only one job runs
            # at a time.
            model = self._load_model_cached(model_name, device)
            volume = read_volume(files[t])
            points, _details = detect_spots_3d(
                volume,
                model,
                prob_thresh=self._prob_thresh_spin.value(),
                min_distance=self._min_distance_spin.value(),
            )
            return points

        worker = FunctionWorker3D(_detect)

        def _on_finished(points):
            # One preview layer, updated in place: tuning means clicking
            # this repeatedly, and a layer per click would bury the viewer.
            preview_name = f"{layer.name} (spot preview)"
            if preview_name in self.viewer.layers:
                self.viewer.layers[preview_name].data = points
            else:
                # Points here are (z, y, x) for a single volume, so the
                # source layer's scale minus its leading T entry.
                self.viewer.add_points(
                    points,
                    name=preview_name,
                    scale=layer.scale[1:],
                    face_color="transparent",
                    border_color="red",
                )
            show_info(
                f"Preview: {len(points)} spots on timepoint {t} "
                "(nothing written to disk)."
            )

        self._begin_run(
            worker, total=1, description=f"Previewing detection on timepoint {t}",
            on_finished=_on_finished,
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py -v`
Expected: PASS — existing tests through the seam plus the six new ones.

- [ ] **Step 7: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_detection_widget3d.py src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py
git commit -m "feat(nd): thread detection, and load the model on the worker thread"
```

---

### Task 5: The threaded tests — what actually proves this works

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_tests/test_threading.py`

**Interfaces:**
- Consumes: both widgets from Tasks 3 and 4, with their real `_launch_worker` (no seam override).
- Produces: nothing other tasks use.

These are the only tests that exercise real concurrency. The seam used everywhere else covers none of it — no thread affinity, no event-loop delivery, no teardown. Treat gaps here as gaps in the feature.

- [ ] **Step 1: Write the tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_threading.py`:

```python
"""The tests that actually prove the threading works.

Every other widget test overrides `_launch_worker` to run the worker
synchronously, which exercises the wiring but none of the concurrency.
These use the real `.start()` and wait on signals, so they are what stands
between this feature and a UI that freezes, crashes on close, or reports a
run as finished while it is still going.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

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

        original = widget._calc_op()[0]

        def _spy(volume, **kwargs):
            worker_threads.append(threading.get_ident())
            return original(volume, **kwargs)

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
        original_add_image = viewer.add_image

        def _spy_add_image(*args, **kwargs):
            handler_threads.append(threading.get_ident())
            return original_add_image(*args, **kwargs)

        viewer.add_image = _spy_add_image
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
        widget._calc_value_spin.setValue(5)

        # Click Cancel as soon as the first volume lands.
        widget._progress_bar.valueChanged.connect(
            lambda value: widget._cancel_btn.click() if value == 1 else None
        )

        widget._run_arithmetic()
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
```

- [ ] **Step 2: Run them**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_threading.py -v`
Expected: PASS — five tests. If `test_the_work_happens_off_the_main_thread` fails, `.start()` is not actually being used; if `test_the_gui_is_updated_from_the_main_thread` fails, the finished handler is being called on the worker thread and the widget code must move GUI work into a slot.

- [ ] **Step 3: Run the whole suite**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/ -v`
Expected: PASS, with no `QThread: Destroyed while thread is still running` warnings in the output. Treat that warning as a failure even if the suite is green.

- [ ] **Step 4: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_tests/test_threading.py
git commit -m "test(nd): real-threading tests for start, cancel and teardown"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-09-22-nd-plugin-phase1.md`

**Interfaces:**
- Consumes: the finished behaviour from Tasks 3–5.
- Produces: nothing code depends on.

- [ ] **Step 1: Replace the blocking-run warning in `README.md`**

Delete the blockquote beginning `> **A run blocks the napari window for its full duration.**` and put in its place:

```markdown
> **Runs happen in the background.** Pre-processing and detection run on a
> worker thread, so napari stays responsive: the progress bar shows
> `120 / 349` as it goes, and **Cancel** stops the run after the volume
> currently being processed. A cancelled run keeps everything already
> written and loads it as a layer labelled `[cancelled 120/349]` —
> nothing is discarded. A *preview* is a single uninterruptible call, so
> cancelling one discards its result rather than stopping the work.
```

- [ ] **Step 2: Update the quick-mode bullet**

In the "Quick mode (Preview)" bullet, replace the closing sentence
`Repeated previews update a single layer rather than stacking one per click, and the detection model stays loaded between previews.` with:

```markdown
  Repeated previews update a single layer rather than stacking one per
  click, and the detection model stays loaded between previews. Previews
  run on a worker thread too, so the UI stays live while one is in
  flight.
```

- [ ] **Step 3: Fix the Phase 1 plan's self-contradiction**

In `docs/superpowers/plans/2026-09-22-nd-plugin-phase1.md`, Task 4's **Interfaces** block says the widgets "construct and `.start()` these", while Task 5's note mandates `.run()`. The implementation followed `.run()`. Replace that phrase with:

```markdown
Task 5 and Task 6's widgets construct and run these (Phase 1 called `.run()` synchronously; background threading landed later — see `docs/superpowers/specs/2026-09-23-background-threading-design.md`).
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/plans/2026-09-22-nd-plugin-phase1.md
git commit -m "docs(nd): background runs, cancellation, and a stale plan note"
```

---

## Self-Review Notes

- **Spec coverage:** worker layer + `FunctionWorker3D` → Task 1; one-job-in-flight, button states, cancel semantics, progress, status, teardown → Task 2 (machine) and Tasks 3–4 (wiring); model load moved onto the thread → Task 4; partial results on cancel → Tasks 3 and 4; the `qtbot` tests the spec calls "the tests for this change" → Task 5; the false blocking-note and README → Tasks 3, 4 and 6.
- **Placeholder scan:** none — every step carries real code or an exact command with its expected output.
- **Type consistency:** `_init_run_state`, `_begin_run`, `_launch_worker`, `cancelled_suffix`, `_is_running`, `_request_cancel` are defined in Task 2 and used with those exact names and signatures in Tasks 3–5. `FunctionWorker3D(fn, args, kwargs)` and its `run_finished(object)` / `errored(str)` signals are defined in Task 1 and used identically in Tasks 3 and 4. `run_finished` (never `finished`) throughout.
- **Review Focus coverage:** double-start → Task 2 `TestSecondJobRefused`; cancel before any progress → Task 2 `test_cancel_before_any_progress_does_not_produce_a_nonsense_suffix`; errored path re-enabling the UI → Task 2 `TestErrorPath`; close mid-run → Task 5 `test_closing_the_widget_mid_run_does_not_crash`; detection cancelled with no spots yet → Task 4 `test_a_cancelled_run_with_no_spots_yet_adds_no_layer`.
- **Revised during self-review:** an earlier draft moved the model load onto the thread with a method-local subclass of `DetectionWorker3D` overriding `run()`. That was clever rather than clear, and easy for a fresh implementer to get subtly wrong. It is replaced by one additive change to the worker (`model` may be a zero-arg callable, Task 1) and a one-line lambda at the call site (Task 4) — same effect, readable, and tested directly in Task 1.
- **Worth a reviewer's attention:** the callable-model check in Task 1 is `callable(self._model) and not hasattr(self._model, "predict")`, because a real Spotiflow model object may itself be callable. A model class that is both callable and lacks `.predict` would be mis-resolved; no such thing exists in this codebase, and `detect_spots_3d` already requires `.predict`.
