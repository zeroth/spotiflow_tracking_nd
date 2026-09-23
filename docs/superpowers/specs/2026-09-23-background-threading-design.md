# Background threading, cancellation and live progress — design

Status: approved in chat, pending written spec review.
Follows: `2026-09-22-nd-plugin-phase1-design.md` (Phase 1), which deliberately
deferred this.

## Problem

Phase 1 runs every job synchronously: both widgets end with `worker.run()`
on the calling thread. Consequences on the real dataset (349 timepoints):

- napari stops repainting for the whole run — minutes to hours — and is
  indistinguishable from a hang. A `QLabel` currently apologises for this.
- `cancel()` is implemented and unit-tested on both workers, but no widget
  exposes it, and a blocked event loop could not deliver the click anyway.
  A spec'd Phase 1 feature is unreachable.
- The progress bar is wired to the `progress` signal but never repaints,
  for the same reason.

## Scope

**In:** both widgets' full runs *and* previews move onto a background
thread; a Cancel button that works; a progress bar that animates; partial
results kept and loaded on cancel; safe thread teardown.

**Out:** any change to what the ops compute; the batch/CLI story; Phase 2
tracking; multi-run concurrency (exactly one job in flight at a time, by
design — see below).

## Approach

Keep the existing `QThread` subclasses and call `.start()` instead of
`.run()`. Phase 1's own call-site comment records that the QThread API was
kept precisely so this switch would not disturb the signal wiring.

Rejected: rewriting the workers as `QObject` + `moveToThread` (Qt's
canonical pattern, but a rewrite of two reviewed, tested classes for no
functional gain and more lifetime choreography to get wrong); napari's
`@thread_worker` (most idiomatic for plugins, but discards those workers
and couples them to napari — they currently need only `qtpy`, which is
worth keeping).

**Verified, not assumed:** a signal emitted from the worker thread and
connected to a plain closure is delivered on the *main* thread, because
PyQt binds the connection to the thread that called `connect()`. Both the
closure style used in `_on_finished` and bound-method slots were tested
against the installed Qt binding and both reported the main thread id.
So the existing `_on_finished` closures may keep calling
`viewer.add_image` unchanged — the scariest part of this change is a
non-issue.

## Design

### 1. Worker layer

`VolumeOpWorker3D` and `DetectionWorker3D` are unchanged. Both already
support "cancel keeps partial output": each checks `self._cancelled` at
the top of its loop, breaks, and still emits `run_finished` with whatever
was written.

Preview currently uses **no worker at all** — both widgets call the op
inline. A new `FunctionWorker3D(QThread)` in `_workers3d.py` carries any
single callable onto a thread:

- constructor: `(fn, args=(), kwargs=None, parent=None)`
- signals: `run_finished(object)` — the callable's return value —
  and `errored(str)`
- `cancel()` sets a flag that suppresses `run_finished`; it cannot
  interrupt `fn` itself.

### 2. One job in flight, enforced by the UI

Starting a job disables Run and Preview on that widget and enables Cancel;
`run_finished`/`errored` restore them. Each widget also refuses to start a
second job while `self._worker` is running, so a missed button state can't
produce two concurrent jobs.

This invariant is load-bearing: it is what makes the detection model cache
safe without a lock (see 3).

### 3. Model loading moves onto the thread

`_load_model_cached` currently runs on the main thread in both
`_run_preview` and `_run_detection`. A first load downloads and
initialises weights, so the UI would still freeze before the worker even
starts. The load therefore moves *inside* the threaded work.

The cache stays a plain attribute, unsynchronised, because of the
one-job-in-flight invariant. If that invariant is ever relaxed, the cache
needs a lock — recorded here so the coupling is not rediscovered the hard
way.

### 4. Cancellation semantics, stated honestly

- **Series runs** (both widgets' Run): the volume currently being
  processed finishes, then the loop stops. Cancellation is therefore
  granular to one volume — on this dataset, seconds.
- **Preview**: a single op on a single volume is one blocking call that
  cannot be interrupted. Cancel marks the job abandoned and the UI stops
  waiting; the thread still runs to completion in the background and its
  result is discarded. The button's tooltip says so.

### 5. Partial results on cancel

A cancelled run keeps everything already written and still loads it, named
so the partial state is unmissable — `vol (blur) [cancelled 120/349]` for
a pre-processing run, `vol (spots) [cancelled 120/349]` for a detection
run. The accompanying message names the output folder or CSV. Nothing is
deleted.

**How the widget knows.** Neither worker reports cancellation or a count
in `run_finished`, and that stays true — the widget derives both itself:
it sets a `_cancel_requested` flag when Cancel is clicked, and records the
last `(current, total)` it saw on the `progress` signal. The suffix is
built from those. Deriving it from the payload instead would work for
`VolumeOpWorker3D` (whose `run_finished` carries the written list) but not
for `DetectionWorker3D` (which carries only a CSV path), so the widget-side
flag is the one mechanism that covers both.

### 6. Progress and status

The progress bar animates once the event loop is free; its text format
becomes `%v / %m` so it reads `120 / 349`. The `_blocking_note` label,
which will have become false, is replaced by a status line reporting
idle / running / cancelled / failed.

### 7. Thread lifetime

Destroying a `QThread` while it runs is a hard crash. Each widget gains a
`closeEvent` that cancels the worker and `wait()`s for it. A series run
returns within one volume; a preview may take its full duration, which is
the honest cost of point 4.

## Testing

Hybrid, with the gap stated rather than papered over.

- **Worker unit tests** keep calling `.run()` directly. That is legitimate
  — they test the loop, not the threading.
- **Widget tests** get a seam: `_start_worker(worker)` defaults to
  `worker.start()`, and tests override it to call `worker.run()`. The
  ~20 existing widget tests then keep working almost unchanged, fast and
  deterministic.
- **Threaded tests** — a small number, using `pytest-qt`'s
  `qtbot.waitSignal` — are what actually prove the feature: that `.start()`
  runs off the main thread, that Cancel stops a run and yields partial
  output, that buttons enable/disable correctly, and that `closeEvent`
  during a run does not crash.

**What the seam does not cover, by construction:** anything about real
concurrency — races, thread affinity of GUI calls, event-loop delivery.
Only the `qtbot` tests exercise that, so they are not optional garnish;
they are the tests for this change.

## Risks

- A test suite that passes through the seam while the threaded path is
  broken. Mitigated by the `qtbot` tests above; the risk is that those
  are too few. Reviewers should weigh their coverage specifically.
- Qt teardown at interpreter exit with a live thread. Mitigated by
  `closeEvent`; not fully preventable if the process is killed outright.
- `show_error`/`show_info` called from a queued slot is fine (it lands on
  the main thread), but any *new* code that touches the viewer must stay
  in a slot rather than inside the worker body.

## Out of scope / follow-ups

- Anisotropic sigma in the UI, the `squeeze()` Z==1 bug, layer-rename
  tracking, output-TIFF resolution metadata and the other parked Phase 1
  minors.
- Extracting the duplicated `_refresh_image_combo` — still waiting on a
  third widget (Phase 2 tracking).
