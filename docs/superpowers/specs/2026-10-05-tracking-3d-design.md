# Phase 2 — 3D spot tracking — design

Status: approved in chat, pending written spec review.
Follows: `2026-09-22-nd-plugin-phase1-design.md`, which scoped Phase 2 as
"port `_tracking.py`'s wrapper and the Tracking widget with `pos_columns`
fixed to 3D", and `2026-09-24-gaussian-fit-and-mask-design.md`, whose fit
columns this phase consumes.

## Purpose

Link the detection pass's per-timepoint spots into tracks. The user named
two downstream uses, weighted equally:

- **lifetimes / dynamics** of each structure (CLC-DsRed puncta): when it
  appears, how long it lasts, its intensity over time — so track starts
  and ends must be trustworthy, short gaps bridged, and tracks cut off by
  the movie's ends marked as such;
- **motion**: displacement, MSD, speed — so linking must be accurate and
  positions as precise as the data allows.

The analysis itself (lifetimes, MSD, steps) is Phase 3. This phase
produces the tracks table and layer that Phase 3 reads.

## Measured constraints

From the user's own detection output in `D:\Data\3d_Tracking\23-09-2026-01`:

| Source | Frames | Spots / frame (median, max) | Rows |
|---|---|---|---|
| `endo.csv` (deskewed) | 349 | 117, 1,371 | ~106k |
| `endo2.csv` (deconvolved) | 10 | 3,101, 3,160 | ~31k (≈1.08M for 349) |

A full deconvolved series is ~1M rows — seconds to minutes for trackpy, so
the run must be threaded with progress, but needs no special engineering.
The whole table (~200 MB) fits in RAM, unlike the image data.

Both files' sidecars (`endo.json`, `endo2.json`) hold **all-null
calibration** — detection was run on uncalibrated layers. The widget
therefore cannot rely on the sidecar for calibration and carries its own
editable fields.

Verified against trackpy 0.7: `link_df_iter` yields one linked frame at a
time, and `search_range` accepts a per-axis `(z, y, x)` tuple.

## Decisions (from the brainstorm)

- **Link in pixels, scale the output.** Linking runs in voxel coordinates,
  so it works with or without calibration. Physical units are added as
  extra columns and as the Tracks layer's `scale`, never by rewriting the
  coordinates.
- **Anisotropic search range, linked to calibration.** The user enters the
  search range in xy pixels; with **Link to calibration** ticked the z
  component is derived so the range is the same physical distance on every
  axis — the same rule, and the same formula, as Pre-Processing's
  **Link σ to calibration**.
- **Fitted centres when available.** Link on `fit_z/y/x` for spots whose
  Gaussian fit converged, Spotiflow's `z/y/x` otherwise.
- **Approach A: a standalone Spot Tracking widget reading the detection
  CSV**, not a verbatim port of the 2D widget (no progress, uncancellable,
  needs 1M points loaded in napari first) and not a section of the
  Detection widget (couples two independently re-run stages).

## Design

### 1. `_tracking3d.py` — logic, no Qt, no napari

```python
def choose_link_coordinates(df: pd.DataFrame, use_fit: bool = True) -> pd.DataFrame
def iter_link(df, search_range_zyx, memory, *, adaptive=False,
              should_stop=lambda: False) -> Iterator[tuple[int, pd.DataFrame]]
def filter_short_tracks(df: pd.DataFrame, min_length: int) -> pd.DataFrame
def flag_censored(df: pd.DataFrame, first_frame: int, last_frame: int) -> pd.DataFrame
def add_physical_columns(df: pd.DataFrame, calibration: dict) -> pd.DataFrame
```

- **`choose_link_coordinates`** adds `link_z, link_y, link_x` and
  `coord_source` (`"fit"` / `"detected"`). A row uses its fitted centre
  only when `use_fit` is on, `converged` is true and all three `fit_*`
  values are finite; otherwise its detected `z, y, x`. A CSV without fit
  columns is all `"detected"`. Original `z/y/x` and `fit_*` are untouched.
- **`iter_link`** groups by `frame` and drives `trackpy.link_df_iter` with
  `pos_columns=["link_z", "link_y", "link_x"]`, `t_column="frame"`,
  `search_range=search_range_zyx`, `memory=memory`. It yields
  `(frame, linked_frame_df)` per frame so the caller can report progress
  and stop between frames (`should_stop` is checked before each frame).
  **Every frame from the first to the last is fed to trackpy, a frame
  with no spots as an empty frame.** Measured on trackpy 0.7: trackpy
  counts `memory` in frames it has been fed, not in `frame` values, so
  skipping a spotless frame lets `memory=0` bridge it. With `adaptive=True` it passes `adaptive_stop=0.1 * min(search_range_zyx)`
  and `adaptive_step=0.95`. trackpy's `SubnetOversizeException` is
  re-raised as a `ValueError` with a plain message: lower the search range
  or enable adaptive search.
- **`filter_short_tracks`** drops tracks with fewer than `min_length`
  rows. Default 1 — keep everything; short-lived structures are real data
  for lifetime analysis.
- **`flag_censored`** adds a boolean `censored` column, true on every row
  of a track that has a row at `first_frame` or `last_frame`: its true
  lifetime is unknown. `first_frame`/`last_frame` are the detection CSV's
  first and last frame (the last frame actually linked, on a cancelled
  run), not the tracks'.
- **`add_physical_columns`** adds `z_um` (from `link_z × voxel_size_z_um`),
  `y_um`/`x_um` (from `link_y/x × pixel_size_xy_um`), and
  `time_s = frame × frame_interval_s` — each only when its own calibration
  value is known; nothing is guessed.

Shared helper, moved to `_io3d.py` beside `compute_layer_scale`:

```python
def isotropic_zyx(xy_value, pixel_size_xy_um, voxel_size_z_um) -> tuple[float, float, float]
    # (xy_value * pixel_size_xy_um / voxel_size_z_um, xy_value, xy_value)
```

The Pre-Processing widget's inline `σ_xy × (pixel_size_xy_um /
voxel_size_z_um)` (`_cleanup_widget3d.py`) switches to it — one formula,
one copy. On the reference calibration, 4 → `(1.98, 4, 4)`.

**Output columns**, one row per spot: every detection-CSV column, plus
`link_z/y/x`, `coord_source`, `particle` (trackpy's name, as in the 2D
plugin's CSVs), `censored`, and `z_um/y_um/x_um/time_s` where calibrated.

### 2. `TrackingWorker3D` and files

`TrackingWorker3D(QThread)` in `_workers3d.py`, shaped like its siblings:
signals `progress(int, int)`, `run_finished(str)` (tracks CSV path),
`errored(str)`, and `cancel()`.

In the thread: read the detection CSV (seconds at 1M rows — kept off the
UI thread), `choose_link_coordinates`, `iter_link` with progress per
linked frame (`120 / 349`), then `filter_short_tracks`, `flag_censored`,
`add_physical_columns`, and write the CSV.

- **Written once at the end**, not appended per frame like detection:
  the min-length filter and the `censored` flag both need each track
  whole. It is written to `<name>.partial` and renamed into place, so a
  failed run neither leaves a valid-looking half table nor destroys a
  previous run's output under the same name.
- A run cancelled before any spot is linked writes nothing and reports
  an empty path; the widget then adds no layer.
- **Cancel** stops between frames; the frames linked so far are still
  filtered, flagged (series end = last frame linked) and written, and the
  layer gets the existing `[cancelled 120/349]` suffix.
- **Errors**: the `.partial` file is removed and the message goes out on
  `errored`; any earlier file at the output path is left untouched.
- Runs through `RunStateMixin`, including the **process-wide one-job
  lock**: tracking is CPU-only, but sharing the lock means it can never
  read a CSV a detection run is still writing.

Files:

- **Tracks CSV**: chosen in a save dialog, defaulting to
  `<detection stem>_tracks.csv` beside the detection CSV. A path equal to
  the input CSV is refused.
- **Sidecar** at `calibration_sidecar_path(tracks_csv)`
  (`<stem>_tracks.json`): the calibration used, plus the run parameters —
  `search_range_zyx`, `link_to_calibration`, `memory`, `min_length`,
  `use_fit`, `adaptive`, `source_csv`, and `trackpy_version` — so the CSV
  can be interpreted and reproduced without the widget. A failed sidecar
  write is a `show_error`; the CSV and layer are kept.

### 3. `TrackingWidget3D` — `_tracking_widget3d.py`

Registered in `napari.yaml` as **Spot Tracking**. One panel, no tabs.

- **Input**: a combo of Points layers whose metadata carries `csv_path`
  (detection output), plus **Browse CSV…** to track a file without loading
  it. A label shows the active CSV with its row and frame counts.
- **Calibration**: pixel xy (µm), voxel z (µm), frame interval (s).
  Prefilled from the Points layer's metadata, else the CSV's sidecar, else
  blank (= unknown). Always editable.
- **Linking**: search range (xy px, default 4); **Link to calibration**
  (off by default, like Link σ — the user's existing sidecars are
  uncalibrated, so on-by-default would refuse the first run; when ticked, z is shown read-only, e.g. `z = 1.98 px`; when unticked, a
  z spin box, default equal to xy, is editable; ticked with no
  calibration, Run is refused with a message, as Link σ does); memory
  (frames, default 3); min track length (default 1); **Use fitted
  centres** (default on); **Adaptive search** (default off).
- **Run / Cancel / progress bar / status** via `RunStateMixin`. No
  Preview — linking one frame is meaningless, and a full run is fast.

Output layer: a Tracks layer named `<points name> (tracks)` (or
`<csv stem> (tracks)` when browsed), plus the cancel suffix.

- `data = [particle, frame, link_z, link_y, link_x]`, in pixels.
- `scale` from `compute_layer_scale` on the calibration fields, falling
  back to the Points layer's `scale`, so it overlays the image correctly.
- `features`: `coord_source`, `prob`, `censored`, and any fit intensity
  columns present (`amplitude`, `integrated_intensity`, `background`).
- `metadata`: `tracks_csv_path` and the calibration — for Phase 3.
- Status: `N tracks, M spots, K censored, across F frames`.

**Shared combo helper.** The layer-combo refresh each widget re-implements
becomes one `refresh_layer_combo(combo, viewer, layer_type, predicate=None)`
(new `_layer_utils.py`), used by all three widgets — the dedup the
threading spec parked until a third widget existed.

### 4. Errors and edge cases

- CSV missing any of `frame, z, y, x` → refused, naming the missing
  columns (this also rejects picking a tracks CSV that lacks them).
- Fewer than two frames with spots → refused: nothing to link.
- Link to calibration ticked, xy or z calibration blank → refused.
- Output path equals input path → refused.
- `SubnetOversizeException` → plain message, nothing written.
- Any other worker error → `.partial` removed, previous output kept,
  status shows the error.

### 5. Dependency

`trackpy>=0.6` added to `pyproject.toml` dependencies and installed into
`.venv` (currently absent there). Verified on 0.7.

### 6. Testing

Mirrors the existing convention: no GPU, no real data, synthetic inputs.

- **`_tracking3d`**: coordinate choice (converged / failed / no-fit CSV /
  `use_fit=False`); an anisotropic range linking a z-step an isotropic one
  of the same xy value rejects, and vice versa; memory bridging a one-frame
  gap; `min_length` filtering; `censored` at first and last frame and with
  an early `last_frame`; physical columns present only per known value;
  `should_stop` halting between frames; the `SubnetOversize` message.
- **`isotropic_zyx`**: 4 → `(1.98, 4, 4)` on the reference calibration;
  the Pre-Processing widget's existing linked-σ tests still pass.
- **`TrackingWorker3D`**: progress counts; cancel mid-run writes a valid
  partial CSV with the right `censored`; partial file removed on error;
  sidecar contents.
- **`TrackingWidget3D`**, via the `_launch_worker` seam: calibration
  prefill order (layer → sidecar → blank); linked-but-uncalibrated
  refusal; input-overwrite refusal; Tracks layer data, scale and features;
  cancelled suffix.
- **One real-thread** start/cancel/teardown test, like the existing ones.
- Whole suite green after the `isotropic_zyx` and `refresh_layer_combo`
  refactors.
- **README**: a Tracking section explaining the anisotropic search range.

## Risks

- **Dense deconvolved frames.** ~3,100 spots/frame with a generous search
  range can trigger `SubnetOversizeException`. Mitigated by the message
  and the adaptive option; the right default range is unmeasured on real
  data and will be tuned by the user.
- **Fitted vs detected centres mixed within one track.** A track can
  alternate between `"fit"` and `"detected"` rows where fits fail
  intermittently, adding sub-voxel jitter to MSD. `coord_source` is kept
  per row so Phase 3 can measure or exclude it.
- **`memory` and lifetimes.** Gap-bridging merges a structure that
  disappears and a different one that appears nearby within `memory`
  frames. Default 3 follows the 2D plugin; it is a user parameter.

## Out of scope

- Phase 3 analysis: lifetimes, MSD, speed, step detection, the 2D
  Analysis tab, and the 2D filter panel.
- Sampling intensity from an image layer — the fit columns already carry
  per-spot intensity.
- Gap-closing / split / merge beyond trackpy's `memory`.
- A napari reader for tracks CSVs.
- A headless batch CLI.
