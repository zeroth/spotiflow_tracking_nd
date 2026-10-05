# Phase 3A — 3D track analysis engine and results files — design

Status: approved in chat, pending written spec review.
Follows: `2026-10-05-tracking-3d-design.md` (Phase 2), whose tracks CSV
and sidecar this phase consumes; `2026-09-22-nd-plugin-phase1-design.md`
scoped Phase 3 as "port `_track_properties.py`, extended for anisotropic
3D calibration".

## Purpose and scope

Turn Phase 2's tracks into numbers for the user's two named uses, weighted
equally: **lifetimes / dynamics** of CLC-DsRed structures and **motion**
(MSD, diffusion, motility class). The user wants both the numbers (to
analyse elsewhere) and interactive exploration in napari. That is split
into two sub-projects, each with its own spec and plan:

- **3A (this spec):** the analysis engine, a *Track Analysis* widget that
  runs it, and a folder of results CSVs.
- **3B (next):** interactive exploration — filter panel, layer colouring,
  per-track MSD / step / intensity charts — built on 3A's tables.

## Decisions (from the brainstorm)

- **Intensity comes from the Gaussian-fit columns only** — no image
  sampling. A track's intensity trace is its *converged* spots'
  `integrated_intensity`, gaps skipped, never interpolated. On the user's
  `decon_10` data ~75 % of fits fail, so traces are sparse; the per-track
  `n_intensity_points` makes that visible.
- **MSD in both 3D and xy, side by side**: `D_3d, alpha_3d` from
  `MSD = 6·D·t^α`, `D_xy, alpha_xy` from `MSD = 4·D·t^α`. Axial
  localisation is typically 2–3× worse than lateral; reporting both lets
  the user choose. The motility class uses one of them, chosen in the
  widget (default 3D).
- **Approach A: a standalone Track Analysis widget** with its own logic
  module and worker — not a tab on Spot Tracking (couples re-analysis to
  re-tracking) and not script-only (3B would have to build all UI at once).

## Design

### 1. `_track_analysis3d.py` — logic, no Qt

**Input**: a Phase 2 tracks table — `particle, frame, link_z, link_y,
link_x, censored`, and when detection fitted Gaussians, `converged,
integrated_intensity`.

**Calibration is required**: `pixel_size_xy_um`, `voxel_size_z_um`,
`frame_interval_s`. A 3D distance in raw voxels mixes 0.104 µm and
0.210 µm steps, so there is no voxel-unit fallback.

**Positions** in µm per axis: `z = link_z × voxel_size_z_um`,
`y = link_y × pixel_size_xy_um`, `x = link_x × pixel_size_xy_um`. Each
track is laid on its full frame span `first_frame..last_frame` with NaN
rows for frames bridged by tracking `memory`, so lags are real frame
differences and gaps drop out of averages.

**Per-track table** (`track_properties.csv`, one row per track, sorted by
`track_id`):

| Group | Columns |
|---|---|
| Identity / time | `track_id, n_spots, first_frame, last_frame, n_gap_frames, censored` |
| Lifetime | `lifetime_frames = last_frame − first_frame + 1`; `lifetime_s = lifetime_frames × frame_interval_s` |
| Geometry | `path_length_um` (sum of consecutive *present*-spot distances), `net_displacement_um` (first to last spot), `mean_speed_um_s = path_length_um / ((last_frame − first_frame) × frame_interval_s)` — NaN for a single-spot track |
| Motion | `D_3d, alpha_3d, D_xy, alpha_xy, msd_lags_fitted, motility_class, motility_code` |
| Intensity | `n_intensity_points, mean_intensity, max_intensity` |
| Steps | `n_steps, mean_step_height` |

Rules:

- **Lifetime** counts frames inclusively (seen in frames 10–14 → 5
  frames), the usual coated-pit convention. Censored tracks stay in the
  table, flagged.
- **MSD** per track and per dims (`3d`: z, y, x; `xy`: y, x): for lag
  `L = 1..min(max_lag, span − 1)`, the mean over all pairs of present
  spots exactly `L` frames apart of the squared distance, with its SEM
  and pair count (the 2D `msd_stats` semantics, generalised to NaN-padded
  N-D positions). A lag with zero pairs is omitted.
- **MSD fit**: `curve_fit` of `k·D·t^α` (`k = 6` for 3d, `4` for xy) with
  `t = L × frame_interval_s`, `p0 = (0.001, 0.01)`, bounded `maxfev`, in
  try/except, over the first `n_fit = min(usable lags, max_lag,
  max(4, span // 4))` usable lags (`span = last_frame − first_frame + 1`;
  the usual "first quarter of the lags" rule — long lags rest on few,
  overlapping pairs and would dominate an unweighted fit). Fitted only
  when the track has `≥ min_msd_spots` spots (default 10) and `≥ 4`
  usable lags; otherwise, or on any fit failure or non-finite result,
  `D`/`alpha` are NaN. `msd_curves.csv` still reports every lag up to
  `max_lag`, with `msd_fit` evaluated at each. `msd_lags_fitted` is
  `n_fit` for the 3d fit (0 when not fitted).
- **Motility class**: the 2D thresholds — `alpha < 0.4` constrained,
  `alpha > 1.2` directed, otherwise diffusive (boundaries diffusive),
  `unclassified` when alpha is NaN — applied to `alpha_3d` or `alpha_xy`
  per the `motility_from` parameter. `motility_code` is the same class
  as an integer (0 unclassified, 1 constrained, 2 diffusive, 3 directed),
  since napari's colour-by cannot use a string column.
- **Intensity trace**: `integrated_intensity` of the track's spots with
  `converged == True` and a finite value, in frame order. Without those
  columns every track has `n_intensity_points = 0` and NaN intensities.
- **Steps**: detected with the ported 2D `find_steps(trace, window,
  threshold)` (unchanged) on the intensity trace, each split refined by
  ±1 point to the least-squares boundary (`find_steps` reports the last
  point of the old level, which biased position and height); defaults
  `window = 3` (2D used 20; at 3 s/frame a coated-pit track is ~10–30
  frames), `threshold = 0.5`. A trace shorter than `2 × window + 1`
  points gets `n_steps = NaN` and `mean_step_height = NaN` — "not
  assessed", distinct from zero steps. From the refined indices: a
  step's `frame` is the frame of the first point of the new level;
  `level_before`/`level_after` are the means of the adjacent segments
  between neighbouring steps; `step_height = level_after − level_before`;
  `dwell_before`/`dwell_after` are in **frames** (from the trace's frame
  numbers, so failed fits do not shorten them), the last level lasting
  to the trace's last frame inclusive.

**Population and detail outputs**:

- `lifetime_distribution.csv` — uncensored tracks only, one bin per
  lifetime in frames: `lifetime_frames, lifetime_s, count, fraction`
  (`fraction` of uncensored tracks).
- `msd_curves.csv` — long format for every fitted-or-not track with at
  least one lag: `track_id, dims, lag_frames, lag_s, msd_um2, sem,
  n_pairs, msd_fit` (`msd_fit` NaN when not fitted).
- `steps.csv` — one row per detected step: `track_id, frame,
  level_before, level_after, step_height, dwell_before, dwell_after`
  (dwells in frames).

### 2. `TrackAnalysisWorker3D` — in `_workers3d.py`

Signals `progress(int, int)`, `run_finished(str)` (results folder, or
`""` when cancelled), `errored(str)`; `cancel()`.

In the thread: read the tracks CSV, validate columns, analyse one track
at a time (progress `i / n_tracks`), assemble the tables, write them.

- **Cancel writes nothing.** A partial analysis is a biased sample (the
  first tracks by ID), unlike partial tracking. Reported as "cancelled,
  nothing written".
- **Atomic output**: tables are written into `<folder>.partial/`, then
  the folder is renamed into place (an existing folder of that name is
  replaced only after the new one is complete, and put back if the final
  rename fails). On error the `.partial` folder is removed and any
  earlier results are untouched.
- **Only earlier results are ever replaced.** An existing results folder
  (or a stale `<folder>.old`) is replaced or deleted only when it is
  recognisably an earlier run's — `analysis.json` plus nothing but the
  five result files, no subfolders; a stale `<folder>.partial` only when
  it holds nothing but result files. Anything else is refused with an
  error, before any track is analysed, and left untouched.
- Runs through `RunStateMixin`, sharing the process-wide one-job lock.

### 3. `TrackAnalysisWidget3D` — `_track_analysis_widget3d.py`

Registered in `napari.yaml` as **Track Analysis**.

- **Input**: a combo of Tracks layers carrying `tracks_csv_path` metadata
  (Phase 2 output) plus **Browse CSV…**; editing the path re-reads it
  (the Phase 2 widget's `editingFinished` rule). A label shows the track
  and spot counts.
- **Calibration**: xy (µm), z (µm), dt (s) — prefilled from the layer's
  metadata, then the tracks sidecar (`calibration_sidecar_path`), else
  blank. All three required; blank → run refused.
- **Parameters**: max lag (25), min spots for MSD (10), motility from
  (3D | xy, default 3D), step window (3), step threshold (0.5).
- **Output folder**: defaults to `<tracks stem>_analysis` beside the
  tracks CSV; refused when it resolves to the tracks CSV's own folder or
  any folder above it, or when it exists and is not an earlier results
  folder.
- **Run / Cancel / progress / status** via `RunStateMixin`. Status
  summary: `N tracks — U uncensored, median lifetime X s; F with D/α
  (P % diffusive)`.
- **Layer features**: when the input came from a Tracks layer, the
  per-track columns are joined onto its `features` by `track_id`, so
  napari's own colour-by-feature works immediately for the columns
  without blanks (`lifetime_s, lifetime_frames, n_spots, path_length_um,
  net_displacement_um, motility_code`). napari's colour-by raises on a
  string column and hides every track when a column has any NaN, so the
  others wait for 3B's exploration.

`analysis.json` in the results folder records the calibration, every
parameter, the source tracks CSV, the tracking sidecar's `cancelled`,
`first_frame`, `last_frame` when present, and the trackpy / numpy /
scipy / plugin versions.

### 4. Errors and edge cases

- Tracks CSV missing `particle, frame, link_z, link_y, link_x` or
  `censored` → refused, naming the columns.
- Calibration incomplete → refused.
- Output folder is the tracks CSV's folder or above it, or an existing
  folder that is not an earlier results folder → refused, nothing
  touched.
- Tracking sidecar says `cancelled: true` → a warning; analysis runs.
- A per-track MSD fit failure → NaN and `unclassified`, never a run
  failure. A per-track step-detection failure → NaN, never a run failure.
- Empty tracks table → refused ("no tracks").

### 5. Testing

Synthetic tracks with known answers, no real data:

- 3D Brownian walk with known D (many long tracks): `alpha_3d ≈ 1` and
  `D_3d ≈ D` within tolerance; same for `xy`.
- Directed (constant-velocity) track → `alpha > 1.2`, `directed`.
- Stationary-with-jitter track → `constrained`.
- Anisotropic calibration: a z-only step of 1 voxel gives 0.210 µm, an
  x-only step 0.104 µm.
- Gap-bridged track: lags use real frame differences; `n_gap_frames`
  correct.
- Lifetime / censored bookkeeping; `lifetime_distribution` excludes
  censored tracks.
- Short track: NaN D/α, `unclassified`, `msd_lags_fitted = 0`.
- Steps on a synthetic two-level trace with NaN gaps → one step at the
  right frame, dwells in frames; clean steps measured exactly; a
  too-short trace → NaN, not 0.
- 400 short (30-frame) Brownian tracks → at least 75 % diffusive.
- No fit columns → zero intensity points, NaN intensities, NaN steps.
- Worker: progress; cancel writes nothing; error leaves earlier output;
  an existing non-results folder (or stale `.old` / `.partial`) refused
  and untouched.
- Widget (via `_launch_worker` seam): calibration prefill order;
  incomplete-calibration refusal; output-folder refusal; `features`
  joined onto the Tracks layer; cancelled-tracking warning.
- One real-thread cancel test.

## Out of scope

- Everything in 3B: filter panel, interactive charts, per-track plots.
- Image-sampled intensity (decided against; fit intensity only).
- Localisation-error offsets in the MSD model.
- Ensemble (population-averaged) MSD.
- Fixing the low Gaussian-fit convergence on `decon_10`.
