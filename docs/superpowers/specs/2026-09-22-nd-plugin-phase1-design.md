# napari-spotiflow-tracking-nd — Phase 1 design

Status: approved (chat), pending written spec review.
Sibling project: `D:\Code\Lab\spotiflow_tracking` (`napari-spotiflow-tracking`, 2D).
Example dataset: `D:\Data\3d_Tracking\20260904_CLC_DsRed_DKO_Only`.

## Purpose

Port `napari-spotiflow-tracking` to 4D (T, Z, Y, X) volume timeseries. The
2D plugin holds a whole movie in RAM; a 3D timeseries does not fit that
model (the example dataset is 349 timepoints × 155×1024×1019 uint16, one
TIFF per timepoint, ~323 MB/volume, ~110 GB total). Phase 1 covers data
loading, pre-processing, and 3D spot detection. Tracking (Phase 2) and
track-property/step analysis (Phase 3) are scoped here but not built yet.
PAPA is explicitly out of scope — it will not be ported.

## Example dataset facts (drive defaults, not hard-coded behavior)

- One TIFF per timepoint, `ZYX` axes, `(155, 1024, 1019)`, `uint16`, no
  OME/ImageJ metadata (no pixel size or interval on disk).
- Calibration supplied by the user: **xy pixel size = 0.104 µm, z step =
  0.210 µm, imaging frequency = 3 s/volume.** These are per-dataset values
  a user enters in the widget, not constants — the example dataset just
  supplies the numbers to prefill/test with.
- GPU: 17 GB VRAM available; one float32 volume is ~650 MB, comfortably
  fits for per-volume torch ops.
- Spotiflow ships two pretrained 3D-capable models: `synth_3d`,
  `smfish_3d` (`is_3d=True` in its registry) — 3D detection works without
  training a new model first.

## Data model — lazy, per-volume, TIFF-native

A new `_io3d.py` builds a lazy 4D view over a folder of per-timepoint
TIFFs: one dask block per timepoint via
`dask.array.from_delayed(tifffile.imread(path), shape=(Z,Y,X), dtype)`,
stacked on a new leading T axis. This becomes a single lazy
`napari.layers.Image` the user can scrub through (T slider, 3D render
toggle) without loading the series into RAM. `shape`/`dtype` for the
delayed blocks come from probing the first file; a mismatched later file
raises rather than silently reshaping.

**Calibration → layer scale.** The widget takes `pixel_size_xy_um`,
`voxel_size_z_um`, and `frame_interval_s` as inputs (all optional; `None`
until the user sets them). When both spatial values are set, the loaded
layer's `scale` is `(1.0, voxel_size_z_um, pixel_size_xy_um,
pixel_size_xy_um)` (T left at 1 — napari does not scale playback) so 3D
rendering has the correct aspect ratio. `frame_interval_s` is not a layer
property; it is threaded through to per-run metadata/CSV output for later
phases (physical-unit track properties).

## Pre-processing — stream volume-by-volume, write sibling TIFFs

`_preprocessing3d.py` ports three of the five 2D operations, chosen to
cover this round (crop, N2V, walking average, and morphological-erosion
background removal are deferred, not deleted):

- `remove_background_high_pass_3d(volume, sigma, device)` — separable 3D
  Gaussian via `torch.conv3d` (falls back to `scipy.ndimage.gaussian_filter`
  3D on CPU when torch is unavailable), reusing the existing
  `_rebaseline` dtype-clip logic unchanged. `sigma` accepts a single float
  (isotropic) or a `(sigma_z, sigma_y, sigma_x)` tuple, since z and xy
  sampling differ (0.210 vs 0.104 µm here).
- `gaussian_blur_3d(volume, sigma)` — `scipy.ndimage.gaussian_filter`
  applied directly in 3D (no per-frame loop; a volume *is* the unit now).
  Same `sigma` tuple-or-scalar convention.
- `apply_calculation(image, op, value)` — reused as-is from the 2D
  plugin; it is already dtype-generic and shape-agnostic.

Processing is one timepoint at a time: read one input TIFF → run the
selected op(s) → write one output TIFF, into an output folder that
mirrors the input filenames (e.g. `..._T_000_C_00.tif`). No zarr step —
output is plain per-volume TIFF so a run can be shared by copying the
folder. A `QThread` worker (`_workers3d.py`) drives the loop and reports
progress per timepoint; cancellation stops before the next file starts
(the current file always finishes, so output is never a half-written
TIFF).

## Spot detection — Phase 1, ported

`_segmentation3d.py` wraps Spotiflow for 3D:

- `load_model(name_or_path, device)` — `Spotiflow.from_pretrained(name)`
  for `synth_3d` / `smfish_3d` (or any other registered 3D name), or
  `Spotiflow(...)` loaded from a local checkpoint directory for a
  custom-trained 3D model.
- `detect_spots_3d(model, volume, **predict_kwargs)` — calls
  `model.predict(volume, ...)` per timepoint; `predict_kwargs` passes
  through Spotiflow's own knobs (`prob_thresh`, `min_distance`,
  `exclude_border`, `n_tiles`, etc.) unchanged.

Output is `(t, z, y, x, prob, ...)` rows appended per timepoint and
written to a CSV mirroring the 2D plugin's Points-layer/CSV convention,
plus a Points layer in the viewer (`scale` inherited from the image
layer's calibration so points align with the volume).

## Widgets

Same UI/logic split as the 2D plugin — Qt widgets are thin shells with no
algorithm code, calling the pure logic modules above:

- `_cleanup_widget3d.py` — Pre-Processing: pick input folder, calibration
  fields (xy/z/dt), background removal + Gaussian blur + arithmetic
  controls, output folder, run/cancel via a `QThread` worker.
- `_detection_widget3d.py` — Spot Detection: pick a loaded 4D image layer
  or input folder, model selector (pretrained name or local checkpoint
  path), Spotiflow predict knobs, preview on the current timepoint, run
  full series via worker, export CSV.

Both are registered as dock widgets in a new `napari.yaml` manifest for
the `napari-spotiflow-tracking-nd` package, following the same
`pyproject.toml` / `src/` layout as the 2D plugin (own package name, own
entry point, no shared runtime dependency on the 2D package — logic is
ported/adapted, not imported cross-package).

## Deferred (spec'd as future phases, not built now)

- **Phase 2 — Tracking:** `trackpy.link(f, search_range, pos_columns=['z','y','x'], t_column='frame')` already supports 3D; port `_tracking.py`'s wrapper and the Tracking widget with `pos_columns` fixed to 3D.
- **Phase 3 — Track properties / step analysis:** port `_track_properties.py`, extended for anisotropic 3D calibration — per-axis µm conversion (`dz_um = dz_px * voxel_size_z_um`, `dy_um`/`dx_um = d*_px * pixel_size_xy_um`) combined as a Euclidean 3D displacement, plus `frame_interval_s` for durations/speeds, and the existing step-detection logic ported unchanged (it operates on the physical/pixel columns, not on shape).

## Explicitly out of scope

- PAPA (cycle analysis widget) — not ported.
- N2V denoise, walking average, morphological-erosion background removal,
  Shapes-based crop — not ported this round; can be added later without
  restructuring the above.
- Zarr conversion tooling — pre-processing output is TIFF-in/TIFF-out by
  design, for shareability.
- The `batch/` headless CLI pipeline — may follow the same pattern later
  once the interactive widgets are validated; not part of this spec.

## Testing

Mirrors the 2D plugin's convention: pure-Python logic modules
(`_preprocessing3d.py`, `_segmentation3d.py`, `_io3d.py`) get direct
pytest unit tests with small synthetic volumes (no napari/Qt needed,
no GPU required — CPU fallback paths are what CI exercises). Widget
tests use `pytest-qt` against the same small synthetic volumes, following
the existing `test_cleanup_widget.py` / `test_detection_widget.py`
pattern. No test depends on the real 110 GB dataset; a fixture builds a
tiny multi-timepoint folder (e.g. 3 timepoints × 8×32×32) on the fly.
