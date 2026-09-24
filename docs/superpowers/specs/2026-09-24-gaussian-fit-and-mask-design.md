# 3D Gaussian spot fitting and labelled masks — design

Status: approved in chat, pending written spec review.
Follows: `2026-09-22-nd-plugin-phase1-design.md` (detection), which scoped
masks out of Phase 1.

## Purpose

Detection currently yields a spot's position and Spotiflow's confidence,
and nothing about the spot itself. Fitting a 3D Gaussian to each detection
adds its size, brightness and local background, which serves three uses
named by the user:

- **per-spot intensity measurement**, exported alongside the detections;
- **a label volume exported as TIFF**, on demand, shareable like the
  pre-processed data;
- **a visual check** in napari that detection is picking up real puncta.

## Measured constraints

Numbers taken from the user's own data and detection output, not estimated.

| Source | Spots / timepoint (median) | Timepoints | Total fits |
|---|---|---|---|
| `endo.csv` (deskewed) | 117 (max 1,371) | 349 | ~106k |
| `endo2.csv` (deconvolved) | 3,101 (max 3,160) | 10 run, 349 available | ~1.08M |

Deconvolution raises detections ~26×, so the deconvolved series is the
case the design must survive.

A single `scipy.optimize.curve_fit` of an 8-parameter 3D Gaussian over a
7×11×11 patch, timed against `decon_10` frame 0: **12.5 ms**, with
**92.6% convergence** (275 of 297; 22 failed). Extrapolated:

| Dataset | 1 core | 8 processes |
|---|---|---|
| deskew, 41k fits | 8 min | ~1 min |
| decon, 1.08M fits | 3.7 h | **28 min** |

**This settles the backend.** scipy plus `ProcessPoolExecutor` is
sufficient; 28 minutes is acceptable for an offline step. `torchimize` —
the 2D plugin's GPU fitting backend — is not installed here and is not
needed, so no new dependency and no port of that code. A GPU path remains
possible later if the numbers change.

The ~7% failure rate is not an edge case: on the deconvolved series it is
roughly 80,000 spots. The fallback path is a main path.

**Storage.** A uint16 label volume is 323 MB per timepoint — 113 GB for
349, more than doubling a 110 GB dataset, to store what nine numbers per
spot already determine. A fitted Gaussian is analytic; the mask is
reconstructible from its parameters. Hence: parameters always, label TIFFs
only on request.

## Approach

Fitting runs **inside the detection pass**, behind a checkbox. Detection
already holds each volume in memory, so fitting reuses it: one pass over
110 GB instead of two. The checkbox keeps threshold-tuning runs fast.

Rejected: a separate "Fit & Mask" action on an existing Points layer (the
2D plugin's shape) — it needs a second full read of every volume, ~9
minutes of pure I/O on this dataset before any fitting, and re-fitting
with a different patch radius is not a frequent enough need to pay that.
It stays additive if it ever is.

## Design

### 1. `_fitting3d.py` — logic, no Qt

```python
@dataclass
class SpotFit3D:
    z0: float; y0: float; x0: float          # refined centre, volume coords
    sigma_z: float; sigma_y: float; sigma_x: float
    amplitude: float                          # peak above background
    background: float
    converged: bool                           # False => fallback values

def fit_spots_3d(volume, points, patch_radius_zyx=(3, 5, 5),
                 max_workers=None) -> list[SpotFit3D]

def paint_label_mask(shape, points, fits, n_sigma=2.0,
                     fallback_radius_zyx=(1.5, 2.5, 2.5)) -> np.ndarray  # uint16
#   fallback_radius_zyx is in voxels, anisotropic in the same ratio as
#   the patch; it is what a failed fit is painted with.
```

Patches are anisotropic by default — `(3, 5, 5)`, i.e. 7 z-slices by 11 by
11 — because z is sampled at 0.210 µm against 0.104 µm laterally. This is
the geometry the 12.5 ms figure was measured against.

Two deliberate departures from the 2D plugin, both consequences of 3D
scale:

- **Worker processes receive patches, not the volume.** `fit_and_mask_2d`
  builds `args_list = [(image, centre, radius), ...]` — one copy of the
  whole image per spot. At 323 MB × 3,101 spots that cannot work.
  Extracting patches in the parent first makes it ~6.8 KB × 3,101 ≈ 21 MB
  per timepoint.
- **One process pool per run, not per timepoint.** Windows spawns rather
  than forks, at roughly 0.5–1 s per process; a pool per volume would
  spend ~6 minutes across 349 timepoints doing nothing but starting
  processes. The pool is created once in the worker and passed down.

A spot whose fit fails — or that sits too close to a volume edge for a
full patch, where no fit is attempted at all — keeps its row with
`converged=False` and the fallback σ. A detection never disappears from
the CSV because its fit did not work out.

### 2. Measurements are analytic, not regionprops

The fit yields integrated intensity in closed form:

```
integrated_intensity = amplitude · (2π)^(3/2) · σz · σy · σx
```

So an ordinary run pays only for fitting: no 323 MB label volume has to be
painted merely to measure. Painting happens only for the overlay or an
export.

Columns added to the spots CSV, after the existing `frame, z, y, x, prob`:

```
fit_z, fit_y, fit_x, sigma_z, sigma_y, sigma_x,
amplitude, background, integrated_intensity, converged
```

`z, y, x` stay exactly as Spotiflow detected them; the refined centre is
`fit_z, fit_y, fit_x` beside them. Nothing downstream — Phase 2 tracking
in particular — has its coordinates shifted underneath it by turning
fitting on.

All in **voxel units**, consistent with the existing rule that the CSV is
pixel/frame space and the JSON sidecar carries the calibration needed to
convert. Physical units remain Phase 3's job.

The cost of this choice, stated plainly: there is no "mean intensity
within the segmented region" column of the kind the 2D plugin's regionprops
export produces. `amplitude` (background-subtracted peak) and
`integrated_intensity` replace it.

### 3. Mask geometry

One label per spot, `i + 1`, painted as the ellipsoid

```
((z-z0)/(n_sigma·σz))² + ((y-y0)/(n_sigma·σy))² + ((x-x0)/(n_sigma·σx))² ≤ 1
```

with `n_sigma = 2.0` by default (~95% of each axis). Spots are painted in
order, so where two overlap the later label wins — the same rule the 2D
plugin uses. Failed fits get a fixed fallback ellipsoid rather than being
skipped, so every spot is represented.

### 4. Worker and widget

`DetectionWorker3D` gains `fit: bool`, `patch_radius_zyx`, and
`mask_output_dir: Path | None` — additive, preserving the single pass and
the existing signature for callers that do not fit. When
`mask_output_dir` is set it writes one uint16 label TIFF per timepoint,
named to mirror the input, exactly as `VolumeOpWorker3D` does.

`DetectionWidget3D` gains a **Fit Gaussians to spots** checkbox, two
patch-radius spin boxes (z and xy), and an **Export label masks**
checkbox with its own output folder. The existing output-folder guard
applies: the mask folder may not be the input folder.

Preview with fitting enabled adds a Labels layer for the previewed
timepoint — the visual check, in memory, nothing written.

### 5. Testing

The test that earns its place: **synthesise a volume containing Gaussians
at known centres with known anisotropic σ, then assert the fit recovers
them** within tolerance. That proves the fitting works rather than merely
runs, and it is the only test that would catch a wrong model function, a
transposed axis, or a bad initial guess.

Alongside it:

- a flat/noise patch yields `converged=False` and a fallback mask, since
  ~7% of real spots take this path;
- analytic `integrated_intensity` agrees with a numeric sum over a
  synthesised spot;
- the mask has one label per spot, is anisotropic in the direction the σ
  say, and represents failed fits too;
- the CSV gains exactly the named columns, and only when fitting is on;
- mask TIFFs appear only when a mask folder is given;
- a spot too close to an edge for a full patch keeps its row with
  `converged=False` and the fallback σ — no fit is attempted, nothing
  raises, and the detection is not dropped.

Tests pass `max_workers=1` so no processes spawn under pytest; the
multiprocessing path itself is covered by one test asserting that what
crosses the process boundary is patches, not the volume.

## Risks

- **Process pool inside a QThread inside napari.** Spawned children must
  import the module cleanly; the fitting entry point therefore stays a
  module-level function with picklable arguments. If this proves fragile
  on Windows, the fallback is `max_workers=1`, which costs 3.7 h on the
  deconvolved series — usable but unattractive.
- **Initial-guess sensitivity.** 12.5 ms and 92.6% were measured with
  `p0 = (0, 0, 0, 1.5, 2.0, 2.0, max-min, min)` on deconvolved data.
  Deskewed (non-deconvolved) spots are dimmer and broader; convergence
  there is unmeasured and may differ.
- **Fitting silently changes what a "spot" is.** The fitted centre differs
  from Spotiflow's. The CSV keeps `z, y, x` as detected and adds the
  fitted centre separately, so nothing downstream shifts underneath
  Phase 2 tracking.

## Out of scope

- GPU/`torchimize` fitting.
- A standalone "fit an existing Points layer" action.
- Physical-unit conversion of σ (Phase 3, from the sidecar).
- regionprops-style measurements over the painted mask.
