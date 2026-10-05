# napari-spotiflow-tracking-nd

A [napari](https://napari.org) plugin for **3D-volume timeseries** spot
detection and tracking, built for datasets too large to hold in memory:
one TIFF per timepoint, `(Z, Y, X)` per file, streamed one volume at a
time.

Sibling project: napari-spotiflow-tracking — the 2D (single-plane,
whole-movie-in-RAM) version this plugin is ported from.

## Status: Phase 1

- **Data loading** — a folder of per-timepoint TIFFs opens as a single
  lazy `(T, Z, Y, X)` image layer (`dask`-backed; nothing is loaded until
  you scrub to it) via a napari **reader**: `File > Open Folder…`,
  drag-and-drop, `napari path/to/folder`, or opening a single file inside
  the series all work with no widget interaction. Because a reader can't
  prompt for input, the layer opens at unit scale; use the Pre-Processing
  widget's **Apply Calibration** action afterward to set pixel size, voxel
  size and frame interval on it.
- **Pre-processing** — background removal (3D high-pass), Gaussian blur,
  and add/subtract arithmetic, each streamed volume-by-volume and written
  to an output folder of per-timepoint TIFFs (same filenames as the
  input — no zarr).
- **Quick mode (Preview)** — every operation, and detection, has a
  **Preview** button that runs on the **currently displayed timepoint
  only**, in memory: one file read instead of the whole series, nothing
  written to disk, and no output folder or CSV path required. Tune sigma
  / `prob_thresh` / `min_distance` against one volume, then commit to the
  full run. Repeated previews update a single layer rather than stacking
  one per click, and the detection model stays loaded between previews.
  Previews run on a worker thread too, so the UI stays live while one is
  in flight.
- **Spot detection** — [Spotiflow](https://github.com/weigertlab/spotiflow)
  3D models, picked from a drop-down of the built-ins (`synth_3d`,
  `smfish_3d`) or set to a custom checkpoint folder (the drop-down is
  editable; **Browse…** fills it in). Runs per volume, exported to a CSV
  (`frame, z, y, x, prob`) and loaded back as a Points layer. The calibration in force for the run is written beside the
  CSV as a JSON sidecar (`spots.csv` → `spots.json`) and carried on the
  Points layer's metadata.
- **Gaussian fitting (optional)** — fits a 3D Gaussian to each detected
  spot and adds its refined centre, per-axis σ, amplitude, local
  background and analytic integrated intensity to the spots CSV. Roughly
  12 ms per spot, parallelised across processes: about 28 minutes for a
  dense 349-timepoint series, so it is off by default and the threshold
  tuning you do with **Preview** stays fast. `z, y, x` remain exactly as
  detected — the fit's centre is added as `fit_z, fit_y, fit_x`, so
  turning fitting on never moves coordinates under anything downstream.
  A spot whose fit fails, or which sits too close to a volume edge to cut
  a full patch, keeps its row with `converged=False` and a fallback σ
  rather than disappearing; its `amplitude`, `background` and
  `integrated_intensity` are `NaN`, because nothing was measured — they
  drop out of a mean instead of contaminating it. A fit that slides more
  than two voxels off the detected centre is rejected the same way: on
  crowded data an unbounded fit will otherwise migrate onto a brighter
  neighbour and report *its* amplitude under this spot's identity.
- **Label masks (optional)** — each fitted spot painted as an ellipsoid
  out to 2σ, one label per spot. Needs Gaussian fitting on, since that is
  what the ellipsoids come from; the run is refused rather than writing a
  series of empty volumes. The mask folder may not be the input folder —
  masks are written under the input filenames, so that would overwrite
  the acquisition. Shown as a Labels layer when you
  **Preview** with fitting on; written to disk as one label TIFF per
  timepoint only when you ask, because a label volume is the same size as
  the data (a 349-timepoint series is ~113 GB). The masks are
  reconstructible from the CSV's fit columns, which is why the parameters
  are what's always saved.

> **Runs happen in the background.** Pre-processing and detection run on a
> worker thread, so napari stays responsive: the progress bar shows
> `120 / 349` as it goes, and **Cancel** stops the run after the volume
> currently being processed. A cancelled run keeps everything already
> written and loads it as a layer labelled `[cancelled 120/349]` —
> nothing is discarded. A *preview* is a single uninterruptible call, so
> cancelling one discards its result rather than stopping the work.

- **Spot Tracking** — links the detection CSV into tracks with trackpy,
  frame by frame, in the background (progress, Cancel keeps the frames
  linked so far). Pick a spots layer from detection, or **Browse CSV…**
  to track a file without loading it. Links on the Gaussian-fit centre
  where the fit converged (**Use fitted centres**), the detected centre
  elsewhere; `coord_source` records which. Writes `<name>_tracks.csv` —
  every detection column plus `link_z/y/x`, `particle` (track ID),
  `censored` (the track starts or ends within `memory` frames of the
  first or last frame, so its true lifetime is unknown) and, where calibrated, `z_um, y_um, x_um,
  time_s` — and a `<name>_tracks.json` sidecar with the calibration and
  every parameter used, plus a `cancelled` flag and the first and last frame
  linked (so a stopped run is recognisable on disk). Shown as a Tracks layer
  at the spots layer's scale.

- **Track Analysis** — turns a tracks CSV into per-track numbers, in the
  background (progress; Cancel writes nothing). Pick a Tracks layer from
  Spot Tracking, or **Browse CSV…**. Calibration (xy, z, dt) is required
  and prefills from the tracks sidecar. Writes a results folder
  (`<tracks>_analysis/`) with `track_properties.csv` (one row per track:
  lifetime, censored, path length, net displacement, mean speed,
  `D_3d`/`alpha_3d` and `D_xy`/`alpha_xy`, motility class, intensity and
  step counts), `lifetime_distribution.csv` (uncensored tracks only),
  `msd_curves.csv`, `steps.csv` and `analysis.json` (every parameter).
  The per-track columns are also attached to the Tracks layer, so its
  colour-by can show e.g. `lifetime_s` or `motility_class`.

Deferred to later phases: interactive track exploration (filter panel,
per-track charts), PAPA (not planned), N2V denoise, walking average,
erosion background removal, Shapes-based crop.

## Install

```bash
pip install -e ".[testing]"   # brings napari + PyQt6

# GPU build of torch — do this explicitly; the default PyPI wheel is
# CPU-only, which makes the widgets' "Use GPU" checkboxes useless.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

napari                        # Plugins → Spotiflow ND Tracking → <widget>
```

Check the GPU is actually live before a long run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# want: 2.x.y+cu126 True   —   a bare '2.x.y' or '+cpu' means CPU-only
```

> If a CPU build is already installed, `pip install torch --index-url …`
> **silently does nothing** — pip treats the existing version as
> satisfying the unpinned requirement and never notices `+cpu` vs
> `+cu126`. Uninstall first (`pip uninstall torch torchvision -y`), and
> close any napari/python process using the venv, or the reinstall fails
> with `Access is denied` on `torch\_C.pyd` and leaves a broken install.

> **Use PyQt6, not PyQt5.** On Windows, PyQt5's bundled Qt DLLs collide
> with torch's: because napari loads Qt before any plugin code runs, the
> first `import torch` (i.e. the first model load) then dies with
> `OSError: [WinError 1114] ... Error loading "torch\lib\c10.dll"`. PyQt6
> has no such conflict. If you hit that error, check with
> `python -c "import qtpy; print(qtpy.API_NAME)"` — if it says PyQt5,
> `pip uninstall PyQt5 PyQt5-Qt5 PyQt5_sip` and `pip install PyQt6`
> (qtpy prefers PyQt5 whenever it is installed, so removing it is more
> reliable than setting `QT_API`).
>
> `torchvision` must also be installed — Spotiflow imports it at module
> load, so without it `spotiflow.model` fails with
> `ModuleNotFoundError: No module named 'torchvision'`. Install it from
> the same index as torch so the pair matches (e.g. torch 2.14.0+cu126 ↔
> torchvision 0.29.0+cu126); a mismatched pair fails with
> `ImportError: DLL load failed while importing _C`.

## Calibration

Pixel size (xy, µm) and voxel size (z, µm) are entered in the
Pre-Processing widget's **Apply Calibration** action, which sets the
selected layer's `scale` (for correct 3D aspect ratio) and its metadata —
works on a layer opened through the reader, or built any other way, as
long as it carries the same metadata keys. Frame interval (s) is carried
in layer metadata for later phases. None of these are hard-coded — enter
the values for your own acquisition.

### Link σ to calibration

σ for background removal and Gaussian blur is measured in **voxels**, and
voxels are rarely cubes — at 0.104 µm laterally against 0.210 µm axially, a
σ of 4 spans 0.416 µm across x/y but 0.840 µm along z. The filter is then
isotropic in array indices and twice as wide in z in physical space, which
smears spots axially and leaves axial background under-subtracted.

Tick **Link σ to calibration** and the σ you type stays the lateral value
while the axial one is derived:

```
σ_z = σ_xy × (pixel_size_xy_um / voxel_size_z_um)
```

so σ = 4 is applied as `(1.98, 4, 4)` — the same 0.416 µm on every axis.
The applied value appears in the output layer's name, e.g.
`vol (blur s=(1.98, 4, 4))`. Linking needs a calibrated layer: with the box
ticked and no calibration, the run is refused rather than silently falling
back to isotropic voxels.

Isotropic-in-µm is not automatically the right target — axial optical
resolution is typically 2–3× worse than lateral, so a filter matched to the
PSF's own anisotropy is also defensible. Leave the box unticked for that.

### Tracking search range

Linking runs in voxels, so the search range has the same problem as σ: 4
voxels is 0.416 µm across x/y but 0.840 µm along z, and an isotropic
4-voxel range lets a spot jump twice as far axially as laterally. Tick
**Link to calibration** and the z range is derived from the xy one,

```
z = xy × (pixel_size_xy_um / voxel_size_z_um)
```

so 4 px becomes `(1.98, 4, 4)` — the same 0.416 µm on every axis. Unticked,
the z range is whatever you type. Linked with blank calibration, the run
is refused. The calibration fields prefill from the spots layer, then the
detection CSV's sidecar; fill them in by hand for a series detected
before calibration was applied.

**Memory** is counted in timepoints, including timepoints with no spots
at all. **Min track length** defaults to 1 — short-lived structures are
data for lifetime analysis, not noise to drop by default.

### Track analysis conventions

- **Lifetime** counts frames inclusively — a structure seen in frames
  10–14 lives 5 frames (`lifetime_s = 5 × dt`). Censored tracks (within
  `memory` frames of either end of the series) keep their row but are
  left out of `lifetime_distribution.csv`.
- **MSD** is computed per track in µm on the real frame grid — frames
  bridged by tracking `memory` are gaps, not neighbours — for lags 1 to
  **Max lag**. `MSD = 6·D·t^α` is fitted in 3D and `MSD = 4·D·t^α` in xy,
  only for tracks with at least **Min spots for MSD** spots and 4 usable
  lags. **Motility from** picks which α sets the class: `α < 0.4`
  constrained, `α > 1.2` directed, otherwise diffusive.
- **Intensity and steps** use the Gaussian-fit `integrated_intensity` of
  spots whose fit converged; failed fits are gaps, never interpolated. A
  trace shorter than `2 × Step window + 1` points is not assessed
  (`n_steps` is blank, not 0).

## Tests

```bash
pytest   # testpaths = src/napari_spotiflow_tracking_nd/_tests
```

No test requires a GPU, real Spotiflow weights, or the real dataset —
fixtures build tiny synthetic per-timepoint TIFF folders on the fly.
