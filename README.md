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
  rather than disappearing.
- **Label masks (optional)** — each fitted spot painted as an ellipsoid
  out to 2σ, one label per spot. Shown as a Labels layer when you
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

Deferred to later phases: tracking (trackpy, 3D), track-property/step
analysis, PAPA (not planned), N2V denoise, walking average, erosion
background removal, Shapes-based crop.

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

## Tests

```bash
pytest   # testpaths = src/napari_spotiflow_tracking_nd/_tests
```

No test requires a GPU, real Spotiflow weights, or the real dataset —
fixtures build tiny synthetic per-timepoint TIFF folders on the fly.
