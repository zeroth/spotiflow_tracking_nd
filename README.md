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
- **Spot detection** — [Spotiflow](https://github.com/weigertlab/spotiflow)
  3D models, picked from a drop-down of the built-ins (`synth_3d`,
  `smfish_3d`) or set to a custom checkpoint folder (the drop-down is
  editable; **Browse…** fills it in). Runs per volume, exported to a CSV
  (`frame, z, y, x, prob`) and loaded back as a Points layer. The calibration in force for the run is written beside the
  CSV as a JSON sidecar (`spots.csv` → `spots.json`) and carried on the
  Points layer's metadata.

> **A run blocks the napari window for its full duration.** Phase 1
> deliberately runs each pre-processing / detection job synchronously
> (`worker.run()`, not `worker.start()`), so napari stops repainting until
> the last timepoint is written — on a large acquisition that can be many
> minutes. It is working, not hung; watch the output folder (or the CSV)
> growing. Background execution with a live progress bar and a working
> Cancel button is Phase 2.

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

## Tests

```bash
pytest   # testpaths = src/napari_spotiflow_tracking_nd/_tests
```

No test requires a GPU, real Spotiflow weights, or the real dataset —
fixtures build tiny synthetic per-timepoint TIFF folders on the fly.
