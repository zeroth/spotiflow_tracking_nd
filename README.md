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
- **Spot detection** — [Spotiflow](https://github.com/weigertlab/spotiflow)
  3D models (`synth_3d`, `smfish_3d`, or a custom checkpoint), run per
  volume, exported to a CSV (`frame, z, y, x, prob`) and loaded back as a
  Points layer. The calibration in force for the run is written beside the
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
pip install -e ".[testing]"
pip install "napari[all]"   # if napari + a Qt backend aren't already present

napari   # then open Plugins → Spotiflow ND Tracking → <widget>
```

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
