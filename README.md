# napari-spotiflow-tracking-nd

A [napari](https://napari.org) plugin for **3D-volume timeseries** spot
detection and tracking, built for datasets too large to hold in memory:
one TIFF per timepoint, `(Z, Y, X)` per file, streamed one volume at a
time.

Sibling project: [napari-spotiflow-tracking](https://github.com/) — the
2D (single-plane, whole-movie-in-RAM) version this plugin is ported from.

## Status: Phase 1

- **Data loading** — a folder of per-timepoint TIFFs opens as a single
  lazy `(T, Z, Y, X)` image layer (`dask`-backed; nothing is loaded until
  you scrub to it).
- **Pre-processing** — background removal (3D high-pass), Gaussian blur,
  and add/subtract arithmetic, each streamed volume-by-volume and written
  to an output folder of per-timepoint TIFFs (same filenames as the
  input — no zarr).
- **Spot detection** — [Spotiflow](https://github.com/weigertlab/spotiflow)
  3D models (`synth_3d`, `smfish_3d`, or a custom checkpoint), run per
  volume, exported to a CSV (`frame, z, y, x, prob`) and loaded back as a
  Points layer.

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

Pixel size (xy, µm) and voxel size (z, µm) are entered when loading a
folder in the Pre-Processing widget; they set the loaded layer's `scale`
so 3D rendering has the correct aspect ratio. Frame interval (s) is
carried in layer metadata for later phases. None of these are hard-coded
— enter the values for your own acquisition.

## Tests

```bash
pytest   # testpaths = src/napari_spotiflow_tracking_nd/_tests
```

No test requires a GPU, real Spotiflow weights, or the real dataset —
fixtures build tiny synthetic per-timepoint TIFF folders on the fly.
