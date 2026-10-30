# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A napari (npe2) plugin for spot detection and tracking in **3D-volume timeseries too large for RAM**: one `(Z, Y, X)` TIFF per timepoint, streamed one volume at a time. It is a port of the sibling 2D plugin napari-spotiflow-tracking (whole movie in RAM); `find_steps` in `_track_analysis3d.py` is copied from there. The README documents user-facing behaviour and the scientific conventions (lifetime, MSD fit window, motility thresholds, step boundaries) in detail; read it before changing analysis output.

## Commands

```bash
pip install -e ".[testing]"          # napari + PyQt6 + pytest-qt
pip install -e ".[dev]"              # ruff, pre-commit
pytest                               # testpaths = src/napari_spotiflow_tracking_nd/_tests
pytest src/napari_spotiflow_tracking_nd/_tests/test_tracking3d.py::test_name
ruff check src
napari path/to/timepoint_folder      # opens through the plugin's reader
```

Tests need no GPU, Spotiflow weights or real data: `conftest.py` writes tiny synthetic timepoint folders (`write_timepoint_folder`) and adds them to a viewer the way the reader would (`add_layer_from_folder`).

Environment pitfalls (Windows): use **PyQt6, not PyQt5** (PyQt5's Qt DLLs break `import torch` with WinError 1114 on `c10.dll`); install `torch` and `torchvision` together from the CUDA index (`--index-url https://download.pytorch.org/whl/cu126`), uninstalling a CPU build first since pip won't replace it.

## Architecture

Pipeline: **reader → Pre-Processing → Spot Detection → Spot Tracking → Track Analysis**, each stage a widget registered in `napari.yaml`. Stages hand off through **files on disk plus layer metadata**, not in-memory state, so each widget can also start from a file chosen with Browse…

Layering — keep it:
- **Pure logic, no Qt/napari**: `_io3d.py` (timepoint file listing by the `_T_<n>_` filename token, lazy dask stacks, hyperstack splitting, scale helpers), `_preprocessing3d.py` (torch with scipy fallback, identical kernels), `_segmentation3d.py` (Spotiflow model loading/detection), `_fitting3d.py` (per-spot 3D Gaussian fit, label masks), `_tracking3d.py` (trackpy linking), `_track_analysis3d.py` (per-track properties, MSD, steps).
- **Workers** (`_workers3d.py`): one `QThread` subclass per long job (`VolumeOpWorker3D`, `DetectionWorker3D`, `TrackingWorker3D`, `TrackAnalysisWorker3D`) plus `FunctionWorker3D` for single-call previews. All share the signals `progress(int, int)`, `run_finished(...)`, `errored(str)` and a cooperative `cancel()` checked between volumes/frames. Gaussian fitting fans out to a `ProcessPoolExecutor`.
- **Widgets** (`_*_widget3d.py`): build the UI, validate inputs, launch workers through `RunStateMixin` (`_run_state.py`), and turn results into layers.

Cross-cutting conventions:
- **One job process-wide.** `_run_state._ACTIVE_RUN` enforces a single running job across *all* widgets (they share the GPU). Use `RunStateMixin._begin_run` for any new background job; it handles busy state, progress, Cancel and safe teardown on close.
- **Cancel keeps partial work** for pre-processing, detection and tracking (layer named with a `[cancelled n/N]` suffix); track analysis writes nothing on cancel. Previews run one timepoint in memory and are not interruptible.
- **Calibration** lives only in layer metadata under `CALIBRATION_KEYS = ("pixel_size_xy_um", "voxel_size_z_um", "frame_interval_s")` (defined in `_detection_widget3d.py`). The reader opens at unit scale with these keys `None`; Pre-Processing's Apply Calibration fills them. Never hard-code dataset values.
- **JSON sidecars** next to each CSV (`spots.csv` → `spots.json`, `<name>_tracks.json`, `analysis.json`) record the calibration and parameters used; downstream widgets prefill from them.
- **Coordinates stay in voxels**; physical units are added as extra columns (`*_um`, `time_s`), never by rewriting coordinates. Detected `z, y, x` are never moved by fitting (fit centre goes in `fit_z/y/x`); tracking links on `link_z/y/x`, chosen per row and recorded in `coord_source`.
- **Anisotropic voxels**: σ and tracking search range are in voxels; "Link to calibration" derives the z value as `xy × (pixel_size_xy_um / voxel_size_z_um)` and refuses to run without calibration rather than falling back.
- **Never lose data silently**: failed fits keep their row (`converged=False`, measurement columns `NaN`); missing values are blank, not 0; outputs that would overwrite the input folder, or a non-results folder, are refused.
- Don't import spotiflow/torch at widget construction — it is slow and is deferred until a model is actually loaded.

Design specs and implementation plans for each feature are in `docs/superpowers/specs/` and `docs/superpowers/plans/`.
