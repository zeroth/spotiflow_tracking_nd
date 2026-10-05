# Phase 3A — 3D Track Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A *Track Analysis* napari widget that turns the Spot Tracking widget's tracks CSV into per-track lifetimes, geometry, 3D and xy MSD fits, motility classes and intensity steps, written as a folder of CSVs and attached to the Tracks layer's `features`.

**Architecture:** Pure per-track analysis in `_track_analysis3d.py` (numpy/pandas/scipy, no Qt); a `TrackAnalysisWorker3D` QThread in `_workers3d.py` that analyses one track at a time with progress and cancel and writes the results folder atomically; a `TrackAnalysisWidget3D` driven by the existing `RunStateMixin`, mirroring the Spot Tracking widget's input/calibration patterns.

**Tech Stack:** numpy, pandas, scipy (`curve_fit`, `gaussian_filter1d`), qtpy, napari 0.9, pytest + pytest-qt.

**Spec:** `docs/superpowers/specs/2026-10-05-track-analysis-3d-design.md`

## Global Constraints

- Calibration is **required**: `pixel_size_xy_um`, `voxel_size_z_um`, `frame_interval_s`; no voxel-unit fallback. Positions in µm: `z = link_z × voxel_size_z_um`, `y = link_y × pixel_size_xy_um`, `x = link_x × pixel_size_xy_um`.
- Each track is laid on its full frame span with NaN rows for gap frames; MSD lags are real frame differences.
- `lifetime_frames = last_frame − first_frame + 1`; `lifetime_s = lifetime_frames × frame_interval_s`.
- MSD fit `k·D·t^α`, `k = 6` (3d: z, y, x) and `k = 4` (xy: y, x), `t = lag × frame_interval_s`, `p0 = (0.001, 0.01)`, only when `n_spots ≥ min_msd_spots` (default 10) and `≥ 4` usable lags; otherwise NaN. Lags `1..min(max_lag, span − 1)` (default `max_lag` 25); a lag with zero pairs is omitted.
- Motility: `alpha < 0.4` constrained, `alpha > 1.2` directed, else diffusive (boundaries diffusive), NaN → `unclassified`; from `alpha_3d` or `alpha_xy` per `motility_from` (default `"3d"`).
- Intensity trace: `integrated_intensity` of spots with `converged == True` and a finite value, frame order, gaps skipped, never interpolated.
- Steps: the 2D `find_steps` ported unchanged; defaults `step_window = 3`, `step_threshold = 0.5`; a trace shorter than `2 × window + 1` → `n_steps = NaN`, `mean_step_height = NaN`.
- `lifetime_distribution.csv` counts **uncensored** tracks only.
- Per-track failures (fit, steps) become NaN, never a run failure.
- Cancel **writes nothing**. Output goes to `<folder>.partial/`, renamed into place on success; on error `.partial` is removed and earlier results are untouched.
- Output files: `track_properties.csv`, `lifetime_distribution.csv`, `msd_curves.csv`, `steps.csv`, `analysis.json`.
- Tests: no GPU, no real data. Widget tests run workers synchronously via the `_launch_worker` seam; only `test_threading.py` uses real threads.
- Run commands from the repo root `D:\Code\Lab\spotiflow_tracking_nd` with `.venv/Scripts/python -m pytest …`. 4-space indentation.

## Review Focus

Inputs the spec implies but its listed tests would not exercise — each has a pinning test in the owning task:

1. **Tracks from a cancelled tracking run** (`cancelled: true` in the tracks sidecar) — the user must be warned, and `analysis.json` must say so. → Task 4, `test_tracks_from_a_cancelled_tracking_run_warn_and_are_recorded`.
2. **A single-spot track** — path 0, net 0, speed NaN, D/α NaN, `unclassified`, no exception. → Task 2, `test_a_single_spot_track_is_all_nan_motion_and_no_crash`.
3. **Re-running into an existing results folder** — the old folder's files must not mix with the new ones. → Task 3, `test_a_rerun_replaces_the_previous_results_folder`.
4. **An output path that is an existing file** — refused; the file must not be moved or replaced. → Task 4, `test_an_output_path_that_is_a_file_is_refused`.
5. **Mostly-failed fits** (the `decon_10` reality: ~75 % of `converged` False) — sparse traces give NaN steps, never an error. → Task 2, `test_a_mostly_failed_fit_trace_gives_nan_steps`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/napari_spotiflow_tracking_nd/_track_analysis3d.py` (create) | per-track analysis logic, no Qt |
| `src/napari_spotiflow_tracking_nd/_workers3d.py` (modify) | `TrackAnalysisWorker3D`, `replace_dir` |
| `src/napari_spotiflow_tracking_nd/_layer_utils.py` (modify) | gains `calibration_spin` (moved from the tracking widget) |
| `src/napari_spotiflow_tracking_nd/_tracking_widget3d.py` (modify) | uses `calibration_spin`; gains `read_sidecar` |
| `src/napari_spotiflow_tracking_nd/_track_analysis_widget3d.py` (create) | the Track Analysis widget |
| `src/napari_spotiflow_tracking_nd/napari.yaml` (modify) | register the widget |
| `README.md` (modify) | Track Analysis section |
| `src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py` (create) | logic tests |
| `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py` (modify) | worker tests |
| `src/napari_spotiflow_tracking_nd/_tests/test_track_analysis_widget3d.py` (create) | widget tests |
| `src/napari_spotiflow_tracking_nd/_tests/test_threading.py` (modify) | real-thread cancel test |

---

### Task 1: `_track_analysis3d.py` — parameters, positions, MSD and its fit

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_track_analysis3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py`

**Interfaces:**
- Consumes: nothing.
- Produces, in `napari_spotiflow_tracking_nd._track_analysis3d`:
  - constants `REQUIRED_COLUMNS = ("particle", "frame", "link_z", "link_y", "link_x", "censored")`, `ALPHA_CONSTRAINED_MAX = 0.4`, `ALPHA_DIRECTED_MIN = 1.2`, `MOTILITY_CLASSES`, `MIN_FIT_LAGS = 4`, `MSD_DIMS = {"3d": (slice(0, 3), 6.0), "xy": (slice(1, 3), 4.0)}`
  - `@dataclass(frozen=True) AnalysisParams(pixel_size_xy_um, voxel_size_z_um, frame_interval_s, max_lag=25, min_msd_spots=10, motility_from="3d", step_window=3, step_threshold=0.5)`
  - `missing_columns(df) -> list[str]`
  - `classify_motility(alpha) -> str`
  - `positions_um(track: pd.DataFrame, params) -> tuple[np.ndarray, np.ndarray]` — `(frames, pos)`, `pos` shape `(span, 3)` z,y,x µm, NaN for gap frames
  - `msd_table(pos: np.ndarray, max_lag: int) -> pd.DataFrame` — index `lag_frames` (int), columns `msd, sem, n_pairs`
  - `fit_msd(lag_s, msd, k: float) -> tuple[float, float]` — `(D, alpha)` or `(nan, nan)`, never raises

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking_nd._track_analysis3d import AnalysisParams

# The reference dataset's calibration: anisotropic voxels, 3 s per volume.
PARAMS = AnalysisParams(pixel_size_xy_um=0.104, voxel_size_z_um=0.210, frame_interval_s=3.0)


def _track(pos_um, track_id=0, frames=None, censored=False, **extra):
    """A tracks-CSV-shaped frame for one track, from `(z, y, x)` µm
    positions — converted to the voxel `link_*` columns the tracker writes."""
    pos_um = np.asarray(pos_um, dtype=float)
    frames = np.arange(len(pos_um)) if frames is None else np.asarray(frames)
    df = pd.DataFrame({
        "particle": track_id,
        "frame": frames,
        "link_z": pos_um[:, 0] / 0.210,
        "link_y": pos_um[:, 1] / 0.104,
        "link_x": pos_um[:, 2] / 0.104,
        "censored": censored,
    })
    for name, value in extra.items():
        df[name] = value
    return df


class TestMissingColumns:
    def test_a_tracks_table_is_complete(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import missing_columns

        assert missing_columns(_track([[0, 0, 0]])) == []

    def test_names_what_is_missing_in_order(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import missing_columns

        df = pd.DataFrame({"particle": [0], "frame": [0], "link_y": [0.0]})
        assert missing_columns(df) == ["link_z", "link_x", "censored"]


class TestClassifyMotility:
    @pytest.mark.parametrize("alpha, expected", [
        (0.1, "constrained"), (0.4, "diffusive"), (1.0, "diffusive"),
        (1.2, "diffusive"), (1.9, "directed"), (np.nan, "unclassified"),
        (None, "unclassified"),
    ])
    def test_thresholds(self, alpha, expected):
        from napari_spotiflow_tracking_nd._track_analysis3d import classify_motility

        assert classify_motility(alpha) == expected


class TestPositionsUm:
    def test_scales_each_axis_by_its_own_voxel_size(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import positions_um

        track = pd.DataFrame({
            "particle": 0, "frame": [0], "link_z": [1.0], "link_y": [1.0],
            "link_x": [1.0], "censored": False,
        })

        _, pos = positions_um(track, PARAMS)

        assert pos[0].tolist() == pytest.approx([0.210, 0.104, 0.104])

    def test_gap_frames_are_nan_rows(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import positions_um

        frames, pos = positions_um(_track(np.zeros((3, 3)), frames=[4, 5, 7]), PARAMS)

        assert frames.tolist() == [4, 5, 6, 7]
        assert np.isnan(pos[2]).all()
        assert np.isfinite(pos[[0, 1, 3]]).all()


class TestMsdTable:
    def test_a_constant_step_gives_lag_squared_growth(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import msd_table

        pos = np.column_stack([np.zeros(5), np.zeros(5), np.arange(5.0)])

        table = msd_table(pos, max_lag=25)

        assert table.index.tolist() == [1, 2, 3, 4]
        assert table["msd"].tolist() == pytest.approx([1.0, 4.0, 9.0, 16.0])
        assert table["n_pairs"].tolist() == [4, 3, 2, 1]
        assert np.isnan(table["sem"].iloc[-1])  # one pair: no spread

    def test_gaps_drop_out_and_empty_lags_are_omitted(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import msd_table

        # Present at frames 0, 1, 3 only (frame 2 a gap).
        pos = np.array([[0, 0, 0], [0, 0, 1.0], [np.nan] * 3, [0, 0, 3.0]])

        table = msd_table(pos, max_lag=25)

        assert table.index.tolist() == [1, 2, 3]
        assert table["n_pairs"].tolist() == [1, 1, 1]
        assert table["msd"].tolist() == pytest.approx([1.0, 4.0, 9.0])

    def test_max_lag_caps_the_lags(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import msd_table

        pos = np.column_stack([np.zeros(10), np.zeros(10), np.arange(10.0)])

        assert msd_table(pos, max_lag=3).index.tolist() == [1, 2, 3]


class TestFitMsd:
    def test_recovers_d_and_alpha_from_an_exact_curve(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import fit_msd

        t = np.arange(1, 11) * 3.0
        d, alpha = fit_msd(t, 6 * 0.02 * t ** 0.8, k=6.0)

        assert d == pytest.approx(0.02, rel=1e-3)
        assert alpha == pytest.approx(0.8, rel=1e-3)

    def test_a_hopeless_curve_returns_nan_instead_of_raising(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import fit_msd

        d, alpha = fit_msd([3.0, 6.0], [np.nan, np.inf], k=6.0)

        assert np.isnan(d) and np.isnan(alpha)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._track_analysis3d'`.

- [ ] **Step 3: Implement it**

Create `src/napari_spotiflow_tracking_nd/_track_analysis3d.py`:

```python
"""Per-track analysis of 3D tracks — lifetimes, geometry, MSD, intensity
steps. Logic only, no Qt.

Input is the Spot Tracking widget's tracks table, in voxel coordinates
(`link_z/y/x`). Everything here is computed in µm and seconds, so
calibration is required: a distance in raw voxels mixes 0.104 µm lateral
and 0.210 µm axial steps and means nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Columns a tracks CSV must have to be analysed.
REQUIRED_COLUMNS = ("particle", "frame", "link_z", "link_y", "link_x", "censored")

# Motility classification thresholds on the anomalous-diffusion exponent α,
# after Nature Communications Biology 4, 137 (2021), "Motility analysis":
# α < 0.4 constrained, 0.4 < α < 1.2 diffusive, α > 1.2 directed.
# The exact boundary values are assigned to "diffusive".
ALPHA_CONSTRAINED_MAX = 0.4
ALPHA_DIRECTED_MIN = 1.2
MOTILITY_CLASSES = ("constrained", "diffusive", "directed")

#: Fewer usable lags than this and an MSD fit is not attempted.
MIN_FIT_LAGS = 4
#: Position columns (of `positions_um`'s z, y, x) and the `k` in
#: `MSD = k·D·t^α` — twice the number of dimensions.
MSD_DIMS = {"3d": (slice(0, 3), 6.0), "xy": (slice(1, 3), 4.0)}


@dataclass(frozen=True)
class AnalysisParams:
    pixel_size_xy_um: float
    voxel_size_z_um: float
    frame_interval_s: float
    max_lag: int = 25
    min_msd_spots: int = 10
    motility_from: str = "3d"  # "3d" or "xy"
    step_window: int = 3
    step_threshold: float = 0.5


def missing_columns(df: pd.DataFrame) -> list[str]:
    """The `REQUIRED_COLUMNS` that `df` lacks, in order."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def classify_motility(alpha) -> str:
    """Bin an anomalous-diffusion exponent into a motility class, or
    `"unclassified"` when `alpha` is None/NaN (no MSD fit)."""
    if alpha is None:
        return "unclassified"
    a = float(alpha)
    if not np.isfinite(a):
        return "unclassified"
    if a < ALPHA_CONSTRAINED_MAX:
        return "constrained"
    if a > ALPHA_DIRECTED_MIN:
        return "directed"
    return "diffusive"


def positions_um(track: pd.DataFrame, params: AnalysisParams) -> tuple[np.ndarray, np.ndarray]:
    """`(frames, pos)` for one track laid on its full frame span: `frames`
    is `first..last`, `pos` is `(span, 3)` `(z, y, x)` in µm with NaN rows
    for frames tracking `memory` bridged, so a lag is a real frame
    difference rather than a row difference."""
    frames_present = track["frame"].to_numpy(dtype=int)
    first, last = int(frames_present.min()), int(frames_present.max())
    frames = np.arange(first, last + 1)
    pos = np.full((len(frames), 3), np.nan)
    scale = np.array(
        [params.voxel_size_z_um, params.pixel_size_xy_um, params.pixel_size_xy_um]
    )
    pos[frames_present - first] = (
        track[["link_z", "link_y", "link_x"]].to_numpy(dtype=float) * scale
    )
    return frames, pos


def msd_table(pos: np.ndarray, max_lag: int) -> pd.DataFrame:
    """Per-lag MSD of NaN-padded positions: for lag `L`, the mean over every
    pair of present points exactly `L` frames apart of the squared
    distance, with its SEM and pair count. Lags with no pairs are omitted.

    The SEM treats overlapping displacements as independent, so it
    understates the true uncertainty at long lags — indicative only.
    """
    lags, means, sems, counts = [], [], [], []
    for lag in range(1, min(max_lag, len(pos) - 1) + 1):
        diff = pos[lag:] - pos[:-lag]
        squared = np.sum(diff ** 2, axis=1)  # NaN if either end is a gap
        squared = squared[np.isfinite(squared)]
        if squared.size == 0:
            continue
        lags.append(lag)
        means.append(float(squared.mean()))
        counts.append(int(squared.size))
        sems.append(
            float(squared.std(ddof=1) / np.sqrt(squared.size)) if squared.size > 1 else np.nan
        )
    return pd.DataFrame(
        {"msd": means, "sem": sems, "n_pairs": counts},
        index=pd.Index(lags, name="lag_frames", dtype=int),
    )


def fit_msd(lag_s, msd, k: float) -> tuple[float, float]:
    """Fit `msd = k·D·t^α`; `(D, α)`, or `(nan, nan)` when the fit fails or
    returns something non-finite. Never raises."""
    from scipy.optimize import curve_fit

    def _model(t, d, alpha):
        return k * d * np.power(t, alpha)

    try:
        popt, _ = curve_fit(
            _model, np.asarray(lag_s, dtype=float), np.asarray(msd, dtype=float),
            p0=(0.001, 0.01), maxfev=10_000,
        )
    except Exception:
        return np.nan, np.nan
    d, alpha = float(popt[0]), float(popt[1])
    if not (np.isfinite(d) and np.isfinite(alpha)):
        return np.nan, np.nan
    return d, alpha
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_track_analysis3d.py src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py
git commit -m "feat(nd): calibrated positions, MSD and its fit for 3D track analysis"
```

---

### Task 2: `_track_analysis3d.py` — per-track analysis, steps, lifetime distribution

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_track_analysis3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py`

**Interfaces:**
- Consumes: everything Task 1 produced.
- Produces, in `_track_analysis3d`:
  - column lists `PROPERTY_COLUMNS`, `MSD_COLUMNS`, `STEP_COLUMNS`, `LIFETIME_COLUMNS` (exact values in Step 3)
  - `find_steps(data, window=20, threshold=0.5)` — the 2D function, unchanged
  - `intensity_trace(track) -> tuple[np.ndarray, np.ndarray]` — `(frames, values)`
  - `analyse_track(track: pd.DataFrame, params: AnalysisParams) -> tuple[dict, list[dict], list[dict]]` — `(properties_row, msd_rows, step_rows)`; keys exactly `PROPERTY_COLUMNS`, `MSD_COLUMNS`, `STEP_COLUMNS`
  - `lifetime_distribution(properties: pd.DataFrame, frame_interval_s: float) -> pd.DataFrame` — columns `LIFETIME_COLUMNS`

- [ ] **Step 1: Write the failing tests**

Append to `test_track_analysis3d.py`:

```python
class TestAnalyseTrackMotion:
    def test_a_brownian_walk_recovers_d_and_alpha_in_3d_and_xy(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        rng = np.random.default_rng(1)
        true_d, dt = 0.01, 3.0
        d3, a3, dxy, axy = [], [], [], []
        for track_id in range(150):
            steps = rng.normal(0, np.sqrt(2 * true_d * dt), size=(100, 3))
            row, _, _ = analyse_track(_track(np.cumsum(steps, axis=0), track_id), PARAMS)
            d3.append(row["D_3d"])
            a3.append(row["alpha_3d"])
            dxy.append(row["D_xy"])
            axy.append(row["alpha_xy"])

        assert np.median(a3) == pytest.approx(1.0, abs=0.15)
        assert np.median(d3) == pytest.approx(true_d, rel=0.25)
        assert np.median(axy) == pytest.approx(1.0, abs=0.15)
        assert np.median(dxy) == pytest.approx(true_d, rel=0.25)

    def test_constant_velocity_is_directed(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        rng = np.random.default_rng(2)
        t = np.arange(40)
        pos = np.column_stack([0 * t, 0 * t, 0.2 * t]) + rng.normal(0, 0.01, (40, 3))

        row, _, _ = analyse_track(_track(pos), PARAMS)

        assert row["alpha_3d"] > 1.2
        assert row["motility_class"] == "directed"

    def test_jitter_in_place_is_constrained(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        rng = np.random.default_rng(3)

        row, _, _ = analyse_track(_track(rng.normal(0, 0.02, (40, 3))), PARAMS)

        assert row["motility_class"] == "constrained"

    def test_motility_from_xy_uses_alpha_xy(self):
        from dataclasses import replace

        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        # Directed along z only: xy is stationary jitter.
        rng = np.random.default_rng(4)
        t = np.arange(40)
        pos = np.column_stack([0.2 * t, 0 * t, 0 * t]) + rng.normal(0, 0.01, (40, 3))

        row_3d, _, _ = analyse_track(_track(pos), PARAMS)
        row_xy, _, _ = analyse_track(_track(pos), replace(PARAMS, motility_from="xy"))

        assert row_3d["motility_class"] == "directed"
        assert row_xy["motility_class"] == "constrained"

    def test_a_short_track_is_not_fitted(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, msd_rows, _ = analyse_track(_track(np.random.default_rng(5).normal(0, 0.1, (9, 3))), PARAMS)

        assert np.isnan(row["D_3d"]) and np.isnan(row["alpha_xy"])
        assert row["msd_lags_fitted"] == 0
        assert row["motility_class"] == "unclassified"
        # The curve is still reported, unfitted.
        assert msd_rows and all(np.isnan(r["msd_fit"]) for r in msd_rows)

    def test_a_fitted_track_reports_its_curve_and_fit(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import MSD_COLUMNS, analyse_track

        rng = np.random.default_rng(6)
        row, msd_rows, _ = analyse_track(_track(np.cumsum(rng.normal(0, 0.2, (30, 3)), axis=0)), PARAMS)

        assert row["msd_lags_fitted"] == 25
        assert {r["dims"] for r in msd_rows} == {"3d", "xy"}
        assert all(set(r) == set(MSD_COLUMNS) for r in msd_rows)
        first = next(r for r in msd_rows if r["dims"] == "3d" and r["lag_frames"] == 1)
        assert first["lag_s"] == 3.0
        assert np.isfinite(first["msd_fit"])


class TestAnalyseTrackGeometryAndLifetime:
    def test_anisotropic_steps_and_gap_bookkeeping(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        # One z-voxel step (0.210 µm), then a gap, then one x-pixel step (0.104 µm).
        row, _, _ = analyse_track(
            _track([[0, 0, 0], [0.210, 0, 0], [0.210, 0, 0.104]], frames=[0, 1, 3]), PARAMS,
        )

        assert row["path_length_um"] == pytest.approx(0.314)
        assert row["net_displacement_um"] == pytest.approx(np.hypot(0.210, 0.104))
        assert row["n_spots"] == 3
        assert row["n_gap_frames"] == 1
        assert row["lifetime_frames"] == 4
        assert row["lifetime_s"] == 12.0
        assert row["mean_speed_um_s"] == pytest.approx(0.314 / 9.0)

    def test_censored_is_carried_through(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, _, _ = analyse_track(_track(np.zeros((3, 3)), censored=True), PARAMS)

        assert row["censored"] is True

    def test_a_single_spot_track_is_all_nan_motion_and_no_crash(self):
        # Review focus: tracking's min length defaults to 1.
        from napari_spotiflow_tracking_nd._track_analysis3d import PROPERTY_COLUMNS, analyse_track

        row, msd_rows, step_rows = analyse_track(_track([[1.0, 2.0, 3.0]], frames=[7]), PARAMS)

        assert list(row) == PROPERTY_COLUMNS
        assert row["lifetime_frames"] == 1 and row["lifetime_s"] == 3.0
        assert row["path_length_um"] == 0.0 and row["net_displacement_um"] == 0.0
        assert np.isnan(row["mean_speed_um_s"])
        assert np.isnan(row["D_3d"]) and row["motility_class"] == "unclassified"
        assert msd_rows == [] and step_rows == []


class TestAnalyseTrackIntensity:
    def test_a_two_level_trace_with_failed_fits_has_one_step(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import STEP_COLUMNS, analyse_track

        frames = np.arange(30)
        values = np.where(frames < 15, 1000.0, 400.0)
        converged = np.ones(30, dtype=bool)
        converged[[3, 20]] = False

        row, _, steps = analyse_track(
            _track(np.zeros((30, 3)), integrated_intensity=values, converged=converged), PARAMS,
        )

        assert row["n_intensity_points"] == 28
        assert row["mean_intensity"] == pytest.approx((14 * 1000 + 14 * 400) / 28)
        assert row["max_intensity"] == 1000.0
        assert row["n_steps"] == 1
        assert set(steps[0]) == set(STEP_COLUMNS)
        # The ported 2D detector lands on the last high point, one early.
        assert steps[0]["frame"] in (14, 15)
        assert steps[0]["step_height"] == pytest.approx(-600, rel=0.1)
        assert row["mean_step_height"] == steps[0]["step_height"]

    def test_a_trace_too_short_for_the_window_is_not_assessed(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, _, steps = analyse_track(
            _track(np.zeros((6, 3)), integrated_intensity=1000.0, converged=True), PARAMS,
        )

        # 6 points < 2 × 3 + 1: NaN, not zero steps.
        assert np.isnan(row["n_steps"]) and np.isnan(row["mean_step_height"])
        assert steps == []

    def test_a_flat_trace_has_zero_steps(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, _, steps = analyse_track(
            _track(np.zeros((20, 3)), integrated_intensity=1000.0, converged=True), PARAMS,
        )

        assert row["n_steps"] == 0 and np.isnan(row["mean_step_height"])
        assert steps == []

    def test_no_fit_columns_means_no_intensity(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        row, _, _ = analyse_track(_track(np.zeros((20, 3))), PARAMS)

        assert row["n_intensity_points"] == 0
        assert np.isnan(row["mean_intensity"]) and np.isnan(row["max_intensity"])
        assert np.isnan(row["n_steps"])

    def test_a_mostly_failed_fit_trace_gives_nan_steps(self):
        # Review focus: on decon_10 ~75 % of fits fail.
        from napari_spotiflow_tracking_nd._track_analysis3d import analyse_track

        converged = np.zeros(20, dtype=bool)
        converged[::4] = True  # 5 of 20
        values = np.where(converged, 1000.0, np.nan)

        row, _, steps = analyse_track(
            _track(np.zeros((20, 3)), integrated_intensity=values, converged=converged), PARAMS,
        )

        assert row["n_intensity_points"] == 5
        assert np.isnan(row["n_steps"])
        assert steps == []


class TestLifetimeDistribution:
    def test_counts_uncensored_tracks_only(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import (
            LIFETIME_COLUMNS,
            lifetime_distribution,
        )

        props = pd.DataFrame({
            "lifetime_frames": [3, 3, 5, 9],
            "censored": [False, False, False, True],
        })

        dist = lifetime_distribution(props, 3.0)

        assert list(dist.columns) == LIFETIME_COLUMNS
        assert dist["lifetime_frames"].tolist() == [3, 5]
        assert dist["lifetime_s"].tolist() == [9.0, 15.0]
        assert dist["count"].tolist() == [2, 1]
        assert dist["fraction"].tolist() == pytest.approx([2 / 3, 1 / 3])

    def test_all_censored_gives_an_empty_table_with_columns(self):
        from napari_spotiflow_tracking_nd._track_analysis3d import (
            LIFETIME_COLUMNS,
            lifetime_distribution,
        )

        dist = lifetime_distribution(
            pd.DataFrame({"lifetime_frames": [4], "censored": [True]}), 3.0,
        )

        assert dist.empty and list(dist.columns) == LIFETIME_COLUMNS
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py -v`
Expected: the new classes FAIL with `ImportError: cannot import name 'analyse_track'` (and `lifetime_distribution`, `PROPERTY_COLUMNS`, …); Task 1's tests still PASS.

- [ ] **Step 3: Implement it**

In `_track_analysis3d.py`, after `MSD_DIMS`, add the column lists:

```python
PROPERTY_COLUMNS = [
    "track_id", "n_spots", "first_frame", "last_frame", "n_gap_frames", "censored",
    "lifetime_frames", "lifetime_s",
    "path_length_um", "net_displacement_um", "mean_speed_um_s",
    "D_3d", "alpha_3d", "D_xy", "alpha_xy", "msd_lags_fitted", "motility_class",
    "n_intensity_points", "mean_intensity", "max_intensity",
    "n_steps", "mean_step_height",
]
MSD_COLUMNS = ["track_id", "dims", "lag_frames", "lag_s", "msd_um2", "sem", "n_pairs", "msd_fit"]
STEP_COLUMNS = [
    "track_id", "frame", "level_before", "level_after",
    "step_height", "dwell_before", "dwell_after",
]
LIFETIME_COLUMNS = ["lifetime_frames", "lifetime_s", "count", "fraction"]
```

and append to the end of the module:

```python
def find_steps(data, window: int = 20, threshold: float = 0.5):
    """Detect step-like transitions in a 1-D time series.

    Returns ``(table, fitx, gaussian_normalise)``:
      - ``table``: list of ``[index, level_before, level_after, step_height,
        dwell_before, dwell_after, step_error]`` per detected step.
      - ``fitx``: piecewise-constant fit aligned with ``data``.
      - ``gaussian_normalise``: the normalized gaussian-filtered derivative
        used for peak detection (useful for diagnostics).

    Ported unchanged from the 2D plugin. Adopted from
    https://github.com/zeroth/step_detection
    """
    from math import sqrt

    from scipy.ndimage import gaussian_filter1d

    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return [], np.array([]), np.array([])

    gaussian_data = gaussian_filter1d(data, window, order=1)
    peak_amp = float(np.abs(gaussian_data).max())
    if peak_amp > 0:
        gaussian_normalise = gaussian_data / peak_amp
    else:
        gaussian_normalise = gaussian_data
    gaussian_normalise = np.abs(gaussian_normalise)

    peaks = np.where(gaussian_normalise > threshold, 1, 0)
    peaks_dif = np.diff(peaks)
    ups = np.where(peaks_dif == 1)[0]
    dns = np.where(peaks_dif == -1)[0]
    indices = []
    for u, d in zip(ups, dns):
        g_slice = gaussian_normalise[u:d]
        if not len(g_slice):
            continue
        loc = int(np.argmax(g_slice))
        indices.append(loc + int(u))

    last = len(indices) - 1
    table = []
    fitx = np.zeros(data.shape)
    for i, index in enumerate(indices):
        if i == 0:
            level_before = data[0:index]
            if i == last:
                level_after = data[index:]
                dwell_after = len(data) - index
                fitx[index:] = level_after.mean()
            else:
                level_after = data[index:indices[i + 1]]
                dwell_after = indices[i + 1] - index
                fitx[index:indices[i + 1]] = level_after.mean()
            dwell_before = index
            fitx[0:index] = level_before.mean() if level_before.size else 0.0
        elif i == last:
            level_before = data[indices[i - 1]:index]
            level_after = data[index:]
            dwell_before = index - indices[i - 1]
            dwell_after = len(data) - index
            fitx[indices[i - 1]:index] = level_before.mean()
            fitx[index:] = level_after.mean()
        else:
            level_before = data[indices[i - 1]:index]
            level_after = data[index:indices[i + 1]]
            dwell_before = index - indices[i - 1]
            dwell_after = indices[i + 1] - index
            fitx[indices[i - 1]:index] = level_before.mean()
            fitx[index:indices[i + 1]] = level_after.mean()

        if level_before.size and level_after.size:
            step_error = sqrt(level_after.var() + level_before.var())
            step_height = float(level_after.mean() - level_before.mean())
            level_before_mean = float(level_before.mean())
            level_after_mean = float(level_after.mean())
        else:
            step_error = 0.0
            step_height = 0.0
            level_before_mean = float(level_before.mean()) if level_before.size else 0.0
            level_after_mean = float(level_after.mean()) if level_after.size else 0.0
        table.append([index, level_before_mean, level_after_mean,
                      step_height, dwell_before, dwell_after, step_error])

    # If no steps detected, fitx is all-zero; fill with the data mean so the
    # plot still shows a reasonable baseline.
    if len(indices) == 0:
        fitx[:] = float(data.mean())

    return table, fitx, gaussian_normalise


def intensity_trace(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(frames, values)`: the track's converged spots' finite
    `integrated_intensity`, in frame order. Empty without fit columns —
    failed fits are skipped, never interpolated."""
    if "converged" not in track.columns or "integrated_intensity" not in track.columns:
        return np.array([], dtype=int), np.array([], dtype=float)
    values = track["integrated_intensity"].to_numpy(dtype=float)
    # `.eq(True)`: a blank `converged` cell reads back as NaN, which is truthy.
    keep = track["converged"].eq(True).to_numpy() & np.isfinite(values)
    return track["frame"].to_numpy(dtype=int)[keep], values[keep]


def analyse_track(
    track: pd.DataFrame, params: AnalysisParams,
) -> tuple[dict, list[dict], list[dict]]:
    """One track's `(properties_row, msd_rows, step_rows)`.

    Per-track failures — a fit that won't converge, a trace too short for
    steps — become NaN, never an exception: one awkward track must not
    sink a run over thousands.
    """
    track = track.sort_values("frame", kind="stable")
    track_id = int(track["particle"].iloc[0])
    frames, pos = positions_um(track, params)
    first, last = int(frames[0]), int(frames[-1])
    dt = float(params.frame_interval_s)

    present = pos[np.isfinite(pos).all(axis=1)]
    path = float(np.linalg.norm(np.diff(present, axis=0), axis=1).sum())
    net = float(np.linalg.norm(present[-1] - present[0])) if len(present) > 1 else 0.0

    row = {
        "track_id": track_id,
        "n_spots": int(len(track)),
        "first_frame": first,
        "last_frame": last,
        "n_gap_frames": int(len(frames) - len(track)),
        "censored": bool(track["censored"].astype(bool).any()),
        "lifetime_frames": last - first + 1,
        "lifetime_s": (last - first + 1) * dt,
        "path_length_um": path,
        "net_displacement_um": net,
        "mean_speed_um_s": path / ((last - first) * dt) if last > first else np.nan,
    }

    msd_rows = []
    lags_fitted = 0
    for dims, (axes, k) in MSD_DIMS.items():
        table = msd_table(pos[:, axes], params.max_lag)
        lag_s = table.index.to_numpy(dtype=float) * dt
        d = alpha = np.nan
        fit = np.full(len(table), np.nan)
        if len(track) >= params.min_msd_spots and len(table) >= MIN_FIT_LAGS:
            d, alpha = fit_msd(lag_s, table["msd"].to_numpy(), k)
            if np.isfinite(d):
                fit = k * d * np.power(lag_s, alpha)
                if dims == "3d":
                    lags_fitted = len(table)
        row[f"D_{dims}"], row[f"alpha_{dims}"] = d, alpha
        for (lag, rec), lag_seconds, fitted in zip(table.iterrows(), lag_s, fit):
            msd_rows.append({
                "track_id": track_id, "dims": dims, "lag_frames": int(lag),
                "lag_s": float(lag_seconds), "msd_um2": rec["msd"], "sem": rec["sem"],
                "n_pairs": int(rec["n_pairs"]), "msd_fit": float(fitted),
            })
    row["msd_lags_fitted"] = lags_fitted
    row["motility_class"] = classify_motility(row[f"alpha_{params.motility_from}"])

    trace_frames, trace = intensity_trace(track)
    row["n_intensity_points"] = int(trace.size)
    row["mean_intensity"] = float(trace.mean()) if trace.size else np.nan
    row["max_intensity"] = float(trace.max()) if trace.size else np.nan

    step_rows = []
    row["n_steps"] = np.nan
    row["mean_step_height"] = np.nan
    if trace.size >= 2 * params.step_window + 1:
        try:
            steps, _, _ = find_steps(trace, params.step_window, params.step_threshold)
        except Exception:
            steps = None
        if steps is not None:
            row["n_steps"] = len(steps)
            if steps:
                row["mean_step_height"] = float(np.mean([s[3] for s in steps]))
            for index, before, after, height, dwell_before, dwell_after, _error in steps:
                step_rows.append({
                    "track_id": track_id, "frame": int(trace_frames[index]),
                    "level_before": before, "level_after": after, "step_height": height,
                    "dwell_before": int(dwell_before), "dwell_after": int(dwell_after),
                })
    return {c: row[c] for c in PROPERTY_COLUMNS}, msd_rows, step_rows


def lifetime_distribution(properties: pd.DataFrame, frame_interval_s: float) -> pd.DataFrame:
    """Lifetimes of the *uncensored* tracks, one bin per frame count, with
    each bin's fraction of the uncensored tracks. A censored track's true
    lifetime is unknown, so it has no place in this distribution."""
    done = properties.loc[~properties["censored"].astype(bool), "lifetime_frames"]
    if done.empty:
        return pd.DataFrame(columns=LIFETIME_COLUMNS)
    counts = done.value_counts().sort_index()
    return pd.DataFrame({
        "lifetime_frames": counts.index.astype(int),
        "lifetime_s": counts.index.astype(float) * float(frame_interval_s),
        "count": counts.to_numpy(dtype=int),
        "fraction": counts.to_numpy(dtype=float) / len(done),
    })
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_track_analysis3d.py src/napari_spotiflow_tracking_nd/_tests/test_track_analysis3d.py
git commit -m "feat(nd): per-track lifetimes, motion, intensity steps for 3D tracks"
```

---

### Task 3: `TrackAnalysisWorker3D`

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_workers3d.py` (imports; `replace_dir`; new class before `class FunctionWorker3D`)
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py`

**Interfaces:**
- Consumes: `AnalysisParams`, `analyse_track`, `lifetime_distribution`, `missing_columns`, `PROPERTY_COLUMNS`, `MSD_COLUMNS`, `STEP_COLUMNS` (Tasks 1–2).
- Produces: `replace_dir(src: Path, dst: Path) -> None`; `TrackAnalysisWorker3D(csv_path, output_dir, params: AnalysisParams, record: dict | None = None, parent=None)` with signals `progress(int, int)`, `run_finished(str)` (results folder, or `""` when cancelled), `errored(str)`, and `cancel()`. It calls `analyse_track` as a `_workers3d` module global (Task 5 monkeypatches it). Writes `track_properties.csv`, `lifetime_distribution.csv`, `msd_curves.csv`, `steps.csv`, `analysis.json` (the `record`, as JSON).

- [ ] **Step 1: Write the failing tests**

Append to `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py` (it already imports `numpy as np`, `pandas as pd`, `pytest`):

```python
def _write_tracks_csv(path, n_tracks=3, n_frames=12):
    """Three jittering tracks, the first censored."""
    rng = np.random.default_rng(0)
    rows = []
    for track_id in range(n_tracks):
        for t in range(n_frames):
            rows.append({
                "particle": track_id, "frame": t,
                "link_z": 5 + rng.normal(), "link_y": 10.0 * track_id + rng.normal(),
                "link_x": 10.0, "censored": track_id == 0,
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _analysis_params():
    from napari_spotiflow_tracking_nd._track_analysis3d import AnalysisParams

    return AnalysisParams(pixel_size_xy_um=0.104, voxel_size_z_um=0.210, frame_interval_s=3.0)


def _run_analysis_worker(worker):
    seen = {"progress": [], "finished": [], "errored": []}
    worker.progress.connect(lambda c, t: seen["progress"].append((c, t)))
    worker.run_finished.connect(seen["finished"].append)
    worker.errored.connect(seen["errored"].append)
    worker.run()
    return seen


RESULT_FILES = [
    "analysis.json", "lifetime_distribution.csv", "msd_curves.csv",
    "steps.csv", "track_properties.csv",
]


class TestTrackAnalysisWorker3D:
    def test_writes_the_results_folder_and_reports_progress_per_track(self, tmp_path):
        import json

        from napari_spotiflow_tracking_nd._track_analysis3d import PROPERTY_COLUMNS
        from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

        src = _write_tracks_csv(tmp_path / "t_tracks.csv")
        out = tmp_path / "t_tracks_analysis"

        seen = _run_analysis_worker(
            TrackAnalysisWorker3D(src, out, _analysis_params(), record={"source_csv": str(src)})
        )

        assert seen["errored"] == []
        assert seen["progress"] == [(1, 3), (2, 3), (3, 3)]
        assert seen["finished"] == [str(out)]
        assert sorted(p.name for p in out.iterdir()) == RESULT_FILES
        props = pd.read_csv(out / "track_properties.csv")
        assert list(props.columns) == PROPERTY_COLUMNS
        assert props["track_id"].tolist() == [0, 1, 2]
        assert json.loads((out / "analysis.json").read_text()) == {"source_csv": str(src)}
        # Only the two uncensored tracks are in the lifetime distribution.
        assert pd.read_csv(out / "lifetime_distribution.csv")["count"].sum() == 2

    def test_no_partial_folder_is_left_behind(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

        src = _write_tracks_csv(tmp_path / "t.csv")

        _run_analysis_worker(TrackAnalysisWorker3D(src, tmp_path / "res", _analysis_params()))

        assert sorted(p.name for p in tmp_path.iterdir()) == ["res", "t.csv"]

    def test_a_rerun_replaces_the_previous_results_folder(self, tmp_path):
        # Review focus: an earlier run's files must not mix with this one's.
        from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

        src = _write_tracks_csv(tmp_path / "t.csv")
        out = tmp_path / "res"
        out.mkdir()
        (out / "stale.csv").write_text("from an earlier run")

        _run_analysis_worker(TrackAnalysisWorker3D(src, out, _analysis_params()))

        assert sorted(p.name for p in out.iterdir()) == RESULT_FILES
        assert sorted(p.name for p in tmp_path.iterdir()) == ["res", "t.csv"]

    def test_cancel_writes_nothing(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

        src = _write_tracks_csv(tmp_path / "t.csv")
        out = tmp_path / "res"
        worker = TrackAnalysisWorker3D(src, out, _analysis_params())
        worker.progress.connect(lambda current, total: worker.cancel())

        seen = _run_analysis_worker(worker)

        assert seen["finished"] == [""]
        assert not out.exists()
        assert not (tmp_path / "res.partial").exists()

    def test_a_cancel_keeps_an_earlier_results_folder(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

        src = _write_tracks_csv(tmp_path / "t.csv")
        out = tmp_path / "res"
        out.mkdir()
        (out / "earlier.csv").write_text("keep me")
        worker = TrackAnalysisWorker3D(src, out, _analysis_params())
        worker.cancel()

        _run_analysis_worker(worker)

        assert (out / "earlier.csv").read_text() == "keep me"

    def test_missing_columns_are_refused_by_name_and_earlier_output_kept(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

        bad = tmp_path / "bad.csv"
        pd.DataFrame({"particle": [0], "frame": [0]}).to_csv(bad, index=False)
        out = tmp_path / "res"
        out.mkdir()
        (out / "earlier.csv").write_text("keep me")

        seen = _run_analysis_worker(TrackAnalysisWorker3D(bad, out, _analysis_params()))

        assert len(seen["errored"]) == 1
        assert "link_z" in seen["errored"][0] and "censored" in seen["errored"][0]
        assert (out / "earlier.csv").read_text() == "keep me"
        assert not (tmp_path / "res.partial").exists()

    def test_an_empty_tracks_table_is_refused(self, tmp_path):
        from napari_spotiflow_tracking_nd._track_analysis3d import REQUIRED_COLUMNS
        from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

        empty = tmp_path / "empty.csv"
        pd.DataFrame(columns=list(REQUIRED_COLUMNS)).to_csv(empty, index=False)

        seen = _run_analysis_worker(TrackAnalysisWorker3D(empty, tmp_path / "res", _analysis_params()))

        assert "no tracks" in seen["errored"][0]
        assert not (tmp_path / "res").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py::TestTrackAnalysisWorker3D -v`
Expected: FAIL with `ImportError: cannot import name 'TrackAnalysisWorker3D'`.

- [ ] **Step 3: Implement it**

In `_workers3d.py`, add `import json` and `import shutil` to the stdlib imports, and after the `_tracking3d` import block:

```python
from napari_spotiflow_tracking_nd._track_analysis3d import (
    MSD_COLUMNS,
    PROPERTY_COLUMNS,
    STEP_COLUMNS,
    AnalysisParams,
    analyse_track,
    lifetime_distribution,
)
from napari_spotiflow_tracking_nd._track_analysis3d import (
    missing_columns as missing_track_columns,
)
```

(`missing_columns` is already imported from `_tracking3d` for the tracking worker; the alias keeps the two apart.)

Insert, directly before `class FunctionWorker3D`:

```python
def replace_dir(src: Path, dst: Path) -> None:
    """Move the finished folder `src` to `dst`, replacing any earlier `dst`
    only once `src` is complete — so a failed run can never leave `dst`
    half old, half new."""
    old = dst.with_name(dst.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if dst.exists():
        os.replace(dst, old)
    os.replace(src, dst)
    shutil.rmtree(old, ignore_errors=True)


class TrackAnalysisWorker3D(QThread):
    """Analyse a tracks CSV one track at a time and write the results
    folder once every track is done.

    Cancellation (`cancel()`) is checked between tracks and writes
    *nothing*: unlike a cancelled tracking run, a partial analysis is a
    biased sample (the first tracks by ID), not a shorter series. Output
    is assembled in `<folder>.partial` and swapped into place by
    `replace_dir`, so an error never touches an earlier results folder.
    """

    progress = Signal(int, int)  # (current, total), 1-indexed current
    # `run_finished`, not `finished` — see `VolumeOpWorker3D` for why.
    run_finished = Signal(str)   # the results folder, or "" if cancelled
    errored = Signal(str)

    def __init__(
        self,
        csv_path: Path,
        output_dir: Path,
        params: AnalysisParams,
        record: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._csv_path = Path(csv_path)
        self._output_dir = Path(output_dir)
        self._params = params
        self._record = dict(record or {})
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        partial = self._output_dir.with_name(self._output_dir.name + ".partial")
        try:
            tracks = pd.read_csv(self._csv_path)
            missing = missing_track_columns(tracks)
            if missing:
                raise ValueError(
                    f"{self._csv_path.name} is not a tracks CSV — missing "
                    f"column(s): {', '.join(missing)}."
                )
            if tracks.empty:
                raise ValueError(f"{self._csv_path.name} has no tracks.")

            groups = [group for _, group in tracks.groupby("particle", sort=True)]
            properties, msd_rows, step_rows = [], [], []
            for i, track in enumerate(groups):
                if self._cancelled:
                    self.run_finished.emit("")
                    return
                row, msd, steps = analyse_track(track, self._params)
                properties.append(row)
                msd_rows.extend(msd)
                step_rows.extend(steps)
                self.progress.emit(i + 1, len(groups))

            table = pd.DataFrame(properties, columns=PROPERTY_COLUMNS)
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(parents=True)
            table.to_csv(partial / "track_properties.csv", index=False)
            lifetime_distribution(table, self._params.frame_interval_s).to_csv(
                partial / "lifetime_distribution.csv", index=False
            )
            pd.DataFrame(msd_rows, columns=MSD_COLUMNS).to_csv(
                partial / "msd_curves.csv", index=False
            )
            pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(
                partial / "steps.csv", index=False
            )
            (partial / "analysis.json").write_text(
                json.dumps(self._record, indent=2), encoding="utf-8"
            )
            replace_dir(partial, self._output_dir)
            self.run_finished.emit(str(self._output_dir))
        except Exception as exc:
            shutil.rmtree(partial, ignore_errors=True)
            self.errored.emit(str(exc))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py -v`
Expected: all PASS (old and new).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_workers3d.py src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py
git commit -m "feat(nd): TrackAnalysisWorker3D writes the results folder atomically"
```

---

### Task 4: `TrackAnalysisWidget3D`, shared widget helpers, manifest

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_layer_utils.py` (add `calibration_spin`)
- Modify: `src/napari_spotiflow_tracking_nd/_tracking_widget3d.py` (use `calibration_spin`; add `read_sidecar`)
- Create: `src/napari_spotiflow_tracking_nd/_track_analysis_widget3d.py`
- Modify: `src/napari_spotiflow_tracking_nd/napari.yaml`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_track_analysis_widget3d.py`

**Interfaces:**
- Consumes: `TrackAnalysisWorker3D` (Task 3); `AnalysisParams` (Task 1); from `_tracking_widget3d`: `same_file(a, b)`, `read_sidecar_calibration(csv_path)`; from `_detection_widget3d`: `CALIBRATION_KEYS`; `refresh_layer_combo`; `RunStateMixin`.
- Produces: `_layer_utils.calibration_spin(maximum: float, decimals: int) -> QDoubleSpinBox`; `_tracking_widget3d.read_sidecar(csv_path) -> dict`; `_track_analysis_widget3d.default_analysis_dir(csv_path) -> Path`, `summarise(properties: pd.DataFrame) -> str`, `TrackAnalysisWidget3D(napari_viewer)` with attributes `_tracks_combo, _csv_path_edit, _browse_csv_btn, _csv_info_label, _pixel_size_xy_spin, _voxel_size_z_spin, _frame_interval_spin, _max_lag_spin, _min_msd_spots_spin, _motility_combo, _step_window_spin, _step_threshold_spin, _output_dir_edit, _run_btn, _cancel_btn, _progress_bar, _status_label` and methods `_set_input_csv(path, layer=None)`, `_run_analysis()`.

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_track_analysis_widget3d.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CALIBRATION = {"pixel_size_xy_um": 0.104, "voxel_size_z_um": 0.210, "frame_interval_s": 3.0}


@pytest.fixture
def analysis_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._track_analysis_widget3d import TrackAnalysisWidget3D

    viewer = make_napari_viewer()
    widget = TrackAnalysisWidget3D(viewer)
    # Synchronous through the seam; real threading is covered by
    # test_threading.py.
    widget._launch_worker = lambda worker: worker.run()
    return widget, viewer


def _write_tracks(folder: Path, name="spots_tracks.csv", sidecar=CALIBRATION, n_frames=12):
    """Three tracks — two jittering in place (censored: they span the
    series), one living frames 3..7 — plus the tracking sidecar."""
    rng = np.random.default_rng(0)
    rows = []
    for track_id, frames in ((0, range(n_frames)), (1, range(n_frames)), (2, range(3, 8))):
        for t in frames:
            rows.append({
                "particle": track_id, "frame": t,
                "link_z": 5 + 0.1 * rng.normal(), "link_y": 20.0 * track_id + 0.1 * rng.normal(),
                "link_x": 10.0, "censored": track_id != 2,
            })
    folder.mkdir(parents=True, exist_ok=True)
    csv = folder / name
    pd.DataFrame(rows).to_csv(csv, index=False)
    if sidecar is not None:
        csv.with_suffix(".json").write_text(json.dumps(sidecar))
    return csv


def _add_tracks_layer(viewer, csv, name="vol (tracks)", **metadata):
    df = pd.read_csv(csv).sort_values(["particle", "frame"])
    return viewer.add_tracks(
        df[["particle", "frame", "link_z", "link_y", "link_x"]].to_numpy(dtype=float),
        name=name,
        metadata={"tracks_csv_path": str(csv), **metadata},
    )


def _output_folder(tmp_path):
    return tmp_path / "out" / "spots_tracks_analysis"


class TestCreation:
    def test_has_the_expected_controls(self, analysis_widget):
        widget, _ = analysis_widget
        for attr in (
            "_tracks_combo", "_csv_path_edit", "_browse_csv_btn", "_csv_info_label",
            "_pixel_size_xy_spin", "_voxel_size_z_spin", "_frame_interval_spin",
            "_max_lag_spin", "_min_msd_spots_spin", "_motility_combo",
            "_step_window_spin", "_step_threshold_spin", "_output_dir_edit",
            "_run_btn", "_cancel_btn", "_progress_bar", "_status_label",
        ):
            assert hasattr(widget, attr), f"missing {attr}"

    def test_defaults(self, analysis_widget):
        widget, _ = analysis_widget
        assert widget._max_lag_spin.value() == 25
        assert widget._min_msd_spots_spin.value() == 10
        assert widget._motility_combo.currentText() == "3D"
        assert widget._step_window_spin.value() == 3
        assert widget._step_threshold_spin.value() == 0.5

    def test_registered_in_the_manifest(self):
        from importlib.resources import files

        manifest = files("napari_spotiflow_tracking_nd").joinpath("napari.yaml").read_text()
        assert "_track_analysis_widget3d:TrackAnalysisWidget3D" in manifest
        assert "Track Analysis" in manifest


class TestInput:
    def test_only_tracks_layers_from_tracking_are_listed(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        viewer.add_tracks(np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1.0]]), name="other")
        _add_tracks_layer(viewer, _write_tracks(tmp_path))

        listed = [widget._tracks_combo.itemText(i) for i in range(widget._tracks_combo.count())]

        assert listed == ["vol (tracks)"]

    def test_a_tracks_layer_fills_input_output_and_counts(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        csv = _write_tracks(tmp_path)

        _add_tracks_layer(viewer, csv)

        assert widget._csv_path_edit.text() == str(csv)
        assert widget._output_dir_edit.text() == str(tmp_path / "spots_tracks_analysis")
        assert "3 tracks" in widget._csv_info_label.text()

    def test_a_typed_path_is_adopted(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        csv = _write_tracks(tmp_path / "b")

        widget._csv_path_edit.setText(str(csv))
        widget._csv_path_edit.editingFinished.emit()

        assert widget._output_dir_edit.text() == str(tmp_path / "b" / "spots_tracks_analysis")
        assert widget._pixel_size_xy_spin.value() == pytest.approx(0.104)


class TestCalibrationPrefill:
    def test_layer_values_win_then_the_sidecar_fills_the_gaps(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        csv = _write_tracks(tmp_path, sidecar={**CALIBRATION, "pixel_size_xy_um": 0.1})

        _add_tracks_layer(viewer, csv, pixel_size_xy_um=0.2)

        assert widget._pixel_size_xy_spin.value() == pytest.approx(0.2)
        assert widget._voxel_size_z_spin.value() == pytest.approx(0.210)
        assert widget._frame_interval_spin.value() == pytest.approx(3.0)


class TestRefusals:
    def test_incomplete_calibration_is_refused(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path, sidecar=None))
        widget._output_dir_edit.setText(str(_output_folder(tmp_path)))
        widget._pixel_size_xy_spin.setValue(0.104)
        widget._voxel_size_z_spin.setValue(0.210)  # frame interval left blank

        widget._run_analysis()

        assert not _output_folder(tmp_path).exists()
        assert not widget._is_running

    def test_the_tracks_csvs_own_folder_is_refused(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path))
        widget._output_dir_edit.setText(str(tmp_path / "sub" / ".."))
        before = sorted(p.name for p in tmp_path.iterdir())

        widget._run_analysis()

        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_an_output_path_that_is_a_file_is_refused(self, analysis_widget, tmp_path):
        # Review focus: replace_dir would otherwise move the file aside.
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path))
        target = tmp_path / "out" / "results.csv"
        target.parent.mkdir()
        target.write_text("precious")
        widget._output_dir_edit.setText(str(target))

        widget._run_analysis()

        assert target.read_text() == "precious"
        assert sorted(p.name for p in target.parent.iterdir()) == ["results.csv"]

    def test_a_missing_tracks_csv_is_refused(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        csv = _write_tracks(tmp_path)
        widget._set_input_csv(csv)
        widget._output_dir_edit.setText(str(_output_folder(tmp_path)))
        csv.unlink()

        widget._run_analysis()

        assert not _output_folder(tmp_path).exists()
        assert not widget._is_running


class TestRun:
    def test_writes_the_results_folder_with_its_record(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        csv = _write_tracks(tmp_path)
        widget._set_input_csv(csv)
        out = _output_folder(tmp_path)
        widget._output_dir_edit.setText(str(out))
        widget._motility_combo.setCurrentText("xy")

        widget._run_analysis()

        assert sorted(p.name for p in out.iterdir()) == [
            "analysis.json", "lifetime_distribution.csv", "msd_curves.csv",
            "steps.csv", "track_properties.csv",
        ]
        record = json.loads((out / "analysis.json").read_text())
        assert record["pixel_size_xy_um"] == pytest.approx(0.104)
        assert record["frame_interval_s"] == 3.0
        assert record["max_lag"] == 25 and record["min_msd_spots"] == 10
        assert record["motility_from"] == "xy"
        assert record["step_window"] == 3 and record["step_threshold"] == 0.5
        assert record["source_csv"] == str(csv)
        assert record["tracking_cancelled"] is None
        assert {"numpy", "scipy", "pandas", "plugin"} <= set(record["versions"])

    def test_per_track_results_are_joined_onto_the_tracks_layer(self, analysis_widget, tmp_path):
        widget, viewer = analysis_widget
        layer = _add_tracks_layer(viewer, _write_tracks(tmp_path))
        widget._output_dir_edit.setText(str(_output_folder(tmp_path)))
        n_vertices = len(layer.data)

        widget._run_analysis()

        features = layer.features
        assert len(features) == n_vertices
        for name in ("lifetime_s", "D_3d", "alpha_3d", "motility_class", "n_steps"):
            assert name in features.columns, f"missing feature {name}"
        short = features.loc[features["track_id"] == 2, "lifetime_frames"]
        assert (short == 5).all()

    def test_the_status_reports_the_summary(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path))
        widget._output_dir_edit.setText(str(_output_folder(tmp_path)))

        widget._run_analysis()

        text = widget._status_label.text()
        assert "3 tracks" in text
        assert "1 uncensored" in text
        assert "median lifetime 15 s" in text

    def test_tracks_from_a_cancelled_tracking_run_warn_and_are_recorded(
        self, analysis_widget, tmp_path, monkeypatch
    ):
        # Review focus: lifetimes near the end of a cancelled run are cut short.
        from napari_spotiflow_tracking_nd import _track_analysis_widget3d

        warnings = []
        monkeypatch.setattr(_track_analysis_widget3d, "show_warning", warnings.append)
        widget, _ = analysis_widget
        csv = _write_tracks(
            tmp_path, sidecar={**CALIBRATION, "cancelled": True, "first_frame": 0, "last_frame": 11},
        )
        widget._set_input_csv(csv)
        out = _output_folder(tmp_path)
        widget._output_dir_edit.setText(str(out))

        widget._run_analysis()

        assert len(warnings) == 1 and "cancelled" in warnings[0]
        record = json.loads((out / "analysis.json").read_text())
        assert record["tracking_cancelled"] is True
        assert record["tracking_last_frame"] == 11

    def test_a_cancelled_analysis_writes_nothing(self, analysis_widget, tmp_path):
        widget, _ = analysis_widget
        widget._set_input_csv(_write_tracks(tmp_path))
        out = _output_folder(tmp_path)
        widget._output_dir_edit.setText(str(out))

        def _launch(worker):
            # Through the widget's own cancel path, as the Cancel button
            # does — so the status reports the cancel.
            widget._request_cancel()
            worker.run()

        widget._launch_worker = _launch
        widget._run_analysis()

        assert not out.exists()
        assert "Cancelled" in widget._status_label.text()


class TestSummarise:
    def test_counts_and_percentages(self):
        from napari_spotiflow_tracking_nd._track_analysis_widget3d import summarise

        props = pd.DataFrame({
            "censored": [False, False, True, False],
            "lifetime_s": [9.0, 15.0, 30.0, 21.0],
            "motility_class": ["diffusive", "directed", "diffusive", "unclassified"],
        })

        assert summarise(props) == (
            "4 tracks — 3 uncensored, median lifetime 15 s; "
            "3 with D/α (67% diffusive)."
        )

    def test_nothing_uncensored_or_fitted(self):
        from napari_spotiflow_tracking_nd._track_analysis_widget3d import summarise

        props = pd.DataFrame({
            "censored": [True], "lifetime_s": [9.0], "motility_class": ["unclassified"],
        })

        assert summarise(props) == "1 tracks — 0 uncensored; 0 with D/α."
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_track_analysis_widget3d.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._track_analysis_widget3d'`.

- [ ] **Step 3: Move `calibration_spin` and add `read_sidecar`**

In `_layer_utils.py`, add after the existing imports:

```python
from qtpy.QtWidgets import QDoubleSpinBox
```

and append:

```python
def calibration_spin(maximum: float, decimals: int) -> QDoubleSpinBox:
    """A calibration field where 0 reads as "unknown" — the plugin-wide
    convention that a blank calibration value is `None`, not zero."""
    spin = QDoubleSpinBox()
    spin.setRange(0.0, maximum)
    spin.setDecimals(decimals)
    spin.setSpecialValueText("unknown")
    return spin
```

In `_tracking_widget3d.py`: delete the `_calibration_spin` function; change the `_layer_utils` import to `from napari_spotiflow_tracking_nd._layer_utils import calibration_spin, refresh_layer_combo`; replace the three `_calibration_spin(` calls with `calibration_spin(`; and replace `read_sidecar_calibration` with:

```python
def read_sidecar(csv_path) -> dict:
    """`csv_path`'s whole JSON sidecar; `{}` when there is none or it
    cannot be read."""
    try:
        data = json.loads(calibration_sidecar_path(csv_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_sidecar_calibration(csv_path) -> dict:
    """The calibration recorded in `csv_path`'s JSON sidecar; `{}` when
    there is none or it cannot be read."""
    data = read_sidecar(csv_path)
    if not data:
        return {}
    return {k: data.get(k) for k in CALIBRATION_KEYS}
```

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_tracking_widget3d.py -q`
Expected: all PASS (refactor only).

- [ ] **Step 4: Implement the widget**

Create `src/napari_spotiflow_tracking_nd/_track_analysis_widget3d.py`:

```python
"""The Track Analysis widget: per-track lifetimes, motion and intensity
steps from a tracks CSV, written to a results folder."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import scipy
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_spotiflow_tracking_nd import __version__ as PLUGIN_VERSION
from napari_spotiflow_tracking_nd._detection_widget3d import CALIBRATION_KEYS
from napari_spotiflow_tracking_nd._layer_utils import calibration_spin, refresh_layer_combo
from napari_spotiflow_tracking_nd._run_state import RunStateMixin
from napari_spotiflow_tracking_nd._track_analysis3d import AnalysisParams
from napari_spotiflow_tracking_nd._tracking_widget3d import (
    read_sidecar,
    read_sidecar_calibration,
    same_file,
)
from napari_spotiflow_tracking_nd._workers3d import TrackAnalysisWorker3D

#: The motility combo's labels and the `AnalysisParams.motility_from` they mean.
MOTILITY_FROM = {"3D": "3d", "xy": "xy"}


def default_analysis_dir(csv_path) -> Path:
    """`spots_tracks.csv` -> `spots_tracks_analysis/`, beside it."""
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_analysis")


def summarise(properties: pd.DataFrame) -> str:
    """One status line: track count, uncensored lifetimes, motility."""
    n = len(properties)
    done = properties.loc[~properties["censored"].astype(bool), "lifetime_s"]
    text = f"{n} tracks — {len(done)} uncensored"
    if len(done):
        text += f", median lifetime {done.median():.3g} s"
    fitted = properties["motility_class"] != "unclassified"
    n_fitted = int(fitted.sum())
    text += f"; {n_fitted} with D/α"
    if n_fitted:
        diffusive = (properties.loc[fitted, "motility_class"] == "diffusive").mean()
        text += f" ({diffusive * 100:.0f}% diffusive)"
    return text + "."


class TrackAnalysisWidget3D(RunStateMixin, QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        # The Tracks layer the input CSV came from, if any — the per-track
        # results are joined onto its `features`.
        self._source_layer = None
        # The input last set through `_set_input_csv`, to tell a typed or
        # pasted path from the one already adopted.
        self._last_input_csv = ""
        self._summary = ""
        self._setup_ui()
        self._connect_events()

    # ── UI ───────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        in_group = QGroupBox("Input")
        in_layout = QVBoxLayout()
        in_group.setLayout(in_layout)
        in_layout.addWidget(QLabel("Tracks layer:"))
        self._tracks_combo = QComboBox()
        in_layout.addWidget(self._tracks_combo)
        csv_row = QHBoxLayout()
        self._csv_path_edit = QLineEdit()
        self._csv_path_edit.setPlaceholderText("Tracks CSV")
        csv_row.addWidget(self._csv_path_edit)
        self._browse_csv_btn = QPushButton("Browse CSV…")
        self._browse_csv_btn.clicked.connect(self._browse_input_csv)
        csv_row.addWidget(self._browse_csv_btn)
        in_layout.addLayout(csv_row)
        self._csv_info_label = QLabel()
        in_layout.addWidget(self._csv_info_label)
        layout.addWidget(in_group)

        calib_group = QGroupBox("Calibration (all required)")
        calib_layout = QHBoxLayout()
        calib_group.setLayout(calib_layout)
        calib_layout.addWidget(QLabel("xy (µm):"))
        self._pixel_size_xy_spin = calibration_spin(100.0, 4)
        calib_layout.addWidget(self._pixel_size_xy_spin)
        calib_layout.addWidget(QLabel("z (µm):"))
        self._voxel_size_z_spin = calibration_spin(100.0, 4)
        calib_layout.addWidget(self._voxel_size_z_spin)
        calib_layout.addWidget(QLabel("dt (s):"))
        self._frame_interval_spin = calibration_spin(10000.0, 3)
        calib_layout.addWidget(self._frame_interval_spin)
        layout.addWidget(calib_group)

        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        params_group.setLayout(params_layout)

        msd_row = QHBoxLayout()
        msd_row.addWidget(QLabel("Max lag (frames):"))
        self._max_lag_spin = QSpinBox()
        self._max_lag_spin.setRange(1, 1000)
        self._max_lag_spin.setValue(25)
        msd_row.addWidget(self._max_lag_spin)
        msd_row.addWidget(QLabel("Min spots for MSD:"))
        self._min_msd_spots_spin = QSpinBox()
        self._min_msd_spots_spin.setRange(2, 100000)
        self._min_msd_spots_spin.setValue(10)
        msd_row.addWidget(self._min_msd_spots_spin)
        params_layout.addLayout(msd_row)

        motility_row = QHBoxLayout()
        motility_row.addWidget(QLabel("Motility from:"))
        self._motility_combo = QComboBox()
        self._motility_combo.addItems(list(MOTILITY_FROM))
        self._motility_combo.setToolTip(
            "Which α sets the motility class. Axial localisation is usually "
            "2–3× worse than lateral; xy ignores it."
        )
        motility_row.addWidget(self._motility_combo)
        motility_row.addStretch()
        params_layout.addLayout(motility_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step window:"))
        self._step_window_spin = QSpinBox()
        self._step_window_spin.setRange(1, 200)
        self._step_window_spin.setValue(3)
        step_row.addWidget(self._step_window_spin)
        step_row.addWidget(QLabel("Step threshold:"))
        self._step_threshold_spin = QDoubleSpinBox()
        self._step_threshold_spin.setRange(0.0, 1.0)
        self._step_threshold_spin.setSingleStep(0.05)
        self._step_threshold_spin.setDecimals(2)
        self._step_threshold_spin.setValue(0.5)
        step_row.addWidget(self._step_threshold_spin)
        params_layout.addLayout(step_row)
        layout.addWidget(params_group)

        out_group = QGroupBox("Results folder")
        out_layout = QHBoxLayout()
        out_group.setLayout(out_layout)
        self._output_dir_edit = QLineEdit()
        out_layout.addWidget(self._output_dir_edit)
        browse_out_btn = QPushButton("Browse…")
        browse_out_btn.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(browse_out_btn)
        layout.addWidget(out_group)

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.clicked.connect(self._run_analysis)
        layout.addWidget(self._run_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("%v / %m")
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip(
            "Stops after the current track and writes nothing — a partial "
            "analysis would be a biased sample of the tracks."
        )
        layout.addWidget(self._cancel_btn)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

        self._init_run_state(
            run_buttons=[self._run_btn],
            cancel_button=self._cancel_btn,
            progress_bar=self._progress_bar,
            status_label=self._status_label,
        )

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_tracks_combo)
        self.viewer.layers.events.removed.connect(self._refresh_tracks_combo)
        self._tracks_combo.currentTextChanged.connect(self._on_tracks_layer_chosen)
        self._csv_path_edit.editingFinished.connect(self._on_csv_path_typed)
        self._refresh_tracks_combo()

    # ── input ────────────────────────────────────────────────────────

    def _refresh_tracks_combo(self, event=None):
        refresh_layer_combo(
            self._tracks_combo, self.viewer, napari.layers.Tracks,
            predicate=lambda layer: bool(layer.metadata.get("tracks_csv_path")),
        )
        # The refresh is silent, so the first tracks layer is adopted here
        # — but only into an empty input, never over a browsed CSV.
        if not self._csv_path_edit.text() and self._tracks_combo.currentText():
            self._on_tracks_layer_chosen(self._tracks_combo.currentText())

    def _on_tracks_layer_chosen(self, name: str):
        if not name or name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        self._set_input_csv(layer.metadata["tracks_csv_path"], layer)

    def _on_csv_path_typed(self):
        text = self._csv_path_edit.text().strip()
        if text and text != self._last_input_csv:
            self._set_input_csv(text)

    def _browse_input_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open tracks CSV", "", "CSV files (*.csv)")
        if path:
            self._set_input_csv(path)

    def _browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select results folder")
        if folder:
            self._output_dir_edit.setText(folder)

    def _set_input_csv(self, csv_path, layer=None):
        """Make `csv_path` the input: default the results folder beside it
        and prefill calibration — the layer's values first, then the
        tracks sidecar, else blank."""
        self._source_layer = layer
        self._last_input_csv = str(csv_path)
        self._csv_path_edit.setText(str(csv_path))
        self._output_dir_edit.setText(str(default_analysis_dir(csv_path)))

        from_sidecar = read_sidecar_calibration(csv_path)
        from_layer = layer.metadata if layer is not None else {}
        calibration = {k: from_layer.get(k) or from_sidecar.get(k) for k in CALIBRATION_KEYS}
        self._pixel_size_xy_spin.setValue(calibration["pixel_size_xy_um"] or 0.0)
        self._voxel_size_z_spin.setValue(calibration["voxel_size_z_um"] or 0.0)
        self._frame_interval_spin.setValue(calibration["frame_interval_s"] or 0.0)

        try:
            ids = pd.read_csv(csv_path, usecols=["particle"])["particle"]
        except (OSError, ValueError) as exc:
            self._csv_info_label.setText(f"Cannot read this CSV: {exc}")
        else:
            self._csv_info_label.setText(f"{ids.nunique()} tracks, {len(ids)} spots.")

    # ── parameters ───────────────────────────────────────────────────

    def _params(self) -> AnalysisParams | None:
        """The run's parameters, or None (after `show_error`) when any
        calibration value is blank — a 3D distance in raw voxels mixes
        lateral and axial steps of different sizes."""
        pixel_xy = self._pixel_size_xy_spin.value()
        voxel_z = self._voxel_size_z_spin.value()
        frame_s = self._frame_interval_spin.value()
        if not (pixel_xy and voxel_z and frame_s):
            show_error(
                "Track analysis needs the pixel size (xy), voxel size (z) and "
                "frame interval — fill in all three."
            )
            return None
        return AnalysisParams(
            pixel_size_xy_um=pixel_xy,
            voxel_size_z_um=voxel_z,
            frame_interval_s=frame_s,
            max_lag=self._max_lag_spin.value(),
            min_msd_spots=self._min_msd_spots_spin.value(),
            motility_from=MOTILITY_FROM[self._motility_combo.currentText()],
            step_window=self._step_window_spin.value(),
            step_threshold=self._step_threshold_spin.value(),
        )

    # ── run ──────────────────────────────────────────────────────────

    def _run_analysis(self):
        csv_text = self._csv_path_edit.text().strip()
        if not csv_text:
            show_error("No tracks CSV selected.")
            return
        csv_path = Path(csv_text)
        if not csv_path.is_file():
            show_error(f"Tracks CSV not found: {csv_path}")
            return
        out_text = self._output_dir_edit.text().strip()
        if not out_text:
            show_error("No results folder selected.")
            return
        out_dir = Path(out_text)
        if same_file(out_dir, csv_path.parent):
            show_error(
                "The results folder would be the tracks CSV's own folder — "
                "pick a different one."
            )
            return
        if out_dir.exists() and not out_dir.is_dir():
            show_error(f"{out_dir} is a file, not a folder — pick a different results folder.")
            return
        params = self._params()
        if params is None:
            return

        sidecar = read_sidecar(csv_path)
        if sidecar.get("cancelled"):
            show_warning(
                "These tracks come from a cancelled tracking run (frames "
                f"{sidecar.get('first_frame')}–{sidecar.get('last_frame')}) — "
                "lifetimes near the end of that range are cut short."
            )
        record = {
            **asdict(params),
            "source_csv": str(csv_path),
            "tracking_cancelled": sidecar.get("cancelled"),
            "tracking_first_frame": sidecar.get("first_frame"),
            "tracking_last_frame": sidecar.get("last_frame"),
            "versions": {
                "plugin": PLUGIN_VERSION,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
        }
        source_layer = self._source_layer
        if source_layer is not None and (
            source_layer not in self.viewer.layers
            or not same_file(source_layer.metadata.get("tracks_csv_path", ""), csv_path)
        ):
            source_layer = None

        worker = TrackAnalysisWorker3D(csv_path, out_dir, params, record)

        def _on_finished(written: str):
            if not written:
                show_info("Analysis cancelled — nothing written.")
                return
            properties = pd.read_csv(Path(written) / "track_properties.csv")
            if source_layer is not None:
                self._attach_features(source_layer, properties)
            self._summary = summarise(properties)
            show_info(self._summary)

        self._summary = ""
        # The track count is only known once the worker has read the CSV;
        # its first `progress` sets the bar's maximum.
        self._begin_run(worker, total=0, description="Analysing tracks", on_finished=_on_finished)

    @staticmethod
    def _attach_features(layer, properties: pd.DataFrame):
        """Join the per-track columns onto `layer`'s per-vertex `features`
        by `track_id`, so napari's colour-by-feature can use them. Columns
        the layer already has (e.g. `censored`) are left as they are."""
        features = layer.features.copy()
        table = properties.set_index("track_id")
        for column in table.columns:
            if column not in features.columns:
                features[column] = features["track_id"].map(table[column]).to_numpy()
        layer.features = features

    def _on_run_finished(self, payload=None):
        # The mixin writes "Done." / "Cancelled at …" after the host's
        # handler; the summary goes after that so it is what stays visible.
        super()._on_run_finished(payload)
        if self._summary:
            self._set_status(f"{self._status_label.text()} {self._summary}")
```

- [ ] **Step 5: Register the widget**

In `src/napari_spotiflow_tracking_nd/napari.yaml`, add under `commands:` (after the tracking widget command):

```yaml
    - id: napari-spotiflow-tracking-nd.track_analysis_widget
      python_name: napari_spotiflow_tracking_nd._track_analysis_widget3d:TrackAnalysisWidget3D
      title: Track Analysis
```

and under `widgets:` (after the tracking widget):

```yaml
    - command: napari-spotiflow-tracking-nd.track_analysis_widget
      display_name: Track Analysis
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_track_analysis_widget3d.py src/napari_spotiflow_tracking_nd/_tests/test_tracking_widget3d.py -v`
Expected: all PASS.

- [ ] **Step 7: Run the whole suite**

Run: `.venv/Scripts/python -m pytest`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_layer_utils.py src/napari_spotiflow_tracking_nd/_tracking_widget3d.py src/napari_spotiflow_tracking_nd/_track_analysis_widget3d.py src/napari_spotiflow_tracking_nd/napari.yaml src/napari_spotiflow_tracking_nd/_tests/test_track_analysis_widget3d.py
git commit -m "feat(nd): Track Analysis widget"
```

---

### Task 5: Real-thread test and README

**Files:**
- Modify: `src/napari_spotiflow_tracking_nd/_tests/test_threading.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `TrackAnalysisWidget3D` (Task 4); `napari_spotiflow_tracking_nd._workers3d.analyse_track` (the module global the worker calls — Task 3).
- Produces: nothing new.

- [ ] **Step 1: Write the test**

Append to `src/napari_spotiflow_tracking_nd/_tests/test_threading.py` (it already imports `time`, `pytest`, `numpy as np`):

```python
@pytest.fixture
def track_analysis(make_napari_viewer, monkeypatch):
    from napari_spotiflow_tracking_nd import _workers3d
    from napari_spotiflow_tracking_nd._track_analysis_widget3d import TrackAnalysisWidget3D

    # Analysis is fast; slow each track so a human-speed Cancel can land
    # mid-run (see the pre-processing cancel test for why).
    real_analyse_track = _workers3d.analyse_track

    def _slow_analyse_track(*args, **kwargs):
        time.sleep(0.05)
        return real_analyse_track(*args, **kwargs)

    monkeypatch.setattr(_workers3d, "analyse_track", _slow_analyse_track)
    viewer = make_napari_viewer()
    return TrackAnalysisWidget3D(viewer), viewer


class TestRealThreadingTrackAnalysis:
    def test_cancelling_a_real_analysis_writes_nothing(self, track_analysis, tmp_path, qtbot):
        import json

        import pandas as pd

        widget, _ = track_analysis
        rows = [
            {"particle": p, "frame": t, "link_z": 5.0, "link_y": 10.0 * p,
             "link_x": 10.0, "censored": False}
            for p in range(20) for t in range(5)
        ]
        csv = tmp_path / "spots_tracks.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        csv.with_suffix(".json").write_text(json.dumps(
            {"pixel_size_xy_um": 0.104, "voxel_size_z_um": 0.210, "frame_interval_s": 3.0}
        ))
        widget._set_input_csv(csv)
        out = tmp_path / "results"
        widget._output_dir_edit.setText(str(out))
        widget._progress_bar.valueChanged.connect(
            lambda value: widget._cancel_btn.click() if value == 1 else None
        )

        widget._run_analysis()
        qtbot.waitUntil(lambda: not widget._is_running, timeout=15000)

        assert not out.exists()
        assert not (tmp_path / "results.partial").exists()
        assert "Cancelled" in widget._status_label.text()
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python -m pytest src/napari_spotiflow_tracking_nd/_tests/test_threading.py -v`
Expected: all PASS.

- [ ] **Step 3: Document the feature**

In `README.md`, replace the line

```
Deferred to later phases: track-property/step analysis, PAPA (not
planned), N2V denoise, walking average, erosion background removal,
Shapes-based crop.
```

(if its wording differs, replace the paragraph beginning `Deferred to later phases:`) with:

```
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
```

and add, before `## Tests`:

````
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
````

- [ ] **Step 4: Run the whole suite**

Run: `.venv/Scripts/python -m pytest`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_tests/test_threading.py README.md
git commit -m "docs(nd): 3D track analysis; real-thread analysis test"
```
