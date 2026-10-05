"""Per-track analysis of 3D tracks — lifetimes, geometry, MSD, intensity
steps. Logic only, no Qt.

Input is the Spot Tracking widget's tracks table, in voxel coordinates
(`link_z/y/x`). Everything here is computed in µm and seconds, so
calibration is required: a distance in raw voxels mixes 0.104 µm lateral
and 0.210 µm axial steps and means nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
#: `motility_class` as an integer: napari's colour-by cannot use a
#: string column.
MOTILITY_CODES = {"unclassified": 0, "constrained": 1, "diffusive": 2, "directed": 3}

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


PROPERTY_COLUMNS = [
    "track_id", "n_spots", "first_frame", "last_frame", "n_gap_frames", "censored",
    "lifetime_frames", "lifetime_s",
    "path_length_um", "net_displacement_um", "mean_speed_um_s",
    "D_3d", "alpha_3d", "D_xy", "alpha_xy", "msd_lags_fitted",
    "motility_class", "motility_code",
    "n_intensity_points", "mean_intensity", "max_intensity",
    "n_steps", "mean_step_height",
]
MSD_COLUMNS = ["track_id", "dims", "lag_frames", "lag_s", "msd_um2", "sem", "n_pairs", "msd_fit"]
STEP_COLUMNS = [
    "track_id", "frame", "level_before", "level_after",
    "step_height", "dwell_before", "dwell_after",
]
LIFETIME_COLUMNS = ["lifetime_frames", "lifetime_s", "count", "fraction"]

#: Everything a results folder holds — and all a run may ever replace.
RESULT_FILES = (
    "track_properties.csv", "lifetime_distribution.csv", "msd_curves.csv",
    "steps.csv", "analysis.json",
)


def holds_only_result_files(path) -> bool:
    """True when `path` is a real folder (not a link) whose every entry is
    a regular file named in `RESULT_FILES` — nothing else, no subfolders.
    An empty folder qualifies."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        return False
    return all(
        entry.name in RESULT_FILES and entry.is_file() and not entry.is_symlink()
        for entry in path.iterdir()
    )


def is_results_folder(path) -> bool:
    """True only for a folder that is recognisably an earlier run's
    results: `analysis.json` plus nothing but other `RESULT_FILES`. The
    one kind of existing folder a run may delete."""
    return holds_only_result_files(path) and (Path(path) / "analysis.json").is_file()


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
    row["motility_code"] = MOTILITY_CODES[row["motility_class"]]

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
