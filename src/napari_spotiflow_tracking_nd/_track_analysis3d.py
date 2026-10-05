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
