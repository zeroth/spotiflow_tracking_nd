"""Link the detection pass's spots into 3D tracks — logic only, no Qt.

The detection CSV holds one row per spot per timepoint, in voxel
coordinates. Linking runs in those same voxel coordinates, with a
per-axis search range so an anisotropic voxel does not make the range
anisotropic in µm; physical units are added afterwards, as extra columns,
never by rewriting the coordinates linking used.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd
import trackpy as tp
from trackpy.linking import SubnetOversizeException

#: Columns a detection CSV must have to be linked.
REQUIRED_COLUMNS = ("frame", "z", "y", "x")
#: The Gaussian-fit centre, present when detection ran with fitting on.
FIT_COLUMNS = ("fit_z", "fit_y", "fit_x")
#: The coordinates linking actually uses, chosen per row.
LINK_COLUMNS = ("link_z", "link_y", "link_x")


def missing_columns(df: pd.DataFrame) -> list[str]:
    """The `REQUIRED_COLUMNS` that `df` lacks, in order."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def choose_link_coordinates(df: pd.DataFrame, use_fit: bool = True) -> pd.DataFrame:
    """Add `link_z/y/x` and `coord_source` to a copy of `df`.

    A row links on its fitted centre only when fitting is wanted, the fit
    converged, and all three fitted values are finite; otherwise on
    Spotiflow's detected `z, y, x`. The original columns are untouched.
    """
    out = df.copy()
    # `copy=True`: under pandas copy-on-write `to_numpy` may hand back a
    # read-only view, and the fitted rows are written into this array.
    coords = out[["z", "y", "x"]].to_numpy(dtype=float, copy=True)
    source = np.full(len(out), "detected", dtype=object)
    if use_fit and all(c in out.columns for c in FIT_COLUMNS) and "converged" in out.columns:
        fitted = out[list(FIT_COLUMNS)].to_numpy(dtype=float, copy=True)
        # `.eq(True)`, not `.astype(bool)`: a blank cell reads back as NaN,
        # and `bool(nan)` is True — a missing fit would count as converged.
        ok = out["converged"].eq(True).to_numpy() & np.isfinite(fitted).all(axis=1)
        coords[ok] = fitted[ok]
        source[ok] = "fit"
    for i, name in enumerate(LINK_COLUMNS):
        out[name] = coords[:, i]
    out["coord_source"] = source
    return out


def iter_link(
    df: pd.DataFrame,
    search_range_zyx: tuple[float, float, float],
    memory: int,
    *,
    adaptive: bool = False,
    should_stop: Callable[[], bool] = lambda: False,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Link `df` (with `LINK_COLUMNS`) frame by frame, yielding
    `(frame, linked_rows)` for every frame from the first to the last.

    Frames with no spots are fed to trackpy as empty frames rather than
    skipped: trackpy counts `memory` in frames *it has seen*, so skipping
    one would let a track bridge a gap `memory` should have broken.

    `should_stop` is checked before each frame; returning True ends the
    iteration cleanly after the frames already yielded.

    Raises:
        ValueError: a region is too dense to link at this search range
            (trackpy's `SubnetOversizeException`), with what to change.
    """
    frames = df["frame"].astype(int)
    groups = {int(t): g for t, g in df.groupby(frames, sort=True)}
    if not groups:
        return
    first, last = min(groups), max(groups)
    empty = df.iloc[0:0]
    order = range(first, last + 1)

    def _feed():
        for t in order:
            if should_stop():
                return
            yield groups.get(t, empty)

    search_range = tuple(float(v) for v in search_range_zyx)
    kwargs = {}
    if adaptive:
        kwargs = {"adaptive_stop": 0.1 * min(search_range), "adaptive_step": 0.95}

    # trackpy otherwise logs a line per frame — 349 of them per run.
    tp.quiet()
    try:
        linked = tp.link_df_iter(
            _feed(),
            search_range=search_range,
            memory=memory,
            pos_columns=list(LINK_COLUMNS),
            t_column="frame",
            **kwargs,
        )
        for t, frame_df in zip(order, linked):
            yield t, frame_df
    except SubnetOversizeException as exc:
        raise ValueError(
            "Too many candidate links in a dense region to solve at this "
            "search range — lower the search range, or tick Adaptive search."
        ) from exc


def filter_short_tracks(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Drop tracks with fewer than `min_length` spots."""
    if min_length <= 1 or df.empty:
        return df
    lengths = df.groupby("particle")["particle"].transform("size")
    return df[lengths >= min_length].reset_index(drop=True)


def flag_censored(
    df: pd.DataFrame, first_frame: int, last_frame: int, memory: int = 0,
) -> pd.DataFrame:
    """Add `censored`: True on every row of a track that starts within
    `memory` frames of `first_frame` or ends within `memory` frames of
    `last_frame`. Its true lifetime is unknown — it may have started before
    the series or outlived it, and with `memory` > 0 a track missing for up
    to `memory` frames would still have been continued past the edge."""
    out = df.copy()
    if out.empty:
        out["censored"] = pd.Series(dtype=bool)
        return out
    by_track = out.groupby("particle")["frame"]
    starts, ends = by_track.transform("min"), by_track.transform("max")
    out["censored"] = (starts <= first_frame + memory) | (ends >= last_frame - memory)
    return out


def add_physical_columns(df: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    """Add `z_um, y_um, x_um, time_s` from the link coordinates — each
    only when its own calibration value is known (not None, not 0)."""
    out = df.copy()
    pixel_xy = calibration.get("pixel_size_xy_um")
    voxel_z = calibration.get("voxel_size_z_um")
    frame_s = calibration.get("frame_interval_s")
    if voxel_z:
        out["z_um"] = out["link_z"] * voxel_z
    if pixel_xy:
        out["y_um"] = out["link_y"] * pixel_xy
        out["x_um"] = out["link_x"] * pixel_xy
    if frame_s:
        out["time_s"] = out["frame"] * frame_s
    return out
