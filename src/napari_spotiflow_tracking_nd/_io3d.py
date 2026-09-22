from __future__ import annotations

import re
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile
from dask import delayed

_TIMEPOINT_RE = re.compile(r"_T_(\d+)_")


def list_timepoint_files(folder: str | Path, pattern: str = "*.tif") -> list[Path]:
    """Return files in `folder` matching `pattern`, ordered by the numeric
    `_T_<n>_` token in their filename (a plain lexicographic sort would put
    `_T_10_` before `_T_9_`).

    Raises:
        FileNotFoundError: no file in `folder` matches `pattern`.
        ValueError: a matching file has no `_T_<n>_` token in its name.
    """
    folder = Path(folder)
    files = list(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")

    def _timepoint(path: Path) -> int:
        match = _TIMEPOINT_RE.search(path.name)
        if match is None:
            raise ValueError(
                f"{path.name!r} has no '_T_<n>_' timepoint token in its "
                "filename; cannot order it into a timeseries."
            )
        return int(match.group(1))

    return sorted(files, key=_timepoint)


def probe_volume_shape(path: str | Path) -> tuple[tuple[int, int, int], np.dtype]:
    """Read a (Z, Y, X) volume's shape and dtype from its TIFF header,
    without decoding pixel data."""
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        return tuple(series.shape), series.dtype


def load_folder_as_lazy_stack(
    folder: str | Path, pattern: str = "*.tif"
) -> tuple[da.Array, list[Path]]:
    """Build a lazy (T, Z, Y, X) dask array over a folder of per-timepoint
    TIFFs, one delayed block per file, ordered by `_T_<n>_`.

    Every file's (shape, dtype) is checked against the first file's; a
    mismatch raises `ValueError` naming the offending file rather than
    silently reshaping or truncating the series.

    Returns:
        (stack, files) where `files[t]` is the source path for `stack[t]`.
    """
    files = list_timepoint_files(folder, pattern)
    ref_shape, ref_dtype = probe_volume_shape(files[0])

    blocks = []
    for path in files:
        shape, dtype = probe_volume_shape(path)
        if (shape, dtype) != (ref_shape, ref_dtype):
            raise ValueError(
                f"{path.name!r} has shape {shape}/{dtype}, expected "
                f"{ref_shape}/{ref_dtype} (from {files[0].name!r}); "
                "ragged timeseries are not supported."
            )
        block = da.from_delayed(
            delayed(tifffile.imread)(str(path)), shape=ref_shape, dtype=ref_dtype
        )
        blocks.append(block)

    return da.stack(blocks, axis=0), files


def compute_layer_scale(
    pixel_size_xy_um: float | None, voxel_size_z_um: float | None
) -> tuple[float, float, float, float]:
    """A napari Image layer's (T, Z, Y, X) `scale`. T is never scaled.
    Returns unit scale unless *both* spatial values are given — half a
    calibration is no calibration."""
    if pixel_size_xy_um is None or voxel_size_z_um is None:
        return (1.0, 1.0, 1.0, 1.0)
    return (1.0, float(voxel_size_z_um), float(pixel_size_xy_um), float(pixel_size_xy_um))
