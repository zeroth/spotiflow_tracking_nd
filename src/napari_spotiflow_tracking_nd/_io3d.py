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
    without decoding pixel data.

    Raises:
        ValueError: the file is not a 3D (Z, Y, X) volume. Pointing the
            loader at a folder of 2D planes would otherwise build a
            (T, Y, X) stack that later gets a 4-element `scale`, failing
            deep inside napari instead of here with a clear message.
    """
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        shape = tuple(series.shape)
        if len(shape) != 3:
            raise ValueError(
                f"Expected a 3D (Z, Y, X) volume, got shape {shape} from "
                f"{Path(path).name!r}; this plugin streams one 3D volume "
                "per timepoint file."
            )
        return shape, series.dtype


def read_volume(path: str | Path) -> np.ndarray:
    """Read one (Z, Y, X) volume off disk, eagerly.

    The lazy stack is for viewing; this is for the quick-preview path,
    which deliberately pulls a single timepoint into memory instead of
    streaming the whole series.
    """
    return tifffile.imread(str(path))


def build_lazy_stack_from_files(
    files: list[str | Path],
) -> tuple[da.Array, list[Path]]:
    """Build a lazy (T, Z, Y, X) dask array over an explicit, already-ordered
    list of per-timepoint TIFFs, one delayed block per file.

    `load_folder_as_lazy_stack` is this plus a folder glob; callers that
    already hold the exact file list (a worker's written outputs, say)
    should use this so an unrelated `.tif` sitting in the same folder
    can't silently join the series.

    Every file's (shape, dtype) is checked against the first file's; a
    mismatch raises `ValueError` naming the offending file rather than
    silently reshaping or truncating the series.

    Returns:
        (stack, files) where `files[t]` is the source path for `stack[t]`.

    Raises:
        ValueError: `files` is empty, or a file's (shape, dtype) differs
            from the first file's.
    """
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("No files given; cannot build a timeseries stack.")
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


def load_folder_as_lazy_stack(
    folder: str | Path, pattern: str = "*.tif"
) -> tuple[da.Array, list[Path]]:
    """Build a lazy (T, Z, Y, X) dask array over a folder of per-timepoint
    TIFFs, ordered by the `_T_<n>_` token in each filename.

    Returns:
        (stack, files) where `files[t]` is the source path for `stack[t]`.

    Raises:
        FileNotFoundError: no file in `folder` matches `pattern`.
        ValueError: a file has no `_T_<n>_` token, is not 3D, or does not
            match the first file's (shape, dtype).
    """
    return build_lazy_stack_from_files(list_timepoint_files(folder, pattern))


def compute_layer_scale(
    pixel_size_xy_um: float | None, voxel_size_z_um: float | None
) -> tuple[float, float, float, float]:
    """A napari Image layer's (T, Z, Y, X) `scale`. T is never scaled.
    Returns unit scale unless *both* spatial values are given — half a
    calibration is no calibration."""
    if pixel_size_xy_um is None or voxel_size_z_um is None:
        return (1.0, 1.0, 1.0, 1.0)
    return (1.0, float(voxel_size_z_um), float(pixel_size_xy_um), float(pixel_size_xy_um))


def isotropic_zyx(
    xy_value: float, pixel_size_xy_um: float, voxel_size_z_um: float
) -> tuple[float, float, float]:
    """A lateral length in voxels, as `(z, y, x)` voxels spanning the same
    physical distance on every axis.

    Voxels are rarely cubes — at 0.104 µm laterally against 0.210 µm
    axially, 4 voxels span 0.416 µm across x/y but 0.840 µm along z — so a
    σ or a search range that should be isotropic in µm needs fewer voxels
    along z. Shared by the Pre-Processing widget's Link σ and the Tracking
    widget's Link search range.
    """
    xy = float(xy_value)
    return (xy * (float(pixel_size_xy_um) / float(voxel_size_z_um)), xy, xy)


def validate_output_dir(folder, src_files) -> tuple[Path | None, str | None]:
    """Check an output folder, returning `(path, None)` or `(None, error)`.

    Every writer in this plugin — pre-processing, and the detection pass's
    label masks — writes `output_dir / input_file.name`, mirroring the
    input filenames exactly. An output folder that *is* the input folder
    would therefore overwrite the source acquisition in place, file by
    file, with no confirmation and no undo. The folder pickers are
    adjacent `Browse…` dialogs, so that slip is a plausible one.

    No Qt here: the caller turns the message into `show_error`.
    """
    if not folder:
        return None, "No output folder selected."

    try:
        out = Path(folder).resolve()
    except OSError as exc:
        return None, str(exc)

    if src_files:
        try:
            src_dir = Path(src_files[0]).resolve().parent
        except OSError:
            src_dir = None
        if src_dir is not None and out == src_dir:
            return None, (
                "Output folder is the input folder — pick a different one "
                "to avoid overwriting your source data."
            )
    return out, None
