from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def write_timepoint_folder(
    folder: Path,
    n_timepoints: int = 3,
    shape: tuple[int, int, int] = (8, 32, 32),
    dtype=np.uint16,
    prefix: str = "stack",
) -> list[Path]:
    """Write `n_timepoints` synthetic ZYX volumes as one TIFF per timepoint,
    named like the real dataset (`..._T_000_C_00.tif`), and return the
    written paths in timepoint order.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for t in range(n_timepoints):
        volume = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
        path = folder / f"{prefix}_T_{t:03d}_C_00.tif"
        tifffile.imwrite(str(path), volume)
        paths.append(path)
    return paths


def add_layer_from_folder(viewer, folder: Path):
    """Add `folder` to `viewer` exactly as the napari reader
    (`_reader3d.get_reader`) would: a lazy image layer, unit scale, and
    the three calibration metadata keys present but `None`.

    Tests that only care about widget/worker behavior use this instead of
    driving the reader itself, so they don't have to re-derive what the
    reader returns; `test_reader3d.py` is what actually exercises the
    reader.
    """
    from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

    stack, files = load_folder_as_lazy_stack(folder)
    return viewer.add_image(
        stack,
        name=Path(folder).name,
        metadata={
            "source_files": files,
            "pixel_size_xy_um": None,
            "voxel_size_z_um": None,
            "frame_interval_s": None,
        },
    )
