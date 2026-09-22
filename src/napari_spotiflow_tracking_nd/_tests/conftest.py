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
