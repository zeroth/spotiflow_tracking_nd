from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import sys
import platform

import numpy as np
import tifffile
import pytest


def pytest_configure(config):
    """Mock torch and spotiflow on Windows before any tests run to prevent
    DLL load failures during @patch decorators.
    """
    if platform.system() != "Windows":
        return

    # Mock torch before spotiflow tries to import it
    if "torch" not in sys.modules:
        mock_torch = MagicMock()
        sys.modules["torch"] = mock_torch
        sys.modules["torch.nn"] = MagicMock()
        sys.modules["torch.optim"] = MagicMock()
        sys.modules["torch.cuda"] = MagicMock()

    # Mock spotiflow modules if not already imported
    if "spotiflow" not in sys.modules:
        sys.modules["spotiflow"] = MagicMock()
        sys.modules["spotiflow.model"] = MagicMock()
        sys.modules["spotiflow.model.backbones"] = MagicMock()
        sys.modules["spotiflow.model.backbones.resnet"] = MagicMock()


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
