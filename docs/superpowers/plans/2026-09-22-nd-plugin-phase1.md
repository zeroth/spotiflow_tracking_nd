# napari-spotiflow-tracking-nd Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 slice of `napari-spotiflow-tracking-nd`: a lazy TIFF-folder loader, 3D pre-processing (background removal, Gaussian blur, arithmetic), Spotiflow 3D spot detection, and the two napari dock widgets that drive them — all streaming one volume at a time so a 110 GB timeseries never has to fit in RAM.

**Architecture:** Pure-Python logic modules (no Qt) do the array math and are unit-tested directly; two `QThread` workers stream files through those functions one at a time; two thin Qt widgets wire buttons/comboboxes to the workers. Data on disk stays TIFF-in/TIFF-out (one file per timepoint) — no zarr conversion.

**Tech Stack:** napari, qtpy, numpy, dask, tifffile, scipy, torch (via spotiflow, optional/lazy for the background-removal GPU path), spotiflow, pandas, pytest + pytest-qt.

**Spec:** [docs/superpowers/specs/2026-09-22-nd-plugin-phase1-design.md](../specs/2026-09-22-nd-plugin-phase1-design.md)

## Global Constraints

- No shared runtime dependency on `napari-spotiflow-tracking` (the 2D package) — ported helpers (`_get_device`, `_rebaseline`, `apply_calculation`) are copied into this package, not imported cross-package.
- Pre-processing output is one TIFF per volume, written to an output folder whose filenames mirror the input filenames exactly (e.g. `..._T_000_C_00.tif` in → same name out). No zarr anywhere in this phase.
- Calibration (`pixel_size_xy_um`, `voxel_size_z_um`, `frame_interval_s`) is optional and per-dataset — `None`/unset unless the user enters it in the widget. Never hard-code the example dataset's values (0.104 µm xy / 0.210 µm z / 3 s) into logic code; they only appear as widget defaults/test fixtures.
- A napari `Image` layer's `scale` is `(1.0, voxel_size_z_um, pixel_size_xy_um, pixel_size_xy_um)` — the T axis is never scaled.
- Every logic module is unit-tested with small synthetic volumes (no GPU, no real Spotiflow weights, no real 110 GB dataset required by any test).
- Package layout mirrors the sibling repo: `src/napari_spotiflow_tracking_nd/`, `pytest` testpaths under `_tests/`, npe2 `napari.yaml` manifest, setuptools `src` layout.

---

### Task 1: Project scaffold + lazy TIFF-folder loader (`_io3d.py`)

**Files:**
- Create: `pyproject.toml`
- Create: `src/napari_spotiflow_tracking_nd/__init__.py`
- Create: `src/napari_spotiflow_tracking_nd/_version.py`
- Create: `src/napari_spotiflow_tracking_nd/_io3d.py`
- Create: `src/napari_spotiflow_tracking_nd/_tests/__init__.py`
- Create: `src/napari_spotiflow_tracking_nd/_tests/conftest.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_io3d.py`

**Interfaces:**
- Produces: `list_timepoint_files(folder: str | Path, pattern: str = "*.tif") -> list[Path]`; `probe_volume_shape(path: str | Path) -> tuple[tuple[int, int, int], np.dtype]`; `load_folder_as_lazy_stack(folder: str | Path, pattern: str = "*.tif") -> tuple[dask.array.Array, list[Path]]`; `compute_layer_scale(pixel_size_xy_um: float | None, voxel_size_z_um: float | None) -> tuple[float, float, float, float]`. All in `napari_spotiflow_tracking_nd._io3d`. Later tasks import these four names.
- Consumes: nothing (first task).

- [ ] **Step 1: Scaffold the package**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "napari-spotiflow-tracking-nd"
version = "0.1.0"
description = "Napari plugin for 3D-volume-timeseries spot detection and tracking using Spotiflow and trackpy"
requires-python = ">=3.9"
license = { text = "BSD-3-Clause" }
dependencies = [
    "numpy",
    "spotiflow>=0.6.0",
    "qtpy",
    "superqt",
    "pandas",
    "dask",
    "scipy",
    "tifffile",
]

[project.optional-dependencies]
testing = [
    "pytest",
    "pytest-cov",
    "pytest-qt",
    "napari",
    "pyqt5",
]
dev = [
    "pre-commit",
    "ruff",
]

[project.entry-points."napari.manifest"]
napari-spotiflow-tracking-nd = "napari_spotiflow_tracking_nd:napari.yaml"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
napari_spotiflow_tracking_nd = ["napari.yaml"]

[tool.pytest.ini_options]
testpaths = ["src/napari_spotiflow_tracking_nd/_tests"]
```

Create `src/napari_spotiflow_tracking_nd/__init__.py`:

```python
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
```

Create `src/napari_spotiflow_tracking_nd/_version.py`:

```python
version = "0.1.0"
```

Create `src/napari_spotiflow_tracking_nd/_tests/__init__.py` (empty file).

Create `src/napari_spotiflow_tracking_nd/_tests/conftest.py`:

```python
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
```

- [ ] **Step 2: Write the failing tests for `_io3d.py`**

Create `src/napari_spotiflow_tracking_nd/_tests/test_io3d.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


class TestListTimepointFiles:
    def test_orders_numerically_not_lexicographically(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import list_timepoint_files

        # Write T_9 and T_10 so a lexicographic sort ("T_10" < "T_9") would
        # misorder them; the real dataset goes past 99 timepoints too.
        for t in (10, 2, 9):
            tifffile.imwrite(str(tmp_path / f"stack_T_{t:03d}_C_00.tif"),
                              np.zeros((2, 4, 4), dtype=np.uint16))

        files = list_timepoint_files(tmp_path)

        assert [f.name for f in files] == [
            "stack_T_002_C_00.tif", "stack_T_009_C_00.tif", "stack_T_010_C_00.tif",
        ]

    def test_no_matching_files_raises(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import list_timepoint_files

        with pytest.raises(FileNotFoundError):
            list_timepoint_files(tmp_path)

    def test_file_without_timepoint_token_raises(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import list_timepoint_files

        tifffile.imwrite(str(tmp_path / "no_token_here.tif"),
                          np.zeros((2, 4, 4), dtype=np.uint16))

        with pytest.raises(ValueError, match="no_token_here.tif"):
            list_timepoint_files(tmp_path)


class TestProbeVolumeShape:
    def test_returns_shape_and_dtype_without_reading_pixels(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import probe_volume_shape

        path = tmp_path / "stack_T_000_C_00.tif"
        tifffile.imwrite(str(path), np.zeros((5, 10, 12), dtype=np.uint16))

        shape, dtype = probe_volume_shape(path)

        assert shape == (5, 10, 12)
        assert dtype == np.uint16


class TestLoadFolderAsLazyStack:
    def test_builds_a_lazy_t_z_y_x_stack(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

        write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))

        stack, files = load_folder_as_lazy_stack(tmp_path)

        assert stack.shape == (3, 4, 8, 8)
        assert stack.dtype == np.uint16
        assert len(files) == 3

    def test_stack_is_lazy_until_computed(self, tmp_path):
        import dask.array as da
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))

        stack, _ = load_folder_as_lazy_stack(tmp_path)

        assert isinstance(stack, da.Array)

    def test_values_match_the_source_files(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

        paths = write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))
        expected_t0 = tifffile.imread(str(paths[0]))

        stack, files = load_folder_as_lazy_stack(tmp_path)

        assert np.array_equal(np.asarray(stack[0]), expected_t0)
        assert files[0] == paths[0]

    def test_mismatched_volume_raises_with_offending_filename(self, tmp_path):
        from napari_spotiflow_tracking_nd._io3d import load_folder_as_lazy_stack

        write_timepoint_folder(tmp_path, n_timepoints=1, shape=(4, 8, 8))
        tifffile.imwrite(str(tmp_path / "stack_T_001_C_00.tif"),
                          np.zeros((4, 8, 9), dtype=np.uint16))  # X differs

        with pytest.raises(ValueError, match="stack_T_001_C_00.tif"):
            load_folder_as_lazy_stack(tmp_path)


class TestComputeLayerScale:
    def test_full_calibration_gives_z_y_x_scale(self):
        from napari_spotiflow_tracking_nd._io3d import compute_layer_scale

        assert compute_layer_scale(pixel_size_xy_um=0.104, voxel_size_z_um=0.210) == (
            1.0, 0.210, 0.104, 0.104,
        )

    def test_missing_calibration_gives_unit_scale(self):
        from napari_spotiflow_tracking_nd._io3d import compute_layer_scale

        assert compute_layer_scale(None, None) == (1.0, 1.0, 1.0, 1.0)
        assert compute_layer_scale(0.104, None) == (1.0, 1.0, 1.0, 1.0)
        assert compute_layer_scale(None, 0.210) == (1.0, 1.0, 1.0, 1.0)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_io3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._io3d'` (the package itself isn't importable yet either, so install it editable first: `pip install -e ".[testing]"`, then re-run to see the `_io3d` import fail).

- [ ] **Step 4: Implement `_io3d.py`**

Create `src/napari_spotiflow_tracking_nd/_io3d.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pip install -e ".[testing]"` then `pytest src/napari_spotiflow_tracking_nd/_tests/test_io3d.py -v`
Expected: PASS, all tests green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/napari_spotiflow_tracking_nd
git commit -m "feat(nd): scaffold package + lazy TIFF-folder loader"
```

---

### Task 2: 3D pre-processing logic (`_preprocessing3d.py`)

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_preprocessing3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_preprocessing3d.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent logic module).
- Produces: `remove_background_high_pass_3d(volume: np.ndarray, sigma: float | tuple[float, float, float] = 10.0, device: str = "cpu") -> np.ndarray`; `gaussian_blur_3d(volume: np.ndarray, sigma: float | tuple[float, float, float]) -> np.ndarray`; `apply_calculation(image: np.ndarray, op: str, value: int) -> np.ndarray`. All in `napari_spotiflow_tracking_nd._preprocessing3d`. Task 4's workers and Task 5's widget call these three by name.

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_preprocessing3d.py`:

```python
from __future__ import annotations

import sys

import numpy as np
import pytest


@pytest.fixture(params=["torch", "scipy"])
def backend(request, monkeypatch):
    """Run background-removal tests against both the torch path and the
    scipy fallback (forced by making `import torch` raise ImportError)."""
    if request.param == "scipy":
        monkeypatch.setitem(sys.modules, "torch", None)
    return request.param


def _blob_volume(dtype=np.uint16):
    vol = np.full((16, 32, 32), 100, dtype=np.float64)
    vol[6:10, 12:20, 12:20] = 3000.0
    return vol.astype(dtype)


class TestRemoveBackgroundHighPass3D:
    def test_preserves_input_dtype(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.uint16)
        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.dtype == np.uint16
        assert out.shape == vol.shape

    def test_restores_the_original_mean(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.uint16)
        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.mean() == pytest.approx(vol.mean(), rel=0.2)

    def test_integer_result_does_not_wrap_around(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.uint16)
        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.min() >= 0
        assert out.max() <= int(vol.max()) + int(round(vol.mean())) + 1

    def test_accepts_anisotropic_sigma_tuple(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.uint16)
        out = remove_background_high_pass_3d(vol, sigma=(2.0, 4.0, 4.0))
        assert out.shape == vol.shape

    def test_float_input_is_not_clipped(self, backend):
        from napari_spotiflow_tracking_nd._preprocessing3d import (
            remove_background_high_pass_3d,
        )

        vol = _blob_volume(np.float32)
        out = remove_background_high_pass_3d(vol, sigma=3.0)
        assert out.dtype == np.float32
        assert out.min() < 0


class TestGaussianBlur3D:
    def test_blurs_and_preserves_dtype(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        vol = _blob_volume(np.uint16)
        out = gaussian_blur_3d(vol, sigma=2.0)
        assert out.dtype == np.uint16
        assert out.shape == vol.shape
        # Blurring spreads the peak: the max drops, the total mass is kept.
        assert out.max() < vol.max()
        assert out.sum() == pytest.approx(vol.sum(), rel=0.05)

    def test_non_positive_scalar_sigma_returns_input_unchanged(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        vol = _blob_volume(np.uint16)
        out = gaussian_blur_3d(vol, sigma=0.0)
        assert out is vol

    def test_all_non_positive_tuple_sigma_returns_input_unchanged(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        vol = _blob_volume(np.uint16)
        out = gaussian_blur_3d(vol, sigma=(0.0, 0.0, 0.0))
        assert out is vol

    def test_anisotropic_sigma_blurs_axes_independently(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        vol = _blob_volume(np.uint16)
        blurred_z_only = gaussian_blur_3d(vol, sigma=(4.0, 0.001, 0.001))
        blurred_isotropic = gaussian_blur_3d(vol, sigma=(4.0, 4.0, 4.0))
        # Blurring only along Z spreads less total variance than blurring
        # all three axes, so the isotropic peak is lower.
        assert blurred_isotropic.max() <= blurred_z_only.max()

    def test_wrong_ndim_raises(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import gaussian_blur_3d

        with pytest.raises(ValueError, match="3D"):
            gaussian_blur_3d(np.zeros((8, 8), dtype=np.uint16), sigma=2.0)


class TestApplyCalculation:
    def test_add_clips_at_dtype_max(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        vol = np.full((4, 8, 8), 65530, dtype=np.uint16)
        out = apply_calculation(vol, "add", 100)
        assert out.dtype == np.uint16
        assert out.max() == 65535

    def test_subtract_clips_at_zero(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        vol = np.full((4, 8, 8), 5, dtype=np.uint16)
        out = apply_calculation(vol, "subtract", 100)
        assert out.min() == 0

    def test_unknown_op_raises(self):
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        with pytest.raises(ValueError):
            apply_calculation(np.zeros((2, 4, 4), dtype=np.uint16), "multiply", 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_preprocessing3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._preprocessing3d'`

- [ ] **Step 3: Implement `_preprocessing3d.py`**

Create `src/napari_spotiflow_tracking_nd/_preprocessing3d.py`:

```python
from __future__ import annotations

import numpy as np


def _get_device(use_gpu: bool = False) -> str:
    """Return 'cuda' if requested and available, else 'cpu'."""
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def _rebaseline(residual: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Shift a zero-centred residual onto the original's mean, in its dtype.

    See the 2D plugin's `_preprocessing._rebaseline` for the rationale —
    ported unchanged: integer dtypes are clipped to their representable
    range before the cast; floats are cast without clipping.
    """
    out = residual + float(np.mean(original))
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        out = np.clip(out, info.min, info.max)
    return out.astype(original.dtype)


def _sigma_zyx(sigma: float | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(sigma, (int, float)):
        return (float(sigma), float(sigma), float(sigma))
    sz, sy, sx = sigma
    return (float(sz), float(sy), float(sx))


def remove_background_high_pass_3d(
    volume: np.ndarray,
    sigma: float | tuple[float, float, float] = 10.0,
    device: str = "cpu",
) -> np.ndarray:
    """Remove background from a single (Z, Y, X) volume by subtracting a
    Gaussian low-pass filtered copy.

    Uses a separable PyTorch conv3d for GPU acceleration when torch is
    importable; falls back to `scipy.ndimage.gaussian_filter` on CPU
    otherwise. The subtraction is re-baselined onto the volume's own mean
    (see `_rebaseline`) exactly as the 2D plugin does per-frame.

    Args:
        volume: 3D array (Z, Y, X).
        sigma: one float (isotropic) or (sigma_z, sigma_y, sigma_x) — z and
            xy sampling often differ, so anisotropic sigma is supported.
        device: 'cuda' for GPU, 'cpu' for CPU.

    Returns:
        Background-subtracted volume, re-baselined and cast back to the
        input dtype.
    """
    sigma_z, sigma_y, sigma_x = _sigma_zyx(sigma)
    try:
        import torch
        import torch.nn.functional as F
        return _remove_background_torch_3d(volume, sigma_z, sigma_y, sigma_x, device, torch, F)
    except ImportError:
        from scipy.ndimage import gaussian_filter
        background = gaussian_filter(
            volume.astype(np.float64), sigma=(sigma_z, sigma_y, sigma_x)
        )
        return _rebaseline(volume.astype(np.float64) - background, volume)


def _remove_background_torch_3d(volume, sigma_z, sigma_y, sigma_x, device, torch, F):
    """Separable 3D Gaussian background subtraction via torch conv3d."""

    def _kernel(sigma):
        radius = max(int(np.ceil(sigma * 2)), 1)
        size = 2 * radius + 1
        t = torch.linspace(-2, 2, size, dtype=torch.float32)
        h = torch.exp(-t ** 2)
        return h / h.sum(), radius

    kz, rz = _kernel(sigma_z)
    ky, ry = _kernel(sigma_y)
    kx, rx = _kernel(sigma_x)

    wz = kz.reshape(1, 1, -1, 1, 1)
    wy = ky.reshape(1, 1, 1, -1, 1)
    wx = kx.reshape(1, 1, 1, 1, -1)

    x = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

    if device == "cuda":
        x = x.cuda()
        wz, wy, wx = wz.cuda(), wy.cuda(), wx.cuda()

    y = F.pad(x, pad=(0, 0, 0, 0, rz, rz), mode="reflect")
    y = F.conv3d(y, weight=wz, padding="valid")
    y = F.pad(y, pad=(0, 0, ry, ry, 0, 0), mode="reflect")
    y = F.conv3d(y, weight=wy, padding="valid")
    y = F.pad(y, pad=(rx, rx, 0, 0, 0, 0), mode="reflect")
    y = F.conv3d(y, weight=wx, padding="valid")

    residual = (x - y).squeeze().cpu().numpy().astype(np.float64)
    return _rebaseline(residual, volume)


def gaussian_blur_3d(
    volume: np.ndarray, sigma: float | tuple[float, float, float]
) -> np.ndarray:
    """3D Gaussian blur of a single (Z, Y, X) volume.

    `sigma` is one float (isotropic) or (sigma_z, sigma_y, sigma_x). A
    non-positive scalar, or an all non-positive tuple, returns the input
    unchanged — extending the 2D plugin's `sigma <= 0` convention to 3D's
    anisotropic sigma. Computed in float64, cast back to the input dtype.

    Raises:
        ValueError: if `volume` is not 3D.
    """
    from scipy.ndimage import gaussian_filter

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X) volume, got {volume.ndim}D")

    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            return volume
        sigma_arg = float(sigma)
    else:
        sigma_arg = tuple(float(s) for s in sigma)
        if all(s <= 0 for s in sigma_arg):
            return volume

    return gaussian_filter(volume.astype(np.float64), sigma=sigma_arg).astype(volume.dtype)


def apply_calculation(image: np.ndarray, op: str, value: int) -> np.ndarray:
    """Add or subtract an integer scalar from an image or volume.

    Ported unchanged from the 2D plugin — already shape-agnostic. Returns
    a new array with the same dtype as the input; integer dtypes clip to
    their representable range, floats do not.

    Raises:
        ValueError: if `op` is not one of "add" or "subtract".
    """
    if op not in ("add", "subtract"):
        raise ValueError(f"Unknown op: {op!r}")

    sign = 1 if op == "add" else -1

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        wide = image.astype(np.int64) + sign * int(value)
        clipped = np.clip(wide, info.min, info.max)
        return clipped.astype(image.dtype)

    return (image + sign * value).astype(image.dtype, copy=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_preprocessing3d.py -v`
Expected: PASS, all tests green (runs twice per background-removal test — once per `backend` param).

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_preprocessing3d.py src/napari_spotiflow_tracking_nd/_tests/test_preprocessing3d.py
git commit -m "feat(nd): 3D background removal, Gaussian blur, arithmetic"
```

---

### Task 3: Spotiflow 3D detection wrapper (`_segmentation3d.py`)

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_segmentation3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_segmentation3d.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `load_model(model_name: str, device: str = "cpu")`; `detect_spots_3d(volume: np.ndarray, model, prob_thresh: float | None = 0.5, min_distance: int = 2) -> tuple[np.ndarray, object]` (points is `(N, 3)` `[z, y, x]`; `details.prob` is the parallel `(N,)` probability array). Both in `napari_spotiflow_tracking_nd._segmentation3d`. Task 4's `DetectionWorker3D` calls both by name.

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_segmentation3d.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def _make_mock_model():
    model = MagicMock()
    details = MagicMock()
    details.prob = np.array([0.9, 0.7])
    model.predict.return_value = (
        np.array([[3, 10, 20], [5, 30, 40]]),  # 2 spots at (z, y, x)
        details,
    )
    return model


class TestDetectSpots3D:
    def test_returns_points_and_details(self):
        from napari_spotiflow_tracking_nd._segmentation3d import detect_spots_3d

        model = _make_mock_model()
        volume = np.random.rand(16, 64, 64).astype(np.float32)

        points, details = detect_spots_3d(volume, model, prob_thresh=0.5, min_distance=2)

        assert points.shape == (2, 3)
        assert list(points[0]) == [3, 10, 20]
        assert list(details.prob) == [0.9, 0.7]

    def test_calls_model_predict_with_params(self):
        from napari_spotiflow_tracking_nd._segmentation3d import detect_spots_3d

        model = _make_mock_model()
        volume = np.random.rand(16, 64, 64).astype(np.float32)

        detect_spots_3d(volume, model, prob_thresh=0.3, min_distance=5)

        model.predict.assert_called_once_with(
            volume, prob_thresh=0.3, min_distance=5, verbose=False,
        )

    def test_returns_empty_array_when_no_spots(self):
        from napari_spotiflow_tracking_nd._segmentation3d import detect_spots_3d

        model = MagicMock()
        details = MagicMock()
        details.prob = np.empty((0,))
        model.predict.return_value = (np.empty((0, 3)), details)
        volume = np.random.rand(16, 64, 64).astype(np.float32)

        points, details = detect_spots_3d(volume, model)

        assert points.shape == (0, 3)


class TestLoadModel:
    @patch("spotiflow.model.Spotiflow")
    def test_loads_a_registered_3d_pretrained_model(self, MockSpotiflow):
        from napari_spotiflow_tracking_nd._segmentation3d import load_model

        mock_model = MagicMock()
        MockSpotiflow.from_pretrained.return_value = mock_model

        result = load_model("synth_3d", device="cpu")

        MockSpotiflow.from_pretrained.assert_called_once_with("synth_3d", map_location="cpu")
        assert result is mock_model

    @patch("spotiflow.model.Spotiflow")
    def test_loads_a_custom_model_from_path(self, MockSpotiflow):
        from napari_spotiflow_tracking_nd._segmentation3d import load_model

        mock_model = MagicMock()
        MockSpotiflow.from_folder.return_value = mock_model

        result = load_model("/path/to/custom_3d_model", device="cuda")

        MockSpotiflow.from_folder.assert_called_once_with(
            "/path/to/custom_3d_model", map_location="cuda"
        )
        assert result is mock_model
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_segmentation3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._segmentation3d'`

- [ ] **Step 3: Implement `_segmentation3d.py`**

Create `src/napari_spotiflow_tracking_nd/_segmentation3d.py`:

```python
from __future__ import annotations

import numpy as np

# Spotiflow's own pretrained-model registry currently marks these two
# `is_3d=True` (see spotiflow.model.pretrained._REGISTERED). Any other
# registered 3D model added upstream can still be loaded — pass its name
# here too, or use a local checkpoint path, which always goes through
# `Spotiflow.from_folder`.
_PRETRAINED_3D_MODELS = {"synth_3d", "smfish_3d"}


def load_model(model_name: str, device: str = "cpu"):
    """Load a Spotiflow model for 3D detection.

    `model_name` is either a registered pretrained 3D model name
    (`"synth_3d"`, `"smfish_3d"`, ...) or a filesystem path to a local
    checkpoint folder.
    """
    from spotiflow.model import Spotiflow

    if model_name in _PRETRAINED_3D_MODELS:
        return Spotiflow.from_pretrained(model_name, map_location=device)
    return Spotiflow.from_folder(model_name, map_location=device)


def detect_spots_3d(
    volume: np.ndarray,
    model,
    prob_thresh: float | None = 0.5,
    min_distance: int = 2,
) -> tuple[np.ndarray, object]:
    """Detect spots in a single (Z, Y, X) volume using a loaded 3D
    Spotiflow model.

    Returns (points, details) where points is an (N, 3) array of
    [z, y, x] and details.prob is the parallel (N,) probability array
    (Spotiflow's own return shape — see `Spotiflow.predict`).
    """
    points, details = model.predict(
        volume,
        prob_thresh=prob_thresh,
        min_distance=min_distance,
        verbose=False,
    )
    return points, details
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_segmentation3d.py -v`
Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_segmentation3d.py src/napari_spotiflow_tracking_nd/_tests/test_segmentation3d.py
git commit -m "feat(nd): Spotiflow 3D detection wrapper"
```

---

### Task 4: Streaming QThread workers (`_workers3d.py`)

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_workers3d.py`
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py`

**Interfaces:**
- Consumes: `remove_background_high_pass_3d`, `gaussian_blur_3d`, `apply_calculation` from `napari_spotiflow_tracking_nd._preprocessing3d` (Task 2); `detect_spots_3d` from `napari_spotiflow_tracking_nd._segmentation3d` (Task 3).
- Produces: `VolumeOpWorker3D(QThread)` — signals `progress(int, int)` (current, total), `finished(list)` (output file paths as `str`), `errored(str)`; constructor `__init__(self, files: list[Path], output_dir: Path, op: Callable, op_kwargs: dict, parent=None)`; method `cancel()`. `DetectionWorker3D(QThread)` — signals `progress(int, int)`, `finished(str)` (the CSV path written), `errored(str)`; constructor `__init__(self, files: list[Path], model, csv_path: Path, prob_thresh: float | None = 0.5, min_distance: int = 2, parent=None)`; method `cancel()`. Both in `napari_spotiflow_tracking_nd._workers3d`. Task 5 and Task 6's widgets construct and `.start()` these.

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


class TestVolumeOpWorker3D:
    def test_writes_one_output_file_per_input_with_matching_names(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        files = write_timepoint_folder(in_dir, n_timepoints=3, shape=(4, 8, 8))

        progress_calls = []
        finished = {}
        worker = VolumeOpWorker3D(
            files=files, output_dir=out_dir, op=apply_calculation,
            op_kwargs={"op": "add", "value": 5},
        )
        worker.progress.connect(lambda cur, total: progress_calls.append((cur, total)))
        worker.finished.connect(lambda paths: finished.update(paths=paths))

        worker.run()  # call directly — no real threading in the test

        written = sorted(p.name for p in out_dir.iterdir())
        assert written == [f.name for f in files]
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]
        assert len(finished["paths"]) == 3

    def test_output_matches_the_op_applied_to_the_input(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        files = write_timepoint_folder(in_dir, n_timepoints=1, shape=(4, 8, 8))
        original = tifffile.imread(str(files[0]))

        worker = VolumeOpWorker3D(
            files=files, output_dir=out_dir, op=apply_calculation,
            op_kwargs={"op": "add", "value": 7},
        )
        worker.run()

        out = tifffile.imread(str(out_dir / files[0].name))
        expected = apply_calculation(original, "add", 7)
        assert np.array_equal(out, expected)

    def test_cancel_stops_before_the_next_file(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D
        from napari_spotiflow_tracking_nd._preprocessing3d import apply_calculation

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        files = write_timepoint_folder(in_dir, n_timepoints=5, shape=(4, 8, 8))

        worker = VolumeOpWorker3D(
            files=files, output_dir=out_dir, op=apply_calculation,
            op_kwargs={"op": "add", "value": 1},
        )
        # Cancel as soon as the first file's progress is reported.
        worker.progress.connect(lambda cur, total: worker.cancel())
        worker.run()

        assert len(list(out_dir.iterdir())) == 1

    def test_errored_signal_carries_the_exception_message(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        files = write_timepoint_folder(in_dir, n_timepoints=1, shape=(4, 8, 8))

        def _broken_op(volume, **kwargs):
            raise RuntimeError("boom")

        errors = []
        worker = VolumeOpWorker3D(files=files, output_dir=out_dir, op=_broken_op, op_kwargs={})
        worker.errored.connect(errors.append)
        worker.run()

        assert errors == ["boom"]


class TestDetectionWorker3D:
    def test_writes_a_csv_with_frame_z_y_x_prob_columns(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        in_dir = tmp_path / "in"
        files = write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 8, 8))
        csv_path = tmp_path / "spots.csv"

        model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        model.predict.return_value = (np.array([[1, 2, 3]]), details)

        finished = {}
        worker = DetectionWorker3D(files=files, model=model, csv_path=csv_path)
        worker.finished.connect(lambda path: finished.update(path=path))
        worker.run()

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["frame", "z", "y", "x", "prob"]
        assert list(df["frame"]) == [0, 1]
        assert finished["path"] == str(csv_path)

    def test_frame_with_no_spots_writes_no_rows_for_that_frame(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        in_dir = tmp_path / "in"
        files = write_timepoint_folder(in_dir, n_timepoints=2, shape=(4, 8, 8))
        csv_path = tmp_path / "spots.csv"

        model = MagicMock()
        details_empty = MagicMock()
        details_empty.prob = np.empty((0,))
        details_full = MagicMock()
        details_full.prob = np.array([0.9])
        model.predict.side_effect = [
            (np.empty((0, 3)), details_empty),
            (np.array([[1, 2, 3]]), details_full),
        ]

        worker = DetectionWorker3D(files=files, model=model, csv_path=csv_path)
        worker.run()

        df = pd.read_csv(csv_path)
        assert list(df["frame"]) == [1]

    def test_errored_signal_carries_the_exception_message(self, tmp_path):
        from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D

        in_dir = tmp_path / "in"
        files = write_timepoint_folder(in_dir, n_timepoints=1, shape=(4, 8, 8))
        csv_path = tmp_path / "spots.csv"

        model = MagicMock()
        model.predict.side_effect = RuntimeError("model exploded")

        errors = []
        worker = DetectionWorker3D(files=files, model=model, csv_path=csv_path)
        worker.errored.connect(errors.append)
        worker.run()

        assert errors == ["model exploded"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._workers3d'`

- [ ] **Step 3: Implement `_workers3d.py`**

Create `src/napari_spotiflow_tracking_nd/_workers3d.py`:

```python
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from qtpy.QtCore import QThread, Signal

from napari_spotiflow_tracking_nd._segmentation3d import detect_spots_3d


class VolumeOpWorker3D(QThread):
    """Stream a pre-processing op over a folder of per-timepoint TIFFs,
    one volume at a time: read the input file, apply `op`, write the
    result to `output_dir` under the same filename.

    Cancellation (`cancel()`) is checked between files — the file
    currently being written always finishes, so output is never a
    half-written TIFF.
    """

    progress = Signal(int, int)  # (current, total), 1-indexed current
    finished = Signal(list)      # output paths, as str, in input order
    errored = Signal(str)

    def __init__(self, files: list[Path], output_dir: Path, op, op_kwargs: dict, parent=None):
        super().__init__(parent)
        self._files = list(files)
        self._output_dir = Path(output_dir)
        self._op = op
        self._op_kwargs = dict(op_kwargs)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            written = []
            total = len(self._files)
            for i, path in enumerate(self._files):
                if self._cancelled:
                    break
                volume = tifffile.imread(str(path))
                result = self._op(volume, **self._op_kwargs)
                out_path = self._output_dir / path.name
                tifffile.imwrite(str(out_path), result)
                written.append(str(out_path))
                self.progress.emit(i + 1, total)
            self.finished.emit(written)
        except Exception as exc:
            self.errored.emit(str(exc))


class DetectionWorker3D(QThread):
    """Stream Spotiflow 3D detection over a folder of per-timepoint TIFFs,
    appending each timepoint's spots to a CSV as they're found so a long
    run's progress survives a cancel or a crash.
    """

    progress = Signal(int, int)  # (current, total), 1-indexed current
    finished = Signal(str)       # the CSV path written
    errored = Signal(str)

    def __init__(
        self,
        files: list[Path],
        model,
        csv_path: Path,
        prob_thresh: float | None = 0.5,
        min_distance: int = 2,
        parent=None,
    ):
        super().__init__(parent)
        self._files = list(files)
        self._model = model
        self._csv_path = Path(csv_path)
        self._prob_thresh = prob_thresh
        self._min_distance = min_distance
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._csv_path.exists():
                self._csv_path.unlink()
            header_written = False
            total = len(self._files)
            for t, path in enumerate(self._files):
                if self._cancelled:
                    break
                volume = tifffile.imread(str(path))
                points, details = detect_spots_3d(
                    volume, self._model,
                    prob_thresh=self._prob_thresh,
                    min_distance=self._min_distance,
                )
                if len(points) > 0:
                    frame_col = np.full((len(points), 1), t)
                    prob_col = np.asarray(details.prob).reshape(-1, 1)
                    rows = np.hstack([frame_col, points, prob_col])
                    df = pd.DataFrame(rows, columns=["frame", "z", "y", "x", "prob"])
                    df.to_csv(self._csv_path, mode="a", header=not header_written, index=False)
                    header_written = True
                self.progress.emit(t + 1, total)
            self.finished.emit(str(self._csv_path))
        except Exception as exc:
            self.errored.emit(str(exc))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py -v`
Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_workers3d.py src/napari_spotiflow_tracking_nd/_tests/test_workers3d.py
git commit -m "feat(nd): streaming QThread workers for pre-processing and detection"
```

---

### Task 5: Pre-Processing widget + napari manifest

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_cleanup_widget3d.py`
- Create: `src/napari_spotiflow_tracking_nd/napari.yaml`
- Modify: `pyproject.toml` (add `package-data` entry already present from Task 1 — confirm `napari.yaml` is listed; no change needed if Task 1's list already includes it)
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py`

**Interfaces:**
- Consumes: `load_folder_as_lazy_stack`, `compute_layer_scale` from `_io3d` (Task 1); `remove_background_high_pass_3d`, `gaussian_blur_3d`, `apply_calculation` from `_preprocessing3d` (Task 2); `VolumeOpWorker3D` from `_workers3d` (Task 4).
- Produces: `PreProcessingWidget3D(QWidget)` — public attributes used by tests: `_input_folder_edit`, `_load_btn`, `_pixel_size_xy_spin`, `_voxel_size_z_spin`, `_frame_interval_spin`, `_image_combo`, `_output_folder_edit`, `_bg_sigma_spin`, `_bg_use_gpu_check`, `_bg_run_btn`, `_blur_sigma_spin`, `_blur_run_btn`, `_calc_op_combo`, `_calc_value_spin`, `_calc_run_btn`, `_progress_bar`; methods `_load_folder()`, `_run_remove_background()`, `_run_gaussian_blur()`, `_run_arithmetic()`. Registered in `napari.yaml` as command id `napari-spotiflow-tracking-nd.preprocessing_widget`. Task 6 follows the same manifest-extension pattern.

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


@pytest.fixture
def preproc_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._cleanup_widget3d import PreProcessingWidget3D

    viewer = make_napari_viewer()
    widget = PreProcessingWidget3D(viewer)
    return widget, viewer


class TestPreProcessingWidget3DCreation:
    def test_creates_without_error(self, preproc_widget):
        widget, _ = preproc_widget
        assert widget is not None

    def test_has_the_expected_controls(self, preproc_widget):
        widget, _ = preproc_widget
        for attr in (
            "_input_folder_edit", "_load_btn",
            "_pixel_size_xy_spin", "_voxel_size_z_spin", "_frame_interval_spin",
            "_image_combo", "_output_folder_edit",
            "_bg_sigma_spin", "_bg_run_btn",
            "_blur_sigma_spin", "_blur_run_btn",
            "_calc_op_combo", "_calc_value_spin", "_calc_run_btn",
        ):
            assert hasattr(widget, attr), f"missing {attr}"


class TestLoadFolder:
    def test_load_folder_adds_a_scaled_image_layer(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=3, shape=(4, 8, 8))

        widget._input_folder_edit.setText(str(tmp_path))
        widget._pixel_size_xy_spin.setValue(0.104)
        widget._voxel_size_z_spin.setValue(0.210)
        widget._load_folder()

        assert len(viewer.layers) == 1
        layer = viewer.layers[0]
        assert layer.data.shape == (3, 4, 8, 8)
        assert layer.scale == (1.0, 0.210, 0.104, 0.104)
        assert len(layer.metadata["source_files"]) == 3

    def test_load_folder_without_calibration_uses_unit_scale(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path, n_timepoints=2, shape=(4, 8, 8))

        widget._input_folder_edit.setText(str(tmp_path))
        widget._load_folder()

        assert tuple(viewer.layers[0].scale) == (1.0, 1.0, 1.0, 1.0)


class TestRunRemoveBackground:
    def test_writes_output_folder_and_loads_it_as_a_new_layer(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        widget._input_folder_edit.setText(str(tmp_path / "in"))
        widget._load_folder()
        widget._image_combo.setCurrentIndex(0)
        widget._output_folder_edit.setText(str(tmp_path / "out_bg"))
        widget._bg_sigma_spin.setValue(2.0)

        widget._run_remove_background()  # calls worker.run() synchronously (see widget code)

        assert (tmp_path / "out_bg").exists()
        assert len(list((tmp_path / "out_bg").iterdir())) == 2
        assert len(viewer.layers) == 2


class TestRunGaussianBlur:
    def test_writes_output_folder(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        widget._input_folder_edit.setText(str(tmp_path / "in"))
        widget._load_folder()
        widget._image_combo.setCurrentIndex(0)
        widget._output_folder_edit.setText(str(tmp_path / "out_blur"))
        widget._blur_sigma_spin.setValue(1.5)

        widget._run_gaussian_blur()

        assert len(list((tmp_path / "out_blur").iterdir())) == 2


class TestRunArithmetic:
    def test_writes_output_folder(self, preproc_widget, tmp_path):
        widget, viewer = preproc_widget
        write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        widget._input_folder_edit.setText(str(tmp_path / "in"))
        widget._load_folder()
        widget._image_combo.setCurrentIndex(0)
        widget._output_folder_edit.setText(str(tmp_path / "out_calc"))
        widget._calc_op_combo.setCurrentText("add")
        widget._calc_value_spin.setValue(10)

        widget._run_arithmetic()

        assert len(list((tmp_path / "out_calc").iterdir())) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._cleanup_widget3d'`

- [ ] **Step 3: Implement `_cleanup_widget3d.py`**

Create `src/napari_spotiflow_tracking_nd/_cleanup_widget3d.py`:

```python
from __future__ import annotations

from pathlib import Path

import napari
from napari.utils.notifications import show_error, show_info
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

from napari_spotiflow_tracking_nd._io3d import compute_layer_scale, load_folder_as_lazy_stack
from napari_spotiflow_tracking_nd._preprocessing3d import (
    apply_calculation,
    gaussian_blur_3d,
    remove_background_high_pass_3d,
)
from napari_spotiflow_tracking_nd._workers3d import VolumeOpWorker3D


class PreProcessingWidget3D(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker: VolumeOpWorker3D | None = None
        self._setup_ui()
        self._connect_events()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── Load folder ──────────────────────────────────────────────
        load_group = QGroupBox("Load timeseries folder")
        load_layout = QVBoxLayout()
        load_group.setLayout(load_layout)

        folder_row = QHBoxLayout()
        self._input_folder_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_input_folder)
        folder_row.addWidget(self._input_folder_edit)
        folder_row.addWidget(browse_btn)
        load_layout.addLayout(folder_row)

        calib_row = QHBoxLayout()
        calib_row.addWidget(QLabel("xy (µm):"))
        self._pixel_size_xy_spin = QDoubleSpinBox()
        self._pixel_size_xy_spin.setRange(0.0, 100.0)
        self._pixel_size_xy_spin.setDecimals(4)
        calib_row.addWidget(self._pixel_size_xy_spin)
        calib_row.addWidget(QLabel("z (µm):"))
        self._voxel_size_z_spin = QDoubleSpinBox()
        self._voxel_size_z_spin.setRange(0.0, 100.0)
        self._voxel_size_z_spin.setDecimals(4)
        calib_row.addWidget(self._voxel_size_z_spin)
        calib_row.addWidget(QLabel("dt (s):"))
        self._frame_interval_spin = QDoubleSpinBox()
        self._frame_interval_spin.setRange(0.0, 10000.0)
        self._frame_interval_spin.setDecimals(3)
        calib_row.addWidget(self._frame_interval_spin)
        load_layout.addLayout(calib_row)

        self._load_btn = QPushButton("Load Folder")
        self._load_btn.clicked.connect(self._load_folder)
        load_layout.addWidget(self._load_btn)
        layout.addWidget(load_group)

        # ── Input / output selection ─────────────────────────────────
        io_group = QGroupBox("Input / Output")
        io_layout = QVBoxLayout()
        io_group.setLayout(io_layout)

        self._image_combo = QComboBox()
        io_layout.addWidget(QLabel("Input layer:"))
        io_layout.addWidget(self._image_combo)

        out_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        out_browse_btn = QPushButton("Browse…")
        out_browse_btn.clicked.connect(self._browse_output_folder)
        out_row.addWidget(self._output_folder_edit)
        out_row.addWidget(out_browse_btn)
        io_layout.addWidget(QLabel("Output folder:"))
        io_layout.addLayout(out_row)
        layout.addWidget(io_group)

        # ── Background removal ───────────────────────────────────────
        bg_group = QGroupBox("Background Removal (3D high-pass)")
        bg_layout = QHBoxLayout()
        bg_group.setLayout(bg_layout)
        bg_layout.addWidget(QLabel("sigma:"))
        self._bg_sigma_spin = QDoubleSpinBox()
        self._bg_sigma_spin.setRange(0.1, 200.0)
        self._bg_sigma_spin.setValue(10.0)
        bg_layout.addWidget(self._bg_sigma_spin)
        self._bg_run_btn = QPushButton("Run")
        bg_layout.addWidget(self._bg_run_btn)
        layout.addWidget(bg_group)

        # ── Gaussian blur ────────────────────────────────────────────
        blur_group = QGroupBox("Gaussian Blur")
        blur_layout = QHBoxLayout()
        blur_group.setLayout(blur_layout)
        blur_layout.addWidget(QLabel("sigma:"))
        self._blur_sigma_spin = QDoubleSpinBox()
        self._blur_sigma_spin.setRange(0.0, 200.0)
        self._blur_sigma_spin.setValue(1.0)
        blur_layout.addWidget(self._blur_sigma_spin)
        self._blur_run_btn = QPushButton("Run")
        blur_layout.addWidget(self._blur_run_btn)
        layout.addWidget(blur_group)

        # ── Arithmetic ───────────────────────────────────────────────
        calc_group = QGroupBox("Arithmetic")
        calc_layout = QHBoxLayout()
        calc_group.setLayout(calc_layout)
        self._calc_op_combo = QComboBox()
        self._calc_op_combo.addItems(["add", "subtract"])
        calc_layout.addWidget(self._calc_op_combo)
        self._calc_value_spin = QSpinBox()
        self._calc_value_spin.setRange(0, 65535)
        calc_layout.addWidget(self._calc_value_spin)
        self._calc_run_btn = QPushButton("Run")
        calc_layout.addWidget(self._calc_run_btn)
        layout.addWidget(calc_group)

        # ── Progress ─────────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)
        layout.addStretch()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_image_combo)
        self.viewer.layers.events.removed.connect(self._refresh_image_combo)
        self._bg_run_btn.clicked.connect(self._run_remove_background)
        self._blur_run_btn.clicked.connect(self._run_gaussian_blur)
        self._calc_run_btn.clicked.connect(self._run_arithmetic)

    def _refresh_image_combo(self, event=None):
        current = self._image_combo.currentText()
        self._image_combo.clear()
        names = [l.name for l in self.viewer.layers if isinstance(l, napari.layers.Image)]
        self._image_combo.addItems(names)
        if current in names:
            self._image_combo.setCurrentText(current)

    # ── Folder browsing ──────────────────────────────────────────────

    def _browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self._input_folder_edit.setText(folder)

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._output_folder_edit.setText(folder)

    # ── Load ──────────────────────────────────────────────────────────

    def _load_folder(self):
        folder = self._input_folder_edit.text()
        if not folder:
            show_error("No input folder selected.")
            return
        try:
            stack, files = load_folder_as_lazy_stack(folder)
        except (FileNotFoundError, ValueError) as exc:
            show_error(str(exc))
            return

        pixel_size_xy_um = self._pixel_size_xy_spin.value() or None
        voxel_size_z_um = self._voxel_size_z_spin.value() or None
        frame_interval_s = self._frame_interval_spin.value() or None
        scale = compute_layer_scale(pixel_size_xy_um, voxel_size_z_um)

        layer = self.viewer.add_image(
            stack,
            name=Path(folder).name,
            scale=scale,
            metadata={
                "source_files": files,
                "pixel_size_xy_um": pixel_size_xy_um,
                "voxel_size_z_um": voxel_size_z_um,
                "frame_interval_s": frame_interval_s,
            },
        )
        show_info(f"Loaded {len(files)} timepoints from {folder}")
        return layer

    # ── Shared run machinery ─────────────────────────────────────────

    def _get_input_layer(self):
        name = self._image_combo.currentText()
        if not name:
            show_error("No input layer selected.")
            return None
        return self.viewer.layers[name]

    def _get_output_dir(self) -> Path | None:
        folder = self._output_folder_edit.text()
        if not folder:
            show_error("No output folder selected.")
            return None
        return Path(folder)

    def _run_op(self, op, op_kwargs: dict, suffix: str):
        layer = self._get_input_layer()
        if layer is None:
            return
        output_dir = self._get_output_dir()
        if output_dir is None:
            return
        files = layer.metadata.get("source_files")
        if not files:
            show_error("Selected layer has no source files (load it via 'Load Folder').")
            return

        self._progress_bar.setMaximum(len(files))
        self._progress_bar.setValue(0)

        worker = VolumeOpWorker3D(files=files, output_dir=output_dir, op=op, op_kwargs=op_kwargs)
        worker.progress.connect(lambda cur, total: self._progress_bar.setValue(cur))
        worker.errored.connect(show_error)

        def _on_finished(written_paths):
            if not written_paths:
                return
            stack, out_files = load_folder_as_lazy_stack(output_dir)
            self.viewer.add_image(
                stack,
                name=f"{layer.name} ({suffix})",
                scale=layer.scale,
                metadata={
                    "source_files": out_files,
                    "pixel_size_xy_um": layer.metadata.get("pixel_size_xy_um"),
                    "voxel_size_z_um": layer.metadata.get("voxel_size_z_um"),
                    "frame_interval_s": layer.metadata.get("frame_interval_s"),
                },
            )
            show_info(f"Wrote {len(written_paths)} volumes to {output_dir}")

        worker.finished.connect(_on_finished)
        self._worker = worker
        worker.run()  # Phase 1: run synchronously on the calling thread; the
        # `QThread` API is kept so a future phase can switch to `worker.start()`
        # without changing this call site's signal wiring.

    def _run_remove_background(self):
        self._run_op(
            remove_background_high_pass_3d,
            {"sigma": self._bg_sigma_spin.value()},
            f"bg s={self._bg_sigma_spin.value()}",
        )

    def _run_gaussian_blur(self):
        self._run_op(
            gaussian_blur_3d,
            {"sigma": self._blur_sigma_spin.value()},
            f"blur s={self._blur_sigma_spin.value()}",
        )

    def _run_arithmetic(self):
        op = self._calc_op_combo.currentText()
        value = self._calc_value_spin.value()
        self._run_op(apply_calculation, {"op": op, "value": value}, f"{op} {value}")
```

**Note on `worker.run()` vs `worker.start()`:** Phase 1 calls `worker.run()` directly (same thread) so the widget stays simple and testable without `qtbot`/event-loop plumbing. This keeps the UI blocking during a run — acceptable for Phase 1's interactive validation, and flagged here explicitly so a later phase can switch the two call sites (`_run_op` here, and Task 6's detection run) to `worker.start()` plus `qtbot.waitSignal`-based tests without touching the worker classes themselves.

Create `src/napari_spotiflow_tracking_nd/napari.yaml`:

```yaml
name: napari-spotiflow-tracking-nd
display_name: Spotiflow ND Tracking
contributions:
  commands:
    - id: napari-spotiflow-tracking-nd.preprocessing_widget
      python_name: napari_spotiflow_tracking_nd._cleanup_widget3d:PreProcessingWidget3D
      title: Pre-Processing
  widgets:
    - command: napari-spotiflow-tracking-nd.preprocessing_widget
      display_name: Pre-Processing
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py -v`
Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_cleanup_widget3d.py src/napari_spotiflow_tracking_nd/napari.yaml src/napari_spotiflow_tracking_nd/_tests/test_cleanup_widget3d.py
git commit -m "feat(nd): Pre-Processing widget + napari manifest"
```

---

### Task 6: Spot Detection widget + manifest entry

**Files:**
- Create: `src/napari_spotiflow_tracking_nd/_detection_widget3d.py`
- Modify: `src/napari_spotiflow_tracking_nd/napari.yaml` (add the detection command/widget entries)
- Test: `src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py`

**Interfaces:**
- Consumes: `load_model`, `detect_spots_3d` (via `DetectionWorker3D`) from Task 3/4; `DetectionWorker3D` from `_workers3d` (Task 4).
- Produces: `DetectionWidget3D(QWidget)` with attributes: `_image_combo`, `_model_name_edit`, `_browse_model_btn`, `_use_gpu_check`, `_prob_thresh_spin`, `_min_distance_spin`, `_csv_path_edit`, `_browse_csv_btn`, `_run_btn`, `_progress_bar`; methods `_run_detection()`. Registered as `napari-spotiflow-tracking-nd.detection_widget` in `napari.yaml`.

- [ ] **Step 1: Write the failing tests**

Create `src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from napari_spotiflow_tracking_nd._tests.conftest import write_timepoint_folder


@pytest.fixture
def detection_widget(make_napari_viewer):
    from napari_spotiflow_tracking_nd._detection_widget3d import DetectionWidget3D

    viewer = make_napari_viewer()
    widget = DetectionWidget3D(viewer)
    return widget, viewer


class TestDetectionWidget3DCreation:
    def test_creates_without_error(self, detection_widget):
        widget, _ = detection_widget
        assert widget is not None

    def test_has_the_expected_controls(self, detection_widget):
        widget, _ = detection_widget
        for attr in (
            "_image_combo", "_model_name_edit", "_browse_model_btn",
            "_prob_thresh_spin", "_min_distance_spin",
            "_csv_path_edit", "_browse_csv_btn", "_run_btn", "_progress_bar",
        ):
            assert hasattr(widget, attr), f"missing {attr}"


class TestRunDetection:
    @patch("napari_spotiflow_tracking_nd._detection_widget3d.load_model")
    def test_writes_csv_and_adds_a_scaled_points_layer(
        self, mock_load_model, detection_widget, tmp_path
    ):
        widget, viewer = detection_widget
        files = write_timepoint_folder(tmp_path / "in", n_timepoints=2, shape=(4, 16, 16))

        mock_model = MagicMock()
        details = MagicMock()
        details.prob = np.array([0.8])
        mock_model.predict.return_value = (np.array([[1, 2, 3]]), details)
        mock_load_model.return_value = mock_model

        viewer.add_image(
            np.zeros((2, 4, 16, 16), dtype=np.uint16),
            name="vol",
            scale=(1.0, 0.210, 0.104, 0.104),
            metadata={"source_files": files},
        )
        widget._image_combo.clear()
        widget._image_combo.addItem("vol")
        widget._image_combo.setCurrentText("vol")
        widget._model_name_edit.setText("synth_3d")
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))

        widget._run_detection()

        df = pd.read_csv(tmp_path / "spots.csv")
        assert list(df.columns) == ["frame", "z", "y", "x", "prob"]
        assert len(df) == 2  # one row per timepoint (2 timepoints, 1 spot each)

        points_layers = [l for l in viewer.layers if l.__class__.__name__ == "Points"]
        assert len(points_layers) == 1
        # Points are (frame, z, y, x) — 4 columns — so scale carries all 4
        # entries from the source image layer, including the unscaled T axis.
        assert tuple(points_layers[0].scale) == (1.0, 0.210, 0.104, 0.104)

    def test_missing_input_layer_shows_error_not_exception(self, detection_widget, tmp_path):
        widget, viewer = detection_widget
        widget._image_combo.clear()
        widget._csv_path_edit.setText(str(tmp_path / "spots.csv"))

        widget._run_detection()  # must not raise

        assert not (tmp_path / "spots.csv").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_spotiflow_tracking_nd._detection_widget3d'`

- [ ] **Step 3: Implement `_detection_widget3d.py`**

Create `src/napari_spotiflow_tracking_nd/_detection_widget3d.py`:

```python
from __future__ import annotations

from pathlib import Path

import napari
import pandas as pd
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
    QCheckBox,
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

from napari_spotiflow_tracking_nd._segmentation3d import load_model
from napari_spotiflow_tracking_nd._workers3d import DetectionWorker3D


class DetectionWidget3D(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker: DetectionWorker3D | None = None
        self._setup_ui()
        self._connect_events()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        io_group = QGroupBox("Input")
        io_layout = QVBoxLayout()
        io_group.setLayout(io_layout)
        self._image_combo = QComboBox()
        io_layout.addWidget(QLabel("Input layer:"))
        io_layout.addWidget(self._image_combo)
        layout.addWidget(io_group)

        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()
        model_group.setLayout(model_layout)
        self._model_name_edit = QLineEdit("synth_3d")
        model_layout.addWidget(self._model_name_edit)
        self._browse_model_btn = QPushButton("Browse…")
        self._browse_model_btn.clicked.connect(self._browse_custom_model)
        model_layout.addWidget(self._browse_model_btn)
        self._use_gpu_check = QCheckBox("Use GPU")
        model_layout.addWidget(self._use_gpu_check)
        layout.addWidget(model_group)

        params_group = QGroupBox("Detection parameters")
        params_layout = QHBoxLayout()
        params_group.setLayout(params_layout)
        params_layout.addWidget(QLabel("prob_thresh:"))
        self._prob_thresh_spin = QDoubleSpinBox()
        self._prob_thresh_spin.setRange(0.0, 1.0)
        self._prob_thresh_spin.setSingleStep(0.05)
        self._prob_thresh_spin.setValue(0.5)
        params_layout.addWidget(self._prob_thresh_spin)
        params_layout.addWidget(QLabel("min_distance:"))
        self._min_distance_spin = QSpinBox()
        self._min_distance_spin.setRange(1, 100)
        self._min_distance_spin.setValue(2)
        params_layout.addWidget(self._min_distance_spin)
        layout.addWidget(params_group)

        out_group = QGroupBox("Output")
        out_layout = QHBoxLayout()
        out_group.setLayout(out_layout)
        self._csv_path_edit = QLineEdit()
        out_layout.addWidget(self._csv_path_edit)
        self._browse_csv_btn = QPushButton("Browse…")
        self._browse_csv_btn.clicked.connect(self._browse_csv_path)
        out_layout.addWidget(self._browse_csv_btn)
        layout.addWidget(out_group)

        self._run_btn = QPushButton("Run Detection")
        self._run_btn.clicked.connect(self._run_detection)
        layout.addWidget(self._run_btn)

        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)
        layout.addStretch()

    def _connect_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_image_combo)
        self.viewer.layers.events.removed.connect(self._refresh_image_combo)

    def _refresh_image_combo(self, event=None):
        current = self._image_combo.currentText()
        self._image_combo.clear()
        names = [l.name for l in self.viewer.layers if isinstance(l, napari.layers.Image)]
        self._image_combo.addItems(names)
        if current in names:
            self._image_combo.setCurrentText(current)

    def _browse_custom_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select model checkpoint folder")
        if folder:
            self._model_name_edit.setText(folder)

    def _browse_csv_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save spots CSV", "", "CSV files (*.csv)")
        if path:
            self._csv_path_edit.setText(path)

    def _run_detection(self):
        name = self._image_combo.currentText()
        if not name:
            show_error("No input layer selected.")
            return
        layer = self.viewer.layers[name]
        files = layer.metadata.get("source_files")
        if not files:
            show_error("Selected layer has no source files (load it via Pre-Processing's 'Load Folder').")
            return
        csv_path = self._csv_path_edit.text()
        if not csv_path:
            show_error("No output CSV path selected.")
            return

        device = "cuda" if self._use_gpu_check.isChecked() else "cpu"
        model = load_model(self._model_name_edit.text(), device=device)

        self._progress_bar.setMaximum(len(files))
        self._progress_bar.setValue(0)

        worker = DetectionWorker3D(
            files=files,
            model=model,
            csv_path=Path(csv_path),
            prob_thresh=self._prob_thresh_spin.value(),
            min_distance=self._min_distance_spin.value(),
        )
        worker.progress.connect(lambda cur, total: self._progress_bar.setValue(cur))
        worker.errored.connect(show_error)

        def _on_finished(written_csv_path):
            df = pd.read_csv(written_csv_path)
            if len(df) == 0:
                show_info("Detection finished — no spots found.")
                return
            points = df[["frame", "z", "y", "x"]].to_numpy()
            # layer.scale is (T, Z, Y, X) — one entry per point column above,
            # since points here are (frame, z, y, x), not just (z, y, x).
            self.viewer.add_points(
                points,
                name=f"{layer.name} (spots)",
                scale=layer.scale,
                metadata={"csv_path": written_csv_path},
                properties={"prob": df["prob"].to_numpy()},
            )
            show_info(f"Detected {len(df)} spots across {df['frame'].nunique()} timepoints.")

        worker.finished.connect(_on_finished)
        self._worker = worker
        worker.run()  # Phase 1: synchronous, same rationale as PreProcessingWidget3D._run_op.
```

Modify `src/napari_spotiflow_tracking_nd/napari.yaml` to add the detection command/widget:

```yaml
name: napari-spotiflow-tracking-nd
display_name: Spotiflow ND Tracking
contributions:
  commands:
    - id: napari-spotiflow-tracking-nd.preprocessing_widget
      python_name: napari_spotiflow_tracking_nd._cleanup_widget3d:PreProcessingWidget3D
      title: Pre-Processing
    - id: napari-spotiflow-tracking-nd.detection_widget
      python_name: napari_spotiflow_tracking_nd._detection_widget3d:DetectionWidget3D
      title: Spot Detection
  widgets:
    - command: napari-spotiflow-tracking-nd.preprocessing_widget
      display_name: Pre-Processing
    - command: napari-spotiflow-tracking-nd.detection_widget
      display_name: Spot Detection
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py -v`
Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add src/napari_spotiflow_tracking_nd/_detection_widget3d.py src/napari_spotiflow_tracking_nd/napari.yaml src/napari_spotiflow_tracking_nd/_tests/test_detection_widget3d.py
git commit -m "feat(nd): Spot Detection widget + manifest entry"
```

---

### Task 7: Package finalization — editable install, full suite, README

**Files:**
- Create: `README.md`
- Modify: `pyproject.toml` (confirm `package-data` includes `napari.yaml`; add `[project.urls]`/classifiers only if missing — this task verifies, not restructures)

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: nothing new for later tasks — this is the phase's final checkpoint.

- [ ] **Step 1: Confirm the editable install picks up the plugin**

Run: `pip install -e ".[testing]"`
Expected: installs cleanly; then run
`python -c "from napari.plugins import plugin_manager; plugin_manager.discover(); print('napari-spotiflow-tracking-nd' in plugin_manager.plugins)"`
Expected: prints `True` (the manifest is discovered).

- [ ] **Step 2: Run the full test suite**

Run: `pytest -v`
Expected: PASS — all tests from Tasks 1–6 green, none skipped for missing imports.

- [ ] **Step 3: Write `README.md`**

Create `README.md`:

```markdown
# napari-spotiflow-tracking-nd

A [napari](https://napari.org) plugin for **3D-volume timeseries** spot
detection and tracking, built for datasets too large to hold in memory:
one TIFF per timepoint, `(Z, Y, X)` per file, streamed one volume at a
time.

Sibling project: [napari-spotiflow-tracking](https://github.com/) — the
2D (single-plane, whole-movie-in-RAM) version this plugin is ported from.

## Status: Phase 1

- **Data loading** — a folder of per-timepoint TIFFs opens as a single
  lazy `(T, Z, Y, X)` image layer (`dask`-backed; nothing is loaded until
  you scrub to it).
- **Pre-processing** — background removal (3D high-pass), Gaussian blur,
  and add/subtract arithmetic, each streamed volume-by-volume and written
  to an output folder of per-timepoint TIFFs (same filenames as the
  input — no zarr).
- **Spot detection** — [Spotiflow](https://github.com/weigertlab/spotiflow)
  3D models (`synth_3d`, `smfish_3d`, or a custom checkpoint), run per
  volume, exported to a CSV (`frame, z, y, x, prob`) and loaded back as a
  Points layer.

Deferred to later phases: tracking (trackpy, 3D), track-property/step
analysis, PAPA (not planned), N2V denoise, walking average, erosion
background removal, Shapes-based crop.

## Install

```bash
pip install -e ".[testing]"
pip install "napari[all]"   # if napari + a Qt backend aren't already present

napari   # then open Plugins → Spotiflow ND Tracking → <widget>
```

## Calibration

Pixel size (xy, µm) and voxel size (z, µm) are entered when loading a
folder in the Pre-Processing widget; they set the loaded layer's `scale`
so 3D rendering has the correct aspect ratio. Frame interval (s) is
carried in layer metadata for later phases. None of these are hard-coded
— enter the values for your own acquisition.

## Tests

```bash
pytest   # testpaths = src/napari_spotiflow_tracking_nd/_tests
```

No test requires a GPU, real Spotiflow weights, or the real dataset —
fixtures build tiny synthetic per-timepoint TIFF folders on the fly.
```

- [ ] **Step 4: Commit**

```bash
git add README.md pyproject.toml
git commit -m "docs(nd): Phase 1 README"
```

---

## Self-Review Notes

- **Spec coverage:** data model + calibration → Task 1; background removal + blur + arithmetic → Task 2; Spotiflow 3D detection → Task 3; streaming workers → Task 4; Pre-Processing widget/manifest → Task 5; Detection widget/manifest → Task 6; package/README checkpoint → Task 7. Deferred items (Phase 2/3, PAPA, N2V, walking average, erosion, crop, zarr) are named explicitly in Global Constraints and the README rather than left implicit.
- **Placeholder scan:** no TBD/TODO; every step carries real code or an exact command + expected output.
- **Type/name consistency:** `remove_background_high_pass_3d`, `gaussian_blur_3d`, `apply_calculation` (Task 2) are the exact names imported in Task 4's workers and Task 5's widget. `detect_spots_3d`/`load_model` (Task 3) match Task 4/Task 6's imports. `VolumeOpWorker3D`/`DetectionWorker3D` signal names (`progress`, `finished`, `errored`) and constructor kwargs match between Task 4's implementation and Tasks 5–6's usage. `compute_layer_scale`/`load_folder_as_lazy_stack` (Task 1) match Task 5's imports. Task 6's detected points are `(frame, z, y, x)` — 4 columns — so the Points layer's `scale` is the full 4-entry `layer.scale`, not a 3-entry slice; verified consistent between the widget code and its test.
- **Scope check:** single phase, one plan; Phase 2 (tracking) and Phase 3 (track properties) are out of this plan's scope by design and will get their own brainstorm → spec → plan cycle.
