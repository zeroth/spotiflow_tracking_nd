"""npe2 reader contribution: open a folder of per-timepoint 3D TIFFs (or any
single file inside one) as a lazy (T, Z, Y, X) Image layer.

Registered in ``napari.yaml`` with ``accepts_directories: true``, so
File -> Open Folder..., drag-and-drop, and ``napari path/to/folder`` all
work with no widget interaction — the user never has to go through the
Pre-Processing widget just to look at a dataset. Selecting a single file
that belongs to such a series also works: the whole parent folder is
loaded, same as if the folder itself had been opened.

Calibration (pixel size, voxel size, frame interval) can't be prompted for
from a reader — there is no UI to ask through — so the layer opens at unit
scale with the three calibration metadata keys present but ``None``. Use
the Pre-Processing widget's "Apply Calibration" action to fill them in on
the loaded layer afterward; every other part of the plugin only reads
these keys from layer metadata; it doesn't matter whether a layer arrived
via this reader or was built some other way, as long as the keys are
there.
"""
from __future__ import annotations

from pathlib import Path

from napari_spotiflow_tracking_nd._io3d import list_timepoint_files, load_folder_as_lazy_stack


def get_reader(path):
    """Return a reader function if `path` is a per-timepoint 3D TIFF series
    (a folder of them, or a single file that belongs to one), else None.
    """
    # A list means napari is offering several paths from one selection;
    # this reader handles a single folder or file and lets napari fan out.
    if not isinstance(path, str):
        return None

    candidate = Path(path)
    if candidate.is_dir():
        folder = candidate
    elif candidate.is_file():
        folder = candidate.parent
    else:
        return None

    try:
        files = list_timepoint_files(folder)
    except (FileNotFoundError, ValueError):
        return None

    if candidate.is_file() and candidate not in files:
        # A file in the folder that isn't part of the recognised series
        # (no `_T_<n>_` token, or a different extension) — decline, so
        # another reader (e.g. napari's builtin TIFF reader) can try it.
        return None

    return _make_reader(folder)


def _make_reader(folder: Path):
    """Bind a reader function to `folder`, ignoring whatever single path
    napari originally passed `get_reader` (a directory or one file inside
    it) — either way, the whole series is what gets loaded."""

    def read_volume_timeseries(_path):
        stack, files = load_folder_as_lazy_stack(folder)
        attrs = {
            "name": folder.name,
            "scale": (1.0, 1.0, 1.0, 1.0),
            "metadata": {
                "source_files": files,
                "pixel_size_xy_um": None,
                "voxel_size_z_um": None,
                "frame_interval_s": None,
            },
        }
        return [(stack, attrs, "image")]

    return read_volume_timeseries
