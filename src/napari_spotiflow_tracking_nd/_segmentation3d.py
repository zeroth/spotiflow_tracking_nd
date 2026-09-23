from __future__ import annotations

import numpy as np

# Spotiflow's own pretrained-model registry currently marks these two
# `is_3d=True` (see spotiflow.model.pretrained._REGISTERED). Any other
# registered 3D model added upstream can still be loaded — pass its name
# here too, or use a local checkpoint path, which always goes through
# `Spotiflow.from_folder`.
#
# Hard-coded rather than queried from spotiflow at import time on purpose:
# reading the live registry means importing spotiflow (and so torch), which
# the Spot Detection widget must not do just to populate a combo box.
PRETRAINED_3D_MODELS: tuple[str, ...] = ("synth_3d", "smfish_3d")

_PRETRAINED_3D_MODELS = set(PRETRAINED_3D_MODELS)


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
