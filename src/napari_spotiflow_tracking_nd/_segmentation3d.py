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


# --- Classical (model-free) detectors: LoG and DoG --------------------------
#
# Single-scale, TrackMate-style: the user gives the expected spot size as a
# Gaussian sigma per axis (in voxels, anisotropic, like pre-processing), the
# volume is filtered once at that scale, and spots are local maxima of the
# response above a noise-relative threshold.

CLASSICAL_METHODS: tuple[str, ...] = ("log", "dog")

# The usual DoG approximation of LoG: G(sigma) - G(1.6 * sigma).
DOG_SIGMA_RATIO = 1.6

# Peak response of each filter to a spot that exactly matches `sigma`, per
# unit of spot amplitude (derived for a Gaussian spot of width s filtered
# at sigma = s on every axis). Dividing by it makes `response` read as
# roughly "peak brightness above local background".
_LOG_MATCHED_GAIN = 1.5 * 2.0 ** -1.5
_DOG_MATCHED_GAIN = 2.0 ** -1.5 - (1.0 + DOG_SIGMA_RATIO ** 2) ** -1.5

# Noise is estimated on at most about this many voxels of the response.
_NOISE_SAMPLE_VOXELS = 2_000_000


def _gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    radius = max(1, int(np.ceil(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _second_derivative_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Second derivative of a Gaussian, corrected to sum to exactly zero.

    `scipy.ndimage.gaussian_filter(order=2)`'s kernel does not quite sum to
    zero, so its LoG of a bright flat region is not zero; the residue grows
    with brightness and would show up as spurious response on bright cells.
    """
    radius = max(1, int(np.ceil(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = _gaussian_kernel(sigma, truncate)
    k = g * (x ** 2 - sigma ** 2) / sigma ** 4
    k -= g * (k.sum() / g.sum())        # zero sum: flat input -> zero output
    k *= 2.0 / float((k * x ** 2).sum())  # unit second moment: exact on x^2
    return k


def filter_response_3d(
    volume: np.ndarray, method: str, sigma_zyx: tuple[float, float, float],
) -> np.ndarray:
    """LoG or DoG response of a (Z, Y, X) volume, as float32, positive on
    bright spots and scaled so a spot matching `sigma_zyx` responds with
    about its own amplitude above background.

    LoG is scale-normalised per axis (sum of sigma_i^2 * d^2/dx_i^2), which
    is what makes anisotropic sigmas comparable across z and xy.
    """
    from scipy.ndimage import correlate1d, gaussian_filter

    if method not in CLASSICAL_METHODS:
        raise ValueError(f"Unknown detection method {method!r}; expected one of {CLASSICAL_METHODS}.")
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D (Z, Y, X) volume, got {volume.ndim}D.")
    sigma = tuple(float(s) for s in sigma_zyx)
    # float32 throughout: a full-size volume is large, and every extra
    # float64 copy doubles the peak memory of a detection run.
    vol = volume.astype(np.float32, copy=False)

    if method == "log":
        response = np.zeros(vol.shape, dtype=np.float32)
        for axis, s in enumerate(sigma):
            # sigma^2 * d^2/dx^2 along `axis`, Gaussian along the others.
            term = vol
            for other in range(3):
                kernel = _second_derivative_kernel(sigma[other]) if other == axis                     else _gaussian_kernel(sigma[other])
                term = correlate1d(term, kernel, axis=other, output=np.float32, mode="reflect")
            response -= (s * s) * term
        response /= _LOG_MATCHED_GAIN
        return response

    response = gaussian_filter(vol, sigma, output=np.float32)
    response -= gaussian_filter(
        vol, tuple(DOG_SIGMA_RATIO * s for s in sigma), output=np.float32,
    )
    response /= _DOG_MATCHED_GAIN
    return response


def _noise_level(response: np.ndarray, volume_dtype) -> float:
    """Robust standard deviation of the filter response (MAD x 1.4826),
    estimated on a regular subsample of the volume.

    The MAD ignores the few spot voxels, so it measures background noise.
    For integer images it is floored at the quantisation noise (1/sqrt(12)
    counts), so a noise-free volume doesn't divide by zero and turn
    rounding error into "spots".
    """
    step = max(1, int(np.ceil((response.size / _NOISE_SAMPLE_VOXELS) ** (1 / 3))))
    sample = response[::step, ::step, ::step].ravel()
    mad = float(np.median(np.abs(sample - np.median(sample)))) * 1.4826
    floor = 1.0 / np.sqrt(12.0) if np.issubdtype(volume_dtype, np.integer) else 0.0
    floor = max(floor, 1e-6 * float(np.max(np.abs(response), initial=0.0)))
    return max(mad, floor, np.finfo(np.float32).tiny)


def _local_maxima(response: np.ndarray, min_distance: int, threshold: float) -> np.ndarray:
    """(N, 3) int coordinates of local maxima of `response` above
    `threshold`, no two closer than `min_distance` voxels on any axis,
    brightest first.

    Equivalent to `skimage.feature.peak_local_max(..., exclude_border=False)`
    with its default Chebyshev distance, but up to ~1.8x faster on a
    full-size volume: a `(2 * min_distance + 1)` maximum filter already
    enforces the spacing, so only tied neighbours (plateaus) are left to
    thin out. Border voxels are kept on purpose: volumes are often only
    tens of planes deep, and dropping peaks near the top and bottom would
    lose real spots.
    """
    from scipy.ndimage import maximum_filter
    from scipy.spatial import cKDTree

    size = 2 * max(min_distance, 0) + 1
    is_peak = maximum_filter(response, size=size, mode="nearest")
    np.equal(response, is_peak, out=is_peak)  # reuse the buffer: 1.0 where max
    peak = is_peak.astype(bool, copy=False)
    del is_peak
    peak &= response > threshold
    coords = np.argwhere(peak)
    if len(coords) == 0:
        return coords.reshape(0, 3)
    coords = coords[np.argsort(-response[tuple(coords.T)], kind="stable")]

    if min_distance >= 1:
        # Only equal-valued neighbours can both survive the filter.
        pairs = cKDTree(coords).query_pairs(r=min_distance, p=np.inf, output_type="ndarray")
        if len(pairs):
            keep = np.ones(len(coords), dtype=bool)
            neighbours: dict[int, list[int]] = {}
            for a, b in pairs:
                neighbours.setdefault(min(a, b), []).append(max(a, b))
            for i in range(len(coords)):
                if keep[i]:
                    for j in neighbours.get(i, ()):
                        keep[j] = False
            coords = coords[keep]
    return coords


class ClassicalDetector3D:
    """LoG / DoG spot detector with the same `predict` interface as a
    Spotiflow model, so the detection worker and preview take either.

    Args:
        method: "log" or "dog".
        sigma_zyx: expected spot sigma per axis, in voxels.
        threshold: minimum signal-to-noise ratio — the response divided by
            the volume's own noise level (see `_noise_level`). Relative to
            each volume, so it carries across datasets and through
            photobleaching without retuning.
    """

    def __init__(self, method: str, sigma_zyx: tuple[float, float, float], threshold: float = 5.0):
        if method not in CLASSICAL_METHODS:
            raise ValueError(f"Unknown detection method {method!r}; expected one of {CLASSICAL_METHODS}.")
        self.method = method
        self.sigma_zyx = tuple(float(s) for s in sigma_zyx)
        self.threshold = float(threshold)

    def params(self) -> dict:
        """The settings that define this detector, for the JSON sidecar."""
        out = {
            "method": self.method,
            "sigma_zyx": list(self.sigma_zyx),
            "threshold_snr": self.threshold,
        }
        if self.method == "dog":
            out["dog_sigma_ratio"] = DOG_SIGMA_RATIO
        return out

    def predict(self, volume, prob_thresh=None, min_distance: int = 2, verbose: bool = False):
        """Return (points, details): points is (N, 3) float [z, y, x];
        details has parallel `prob` (all NaN — there is no probability),
        `response` (≈ amplitude above background) and `snr`.

        `prob_thresh` is accepted for interface compatibility and ignored.
        """
        from types import SimpleNamespace

        response = filter_response_3d(volume, self.method, self.sigma_zyx)
        noise = _noise_level(response, volume.dtype)
        coords = _local_maxima(response, int(min_distance), self.threshold * noise)
        values = response[tuple(coords.T)].astype(np.float64)
        details = SimpleNamespace(
            prob=np.full(len(coords), np.nan),
            response=values,
            snr=values / noise,
        )
        return coords.astype(np.float64).reshape(-1, 3), details
