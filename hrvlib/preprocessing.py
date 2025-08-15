import numpy as np

def detect_artifacts(rr: list[float], threshold_ms: int = 2000) -> list[bool]:
    """Detect artifacts based on a threshold.
    TODO: Add alternative detection methods (e.g., median absolute deviation).
    """
    return [abs(x) > threshold_ms for x in rr]

def correct_artifacts(rr: list[float], mask: list[bool]) -> list[float]:
    """Replace artifacts using linear interpolation.
    TODO: Allow option for cubic spline interpolation.
    """
    rr_array = np.array(rr, dtype=float)
    for i, is_artifact in enumerate(mask):
        if is_artifact:
            rr_array[i] = np.nan
    nans = np.isnan(rr_array)
    rr_array[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), rr_array[~nans])
    return rr_array.tolist()

def detrend_rr(rr: list[float]) -> list[float]:
    """Remove low-frequency trends from RR intervals.
    TODO: Implement using Smoothness Priors or polynomial fitting.
    """
    raise NotImplementedError("Detrending not yet implemented.")
