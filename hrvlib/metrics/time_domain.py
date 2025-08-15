import numpy as np

def sdnn(rr: list[float]) -> float:
    """Standard deviation of NN intervals.
    TODO: Ensure proper unit consistency (ms vs seconds).
    """
    return np.std(rr, ddof=1)

def rmssd(rr: list[float]) -> float:
    """Root mean square of successive differences.
    TODO: Allow filtering of ectopic beats before calculation.
    """
    diff_rr = np.diff(rr)
    return np.sqrt(np.mean(diff_rr ** 2))

def pnn50(rr: list[float]) -> float:
    """Percentage of successive RR intervals differing by > 50 ms.
    TODO: Make threshold configurable.
    """
    diff_rr = np.abs(np.diff(rr))
    count = np.sum(diff_rr > 50)
    return 100.0 * count / len(diff_rr)
