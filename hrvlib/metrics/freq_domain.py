import numpy as np
from scipy.signal import welch

def band_powers(rr: list[float], fs: float = 4.0):
    """Compute VLF, LF, HF, and LF/HF ratio using Welch's method.
    TODO: Allow choice of window function and segment length.
    TODO: Add Lomb-Scargle periodogram for uneven sampling.
    """
    t = np.cumsum(rr) / 1000.0
    interpolated = np.interp(np.arange(t[0], t[-1], 1/fs), t, rr)
    f, pxx = welch(interpolated, fs=fs)

    vlf_band = (0.0033, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    vlf = np.trapz(pxx[(f >= vlf_band[0]) & (f < vlf_band[1])], f[(f >= vlf_band[0]) & (f < vlf_band[1])])
    lf = np.trapz(pxx[(f >= lf_band[0]) & (f < lf_band[1])], f[(f >= lf_band[0]) & (f < lf_band[1])])
    hf = np.trapz(pxx[(f >= hf_band[0]) & (f < hf_band[1])], f[(f >= hf_band[0]) & (f < hf_band[1])])

    lf_hf = lf / hf if hf != 0 else np.nan
    return vlf, lf, hf, lf_hf
