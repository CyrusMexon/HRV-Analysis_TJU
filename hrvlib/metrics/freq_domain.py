import numpy as np
from scipy import signal, interpolate
from typing import Tuple, Dict, Optional, Union
import warnings

# Keep this import (you had it in the original file).
from hrvlib.preprocessing import PreprocessingResult

from hrvlib.signal_processing.smoothness_priors import detrend_with_smoothness_priors


class HRVFreqDomainAnalysis:
    """
    Patched HRV frequency-domain analysis with robust, adaptive AR fitting.

    Key behaviours:
    - Input RR intervals are expected in milliseconds.
    - Internally we convert RR to seconds for interpolation/detrending/AR fitting.
    - PSD outputs (Welch & AR) are in ms^2/Hz.
    - AR order adapts to data length (safe default: min(requested, max(4, n//6))).
    - Burg estimation is tried first; fallback to Yule-Walker if necessary.
    - If AR cannot be estimated, AR PSD falls back to Welch PSD (no-all-zero AR).
    """

    VALID_WINDOWS = [
        "hann",
        "hamming",
        "blackman",
        "bartlett",
        "flattop",
        "parzen",
        "bohman",
        "nuttall",
    ]
    VALID_DETRENDS = ["linear", "constant", "smoothness_priors", None]
    DEFAULT_FREQ_BANDS = {
        "ulf": (0.0, 0.003),
        "vlf": (0.003, 0.04),
        "lf": (0.04, 0.15),
        "hf": (0.15, 0.4),
        "lf_hf_ratio": (0.04, 0.4),
    }

    def __init__(
        self,
        preprocessed_rri: np.ndarray,
        preprocessing_result: Optional[PreprocessingResult] = None,
        sampling_rate: float = 4.0,
        detrend_method: Optional[str] = "linear",
        detrend_lambda: float = 500,
        window_type: str = "hann",
        segment_length: float = 120.0,
        overlap_ratio: float = 0.75,
        ar_order: int = 16,
        analysis_window: Optional[Tuple[float, float]] = None,
    ):
        self.rr_intervals_ms = np.array(preprocessed_rri, dtype=float)
        self.preprocessing_result = preprocessing_result
        self.sampling_rate = float(sampling_rate)
        self.detrend_method = detrend_method
        self.detrend_lambda = detrend_lambda
        self.window_type = window_type
        self.segment_length = float(segment_length)
        self.overlap_ratio = float(overlap_ratio)
        self.ar_order = int(ar_order)
        self.analysis_window = analysis_window

        self._validate_input()

        if self.analysis_window is not None:
            self.rr_intervals_ms = self._apply_analysis_window(self.rr_intervals_ms)

        # Basic sanity checks on rr_intervals
        if np.any(np.isnan(self.rr_intervals_ms)):
            warnings.warn(
                "RR intervals contain NaN values; these will be removed before analysis."
            )
            self.rr_intervals_ms = self.rr_intervals_ms[~np.isnan(self.rr_intervals_ms)]

        # Keep a quick print/log of length for debugging
        # (You can remove/replace with logger in production)
        if len(self.rr_intervals_ms) == 0:
            warnings.warn("Empty RR input received.")

        # Create time domain signal in seconds (RR_s)
        self.time_domain_s = self._create_time_domain_signal()

        # Compute Welch PSD
        self.freqs, self.psd = self._compute_welch_psd()

        # Compute FFT PSD
        self.fft_freqs, self.fft_psd = self._compute_fft_psd()

        # Compute AR PSD (adaptive)
        self.ar_freqs, self.ar_psd = self._compute_ar_psd()

        # Spectral metrics
        self.spectral_metrics = self._compute_spectral_metrics()
        self.fft_spectral_metrics = self._compute_spectral_metrics(use_fft=True)
        self.ar_spectral_metrics = self._compute_spectral_metrics(use_ar=True)

    def _validate_input(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.detrend_method not in self.VALID_DETRENDS:
            raise ValueError(f"Detrend method must be one of: {self.VALID_DETRENDS}")
        if self.window_type not in self.VALID_WINDOWS:
            raise ValueError(
                f"Invalid window function: {self.window_type}. Valid options: {self.VALID_WINDOWS}"
            )
        if self.segment_length <= 0:
            raise ValueError("Segment length must be positive")
        if not (0 <= self.overlap_ratio < 1):
            raise ValueError("Overlap ratio must be in [0,1) range")

    def _apply_analysis_window(self, rr_ms: np.ndarray) -> np.ndarray:
        start_time, end_time = self.analysis_window
        rr_s = rr_ms.astype(float) / 1000.0
        time_points = np.cumsum(rr_s)
        time_points = np.concatenate([[0.0], time_points[:-1]])
        mask = (time_points >= start_time) & (time_points <= end_time)
        if not np.any(mask):
            raise ValueError(
                f"No data found in analysis window [{start_time}, {end_time}]"
            )
        return rr_ms[mask]

    def _create_time_domain_signal(self) -> np.ndarray:
        if len(self.rr_intervals_ms) == 0:
            return np.array([])

        rr_s = self.rr_intervals_ms.astype(float) / 1000.0
        time_points = np.cumsum(rr_s)
        time_points = np.concatenate([[0.0], time_points[:-1]])

        duration = time_points[-1]
        if duration <= 0:
            warnings.warn("Non-positive signal duration. Returning empty signal.")
            return np.array([])

        if duration < 60:
            warnings.warn(
                "Signal duration < 1 minute. Frequency domain results may be unreliable."
            )

        try:
            interp_func = interpolate.CubicSpline(time_points, rr_s, bc_type="natural")
        except Exception as e:
            warnings.warn(
                f"Cubic spline interpolation failed: {e}. Falling back to linear."
            )
            interp_func = interpolate.interp1d(
                time_points,
                rr_s,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

        dt = 1.0 / self.sampling_rate
        new_time_axis = np.arange(0.0, duration + 1e-12, dt)
        if new_time_axis.size < 4:
            warnings.warn(
                "Very few resampled samples for frequency analysis. Results may be unreliable."
            )

        resampled_rr_s = interp_func(new_time_axis)
        resampled_rr_s = np.clip(resampled_rr_s, 0.2, 3.0)
        return resampled_rr_s

    def _compute_welch_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.time_domain_s) == 0:
            return np.array([]), np.array([])

        requested = int(self.segment_length * self.sampling_rate)
        nperseg = min(requested, len(self.time_domain_s))
        if nperseg < 8:
            warnings.warn(
                f"Segment length ({nperseg}) too small for reliable Welch computation"
            )
            return np.array([]), np.array([])

        noverlap = int(nperseg * self.overlap_ratio)
        try:
            window = self._get_window(nperseg)
        except Exception as e:
            warnings.warn(
                f"Failed to create {self.window_type} window: {e}. Using Hann window."
            )
            window = signal.windows.hann(nperseg)

        if self.detrend_method == "smoothness_priors":
            try:
                detrended_signal = detrend_with_smoothness_priors(
                    self.time_domain_s,
                    lambda_param=self.detrend_lambda,
                    fs=self.sampling_rate,
                    return_trend=False,
                )
                freqs, psd_seconds = signal.welch(
                    x=detrended_signal,
                    fs=self.sampling_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=False,
                    scaling="density",
                    average="mean",
                )
            except Exception as e:
                warnings.warn(
                    f"Smoothness priors detrending failed: {e}. Falling back to linear detrend."
                )
                freqs, psd_seconds = signal.welch(
                    x=self.time_domain_s,
                    fs=self.sampling_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend="linear",
                    scaling="density",
                    average="mean",
                )
        else:
            detrend_param = self.detrend_method if self.detrend_method else False
            try:
                freqs, psd_seconds = signal.welch(
                    x=self.time_domain_s,
                    fs=self.sampling_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=detrend_param,
                    scaling="density",
                    average="mean",
                )
            except Exception as e:
                warnings.warn(f"Welch PSD computation failed: {e}")
                return np.array([]), np.array([])

        psd_ms2 = psd_seconds * 1e6
        if np.all(psd_ms2 == 0):
            warnings.warn("Welch PSD is zero everywhere. Check signal quality.")
        return freqs, psd_ms2

    def _compute_fft_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PSD using FFT-based periodogram method.
        This is a simple, non-averaged method that uses the entire signal.
        """
        if len(self.time_domain_s) == 0:
            return np.array([]), np.array([])

        n = len(self.time_domain_s)
        if n < 8:
            warnings.warn(
                f"Signal length ({n}) too small for reliable FFT computation"
            )
            return np.array([]), np.array([])

        # Apply detrending
        if self.detrend_method == "smoothness_priors":
            try:
                detrended_signal = detrend_with_smoothness_priors(
                    self.time_domain_s,
                    lambda_param=self.detrend_lambda,
                    fs=self.sampling_rate,
                    return_trend=False,
                )
            except Exception as e:
                warnings.warn(
                    f"Smoothness priors detrending failed: {e}. Falling back to linear detrend."
                )
                detrended_signal = signal.detrend(self.time_domain_s, type="linear")
        elif self.detrend_method == "linear":
            detrended_signal = signal.detrend(self.time_domain_s, type="linear")
        elif self.detrend_method == "constant":
            detrended_signal = signal.detrend(self.time_domain_s, type="constant")
        else:
            detrended_signal = self.time_domain_s.copy()

        # Apply window to entire signal
        try:
            window = self._get_window(n)
        except Exception as e:
            warnings.warn(
                f"Failed to create {self.window_type} window: {e}. Using Hann window."
            )
            window = signal.windows.hann(n)

        windowed_signal = detrended_signal * window

        # Compute FFT
        try:
            fft_result = np.fft.rfft(windowed_signal)
            freqs = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate)

            # Compute PSD (periodogram) in seconds^2/Hz
            # Scale factor accounts for windowing and sampling
            window_power = np.sum(window**2)
            psd_seconds = (np.abs(fft_result) ** 2) / (self.sampling_rate * window_power)

            # Convert to ms^2/Hz
            psd_ms2 = psd_seconds * 1e6

            if np.all(psd_ms2 == 0):
                warnings.warn("FFT PSD is zero everywhere. Check signal quality.")

            return freqs, psd_ms2

        except Exception as e:
            warnings.warn(f"FFT PSD computation failed: {e}")
            return np.array([]), np.array([])

    # Robust Burg (attempt) and Yule-Walker fallback
    @staticmethod
    def _burg_try(data: np.ndarray, order: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Try Burg estimate but be defensive. Return (ar_coefs, sigma2) or None on failure.
        Note: this implementation is conservative and will return None on any suspicious condition.
        """
        x = np.asarray(data, dtype=float)
        n = x.size
        if n <= order:
            return None

        # safe initializations
        f = x.copy()
        b = x.copy()
        ar = np.zeros(order, dtype=float)
        sigma2 = np.dot(x, x) / float(n)

        try:
            for m in range(order):
                # ensure valid slice lengths
                if (m + 1) >= n:
                    return None
                ef = f[m + 1 :]
                eb = b[m : n - 1]
                if ef.size == 0 or eb.size == 0:
                    return None

                num = -2.0 * np.dot(ef, eb)
                den = np.dot(ef, ef) + np.dot(eb, eb)
                if den == 0.0:
                    k = 0.0
                else:
                    k = num / den

                # update coefficients
                if m == 0:
                    ar[0] = k
                else:
                    ar_prev = ar[:m].copy()
                    ar[:m] = ar_prev + k * ar_prev[::-1]
                    ar[m] = k

                # update errors
                f_new = f.copy()
                b_new = b.copy()
                # slices must match - check sizes
                if (m + 1) < n:
                    f[m + 1 :] = f_new[m + 1 :] + k * b_new[m : n - 1]
                    b[m + 1 :] = b_new[m + 1 :] + k * f_new[m : n - 1]
                else:
                    return None

                sigma2 *= 1.0 - k * k
                # guard numerical issues
                if not np.isfinite(sigma2) or sigma2 <= 0:
                    return None
        except Exception:
            return None

        return ar, sigma2

    @staticmethod
    def _yule_walker_estimate(
        data: np.ndarray, order: int
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Yule-Walker estimator (biased autocorrelation).
        Returns (ar_coefs, sigma2) with convention A(z) = 1 + a1 z^-1 + ...
        """
        x = np.asarray(data, dtype=float)
        n = x.size
        if n <= order:
            return None

        # biased autocorrelation
        r = np.correlate(x, x, mode="full")
        mid = len(r) // 2
        r = r[mid : mid + order + 1] / float(n)

        R = signal.toeplitz(r[:order])
        rhs = -r[1 : order + 1]

        try:
            a = np.linalg.solve(R, rhs)
        except np.linalg.LinAlgError:
            return None
        # innovation variance
        sigma2 = r[0] + np.dot(a, r[1 : order + 1])
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return None
        return a, sigma2

    def _compute_ar_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute AR PSD using adaptive-order Burg (fallback Yule-Walker).
        If AR cannot be estimated, fall back to Welch PSD to avoid empty AR outputs.
        """
        # fallback default: return Welch PSD if AR fails
        fallback_freqs, fallback_psd = getattr(self, "freqs", np.array([])), getattr(
            self, "psd", np.array([])
        )

        if len(self.time_domain_s) == 0:
            return fallback_freqs, fallback_psd

        try:
            # detrend safely, with fallback
            try:
                if self.detrend_method == "smoothness_priors":
                    detrended_signal = detrend_with_smoothness_priors(
                        self.time_domain_s,
                        lambda_param=self.detrend_lambda,
                        fs=self.sampling_rate,
                        return_trend=False,
                    )
                elif self.detrend_method == "linear":
                    detrended_signal = signal.detrend(self.time_domain_s, type="linear")
                elif self.detrend_method == "constant":
                    detrended_signal = signal.detrend(
                        self.time_domain_s, type="constant"
                    )
                else:
                    detrended_signal = self.time_domain_s.copy()
            except Exception as e:
                warnings.warn(
                    f"Smoothness priors detrending failed (or other detrend issue): {e}. Falling back to linear/none."
                )
                if self.detrend_method != "linear":
                    detrended_signal = signal.detrend(self.time_domain_s, type="linear")
                else:
                    detrended_signal = self.time_domain_s.copy()

            n = len(detrended_signal)
            if n < 8:
                warnings.warn(
                    "Very short signal for AR estimation. Using Welch PSD as AR fallback."
                )
                return fallback_freqs, fallback_psd

            # determine adaptive order (heuristic)
            adaptive_max_order = max(4, n // 6)
            adaptive_order = min(self.ar_order, adaptive_max_order)
            adaptive_order = max(4, adaptive_order)  # ensure >=4

            ar_coefs = None
            sigma2 = None
            effective_order = None

            # try descending from adaptive_order down to 4
            for order_attempt in range(adaptive_order, 3, -1):
                # first try Burg (defensive)
                res = self._burg_try(detrended_signal, order_attempt)
                if res is not None:
                    ar_tmp, sigma2_tmp = res
                else:
                    # fallback to Yule-Walker
                    res_yw = self._yule_walker_estimate(detrended_signal, order_attempt)
                    if res_yw is None:
                        continue
                    ar_tmp, sigma2_tmp = res_yw

                # check sigma2 validity
                if sigma2_tmp is None or not np.isfinite(sigma2_tmp) or sigma2_tmp <= 0:
                    continue

                # Accept the model (we avoid complicated root checks here)
                ar_coefs = ar_tmp
                sigma2 = sigma2_tmp
                effective_order = order_attempt
                break

            if ar_coefs is None:
                warnings.warn(
                    "Could not estimate AR model reliably; using Welch PSD as AR fallback."
                )
                return fallback_freqs, fallback_psd

            # Build AR polynomial A(z) = 1 + a1 z^-1 + ...
            a_poly = np.concatenate([[1.0], ar_coefs])

            # compute frequency response
            nfft = 4096
            w, h = signal.freqz(b=[1.0], a=a_poly, worN=nfft, fs=self.sampling_rate)
            half = nfft // 2 + 1
            freqs = w[:half]
            h = h[:half]

            # PSD in seconds^2/Hz
            psd_seconds = (sigma2 / self.sampling_rate) * (np.abs(h) ** 2)
            psd_ms2 = psd_seconds * 1e6

            # sanity: if PSD all zeros or NaN, fallback
            if (
                psd_ms2.size == 0
                or not np.any(np.isfinite(psd_ms2))
                or np.all(psd_ms2 == 0)
            ):
                warnings.warn("AR PSD invalid (zeros/NaN). Falling back to Welch PSD.")
                return fallback_freqs, fallback_psd

            return freqs, psd_ms2

        except Exception as e:
            warnings.warn(
                f"AR spectrum computation failed: {e}. Falling back to Welch PSD."
            )
            return fallback_freqs, fallback_psd

    def _get_window(self, nperseg: int) -> np.ndarray:
        window_functions = {
            "hann": signal.windows.hann,
            "hamming": signal.windows.hamming,
            "blackman": signal.windows.blackman,
            "bartlett": signal.windows.bartlett,
            "flattop": signal.windows.flattop,
            "parzen": signal.windows.parzen,
            "bohman": signal.windows.bohman,
            "nuttall": signal.windows.nuttall,
        }

        window_func = window_functions.get(self.window_type)
        if window_func is None:
            raise ValueError(f"Unknown window function type: {self.window_type}")
        return window_func(nperseg)

    def _compute_spectral_metrics(self, use_ar: bool = False, use_fft: bool = False) -> Dict[str, float]:
        if use_ar:
            freqs = self.ar_freqs
            psd = self.ar_psd
        elif use_fft:
            freqs = self.fft_freqs
            psd = self.fft_psd
        else:
            freqs = self.freqs
            psd = self.psd

        default_results = {
            "ulf_power": 0.0,
            "ulf_power_nu": 0.0,
            "vlf_power": 0.0,
            "vlf_power_nu": 0.0,
            "lf_power": 0.0,
            "lf_power_nu": 0.0,
            "hf_power": 0.0,
            "hf_power_nu": 0.0,
            "lf_hf_ratio": float("nan"),
            "total_power": 0.0,
            "peak_freq_lf": float("nan"),
            "peak_freq_hf": float("nan"),
            "relative_lf_power": 0.0,
            "relative_hf_power": 0.0,
            "lf_nu": 0.0,
            "hf_nu": 0.0,
        }

        if len(psd) == 0 or len(freqs) == 0:
            warnings.warn(
                "PSD computation resulted in empty arrays. Returning default metrics."
            )
            return default_results

        try:
            total_power = np.trapz(psd, freqs)
            if total_power <= 0:
                warnings.warn("Total power is zero or negative. Returning defaults.")
                return default_results
        except Exception as e:
            warnings.warn(f"Total power calculation failed: {e}")
            return default_results

        results = {"total_power": total_power}

        for band, (low, high) in self.DEFAULT_FREQ_BANDS.items():
            if band == "lf_hf_ratio":
                continue

            if band == "vlf":
                first_idx = 1 if len(freqs) > 1 and freqs[0] == 0.0 else 0
                mask = (freqs >= low) & (freqs <= high)
                if first_idx == 1:
                    mask[0] = False
            else:
                mask = (freqs >= low) & (freqs <= high)

            if not np.any(mask):
                warnings.warn(
                    f"No frequency points found in {band} band [{low}, {high}] Hz"
                )
                band_power = 0.0
            else:
                try:
                    band_power = np.trapz(psd[mask], freqs[mask])
                    band_power = max(0.0, band_power)
                except Exception as e:
                    warnings.warn(f"Power calculation failed for {band} band: {e}")
                    band_power = 0.0

            band_pct_of_total = (
                (band_power / total_power) * 100.0 if total_power > 0 else 0.0
            )
            results[f"{band}_power"] = band_power
            results[f"{band}_power_nu"] = band_pct_of_total

        lf_power = results.get("lf_power", 0.0)
        hf_power = results.get("hf_power", 0.0)
        if hf_power > 1e-10:
            results["lf_hf_ratio"] = lf_power / hf_power
        else:
            results["lf_hf_ratio"] = float("inf") if lf_power > 1e-10 else float("nan")

        lf_hf_sum = lf_power + hf_power
        if lf_hf_sum > 0:
            results["relative_lf_power"] = (lf_power / lf_hf_sum) * 100.0
            results["relative_hf_power"] = (hf_power / lf_hf_sum) * 100.0
            results["lf_nu"] = (lf_power / lf_hf_sum) * 100.0
            results["hf_nu"] = (hf_power / lf_hf_sum) * 100.0
        else:
            results["relative_lf_power"] = 0.0
            results["relative_hf_power"] = 0.0
            results["lf_nu"] = 0.0
            results["hf_nu"] = 0.0

        results["peak_freq_lf"] = self._find_peak_frequency("lf", freqs, psd)
        results["peak_freq_hf"] = self._find_peak_frequency("hf", freqs, psd)

        return results

    def _find_peak_frequency(
        self, band: str, freqs: np.ndarray = None, psd: np.ndarray = None
    ) -> float:
        if freqs is None:
            freqs = self.freqs
        if psd is None:
            psd = self.psd

        if len(freqs) == 0 or len(psd) == 0:
            return float("nan")

        band_range = self.DEFAULT_FREQ_BANDS.get(band)
        if band_range is None:
            return float("nan")

        low, high = band_range
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return float("nan")

        band_psd = psd[mask]
        band_freqs = freqs[mask]
        peak_idx = np.argmax(band_psd)
        return float(band_freqs[peak_idx])

    def get_results(self) -> Dict[str, Union[float, Dict]]:
        results = {}
        results.update(self.spectral_metrics.copy())
        for key, value in self.spectral_metrics.items():
            results[f"welch_{key}"] = value
        for key, value in self.fft_spectral_metrics.items():
            results[f"fft_{key}"] = value
        for key, value in self.ar_spectral_metrics.items():
            results[f"ar_{key}"] = value

        results["welch_method"] = "Welch"
        results["fft_method"] = "FFT"
        results["ar_method"] = f"AR (order={self.ar_order})"

        results["analysis_info"] = {
            "sampling_rate": self.sampling_rate,
            "window_type": self.window_type,
            "detrend_method": self.detrend_method,
            "detrend_lambda": getattr(self, "detrend_lambda", None),
            "ar_order": getattr(self, "ar_order", None),
            "segment_length_s": self.segment_length,
            "overlap_ratio": self.overlap_ratio,
            "signal_duration_s": (
                len(self.time_domain_s) / self.sampling_rate
                if len(self.time_domain_s) > 0
                else 0
            ),
            "frequency_resolution_welch": (
                (self.freqs[1] - self.freqs[0]) if len(self.freqs) > 1 else 0
            ),
            "frequency_resolution_fft": (
                (self.fft_freqs[1] - self.fft_freqs[0]) if len(self.fft_freqs) > 1 else 0
            ),
            "frequency_resolution_ar": (
                (self.ar_freqs[1] - self.ar_freqs[0]) if len(self.ar_freqs) > 1 else 0
            ),
            "preprocessing_applied": self.preprocessing_result is not None,
            "analysis_window": self.analysis_window,
        }

        if self.preprocessing_result is not None:
            try:
                results["preprocessing_stats"] = {
                    "artifacts_detected": self.preprocessing_result.stats[
                        "artifacts_detected"
                    ],
                    "artifacts_corrected": self.preprocessing_result.stats[
                        "artifacts_corrected"
                    ],
                    "artifact_percentage": self.preprocessing_result.stats[
                        "artifact_percentage"
                    ],
                    "noise_segments": len(self.preprocessing_result.noise_segments),
                    "correction_method": self.preprocessing_result.correction_method,
                    "quality_flags": self.preprocessing_result.quality_flags,
                }
            except Exception:
                results["preprocessing_stats"] = self.preprocessing_result.__dict__

        return results

    def get_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.freqs.copy() if hasattr(self, "freqs") else np.array([]),
            self.psd.copy() if hasattr(self, "psd") else np.array([]),
        )

    def get_band_powers_summary(self) -> Dict[str, Dict[str, float]]:
        bands = ["ulf", "vlf", "lf", "hf"]
        summary = {}
        for band in bands:
            freq_range = self.DEFAULT_FREQ_BANDS[band]
            summary[band] = {
                "frequency_range_hz": freq_range,
                "absolute_power": self.spectral_metrics.get(f"{band}_power", 0.0),
                "relative_power_pct": self.spectral_metrics.get(
                    f"{band}_power_nu", 0.0
                ),
                "peak_frequency": self.spectral_metrics.get(
                    f"peak_freq_{band}", float("nan")
                ),
            }

        summary["lf_hf_ratio"] = {
            "value": self.spectral_metrics.get("lf_hf_ratio", float("nan")),
            "relative_lf_pct": self.spectral_metrics.get("relative_lf_power", 0.0),
            "relative_hf_pct": self.spectral_metrics.get("relative_hf_power", 0.0),
            "lf_nu": self.spectral_metrics.get("lf_nu", 0.0),
            "hf_nu": self.spectral_metrics.get("hf_nu", 0.0),
        }

        return summary

    def validate_frequency_analysis(self) -> Dict[str, Union[bool, str, float]]:
        validation = {
            "is_valid": True,
            "warnings": [],
            "recommendations": [],
            "signal_duration_s": (
                len(self.time_domain_s) / self.sampling_rate
                if len(self.time_domain_s) > 0
                else 0
            ),
            "frequency_resolution_hz": (
                (self.freqs[1] - self.freqs[0]) if len(self.freqs) > 1 else 0
            ),
        }

        duration = validation["signal_duration_s"]
        if duration < 120:
            validation["warnings"].append(
                "Signal duration < 2 minutes may produce unreliable frequency metrics"
            )
            if duration < 60:
                validation["is_valid"] = False

        freq_res = validation["frequency_resolution_hz"]
        if freq_res > 0.01:
            validation["warnings"].append(
                f"Poor frequency resolution ({freq_res:.4f} Hz). Consider longer segments."
            )

        total_power = self.spectral_metrics.get("total_power", 0)
        if total_power < 1e-10:
            validation["warnings"].append("Very low total power. Check signal quality.")
            validation["is_valid"] = False

        if self.preprocessing_result:
            try:
                artifact_pct = self.preprocessing_result.stats.get(
                    "artifact_percentage", 0
                )
                if artifact_pct > 10:
                    validation["warnings"].append(
                        f"High artifact percentage ({artifact_pct:.1f}%) may affect frequency metrics"
                    )
                if (
                    self.preprocessing_result.quality_flags
                    and self.preprocessing_result.quality_flags.get(
                        "poor_signal_quality"
                    )
                ):
                    validation["warnings"].append("Poor signal quality detected")
                    validation["is_valid"] = False
            except Exception:
                pass

        if duration < 300:
            validation["recommendations"].append(
                "Consider longer recordings (≥5 minutes) for stable frequency metrics"
            )
        if freq_res > 0.005:
            validation["recommendations"].append(
                "Consider longer segment lengths for better frequency resolution"
            )

        return validation
