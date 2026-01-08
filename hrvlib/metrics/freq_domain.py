import numpy as np
from scipy import signal, interpolate, linalg
from typing import Tuple, Dict, Optional, Union
import warnings

# Keep this import (you had it in the original file).
from hrvlib.preprocessing import PreprocessingResult

from hrvlib.signal_processing.smoothness_priors import (
    detrend_uniform_with_smoothness_priors,
)


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
        "vlf": (0.0, 0.04),
        "lf": (0.04, 0.15),
        "hf": (0.15, 0.4),
        "lf_hf_ratio": (0.04, 0.4),
    }

    def __init__(
        self,
        preprocessed_rri: np.ndarray,
        preprocessing_result: Optional[PreprocessingResult] = None,
        sampling_rate: float = 4.0,
        detrend_method: Optional[str] = None,
        detrend_lambda: float = 500,
        window_type: str = "hann",
        segment_length: float = 120.0,
        overlap_ratio: float = 0.75,
        ar_order: int = 16,
        clip_rr_resampled: bool = True,
        rr_clip_range: Tuple[float, float] = (0.2, 3.0),
        welch_shorten_if_short: bool = True,
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
        self.clip_rr_resampled = bool(clip_rr_resampled)
        self.rr_clip_range = tuple(rr_clip_range)
        self.welch_shorten_if_short = bool(welch_shorten_if_short)
        self.analysis_window = analysis_window

        self._validate_input()

        # Auto-detect RR units (seconds vs milliseconds vs microseconds).
        mean_rr = np.mean(self.rr_intervals_ms) if self.rr_intervals_ms.size else 0.0
        if mean_rr > 0:
            if mean_rr < 10:
                warnings.warn(
                    f"RR intervals appear to be in seconds (mean={mean_rr:.3f}s); converting to ms."
                )
                self.rr_intervals_ms = self.rr_intervals_ms * 1000.0
            elif mean_rr > 10000:
                warnings.warn(
                    f"RR intervals appear to be in microseconds (mean={mean_rr:.0f}us); converting to ms."
                )
                self.rr_intervals_ms = self.rr_intervals_ms / 1000.0
            elif not (200 < mean_rr < 3000):
                warnings.warn(
                    f"RR intervals mean ({mean_rr:.2f}) is unusual for ms; verify input units."
                )

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
        self.time_domain_s = self._create_time_domain_signal(
            rr_values_ms=self.rr_intervals_ms,
            clip_rr=self.clip_rr_resampled,
            clip_range=self.rr_clip_range,
        )
        self._smoothness_priors_applied = False
        if self.detrend_method == "smoothness_priors":
            try:
                self.time_domain_s = detrend_uniform_with_smoothness_priors(
                    self.time_domain_s,
                    lambda_param=self.detrend_lambda,
                    return_trend=False,
                )
                self._smoothness_priors_applied = True
            except Exception as e:
                warnings.warn(
                    f"Smoothness priors detrending failed on resampled signal: {e}. Falling back to linear detrend."
                )

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
        if self.clip_rr_resampled:
            if (
                not isinstance(self.rr_clip_range, tuple)
                or len(self.rr_clip_range) != 2
            ):
                raise ValueError("rr_clip_range must be a (min, max) tuple")
            clip_min, clip_max = self.rr_clip_range
            if clip_min <= 0 or clip_max <= 0 or clip_min >= clip_max:
                raise ValueError("rr_clip_range must be positive and min < max")

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

    def _create_time_domain_signal(
        self,
        rr_values_ms: np.ndarray,
        rr_time_ms: Optional[np.ndarray] = None,
        clip_rr: bool = True,
        clip_range: Tuple[float, float] = (0.2, 3.0),
    ) -> np.ndarray:
        if len(rr_values_ms) == 0:
            return np.array([])

        rr_values_s = rr_values_ms.astype(float) / 1000.0
        if rr_time_ms is None:
            rr_time_s = rr_values_s
        else:
            if len(rr_time_ms) != len(rr_values_ms):
                raise ValueError("rr_time_ms and rr_values_ms must be the same length")
            rr_time_s = rr_time_ms.astype(float) / 1000.0

        time_points = np.cumsum(rr_time_s)
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
            interp_func = interpolate.CubicSpline(
                time_points, rr_values_s, bc_type="natural"
            )
        except Exception as e:
            warnings.warn(
                f"Cubic spline interpolation failed: {e}. Falling back to linear."
            )
            interp_func = interpolate.interp1d(
                time_points,
                rr_values_s,
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
        if clip_rr:
            resampled_rr_s = np.clip(resampled_rr_s, clip_range[0], clip_range[1])
        return resampled_rr_s

    def _compute_welch_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.time_domain_s) == 0:
            return np.array([]), np.array([])

        requested = int(self.segment_length * self.sampling_rate)
        nperseg = min(requested, len(self.time_domain_s))

        # Optionally shorten when recording is shorter than requested window.
        if self.welch_shorten_if_short and nperseg == len(self.time_domain_s) and nperseg >= 16:
            nperseg = max(8, len(self.time_domain_s) // 2)

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
            if self._smoothness_priors_applied:
                detrended_signal = self.time_domain_s
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
            else:
                warnings.warn(
                    "Smoothness priors pre-detrending failed; falling back to linear detrend."
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
            if self._smoothness_priors_applied:
                detrended_signal = self.time_domain_s
            else:
                warnings.warn(
                    "Smoothness priors pre-detrending failed; falling back to linear detrend."
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
            # Proper scaling for one-sided PSD with windowing
            # Reference: Heinzel et al. "Spectrum and spectral density estimation by the DFT"
            S2 = np.sum(window**2) / n  # Window power normalization factor
            psd_seconds = (2.0 / (self.sampling_rate * n * S2)) * (np.abs(fft_result) ** 2)

            # Correct DC and Nyquist components (should not be doubled)
            psd_seconds[0] /= 2.0
            if n % 2 == 0 and len(psd_seconds) > 1:
                psd_seconds[-1] /= 2.0

            # Convert to ms^2/Hz
            psd_ms2 = psd_seconds * 1e6

            if np.all(psd_ms2 == 0):
                warnings.warn("FFT PSD is zero everywhere. Check signal quality.")

            return freqs, psd_ms2

        except Exception as e:
            warnings.warn(f"FFT PSD computation failed: {e}")
            return np.array([]), np.array([])

    # Burg estimator and Yule-Walker fallback
    @staticmethod
    def _burg_try(data: np.ndarray, order: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Burg AR estimation with stability checks.
        Return (ar_coefs, sigma2) or None on failure.
        """
        x = np.asarray(data, dtype=float)
        n = x.size
        if n <= order:
            return None

        # Initialize forward/backward prediction errors
        ef = x[1:].copy()
        eb = x[:-1].copy()
        a = np.zeros(order + 1, dtype=float)
        a[0] = 1.0
        sigma2 = np.dot(x, x) / float(n)
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return None

        try:
            for m in range(1, order + 1):
                num = -2.0 * np.dot(eb, ef)
                den = np.dot(ef, ef) + np.dot(eb, eb)
                if den <= 0.0:
                    return None
                k = num / den
                if not np.isfinite(k) or abs(k) >= 1.0:
                    return None

                a_prev = a.copy()
                a[1:m] = a_prev[1:m] + k * a_prev[m - 1 : 0 : -1]
                a[m] = k

                ef_new = ef[1:] + k * eb[1:]
                eb_new = eb[:-1] + k * ef[:-1]
                ef, eb = ef_new, eb_new

                sigma2 *= 1.0 - k * k
                if not np.isfinite(sigma2) or sigma2 <= 0:
                    return None
        except Exception:
            return None

        return a[1:], sigma2

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

        R = linalg.toeplitz(r[:order])
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
                    if self._smoothness_priors_applied:
                        detrended_signal = self.time_domain_s
                    else:
                        warnings.warn(
                            "Smoothness priors pre-detrending failed; falling back to linear detrend."
                        )
                        detrended_signal = signal.detrend(
                            self.time_domain_s, type="linear"
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

            ar_input = detrended_signal - np.mean(detrended_signal)

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
                res = self._burg_try(ar_input, order_attempt)
                if res is not None:
                    ar_tmp, sigma2_tmp = res
                else:
                    # fallback to Yule-Walker
                    res_yw = self._yule_walker_estimate(ar_input, order_attempt)
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
            freqs, h = signal.freqz(
                b=[1.0], a=a_poly, worN=nfft, fs=self.sampling_rate
            )

            # PSD in seconds^2/Hz
            psd_seconds = (sigma2 / self.sampling_rate) * (np.abs(h) ** 2)
            # Normalize AR PSD so its integral matches the signal variance.
            target_var = np.var(ar_input)
            if np.isfinite(target_var) and target_var > 0:
                psd_power = np.trapezoid(psd_seconds, freqs)
                if np.isfinite(psd_power) and psd_power > 0:
                    psd_seconds = psd_seconds * (target_var / psd_power)
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

    def _compute_psd_diagnostics(
        self, freqs: np.ndarray, psd: np.ndarray
    ) -> Dict[str, float]:
        diagnostics = {
            "total_power_full": 0.0,
            "total_power_band_0_0p4": 0.0,
            "power_below_vlf": 0.0,
            "power_above_hf": 0.0,
            "fraction_in_band": 0.0,
            "fraction_above_hf": 0.0,
            "freq_min": float("nan"),
            "freq_max": float("nan"),
        }

        if len(psd) == 0 or len(freqs) == 0:
            return diagnostics

        diagnostics["freq_min"] = float(freqs[0])
        diagnostics["freq_max"] = float(freqs[-1])

        full = np.trapezoid(psd, freqs)
        if not np.isfinite(full) or full <= 0:
            return diagnostics

        low = self.DEFAULT_FREQ_BANDS["vlf"][0]
        high = self.DEFAULT_FREQ_BANDS["hf"][1]
        mask_band = (freqs >= low) & (freqs <= high)
        mask_below = freqs < low
        mask_above = freqs > high

        band_power = np.trapezoid(psd[mask_band], freqs[mask_band]) if np.any(mask_band) else 0.0
        below_power = np.trapezoid(psd[mask_below], freqs[mask_below]) if np.any(mask_below) else 0.0
        above_power = np.trapezoid(psd[mask_above], freqs[mask_above]) if np.any(mask_above) else 0.0

        diagnostics["total_power_full"] = float(full)
        diagnostics["total_power_band_0_0p4"] = float(band_power)
        diagnostics["power_below_vlf"] = float(below_power)
        diagnostics["power_above_hf"] = float(above_power)
        diagnostics["fraction_in_band"] = float(band_power / full)
        diagnostics["fraction_above_hf"] = float(above_power / full)

        return diagnostics

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
            "peak_freq_vlf": float("nan"),
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
            # Kubios-style total power: integrate 0.0-0.4 Hz (VLF+LF+HF)
            total_low = self.DEFAULT_FREQ_BANDS["vlf"][0]
            total_high = self.DEFAULT_FREQ_BANDS["hf"][1]
            total_mask = (freqs >= total_low) & (freqs <= total_high)
            if not np.any(total_mask):
                warnings.warn(
                    "No frequency points found in total power band [0.0, 0.4] Hz."
                )
                return default_results
            total_power = np.trapezoid(psd[total_mask], freqs[total_mask])
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
            mask = (freqs >= low) & (freqs <= high)

            if not np.any(mask):
                warnings.warn(
                    f"No frequency points found in {band} band [{low}, {high}] Hz"
                )
                band_power = 0.0
            else:
                try:
                    band_power = np.trapezoid(psd[mask], freqs[mask])
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

        results["peak_freq_vlf"] = self._find_peak_frequency("vlf", freqs, psd)
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
            "clip_rr_resampled": self.clip_rr_resampled,
            "rr_clip_range_s": self.rr_clip_range,
            "welch_shorten_if_short": self.welch_shorten_if_short,
        }

        results["ar_psd_diagnostics"] = self._compute_psd_diagnostics(
            self.ar_freqs, self.ar_psd
        )

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
