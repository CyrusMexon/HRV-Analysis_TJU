import numpy as np
from scipy import signal, interpolate
from typing import Tuple, Dict, Optional, Union
import warnings

# Import from existing modules to maintain consistency
from hrvlib.preprocessing import PreprocessingResult

from hrvlib.signal_processing.smoothness_priors import detrend_with_smoothness_priors


class HRVFreqDomainAnalysis:
    """
    High-precision HRV frequency domain analysis integrated with DataBundle
    Uses Welch method for PSD estimation with adaptive preprocessing
    Implements standardized frequency bands and power calculations
    Integrates with preprocessing.py instead of internal preprocessing
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
        analysis_window: Optional[Tuple[float, float]] = None,
    ):
        """
        preprocessed_rri: Preprocessed RR intervals in milliseconds
            preprocessing_result: Results from preprocessing step
            sampling_rate: Resampling frequency in Hz (default 4 Hz)
            detrend_method: Detrending method ('linear', 'constant', 'smoothness_priors', None)
            detrend_lambda: Lambda parameter for smoothness_priors detrending (default 500)
            window_type: Window function type (default 'hann')
            segment_length: Segment length in seconds for Welch method
            overlap_ratio: Overlap ratio for segments (0-1)
            analysis_window: (start_time, end_time) in seconds for analysis window
        """
        self.rr_intervals = np.array(preprocessed_rri, dtype=float)
        self.preprocessing_result = preprocessing_result
        self.sampling_rate = sampling_rate
        self.detrend_method = detrend_method
        self.detrend_lambda = detrend_lambda
        self.window_type = window_type
        self.segment_length = segment_length
        self.overlap_ratio = overlap_ratio
        self.analysis_window = analysis_window

        self._validate_input()

        # Apply analysis window if specified
        if self.analysis_window is not None:
            self.rr_intervals = self._apply_analysis_window(self.rr_intervals)

        # Create time domain signal
        self.time_domain = self._create_time_domain_signal()

        # Compute PSD
        self.freqs, self.psd = self._compute_welch_psd()

        # Calculate spectral metrics
        self.spectral_metrics = self._compute_spectral_metrics()

    def _validate_input(self) -> None:
        """Validate all input parameters"""
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
        """Apply analysis window to RRI data"""
        start_time, end_time = self.analysis_window

        # Convert RRI to time points
        time_points = np.cumsum(rr_ms) / 1000.0  # Convert to seconds
        time_points = np.concatenate([[0], time_points[:-1]])

        # Find indices within analysis window
        mask = (time_points >= start_time) & (time_points <= end_time)

        if not np.any(mask):
            raise ValueError(
                f"No data found in analysis window [{start_time}, {end_time}]"
            )

        return rr_ms[mask]

    def _create_time_domain_signal(self) -> np.ndarray:
        """Create evenly sampled time series signal with error handling"""
        if len(self.rr_intervals) == 0:
            return np.array([])

        # Calculate cumulative time points in seconds
        time_points = np.cumsum(self.rr_intervals) / 1000.0
        time_points = np.concatenate([[0], time_points[:-1]])  # Start from zero

        # Check for minimum duration
        duration = time_points[-1]
        if duration < 60:  # Less than 1 minute
            warnings.warn(
                "Signal duration < 1 minute. Frequency domain results may be unreliable."
            )

        # Create cubic spline interpolation
        try:
            interp_func = interpolate.CubicSpline(
                time_points, self.rr_intervals, bc_type="natural"
            )
        except Exception as e:
            warnings.warn(
                f"Cubic spline interpolation failed: {e}. Using linear interpolation."
            )
            interp_func = interpolate.interp1d(
                time_points,
                self.rr_intervals,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

        # Create regular time axis
        num_samples = int(duration * self.sampling_rate)
        if num_samples < 10:
            warnings.warn(
                "Very few samples for frequency analysis. Results may be unreliable."
            )

        new_time_axis = np.linspace(0, duration, num_samples)

        # Apply interpolation
        resampled_signal = interp_func(new_time_axis)

        # Remove any remaining NaN values
        if np.any(np.isnan(resampled_signal)):
            warnings.warn("NaN values found in resampled signal. Interpolating.")
            nan_mask = np.isnan(resampled_signal)
            if not np.all(nan_mask):
                resampled_signal[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(~nan_mask)[0],
                    resampled_signal[~nan_mask],
                )

        return resampled_signal

    def _compute_welch_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density using Welch method with adaptive parameters"""
        if len(self.time_domain) == 0:
            return np.array([]), np.array([])

        # Calculate segment parameters
        nperseg = int(self.segment_length * self.sampling_rate)

        # Ensure nperseg doesn't exceed half signal length
        max_nperseg = len(self.time_domain) // 2
        if nperseg > max_nperseg:
            original_nperseg = nperseg
            nperseg = max_nperseg
            warnings.warn(
                f"Segment length ({original_nperseg}) exceeds half signal length. "
                f"Adjusted to {nperseg}"
            )

        # Ensure minimum segment size
        if nperseg < 8:
            warnings.warn(
                f"Segment length ({nperseg}) too small for reliable Welch computation"
            )
            return np.array([]), np.array([])

        noverlap = int(nperseg * self.overlap_ratio)

        # Get window function
        try:
            window = self._get_window(nperseg)
        except Exception as e:
            warnings.warn(
                f"Failed to create {self.window_type} window: {e}. Using Hann window."
            )
            window = signal.windows.hann(nperseg)

        # Handle smoothness priors detrending separately
        if self.detrend_method == "smoothness_priors":
            # Apply smoothness priors to the time domain signal BEFORE Welch
            try:
                detrended_signal = detrend_with_smoothness_priors(
                    self.time_domain,
                    lambda_param=self.detrend_lambda,
                    fs=self.sampling_rate,
                    return_trend=False,
                )

                # Now compute Welch WITHOUT additional detrending (we already detrended)
                freqs, psd = signal.welch(
                    x=detrended_signal,
                    fs=self.sampling_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=False,  # ✅ No detrending - already done!
                    scaling="density",
                    average="mean",
                )

            except Exception as e:
                warnings.warn(
                    f"Smoothness priors detrending failed: {e}. "
                    f"Falling back to linear detrending."
                )
                # Fallback to linear detrending
                freqs, psd = signal.welch(
                    x=self.time_domain,
                    fs=self.sampling_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend="linear",
                    scaling="density",
                    average="mean",
                )

        else:
            # Standard scipy detrending (linear, constant, or None)
            # Map None to False for scipy
            detrend_param = self.detrend_method if self.detrend_method else False

            try:
                freqs, psd = signal.welch(
                    x=self.time_domain,
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

        # Validate PSD results
        if len(psd) == 0 or np.all(psd == 0):
            warnings.warn("PSD computation resulted in empty or zero power spectrum")

        return freqs, psd

    def _get_window(self, nperseg: int) -> np.ndarray:
        """Generate specified window function"""
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

    def _compute_spectral_metrics(self) -> Dict[str, float]:
        """Calculate all frequency domain metrics with robust integration"""
        # Initialize default values
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
        }

        # Check for valid PSD
        if len(self.psd) == 0 or len(self.freqs) == 0:
            warnings.warn(
                "PSD computation resulted in empty arrays. Returning default values."
            )
            return default_results

        # Calculate total power using trapezoidal integration
        try:
            total_power = np.trapezoid(self.psd, self.freqs)
            if total_power <= 0:
                warnings.warn("Total power is zero or negative. Check signal quality.")
                return default_results
        except Exception as e:
            warnings.warn(f"Total power calculation failed: {e}")
            return default_results

        results = {"total_power": total_power}

        # Calculate power in each frequency band
        for band, (low, high) in self.DEFAULT_FREQ_BANDS.items():
            if band == "lf_hf_ratio":
                continue

            # Find frequency indices in band
            mask = (self.freqs >= low) & (self.freqs <= high)

            if not np.any(mask):
                warnings.warn(f"No frequency points in {band} band [{low}, {high}] Hz")
                band_power = 0.0
            else:
                try:
                    band_power = np.trapezoid(self.psd[mask], self.freqs[mask])
                    band_power = max(0, band_power)  # Ensure non-negative
                except Exception as e:
                    warnings.warn(f"Power calculation failed for {band} band: {e}")
                    band_power = 0.0

            # Calculate normalized power (percentage of total)
            norm_power = (band_power / total_power) * 100 if total_power > 0 else 0.0

            results[f"{band}_power"] = band_power
            results[f"{band}_power_nu"] = norm_power

        # Calculate LF/HF ratio with robust handling
        lf_power = results.get("lf_power", 0.0)
        hf_power = results.get("hf_power", 0.0)

        if hf_power > 1e-10:  # Avoid division by very small numbers
            results["lf_hf_ratio"] = lf_power / hf_power
        else:
            if lf_power > 1e-10:
                results["lf_hf_ratio"] = float("inf")
            else:
                results["lf_hf_ratio"] = float("nan")

        # Calculate relative powers (LF and HF as percentage of LF+HF)
        lf_hf_sum = lf_power + hf_power
        if lf_hf_sum > 0:
            results["relative_lf_power"] = (lf_power / lf_hf_sum) * 100
            results["relative_hf_power"] = (hf_power / lf_hf_sum) * 100
        else:
            results["relative_lf_power"] = 0.0
            results["relative_hf_power"] = 0.0

        # Find peak frequencies in LF and HF bands
        results["peak_freq_lf"] = self._find_peak_frequency("lf")
        results["peak_freq_hf"] = self._find_peak_frequency("hf")

        return results

    def _find_peak_frequency(self, band: str) -> float:
        """Find peak frequency within specified band"""
        if len(self.freqs) == 0 or len(self.psd) == 0:
            return float("nan")

        band_range = self.DEFAULT_FREQ_BANDS.get(band)
        if band_range is None:
            return float("nan")

        low, high = band_range
        mask = (self.freqs >= low) & (self.freqs <= high)

        if not np.any(mask):
            return float("nan")

        band_psd = self.psd[mask]
        band_freqs = self.freqs[mask]

        if len(band_psd) == 0:
            return float("nan")

        # Find index of maximum power
        peak_idx = np.argmax(band_psd)
        return band_freqs[peak_idx]

    def get_results(self) -> Dict[str, Union[float, Dict]]:
        """Return comprehensive frequency domain results"""
        results = self.spectral_metrics.copy()

        # Add analysis metadata
        results["analysis_info"] = {
            "sampling_rate": self.sampling_rate,
            "window_type": self.window_type,
            "detrend_method": self.detrend_method,
            "segment_length_s": self.segment_length,
            "overlap_ratio": self.overlap_ratio,
            "signal_duration_s": (
                len(self.time_domain) / self.sampling_rate
                if len(self.time_domain) > 0
                else 0
            ),
            "frequency_resolution": (
                self.freqs[1] - self.freqs[0] if len(self.freqs) > 1 else 0
            ),
            "preprocessing_applied": self.preprocessing_result is not None,
            "analysis_window": self.analysis_window,
        }

        # Add preprocessing statistics if available
        if self.preprocessing_result is not None:
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

        return results

    def get_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return frequency and PSD arrays for plotting"""
        return self.freqs.copy(), self.psd.copy()

    def get_band_powers_summary(self) -> Dict[str, Dict[str, float]]:
        """Get organized summary of power in each frequency band"""
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

        # Add LF/HF ratio
        summary["lf_hf_ratio"] = {
            "value": self.spectral_metrics.get("lf_hf_ratio", float("nan")),
            "relative_lf_pct": self.spectral_metrics.get("relative_lf_power", 0.0),
            "relative_hf_pct": self.spectral_metrics.get("relative_hf_power", 0.0),
        }

        return summary

    def validate_frequency_analysis(self) -> Dict[str, Union[bool, str, float]]:
        """Validate frequency domain analysis results and provide recommendations"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "recommendations": [],
            "signal_duration_s": (
                len(self.time_domain) / self.sampling_rate
                if len(self.time_domain) > 0
                else 0
            ),
            "frequency_resolution_hz": (
                self.freqs[1] - self.freqs[0] if len(self.freqs) > 1 else 0
            ),
        }

        # Check signal duration
        duration = validation["signal_duration_s"]
        if duration < 120:  # Less than 2 minutes
            validation["warnings"].append(
                "Signal duration < 2 minutes may produce unreliable frequency metrics"
            )
            if duration < 60:
                validation["is_valid"] = False

        # Check frequency resolution
        freq_res = validation["frequency_resolution_hz"]
        if freq_res > 0.01:  # Resolution worse than 0.01 Hz
            validation["warnings"].append(
                f"Poor frequency resolution ({freq_res:.4f} Hz). Consider longer segments."
            )

        # Check for sufficient power
        total_power = self.spectral_metrics.get("total_power", 0)
        if total_power < 1e-10:
            validation["warnings"].append("Very low total power. Check signal quality.")
            validation["is_valid"] = False

        # Check preprocessing quality
        if self.preprocessing_result:
            artifact_pct = self.preprocessing_result.stats.get("artifact_percentage", 0)
            if artifact_pct > 10:
                validation["warnings"].append(
                    f"High artifact percentage ({artifact_pct:.1f}%) may affect frequency metrics"
                )

            if self.preprocessing_result.quality_flags:
                if self.preprocessing_result.quality_flags.get("poor_signal_quality"):
                    validation["warnings"].append("Poor signal quality detected")
                    validation["is_valid"] = False

        # Recommendations
        if duration < 300:  # Less than 5 minutes
            validation["recommendations"].append(
                "Consider longer recordings (≥5 minutes) for stable frequency metrics"
            )

        if freq_res > 0.005:  # Resolution worse than 5 mHz
            validation["recommendations"].append(
                "Consider longer segment lengths for better frequency resolution"
            )

        return validation
