"""
Respiratory Analysis Module for HRV
FR-21: EDR-AM respiratory rate estimation from ECG
FR-22: RSA coherence computation (0.1-0.5 Hz band)
FR-23: LF/HF band overlap detection and annotation

Integrates with existing DataBundle and TimeSeries structures.
"""

import warnings
from typing import Optional, List, Dict, Tuple
import numpy as np
from scipy import signal, interpolate
from scipy.signal import coherence, butter, filtfilt, find_peaks

# Import from existing modules to maintain consistency
from hrvlib.data_handler import DataBundle, TimeSeries
from hrvlib.preprocessing import PreprocessingResult


def estimate_respiratory_rate_edr_am(
    ecg_timeseries: TimeSeries,
) -> Tuple[np.ndarray, float, float]:
    """
    FR-21: Estimate respiratory rate from ECG using EDR-AM method.

    Args:
        ecg_timeseries: TimeSeries object containing ECG data

    Returns:
        Tuple of (respiratory_signal, respiratory_rate_bpm, confidence)
    """
    # ADDED: Input validation
    if ecg_timeseries is None or ecg_timeseries.data is None:
        raise ValueError("ECG timeseries or data is None")

    if len(ecg_timeseries.data) == 0:
        raise ValueError("ECG data is empty")

    if np.all(np.isnan(ecg_timeseries.data)) or np.all(ecg_timeseries.data == 0):
        raise ValueError("ECG data contains only NaN or zero values")

    # Continue with existing neurokit2 logic...
    try:
        import neurokit2 as nk
    except ImportError:
        warnings.warn("neurokit2 not available; using fallback peak detection")
        return _estimate_edr_fallback(ecg_timeseries)

    ecg_data = ecg_timeseries.data
    fs = ecg_timeseries.fs

    if len(ecg_data) < fs * 10:  # Less than 10 seconds
        raise ValueError("ECG signal too short for reliable respiratory estimation")

    try:
        # Clean ECG and detect R-peaks
        ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=fs)
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        r_peak_indices = rpeaks["ECG_R_Peaks"]

        # Added: Better valiudation
        if len(r_peak_indices) == 0:
            raise ValueError("No R-peaks detected by NeuroKit2")

        if len(r_peak_indices) < 10:
            raise ValueError("Insufficient R-peaks for EDR analysis")

    except Exception as e:
        warnings.warn(f"NeuroKit2 processing failed: {e}, using fallback")
        return _estimate_edr_fallback(ecg_timeseries)

    # Extract R-peak amplitudes (EDR-AM method)
    r_amplitudes = ecg_cleaned[r_peak_indices]
    r_times = r_peak_indices / fs

    # Validation for r_amplitudes
    if len(r_amplitudes) == 0:
        raise ValueError("No valid R-peak amplitudes extracted")

    # Remove amplitude outliers

    if len(r_amplitudes) > 4:  # Only remove outliers if we have enough data
        q25, q75 = np.percentile(r_amplitudes, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        valid_mask = (r_amplitudes >= lower_bound) & (r_amplitudes <= upper_bound)

        # FIX: Only use filtered data if we still have enough points
        if np.sum(valid_mask) >= 5:
            r_amplitudes_clean = r_amplitudes[valid_mask]
            r_times_clean = r_times[valid_mask]
        else:
            warnings.warn("Too many R-peak outliers; using all peaks")
            r_amplitudes_clean = r_amplitudes
            r_times_clean = r_times
    else:
        # Not enough data to filter outliers
        r_amplitudes_clean = r_amplitudes
        r_times_clean = r_times

    # UPDATED: Ensure minimum data points for interpolation
    if len(r_times_clean) < 2:
        raise ValueError("Insufficient data points for respiratory estimation")

    # Interpolate to regular 4 Hz grid
    fs_edr = 4.0
    time_regular = np.arange(r_times_clean[0], r_times_clean[-1], 1 / fs_edr)

    # Ensure enough samples for analysis
    if len(time_regular) < 10:
        raise ValueError("Signal duration too short for respiratory analysis")

    # Cubic spline interpolation
    cs = interpolate.CubicSpline(r_times_clean, r_amplitudes_clean, bc_type="natural")
    resp_signal = cs(time_regular)

    # Band-pass filter for respiratory frequencies (0.1-0.5 Hz)
    nyquist = fs_edr / 2
    low = 0.1 / nyquist
    high = min(0.5 / nyquist, 0.99)  # Ensure below Nyquist

    try:
        b, a = butter(3, [low, high], btype="band")
        resp_signal = filtfilt(b, a, resp_signal)
    except Exception as e:
        warnings.warn(f"Respiratory filtering failed: {e}")

    # Calculate respiratory rate
    resp_rate_bpm = _calculate_resp_rate_from_signal(resp_signal, fs_edr)

    # Calculate confidence based on signal quality
    confidence = _calculate_edr_confidence(resp_signal, r_amplitudes_clean)

    return resp_signal, resp_rate_bpm, confidence


def _estimate_edr_fallback(
    ecg_timeseries: TimeSeries,
) -> Tuple[np.ndarray, float, float]:
    """Fallback EDR estimation without NeuroKit2"""
    ecg_data = ecg_timeseries.data
    fs = ecg_timeseries.fs

    # Validate input data
    if len(ecg_data) == 0 or np.all(np.isnan(ecg_data)):
        raise ValueError("ECG data is empty or contains only NaN values")

    # Simple R-peak detection using find_peaks
    height_threshold = np.max(np.abs(ecg_data)) * 0.3
    min_distance = int(fs * 0.4)  # Minimum 400ms between peaks

    r_peak_indices, _ = find_peaks(
        ecg_data, height=height_threshold, distance=min_distance
    )

    # Ensure minimum peaks for fallback
    if len(r_peak_indices) < 5:
        # Try with lower threshold
        height_threshold = np.max(np.abs(ecg_data)) * 0.1
        r_peak_indices, _ = find_peaks(
            ecg_data, height=height_threshold, distance=min_distance
        )

    if len(r_peak_indices) < 5:
        # Try even more permissive settings
        min_distance = int(fs * 0.3)  # Reduce minimum distance
        height_threshold = np.std(ecg_data) * 1.5  # Use std-based threshold
        r_peak_indices, _ = find_peaks(
            ecg_data, height=height_threshold, distance=min_distance
        )

    if len(r_peak_indices) < 5:
        raise ValueError("Insufficient R-peaks detected with fallback method")

    # Extract amplitudes and continue with same EDR-AM process
    r_amplitudes = ecg_data[r_peak_indices]
    r_times = r_peak_indices / fs

    if len(r_amplitudes) == 0 or len(r_times) == 0:
        raise ValueError("No valid R-peak amplitudes or times extracted")

    # Ensure interpolation is possible
    if len(r_times) < 2:
        raise ValueError("Insufficient data points for fallback estimation")

    # Same interpolation and filtering as main method
    fs_edr = 4.0
    time_regular = np.arange(r_times[0], r_times[-1], 1 / fs_edr)

    # Bounds checking for interpolation
    if len(time_regular) < 5:
        raise ValueError("Signal duration too short for respiratory analysis")

    cs = interpolate.CubicSpline(r_times, r_amplitudes, bc_type="natural")
    resp_signal = cs(time_regular)

    # Filter
    nyquist = fs_edr / 2
    low, high = 0.1 / nyquist, min(0.5 / nyquist, 0.99)
    b, a = butter(3, [low, high], btype="band")
    resp_signal = filtfilt(b, a, resp_signal)

    resp_rate_bpm = _calculate_resp_rate_from_signal(resp_signal, fs_edr)
    confidence = (
        _calculate_edr_confidence(resp_signal, r_amplitudes) * 0.7
    )  # Lower confidence for fallback

    return resp_signal, resp_rate_bpm, confidence


def _calculate_resp_rate_from_signal(resp_signal: np.ndarray, fs: float) -> float:
    """Calculate respiratory rate from respiratory signal"""
    # Method 1: Peak detection
    min_distance = int(fs * 2.0)  # Min 2 seconds between breaths
    peaks, _ = find_peaks(resp_signal, distance=min_distance)

    if len(peaks) > 2:
        intervals = np.diff(peaks) / fs
        rate_from_peaks = 60.0 / np.median(intervals)
    else:
        rate_from_peaks = 0

    # Method 2: FFT peak
    f, psd = signal.welch(resp_signal, fs, nperseg=min(len(resp_signal), int(fs * 30)))
    resp_mask = (f >= 0.1) & (f <= 0.5)

    if np.any(resp_mask):
        peak_idx = np.argmax(psd[resp_mask])
        peak_freq = f[resp_mask][peak_idx]
        rate_from_fft = peak_freq * 60
    else:
        rate_from_fft = 0

    # Use FFT result if available, otherwise peaks
    if rate_from_fft > 0:
        return rate_from_fft
    elif rate_from_peaks > 0:
        return rate_from_peaks
    else:
        warnings.warn("Could not estimate respiratory rate")
        return 15.0  # Default fallback


def _calculate_edr_confidence(
    resp_signal: np.ndarray, r_amplitudes: np.ndarray
) -> float:
    """Calculate confidence score for EDR signal quality"""
    # Signal quality metrics
    snr = np.var(resp_signal) / (np.var(np.diff(resp_signal)) + 1e-10)
    snr_score = min(snr / 5, 1.0)

    # R-peak amplitude variation (good for EDR)
    amp_cv = np.std(r_amplitudes) / (np.mean(r_amplitudes) + 1e-10)
    cv_score = min(amp_cv * 5, 1.0)

    return (snr_score + cv_score) / 2


def compute_rsa_coherence(
    bundle: DataBundle, fs_target: float = 4.0
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    FR-22: Compute RSA coherence between HRV and respiration in 0.1-0.5 Hz band.

    Args:
        bundle: DataBundle with RRI data and respiratory signal
        fs_target: Target sampling frequency for coherence analysis

    Returns:
        Tuple of (mean_coherence, freq_array, coherence_array, peak_freq)
    """
    # Get RRI data
    if bundle.preprocessing and bundle.preprocessing.corrected_rri is not None:
        rri_ms = bundle.preprocessing.corrected_rri
    elif bundle.rri_ms:
        rri_ms = np.array(bundle.rri_ms)
    else:
        raise ValueError("No RRI data available for RSA coherence")

    if len(rri_ms) < 10:
        raise ValueError("Insufficient RRI data for coherence analysis")

    # Convert RRI to regularly sampled HRV signal
    time_rri = np.cumsum(rri_ms) / 1000.0  # Convert to seconds
    time_rri = np.concatenate([[0], time_rri[:-1]])

    # Create regular time grid
    duration = time_rri[-1]
    time_regular = np.arange(0, duration, 1 / fs_target)

    if len(time_regular) < 20:
        raise ValueError("Signal too short for coherence analysis")

    # Interpolate RRI to regular grid
    cs = interpolate.CubicSpline(time_rri, rri_ms, bc_type="natural")
    hrv_signal = cs(time_regular)

    # Get respiratory signal
    resp_signal = None

    # Priority 1: Use RESP channel if available
    if bundle.resp:
        resp_ts = bundle.resp[0]
        # Resample to target frequency if needed
        if resp_ts.fs != fs_target:
            n_samples = int(len(resp_ts.data) * fs_target / resp_ts.fs)
            resp_signal = signal.resample(resp_ts.data, n_samples)
        else:
            resp_signal = resp_ts.data

        # Trim to match HRV signal length
        min_len = min(len(hrv_signal), len(resp_signal))
        hrv_signal = hrv_signal[:min_len]
        resp_signal = resp_signal[:min_len]

    # Priority 2: Use EDR-AM from ECG
    elif bundle.ecg:
        try:
            resp_signal, _, _ = estimate_respiratory_rate_edr_am(bundle.ecg[0])
            # Trim to match HRV signal length
            min_len = min(len(hrv_signal), len(resp_signal))
            hrv_signal = hrv_signal[:min_len]
            resp_signal = resp_signal[:min_len]
        except Exception as e:
            raise ValueError(f"Failed to derive respiratory signal: {e}")
    else:
        raise ValueError("No respiratory or ECG signals available")

    # Normalize signals
    hrv_signal = (hrv_signal - np.mean(hrv_signal)) / np.std(hrv_signal)
    resp_signal = (resp_signal - np.mean(resp_signal)) / np.std(resp_signal)

    # Compute coherence
    nperseg = min(len(hrv_signal) // 4, int(fs_target * 30))  # 30-second windows max

    try:
        f, Cxy = coherence(hrv_signal, resp_signal, fs_target, nperseg=nperseg)
    except Exception as e:
        warnings.warn(f"Coherence calculation failed: {e}")
        return 0.0, np.array([]), np.array([]), 0.0

    # Focus on RSA band (0.1-0.5 Hz)
    rsa_mask = (f >= 0.1) & (f <= 0.5)

    if not np.any(rsa_mask):
        warnings.warn("No frequencies in RSA band (0.1-0.5 Hz)")
        return 0.0, f, Cxy, 0.0

    rsa_coherence_values = Cxy[rsa_mask]
    mean_coherence = np.mean(rsa_coherence_values)

    # Find peak frequency in RSA band
    peak_idx = np.argmax(rsa_coherence_values)
    peak_freq = f[rsa_mask][peak_idx]

    return mean_coherence, f, Cxy, peak_freq


def check_lf_hf_band_overlap(resp_freq_hz: float) -> Dict:
    """
    FR-23: Check if respiratory frequency overlaps with LF/HF band boundaries.

    Args:
        resp_freq_hz: Respiratory frequency in Hz

    Returns:
        Dictionary with overlap analysis and annotations
    """
    LF_BAND = (0.04, 0.15)  # LF: 0.04-0.15 Hz
    HF_BAND = (0.15, 0.40)  # HF: 0.15-0.40 Hz

    resp_rate_bpm = resp_freq_hz * 60

    result = {
        "resp_freq_hz": resp_freq_hz,
        "resp_rate_bpm": resp_rate_bpm,
        "lf_overlap": False,
        "hf_overlap": False,
        "boundary_overlap": False,
        "annotation": None,
        "warning_msg": None,
    }

    # Check overlaps
    if LF_BAND[0] <= resp_freq_hz <= LF_BAND[1]:
        result["lf_overlap"] = True

    if HF_BAND[0] <= resp_freq_hz <= HF_BAND[1]:
        result["hf_overlap"] = True

    # Check critical boundary overlap (0.15 Hz = 9 bpm)
    boundary_tolerance = 0.02  # ±0.02 Hz tolerance
    if abs(resp_freq_hz - 0.15) <= boundary_tolerance:
        result["boundary_overlap"] = True
        result["annotation"] = "RSA_BOUNDARY_OVERLAP"
        result["warning_msg"] = (
            f"Respiratory rate ({resp_rate_bpm:.1f} bpm) overlaps LF/HF boundary. "
            "Frequency domain HRV metrics may be affected."
        )

    # Additional physiological range checks
    if resp_freq_hz < 0.08:  # < 5 bpm
        result["annotation"] = "RESP_RATE_TOO_LOW"
        result["warning_msg"] = (
            f"Unusually low respiratory rate: {resp_rate_bpm:.1f} bpm"
        )
    elif resp_freq_hz > 0.6:  # > 36 bpm
        result["annotation"] = "RESP_RATE_TOO_HIGH"
        result["warning_msg"] = (
            f"Unusually high respiratory rate: {resp_rate_bpm:.1f} bpm"
        )

    return result


def analyze_respiratory_metrics(bundle: DataBundle) -> Dict:
    """
    Main respiratory analysis function that integrates with existing pipeline.

    Args:
        bundle: DataBundle with ECG/RESP data and preprocessed RRI

    Returns:
        Dictionary with complete respiratory analysis results
    """
    results = {
        "method": None,
        "respiratory_rate_bpm": None,
        "respiratory_freq_hz": None,
        "rsa_coherence": None,
        "coherence_peak_freq": None,
        "lf_hf_analysis": None,
        "confidence": 0.0,
        "warnings": [],
    }

    resp_signal = None
    resp_rate_bpm = None
    resp_freq_hz = None
    confidence = 0.0

    try:
        # Priority 1: Use RESP channel if available
        if bundle.resp:
            resp_ts = bundle.resp[0]
            resp_signal = resp_ts.data
            fs_resp = resp_ts.fs

            # Calculate respiratory rate from RESP signal
            resp_rate_bpm = _calculate_resp_rate_from_signal(resp_signal, fs_resp)
            resp_freq_hz = resp_rate_bpm / 60.0

            results["method"] = "RESP_CHANNEL"
            confidence = 0.95  # High confidence for direct measurement

        # Priority 2: Use EDR-AM from ECG
        elif bundle.ecg:
            ecg_ts = bundle.ecg[0]
            resp_signal, resp_rate_bpm, confidence = estimate_respiratory_rate_edr_am(
                ecg_ts
            )
            resp_freq_hz = resp_rate_bpm / 60.0
            results["method"] = "EDR-AM"

        else:
            raise ValueError("No ECG or RESP signals available")

        # Store basic respiratory metrics
        results["respiratory_rate_bpm"] = resp_rate_bpm
        results["respiratory_freq_hz"] = resp_freq_hz
        results["confidence"] = confidence

        # FR-22: Compute RSA coherence
        try:
            rsa_coherence, freq_array, coherence_array, peak_freq = (
                compute_rsa_coherence(bundle)
            )
            results["rsa_coherence"] = rsa_coherence
            results["coherence_peak_freq"] = peak_freq
            results["_coherence_spectrum"] = {
                "frequencies": freq_array,
                "coherence": coherence_array,
            }
        except Exception as e:
            results["warnings"].append(f"RSA coherence calculation failed: {e}")
            results["rsa_coherence"] = None

        # FR-23: Check LF/HF band overlaps
        if resp_freq_hz is not None:
            lf_hf_analysis = check_lf_hf_band_overlap(resp_freq_hz)
            results["lf_hf_analysis"] = lf_hf_analysis

            if lf_hf_analysis["warning_msg"]:
                results["warnings"].append(lf_hf_analysis["warning_msg"])

    except Exception as e:
        error_msg = f"Respiratory analysis failed: {e}"
        results["warnings"].append(error_msg)
        warnings.warn(error_msg)

    return results


def _calculate_resp_rate_from_signal(resp_signal: np.ndarray, fs: float) -> float:
    """Calculate respiratory rate from signal using peak detection and FFT"""
    # Peak detection method
    min_distance = int(fs * 1.5)  # Minimum 1.5 seconds between breaths
    peaks, _ = find_peaks(resp_signal, distance=min_distance)

    rate_from_peaks = 0
    if len(peaks) > 2:
        intervals = np.diff(peaks) / fs
        rate_from_peaks = 60.0 / np.median(intervals)

    # FFT method
    f, psd = signal.welch(resp_signal, fs, nperseg=min(len(resp_signal), int(fs * 30)))
    resp_mask = (f >= 0.1) & (f <= 0.5)

    rate_from_fft = 0
    if np.any(resp_mask):
        peak_idx = np.argmax(psd[resp_mask])
        peak_freq = f[resp_mask][peak_idx]
        rate_from_fft = peak_freq * 60

    # Return FFT result if available, otherwise peaks
    if rate_from_fft > 6 and rate_from_fft < 40:  # Reasonable range
        return rate_from_fft
    elif rate_from_peaks > 6 and rate_from_peaks < 40:
        return rate_from_peaks
    else:
        return 15.0  # Safe default


def add_respiratory_metrics_to_bundle(bundle: DataBundle) -> DataBundle:
    """
    Add respiratory analysis to existing DataBundle (following your pattern).
    This function modifies the bundle in-place and returns it.

    Args:
        bundle: DataBundle to enhance with respiratory metrics

    Returns:
        Enhanced DataBundle with respiratory analysis in meta
    """
    try:
        respiratory_results = analyze_respiratory_metrics(bundle)

        # Add to bundle metadata (following your existing pattern)
        bundle.meta["respiratory_metrics"] = respiratory_results

        # Log warnings if any
        if respiratory_results["warnings"]:
            for warning in respiratory_results["warnings"]:
                warnings.warn(warning)

        return bundle

    except Exception as e:
        error_msg = f"Failed to add respiratory metrics: {e}"
        warnings.warn(error_msg)
        bundle.meta["respiratory_metrics"] = {"error": error_msg}
        return bundle


# Optional: Function to get respiratory summary for bundle.summary()
def get_respiratory_summary(bundle: DataBundle) -> Optional[Dict]:
    """
    Get respiratory metrics summary for inclusion in bundle.summary()

    Args:
        bundle: DataBundle with respiratory analysis

    Returns:
        Summary dictionary or None if no respiratory analysis
    """
    if "respiratory_metrics" not in bundle.meta:
        return None

    resp_data = bundle.meta["respiratory_metrics"]

    if "error" in resp_data:
        return {"status": "failed", "error": resp_data["error"]}

    summary = {
        "method": resp_data.get("method"),
        "rate_bpm": resp_data.get("respiratory_rate_bpm"),
        "rsa_coherence": resp_data.get("rsa_coherence"),
        "confidence": resp_data.get("confidence"),
        "has_warnings": len(resp_data.get("warnings", [])) > 0,
    }

    if resp_data.get("lf_hf_analysis"):
        lf_hf = resp_data["lf_hf_analysis"]
        summary["band_overlaps"] = {
            "lf": lf_hf["lf_overlap"],
            "hf": lf_hf["hf_overlap"],
            "boundary": lf_hf["boundary_overlap"],
        }

    return summary
