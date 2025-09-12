import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy import interpolate, signal
from scipy.signal import find_peaks


@dataclass
class PreprocessingResult:
    """Results from artifact detection and correction"""

    original_rri: np.ndarray
    corrected_rri: np.ndarray
    artifact_indices: List[int]
    artifact_types: List[str]  # 'missed', 'extra', 'ectopic'
    interpolation_indices: List[int]
    correction_method: str
    stats: Dict
    correction_details: Optional[Dict[str, List[int]]] = None
    noise_segments: Optional[List[Tuple[int, int]]] = None  # New: noisy segments
    quality_flags: Optional[Dict[str, bool]] = None  # New: quality indicators

    def __post_init__(self):
        """Initialize correction_details if not provided"""
        if self.correction_details is None:
            self.correction_details = {
                "extra_beats_removed": [],
                "intervals_interpolated": [],
            }
        if self.noise_segments is None:
            self.noise_segments = []
        if self.quality_flags is None:
            self.quality_flags = {
                "high_noise": False,
                "excessive_artifacts": False,
                "poor_signal_quality": False,
                "irregular_rhythm": False,
            }


def detect_noise_segments(
    rri_ms: np.ndarray,
    noise_threshold: float = 2.0,
    window_size: int = 10,
    min_segment_length: int = 3,
) -> List[Tuple[int, int]]:
    """
    FR-8: Detect noisy or incomplete data segments using statistical methods.

    Args:
        rri_ms: RR intervals in milliseconds
        noise_threshold: Z-score threshold for noise detection
        window_size: Size of sliding window for noise assessment
        min_segment_length: Minimum length of noise segment to report

    Returns:
        List of (start_idx, end_idx) tuples for noisy segments
    """
    if len(rri_ms) < window_size:
        return []

    noise_segments = []

    # Calculate rolling statistics
    half_window = window_size // 2
    noise_scores = np.zeros(len(rri_ms))

    for i in range(len(rri_ms)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(rri_ms), i + half_window + 1)
        window_data = rri_ms[start_idx:end_idx]

        if len(window_data) > 2:
            # Calculate local variability metrics
            local_std = np.std(window_data)
            local_median = np.median(window_data)

            # Detect excessive variability
            cv = local_std / (local_median + 1e-10)  # Coefficient of variation

            # Detect sudden jumps
            if i > 0:
                jump_ratio = abs(rri_ms[i] - rri_ms[i - 1]) / (local_median + 1e-10)
                noise_scores[i] = max(cv * 10, jump_ratio * 5)
            else:
                noise_scores[i] = cv * 10

    # Identify noisy regions using threshold
    noisy_mask = noise_scores > noise_threshold

    # Find contiguous noisy segments
    if np.any(noisy_mask):
        diff_mask = np.diff(np.concatenate([[False], noisy_mask, [False]]).astype(int))
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]

        for start, end in zip(starts, ends):
            if end - start >= min_segment_length:
                noise_segments.append((int(start), int(end)))

    return noise_segments


def assess_signal_quality(
    rri_ms: np.ndarray,
    artifact_percentage: float,
    noise_segments: List[Tuple[int, int]],
) -> Dict[str, bool]:
    """
    Assess overall signal quality and set quality flags.

    Args:
        rri_ms: RR intervals in milliseconds
        artifact_percentage: Percentage of artifacts detected
        noise_segments: List of noisy segments

    Returns:
        Dictionary of quality flags
    """
    flags = {
        "high_noise": False,
        "excessive_artifacts": False,
        "poor_signal_quality": False,
        "irregular_rhythm": False,
    }

    # Check noise level
    total_noise_samples = sum(end - start for start, end in noise_segments)
    noise_percentage = (
        (total_noise_samples / len(rri_ms)) * 100 if len(rri_ms) > 0 else 0
    )

    flags["high_noise"] = noise_percentage > 15.0  # >15% noisy segments
    flags["excessive_artifacts"] = artifact_percentage > 5.0  # >5% artifacts

    # Assess rhythm regularity using coefficient of variation
    if len(rri_ms) > 0:
        cv = np.std(rri_ms) / np.mean(rri_ms)
        flags["irregular_rhythm"] = (
            cv > 0.45
        )  # High variability might indicate AF/irregular rhythm

    # Overall quality assessment
    flags["poor_signal_quality"] = (
        flags["high_noise"] or flags["excessive_artifacts"] or noise_percentage > 25.0
    )

    return flags


def detect_artifacts(
    rri_ms: np.ndarray,
    threshold_low: float = 300.0,
    threshold_high: float = 2000.0,
    ectopic_threshold: float = 0.3,
) -> Tuple[List[int], List[str]]:
    """
    Detect artifacts in RR intervals using threshold-based methods.

    Args:
        rri_ms: RR intervals in milliseconds
        threshold_low: Lower threshold for valid RR intervals (ms)
        threshold_high: Upper threshold for valid RR intervals (ms)
        ectopic_threshold: Relative threshold for ectopic beat detection

    Returns:
        Tuple of (artifact_indices, artifact_types)
    """
    if len(rri_ms) < 3:
        return [], []

    artifact_indices = []
    artifact_types = []

    # Convert to numpy array for easier processing
    rri = np.asarray(rri_ms, dtype=float)

    # 1. Missed beats (very long intervals)
    missed_mask = rri > threshold_high
    missed_indices = np.where(missed_mask)[0].tolist()
    artifact_indices.extend(missed_indices)
    artifact_types.extend(["missed"] * len(missed_indices))

    # 2. Extra beats (very short intervals)
    extra_mask = rri < threshold_low
    extra_indices = np.where(extra_mask)[0].tolist()
    artifact_indices.extend(extra_indices)
    artifact_types.extend(["extra"] * len(extra_indices))

    # 3. Ectopic beats (sudden changes relative to local mean)
    for i in range(1, len(rri) - 1):
        if i in artifact_indices:  # Skip already detected artifacts
            continue

        # Local window analysis
        window_size = min(5, len(rri) // 4)
        start_idx = max(0, i - window_size)
        end_idx = min(len(rri), i + window_size + 1)

        local_rri = rri[start_idx:end_idx]
        local_rri = local_rri[~np.isin(np.arange(start_idx, end_idx), artifact_indices)]

        if len(local_rri) > 2:
            local_mean = np.mean(local_rri)
            relative_change = abs(rri[i] - local_mean) / local_mean

            if relative_change > ectopic_threshold:
                artifact_indices.append(i)
                artifact_types.append("ectopic")

    return artifact_indices, artifact_types


def correct_extra_beats(
    rri_ms: np.ndarray, extra_indices: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """
    Correct extra beats by removing them and merging adjacent RR intervals (Kubios method).

    Args:
        rri_ms: RR intervals in milliseconds
        extra_indices: Indices of detected extra beats

    Returns:
        Tuple of (corrected_rri, original_extra_indices_corrected)
    """
    if not extra_indices:
        return rri_ms.copy(), []

    rri = rri_ms.copy()
    corrected_indices = []

    # Sort indices in descending order to avoid index shifting issues
    extra_indices_sorted = sorted(set(extra_indices), reverse=True)

    for idx in extra_indices_sorted:
        if 0 <= idx < len(rri):
            # For extra beats, we merge the current RR interval with the next one
            if idx < len(rri) - 1:
                # Add current interval to next interval (merge them)
                rri[idx + 1] = rri[idx] + rri[idx + 1]
                # Remove the current (extra beat) interval
                rri = np.delete(rri, idx)
                corrected_indices.append(idx)
            elif idx == len(rri) - 1 and idx > 0:
                # If it's the last interval, merge with previous
                rri[idx - 1] = rri[idx - 1] + rri[idx]
                rri = np.delete(rri, idx)
                corrected_indices.append(idx)

    return rri, sorted(corrected_indices)


def cubic_spline_interpolation(
    rri_ms: np.ndarray, artifact_indices: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """
    Correct artifacts using cubic spline interpolation.

    Args:
        rri_ms: Original RR intervals in milliseconds
        artifact_indices: Indices of detected artifacts

    Returns:
        Tuple of (corrected_rri, interpolation_indices)
    """
    if not artifact_indices or len(rri_ms) < 4:
        return rri_ms.copy(), []

    rri = rri_ms.copy()
    interpolation_indices = []

    # Sort artifact indices
    artifact_indices = sorted(set(artifact_indices))

    # Create time vector
    time_cumsum = np.cumsum(rri) / 1000.0  # Convert to seconds
    time_vector = np.concatenate([[0], time_cumsum[:-1]])

    # Create mask for valid points
    valid_mask = np.ones(len(rri), dtype=bool)
    valid_mask[artifact_indices] = False

    if np.sum(valid_mask) < 4:  # Need at least 4 points for cubic spline
        warnings.warn("Too few valid points for cubic spline interpolation")
        return rri, []

    # Get valid time points and RR intervals
    valid_times = time_vector[valid_mask]
    valid_rri = rri[valid_mask]

    try:
        # Create cubic spline interpolator
        cs = interpolate.CubicSpline(valid_times, valid_rri, bc_type="natural")

        # Interpolate at artifact positions
        for idx in artifact_indices:
            if 0 <= idx < len(time_vector):
                rri[idx] = cs(time_vector[idx])
                interpolation_indices.append(idx)

        # Ensure interpolated values are reasonable
        rri = np.clip(rri, 200, 3000)  # Physiological limits

    except Exception as e:
        warnings.warn(f"Cubic spline interpolation failed: {e}")
        return rri_ms.copy(), []

    return rri, interpolation_indices


def preprocess_rri(
    rri_ms: List[float],
    threshold_low: float = 300.0,
    threshold_high: float = 2000.0,
    ectopic_threshold: float = 0.3,
    correction_method: str = "cubic_spline",
    noise_detection: bool = True,
) -> PreprocessingResult:
    """
    Complete preprocessing pipeline for RR intervals with proper extra beat handling and noise detection.

    Args:
        rri_ms: RR intervals in milliseconds
        threshold_low: Lower threshold for valid RR intervals
        threshold_high: Upper threshold for valid RR intervals
        ectopic_threshold: Relative threshold for ectopic detection
        correction_method: Method for artifact correction
        noise_detection: Whether to perform noise detection (FR-8)

    Returns:
        PreprocessingResult object with all preprocessing information
    """
    if not rri_ms:
        raise ValueError("No RR intervals provided for preprocessing")

    # Validate input data
    rri_array = np.asarray(rri_ms, dtype=float)

    # Remove NaN and infinite values
    valid_mask = np.isfinite(rri_array)
    if not np.all(valid_mask):
        warnings.warn(f"Removed {np.sum(~valid_mask)} invalid RR intervals (NaN/inf)")
        rri_array = rri_array[valid_mask]

    if len(rri_array) == 0:
        raise ValueError("No valid RR intervals after removing NaN/inf values")

    original_rri = rri_array.copy()

    # Step 1: Noise detection (FR-8)
    noise_segments = []
    if noise_detection:
        noise_segments = detect_noise_segments(rri_array)
        if noise_segments:
            warnings.warn(f"Detected {len(noise_segments)} noisy segments in the data")

    # Step 2: Detect artifacts
    artifact_indices, artifact_types = detect_artifacts(
        rri_array, threshold_low, threshold_high, ectopic_threshold
    )

    # Step 3: Separate artifact types for different handling
    extra_indices = [
        i for i, t in zip(artifact_indices, artifact_types) if t == "extra"
    ]
    other_indices = [
        i for i, t in zip(artifact_indices, artifact_types) if t != "extra"
    ]
    other_types = [t for t in artifact_types if t != "extra"]

    # Step 4: Handle extra beats first (removal/merging)
    corrected_rri = rri_array.copy()
    extra_corrected_indices = []

    if extra_indices:
        corrected_rri, extra_corrected_indices = correct_extra_beats(
            corrected_rri, extra_indices
        )

        # Adjust indices for remaining artifacts after extra beat removal
        adjusted_other_indices = []
        for idx in other_indices:
            # Count how many extra beats were removed before this index
            removed_before = sum(
                1 for e_idx in sorted(extra_corrected_indices) if e_idx <= idx
            )
            adjusted_idx = idx - removed_before
            if 0 <= adjusted_idx < len(corrected_rri):
                adjusted_other_indices.append(adjusted_idx)

        other_indices = adjusted_other_indices

    # Step 5: Handle missed beats and ectopic beats (interpolation)
    interpolation_indices = []
    if other_indices and correction_method == "cubic_spline":
        corrected_rri, interpolation_indices = cubic_spline_interpolation(
            corrected_rri, other_indices
        )
    elif other_indices and correction_method != "cubic_spline":
        warnings.warn(f"Unknown correction method: {correction_method}")

    # Step 6: Calculate statistics
    total_artifacts_detected = len(artifact_indices)
    total_artifacts_corrected = len(extra_corrected_indices) + len(
        interpolation_indices
    )
    artifact_percentage = total_artifacts_detected / len(original_rri) * 100

    # Step 7: Assess signal quality
    quality_flags = assess_signal_quality(
        original_rri, artifact_percentage, noise_segments
    )

    stats = {
        "original_count": len(original_rri),
        "final_count": len(corrected_rri),
        "artifacts_detected": total_artifacts_detected,
        "artifacts_corrected": total_artifacts_corrected,
        "extra_beats_removed": len(extra_corrected_indices),
        "intervals_interpolated": len(interpolation_indices),
        "artifact_percentage": artifact_percentage,
        "noise_segments_count": len(noise_segments),
        "noise_percentage": (
            sum(end - start for start, end in noise_segments) / len(original_rri)
        )
        * 100,
        "original_mean": float(np.mean(original_rri)),
        "corrected_mean": float(np.mean(corrected_rri)),
        "original_std": float(np.std(original_rri)),
        "corrected_std": float(np.std(corrected_rri)),
    }

    # Combine all correction information for the result
    all_corrected_indices = extra_corrected_indices + interpolation_indices
    correction_details = {
        "extra_beats_removed": extra_corrected_indices,
        "intervals_interpolated": interpolation_indices,
    }

    return PreprocessingResult(
        original_rri=original_rri,
        corrected_rri=corrected_rri,
        artifact_indices=artifact_indices,
        artifact_types=artifact_types,
        interpolation_indices=all_corrected_indices,  # For backward compatibility
        correction_method=correction_method,
        stats=stats,
        correction_details=correction_details,
        noise_segments=noise_segments,
        quality_flags=quality_flags,
    )


def validate_rri_data(rri_ms: List[float]) -> Tuple[bool, List[str]]:
    """
    Validate RR interval data.

    Args:
        rri_ms: RR intervals in milliseconds

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not rri_ms:
        errors.append("No RR intervals provided")
        return False, errors

    try:
        rri_array = np.asarray(rri_ms, dtype=float)
    except (ValueError, TypeError):
        errors.append("RR intervals must be numeric values")
        return False, errors

    # Check for NaN or infinite values
    if not np.all(np.isfinite(rri_array)):
        nan_count = np.sum(np.isnan(rri_array))
        inf_count = np.sum(np.isinf(rri_array))
        errors.append(f"Found {nan_count} NaN and {inf_count} infinite values")

    # Check for negative values
    negative_count = np.sum(rri_array <= 0)
    if negative_count > 0:
        errors.append(f"Found {negative_count} non-positive RR intervals")

    # Check for extremely unrealistic values
    too_short = np.sum(rri_array < 100)  # < 100ms
    too_long = np.sum(rri_array > 5000)  # > 5s

    if too_short > 0:
        errors.append(f"Found {too_short} extremely short RR intervals (< 100ms)")
    if too_long > 0:
        errors.append(f"Found {too_long} extremely long RR intervals (> 5s)")

    # Check minimum data length
    if len(rri_array) < 10:
        errors.append(
            f"Too few RR intervals for analysis: {len(rri_array)} (minimum: 10)"
        )

    return len(errors) == 0, errors
