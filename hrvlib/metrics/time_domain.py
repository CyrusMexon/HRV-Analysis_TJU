import numpy as np
import warnings
from typing import List, Tuple, Union, Optional, Dict

# Import from existing modules to maintain consistency
from hrvlib.data_handler import DataBundle
from hrvlib.preprocessing import PreprocessingResult


class HRVTimeDomainAnalysis:
    """
    High-precision HRV time domain analysis tool integrated with DataBundle
    Implementation follows clinical and research standards (HRV Guidelines: Task Force of ESC/NASPE)
    Expects preprocessed data from the pipeline - no internal preprocessing
    """

    def __init__(
        self,
        preprocessed_rri: np.ndarray,
        preprocessing_result: Optional[PreprocessingResult] = None,
        analysis_window: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize HRV analyzer with preprocessed data

        Args:
            preprocessed_rri: Preprocessed RR intervals in milliseconds
            preprocessing_result: Results from preprocessing step
            analysis_window: (start_time, end_time) in seconds for analysis window
        """
        self.rr_ms = np.array(preprocessed_rri, dtype=float)
        self.preprocessing_result = preprocessing_result
        self.analysis_window = analysis_window

        if len(self.rr_ms) < 2:
            raise ValueError("At least 2 RR intervals needed for time domain analysis")

        # Apply analysis window if specified
        if self.analysis_window is not None:
            self.rr_ms = self._apply_analysis_window(self.rr_ms)

        # Validate data quality
        self._validate_data_quality()

    def _apply_analysis_window(self, rr_ms: np.ndarray) -> np.ndarray:
        """
        Apply analysis window to RRI data

        Args:
            rr_ms: RR intervals in milliseconds

        Returns:
            Windowed RR intervals
        """
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

    def _validate_data_quality(self) -> None:
        """
        Validate data quality using preprocessing results
        """
        if self.preprocessing_result is not None:
            quality_flags = self.preprocessing_result.quality_flags

            if quality_flags and quality_flags.get("poor_signal_quality", False):
                warnings.warn(
                    "Poor signal quality detected. Results may be unreliable."
                )

            if quality_flags and quality_flags.get("excessive_artifacts", False):
                warnings.warn(
                    "Excessive artifacts detected (>5%). Consider data quality."
                )

            artifact_percentage = self.preprocessing_result.stats.get(
                "artifact_percentage", 0
            )
            if artifact_percentage > 10:
                warnings.warn(f"High artifact percentage: {artifact_percentage:.1f}%")

    def sdnn(self) -> float:
        """
        Calculate SDNN: Standard deviation of RR intervals
        Reflects overall HRV
        """
        return float(np.std(self.rr_ms, ddof=1))

    def rmssd(self) -> float:
        """
        Calculate RMSSD: Root mean square of successive differences
        Reflects short-term HRV and parasympathetic activity
        """
        if len(self.rr_ms) < 2:
            return 0.0

        diff = np.diff(self.rr_ms)
        squared_diff = diff**2
        mean_squared_diff = np.mean(squared_diff)
        return float(np.sqrt(mean_squared_diff))

    def pnn50(self) -> float:
        """
        Calculate pNN50: Percentage of successive RR interval differences > 50ms
        Reflects parasympathetic activity
        """
        if len(self.rr_ms) < 2:
            return 0.0

        diff = np.abs(np.diff(self.rr_ms))
        nn50 = np.sum(diff > 50)
        total = len(diff)
        return (nn50 / total) * 100 if total > 0 else 0.0

    def pnn20(self) -> float:
        """
        Calculate pNN20: Percentage of successive RR interval differences > 20ms
        More sensitive measure of parasympathetic activity
        """
        if len(self.rr_ms) < 2:
            return 0.0

        diff = np.abs(np.diff(self.rr_ms))
        nn20 = np.sum(diff > 20)
        total = len(diff)
        return (nn20 / total) * 100 if total > 0 else 0.0

    def nn50(self) -> int:
        """Count of successive RR interval differences > 50ms"""
        if len(self.rr_ms) < 2:
            return 0

        diff = np.abs(np.diff(self.rr_ms))
        return int(np.sum(diff > 50))

    def nn20(self) -> int:
        """Count of successive RR interval differences > 20ms"""
        if len(self.rr_ms) < 2:
            return 0

        diff = np.abs(np.diff(self.rr_ms))
        return int(np.sum(diff > 20))

    def mean_rr(self) -> float:
        """Mean RR interval in milliseconds"""
        return float(np.mean(self.rr_ms))

    def median_rr(self) -> float:
        """Median RR interval in milliseconds"""
        return float(np.median(self.rr_ms))

    def mean_hr(self) -> float:
        """Mean heart rate in beats per minute"""
        mean_rr_s = np.mean(self.rr_ms) / 1000.0
        return 60.0 / mean_rr_s if mean_rr_s > 0 else 0.0

    def std_hr(self) -> float:
        """Standard deviation of heart rate"""
        rr_s = self.rr_ms / 1000.0
        hr = 60.0 / rr_s
        return float(np.std(hr, ddof=1))

    def cvnn(self) -> float:
        """
        Coefficient of variation of RR intervals (CVNN)
        CVNN = SDNN / Mean RR
        """
        mean_rr = self.mean_rr()
        if mean_rr > 0:
            return self.sdnn() / mean_rr
        return 0.0

    def cvsd(self) -> float:
        """
        Coefficient of variation of successive differences
        CVSD = RMSSD / Mean RR
        """
        mean_rr = self.mean_rr()
        if mean_rr > 0:
            return self.rmssd() / mean_rr
        return 0.0

    def hrv_triangular_index(self) -> float:
        """
        HRV Triangular Index: Number of RR intervals / height of histogram
        Geometric measure of HRV
        """
        if len(self.rr_ms) == 0:
            return 0.0

        # Use 7.8125 ms bin width (1/128 s) as per HRV standards
        bin_width = 7.8125
        bins = np.arange(np.min(self.rr_ms), np.max(self.rr_ms) + bin_width, bin_width)

        hist, _ = np.histogram(self.rr_ms, bins=bins)

        if len(hist) == 0 or np.max(hist) == 0:
            return 0.0

        return len(self.rr_ms) / np.max(hist)

    def tinn(self) -> float:
        """
        Triangular Interpolation of NN interval Histogram (TINN)
        Baseline width of triangular interpolation
        """
        if len(self.rr_ms) < 3:
            return 0.0

        # Use 7.8125 ms bin width
        bin_width = 7.8125
        bins = np.arange(np.min(self.rr_ms), np.max(self.rr_ms) + bin_width, bin_width)

        hist, bin_edges = np.histogram(self.rr_ms, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if len(hist) == 0:
            return 0.0

        # Find histogram peak
        max_idx = np.argmax(hist)
        max_value = hist[max_idx]

        if max_value == 0:
            return 0.0

        # Find left and right boundaries where histogram drops to baseline
        # Use triangular approximation method
        left_idx = 0
        right_idx = len(hist) - 1

        # Find baseline level (typically minimum non-zero value)
        baseline = np.min(hist[hist > 0]) if np.any(hist > 0) else 0

        # Search left boundary
        for i in range(max_idx, -1, -1):
            if hist[i] <= baseline:
                left_idx = i
                break

        # Search right boundary
        for i in range(max_idx, len(hist)):
            if hist[i] <= baseline:
                right_idx = i
                break

        # Calculate TINN as baseline width
        return (
            bin_centers[right_idx] - bin_centers[left_idx]
            if right_idx > left_idx
            else 0.0
        )

    def full_analysis(self) -> Dict[str, Union[float, int, Dict]]:
        """Execute complete time domain analysis"""
        results = {
            # Basic time domain metrics
            "sdnn": self.sdnn(),
            "rmssd": self.rmssd(),
            "pnn50": self.pnn50(),
            "pnn20": self.pnn20(),
            "nn50": self.nn50(),
            "nn20": self.nn20(),
            # Central tendency measures
            "mean_rr": self.mean_rr(),
            "median_rr": self.median_rr(),
            "mean_hr": self.mean_hr(),
            "std_hr": self.std_hr(),
            # Coefficient of variation measures
            "cvnn": self.cvnn(),
            "cvsd": self.cvsd(),
            # Geometric measures
            "hrv_triangular_index": self.hrv_triangular_index(),
            "tinn": self.tinn(),
            # Analysis metadata
            "analysis_info": {
                "total_intervals": len(self.rr_ms),
                "analysis_duration_s": np.sum(self.rr_ms) / 1000.0,
                "preprocessing_applied": self.preprocessing_result is not None,
                "analysis_window": self.analysis_window,
            },
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

    def get_quality_assessment(self) -> Dict[str, Union[str, bool, float]]:
        """
        Get data quality assessment for the analysis

        Returns:
            Dictionary with quality metrics and recommendations
        """
        assessment = {
            "overall_quality": "unknown",
            "data_length_adequate": len(self.rr_ms)
            >= 50,  # At least 50 intervals for reliable metrics
            "duration_adequate_s": np.sum(self.rr_ms) / 1000.0,
            "recommendations": [],
        }

        # Duration check
        duration_s = np.sum(self.rr_ms) / 1000.0
        if duration_s < 120:  # Less than 2 minutes
            assessment["recommendations"].append(
                "Recording duration < 2 minutes may limit metric reliability"
            )
        elif duration_s < 300:  # Less than 5 minutes
            assessment["recommendations"].append(
                "Consider longer recordings (≥5 minutes) for more stable metrics"
            )

        # Preprocessing quality assessment
        if self.preprocessing_result is not None:
            artifact_pct = self.preprocessing_result.stats.get("artifact_percentage", 0)
            quality_flags = self.preprocessing_result.quality_flags

            assessment["artifact_percentage"] = artifact_pct
            assessment["noise_detected"] = (
                len(self.preprocessing_result.noise_segments) > 0
            )

            if quality_flags:
                if quality_flags.get("poor_signal_quality", False):
                    assessment["overall_quality"] = "poor"
                    assessment["recommendations"].append(
                        "Poor signal quality detected - consider data cleaning"
                    )
                elif quality_flags.get("excessive_artifacts", False):
                    assessment["overall_quality"] = "fair"
                    assessment["recommendations"].append(
                        "High artifact rate - verify preprocessing settings"
                    )
                else:
                    assessment["overall_quality"] = "good"

            if artifact_pct > 10:
                assessment["recommendations"].append(
                    f"High artifact percentage ({artifact_pct:.1f}%) - results may be unreliable"
                )

        return assessment

    def compare_with_norms(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Compare calculated metrics with published normative values
        Based on Task Force guidelines and population studies

        Returns:
            Dictionary with metric comparisons and interpretations
        """
        # Normative ranges for healthy adults (approximate values from literature)
        norms = {
            "sdnn": {"low": 30, "normal_low": 50, "normal_high": 100, "high": 150},
            "rmssd": {"low": 15, "normal_low": 25, "normal_high": 50, "high": 80},
            "pnn50": {"low": 2, "normal_low": 5, "normal_high": 25, "high": 40},
            "mean_hr": {"low": 50, "normal_low": 60, "normal_high": 100, "high": 110},
            "hrv_triangular_index": {
                "low": 10,
                "normal_low": 15,
                "normal_high": 35,
                "high": 50,
            },
        }

        comparisons = {}

        for metric in ["sdnn", "rmssd", "pnn50", "mean_hr", "hrv_triangular_index"]:
            if hasattr(self, metric):
                value = getattr(self, metric)()
                norm = norms.get(metric, {})

                if value < norm.get("low", 0):
                    interpretation = "very_low"
                elif value < norm.get("normal_low", 0):
                    interpretation = "low"
                elif value <= norm.get("normal_high", float("inf")):
                    interpretation = "normal"
                elif value <= norm.get("high", float("inf")):
                    interpretation = "high"
                else:
                    interpretation = "very_high"

                comparisons[metric] = {
                    "value": value,
                    "interpretation": interpretation,
                    "normal_range": f"{norm.get('normal_low', '?')}-{norm.get('normal_high', '?')}",
                }

        return comparisons


def validate_rr_intervals_for_time_domain(
    rr_intervals: List[Union[int, float]],
) -> Tuple[bool, List[str]]:
    """
    Validate RR intervals specifically for time domain analysis

    Args:
        rr_intervals: List of RR intervals

    Returns:
        Tuple of (is_valid, error_messages)
    """
    from hrvlib.preprocessing import validate_rri_data

    # Use base validation
    is_valid, errors = validate_rri_data(rr_intervals)

    # Additional time domain specific checks
    if len(rr_intervals) < 50:
        errors.append(
            "Time domain analysis requires at least 50 RR intervals for reliable results"
        )

    if len(rr_intervals) > 0:
        duration_s = sum(rr_intervals) / 1000.0
        if duration_s < 120:  # Less than 2 minutes
            errors.append(
                "Recording duration < 2 minutes may produce unreliable time domain metrics"
            )

    return len(errors) == 0, errors
