import numpy as np
import scipy.stats
import scipy.spatial
import scipy.special
from typing import Tuple, Optional, List, Dict, Union
import warnings

# Import from existing modules to maintain consistency
from hrvlib.preprocessing import PreprocessingResult


class NonlinearHRVAnalysis:
    """
    Nonlinear HRV analysis toolkit
    Implements Poincaré analysis, Sample Entropy, Multiscale Entropy, DFA and RQA
    Expects preprocessed data from the pipeline - no internal preprocessing
    Meets research and enterprise-level accuracy requirements
    """

    def __init__(
        self,
        preprocessed_rri: np.ndarray,
        preprocessing_result: Optional[PreprocessingResult] = None,
        analysis_window: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize nonlinear HRV analyzer with preprocessed data

        Args:
            preprocessed_rri: Preprocessed RR intervals in milliseconds
            preprocessing_result: Results from preprocessing step
            analysis_window: (start_time, end_time) in seconds for analysis window
        """
        self.rr_ms = np.array(preprocessed_rri, dtype=float)
        self.preprocessing_result = preprocessing_result
        self.analysis_window = analysis_window

        if len(self.rr_ms) < 10:
            raise ValueError("At least 10 RR intervals needed for nonlinear analysis")

        # Apply analysis window if specified
        if self.analysis_window is not None:
            self.rr_ms = self._apply_analysis_window(self.rr_ms)

        # Validate data quality
        self._validate_data_quality()

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

    def _validate_data_quality(self) -> None:
        """Validate data quality using preprocessing results"""
        if self.preprocessing_result is not None:
            quality_flags = self.preprocessing_result.quality_flags

            if quality_flags and quality_flags.get("poor_signal_quality", False):
                warnings.warn(
                    "Poor signal quality detected. Nonlinear results may be unreliable."
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

    def poincare_analysis(self) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Perform Poincaré analysis and calculate SD1 and SD2 metrics

        Returns:
            Tuple of (sd1, sd2, sd1_sd2_ratio, additional_metrics)
        """
        if len(self.rr_ms) < 2:
            return 0.0, 0.0, 0.0, {}

        # Create Poincaré plot data points
        x = self.rr_ms[:-1]  # RR(i)
        y = self.rr_ms[1:]  # RR(i+1)

        # Calculate successive differences
        diff = y - x

        # Calculate SD1 (standard deviation perpendicular to line of identity)
        # SD1 = sqrt(var(RR(i+1) - RR(i)) / 2)
        if len(diff) == 1:
            sd1 = 0.0
        else:
            sd1 = np.sqrt(np.var(diff, ddof=1) / 2)

        # Calculate SD2 (standard deviation along line of identity)
        # SD2 = sqrt(2 * SDNN² - 0.5 * SD1²)
        if len(self.rr_ms) == 1:
            sd2 = 0.0
        else:
            sdnn = np.std(self.rr_ms, ddof=1)
            sd2_squared = 2 * sdnn**2 - 0.5 * sd1**2
            sd2 = np.sqrt(max(sd2_squared, 0))  # Ensure non-negative

        # Calculate SD1/SD2 ratio
        sd1_sd2_ratio = sd1 / sd2 if sd2 > 0 else float("nan")

        # Additional Poincaré metrics - only calculate if we have meaningful variation
        if sd1 == 0.0 and sd2 == 0.0:
            # For constant data, return empty additional metrics
            additional_metrics = {}
        else:
            additional_metrics = {
                "ellipse_area": np.pi * sd1 * sd2,  # Area of Poincaré ellipse
                "csi": (
                    sd2 / sd1 if sd1 > 0 else float("inf")
                ),  # Cardiac Sympathetic Index
                "cvi": (
                    np.log10(sd1 * sd2) if sd1 > 0 and sd2 > 0 else float("nan")
                ),  # Cardiac Vagal Index
                "modified_csi": (
                    (sd2**2) / sd1 if sd1 > 0 else float("inf")
                ),  # Modified CSI
            }

        return sd1, sd2, sd1_sd2_ratio, additional_metrics

    def sample_entropy(
        self, m: int = 2, r: float = 0.2, normalize: bool = True
    ) -> float:
        """
        Calculate Sample Entropy of RR interval series

        Args:
            m: Template length (typically 2)
            r: Tolerance coefficient (typically 0.2)
            normalize: Whether to normalize data (use relative tolerance)

        Returns:
            Sample entropy value
        """
        if len(self.rr_ms) < 100:
            raise ValueError(
                "Sample entropy calculation requires at least 100 data points"
            )
        if m < 1:
            raise ValueError("Template length m must be at least 1")
        if r <= 0:
            raise ValueError("Tolerance r must be positive")

        rr_clean = self.rr_ms.copy()

        # Normalize data if requested
        if normalize:
            std = np.std(rr_clean, ddof=1)
            if std < 1e-10:  # Prevent division by zero
                return 0.0
            rr_norm = (rr_clean - np.mean(rr_clean)) / std
            tolerance = r  # r is now relative to std = 1
        else:
            rr_norm = rr_clean.copy()
            tolerance = r * np.std(rr_clean, ddof=1)  # Absolute tolerance

        n = len(rr_norm)

        def _maxdist(xi, xj, m):
            """Calculate maximum distance between templates"""
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            """Calculate phi(m) for sample entropy"""
            patterns = np.array([rr_norm[i : i + m] for i in range(n - m + 1)])
            C = np.zeros(n - m + 1)

            for i in range(n - m + 1):
                template = patterns[i]
                distances = np.array(
                    [
                        _maxdist(template, patterns[j], m)
                        for j in range(n - m + 1)
                        if i != j
                    ]
                )
                C[i] = np.sum(distances <= tolerance)

            phi = np.sum(C) / ((n - m + 1) * (n - m))
            return phi

        # Calculate sample entropy
        phi_m = _phi(m)
        phi_m_plus_1 = _phi(m + 1)

        # Handle edge cases
        if phi_m == 0 or phi_m_plus_1 == 0:
            return float("inf") if phi_m == 0 else 0.0

        return -np.log(phi_m_plus_1 / phi_m)

    def multiscale_entropy(
        self, scale_max: int = 10, m: int = 2, r: float = 0.15
    ) -> np.ndarray:
        """
        Calculate Multiscale Sample Entropy

        Args:
            scale_max: Maximum scale factor
            m: Template length
            r: Tolerance coefficient

        Returns:
            Array of sample entropy values at each scale
        """
        n = len(self.rr_ms)
        # Fixed requirement calculation - be more conservative
        min_required = scale_max * 50  # Ensure enough data for largest scale
        if n < min_required:
            raise ValueError(
                f"Multiscale entropy requires at least {min_required} data points for scale_max={scale_max}"
            )

        mse = np.zeros(scale_max)
        rr_clean = self.rr_ms.copy()

        for scale in range(1, scale_max + 1):
            # Create coarse-grained series
            coarse_grained = []
            for i in range(0, n - scale + 1, scale):
                coarse_grained.append(np.mean(rr_clean[i : i + scale]))

            coarse_grained = np.array(coarse_grained)

            if len(coarse_grained) < 100:  # Need at least 100 for sample entropy
                mse[scale - 1] = np.nan
                continue

            try:
                # Create temporary analyzer for sample entropy calculation
                temp_analyzer = NonlinearHRVAnalysis(
                    preprocessed_rri=coarse_grained, preprocessing_result=None
                )
                sampen = temp_analyzer.sample_entropy(m, r, normalize=True)
                mse[scale - 1] = sampen
            except Exception as e:
                warnings.warn(f"Scale {scale} calculation failed: {str(e)}")
                mse[scale - 1] = np.nan

        return mse

    def detrended_fluctuation_analysis(
        self,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Calculate Detrended Fluctuation Analysis (DFA) metrics

        Returns:
            Tuple of (alpha1, alpha2, box_sizes, fluctuations)
        """
        if len(self.rr_ms) < 100:
            raise ValueError("DFA calculation requires at least 100 data points")

        rr_clean = self.rr_ms.copy()
        n = len(rr_clean)

        # Step 1: Integrate the signal (cumulative sum of deviations from mean)
        y = np.cumsum(rr_clean - np.mean(rr_clean))

        # Step 2: Create box sizes (logarithmic scale)
        min_box_size = 4
        max_box_size = min(n // 4, 64)  # Limit maximum box size
        box_sizes = np.unique(
            np.logspace(
                np.log10(min_box_size), np.log10(max_box_size), num=12, dtype=int
            )
        )

        fluctuations = []

        # Step 3: Calculate fluctuations for each box size
        for box_size in box_sizes:
            # Divide into non-overlapping boxes
            n_boxes = n // box_size

            if n_boxes < 1:
                continue

            f_squared = 0

            for i in range(n_boxes):
                # Extract segment
                start_idx = i * box_size
                end_idx = (i + 1) * box_size
                segment = y[start_idx:end_idx]

                # Linear detrending
                x = np.arange(box_size)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)

                # Calculate fluctuation
                detrended = segment - trend
                f_squared += np.sum(detrended**2)

            # Average fluctuation
            fluctuation = np.sqrt(f_squared / (n_boxes * box_size))
            fluctuations.append(fluctuation)

        fluctuations = np.array(fluctuations)

        # Step 4: Calculate scaling exponents
        if len(fluctuations) < 4:
            return np.nan, np.nan, box_sizes, fluctuations

        log_box_sizes = np.log(box_sizes[: len(fluctuations)])
        log_fluctuations = np.log(fluctuations)

        # Short-term scaling exponent (typically 4-11 beats)
        short_term_mask = (box_sizes[: len(fluctuations)] >= 4) & (
            box_sizes[: len(fluctuations)] <= 11
        )
        # Long-term scaling exponent (typically > 11 beats)
        long_term_mask = box_sizes[: len(fluctuations)] > 11

        alpha1 = np.nan
        alpha2 = np.nan

        if np.sum(short_term_mask) >= 2:
            try:
                alpha1, _ = np.polyfit(
                    log_box_sizes[short_term_mask], log_fluctuations[short_term_mask], 1
                )
            except:
                alpha1 = np.nan

        if np.sum(long_term_mask) >= 2:
            try:
                alpha2, _ = np.polyfit(
                    log_box_sizes[long_term_mask], log_fluctuations[long_term_mask], 1
                )
            except:
                alpha2 = np.nan

        return alpha1, alpha2, box_sizes[: len(fluctuations)], fluctuations

    def recurrence_quantification_analysis(
        self,
        threshold: float = 0.1,
        embedding_dim: int = 3,
        delay: int = 1,
        min_line_length: int = 2,
    ) -> Dict[str, Union[float, int]]:
        """
        Recurrence Quantification Analysis (RQA)

        Args:
            threshold: Recurrence threshold (as fraction of max distance)
            embedding_dim: Embedding dimension
            delay: Time delay
            min_line_length: Minimum line length for determinism calculation

        Returns:
            Dictionary of RQA metrics
        """
        if len(self.rr_ms) < 50:
            raise ValueError("RQA calculation requires at least 50 data points")

        rr_clean = self.rr_ms.copy()
        n = len(rr_clean)

        # Phase space reconstruction
        m = embedding_dim
        tau = delay
        embedded_length = n - (m - 1) * tau

        if embedded_length < 10:
            raise ValueError("Insufficient data for phase space reconstruction")

        embedded = np.zeros((embedded_length, m))
        for i in range(embedded_length):
            for j in range(m):
                embedded[i, j] = rr_clean[i + j * tau]

        # Calculate distance matrix
        dist_matrix = scipy.spatial.distance_matrix(embedded, embedded)

        # Set recurrence threshold
        if threshold <= 1.0:  # Relative threshold
            max_dist = np.max(dist_matrix)
            recurrence_threshold = threshold * max_dist
        else:  # Absolute threshold
            recurrence_threshold = threshold

        # Create recurrence matrix
        recurrence_matrix = (dist_matrix <= recurrence_threshold).astype(int)

        # Remove main diagonal (self-recurrences)
        np.fill_diagonal(recurrence_matrix, 0)

        N = recurrence_matrix.shape[0]
        total_points = N * N

        # Calculate basic RQA measures
        total_recurrent_points = np.sum(recurrence_matrix)

        # Recurrence Rate (RR)
        recurrence_rate = (
            total_recurrent_points / total_points if total_points > 0 else 0.0
        )

        # Find diagonal lines (determinism)
        diagonal_lines = []
        visited = np.zeros_like(recurrence_matrix, dtype=bool)

        for i in range(N):
            for j in range(N):
                if recurrence_matrix[i, j] and not visited[i, j]:
                    # Check diagonal line starting from (i, j)
                    k = 0
                    while (
                        i + k < N
                        and j + k < N
                        and recurrence_matrix[i + k, j + k]
                        and not visited[i + k, j + k]
                    ):
                        visited[i + k, j + k] = True
                        k += 1

                    if k >= min_line_length:
                        diagonal_lines.append(k)

        # Determinism (DET)
        if diagonal_lines:
            diagonal_points = sum(diagonal_lines)
            determinism = (
                diagonal_points / total_recurrent_points
                if total_recurrent_points > 0
                else 0.0
            )
            avg_diagonal_length = float(np.mean(diagonal_lines))
            max_diagonal_length = float(max(diagonal_lines))
        else:
            determinism = 0.0
            avg_diagonal_length = 0.0
            max_diagonal_length = 0.0

        # Find vertical lines (laminarity)
        vertical_lines = []
        visited_vertical = np.zeros_like(recurrence_matrix, dtype=bool)

        for j in range(N):
            i = 0
            while i < N:
                if recurrence_matrix[i, j] and not visited_vertical[i, j]:
                    k = 0
                    while (
                        i + k < N
                        and recurrence_matrix[i + k, j]
                        and not visited_vertical[i + k, j]
                    ):
                        visited_vertical[i + k, j] = True
                        k += 1

                    if k >= min_line_length:
                        vertical_lines.append(k)
                    i += k
                else:
                    i += 1

        # Laminarity (LAM)
        if vertical_lines:
            vertical_points = sum(vertical_lines)
            laminarity = (
                vertical_points / total_recurrent_points
                if total_recurrent_points > 0
                else 0.0
            )
            avg_vertical_length = float(np.mean(vertical_lines))
            max_vertical_length = float(max(vertical_lines))
        else:
            laminarity = 0.0
            avg_vertical_length = 0.0
            max_vertical_length = 0.0

        return {
            "recurrence_rate": float(recurrence_rate),
            "determinism": float(determinism),
            "avg_diagonal_length": avg_diagonal_length,
            "max_diagonal_length": max_diagonal_length,
            "laminarity": float(laminarity),
            "avg_vertical_length": avg_vertical_length,
            "max_vertical_length": max_vertical_length,
            "entropy_diagonal": float(
                -np.sum(
                    [
                        p * np.log(p)
                        for p in np.bincount(diagonal_lines) / len(diagonal_lines)
                        if p > 0
                    ]
                )
                if diagonal_lines
                else 0.0
            ),
            "entropy_vertical": float(
                -np.sum(
                    [
                        p * np.log(p)
                        for p in np.bincount(vertical_lines) / len(vertical_lines)
                        if p > 0
                    ]
                )
                if vertical_lines
                else 0.0
            ),
        }

    def full_nonlinear_analysis(
        self,
        include_mse: bool = True,
        include_dfa: bool = True,
        include_rqa: bool = True,
        mse_scales: int = 10,
        rqa_params: Optional[Dict] = None,
    ) -> Dict[str, Union[float, np.ndarray, Dict, None]]:
        """
        Perform complete nonlinear HRV analysis

        Args:
            include_mse: Whether to calculate multiscale entropy
            include_dfa: Whether to calculate DFA
            include_rqa: Whether to calculate RQA
            mse_scales: Number of scales for MSE
            rqa_params: Parameters for RQA analysis

        Returns:
            Comprehensive nonlinear analysis results
        """
        results = {}

        # Poincaré analysis
        try:
            sd1, sd2, sd1_sd2_ratio, poincare_additional = self.poincare_analysis()
            results["poincare"] = {
                "sd1": sd1,
                "sd2": sd2,
                "sd1_sd2_ratio": sd1_sd2_ratio,
                **poincare_additional,
            }
        except Exception as e:
            warnings.warn(f"Poincaré analysis failed: {e}")
            results["poincare"] = None

        # Sample Entropy
        try:
            sampen = self.sample_entropy()
            results["sample_entropy"] = sampen
        except Exception as e:
            warnings.warn(f"Sample entropy calculation failed: {e}")
            results["sample_entropy"] = None

        # Multiscale Entropy
        if include_mse:
            try:
                mse = self.multiscale_entropy(scale_max=mse_scales)
                results["multiscale_entropy"] = {
                    "values": mse,
                    "scales": list(range(1, len(mse) + 1)),
                    "area_under_curve": (
                        np.trapezoid(mse[~np.isnan(mse)])
                        if not np.all(np.isnan(mse))
                        else 0.0
                    ),
                }
            except Exception as e:
                warnings.warn(f"Multiscale entropy calculation failed: {e}")
                results["multiscale_entropy"] = None
        else:
            results["multiscale_entropy"] = None

        # Detrended Fluctuation Analysis
        if include_dfa:
            try:
                alpha1, alpha2, box_sizes, fluctuations = (
                    self.detrended_fluctuation_analysis()
                )
                results["dfa"] = {
                    "alpha1": alpha1,
                    "alpha2": alpha2,
                    "box_sizes": box_sizes,
                    "fluctuations": fluctuations,
                }
            except Exception as e:
                warnings.warn(f"DFA calculation failed: {e}")
                results["dfa"] = None
        else:
            results["dfa"] = None

        # Recurrence Quantification Analysis
        if include_rqa:
            try:
                rqa_params = rqa_params or {}
                rqa_metrics = self.recurrence_quantification_analysis(**rqa_params)
                results["rqa"] = rqa_metrics
            except Exception as e:
                warnings.warn(f"RQA calculation failed: {e}")
                results["rqa"] = None
        else:
            results["rqa"] = None

        # Analysis metadata
        results["analysis_info"] = {
            "total_intervals": len(self.rr_ms),
            "analysis_duration_s": np.sum(self.rr_ms) / 1000.0,
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
