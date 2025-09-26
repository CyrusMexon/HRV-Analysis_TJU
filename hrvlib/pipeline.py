"""
Unified HRV Analysis Pipeline
Integrates preprocessing → time_domain → freq_domain → nonlinear → respiratory
Follows the SRS requirements and maintains data consistency across modules
"""

import warnings
from typing import Dict, Optional, Tuple, Union, List
import numpy as np
from dataclasses import dataclass

# Import all integrated modules
from hrvlib.data_handler import DataBundle, TimeSeries, load_rr_file
from hrvlib.preprocessing import preprocess_rri, PreprocessingResult
from hrvlib.metrics.time_domain import HRVTimeDomainAnalysis
from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from hrvlib.metrics.nonlinear import NonlinearHRVAnalysis

from hrvlib.metrics.respiratory import (
    analyze_respiratory_metrics,
    add_respiratory_metrics_to_bundle,
)


@dataclass
class HRVAnalysisResults:
    """
    Complete HRV analysis results from unified pipeline
    """

    # Core analysis results
    time_domain: Optional[Dict] = None
    frequency_domain: Optional[Dict] = None
    nonlinear: Optional[Dict] = None
    respiratory: Optional[Dict] = None

    # Data quality and preprocessing info
    preprocessing_stats: Optional[Dict] = None
    quality_assessment: Optional[Dict] = None

    # Analysis metadata
    analysis_info: Dict = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.analysis_info is None:
            self.analysis_info = {}

    def to_dict(self) -> Dict:
        """
        Convert HRVAnalysisResults to dictionary format for display/export

        Returns:
            Dictionary containing all analysis results
        """
        result_dict = {
            "time_domain": self.time_domain,
            "frequency_domain": self.frequency_domain,
            "freq_domain": self.frequency_domain,  # Alternative key for compatibility
            "nonlinear": self.nonlinear,
            "respiratory": self.respiratory,
            "preprocessing_stats": self.preprocessing_stats,
            "quality_assessment": self.quality_assessment,
            "analysis_info": self.analysis_info,
            "warnings": self.warnings or [],
        }

        # Clean out private analyzer references for serialization
        if self.frequency_domain and "_analyzer" in self.frequency_domain:
            cleaned_freq_domain = {
                k: v for k, v in self.frequency_domain.items() if not k.startswith("_")
            }
            result_dict["frequency_domain"] = cleaned_freq_domain
            result_dict["freq_domain"] = cleaned_freq_domain

        return result_dict


class UnifiedHRVPipeline:
    """
    Unified HRV analysis pipeline that orchestrates all analysis modules
    Ensures consistent preprocessing and data flow across all components
    Follows SRS requirements FR-16 through FR-30
    """

    def __init__(
        self,
        bundle: DataBundle,
        preprocessing_config: Optional[Dict] = None,
        analysis_config: Optional[Dict] = None,
    ):
        """
        Initialize unified HRV pipeline

        Args:
            bundle: DataBundle containing physiological data
            preprocessing_config: Configuration for preprocessing step
            analysis_config: Configuration for analysis modules
        """
        self.bundle = bundle
        self.preprocessing_config = preprocessing_config or {}
        self.analysis_config = analysis_config or self._get_default_analysis_config()
        self.preprocessing_result = None

    def _get_default_analysis_config(self) -> Dict:
        """Get default configuration for all analysis modules"""
        return {
            "time_domain": {
                "enabled": True,
                "analysis_window": None,
            },
            "frequency_domain": {
                "enabled": True,
                "sampling_rate": 4.0,
                "detrend_method": "linear",
                "window_type": "hann",
                "segment_length": 120.0,
                "overlap_ratio": 0.75,
                "analysis_window": None,
            },
            "nonlinear": {
                "enabled": True,
                "include_mse": True,
                "include_dfa": True,
                "include_rqa": True,
                "mse_scales": 10,
                "analysis_window": None,
            },
            "respiratory": {
                "enabled": True,
            },
        }

    def run_time_domain_analysis(
        self,
        preprocessed_rri: np.ndarray,
        analysis_window: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict]:
        """Run time domain analysis on preprocessed data"""
        try:
            config = self.analysis_config.get("time_domain", {})
            if not config.get("enabled", True):
                return None

            window = analysis_window or config.get("analysis_window")

            analyzer = HRVTimeDomainAnalysis(
                preprocessed_rri=preprocessed_rri,
                preprocessing_result=self.preprocessing_result,
                analysis_window=window,
            )

            return analyzer.full_analysis()

        except Exception as e:
            warnings.warn(f"Time domain analysis failed: {e}")
            return None

    def run_frequency_domain_analysis(
        self,
        preprocessed_rri: np.ndarray,
        analysis_window: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict]:
        """Run frequency domain analysis on preprocessed data"""
        try:
            config = self.analysis_config.get("freq_domain", {})
            if not config.get("enabled", True):
                return None

            window = analysis_window or config.get("analysis_window")

            analyzer = HRVFreqDomainAnalysis(
                preprocessed_rri=preprocessed_rri,
                preprocessing_result=self.preprocessing_result,
                sampling_rate=config.get("sampling_rate", 4.0),
                detrend_method=config.get("detrend_method", "linear"),
                window_type=config.get("window_type", "hann"),
                segment_length=config.get("segment_length", 120.0),
                overlap_ratio=config.get("overlap_ratio", 0.75),
                analysis_window=window,
            )

            # Get basic results (spectral metrics)
            results = analyzer.get_results()

            if results is None:
                return None

            # ENHANCEMENT: Add PSD data for plotting using your existing method
            try:
                freqs, psd = analyzer.get_psd()
                results["psd_frequencies"] = freqs
                results["psd_power"] = psd
                results["psd_available"] = len(freqs) > 0 and len(psd) > 0
            except Exception as e:
                warnings.warn(f"Failed to get PSD data: {e}")
                results["psd_frequencies"] = np.array([])
                results["psd_power"] = np.array([])
                results["psd_available"] = False

            # ENHANCEMENT: Add organized band powers using your existing method
            try:
                band_summary = analyzer.get_band_powers_summary()
                results["band_powers_summary"] = band_summary

                # Also flatten some key metrics for easier access by widgets
                for band_name, band_data in band_summary.items():
                    if band_name != "lf_hf_ratio":
                        results[f"{band_name}_power"] = band_data.get(
                            "absolute_power", 0.0
                        )
                        results[f"{band_name}_power_rel"] = band_data.get(
                            "relative_power_pct", 0.0
                        )
                        if "peak_frequency" in band_data:
                            results[f"peak_freq_{band_name}"] = band_data[
                                "peak_frequency"
                            ]

                # Add LF/HF ratio
                if "lf_hf_ratio" in band_summary:
                    lf_hf_data = band_summary["lf_hf_ratio"]
                    results["lf_hf_ratio"] = lf_hf_data.get("value", float("nan"))
                    results["relative_lf_power"] = lf_hf_data.get(
                        "relative_lf_pct", 0.0
                    )
                    results["relative_hf_power"] = lf_hf_data.get(
                        "relative_hf_pct", 0.0
                    )

            except Exception as e:
                warnings.warn(f"Failed to get band powers summary: {e}")

            # Store analyzer reference for potential future use
            results["_analyzer"] = analyzer  # Private reference, won't be serialized

            return results

        except Exception as e:
            warnings.warn(f"Frequency domain analysis failed: {e}")
            return None

    def run_nonlinear_analysis(
        self,
        preprocessed_rri: np.ndarray,
        analysis_window: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict]:
        """Run nonlinear analysis on preprocessed data"""
        try:
            config = self.analysis_config.get("nonlinear", {})
            if not config.get("enabled", True):
                return None

            window = analysis_window or config.get("analysis_window")

            analyzer = NonlinearHRVAnalysis(
                preprocessed_rri=preprocessed_rri,
                preprocessing_result=self.preprocessing_result,
                analysis_window=window,
            )

            return analyzer.full_nonlinear_analysis(
                include_mse=config.get("include_mse", True),
                include_dfa=config.get("include_dfa", True),
                include_rqa=config.get("include_rqa", True),
                mse_scales=config.get("mse_scales", 10),
            )

        except Exception as e:
            warnings.warn(f"Nonlinear analysis failed: {e}")
            return None

    def run_respiratory_analysis(
        self, preprocessed_rri: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """Run respiratory analysis"""
        try:
            config = self.analysis_config.get("respiratory", {})
            if not config.get("enabled", True):
                return None

            results = analyze_respiratory_metrics(
                self.bundle, preprocessed_rri=preprocessed_rri
            )

            # Keep bundle.meta in sync
            self.bundle.meta["respiratory_metrics"] = results

            return results

        except Exception as e:
            warnings.warn(f"Respiratory analysis failed: {e}")
            return None

    def _run_preprocessing(self) -> Tuple[np.ndarray, PreprocessingResult]:
        """Run preprocessing on the bundle's RRI data"""
        preprocessing_result = preprocess_rri(
            self.bundle.rri_ms, **self.preprocessing_config
        )
        return preprocessing_result.corrected_rri, preprocessing_result

    def run_all(
        self, analysis_window: Optional[Tuple[float, float]] = None
    ) -> HRVAnalysisResults:
        """
        Run complete HRV analysis pipeline

        Args:
            analysis_window: Optional time window for analysis (start_s, end_s)

        Returns:
            HRVAnalysisResults with all computed metrics
        """
        # Initialize results container
        results = HRVAnalysisResults()

        try:
            # Step 1: Centralized preprocessing of the RR intervals
            # This includes artifact detection and correction if needed
            preprocessed_rri, preprocessing_result = self._run_preprocessing()
            self.preprocessing_result = preprocessing_result

            # Apply analysis window if specified by the user
            # This allows for focused analysis on specific time periods
            if analysis_window is not None:
                preprocessed_rri = self._apply_analysis_window(
                    preprocessed_rri, analysis_window
                )

            # Step 2: Run all analysis modules with preprocessed data
            results.time_domain = self.run_time_domain_analysis(
                preprocessed_rri, analysis_window
            )

            results.frequency_domain = self.run_frequency_domain_analysis(
                preprocessed_rri, analysis_window
            )

            results.nonlinear = self.run_nonlinear_analysis(
                preprocessed_rri, analysis_window
            )

            results.respiratory = self.run_respiratory_analysis(preprocessed_rri)

            # Step 3: Add preprocessing and quality information
            if preprocessing_result is not None:
                results.preprocessing_stats = {
                    "artifacts_detected": preprocessing_result.stats[
                        "artifacts_detected"
                    ],
                    "artifacts_corrected": preprocessing_result.stats[
                        "artifacts_corrected"
                    ],
                    "artifact_percentage": preprocessing_result.stats[
                        "artifact_percentage"
                    ],
                    "noise_segments": len(preprocessing_result.noise_segments),
                    "correction_method": preprocessing_result.correction_method,
                    "quality_flags": preprocessing_result.quality_flags,
                }

            # Step 4: Overall quality assessment
            results.quality_assessment = self._assess_overall_quality(
                preprocessed_rri, preprocessing_result
            )

            # Step 5: Analysis metadata
            results.analysis_info = {
                "total_intervals": len(preprocessed_rri),
                "analysis_duration_s": np.sum(preprocessed_rri) / 1000.0,
                "preprocessing_applied": preprocessing_result is not None,
                "analysis_window": analysis_window,
                "modules_enabled": {
                    module: config.get("enabled", True)
                    for module, config in self.analysis_config.items()
                },
            }

        except Exception as e:
            results.warnings.append(f"Pipeline execution failed: {e}")

        return results

    def _apply_analysis_window(
        self, rr_ms: np.ndarray, analysis_window: Tuple[float, float]
    ) -> np.ndarray:
        """Apply analysis window to RRI data"""
        start_time, end_time = analysis_window

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

    def _assess_overall_quality(
        self, rr_ms: np.ndarray, preprocessing_result: Optional[PreprocessingResult]
    ) -> Dict[str, Union[str, bool, float, List[str]]]:
        """Assess overall data quality"""
        assessment = {
            "overall_quality": "unknown",
            "data_length_adequate": len(rr_ms) >= 50,
            "duration_s": np.sum(rr_ms) / 1000.0,
            "recommendations": [],
        }

        # Duration assessment
        duration_s = np.sum(rr_ms) / 1000.0
        if duration_s < 120:
            assessment["recommendations"].append(
                "Recording duration < 2 minutes may limit metric reliability"
            )
        elif duration_s < 300:
            assessment["recommendations"].append(
                "Consider longer recordings (≥5 minutes) for more stable metrics"
            )

        # Preprocessing quality assessment
        if preprocessing_result is not None:
            artifact_pct = preprocessing_result.stats.get("artifact_percentage", 0)
            quality_flags = preprocessing_result.quality_flags

            assessment["artifact_percentage"] = artifact_pct

            if quality_flags:
                if quality_flags.get("poor_signal_quality", False):
                    assessment["overall_quality"] = "poor"
                    assessment["recommendations"].append("Poor signal quality detected")
                elif quality_flags.get("excessive_artifacts", False):
                    assessment["overall_quality"] = "fair"
                    assessment["recommendations"].append("High artifact rate detected")
                else:
                    assessment["overall_quality"] = "good"

            if artifact_pct > 10:
                assessment["recommendations"].append(
                    f"High artifact percentage ({artifact_pct:.1f}%)"
                )

        #  Respiratory quality assessment
        resp_metrics = self.bundle.meta.get("respiratory_metrics")
        if resp_metrics:
            confidence = resp_metrics.get("confidence", 0.0)
            if confidence < 0.5:
                assessment["overall_quality"] = "fair"
                assessment["recommendations"].append(
                    f"Low respiratory signal confidence ({confidence:.2f})"
                )

            if resp_metrics.get("warnings"):
                for w in resp_metrics["warnings"]:
                    assessment["recommendations"].append(f"Respiratory: {w}")

            # Check LF/HF overlap annotation
            lf_hf = resp_metrics.get("lf_hf_analysis")
            if lf_hf and lf_hf.get("boundary_overlap"):
                assessment["recommendations"].append(
                    "Respiratory frequency overlaps LF/HF boundary – interpret LF/HF ratio cautiously"
                )
        return assessment


# Factory functions for backward compatibility
def create_unified_pipeline(
    bundle: DataBundle,
    preprocessing_config: Optional[Dict] = None,
    analysis_config: Optional[Dict] = None,
) -> UnifiedHRVPipeline:
    """
    Factory function to create unified HRV pipeline

    Args:
        bundle: DataBundle with physiological data
        preprocessing_config: Configuration for preprocessing
        analysis_config: Configuration for analysis modules

    Returns:
        Configured UnifiedHRVPipeline instance
    """
    return UnifiedHRVPipeline(
        bundle=bundle,
        preprocessing_config=preprocessing_config,
        analysis_config=analysis_config,
    )
