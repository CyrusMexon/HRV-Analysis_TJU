import unittest
import numpy as np
import warnings
from unittest.mock import patch, Mock
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from hrvlib.preprocessing import preprocess_rri, PreprocessingResult


class TestHRVFreqDomainAnalysis(unittest.TestCase):
    """Comprehensive test suite for HRV frequency domain analysis with preprocessed RRI input"""

    def setUp(self):
        """Set up test data and preprocessing objects"""
        np.random.seed(42)

        # Create synthetic RR interval data with known spectral characteristics
        self.duration = 300  # 5 minutes
        self.fs = 4.0  # Sampling rate for resampling

        # Generate time vector for original RR intervals
        n_beats = 300  # approximately 300 beats for 5 minutes
        base_rr = 800  # 800ms = 75 BPM

        # Create RR intervals with known frequency components
        beat_times = np.cumsum(np.full(n_beats, base_rr / 1000))  # Convert to seconds

        # Add respiratory modulation (0.25 Hz)
        resp_component = 40 * np.sin(2 * np.pi * 0.25 * beat_times)

        # Add low frequency oscillation (0.1 Hz)
        lf_component = 30 * np.sin(2 * np.pi * 0.1 * beat_times)

        # Add VLF component (0.02 Hz)
        vlf_component = 20 * np.sin(2 * np.pi * 0.02 * beat_times)

        # Create RR intervals with noise
        self.rr_intervals_ms = (
            base_rr
            + resp_component
            + lf_component
            + vlf_component
            + 10 * np.random.randn(n_beats)
        )

        # Ensure positive values
        self.rr_intervals_ms = np.maximum(self.rr_intervals_ms, 400)

        # Create data with artifacts for preprocessing testing
        self.rr_with_artifacts = self.rr_intervals_ms.copy()
        # Add some artifacts
        self.rr_with_artifacts[50] = 150  # Extra beat (too short)
        self.rr_with_artifacts[100] = 2500  # Missed beat (too long)
        self.rr_with_artifacts[150] = self.rr_with_artifacts[149] * 2.5  # Ectopic beat

        # Create short sequence for edge case testing
        self.short_rr_ms = np.array([800, 820, 790, 810, 805, 815, 795, 825])

        # Create very short sequence
        self.very_short_rr_ms = np.array([800, 820])

        # Create real preprocessing results
        self.normal_preprocessing_result = preprocess_rri(
            self.rr_intervals_ms.tolist(),
            threshold_low=300.0,
            threshold_high=2000.0,
            ectopic_threshold=0.3,
            correction_method="cubic_spline",
        )

        self.artifact_preprocessing_result = preprocess_rri(
            self.rr_with_artifacts.tolist(),
            threshold_low=300.0,
            threshold_high=2000.0,
            ectopic_threshold=0.3,
            correction_method="cubic_spline",
        )

        self.short_preprocessing_result = preprocess_rri(
            self.short_rr_ms.tolist(),
            threshold_low=300.0,
            threshold_high=2000.0,
            ectopic_threshold=0.3,
            correction_method="cubic_spline",
        )

    def test_real_preprocessing_result_structure(self):
        """Test that real PreprocessingResult has expected structure"""
        result = self.normal_preprocessing_result

        # Verify all expected attributes exist
        self.assertIsInstance(result, PreprocessingResult)
        self.assertTrue(hasattr(result, "original_rri"))
        self.assertTrue(hasattr(result, "corrected_rri"))
        self.assertTrue(hasattr(result, "artifact_indices"))
        self.assertTrue(hasattr(result, "artifact_types"))
        self.assertTrue(hasattr(result, "interpolation_indices"))
        self.assertTrue(hasattr(result, "correction_method"))
        self.assertTrue(hasattr(result, "stats"))
        self.assertTrue(hasattr(result, "correction_details"))
        self.assertTrue(hasattr(result, "noise_segments"))
        self.assertTrue(hasattr(result, "quality_flags"))

        # Verify stats structure
        self.assertIsInstance(result.stats, dict)
        expected_stats_keys = [
            "original_count",
            "final_count",
            "artifacts_detected",
            "artifacts_corrected",
            "extra_beats_removed",
            "intervals_interpolated",
            "artifact_percentage",
            "noise_segments_count",
            "noise_percentage",
            "original_mean",
            "corrected_mean",
            "original_std",
            "corrected_std",
        ]
        for key in expected_stats_keys:
            self.assertIn(key, result.stats, f"Missing stats key: {key}")

        # Verify quality_flags structure
        self.assertIsInstance(result.quality_flags, dict)
        expected_quality_keys = [
            "high_noise",
            "excessive_artifacts",
            "poor_signal_quality",
            "irregular_rhythm",
        ]
        for key in expected_quality_keys:
            self.assertIn(key, result.quality_flags, f"Missing quality flag: {key}")

    def test_initialization_valid_inputs(self):
        """Test successful initialization with preprocessed RRI"""
        # Basic initialization with preprocessed RRI
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            preprocessing_result=self.normal_preprocessing_result,
        )
        self.assertIsInstance(analyzer, HRVFreqDomainAnalysis)
        self.assertEqual(analyzer.sampling_rate, 4.0)
        self.assertEqual(analyzer.window_type, "hann")

        # Custom parameters
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            preprocessing_result=self.normal_preprocessing_result,
            sampling_rate=2.0,
            window_type="hamming",
            segment_length=180.0,
            overlap_ratio=0.5,
            detrend_method="constant",
        )
        self.assertEqual(analyzer.sampling_rate, 2.0)
        self.assertEqual(analyzer.window_type, "hamming")
        self.assertEqual(analyzer.segment_length, 180.0)
        self.assertEqual(analyzer.overlap_ratio, 0.5)
        self.assertEqual(analyzer.detrend_method, "constant")

        # Without preprocessing result (just preprocessed RRI)
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)
        self.assertIsInstance(analyzer, HRVFreqDomainAnalysis)

    def test_initialization_invalid_inputs(self):
        """Test initialization with invalid parameters"""
        # Invalid sampling rate
        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, sampling_rate=-1.0
            )

        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, sampling_rate=0.0
            )

        # Invalid detrend method
        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, detrend_method="invalid"
            )

        # Invalid window type
        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, window_type="invalid"
            )

        # Invalid segment length
        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, segment_length=-10.0
            )

        # Invalid overlap ratio
        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, overlap_ratio=-0.1
            )

        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, overlap_ratio=1.0
            )

        # Invalid band convention
        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri,
                band_convention="invalid",
            )

    def test_empty_rri_array(self):
        """Test behavior with empty RRI array"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            analyzer = HRVFreqDomainAnalysis(np.array([]))
            results = analyzer.get_results()

            # Should handle empty input gracefully
            self.assertEqual(results["total_power"], 0.0)
            self.assertEqual(len(analyzer.time_domain), 0)

    def test_with_preprocessing_result(self):
        """Test analyzer with existing preprocessing result"""
        analyzer = HRVFreqDomainAnalysis(
            self.artifact_preprocessing_result.corrected_rri,
            preprocessing_result=self.artifact_preprocessing_result,
        )

        # Should use existing preprocessing result
        self.assertIsNotNone(analyzer.preprocessing_result)
        self.assertIsInstance(analyzer.preprocessing_result, PreprocessingResult)

        # Get results to ensure it works
        results = analyzer.get_results()
        self.assertIn("preprocessing_stats", results)

        # Verify actual preprocessing statistics are included
        preprocessing_stats = results["preprocessing_stats"]
        self.assertIsInstance(preprocessing_stats["artifacts_detected"], int)
        self.assertIsInstance(preprocessing_stats["artifacts_corrected"], int)
        self.assertIsInstance(preprocessing_stats["artifact_percentage"], float)
        self.assertIsInstance(preprocessing_stats["noise_segments"], int)
        self.assertEqual(preprocessing_stats["correction_method"], "cubic_spline")
        self.assertIsInstance(preprocessing_stats["quality_flags"], dict)

    def test_without_preprocessing_result(self):
        """Test analyzer without preprocessing result"""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)

        # Should work without preprocessing result
        self.assertIsNone(analyzer.preprocessing_result)

        # Get results
        results = analyzer.get_results()
        self.assertNotIn("preprocessing_stats", results)
        self.assertFalse(results["analysis_info"]["preprocessing_applied"])

    def test_analysis_window(self):
        """Test analysis window functionality"""
        # Test with valid analysis window
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            analysis_window=(30.0, 150.0),  # 30-150 seconds
        )

        results = analyzer.get_results()
        self.assertGreater(results["total_power"], 0)
        self.assertEqual(results["analysis_info"]["analysis_window"], (30.0, 150.0))

        # Test with invalid analysis window (no data)
        with self.assertRaises(ValueError):
            HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri,
                analysis_window=(500.0, 600.0),  # Beyond data range
            )

    def test_spectral_metrics_calculation(self):
        """Test calculation of all spectral metrics"""
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            preprocessing_result=self.normal_preprocessing_result,
            segment_length=240,
        )
        results = analyzer.get_results()

        # Test presence of all expected metrics
        expected_metrics = [
            "ulf_power",
            "ulf_power_nu",
            "vlf_power",
            "vlf_power_nu",
            "lf_power",
            "lf_power_nu",
            "hf_power",
            "hf_power_nu",
            "lf_hf_ratio",
            "total_power",
            "peak_freq_vlf",
            "peak_freq_lf",
            "peak_freq_hf",
            "relative_lf_power",
            "relative_hf_power",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, results, f"Missing metric: {metric}")

        # Test metric relationships
        total_power = results["total_power"]
        self.assertGreater(total_power, 0)

        # Sum of band powers should be less than or equal to total power
        band_sum = (
            results["ulf_power"]
            + results["vlf_power"]
            + results["lf_power"]
            + results["hf_power"]
        )
        self.assertLessEqual(
            band_sum, total_power * 1.01
        )  # Allow small numerical error

        # Normalized powers should sum to approximately 100%
        norm_sum = (
            results["ulf_power_nu"]
            + results["vlf_power_nu"]
            + results["lf_power_nu"]
            + results["hf_power_nu"]
        )
        self.assertAlmostEqual(norm_sum, 100.0, delta=5.0)

        # LF/HF ratio should match calculation
        if results["hf_power"] > 1e-10:
            expected_ratio = results["lf_power"] / results["hf_power"]
            self.assertAlmostEqual(results["lf_hf_ratio"], expected_ratio, places=6)

        # Relative powers should sum to 100%
        rel_sum = results["relative_lf_power"] + results["relative_hf_power"]
        self.assertAlmostEqual(rel_sum, 100.0, delta=0.1)

    def test_standard_band_convention_boundary_masks(self):
        """Standard convention should use mutually exclusive physiological bands."""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)
        freqs = np.array([0.0, 0.001, 0.0029, 0.003, 0.039, 0.04, 0.149, 0.15, 0.4])

        ulf = analyzer._band_mask(freqs, "ulf")
        vlf = analyzer._band_mask(freqs, "vlf")
        lf = analyzer._band_mask(freqs, "lf")
        hf = analyzer._band_mask(freqs, "hf")

        self.assertFalse(ulf[0])  # DC excluded.
        self.assertTrue(ulf[1])
        self.assertTrue(ulf[2])
        self.assertFalse(ulf[3])  # 0.003 belongs to VLF.
        self.assertTrue(vlf[3])
        self.assertTrue(vlf[4])
        self.assertFalse(vlf[5])  # 0.04 belongs to LF.
        self.assertTrue(lf[5])
        self.assertTrue(lf[6])
        self.assertFalse(lf[7])  # 0.15 belongs to HF.
        self.assertTrue(hf[7])
        self.assertTrue(hf[8])

        summed_membership = ulf.astype(int) + vlf.astype(int) + lf.astype(int) + hf.astype(int)
        self.assertLessEqual(np.max(summed_membership), 1)

    def test_standard_total_power_excludes_dc(self):
        """Standard total power should not include the exact 0-Hz bin."""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)
        analyzer.freqs = np.array([0.0, 0.01, 0.02, 0.4])
        analyzer.psd = np.array([1e9, 1.0, 1.0, 1.0])

        metrics = analyzer._compute_spectral_metrics()

        self.assertAlmostEqual(metrics["total_power"], 0.39, places=12)
        self.assertAlmostEqual(metrics["vlf_power"], 0.01, places=12)

    def test_kubios_compatible_mode_retains_legacy_vlf_and_total(self):
        """Kubios-compatible convention should include DC in VLF and total power."""
        standard = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            band_convention="standard",
        )
        kubios = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            band_convention="kubios_compatible",
        )
        freqs = np.array([0.0, 0.01, 0.02, 0.04, 0.4])
        psd = np.array([1e9, 1.0, 1.0, 1.0, 1.0])
        standard.freqs = freqs
        standard.psd = psd
        kubios.freqs = freqs
        kubios.psd = psd

        standard_metrics = standard._compute_spectral_metrics()
        kubios_metrics = kubios._compute_spectral_metrics()

        self.assertTrue(kubios._band_mask(freqs, "vlf")[0])
        self.assertTrue(kubios._total_power_mask(freqs)[0])
        self.assertGreater(kubios_metrics["vlf_power"], standard_metrics["vlf_power"] * 1000)
        self.assertGreater(kubios_metrics["total_power"], standard_metrics["total_power"] * 1000)

    def test_welch_fft_ar_all_use_selected_band_convention(self):
        """All spectral estimators should route through the selected band masks."""
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            band_convention="standard",
        )
        freqs = np.array([0.0, 0.01, 0.02, 0.4])
        psd = np.array([1e9, 1.0, 1.0, 1.0])
        analyzer.freqs = freqs
        analyzer.psd = psd
        analyzer.fft_freqs = freqs
        analyzer.fft_psd = psd
        analyzer.ar_freqs = freqs
        analyzer.ar_psd = psd

        welch = analyzer._compute_spectral_metrics()
        fft = analyzer._compute_spectral_metrics(use_fft=True)
        ar = analyzer._compute_spectral_metrics(use_ar=True)

        for metrics in (welch, fft, ar):
            self.assertAlmostEqual(metrics["total_power"], 0.39, places=12)
            self.assertAlmostEqual(metrics["vlf_power"], 0.01, places=12)

    def test_derived_frequency_metric_identities(self):
        """Derived normalized and ratio metrics should retain their identities."""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)
        analyzer.freqs = np.array([0.003, 0.02, 0.039, 0.04, 0.10, 0.149, 0.15, 0.30, 0.40])
        analyzer.psd = np.array([1.0, 1.5, 1.2, 2.0, 4.0, 2.5, 3.0, 5.0, 4.0])

        metrics = analyzer._compute_spectral_metrics()
        lf_power = metrics["lf_power"]
        hf_power = metrics["hf_power"]
        lf_hf_sum = lf_power + hf_power

        self.assertAlmostEqual(metrics["lf_hf_ratio"], lf_power / hf_power, places=12)
        self.assertAlmostEqual(metrics["lf_nu"], lf_power / lf_hf_sum * 100.0, places=12)
        self.assertAlmostEqual(metrics["hf_nu"], hf_power / lf_hf_sum * 100.0, places=12)
        self.assertAlmostEqual(metrics["relative_lf_power"], metrics["lf_nu"], places=12)
        self.assertAlmostEqual(metrics["relative_hf_power"], metrics["hf_nu"], places=12)

    def test_comprehensive_results_structure_with_preprocessing(self):
        """Test comprehensive structure of results with preprocessing data"""
        analyzer = HRVFreqDomainAnalysis(
            self.artifact_preprocessing_result.corrected_rri,
            preprocessing_result=self.artifact_preprocessing_result,
        )
        results = analyzer.get_results()

        # Test main metrics
        main_metrics = [
            "ulf_power",
            "ulf_power_nu",
            "vlf_power",
            "vlf_power_nu",
            "lf_power",
            "lf_power_nu",
            "hf_power",
            "hf_power_nu",
            "lf_hf_ratio",
            "total_power",
            "peak_freq_vlf",
            "peak_freq_lf",
            "peak_freq_hf",
            "relative_lf_power",
            "relative_hf_power",
        ]

        for metric in main_metrics:
            self.assertIn(metric, results)

        # Test analysis_info structure
        self.assertIn("analysis_info", results)
        analysis_info = results["analysis_info"]

        expected_info_fields = [
            "sampling_rate",
            "window_type",
            "detrend_method",
            "segment_length_s",
            "overlap_ratio",
            "signal_duration_s",
            "frequency_resolution",
            "preprocessing_applied",
            "analysis_window",
            "band_convention",
            "band_definitions",
            "total_power_definition",
        ]

        for field in expected_info_fields:
            self.assertIn(field, analysis_info)
        self.assertEqual(analysis_info["band_convention"], "standard")

        # Test preprocessing_stats structure with real data
        self.assertIn("preprocessing_stats", results)
        preprocessing_stats = results["preprocessing_stats"]

        # Check that all expected preprocessing statistics are present and have correct types
        expected_stats = {
            "artifacts_detected": int,
            "artifacts_corrected": int,
            "artifact_percentage": float,
            "noise_segments": int,
            "correction_method": str,
            "quality_flags": dict,
        }

        for field, expected_type in expected_stats.items():
            self.assertIn(
                field, preprocessing_stats, f"Missing preprocessing stat: {field}"
            )
            self.assertIsInstance(
                preprocessing_stats[field],
                expected_type,
                f"Wrong type for {field}: expected {expected_type}, got {type(preprocessing_stats[field])}",
            )

    def test_window_functions(self):
        """Test all supported window functions"""
        valid_windows = [
            "hann",
            "hamming",
            "blackman",
            "bartlett",
            "flattop",
            "parzen",
            "bohman",
            "nuttall",
        ]

        for window in valid_windows:
            analyzer = HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, window_type=window
            )
            results = analyzer.get_results()
            self.assertGreater(
                results["total_power"], 0, f"Failed for window: {window}"
            )

    def test_detrend_methods(self):
        """Test all supported detrend methods"""
        valid_detrends = ["linear", "constant", None]

        for detrend in valid_detrends:
            analyzer = HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, detrend_method=detrend
            )
            results = analyzer.get_results()
            self.assertGreater(
                results["total_power"], 0, f"Failed for detrend: {detrend}"
            )

    def test_short_signal_handling(self):
        """Test handling of short signals"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            analyzer = HRVFreqDomainAnalysis(
                self.short_preprocessing_result.corrected_rri,
                preprocessing_result=self.short_preprocessing_result,
            )
            results = analyzer.get_results()

            # Should produce warnings for short signals
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(any("duration" in msg.lower() for msg in warning_messages))

            # But should still produce results
            self.assertGreaterEqual(results["total_power"], 0)

    def test_get_psd_method(self):
        """Test get_psd method"""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)
        freqs, psd = analyzer.get_psd()

        self.assertIsInstance(freqs, np.ndarray)
        self.assertIsInstance(psd, np.ndarray)
        self.assertEqual(len(freqs), len(psd))

        if len(freqs) > 0:
            self.assertGreater(freqs[-1], freqs[0])  # Frequencies should be increasing
            self.assertTrue(np.all(psd >= 0))  # PSD should be non-negative

    def test_get_band_powers_summary(self):
        """Test get_band_powers_summary method"""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)
        summary = analyzer.get_band_powers_summary()

        expected_bands = ["ulf", "vlf", "lf", "hf", "lf_hf_ratio"]
        for band in expected_bands:
            self.assertIn(band, summary)

        # Check structure for regular bands
        for band in ["ulf", "vlf", "lf", "hf"]:
            self.assertIn("frequency_range_hz", summary[band])
            self.assertIn("absolute_power", summary[band])
            self.assertIn("relative_power_pct", summary[band])

        # Check LF/HF ratio structure
        self.assertIn("value", summary["lf_hf_ratio"])
        self.assertIn("relative_lf_pct", summary["lf_hf_ratio"])
        self.assertIn("relative_hf_pct", summary["lf_hf_ratio"])

    def test_validate_frequency_analysis_with_preprocessing(self):
        """Test frequency analysis validation with preprocessing data"""
        # Test with normal signal
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            preprocessing_result=self.normal_preprocessing_result,
        )
        validation = analyzer.validate_frequency_analysis()

        self.assertIn("is_valid", validation)
        self.assertIn("warnings", validation)
        self.assertIn("recommendations", validation)
        self.assertIn("signal_duration_s", validation)
        self.assertIn("frequency_resolution_hz", validation)

        # Test with preprocessed signal that had quality issues
        artifact_analyzer = HRVFreqDomainAnalysis(
            self.artifact_preprocessing_result.corrected_rri,
            preprocessing_result=self.artifact_preprocessing_result,
        )
        artifact_validation = artifact_analyzer.validate_frequency_analysis()

        # Check if preprocessing detected quality issues
        preprocessing_result = artifact_analyzer.preprocessing_result
        if preprocessing_result.stats["artifact_percentage"] > 5.0:
            self.assertTrue(len(artifact_validation["warnings"]) > 0)
            artifact_warnings = [
                w for w in artifact_validation["warnings"] if "artifact" in w.lower()
            ]
            self.assertTrue(len(artifact_warnings) > 0)

    def test_raw_vs_preprocessed_comparison(self):
        """Compare results with raw vs preprocessed RRI"""
        # Analyze raw (but clean) data
        raw_analyzer = HRVFreqDomainAnalysis(self.rr_intervals_ms)
        raw_results = raw_analyzer.get_results()

        # Analyze preprocessed data (from data with artifacts)
        preprocessed_analyzer = HRVFreqDomainAnalysis(
            self.artifact_preprocessing_result.corrected_rri,
            preprocessing_result=self.artifact_preprocessing_result,
        )
        preprocessed_results = preprocessed_analyzer.get_results()

        # Both should produce valid results
        self.assertGreater(raw_results["total_power"], 0)
        self.assertGreater(preprocessed_results["total_power"], 0)

        # Preprocessed should have preprocessing stats
        self.assertNotIn("preprocessing_stats", raw_results)
        self.assertIn("preprocessing_stats", preprocessed_results)

        # Analysis info should reflect preprocessing status
        self.assertFalse(raw_results["analysis_info"]["preprocessing_applied"])
        self.assertTrue(preprocessed_results["analysis_info"]["preprocessing_applied"])

    def test_real_artifact_detection_verification(self):
        """Verify that preprocessing result correctly reflects artifact detection"""
        # Create data with known artifacts
        rr_with_known_artifacts = np.array(
            [
                800,
                820,
                150,
                810,
                2500,
                805,
                815,  # extra beat, missed beat
            ]
        )

        preprocessing_result = preprocess_rri(
            rr_with_known_artifacts.tolist(),
            threshold_low=300.0,
            threshold_high=2000.0,
            ectopic_threshold=0.3,
            correction_method="cubic_spline",
        )

        analyzer = HRVFreqDomainAnalysis(
            preprocessing_result.corrected_rri,
            preprocessing_result=preprocessing_result,
        )

        # Should have detected and corrected artifacts
        self.assertGreater(preprocessing_result.stats["artifacts_detected"], 0)
        self.assertGreater(preprocessing_result.stats["artifacts_corrected"], 0)

        # Corrected RRI should be different from original
        self.assertFalse(
            np.array_equal(
                preprocessing_result.original_rri, preprocessing_result.corrected_rri
            )
        )

        # Should have specific correction details
        correction_details = preprocessing_result.correction_details
        self.assertIsInstance(correction_details, dict)
        self.assertIn("extra_beats_removed", correction_details)
        self.assertIn("intervals_interpolated", correction_details)

    def create_comprehensive_validation_plot(self):
        """Create comprehensive validation plot comparing raw vs preprocessed analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Raw data PSD
        raw_analyzer = HRVFreqDomainAnalysis(self.rr_with_artifacts)
        raw_freqs, raw_psd = raw_analyzer.get_psd()
        raw_results = raw_analyzer.get_results()

        if len(raw_freqs) > 0:
            ax1.semilogy(raw_freqs, raw_psd, "b-", linewidth=1.5, label="Raw Data")
            ax1.set_title("Raw Data PSD (With Artifacts)")
            ax1.set_xlabel("Frequency [Hz]")
            ax1.set_ylabel("Power [ms²/Hz]")
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 0.5)
            self._add_frequency_bands(ax1)

        # Plot 2: Preprocessed data PSD
        preprocessed_analyzer = HRVFreqDomainAnalysis(
            self.artifact_preprocessing_result.corrected_rri,
            preprocessing_result=self.artifact_preprocessing_result,
        )
        prep_freqs, prep_psd = preprocessed_analyzer.get_psd()
        prep_results = preprocessed_analyzer.get_results()

        if len(prep_freqs) > 0:
            ax2.semilogy(
                prep_freqs, prep_psd, "r-", linewidth=1.5, label="Preprocessed Data"
            )
            ax2.set_title("Preprocessed Data PSD")
            ax2.set_xlabel("Frequency [Hz]")
            ax2.set_ylabel("Power [ms²/Hz]")
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 0.5)
            self._add_frequency_bands(ax2)

            # Mark peak frequencies if detected
            if not np.isnan(prep_results["peak_freq_lf"]):
                ax2.axvline(
                    prep_results["peak_freq_lf"],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label=f'LF Peak: {prep_results["peak_freq_lf"]:.3f} Hz',
                )
            if not np.isnan(prep_results["peak_freq_hf"]):
                ax2.axvline(
                    prep_results["peak_freq_hf"],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f'HF Peak: {prep_results["peak_freq_hf"]:.3f} Hz',
                )
            ax2.legend()

        # Plot 3: Band power comparison
        band_names = ["ULF", "VLF", "LF", "HF"]
        raw_powers = [
            raw_results["ulf_power"],
            raw_results["vlf_power"],
            raw_results["lf_power"],
            raw_results["hf_power"],
        ]
        prep_powers = [
            prep_results["ulf_power"],
            prep_results["vlf_power"],
            prep_results["lf_power"],
            prep_results["hf_power"],
        ]

        x = np.arange(len(band_names))
        width = 0.35

        bars1 = ax3.bar(
            x - width / 2, raw_powers, width, label="Raw", alpha=0.7, color="blue"
        )
        bars2 = ax3.bar(
            x + width / 2,
            prep_powers,
            width,
            label="Preprocessed",
            alpha=0.7,
            color="red",
        )

        ax3.set_title("Band Power Comparison: Raw vs Preprocessed")
        ax3.set_ylabel("Power [ms²]")
        ax3.set_xticks(x)
        ax3.set_xticklabels(band_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # Plot 4: Preprocessing statistics and quality info
        ax4.axis("off")  # Turn off axes for text display

        # Display preprocessing statistics
        if preprocessed_analyzer.preprocessing_result:
            stats = preprocessed_analyzer.preprocessing_result.stats
            quality_flags = preprocessed_analyzer.preprocessing_result.quality_flags

            stats_text = [
                "Preprocessing Statistics:",
                f"  Original RR count: {stats['original_count']}",
                f"  Final RR count: {stats['final_count']}",
                f"  Artifacts detected: {stats['artifacts_detected']}",
                f"  Artifacts corrected: {stats['artifacts_corrected']}",
                f"  Artifact percentage: {stats['artifact_percentage']:.1f}%",
                f"  Extra beats removed: {stats['extra_beats_removed']}",
                f"  Intervals interpolated: {stats['intervals_interpolated']}",
                f"  Noise segments: {stats['noise_segments_count']}",
                "",
                "Quality Flags:",
                f"  High noise: {quality_flags['high_noise']}",
                f"  Excessive artifacts: {quality_flags['excessive_artifacts']}",
                f"  Poor signal quality: {quality_flags['poor_signal_quality']}",
                f"  Irregular rhythm: {quality_flags['irregular_rhythm']}",
                "",
                "Key Results Comparison:",
                f"  LF/HF Ratio (Raw): {raw_results['lf_hf_ratio']:.3f}",
                f"  LF/HF Ratio (Prep): {prep_results['lf_hf_ratio']:.3f}",
                f"  Total Power (Raw): {raw_results['total_power']:.0f} ms²",
                f"  Total Power (Prep): {prep_results['total_power']:.0f} ms²",
            ]

            ax4.text(
                0.05,
                0.95,
                "\n".join(stats_text),
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(
            "hrv_freq_analysis_preprocessing_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            "Comprehensive validation plot saved as 'hrv_freq_analysis_preprocessing_comparison.png'"
        )
        print(f"\nPreprocessing Results:")
        if preprocessed_analyzer.preprocessing_result:
            print(
                f"  Artifacts detected: {preprocessed_analyzer.preprocessing_result.stats['artifacts_detected']}"
            )
            print(
                f"  Artifacts corrected: {preprocessed_analyzer.preprocessing_result.stats['artifacts_corrected']}"
            )
            print(
                f"  Artifact percentage: {preprocessed_analyzer.preprocessing_result.stats['artifact_percentage']:.1f}%"
            )
            print(
                f"  Quality issues: {any(preprocessed_analyzer.preprocessing_result.quality_flags.values())}"
            )

    def _add_frequency_bands(self, ax):
        """Helper to add frequency band shading to plots"""
        bands = {
            "ulf": (0.0, 0.003),
            "vlf": (0.0, 0.04),
            "lf": (0.04, 0.15),
            "hf": (0.15, 0.4),
        }
        colors = {"ulf": "purple", "vlf": "gray", "lf": "green", "hf": "red"}

        for band, color in colors.items():
            low, high = bands[band]
            ax.axvspan(low, high, alpha=0.1, color=color)

    def test_input_validation(self):
        """Test input validation for different RRI formats"""
        # Test with numpy array (should work)
        analyzer = HRVFreqDomainAnalysis(
            np.array(self.normal_preprocessing_result.corrected_rri)
        )
        self.assertIsInstance(analyzer.rr_intervals, np.ndarray)

        # Test with list (should work)
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri.tolist()
        )
        self.assertIsInstance(analyzer.rr_intervals, np.ndarray)

        # Test with invalid data types
        with self.assertRaises((ValueError, TypeError)):
            HRVFreqDomainAnalysis("invalid_input")

    def test_time_domain_signal_creation(self):
        """Test time domain signal creation from RRI"""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)

        # Should create time domain signal
        self.assertGreater(len(analyzer.time_domain), 0)

        # Check signal properties
        expected_duration = (
            np.sum(self.normal_preprocessing_result.corrected_rri) / 1000.0
        )
        actual_duration = len(analyzer.time_domain) / analyzer.sampling_rate

        # Allow some tolerance for interpolation effects
        self.assertAlmostEqual(expected_duration, actual_duration, delta=5.0)

    def test_welch_psd_computation(self):
        """Test Welch PSD computation"""
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri, segment_length=120.0
        )

        freqs, psd = analyzer.get_psd()

        # Should have valid frequency and PSD arrays
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
        self.assertEqual(len(freqs), len(psd))

        # PSD should be non-negative
        self.assertTrue(np.all(psd >= 0))

        # Frequencies should be monotonically increasing
        if len(freqs) > 1:
            self.assertTrue(np.all(np.diff(freqs) > 0))

    def test_edge_cases(self):
        """Test various edge cases"""
        # Very short RRI sequence
        very_short_rri = np.array([800.0, 820.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            analyzer = HRVFreqDomainAnalysis(very_short_rri)
            results = analyzer.get_results()

            # Should handle gracefully with warnings
            self.assertTrue(len(w) > 0)
            self.assertIsInstance(results, dict)

        # Single RRI value
        single_rri = np.array([800.0])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            analyzer = HRVFreqDomainAnalysis(single_rri)
            results = analyzer.get_results()

            # Should handle gracefully
            self.assertIsInstance(results, dict)

    def test_frequency_resolution_and_validation(self):
        """Test frequency resolution and validation methods"""
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            segment_length=240.0,  # Long segments for better resolution
        )

        validation = analyzer.validate_frequency_analysis()

        # Should have reasonable frequency resolution
        freq_res = validation["frequency_resolution_hz"]
        self.assertGreater(freq_res, 0)
        self.assertLess(freq_res, 0.1)  # Should be better than 0.1 Hz

        # Should validate signal duration
        duration = validation["signal_duration_s"]
        self.assertGreater(duration, 0)

        # Should provide validation status
        self.assertIsInstance(validation["is_valid"], bool)
        self.assertIsInstance(validation["warnings"], list)
        self.assertIsInstance(validation["recommendations"], list)

    def test_different_sampling_rates(self):
        """Test analysis with different sampling rates"""
        sampling_rates = [2.0, 4.0, 8.0]

        for fs in sampling_rates:
            analyzer = HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, sampling_rate=fs
            )

            results = analyzer.get_results()
            self.assertGreater(results["total_power"], 0)
            self.assertEqual(results["analysis_info"]["sampling_rate"], fs)

    def test_segment_length_effects(self):
        """Test effects of different segment lengths"""
        segment_lengths = [60.0, 120.0, 240.0]

        for seg_len in segment_lengths:
            analyzer = HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, segment_length=seg_len
            )

            results = analyzer.get_results()
            self.assertGreater(results["total_power"], 0)
            self.assertEqual(results["analysis_info"]["segment_length_s"], seg_len)

    def test_overlap_ratio_effects(self):
        """Test effects of different overlap ratios"""
        overlap_ratios = [0.0, 0.5, 0.75]

        for overlap in overlap_ratios:
            analyzer = HRVFreqDomainAnalysis(
                self.normal_preprocessing_result.corrected_rri, overlap_ratio=overlap
            )

            results = analyzer.get_results()
            self.assertGreater(results["total_power"], 0)
            self.assertEqual(results["analysis_info"]["overlap_ratio"], overlap)

    def test_frequency_diagnostics_only_when_enabled(self):
        """Frequency diagnostics should be opt-in only."""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)
        self.assertNotIn("frequency_diagnostics", analyzer.get_results())

        analyzer_with_diagnostics = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            enable_diagnostics=True,
        )
        results = analyzer_with_diagnostics.get_results()
        self.assertIn("frequency_diagnostics", results)
        self.assertIn("welch", results["frequency_diagnostics"])
        self.assertIn("fft", results["frequency_diagnostics"])
        self.assertIn("ar", results["frequency_diagnostics"])

    def test_frequency_diagnostics_do_not_change_existing_outputs(self):
        """Enabling diagnostics should not change existing metric outputs."""
        analyzer = HRVFreqDomainAnalysis(self.normal_preprocessing_result.corrected_rri)
        analyzer_with_diagnostics = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            enable_diagnostics=True,
        )

        results = analyzer.get_results()
        diagnostic_results = analyzer_with_diagnostics.get_results()

        for key, value in results.items():
            self.assertIn(key, diagnostic_results)
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(diagnostic_results[key]))
            else:
                self.assertEqual(diagnostic_results[key], value)

    def test_short_recording_frequency_diagnostics_warnings(self):
        """Short recordings should report duration and band reliability warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer = HRVFreqDomainAnalysis(
                self.short_rr_ms,
                enable_diagnostics=True,
            )

        diagnostics = analyzer.get_results()["frequency_diagnostics"]
        self.assertIn(
            "Frequency-domain HRV metrics may be unreliable.",
            diagnostics["duration_warnings"],
        )
        self.assertIn(
            "LF estimates should be interpreted cautiously.",
            diagnostics["duration_warnings"],
        )
        self.assertIn(
            "VLF estimates are likely unreliable.",
            diagnostics["duration_warnings"],
        )
        self.assertTrue(
            diagnostics["welch"]["band_diagnostics"]["lf"]["warnings"]
        )

    def test_ar_fallback_is_reported_in_diagnostics(self):
        """AR diagnostics should identify Welch fallback when AR cannot run."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer = HRVFreqDomainAnalysis(
                self.very_short_rr_ms,
                enable_diagnostics=True,
            )

        ar_diagnostics = analyzer.get_results()["frequency_diagnostics"]["ar"]
        self.assertEqual(ar_diagnostics["estimator_used"], "welch_fallback")
        self.assertIsNotNone(ar_diagnostics["fallback_reason"])

    def test_welch_effective_parameters_are_reported(self):
        """Welch diagnostics should expose the effective SciPy parameters."""
        analyzer = HRVFreqDomainAnalysis(
            self.normal_preprocessing_result.corrected_rri,
            sampling_rate=4.0,
            segment_length=120.0,
            overlap_ratio=0.75,
            enable_diagnostics=True,
        )

        welch = analyzer.get_results()["frequency_diagnostics"]["welch"]
        expected_nperseg = int(120.0 * 4.0)
        self.assertEqual(welch["requested_nperseg"], expected_nperseg)
        self.assertEqual(welch["effective_nperseg"], expected_nperseg)
        self.assertEqual(welch["effective_noverlap"], int(expected_nperseg * 0.75))
        self.assertEqual(welch["effective_nfft"], expected_nperseg)
        self.assertAlmostEqual(
            welch["frequency_resolution_hz"],
            4.0 / expected_nperseg,
        )
        self.assertFalse(welch["whether_welch_shorten_if_short_applied"])
        self.assertEqual(welch["number_of_resampled_samples"], len(analyzer.time_domain_s))
        self.assertGreaterEqual(welch["number_of_segments"], 1)

    def test_none_detrend_removes_welch_dc_mean(self):
        """None detrending should not preserve the RR mean as a giant Welch DC spike."""
        rr_ms = np.full(900, 800.0)
        analyzer = HRVFreqDomainAnalysis(
            rr_ms,
            detrend_method=None,
            segment_length=120.0,
            overlap_ratio=0.75,
            enable_diagnostics=True,
        )

        results = analyzer.get_results()
        self.assertLess(results["vlf_power"], 1e-6)
        self.assertLess(results["total_power"], 1e-6)

        diagnostics = results["frequency_diagnostics"]
        self.assertTrue(diagnostics["global_mean_removed_for_none_detrend"])
        self.assertTrue(diagnostics["mean_removed_for_none_detrend"])
        self.assertEqual(diagnostics["welch"]["detrend_method"], "global_mean_then_none")

    def test_none_detrend_matches_explicit_global_mean_removed_welch_scale(self):
        """None-mode Welch should match global mean removal followed by no segment detrend."""
        analyzer = HRVFreqDomainAnalysis(
            self.rr_intervals_ms,
            detrend_method=None,
            sampling_rate=4.0,
            segment_length=120.0,
            overlap_ratio=0.75,
        )
        nperseg = analyzer._welch_diagnostics["effective_nperseg"]
        noverlap = analyzer._welch_diagnostics["effective_noverlap"]
        freqs, psd_s2 = signal.welch(
            analyzer.time_domain_s - np.mean(analyzer.time_domain_s),
            fs=analyzer.sampling_rate,
            window=analyzer._get_window(nperseg),
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            average="mean",
        )

        # The native PSD itself is the most direct comparison.
        np.testing.assert_allclose(analyzer.freqs, freqs)
        np.testing.assert_allclose(analyzer.psd, psd_s2 * 1e6, rtol=1e-10, atol=1e-10)

    def test_linear_detrend_welch_behavior_matches_scipy_linear(self):
        """Linear detrending should continue to use segment-wise SciPy linear detrending."""
        analyzer = HRVFreqDomainAnalysis(
            self.rr_intervals_ms,
            detrend_method="linear",
            sampling_rate=4.0,
            segment_length=120.0,
            overlap_ratio=0.75,
        )
        nperseg = analyzer._welch_diagnostics["effective_nperseg"]
        noverlap = analyzer._welch_diagnostics["effective_noverlap"]
        freqs, psd_s2 = signal.welch(
            analyzer.time_domain_s,
            fs=analyzer.sampling_rate,
            window=analyzer._get_window(nperseg),
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="linear",
            scaling="density",
            average="mean",
        )

        np.testing.assert_allclose(analyzer.freqs, freqs)
        np.testing.assert_allclose(analyzer.psd, psd_s2 * 1e6, rtol=1e-10, atol=1e-10)
        self.assertEqual(analyzer._welch_diagnostics["detrend_method"], "linear")
        self.assertFalse(analyzer._welch_diagnostics["mean_removed_for_none_detrend"])

    def test_nonpositive_rr_intervals_are_removed_before_interpolation(self):
        """Nonpositive intervals should not create duplicate interpolation time points."""
        rr_ms = self.rr_intervals_ms.copy()
        rr_ms[10] = 0.0
        rr_ms[20] = -5.0
        rr_ms[30] = np.nan
        rr_ms[40] = np.inf

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyzer = HRVFreqDomainAnalysis(
                rr_ms,
                detrend_method=None,
                enable_diagnostics=True,
            )

        results = analyzer.get_results()
        diagnostics = results["frequency_diagnostics"]
        self.assertEqual(diagnostics["invalid_rr_removed_count"], 4)
        self.assertFalse(diagnostics["nonfinite_interpolated_signal"])
        self.assertTrue(np.all(np.isfinite(analyzer.time_domain_s)))
        self.assertGreater(results["total_power"], 0)
        self.assertFalse(
            results["vlf_power"] == 0.0
            and results["lf_power"] == 0.0
            and results["hf_power"] == 0.0
        )
        self.assertTrue(any("invalid RR intervals" in str(w.message) for w in caught))

    def test_nonfinite_psd_returns_nan_metrics_not_false_zeros(self):
        """Non-finite PSD values should not become all-zero physiological powers."""
        analyzer = HRVFreqDomainAnalysis(self.rr_intervals_ms)
        analyzer.freqs = np.array([0.0, 0.01, 0.04, 0.10, 0.20])
        analyzer.psd = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = analyzer._compute_spectral_metrics()

        self.assertTrue(np.isnan(metrics["vlf_power"]))
        self.assertTrue(np.isnan(metrics["lf_power"]))
        self.assertTrue(np.isnan(metrics["hf_power"]))
        self.assertTrue(np.isnan(metrics["total_power"]))
        self.assertTrue(any("non-finite" in str(w.message) for w in caught))


if __name__ == "__main__":
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHRVFreqDomainAnalysis)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Create validation plot if all tests pass
    if result.wasSuccessful():
        print("\nAll tests passed! Creating comprehensive validation plot...")
        test_instance = TestHRVFreqDomainAnalysis()
        test_instance.setUp()
        test_instance.create_comprehensive_validation_plot()
    else:
        print(
            f"\nTests completed with {len(result.failures)} failures and {len(result.errors)} errors."
        )
        if result.failures:
            print("Failures:")
            for test, traceback in result.failures:
                # Extract assertion message
                newline = "\n"
                assertion_part = traceback.split("AssertionError: ")[-1].split(newline)[
                    0
                ]
                print(f"  - {test}: {assertion_part}")
        if result.errors:
            print("Errors:")
            for test, traceback in result.errors:
                newline = "\n"
                error_part = traceback.split(newline)[-2]
                print(f"  - {test}: {error_part}")
