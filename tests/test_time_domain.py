import pytest
import unittest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules under test
from hrvlib.metrics.time_domain import (
    HRVTimeDomainAnalysis,
    create_time_domain_analysis,
    validate_rr_intervals_for_time_domain,
)
from hrvlib.data_handler import DataBundle, TimeSeries
from hrvlib.preprocessing import PreprocessingResult


class TestHRVTimeDomainAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up test data and mock objects"""
        # Generate synthetic RR intervals (5-minute recording, ~60 bpm)
        np.random.seed(42)
        self.base_rr_ms = np.random.normal(loc=1000, scale=50, size=300).tolist()

        # Add some artifacts for testing preprocessing
        self.rr_with_artifacts = self.base_rr_ms.copy()
        self.rr_with_artifacts[10] = 2000  # Long interval
        self.rr_with_artifacts[20] = 500  # Short interval
        self.rr_with_artifacts[30] = 3000  # Extreme value

        # Clean test data - small for specific metric testing
        self.clean_rr_ms = [800, 900, 1000, 1100, 900, 950, 1050, 975, 1025, 990]

        # Longer clean data for validation testing (meets minimum requirements)
        self.long_clean_rr_ms = [
            1000
        ] * 150  # 150 intervals = 150 seconds = 2.5 minutes

        # Create mock DataBundle with clean data
        self.mock_bundle_clean = Mock(spec=DataBundle)
        self.mock_bundle_clean.rri_ms = self.clean_rr_ms
        self.mock_bundle_clean.preprocessing = None

        # Create mock DataBundle with artifacts
        self.mock_bundle_artifacts = Mock(spec=DataBundle)
        self.mock_bundle_artifacts.rri_ms = self.rr_with_artifacts
        self.mock_bundle_artifacts.preprocessing = None

        # Create mock preprocessing result
        self.mock_preprocessing_result = Mock(spec=PreprocessingResult)
        self.mock_preprocessing_result.corrected_rri = np.array(self.clean_rr_ms)
        self.mock_preprocessing_result.stats = {
            "artifacts_detected": 5,
            "artifacts_corrected": 3,
            "artifact_percentage": 1.7,
        }
        self.mock_preprocessing_result.quality_flags = {
            "poor_signal_quality": False,
            "excessive_artifacts": False,
        }
        self.mock_preprocessing_result.noise_segments = []
        self.mock_preprocessing_result.correction_method = "interpolation"

        # Create mock DataBundle with preprocessing
        self.mock_bundle_preprocessed = Mock(spec=DataBundle)
        self.mock_bundle_preprocessed.rri_ms = self.rr_with_artifacts
        self.mock_bundle_preprocessed.preprocessing = self.mock_preprocessing_result

    def test_initialization_with_clean_data(self):
        """Test initialization with clean DataBundle"""
        hrv = HRVTimeDomainAnalysis(self.mock_bundle_clean, use_preprocessing=False)

        self.assertEqual(len(hrv.rr_ms), len(self.clean_rr_ms))
        np.testing.assert_array_equal(hrv.rr_ms, self.clean_rr_ms)
        self.assertIsNone(hrv.preprocessing_result)

    def test_initialization_with_preprocessing(self):
        """Test initialization with preprocessing enabled"""
        with patch("hrvlib.metrics.time_domain.preprocess_rri") as mock_preprocess:
            mock_preprocess.return_value = self.mock_preprocessing_result

            hrv = HRVTimeDomainAnalysis(
                self.mock_bundle_artifacts, use_preprocessing=True
            )

            mock_preprocess.assert_called_once()
            self.assertIsNotNone(hrv.preprocessing_result)
            np.testing.assert_array_equal(hrv.rr_ms, self.clean_rr_ms)

    def test_initialization_with_existing_preprocessing(self):
        """Test initialization with existing preprocessing results"""
        hrv = HRVTimeDomainAnalysis(
            self.mock_bundle_preprocessed, use_preprocessing=True
        )

        self.assertIsNotNone(hrv.preprocessing_result)
        np.testing.assert_array_equal(hrv.rr_ms, self.clean_rr_ms)

    def test_initialization_with_analysis_window(self):
        """Test initialization with analysis window"""
        # Create longer RR sequence for windowing
        long_rr = [1000] * 600  # 10 minutes at 1000ms intervals
        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = long_rr
        mock_bundle.preprocessing = None

        # Test 2-minute window starting at 1 minute
        hrv = HRVTimeDomainAnalysis(
            mock_bundle,
            use_preprocessing=False,
            analysis_window=(60.0, 180.0),  # 1-3 minutes
        )

        # Should have approximately 120 intervals (2 minutes at 1s intervals)
        self.assertGreater(len(hrv.rr_ms), 100)
        self.assertLess(len(hrv.rr_ms), 140)

    def test_initialization_errors(self):
        """Test initialization error conditions"""
        # Empty DataBundle
        empty_bundle = Mock(spec=DataBundle)
        empty_bundle.rri_ms = None
        empty_bundle.preprocessing = None

        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis(empty_bundle)

        # Insufficient data
        short_bundle = Mock(spec=DataBundle)
        short_bundle.rri_ms = [1000]
        short_bundle.preprocessing = None

        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis(short_bundle, use_preprocessing=False)

    def test_sdnn_calculation(self):
        """Test SDNN calculation"""
        hrv = HRVTimeDomainAnalysis(self.mock_bundle_clean, use_preprocessing=False)
        expected_sdnn = np.std(self.clean_rr_ms, ddof=1)

        self.assertAlmostEqual(hrv.sdnn(), expected_sdnn, places=6)

    def test_rmssd_calculation(self):
        """Test RMSSD calculation"""
        hrv = HRVTimeDomainAnalysis(self.mock_bundle_clean, use_preprocessing=False)

        # Manual RMSSD calculation
        diff = np.diff(self.clean_rr_ms)
        expected_rmssd = np.sqrt(np.mean(diff**2))

        self.assertAlmostEqual(hrv.rmssd(), expected_rmssd, places=6)

    def test_pnn50_calculation(self):
        """Test pNN50 calculation"""
        # Create test data with known differences
        test_rr = [1000, 1060, 940, 1000, 1120, 880]  # Diffs: 60, 120, 60, 120, 240
        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = test_rr
        mock_bundle.preprocessing = None

        hrv = HRVTimeDomainAnalysis(mock_bundle, use_preprocessing=False)

        # Differences > 50ms: 60, 120, 60, 120, 240 = 5/5 = 100%
        self.assertEqual(hrv.pnn50(), 100.0)

    def test_pnn20_calculation(self):
        """Test pNN20 calculation"""
        # Create test data with known differences
        test_rr = [1000, 1030, 970, 1000, 1050, 950]  # Diffs: 30, 60, 30, 50, 100
        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = test_rr
        mock_bundle.preprocessing = None

        hrv = HRVTimeDomainAnalysis(mock_bundle, use_preprocessing=False)

        # Differences > 20ms: 30, 60, 30, 50, 100 = 5/5 = 100%
        self.assertEqual(hrv.pnn20(), 100.0)

    def test_nn50_nn20_calculation(self):
        """Test NN50 and NN20 count calculations"""
        test_rr = [1000, 1030, 970, 1000]  # Diffs: 30, 60, 30
        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = test_rr
        mock_bundle.preprocessing = None

        hrv = HRVTimeDomainAnalysis(mock_bundle, use_preprocessing=False)

        self.assertEqual(hrv.nn50(), 1)  # Only 60ms > 50ms
        self.assertEqual(hrv.nn20(), 3)  # All differences > 20ms

    def test_mean_and_median_calculations(self):
        """Test mean and median RR calculations"""
        hrv = HRVTimeDomainAnalysis(self.mock_bundle_clean, use_preprocessing=False)

        expected_mean = np.mean(self.clean_rr_ms)
        expected_median = np.median(self.clean_rr_ms)

        self.assertAlmostEqual(hrv.mean_rr(), expected_mean, places=6)
        self.assertAlmostEqual(hrv.median_rr(), expected_median, places=6)

    def test_heart_rate_calculations(self):
        """Test heart rate related calculations"""
        test_rr = [1000, 1000, 1000, 1000]  # 1000ms = 60 bpm
        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = test_rr
        mock_bundle.preprocessing = None

        hrv = HRVTimeDomainAnalysis(mock_bundle, use_preprocessing=False)

        self.assertAlmostEqual(hrv.mean_hr(), 60.0, places=1)
        self.assertAlmostEqual(hrv.std_hr(), 0.0, places=6)

    def test_coefficient_of_variation_calculations(self):
        """Test CVNN and CVSD calculations"""
        hrv = HRVTimeDomainAnalysis(self.mock_bundle_clean, use_preprocessing=False)

        expected_cvnn = hrv.sdnn() / hrv.mean_rr()
        expected_cvsd = hrv.rmssd() / hrv.mean_rr()

        self.assertAlmostEqual(hrv.cvnn(), expected_cvnn, places=6)
        self.assertAlmostEqual(hrv.cvsd(), expected_cvsd, places=6)

    def test_geometric_measures(self):
        """Test geometric measures (HRV Triangular Index and TINN)"""
        hrv = HRVTimeDomainAnalysis(self.mock_bundle_clean, use_preprocessing=False)

        # These should return reasonable values
        tri_index = hrv.hrv_triangular_index()
        tinn_value = hrv.tinn()

        self.assertGreater(tri_index, 0)
        self.assertGreaterEqual(tinn_value, 0)

    def test_full_analysis(self):
        """Test complete analysis output"""
        hrv = HRVTimeDomainAnalysis(
            self.mock_bundle_preprocessed, use_preprocessing=True
        )
        results = hrv.full_analysis()

        # Check all expected metrics are present
        expected_metrics = [
            "sdnn",
            "rmssd",
            "pnn50",
            "pnn20",
            "nn50",
            "nn20",
            "mean_rr",
            "median_rr",
            "mean_hr",
            "std_hr",
            "cvnn",
            "cvsd",
            "hrv_triangular_index",
            "tinn",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (int, float))

        # Check analysis metadata
        self.assertIn("analysis_info", results)
        self.assertIn("preprocessing_stats", results)

        # Verify consistency with individual method calls
        self.assertAlmostEqual(results["sdnn"], hrv.sdnn(), places=6)
        self.assertAlmostEqual(results["rmssd"], hrv.rmssd(), places=6)

    def test_quality_assessment(self):
        """Test data quality assessment"""
        hrv = HRVTimeDomainAnalysis(
            self.mock_bundle_preprocessed, use_preprocessing=True
        )
        assessment = hrv.get_quality_assessment()

        # Check required fields
        self.assertIn("overall_quality", assessment)
        self.assertIn("data_length_adequate", assessment)
        self.assertIn("duration_adequate_s", assessment)
        self.assertIn("recommendations", assessment)

        # Quality should be good with our mock data
        self.assertEqual(assessment["overall_quality"], "good")

    def test_quality_assessment_poor_conditions(self):
        """Test quality assessment with poor conditions"""
        # Create preprocessing result with poor quality flags
        poor_preprocessing = Mock(spec=PreprocessingResult)
        poor_preprocessing.corrected_rri = np.array(
            self.clean_rr_ms[:10]
        )  # Short sequence
        poor_preprocessing.stats = {
            "artifacts_detected": 50,
            "artifacts_corrected": 30,
            "artifact_percentage": 15.0,  # High artifact percentage
        }
        poor_preprocessing.quality_flags = {
            "poor_signal_quality": True,
            "excessive_artifacts": True,
        }
        poor_preprocessing.noise_segments = [{"start": 0, "end": 10}]
        poor_preprocessing.correction_method = "interpolation"

        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = self.clean_rr_ms[:10]
        mock_bundle.preprocessing = poor_preprocessing

        hrv = HRVTimeDomainAnalysis(mock_bundle, use_preprocessing=True)
        assessment = hrv.get_quality_assessment()

        self.assertEqual(assessment["overall_quality"], "poor")
        self.assertGreater(len(assessment["recommendations"]), 0)

    def test_normative_comparisons(self):
        """Test comparison with normative values"""
        hrv = HRVTimeDomainAnalysis(self.mock_bundle_clean, use_preprocessing=False)
        comparisons = hrv.compare_with_norms()

        # Should contain comparisons for main metrics
        expected_metrics = ["sdnn", "rmssd", "pnn50", "mean_hr", "hrv_triangular_index"]

        for metric in expected_metrics:
            self.assertIn(metric, comparisons)
            self.assertIn("value", comparisons[metric])
            self.assertIn("interpretation", comparisons[metric])
            self.assertIn("normal_range", comparisons[metric])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Constant RR intervals
        constant_rr = [1000] * 10
        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = constant_rr
        mock_bundle.preprocessing = None

        hrv = HRVTimeDomainAnalysis(mock_bundle, use_preprocessing=False)

        self.assertEqual(hrv.sdnn(), 0.0)
        self.assertEqual(hrv.rmssd(), 0.0)
        self.assertEqual(hrv.pnn50(), 0.0)
        self.assertEqual(hrv.nn50(), 0)

    def test_factory_function(self):
        """Test the factory function"""
        hrv = create_time_domain_analysis(
            self.mock_bundle_clean, use_preprocessing=False
        )

        self.assertIsInstance(hrv, HRVTimeDomainAnalysis)
        self.assertEqual(len(hrv.rr_ms), len(self.clean_rr_ms))

    def test_validation_function(self):
        """Test RR interval validation function"""
        # Valid data - use longer sequence that meets minimum requirements
        is_valid, errors = validate_rr_intervals_for_time_domain(self.long_clean_rr_ms)
        self.assertTrue(
            is_valid, f"Expected long valid data to pass validation. Errors: {errors}"
        )
        self.assertEqual(len(errors), 0)

        # Test with shorter data that should fail validation
        is_valid, errors = validate_rr_intervals_for_time_domain(self.clean_rr_ms)
        # This should fail because it doesn't meet minimum requirements
        self.assertFalse(is_valid, "Short data should fail validation")
        self.assertGreater(len(errors), 0)

        # Invalid data - too short
        is_valid, errors = validate_rr_intervals_for_time_domain([1000, 1000])
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

        # Invalid data - negative values
        is_valid, errors = validate_rr_intervals_for_time_domain([1000, -500, 800])
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_analysis_window_edge_cases(self):
        """Test analysis window edge cases"""
        # Window outside data range
        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = [1000, 1000, 1000]  # 3 seconds total
        mock_bundle.preprocessing = None

        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis(
                mock_bundle,
                use_preprocessing=False,
                analysis_window=(10.0, 20.0),  # Outside data range
            )

    def test_preprocessing_failure_fallback(self):
        """Test fallback to raw data when preprocessing fails"""
        with patch("hrvlib.metrics.time_domain.preprocess_rri") as mock_preprocess:
            mock_preprocess.side_effect = Exception("Preprocessing failed")

            with patch("hrvlib.metrics.time_domain.warnings.warn") as mock_warn:
                hrv = HRVTimeDomainAnalysis(
                    self.mock_bundle_artifacts, use_preprocessing=True
                )

                mock_warn.assert_called()
                # Should fall back to raw data
                np.testing.assert_array_equal(hrv.rr_ms, self.rr_with_artifacts)

    def test_warnings_for_poor_quality(self):
        """Test that warnings are issued for poor quality data"""
        # Create preprocessing result that triggers warnings
        warning_preprocessing = Mock(spec=PreprocessingResult)
        warning_preprocessing.corrected_rri = np.array(self.clean_rr_ms)
        warning_preprocessing.stats = {
            "artifacts_detected": 50,
            "artifacts_corrected": 30,
            "artifact_percentage": 12.0,  # >10% triggers warning
        }
        warning_preprocessing.quality_flags = {
            "poor_signal_quality": False,
            "excessive_artifacts": True,
        }
        warning_preprocessing.noise_segments = []
        warning_preprocessing.correction_method = "interpolation"

        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = self.clean_rr_ms
        mock_bundle.preprocessing = warning_preprocessing

        with patch("hrvlib.metrics.time_domain.warnings.warn") as mock_warn:
            hrv = HRVTimeDomainAnalysis(mock_bundle, use_preprocessing=True)
            mock_warn.assert_called()


class TestTimeDomainIntegration(unittest.TestCase):
    """Integration tests for time domain analysis with real-like scenarios"""

    def setUp(self):
        """Set up realistic test scenarios"""
        np.random.seed(123)

        # Generate realistic RR intervals with circadian variation
        base_rr = 800  # Base RR interval (75 bpm)
        n_intervals = 1000

        # Add realistic HRV components
        trend = np.linspace(0, 50, n_intervals)  # Slow trend
        respiratory = 30 * np.sin(2 * np.pi * np.arange(n_intervals) / 15)  # RSA
        noise = np.random.normal(0, 20, n_intervals)  # Random variation

        self.realistic_rr = base_rr + trend + respiratory + noise
        self.realistic_rr = np.maximum(self.realistic_rr, 400)  # Physiological minimum
        self.realistic_rr = np.minimum(self.realistic_rr, 1200)  # Physiological maximum

    def test_realistic_data_analysis(self):
        """Test analysis with realistic RR interval data"""
        mock_bundle = Mock(spec=DataBundle)
        mock_bundle.rri_ms = self.realistic_rr.tolist()
        mock_bundle.preprocessing = None

        hrv = HRVTimeDomainAnalysis(mock_bundle, use_preprocessing=False)
        results = hrv.full_analysis()

        # Check that results are in physiologically reasonable ranges
        self.assertBetween(results["sdnn"], 10, 200)  # Typical SDNN range
        self.assertBetween(results["rmssd"], 5, 150)  # Typical RMSSD range
        self.assertBetween(results["mean_hr"], 50, 120)  # Typical HR range
        self.assertBetween(results["pnn50"], 0, 60)  # Typical pNN50 range

        # Ensure no NaN or infinite values
        for key, value in results.items():
            if isinstance(value, (int, float)):
                self.assertFalse(np.isnan(value), f"{key} is NaN")
                self.assertFalse(np.isinf(value), f"{key} is infinite")

    def assertBetween(self, value, min_val, max_val):
        """Helper method to check if value is within range"""
        self.assertGreaterEqual(
            value, min_val, f"Value {value} below minimum {min_val}"
        )
        self.assertLessEqual(value, max_val, f"Value {value} above maximum {max_val}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(argv=[""], verbosity=2, exit=False)
