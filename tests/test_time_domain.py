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

        # Create mock preprocessing result
        self.mock_preprocessing_result = Mock(spec=PreprocessingResult)
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

    def test_initialization_with_clean_data(self):
        """Test initialization with clean preprocessed RRI data"""
        hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=self.clean_rr_ms, preprocessing_result=None
        )

        self.assertEqual(len(hrv.rr_ms), len(self.clean_rr_ms))
        np.testing.assert_array_equal(hrv.rr_ms, self.clean_rr_ms)
        self.assertIsNone(hrv.preprocessing_result)

    def test_initialization_with_preprocessing_result(self):
        """Test initialization with preprocessing result provided"""
        hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=self.clean_rr_ms,
            preprocessing_result=self.mock_preprocessing_result,
        )

        self.assertIsNotNone(hrv.preprocessing_result)
        np.testing.assert_array_equal(hrv.rr_ms, self.clean_rr_ms)

    def test_initialization_with_analysis_window(self):
        """Test initialization with analysis window"""
        # Create longer RR sequence for windowing
        long_rr = [1000] * 600  # 10 minutes at 1000ms intervals

        # Test 2-minute window starting at 1 minute
        hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=long_rr,
            preprocessing_result=None,
            analysis_window=(60.0, 180.0),  # 1-3 minutes
        )

        # Should have approximately 120 intervals (2 minutes at 1s intervals)
        self.assertGreater(len(hrv.rr_ms), 100)
        self.assertLess(len(hrv.rr_ms), 140)

    def test_initialization_errors(self):
        """Test initialization error conditions"""
        # Empty RRI data
        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis(preprocessed_rri=[])

        # Insufficient data
        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis(preprocessed_rri=[1000])

    def test_analysis_window_application(self):
        """Test analysis window application"""
        # Create RRI sequence where we know the timing
        rr_sequence = [1000] * 200  # 200 seconds of 1000ms intervals

        # Apply window from 50-150 seconds
        hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=rr_sequence, analysis_window=(50.0, 150.0)
        )

        # Should have approximately 100 intervals
        self.assertAlmostEqual(len(hrv.rr_ms), 100, delta=5)

    def test_analysis_window_no_data_error(self):
        """Test error when analysis window contains no data"""
        short_rr = [1000, 1000, 1000]  # 3 seconds total

        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis(
                preprocessed_rri=short_rr,
                analysis_window=(10.0, 20.0),  # Outside data range
            )

    def test_data_quality_validation_warnings(self):
        """Test data quality validation warnings"""
        # Test with poor quality preprocessing result
        poor_preprocessing = Mock(spec=PreprocessingResult)
        poor_preprocessing.stats = {
            "artifacts_detected": 50,
            "artifacts_corrected": 30,
            "artifact_percentage": 12.0,  # >10% triggers warning
        }
        poor_preprocessing.quality_flags = {
            "poor_signal_quality": True,
            "excessive_artifacts": True,
        }
        poor_preprocessing.noise_segments = [{"start": 0, "end": 10}]
        poor_preprocessing.correction_method = "interpolation"

        with patch("hrvlib.metrics.time_domain.warnings.warn") as mock_warn:
            hrv = HRVTimeDomainAnalysis(
                preprocessed_rri=self.clean_rr_ms,
                preprocessing_result=poor_preprocessing,
            )
            # Should have issued warnings for poor quality
            self.assertGreater(mock_warn.call_count, 0)

    def test_sdnn_calculation(self):
        """Test SDNN calculation"""
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)
        expected_sdnn = np.std(self.clean_rr_ms, ddof=1)

        self.assertAlmostEqual(hrv.sdnn(), expected_sdnn, places=6)

    def test_rmssd_calculation(self):
        """Test RMSSD calculation"""
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)

        # Manual RMSSD calculation
        diff = np.diff(self.clean_rr_ms)
        expected_rmssd = np.sqrt(np.mean(diff**2))

        self.assertAlmostEqual(hrv.rmssd(), expected_rmssd, places=6)

    def test_pnn50_calculation(self):
        """Test pNN50 calculation"""
        # Create test data with known differences
        test_rr = [1000, 1060, 940, 1000, 1120, 880]  # Diffs: 60, 120, 60, 120, 240

        hrv = HRVTimeDomainAnalysis(preprocessed_rri=test_rr)

        # Differences > 50ms: 60, 120, 60, 120, 240 = 5/5 = 100%
        self.assertEqual(hrv.pnn50(), 100.0)

    def test_pnn20_calculation(self):
        """Test pNN20 calculation"""
        # Create test data with known differences
        test_rr = [1000, 1030, 970, 1000, 1050, 950]  # Diffs: 30, 60, 30, 50, 100

        hrv = HRVTimeDomainAnalysis(preprocessed_rri=test_rr)

        # Differences > 20ms: 30, 60, 30, 50, 100 = 5/5 = 100%
        self.assertEqual(hrv.pnn20(), 100.0)

    def test_nn50_nn20_calculation(self):
        """Test NN50 and NN20 count calculations"""
        test_rr = [1000, 1030, 970, 1000]  # Diffs: 30, 60, 30

        hrv = HRVTimeDomainAnalysis(preprocessed_rri=test_rr)

        self.assertEqual(hrv.nn50(), 1)  # Only 60ms > 50ms
        self.assertEqual(hrv.nn20(), 3)  # All differences > 20ms

    def test_mean_and_median_calculations(self):
        """Test mean and median RR calculations"""
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)

        expected_mean = np.mean(self.clean_rr_ms)
        expected_median = np.median(self.clean_rr_ms)

        self.assertAlmostEqual(hrv.mean_rr(), expected_mean, places=6)
        self.assertAlmostEqual(hrv.median_rr(), expected_median, places=6)

    def test_heart_rate_calculations(self):
        """Test heart rate related calculations"""
        test_rr = [1000, 1000, 1000, 1000]  # 1000ms = 60 bpm

        hrv = HRVTimeDomainAnalysis(preprocessed_rri=test_rr)

        self.assertAlmostEqual(hrv.mean_hr(), 60.0, places=1)
        self.assertAlmostEqual(hrv.std_hr(), 0.0, places=6)

    def test_coefficient_of_variation_calculations(self):
        """Test CVNN and CVSD calculations"""
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)

        expected_cvnn = hrv.sdnn() / hrv.mean_rr()
        expected_cvsd = hrv.rmssd() / hrv.mean_rr()

        self.assertAlmostEqual(hrv.cvnn(), expected_cvnn, places=6)
        self.assertAlmostEqual(hrv.cvsd(), expected_cvsd, places=6)

    def test_geometric_measures(self):
        """Test geometric measures (HRV Triangular Index and TINN)"""
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)

        # These should return reasonable values
        tri_index = hrv.hrv_triangular_index()
        tinn_value = hrv.tinn()

        self.assertGreater(tri_index, 0)
        self.assertGreaterEqual(tinn_value, 0)

    def test_geometric_measures_edge_cases(self):
        """Test geometric measures with edge cases"""
        # Test with insufficient data
        short_rr = [1000, 1000]
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=short_rr)

        # Should handle gracefully
        self.assertGreaterEqual(hrv.hrv_triangular_index(), 0)
        self.assertEqual(hrv.tinn(), 0.0)  # TINN requires at least 3 intervals

    def test_full_analysis(self):
        """Test complete analysis output"""
        hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=self.clean_rr_ms,
            preprocessing_result=self.mock_preprocessing_result,
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

    def test_full_analysis_without_preprocessing(self):
        """Test full analysis without preprocessing result"""
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)
        results = hrv.full_analysis()

        # Should have analysis_info but no preprocessing_stats
        self.assertIn("analysis_info", results)
        self.assertNotIn("preprocessing_stats", results)

    def test_quality_assessment(self):
        """Test data quality assessment"""
        hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=self.clean_rr_ms,
            preprocessing_result=self.mock_preprocessing_result,
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

        hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=self.clean_rr_ms[:10],  # Short sequence
            preprocessing_result=poor_preprocessing,
        )
        assessment = hrv.get_quality_assessment()

        self.assertEqual(assessment["overall_quality"], "poor")
        self.assertGreater(len(assessment["recommendations"]), 0)

    def test_quality_assessment_without_preprocessing(self):
        """Test quality assessment without preprocessing result"""
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)
        assessment = hrv.get_quality_assessment()

        # Should still provide basic assessment
        self.assertEqual(assessment["overall_quality"], "unknown")
        self.assertIn("data_length_adequate", assessment)

    def test_normative_comparisons(self):
        """Test comparison with normative values"""
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)
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

        hrv = HRVTimeDomainAnalysis(preprocessed_rri=constant_rr)

        self.assertEqual(hrv.sdnn(), 0.0)
        self.assertEqual(hrv.rmssd(), 0.0)
        self.assertEqual(hrv.pnn50(), 0.0)
        self.assertEqual(hrv.nn50(), 0)

    def test_edge_cases_single_interval_functions(self):
        """Test functions with single interval"""
        single_rr = [1000, 1000]  # Minimum for diff-based calculations
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=single_rr)

        # These should work with 2 intervals
        self.assertEqual(hrv.rmssd(), 0.0)
        self.assertEqual(hrv.pnn50(), 0.0)
        self.assertEqual(hrv.nn50(), 0)

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

    def test_array_conversion(self):
        """Test that input RRI data is properly converted to numpy array"""
        # Test with list input
        hrv_list = HRVTimeDomainAnalysis(preprocessed_rri=self.clean_rr_ms)
        self.assertIsInstance(hrv_list.rr_ms, np.ndarray)

        # Test with numpy array input
        hrv_array = HRVTimeDomainAnalysis(preprocessed_rri=np.array(self.clean_rr_ms))
        self.assertIsInstance(hrv_array.rr_ms, np.ndarray)

        # Results should be identical
        self.assertAlmostEqual(hrv_list.sdnn(), hrv_array.sdnn(), places=10)


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
        hrv = HRVTimeDomainAnalysis(preprocessed_rri=self.realistic_rr.tolist())
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

    def test_analysis_window_with_realistic_data(self):
        """Test analysis window functionality with realistic data"""
        # Use first 5 minutes (300 intervals)
        windowed_hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=self.realistic_rr[:300].tolist(),
            analysis_window=(60.0, 180.0),  # 1-3 minutes
        )

        full_hrv = HRVTimeDomainAnalysis(
            preprocessed_rri=self.realistic_rr[:300].tolist()
        )

        # Windowed analysis should have fewer intervals
        self.assertLess(len(windowed_hrv.rr_ms), len(full_hrv.rr_ms))

        # But still produce valid results
        windowed_results = windowed_hrv.full_analysis()
        self.assertGreater(windowed_results["sdnn"], 0)

    def assertBetween(self, value, min_val, max_val):
        """Helper method to check if value is within range"""
        self.assertGreaterEqual(
            value, min_val, f"Value {value} below minimum {min_val}"
        )
        self.assertLessEqual(value, max_val, f"Value {value} above maximum {max_val}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(argv=[""], verbosity=2, exit=False)
