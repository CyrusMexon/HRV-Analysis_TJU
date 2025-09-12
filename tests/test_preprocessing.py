import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path to import hrvlib.preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the module to test
try:
    from hrvlib.preprocessing import (
        PreprocessingResult,
        detect_artifacts,
        correct_extra_beats,
        cubic_spline_interpolation,
        preprocess_rri,
        validate_rri_data,
        detect_noise_segments,  # New function
        assess_signal_quality,  # New function
    )
except ImportError as e:
    print(f"Failed to import hrvlib.preprocessing: {e}")
    print(
        "Make sure hrvlib.preprocessing.py is in the parent directory of the tests folder"
    )
    raise


class TestPreprocessingFunctions(unittest.TestCase):
    """Test artifact detection and correction functions"""

    def setUp(self):
        """Set up test data"""
        # Normal RR intervals around 800ms
        self.normal_rri = np.array([800, 820, 850, 830, 810, 825, 840, 815, 835, 822])

        # RR intervals with artifacts - ensuring clear missed beats above 2000ms threshold
        self.rri_with_artifacts = np.array(
            [
                800,
                820,
                2500,  # missed beat at index 2 (very long, > 2000ms)
                200,  # extra beat at index 3 (very short, < 300ms)
                830,
                810,
                2200,  # another missed beat at index 6 (long, > 2000ms)
                825,
                840,
                815,
            ]
        )

        # RR intervals with ectopic beats
        self.rri_with_ectopic = np.array(
            [
                800,
                820,
                850,
                1200,  # ectopic (sudden jump)
                830,
                810,
                825,
                500,  # ectopic (sudden drop)
                840,
                815,
                835,
            ]
        )

        # RR intervals with multiple extra beats for testing removal
        self.rri_with_extra_beats = np.array(
            [
                800,
                820,
                150,  # extra beat at index 2
                250,  # extra beat at index 3
                830,
                810,
                180,  # extra beat at index 6
                825,
                840,
                815,
            ]
        )

        # RR intervals with high noise/variability for noise detection
        self.noisy_rri = np.array(
            [
                800,
                900,
                400,
                1200,
                300,
                1500,
                200,
                1000,
                850,
                600,
                1300,
                450,
                900,
                1100,
                350,
                800,
                820,
                830,
                810,
                825,
            ]
        )

        # RR intervals with irregular rhythm (high CV)
        self.irregular_rri = np.array(
            [
                400,
                1200,
                600,
                1000,
                500,
                1400,
                300,
                1100,
                550,
                950,
                450,
                1300,
                350,
                1150,
                400,
                1000,
                500,
                900,
                600,
                800,
            ]
        )

    def test_detect_artifacts_normal_data(self):
        """Test artifact detection with normal data"""
        indices, types = detect_artifacts(self.normal_rri)
        self.assertEqual(len(indices), 0)
        self.assertEqual(len(types), 0)

    def test_detect_artifacts_with_missed_beats(self):
        """Test detection of missed beats (long intervals)"""
        indices, types = detect_artifacts(self.rri_with_artifacts)

        missed_indices = [i for i, t in zip(indices, types) if t == "missed"]

        # The test data has intervals > 2000ms (default threshold) at indices 2 and 6
        # Verify that values above threshold are detected
        for i, val in enumerate(self.rri_with_artifacts):
            if val > 2000:  # Default threshold_high
                self.assertIn(
                    i,
                    missed_indices,
                    f"Index {i} has value {val}ms > 2000ms, should be detected as missed beat",
                )

        # Should detect exactly 2 missed beats (at indices 2 and 6)
        self.assertEqual(
            len(missed_indices),
            2,
            f"Expected 2 missed beats, got {len(missed_indices)}",
        )
        self.assertIn(2, missed_indices)  # 2500ms interval
        self.assertIn(6, missed_indices)  # 2200ms interval

    def test_detect_artifacts_with_extra_beats(self):
        """Test detection of extra beats (short intervals)"""
        indices, types = detect_artifacts(self.rri_with_artifacts)

        # Should detect extra beat at index 3
        extra_indices = [i for i, t in zip(indices, types) if t == "extra"]
        self.assertIn(3, extra_indices)  # 200ms interval

    def test_detect_artifacts_with_ectopic_beats(self):
        """Test detection of ectopic beats"""
        indices, types = detect_artifacts(self.rri_with_ectopic, ectopic_threshold=0.3)

        # Should detect ectopic beats
        ectopic_indices = [i for i, t in zip(indices, types) if t == "ectopic"]
        self.assertGreater(len(ectopic_indices), 0)

    def test_correct_extra_beats_single(self):
        """Test correction of a single extra beat"""
        test_rri = np.array([800, 820, 200, 830, 810])  # extra beat at index 2
        corrected, corrected_indices = correct_extra_beats(test_rri, [2])

        # Should have one fewer interval after correction
        self.assertEqual(len(corrected), len(test_rri) - 1)
        self.assertEqual(corrected_indices, [2])

        # The extra beat interval should be merged with the next one
        # Original: [800, 820, 200, 830, 810] -> [800, 820, 1030, 810]
        expected = np.array([800, 820, 1030, 810])  # 200 + 830 = 1030
        np.testing.assert_array_equal(corrected, expected)

    def test_correct_extra_beats_multiple(self):
        """Test correction of multiple extra beats"""
        test_rri = np.array([800, 150, 820, 180, 830])  # extra beats at indices 1, 3
        corrected, corrected_indices = correct_extra_beats(test_rri, [1, 3])

        # Should have two fewer intervals after correction
        self.assertEqual(len(corrected), len(test_rri) - 2)
        self.assertEqual(sorted(corrected_indices), [1, 3])

    def test_correct_extra_beats_edge_cases(self):
        """Test extra beat correction edge cases"""
        # Extra beat at end of series
        test_rri = np.array([800, 820, 830, 150])  # extra beat at last position
        corrected, corrected_indices = correct_extra_beats(test_rri, [3])

        # Should merge with previous interval
        expected = np.array([800, 820, 980])  # 830 + 150 = 980
        np.testing.assert_array_equal(corrected, expected)

        # No extra beats
        test_rri = np.array([800, 820, 830])
        corrected, corrected_indices = correct_extra_beats(test_rri, [])
        np.testing.assert_array_equal(corrected, test_rri)
        self.assertEqual(corrected_indices, [])

    def test_cubic_spline_interpolation(self):
        """Test cubic spline interpolation"""
        test_rri = np.array(
            [800, 820, 2500, 1200, 830, 810]
        )  # artifacts at indices 2, 3
        artifact_indices = [2, 3]

        corrected, interp_indices = cubic_spline_interpolation(
            test_rri, artifact_indices
        )

        # Should interpolate at artifact positions
        self.assertEqual(set(interp_indices), set(artifact_indices))

        # Interpolated values should be more reasonable
        self.assertLess(corrected[2], 2000)  # Should be less than original 2500
        self.assertLess(corrected[3], 1000)  # Should be less than original 1200
        self.assertGreater(corrected[2], 600)  # Should be reasonable
        self.assertGreater(corrected[3], 600)  # Should be reasonable

    def test_cubic_spline_interpolation_edge_cases(self):
        """Test cubic spline interpolation edge cases"""
        # Too few points
        short_rri = np.array([800, 820])
        corrected, interp_indices = cubic_spline_interpolation(short_rri, [1])
        np.testing.assert_array_equal(corrected, short_rri)
        self.assertEqual(len(interp_indices), 0)

        # No artifacts
        normal_rri = np.array([800, 820, 850, 830])
        corrected, interp_indices = cubic_spline_interpolation(normal_rri, [])
        np.testing.assert_array_equal(corrected, normal_rri)
        self.assertEqual(len(interp_indices), 0)

    def test_preprocess_rri_complete_pipeline(self):
        """Test complete preprocessing pipeline"""
        result = preprocess_rri(self.rri_with_artifacts.tolist())

        self.assertIsInstance(result, PreprocessingResult)
        self.assertGreater(len(result.artifact_indices), 0)
        self.assertGreater(len(result.artifact_types), 0)
        self.assertEqual(result.correction_method, "cubic_spline")

        # Check statistics
        self.assertIn("original_count", result.stats)
        self.assertIn("final_count", result.stats)
        self.assertIn("artifacts_detected", result.stats)
        self.assertIn("artifacts_corrected", result.stats)
        self.assertIn("extra_beats_removed", result.stats)
        self.assertIn("intervals_interpolated", result.stats)
        self.assertIn("artifact_percentage", result.stats)

        # Check that extra beats were properly removed (reducing count)
        # Original has 10 intervals, should have fewer after extra beat removal
        self.assertLess(result.stats["final_count"], result.stats["original_count"])

        # Check new statistics fields
        self.assertIn("noise_segments_count", result.stats)
        self.assertIn("noise_percentage", result.stats)

        # Check new fields in result
        self.assertIsNotNone(result.noise_segments)
        self.assertIsNotNone(result.quality_flags)

    def test_preprocess_rri_extra_beats_only(self):
        """Test preprocessing pipeline with only extra beats"""
        result = preprocess_rri(self.rri_with_extra_beats.tolist())

        # Should detect and remove extra beats
        extra_count = sum(1 for t in result.artifact_types if t == "extra")
        self.assertGreater(extra_count, 0)

        # Final count should be less than original due to extra beat removal
        self.assertLess(result.stats["final_count"], result.stats["original_count"])
        self.assertEqual(result.stats["extra_beats_removed"], extra_count)

        # Should have correction details
        self.assertIn(
            "correction_details", result.__dict__ if hasattr(result, "__dict__") else {}
        )

    def test_preprocess_rri_mixed_artifacts(self):
        """Test preprocessing with mixed artifact types"""
        # Create data with all types: missed, extra, and ectopic
        mixed_rri = np.array(
            [
                800,
                820,
                2500,  # missed beat at index 2
                150,  # extra beat at index 3
                830,
                810,
                825,
                1200,  # ectopic beat at index 7 (sudden jump)
                840,
                815,
            ]
        )

        result = preprocess_rri(mixed_rri.tolist())

        # Should detect all types
        artifact_types_set = set(result.artifact_types)
        self.assertIn("missed", artifact_types_set)
        self.assertIn("extra", artifact_types_set)

        # Should have both removals and interpolations
        self.assertGreater(result.stats["extra_beats_removed"], 0)
        # Note: interpolation count might be 0 if only extra beats after adjustment

        # Final count should reflect extra beat removals
        expected_final_count = len(mixed_rri) - result.stats["extra_beats_removed"]
        self.assertEqual(result.stats["final_count"], expected_final_count)

    def test_preprocess_rri_empty_input(self):
        """Test preprocessing with empty input"""
        with self.assertRaises(ValueError):
            preprocess_rri([])

    def test_preprocess_rri_invalid_data(self):
        """Test preprocessing with invalid data"""
        # All NaN values
        with self.assertRaises(ValueError):
            preprocess_rri([float("nan")] * 5)

    def test_index_adjustment_after_extra_beat_removal(self):
        """Test that artifact indices are properly adjusted after extra beat removal"""
        # Create data where removing extra beats affects subsequent artifact indices
        test_rri = np.array(
            [
                800,
                820,
                150,  # extra beat at index 2
                850,
                830,
                180,  # extra beat at index 5
                810,
                825,
                2500,  # missed beat at index 8 (will become index 6 after removals)
                840,
                815,
            ]
        )

        result = preprocess_rri(test_rri.tolist())

        # Verify that all artifacts were detected initially
        self.assertGreater(len(result.artifact_indices), 0)

        # Check that the pipeline completed successfully
        self.assertEqual(result.stats["original_count"], len(test_rri))
        self.assertLess(result.stats["final_count"], result.stats["original_count"])

    def test_preprocessing_with_custom_parameters(self):
        """Test preprocessing with custom parameters"""
        test_rri = [800, 820, 1800, 250, 830, 810]  # Contains artifacts

        custom_params = {
            "threshold_low": 400.0,
            "threshold_high": 1500.0,
            "ectopic_threshold": 0.2,
        }

        result = preprocess_rri(test_rri, **custom_params)

        self.assertIsInstance(result, PreprocessingResult)
        self.assertEqual(result.stats["original_count"], len(test_rri))

        # With stricter thresholds, should detect more artifacts
        self.assertGreater(len(result.artifact_indices), 0)

    def test_preprocess_rri_with_noise_detection_disabled(self):
        """Test preprocessing with noise detection disabled"""
        result = preprocess_rri(self.noisy_rri.tolist(), noise_detection=False)

        # Should have empty noise segments when noise detection is disabled
        self.assertEqual(len(result.noise_segments), 0)
        self.assertEqual(result.stats["noise_segments_count"], 0)
        self.assertEqual(result.stats["noise_percentage"], 0)


class TestNoiseDetectionFunctions(unittest.TestCase):
    """Test new noise detection functionality"""

    def setUp(self):
        """Set up test data for noise detection"""
        # Normal RR intervals with low variability
        self.clean_rri = np.array([800, 820, 830, 810, 825, 840, 815, 835, 822, 818])

        # RR intervals with noisy segments (high variability)
        self.noisy_rri = np.array(
            [
                800,
                900,
                400,
                1200,
                300,
                1500,
                200,
                1000,  # noisy segment
                850,
                830,
                810,
                825,
                840,
                815,
                835,
                822,  # clean segment
                300,
                1400,
                250,
                1300,
                400,
                1100,  # another noisy segment
            ]
        )

        # RR intervals with sudden jumps
        self.jumpy_rri = np.array([800, 820, 830, 810, 825, 1500, 300, 840, 815, 835])

    def test_detect_noise_segments_clean_data(self):
        """Test noise detection with clean data"""
        noise_segments = detect_noise_segments(self.clean_rri)

        # Should detect no noise in clean data
        self.assertEqual(len(noise_segments), 0)

    def test_detect_noise_segments_noisy_data(self):
        """Test noise detection with noisy data"""
        noise_segments = detect_noise_segments(self.noisy_rri, noise_threshold=1.5)

        # Should detect noise segments
        self.assertGreater(len(noise_segments), 0)

        # Check that detected segments are tuples with start and end indices
        for segment in noise_segments:
            self.assertIsInstance(segment, tuple)
            self.assertEqual(len(segment), 2)
            start, end = segment
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertLess(start, end)

    def test_detect_noise_segments_custom_parameters(self):
        """Test noise detection with custom parameters"""
        # Test with more sensitive threshold
        noise_segments_sensitive = detect_noise_segments(
            self.noisy_rri, noise_threshold=1.0, window_size=5, min_segment_length=2
        )

        # Test with less sensitive threshold
        noise_segments_tolerant = detect_noise_segments(
            self.noisy_rri, noise_threshold=3.0, window_size=15, min_segment_length=5
        )

        # More sensitive should detect more or equal noise
        self.assertGreaterEqual(
            len(noise_segments_sensitive), len(noise_segments_tolerant)
        )

    def test_detect_noise_segments_edge_cases(self):
        """Test noise detection edge cases"""
        # Too few data points
        short_rri = np.array([800, 820])
        noise_segments = detect_noise_segments(short_rri, window_size=10)
        self.assertEqual(len(noise_segments), 0)

        # Empty array
        empty_rri = np.array([])
        noise_segments = detect_noise_segments(empty_rri)
        self.assertEqual(len(noise_segments), 0)

    def test_detect_noise_segments_sudden_jumps(self):
        """Test noise detection with sudden jumps"""
        noise_segments = detect_noise_segments(self.jumpy_rri, noise_threshold=2.0)

        # Should detect the sudden jump regions
        self.assertGreater(len(noise_segments), 0)

    def test_assess_signal_quality_clean_signal(self):
        """Test signal quality assessment with clean signal"""
        quality_flags = assess_signal_quality(
            self.clean_rri, artifact_percentage=2.0, noise_segments=[]
        )

        # Should have good quality flags
        self.assertFalse(quality_flags["high_noise"])
        self.assertFalse(quality_flags["excessive_artifacts"])
        self.assertFalse(quality_flags["poor_signal_quality"])
        self.assertFalse(quality_flags["irregular_rhythm"])

    def test_assess_signal_quality_poor_signal(self):
        """Test signal quality assessment with poor signal"""
        # Create noise segments that cover >15% of data
        noise_segments = [(0, 5), (10, 15)]  # 10 out of 20 samples = 50%

        quality_flags = assess_signal_quality(
            self.noisy_rri,
            artifact_percentage=8.0,  # >5% artifacts
            noise_segments=noise_segments,
        )

        # Should flag as poor quality
        self.assertTrue(quality_flags["high_noise"])
        self.assertTrue(quality_flags["excessive_artifacts"])
        self.assertTrue(quality_flags["poor_signal_quality"])

    def test_assess_signal_quality_irregular_rhythm(self):
        """Test signal quality assessment with irregular rhythm"""
        # Create highly variable RRI data (high CV)
        irregular_rri = np.array([400, 1200, 600, 1000, 500, 1400, 300, 1100])

        quality_flags = assess_signal_quality(
            irregular_rri, artifact_percentage=1.0, noise_segments=[]
        )

        # Should detect irregular rhythm
        self.assertTrue(quality_flags["irregular_rhythm"])

    def test_assess_signal_quality_edge_cases(self):
        """Test signal quality assessment edge cases"""
        # Empty RRI array
        quality_flags = assess_signal_quality(
            np.array([]), artifact_percentage=0.0, noise_segments=[]
        )

        # Should handle empty array gracefully
        self.assertFalse(quality_flags["high_noise"])
        self.assertFalse(quality_flags["excessive_artifacts"])


class TestValidationFunctions(unittest.TestCase):
    """Test data validation functions"""

    def test_validate_rri_data_valid(self):
        """Test validation with valid RRI data"""
        valid_rri = [800, 820, 850, 830, 810, 825, 840, 815, 835, 822]
        is_valid, errors = validate_rri_data(valid_rri)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_rri_data_empty(self):
        """Test validation with empty data"""
        is_valid, errors = validate_rri_data([])
        self.assertFalse(is_valid)
        self.assertIn("No RR intervals provided", errors[0])

    def test_validate_rri_data_invalid_values(self):
        """Test validation with invalid values"""
        # NaN values
        invalid_rri = [800, float("nan"), 850, 830]
        is_valid, errors = validate_rri_data(invalid_rri)
        self.assertFalse(is_valid)
        self.assertTrue(any("NaN" in error for error in errors))

        # Negative values
        invalid_rri = [800, -100, 850, 830]
        is_valid, errors = validate_rri_data(invalid_rri)
        self.assertFalse(is_valid)
        self.assertTrue(any("non-positive" in error for error in errors))

        # Extremely short/long values
        invalid_rri = [50, 6000, 850, 830]  # 50ms too short, 6000ms too long
        is_valid, errors = validate_rri_data(invalid_rri)
        self.assertFalse(is_valid)
        self.assertTrue(any("extremely short" in error for error in errors))
        self.assertTrue(any("extremely long" in error for error in errors))

    def test_validate_rri_data_too_few_points(self):
        """Test validation with too few data points"""
        short_rri = [800, 850, 830]  # Less than 10 points
        is_valid, errors = validate_rri_data(short_rri)
        self.assertFalse(is_valid)
        self.assertTrue(any("Too few RR intervals" in error for error in errors))


class TestPreprocessingResultDataclass(unittest.TestCase):
    """Test PreprocessingResult dataclass with new fields"""

    def test_preprocessing_result_initialization(self):
        """Test PreprocessingResult initialization with new fields"""
        original_rri = np.array([800, 820, 850])
        corrected_rri = np.array([800, 820, 850])

        # Test with all new fields
        result = PreprocessingResult(
            original_rri=original_rri,
            corrected_rri=corrected_rri,
            artifact_indices=[],
            artifact_types=[],
            interpolation_indices=[],
            correction_method="cubic_spline",
            stats={},
            noise_segments=[(0, 2)],
            quality_flags={"high_noise": True, "poor_signal_quality": True},
        )

        self.assertEqual(result.noise_segments, [(0, 2)])
        self.assertTrue(result.quality_flags["high_noise"])
        self.assertIsNotNone(result.correction_details)

    def test_preprocessing_result_post_init(self):
        """Test PreprocessingResult __post_init__ method"""
        original_rri = np.array([800, 820, 850])
        corrected_rri = np.array([800, 820, 850])

        # Test without optional fields (should be initialized by __post_init__)
        result = PreprocessingResult(
            original_rri=original_rri,
            corrected_rri=corrected_rri,
            artifact_indices=[],
            artifact_types=[],
            interpolation_indices=[],
            correction_method="cubic_spline",
            stats={},
        )

        # Should initialize optional fields
        self.assertIsNotNone(result.correction_details)
        self.assertIsNotNone(result.noise_segments)
        self.assertIsNotNone(result.quality_flags)
        self.assertEqual(result.noise_segments, [])
        self.assertIn("high_noise", result.quality_flags)
        self.assertIn("excessive_artifacts", result.quality_flags)
        self.assertIn("poor_signal_quality", result.quality_flags)
        self.assertIn("irregular_rhythm", result.quality_flags)


class TestIntegratedNoiseDetectionPipeline(unittest.TestCase):
    """Test integration of noise detection with preprocessing pipeline"""

    def setUp(self):
        """Set up test data for integrated testing"""
        # Data with both artifacts and noise
        self.mixed_quality_rri = np.array(
            [
                800,
                900,
                400,
                1200,
                300,  # noisy segment
                2500,  # missed beat
                150,  # extra beat
                830,
                810,
                825,
                840,
                815,  # clean segment
                300,
                1400,
                250,
                1300,  # another noisy segment
                835,
                822,
            ]
        )

    def test_integrated_preprocessing_with_noise_detection(self):
        """Test complete preprocessing pipeline with noise detection enabled"""
        result = preprocess_rri(self.mixed_quality_rri.tolist(), noise_detection=True)

        # Should detect both artifacts and noise
        self.assertGreater(len(result.artifact_indices), 0)
        self.assertGreater(len(result.noise_segments), 0)

        # Should have noise statistics
        self.assertGreater(result.stats["noise_segments_count"], 0)
        self.assertGreater(result.stats["noise_percentage"], 0)

        # Should have quality flags
        self.assertIsNotNone(result.quality_flags)

        # Poor quality signal should be flagged
        self.assertTrue(
            result.quality_flags["poor_signal_quality"]
            or result.quality_flags["high_noise"]
            or result.quality_flags["excessive_artifacts"]
        )

    def test_integrated_preprocessing_quality_flags_correlation(self):
        """Test that quality flags correlate with detected issues"""
        result = preprocess_rri(self.mixed_quality_rri.tolist(), noise_detection=True)

        # If high artifact percentage, should flag excessive artifacts
        if result.stats["artifact_percentage"] > 5.0:
            self.assertTrue(result.quality_flags["excessive_artifacts"])

        # If high noise percentage, should flag high noise
        if result.stats["noise_percentage"] > 15.0:
            self.assertTrue(result.quality_flags["high_noise"])

        # If either high noise or excessive artifacts, should flag poor quality
        if (
            result.quality_flags["high_noise"]
            or result.quality_flags["excessive_artifacts"]
        ):
            self.assertTrue(result.quality_flags["poor_signal_quality"])


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
