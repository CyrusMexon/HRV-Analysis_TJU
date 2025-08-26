import unittest
import numpy as np
import tempfile
import os
import sys
import json
import csv
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path to import hrvlib.data_handler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the module to test
try:
    from hrvlib.data_handler import (
        DataBundle,
        TimeSeries,
        SourceInfo,
        PreprocessingResult,
        detect_artifacts,
        correct_extra_beats,
        cubic_spline_interpolation,
        preprocess_rri,
        validate_rri_data,
        load_rr_file,
        _coerce_float_list,
        _label_to_kind,
        set_interactive_mode,
        get_supported_formats,
        check_dependencies,
    )
except ImportError as e:
    print(f"Failed to import hrvlib.data_handler: {e}")
    print(
        "Make sure hrvlib.data_handler.py is in the parent directory of the tests folder"
    )
    raise


class TestDataStructures(unittest.TestCase):
    """Test data structure classes"""

    def test_time_series_creation(self):
        """Test TimeSeries dataclass"""
        data = np.array([1, 2, 3, 4, 5])
        ts = TimeSeries(name="ECG", data=data, fs=1000.0, units="mV")
        self.assertEqual(ts.name, "ECG")
        self.assertEqual(ts.fs, 1000.0)
        self.assertEqual(ts.units, "mV")
        np.testing.assert_array_equal(ts.data, data)

    def test_source_info_creation(self):
        """Test SourceInfo dataclass"""
        source = SourceInfo(path="/test/path.csv", filetype=".csv", device="TestDevice")
        self.assertEqual(source.path, "/test/path.csv")
        self.assertEqual(source.filetype, ".csv")
        self.assertEqual(source.device, "TestDevice")

    def test_data_bundle_summary(self):
        """Test DataBundle summary method"""
        # Create test data
        rri = [800, 850, 900, 750, 820]
        bundle = DataBundle(rri_ms=rri)

        summary = bundle.summary()
        self.assertIn("RRI", summary["types_loaded"])
        self.assertEqual(summary["rri_overview"]["n_RRI"], 5)
        self.assertAlmostEqual(summary["rri_overview"]["mean_RRI_ms"], 824.0)


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


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions"""

    def test_coerce_float_list(self):
        """Test _coerce_float_list function"""
        mixed_input = [1, "2.5", None, float("nan"), 4.7, "invalid"]
        result = _coerce_float_list(mixed_input)
        expected = [1.0, 2.5, 4.7]
        self.assertEqual(result, expected)

    def test_label_to_kind(self):
        """Test _label_to_kind function"""
        # ECG labels
        self.assertEqual(_label_to_kind("ECG_I"), "ECG")
        self.assertEqual(_label_to_kind("Lead II"), "ECG")
        self.assertEqual(_label_to_kind("ml-ii"), "ECG")

        # PPG labels
        self.assertEqual(_label_to_kind("PPG"), "PPG")
        self.assertEqual(_label_to_kind("Pulse"), "PPG")
        self.assertEqual(_label_to_kind("Plethysmograph"), "PPG")

        # RESP labels
        self.assertEqual(_label_to_kind("Respiration"), "RESP")
        self.assertEqual(_label_to_kind("Thoracic"), "RESP")
        self.assertEqual(_label_to_kind("Breath"), "RESP")

        # Unknown labels
        self.assertIsNone(_label_to_kind("Unknown"))
        self.assertIsNone(_label_to_kind(""))
        self.assertIsNone(_label_to_kind(None))


class TestFileLoading(unittest.TestCase):
    """Test file loading functionality"""

    def setUp(self):
        """Set up temporary files for testing"""
        self.temp_dir = tempfile.mkdtemp()
        # Ensure we have exactly 10 test values
        self.test_rri = [800, 820, 850, 830, 810, 825, 840, 815, 835, 822]

        # Verify we have 10 values
        assert (
            len(self.test_rri) == 10
        ), f"Test data should have 10 values, got {len(self.test_rri)}"

    def tearDown(self):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_csv_file(self):
        """Test loading CSV file with RRI data"""
        csv_path = os.path.join(self.temp_dir, "test.csv")

        # Create test CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RRI"])
            for rri in self.test_rri:
                writer.writerow([rri])

        # Test loading
        with patch("hrvlib.data_handler.pd") as mock_pd:
            # Mock pandas to be available
            mock_df = MagicMock()
            mock_df.columns = ["RRI"]
            mock_df.__getitem__.return_value.tolist.return_value = self.test_rri
            mock_pd.read_csv.return_value = mock_df
            mock_pd.read_csv = MagicMock(return_value=mock_df)

            bundle = load_rr_file(csv_path)
            self.assertEqual(len(bundle.rri_ms), len(self.test_rri))

    def test_load_txt_file(self):
        """Test loading TXT file with RRI data"""
        txt_path = os.path.join(self.temp_dir, "test.txt")

        # Create test TXT with one number per line
        with open(txt_path, "w") as f:
            for i, rri in enumerate(self.test_rri):
                f.write(f"{rri}\n")

        # Verify file was written correctly
        with open(txt_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Written {len(self.test_rri)} values to file")
        print(f"File has {len(lines)} non-empty lines")
        print(f"Original data: {self.test_rri}")
        print(f"File lines: {lines}")

        # Force use of manual TXT parsing by temporarily disabling pandas
        with patch("hrvlib.data_handler.pd", None):
            bundle = load_rr_file(txt_path)

        print(f"Loaded data: {bundle.rri_ms}")
        print(f"Loaded count: {len(bundle.rri_ms)}")

        # The core assertion
        self.assertEqual(
            len(bundle.rri_ms),
            len(self.test_rri),
            f"Expected {len(self.test_rri)} values, got {len(bundle.rri_ms)}",
        )

        # Check that the values are approximately correct (allowing for float precision)
        for i, (expected, actual) in enumerate(zip(self.test_rri, bundle.rri_ms)):
            self.assertAlmostEqual(
                actual,
                expected,
                places=1,
                msg=f"Value mismatch at index {i}: expected {expected}, got {actual}",
            )

    def test_load_txt_file_with_pandas(self):
        """Test loading TXT file using pandas path"""
        txt_path = os.path.join(self.temp_dir, "test_pandas.txt")

        # Create test TXT
        with open(txt_path, "w") as f:
            f.write("RRI\n")  # Header
            for rri in self.test_rri:
                f.write(f"{rri}\n")

        # This should use pandas if available
        bundle = load_rr_file(txt_path)

        self.assertEqual(len(bundle.rri_ms), len(self.test_rri))

    def test_load_json_file(self):
        """Test loading JSON file (Movesense format)"""
        json_path = os.path.join(self.temp_dir, "test.json")

        # Create test JSON
        test_data = {"RR": self.test_rri}
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        bundle = load_rr_file(json_path)
        self.assertEqual(len(bundle.rri_ms), len(self.test_rri))

    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        with self.assertRaises(FileNotFoundError):
            load_rr_file("/nonexistent/path.csv")

    def test_load_unsupported_format(self):
        """Test loading unsupported file format"""
        unsupported_path = os.path.join(self.temp_dir, "test.xyz")

        # Create empty file
        with open(unsupported_path, "w") as f:
            f.write("dummy content")

        with self.assertRaises(ValueError):
            load_rr_file(unsupported_path)


class TestConfigurationFunctions(unittest.TestCase):
    """Test configuration and utility functions"""

    def test_set_interactive_mode(self):
        """Test setting interactive mode"""
        # Test setting to True
        set_interactive_mode(True)
        from hrvlib.data_handler import INTERACTIVE_MODE

        self.assertTrue(INTERACTIVE_MODE)

        # Test setting to False
        set_interactive_mode(False)
        from hrvlib.data_handler import INTERACTIVE_MODE

        self.assertFalse(INTERACTIVE_MODE)

    def test_get_supported_formats(self):
        """Test getting supported file formats"""
        formats = get_supported_formats()
        expected_formats = [
            ".csv",
            ".txt",
            ".edf",
            ".hrm",
            ".fit",
            ".sml",
            ".json",
            ".acq",
        ]
        self.assertEqual(set(formats), set(expected_formats))

    def test_check_dependencies(self):
        """Test checking dependencies"""
        deps = check_dependencies()

        # These should always be True
        self.assertTrue(deps["numpy"])
        self.assertTrue(deps["scipy"])

        # These depend on availability
        self.assertIsInstance(deps["pandas"], bool)
        self.assertIsInstance(deps["pyedflib"], bool)
        self.assertIsInstance(deps["fitparse"], bool)
        self.assertIsInstance(deps["bioread"], bool)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""

    def test_load_and_preprocess_integration(self):
        """Test complete load and preprocess pipeline"""
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "test.csv")

        # Create CSV with artifacts
        test_rri = [800, 820, 2500, 200, 830, 810, 825, 840, 815, 835]

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["RRI"])
                for rri in test_rri:
                    writer.writerow([rri])

            # Test with pandas mocked
            with patch("hrvlib.data_handler.pd") as mock_pd:
                mock_df = MagicMock()
                mock_df.columns = ["RRI"]
                mock_df.__getitem__.return_value.tolist.return_value = test_rri
                mock_pd.read_csv.return_value = mock_df

                bundle = load_rr_file(csv_path, auto_preprocess=True)

                # Should have preprocessing results
                self.assertIsNotNone(bundle.preprocessing)
                self.assertGreater(len(bundle.preprocessing.artifact_indices), 0)

                # RRI should be corrected
                corrected_rri = np.array(bundle.rri_ms)
                self.assertTrue(np.all(corrected_rri > 300))  # No very short intervals
                self.assertTrue(np.all(corrected_rri < 2000))  # No very long intervals

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

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


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
