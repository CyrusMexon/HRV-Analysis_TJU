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
        load_rr_file,
        _coerce_float_list,
        _label_to_kind,
        set_interactive_mode,
        get_supported_formats,
        check_dependencies,
        load_and_preprocess,
    )
    from hrvlib.preprocessing import PreprocessingResult
except ImportError as e:
    print(f"Failed to import hrvlib modules: {e}")
    print(
        "Make sure hrvlib.data_handler.py and hrvlib.preprocessing.py are in the parent directory of the tests folder"
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

    def test_data_bundle_with_preprocessing(self):
        """Test DataBundle summary with preprocessing results"""
        rri = [800, 850, 900, 750, 820]
        bundle = DataBundle(rri_ms=rri)

        # Mock preprocessing result
        mock_preprocessing = MagicMock(spec=PreprocessingResult)
        mock_preprocessing.artifact_indices = [1, 3]
        mock_preprocessing.artifact_types = ["extra", "missed"]
        mock_preprocessing.correction_method = "cubic_spline"
        mock_preprocessing.interpolation_indices = [1, 3]

        bundle.preprocessing = mock_preprocessing

        summary = bundle.summary()
        self.assertIsNotNone(summary["preprocessing"])
        self.assertEqual(summary["preprocessing"]["artifacts_detected"], 2)
        self.assertEqual(summary["preprocessing"]["correction_method"], "cubic_spline")

    def test_data_bundle_with_waveforms(self):
        """Test DataBundle with waveform data"""
        ecg_data = np.random.randn(1000)
        ppg_data = np.random.randn(500)

        ecg_ts = TimeSeries(name="ECG", data=ecg_data, fs=1000.0)
        ppg_ts = TimeSeries(name="PPG", data=ppg_data, fs=500.0)

        bundle = DataBundle(ecg=[ecg_ts], ppg=[ppg_ts])

        summary = bundle.summary()
        self.assertIn("ECG", summary["types_loaded"])
        self.assertIn("PPG", summary["types_loaded"])
        self.assertEqual(summary["waveform_channels"]["ECG_channels"], 1)
        self.assertEqual(summary["waveform_channels"]["PPG_channels"], 1)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions"""

    def test_coerce_float_list(self):
        """Test _coerce_float_list function"""
        mixed_input = [1, "2.5", None, float("nan"), 4.7, "invalid"]
        result = _coerce_float_list(mixed_input)
        expected = [1.0, 2.5, 4.7]
        self.assertEqual(result, expected)

    def test_coerce_float_list_empty(self):
        """Test _coerce_float_list with empty input"""
        result = _coerce_float_list([])
        self.assertEqual(result, [])

    def test_coerce_float_list_all_invalid(self):
        """Test _coerce_float_list with all invalid values"""
        invalid_input = [None, "invalid", float("nan")]
        result = _coerce_float_list(invalid_input)
        self.assertEqual(result, [])

    def test_label_to_kind(self):
        """Test _label_to_kind function"""
        # ECG labels
        self.assertEqual(_label_to_kind("ECG_I"), "ECG")
        self.assertEqual(_label_to_kind("Lead II"), "ECG")
        self.assertEqual(_label_to_kind("ml-ii"), "ECG")
        self.assertEqual(_label_to_kind("v1"), "ECG")

        # PPG labels
        self.assertEqual(_label_to_kind("PPG"), "PPG")
        self.assertEqual(_label_to_kind("Pulse"), "PPG")
        self.assertEqual(_label_to_kind("Plethysmograph"), "PPG")
        self.assertEqual(_label_to_kind("oxim"), "PPG")

        # RESP labels
        self.assertEqual(_label_to_kind("Respiration"), "RESP")
        self.assertEqual(_label_to_kind("Thoracic"), "RESP")
        self.assertEqual(_label_to_kind("Breath"), "RESP")
        self.assertEqual(_label_to_kind("airflow"), "RESP")

        # Unknown labels
        self.assertIsNone(_label_to_kind("Unknown"))
        self.assertIsNone(_label_to_kind(""))
        self.assertIsNone(_label_to_kind(None))

    def test_label_to_kind_case_insensitive(self):
        """Test that _label_to_kind is case insensitive"""
        self.assertEqual(_label_to_kind("ECG"), "ECG")
        self.assertEqual(_label_to_kind("ecg"), "ECG")
        self.assertEqual(_label_to_kind("Ecg"), "ECG")
        self.assertEqual(_label_to_kind("PPG"), "PPG")
        self.assertEqual(_label_to_kind("ppg"), "PPG")


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

    def test_load_csv_file_with_waveform_data(self):
        """Test loading CSV file with waveform data"""
        csv_path = os.path.join(self.temp_dir, "test_waveform.csv")

        # Create test CSV with ECG data
        ecg_data = np.sin(np.linspace(0, 4 * np.pi, 1000))  # Synthetic ECG-like data

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ECG", "fs"])
            for i, ecg_val in enumerate(ecg_data):
                fs_val = 1000 if i == 0 else ""  # Only write fs in first row
                writer.writerow([ecg_val, fs_val])

        # Test loading
        with patch("hrvlib.data_handler.pd") as mock_pd:
            mock_df = MagicMock()
            mock_df.columns = ["ECG", "fs"]

            # Mock the column access behavior

        def mock_getitem(col):
            if col == "ECG":
                mock_series = MagicMock()
                mock_series.tolist.return_value = ecg_data.tolist()
                return mock_series
            elif col == "fs":
                # Create fs column with 1000 in first position, empty strings elsewhere
                fs_values = [1000] + [""] * (len(ecg_data) - 1)
                mock_series = MagicMock()

                # Mock the iteration behavior for the fs column parsing
                mock_series.__iter__ = lambda: iter(fs_values)
                return mock_series
            return MagicMock()

        mock_df.__getitem__ = mock_getitem
        mock_pd.read_csv.return_value = mock_df

        # Disable neurokit2 to avoid the peak detection warning
        with patch(
            "hrvlib.data_handler._extract_intervals_from_waveforms"
        ) as mock_extract:
            mock_extract.side_effect = lambda bundle: bundle

            bundle = load_rr_file(csv_path)

            self.assertEqual(len(bundle.ecg), 1)
            self.assertEqual(bundle.ecg[0].fs, 1000.0)
            self.assertEqual(len(bundle.ecg[0].data), 1000)

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

    def test_load_txt_file_with_header(self):
        """Test loading TXT file with non-numeric header"""
        txt_path = os.path.join(self.temp_dir, "test_header.txt")

        # Create test TXT with header
        with open(txt_path, "w") as f:
            f.write("RR_Intervals_ms\n")  # Non-numeric header
            for rri in self.test_rri:
                f.write(f"{rri}\n")

        # Force use of manual TXT parsing
        with patch("hrvlib.data_handler.pd", None):
            bundle = load_rr_file(txt_path)

        # Should skip header and load data correctly
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

    def test_load_json_file_nested_format(self):
        """Test loading JSON file with nested samples format"""
        json_path = os.path.join(self.temp_dir, "test_nested.json")

        # Create test JSON with nested structure
        test_data = {
            "samples": [
                {"rr": self.test_rri[0]},
                {"rr": self.test_rri[1]},
                {"rr": self.test_rri[2]},
                {"rr": self.test_rri[3]},
                {"rr": self.test_rri[4]},
            ]
        }
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        bundle = load_rr_file(json_path)
        self.assertEqual(len(bundle.rri_ms), 5)  # First 5 values

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

    def test_load_file_with_auto_preprocessing(self):
        """Test loading file with automatic preprocessing"""
        csv_path = os.path.join(self.temp_dir, "test_preprocess.csv")

        # Create CSV with some artifacts
        test_rri_with_artifacts = [800, 820, 2500, 200, 830, 810, 825, 840, 815, 835]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RRI"])
            for rri in test_rri_with_artifacts:
                writer.writerow([rri])

        # Test with pandas mocked
        with patch("hrvlib.data_handler.pd") as mock_pd:
            mock_df = MagicMock()
            mock_df.columns = ["RRI"]
            mock_df.__getitem__.return_value.tolist.return_value = (
                test_rri_with_artifacts
            )
            mock_pd.read_csv.return_value = mock_df

            bundle = load_rr_file(csv_path, auto_preprocess=True)

            # Should have preprocessing results
            self.assertIsNotNone(bundle.preprocessing)
            self.assertIsInstance(bundle.preprocessing, PreprocessingResult)
            self.assertTrue(bundle.meta.get("preprocessing_applied", False))

    def test_load_file_source_info_generation(self):
        """Test that source info is properly generated"""
        txt_path = os.path.join(self.temp_dir, "test_source.txt")

        with open(txt_path, "w") as f:
            for rri in self.test_rri:
                f.write(f"{rri}\n")

        with patch("hrvlib.data_handler.pd", None):
            bundle = load_rr_file(txt_path)

        # Should have source info
        self.assertIsNotNone(bundle.source)
        self.assertEqual(bundle.source.path, txt_path)
        self.assertEqual(bundle.source.filetype, ".txt")


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
        self.assertIsInstance(deps["xml.etree.ElementTree"], bool)

    def test_check_dependencies_structure(self):
        """Test that check_dependencies returns expected structure"""
        deps = check_dependencies()

        expected_keys = {
            "pandas",
            "pyedflib",
            "fitparse",
            "xml.etree.ElementTree",
            "bioread",
            "scipy",
            "numpy",
        }

        self.assertEqual(set(deps.keys()), expected_keys)

        # All values should be boolean
        for key, value in deps.items():
            self.assertIsInstance(value, bool, f"Dependency {key} should be boolean")


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""

    def setUp(self):
        """Set up temporary files for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_rri = [800, 820, 2500, 200, 830, 810, 825, 840, 815, 835]

    def tearDown(self):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_and_preprocess_function(self):
        """Test load_and_preprocess convenience function"""
        csv_path = os.path.join(self.temp_dir, "test.csv")

        # Create test CSV with artifacts
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RRI"])
            for rri in self.test_rri:
                writer.writerow([rri])

        # Test with pandas mocked
        with patch("hrvlib.data_handler.pd") as mock_pd:
            mock_df = MagicMock()
            mock_df.columns = ["RRI"]
            mock_df.__getitem__.return_value.tolist.return_value = self.test_rri
            mock_pd.read_csv.return_value = mock_df

            bundle = load_and_preprocess(csv_path)

            # Should have preprocessing applied automatically
            self.assertIsNotNone(bundle.preprocessing)
            self.assertTrue(bundle.meta.get("preprocessing_applied", False))

    def test_load_and_preprocess_with_custom_params(self):
        """Test load_and_preprocess with custom preprocessing parameters"""
        csv_path = os.path.join(self.temp_dir, "test_custom.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RRI"])
            for rri in self.test_rri:
                writer.writerow([rri])

        custom_params = {
            "threshold_low": 400.0,
            "threshold_high": 1500.0,
            "ectopic_threshold": 0.2,
        }

        # Test with pandas mocked
        with patch("hrvlib.data_handler.pd") as mock_pd:
            mock_df = MagicMock()
            mock_df.columns = ["RRI"]
            mock_df.__getitem__.return_value.tolist.return_value = self.test_rri
            mock_pd.read_csv.return_value = mock_df

            bundle = load_and_preprocess(csv_path, preprocessing_params=custom_params)

            # Should have preprocessing applied with custom parameters
            self.assertIsNotNone(bundle.preprocessing)
            self.assertEqual(bundle.preprocessing.correction_method, "cubic_spline")


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""

    def setUp(self):
        """Set up temporary files for testing"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_and_preprocess_integration(self):
        """Test complete load and preprocess pipeline"""
        csv_path = os.path.join(self.temp_dir, "test.csv")

        # Create CSV with artifacts
        test_rri = [800, 820, 2500, 200, 830, 810, 825, 840, 815, 835]

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

    def test_error_handling_preprocessing_failure(self):
        """Test graceful handling of preprocessing failures"""
        csv_path = os.path.join(self.temp_dir, "test_preprocess_fail.csv")

        # Create CSV with valid data
        test_rri = [800, 820, 850, 830, 810]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RRI"])
            for rri in test_rri:
                writer.writerow([rri])

        # Mock preprocessing to fail
        with patch("hrvlib.data_handler.pd") as mock_pd, patch(
            "hrvlib.data_handler.preprocess_rri"
        ) as mock_preprocess:

            mock_df = MagicMock()
            mock_df.columns = ["RRI"]
            mock_df.__getitem__.return_value.tolist.return_value = test_rri
            mock_pd.read_csv.return_value = mock_df

            # Make preprocessing raise an exception
            mock_preprocess.side_effect = Exception("Preprocessing failed")

            # Should not raise exception, but warn and continue
            with self.assertWarns(UserWarning):
                bundle = load_rr_file(csv_path, auto_preprocess=True)

            # Should still have loaded the data
            self.assertEqual(len(bundle.rri_ms), len(test_rri))
            self.assertIsNone(bundle.preprocessing)

    def test_bundle_summary_comprehensive(self):
        """Test comprehensive bundle summary with multiple data types"""
        # Create bundle with multiple data types
        rri = [800, 820, 850, 830, 810]
        ppi = [750, 780, 820, 790, 760]

        ecg_data = np.random.randn(1000)
        ppg_data = np.random.randn(500)
        resp_data = np.random.randn(200)

        ecg_ts = TimeSeries(name="ECG", data=ecg_data, fs=1000.0, units="mV")
        ppg_ts = TimeSeries(name="PPG", data=ppg_data, fs=500.0, units="a.u.")
        resp_ts = TimeSeries(name="RESP", data=resp_data, fs=100.0, units="a.u.")

        source = SourceInfo(path="/test/path.csv", filetype=".csv", device="TestDevice")

        bundle = DataBundle(
            rri_ms=rri,
            ppi_ms=ppi,
            ecg=[ecg_ts],
            ppg=[ppg_ts],
            resp=[resp_ts],
            source=source,
            meta={"device": "TestDevice", "fs": 1000},
        )

        summary = bundle.summary()

        # Check all data types are represented
        expected_types = {"RRI", "PPI", "ECG", "PPG", "RESP"}
        self.assertEqual(set(summary["types_loaded"]), expected_types)

        # Check channel counts
        self.assertEqual(summary["waveform_channels"]["ECG_channels"], 1)
        self.assertEqual(summary["waveform_channels"]["PPG_channels"], 1)
        self.assertEqual(summary["waveform_channels"]["RESP_channels"], 1)

        # Check RRI/PPI overviews
        self.assertIsNotNone(summary["rri_overview"])
        self.assertIsNotNone(summary["ppi_overview"])
        self.assertEqual(summary["rri_overview"]["n_RRI"], 5)
        self.assertEqual(summary["ppi_overview"]["n_PPI"], 5)

        # Check source info
        self.assertIsNotNone(summary["source"])
        self.assertEqual(summary["source"]["path"], "/test/path.csv")
        self.assertEqual(summary["source"]["device"], "TestDevice")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
