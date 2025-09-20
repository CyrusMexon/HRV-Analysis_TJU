import unittest
import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Optional, List
import sys
import os

# Add the parent directory to the Python path to import hrvlib.preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the modules under test
from hrvlib.data_handler import DataBundle
from hrvlib.preprocessing import PreprocessingResult
from hrvlib.pipeline import (
    UnifiedHRVPipeline,
    HRVAnalysisResults,
    create_unified_pipeline,
)


class TestHRVAnalysisResults(unittest.TestCase):
    """Test cases for HRVAnalysisResults dataclass"""

    def test_init_with_defaults(self):
        """Test initialization with default values"""
        results = HRVAnalysisResults()

        self.assertIsNone(results.time_domain)
        self.assertIsNone(results.frequency_domain)
        self.assertIsNone(results.nonlinear)
        self.assertIsNone(results.respiratory)
        self.assertIsNone(results.preprocessing_stats)
        self.assertIsNone(results.quality_assessment)
        self.assertEqual(results.analysis_info, {})
        self.assertEqual(results.warnings, [])

    def test_init_with_custom_values(self):
        """Test initialization with custom values"""
        custom_warnings = ["Test warning"]
        custom_info = {"test": "value"}

        results = HRVAnalysisResults(
            time_domain={"sdnn": 50.0},
            warnings=custom_warnings,
            analysis_info=custom_info,
        )

        self.assertEqual(results.time_domain, {"sdnn": 50.0})
        self.assertEqual(results.warnings, custom_warnings)
        self.assertEqual(results.analysis_info, custom_info)


class TestUnifiedHRVPipeline(unittest.TestCase):
    """Test cases for UnifiedHRVPipeline"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock RRI data
        self.mock_rri = [800, 750, 820, 780, 810, 790, 805, 785, 815, 795]

        # Create mock DataBundle
        self.mock_bundle = Mock(spec=DataBundle)
        self.mock_bundle.rri_ms = self.mock_rri
        self.mock_bundle.meta = {}
        self.mock_bundle.preprocessing = None

        # Create mock preprocessing result
        self.mock_preprocessing_result = Mock(spec=PreprocessingResult)
        self.mock_preprocessing_result.corrected_rri = np.array(self.mock_rri)
        self.mock_preprocessing_result.stats = {
            "artifacts_detected": 2,
            "artifacts_corrected": 1,
            "artifact_percentage": 5.0,
        }
        self.mock_preprocessing_result.noise_segments = []
        self.mock_preprocessing_result.correction_method = "interpolation"
        self.mock_preprocessing_result.quality_flags = {
            "poor_signal_quality": False,
            "excessive_artifacts": False,
        }

    def test_init_with_defaults(self):
        """Test pipeline initialization with default configuration"""
        pipeline = UnifiedHRVPipeline(self.mock_bundle)

        self.assertEqual(pipeline.bundle, self.mock_bundle)
        self.assertEqual(pipeline.preprocessing_config, {})
        self.assertIsInstance(pipeline.analysis_config, dict)
        self.assertIn("time_domain", pipeline.analysis_config)
        self.assertIn("frequency_domain", pipeline.analysis_config)
        self.assertIn("nonlinear", pipeline.analysis_config)
        self.assertIn("respiratory", pipeline.analysis_config)

    def test_init_with_custom_config(self):
        """Test pipeline initialization with custom configuration"""
        preprocessing_config = {"remove_artifacts": True}
        analysis_config = {"time_domain": {"enabled": False}}

        pipeline = UnifiedHRVPipeline(
            self.mock_bundle,
            preprocessing_config=preprocessing_config,
            analysis_config=analysis_config,
        )

        self.assertEqual(pipeline.preprocessing_config, preprocessing_config)
        self.assertEqual(pipeline.analysis_config, analysis_config)

    def test_get_default_analysis_config(self):
        """Test default analysis configuration structure"""
        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        config = pipeline._get_default_analysis_config()

        # Check all required modules are present
        required_modules = [
            "time_domain",
            "frequency_domain",
            "nonlinear",
            "respiratory",
        ]
        for module in required_modules:
            self.assertIn(module, config)
            self.assertIn("enabled", config[module])
            self.assertTrue(config[module]["enabled"])

        # Check specific configuration values
        self.assertEqual(config["frequency_domain"]["sampling_rate"], 4.0)
        self.assertEqual(config["frequency_domain"]["window_type"], "hann")
        self.assertEqual(config["nonlinear"]["mse_scales"], 10)

    @patch("hrvlib.pipeline.preprocess_rri")
    def test_run_preprocessing(self, mock_preprocess):
        """Test preprocessing execution"""
        mock_preprocess.return_value = self.mock_preprocessing_result

        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        rri, preprocessing_result = pipeline._run_preprocessing()

        mock_preprocess.assert_called_once_with(self.mock_rri)
        np.testing.assert_array_equal(rri, np.array(self.mock_rri))
        self.assertEqual(preprocessing_result, self.mock_preprocessing_result)

    @patch("hrvlib.pipeline.preprocess_rri")
    def test_run_preprocessing_with_config(self, mock_preprocess):
        """Test preprocessing with custom configuration"""
        mock_preprocess.return_value = self.mock_preprocessing_result

        custom_config = {"threshold": 0.5}
        pipeline = UnifiedHRVPipeline(
            self.mock_bundle, preprocessing_config=custom_config
        )

        pipeline._run_preprocessing()
        mock_preprocess.assert_called_once_with(self.mock_rri, threshold=0.5)

    def test_apply_analysis_window(self):
        """Test analysis window application"""
        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        rri = np.array([800, 750, 820, 780, 810])  # ~4 seconds total

        # Apply window from 1s to 3s
        windowed_rri = pipeline._apply_analysis_window(rri, (1.0, 3.0))

        # Should get intervals that fall within the time window
        self.assertGreater(len(windowed_rri), 0)
        self.assertLessEqual(len(windowed_rri), len(rri))

    def test_apply_analysis_window_no_data(self):
        """Test analysis window with no data in range"""
        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        rri = np.array([800, 750, 820])  # ~2.4 seconds total

        # Try to get window from 5s to 10s (beyond data range)
        with self.assertRaises(ValueError) as context:
            pipeline._apply_analysis_window(rri, (5.0, 10.0))

        self.assertIn("No data found in analysis window", str(context.exception))

    @patch("hrvlib.pipeline.HRVTimeDomainAnalysis")
    def test_run_time_domain_analysis_success(self, mock_time_domain):
        """Test successful time domain analysis"""
        mock_analyzer = Mock()
        mock_analyzer.full_analysis.return_value = {"sdnn": 50.0}
        mock_time_domain.return_value = mock_analyzer

        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        pipeline.preprocessing_result = self.mock_preprocessing_result

        result = pipeline.run_time_domain_analysis(np.array(self.mock_rri))

        self.assertEqual(result, {"sdnn": 50.0})
        mock_time_domain.assert_called_once()
        mock_analyzer.full_analysis.assert_called_once()

    @patch("hrvlib.pipeline.HRVTimeDomainAnalysis")
    def test_run_time_domain_analysis_disabled(self, mock_time_domain):
        """Test time domain analysis when disabled"""
        config = {"time_domain": {"enabled": False}}
        pipeline = UnifiedHRVPipeline(self.mock_bundle, analysis_config=config)

        result = pipeline.run_time_domain_analysis(np.array(self.mock_rri))

        self.assertIsNone(result)
        mock_time_domain.assert_not_called()

    @patch("hrvlib.pipeline.HRVTimeDomainAnalysis")
    def test_run_time_domain_analysis_exception(self, mock_time_domain):
        """Test time domain analysis with exception"""
        mock_time_domain.side_effect = Exception("Test error")

        pipeline = UnifiedHRVPipeline(self.mock_bundle)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.run_time_domain_analysis(np.array(self.mock_rri))

        self.assertIsNone(result)
        self.assertTrue(len(w) > 0)
        self.assertIn("Time domain analysis failed", str(w[0].message))

    @patch("hrvlib.pipeline.HRVFreqDomainAnalysis")
    def test_run_frequency_domain_analysis_success(self, mock_freq_domain):
        """Test successful frequency domain analysis"""
        mock_analyzer = Mock()
        mock_analyzer.get_results.return_value = {"lf_power": 100.0}
        mock_freq_domain.return_value = mock_analyzer

        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        pipeline.preprocessing_result = self.mock_preprocessing_result

        result = pipeline.run_frequency_domain_analysis(np.array(self.mock_rri))

        self.assertEqual(result, {"lf_power": 100.0})
        mock_freq_domain.assert_called_once()
        mock_analyzer.get_results.assert_called_once()

    @patch("hrvlib.pipeline.HRVFreqDomainAnalysis")
    def test_run_frequency_domain_analysis_with_config(self, mock_freq_domain):
        """Test frequency domain analysis with custom configuration"""
        mock_analyzer = Mock()
        mock_analyzer.get_results.return_value = {"lf_power": 100.0}
        mock_freq_domain.return_value = mock_analyzer

        config = {
            "frequency_domain": {
                "enabled": True,
                "sampling_rate": 8.0,
                "window_type": "blackman",
            }
        }
        pipeline = UnifiedHRVPipeline(self.mock_bundle, analysis_config=config)
        pipeline.preprocessing_result = self.mock_preprocessing_result

        result = pipeline.run_frequency_domain_analysis(np.array(self.mock_rri))

        # Verify the analyzer was called with custom config
        call_args = mock_freq_domain.call_args
        self.assertEqual(call_args.kwargs["sampling_rate"], 8.0)
        self.assertEqual(call_args.kwargs["window_type"], "blackman")

    @patch("hrvlib.pipeline.NonlinearHRVAnalysis")
    def test_run_nonlinear_analysis_success(self, mock_nonlinear):
        """Test successful nonlinear analysis"""
        mock_analyzer = Mock()
        mock_analyzer.full_nonlinear_analysis.return_value = {"sd1": 25.0}
        mock_nonlinear.return_value = mock_analyzer

        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        pipeline.preprocessing_result = self.mock_preprocessing_result

        result = pipeline.run_nonlinear_analysis(np.array(self.mock_rri))

        self.assertEqual(result, {"sd1": 25.0})
        mock_nonlinear.assert_called_once()
        mock_analyzer.full_nonlinear_analysis.assert_called_once()

    @patch("hrvlib.pipeline.analyze_respiratory_metrics")
    def test_run_respiratory_analysis_success(self, mock_respiratory):
        """Test successful respiratory analysis"""
        mock_respiratory.return_value = {"respiratory_rate": 15.0}

        pipeline = UnifiedHRVPipeline(self.mock_bundle)

        result = pipeline.run_respiratory_analysis(np.array(self.mock_rri))

        self.assertEqual(result, {"respiratory_rate": 15.0})
        mock_respiratory.assert_called_once()
        # Check that bundle metadata was updated
        self.assertEqual(
            pipeline.bundle.meta["respiratory_metrics"], {"respiratory_rate": 15.0}
        )

    def test_assess_overall_quality_good(self):
        """Test quality assessment with good data"""
        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        rri = np.array([800] * 100)  # 80 seconds of data

        assessment = pipeline._assess_overall_quality(
            rri, self.mock_preprocessing_result
        )

        self.assertEqual(assessment["overall_quality"], "good")
        self.assertTrue(assessment["data_length_adequate"])
        self.assertEqual(assessment["duration_s"], 80.0)
        self.assertEqual(assessment["artifact_percentage"], 5.0)

    def test_assess_overall_quality_short_duration(self):
        """Test quality assessment with short duration"""
        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        rri = np.array([800, 800, 800])  # 2.4 seconds

        assessment = pipeline._assess_overall_quality(
            rri, self.mock_preprocessing_result
        )

        # Fixed: Use partial string matching instead of exact match
        self.assertTrue(
            any(
                "Recording duration < 2 minutes" in rec
                for rec in assessment["recommendations"]
            ),
            f"Expected 'Recording duration < 2 minutes' in recommendations: {assessment['recommendations']}",
        )

    def test_assess_overall_quality_poor_signal(self):
        """Test quality assessment with poor signal quality"""
        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        rri = np.array([800] * 100)

        # Mock poor signal quality
        poor_quality_result = Mock(spec=PreprocessingResult)
        poor_quality_result.stats = {"artifact_percentage": 15.0}
        poor_quality_result.quality_flags = {"poor_signal_quality": True}

        assessment = pipeline._assess_overall_quality(rri, poor_quality_result)

        self.assertEqual(assessment["overall_quality"], "poor")
        self.assertIn("Poor signal quality detected", assessment["recommendations"])

        # Fixed: Use partial string matching for high artifact percentage
        self.assertTrue(
            any(
                "High artifact percentage" in rec
                for rec in assessment["recommendations"]
            ),
            f"Expected 'High artifact percentage' in recommendations: {assessment['recommendations']}",
        )

    def test_assess_overall_quality_with_respiratory_warnings(self):
        """Test quality assessment including respiratory metrics"""
        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        rri = np.array([800] * 100)

        # Add respiratory metrics to bundle
        pipeline.bundle.meta["respiratory_metrics"] = {
            "confidence": 0.3,
            "warnings": ["Low signal quality"],
            "lf_hf_analysis": {"boundary_overlap": True},
        }

        assessment = pipeline._assess_overall_quality(
            rri, self.mock_preprocessing_result
        )

        self.assertEqual(assessment["overall_quality"], "fair")

        # Fixed: Use partial string matching for respiratory confidence
        self.assertTrue(
            any(
                "Low respiratory signal confidence" in rec
                for rec in assessment["recommendations"]
            ),
            f"Expected 'Low respiratory signal confidence' in recommendations: {assessment['recommendations']}",
        )
        self.assertIn("Respiratory: Low signal quality", assessment["recommendations"])
        self.assertTrue(
            any(
                "overlaps LF/HF boundary" in rec
                for rec in assessment["recommendations"]
            ),
            f"Expected 'overlaps LF/HF boundary' in recommendations: {assessment['recommendations']}",
        )

    @patch("hrvlib.pipeline.preprocess_rri")
    @patch("hrvlib.pipeline.HRVTimeDomainAnalysis")
    @patch("hrvlib.pipeline.HRVFreqDomainAnalysis")
    @patch("hrvlib.pipeline.NonlinearHRVAnalysis")
    @patch("hrvlib.pipeline.analyze_respiratory_metrics")
    def test_run_all_success(
        self, mock_resp, mock_nonlinear, mock_freq, mock_time, mock_preprocess
    ):
        """Test complete pipeline execution"""
        # Set up mocks
        mock_preprocess.return_value = self.mock_preprocessing_result

        mock_time_analyzer = Mock()
        mock_time_analyzer.full_analysis.return_value = {"sdnn": 50.0}
        mock_time.return_value = mock_time_analyzer

        mock_freq_analyzer = Mock()
        mock_freq_analyzer.get_results.return_value = {"lf_power": 100.0}
        mock_freq.return_value = mock_freq_analyzer

        mock_nonlinear_analyzer = Mock()
        mock_nonlinear_analyzer.full_nonlinear_analysis.return_value = {"sd1": 25.0}
        mock_nonlinear.return_value = mock_nonlinear_analyzer

        mock_resp.return_value = {"respiratory_rate": 15.0}

        # Run pipeline
        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        results = pipeline.run_all()

        # Verify results structure
        self.assertIsInstance(results, HRVAnalysisResults)
        self.assertEqual(results.time_domain, {"sdnn": 50.0})
        self.assertEqual(results.frequency_domain, {"lf_power": 100.0})
        self.assertEqual(results.nonlinear, {"sd1": 25.0})
        self.assertEqual(results.respiratory, {"respiratory_rate": 15.0})

        # Verify preprocessing stats
        self.assertIsNotNone(results.preprocessing_stats)
        self.assertEqual(results.preprocessing_stats["artifacts_detected"], 2)
        self.assertEqual(results.preprocessing_stats["artifact_percentage"], 5.0)

        # Verify quality assessment
        self.assertIsNotNone(results.quality_assessment)

        # Verify analysis info
        self.assertIsNotNone(results.analysis_info)
        self.assertEqual(results.analysis_info["total_intervals"], len(self.mock_rri))
        self.assertTrue(results.analysis_info["preprocessing_applied"])

    @patch("hrvlib.pipeline.preprocess_rri")
    def test_run_all_with_analysis_window(self, mock_preprocess):
        """Test pipeline execution with analysis window"""
        # Create a mock respiratory bundle with the 'resp' attribute
        mock_respiratory_bundle = Mock()
        mock_respiratory_bundle.resp = Mock()  # Add the resp attribute that was missing

        # Update the preprocessing result to include the respiratory bundle
        mock_preprocessing_result = Mock(spec=PreprocessingResult)
        mock_preprocessing_result.corrected_rri = np.array(self.mock_rri)
        mock_preprocessing_result.stats = {
            "artifacts_detected": 2,
            "artifacts_corrected": 1,
            "artifact_percentage": 5.0,
        }
        mock_preprocessing_result.noise_segments = []
        mock_preprocessing_result.correction_method = "interpolation"
        mock_preprocessing_result.quality_flags = {
            "poor_signal_quality": False,
            "excessive_artifacts": False,
        }

        mock_preprocess.return_value = mock_preprocessing_result

        pipeline = UnifiedHRVPipeline(self.mock_bundle)

        with patch.object(pipeline, "_apply_analysis_window") as mock_window:
            mock_window.return_value = np.array([800, 750, 820])
            results = pipeline.run_all(analysis_window=(1.0, 3.0))

            mock_window.assert_called_once()
            self.assertEqual(results.analysis_info["analysis_window"], (1.0, 3.0))

    @patch("hrvlib.pipeline.preprocess_rri")
    def test_run_all_with_exception(self, mock_preprocess):
        """Test pipeline execution with exception during preprocessing"""
        mock_preprocess.side_effect = Exception("Preprocessing failed")

        pipeline = UnifiedHRVPipeline(self.mock_bundle)
        results = pipeline.run_all()

        self.assertEqual(len(results.warnings), 1)
        self.assertIn("Pipeline execution failed", results.warnings[0])

    def test_run_all_selective_modules(self):
        """Test pipeline with selective module execution"""
        config = {
            "time_domain": {"enabled": True},
            "frequency_domain": {"enabled": False},
            "nonlinear": {"enabled": False},
            "respiratory": {"enabled": False},
        }

        with patch("hrvlib.pipeline.preprocess_rri") as mock_preprocess:
            mock_preprocess.return_value = self.mock_preprocessing_result

            with patch("hrvlib.pipeline.HRVTimeDomainAnalysis") as mock_time:
                mock_analyzer = Mock()
                mock_analyzer.full_analysis.return_value = {"sdnn": 50.0}
                mock_time.return_value = mock_analyzer

                pipeline = UnifiedHRVPipeline(self.mock_bundle, analysis_config=config)
                results = pipeline.run_all()

                # Only time domain should be executed
                self.assertEqual(results.time_domain, {"sdnn": 50.0})
                self.assertIsNone(results.frequency_domain)
                self.assertIsNone(results.nonlinear)
                self.assertIsNone(results.respiratory)

                # Verify module enablement in results
                modules_enabled = results.analysis_info["modules_enabled"]
                self.assertTrue(modules_enabled["time_domain"])
                self.assertFalse(modules_enabled["frequency_domain"])


class TestFactoryFunction(unittest.TestCase):
    """Test cases for factory function"""

    def test_create_unified_pipeline(self):
        """Test factory function creates pipeline correctly"""
        mock_bundle = Mock(spec=DataBundle)
        preprocessing_config = {"test": "config"}
        analysis_config = {"test": "analysis"}

        pipeline = create_unified_pipeline(
            mock_bundle,
            preprocessing_config=preprocessing_config,
            analysis_config=analysis_config,
        )

        self.assertIsInstance(pipeline, UnifiedHRVPipeline)
        self.assertEqual(pipeline.bundle, mock_bundle)
        self.assertEqual(pipeline.preprocessing_config, preprocessing_config)
        self.assertEqual(pipeline.analysis_config, analysis_config)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""

    def setUp(self):
        """Set up realistic test data"""
        # Generate realistic RRI data (normal sinus rhythm around 1000ms)
        np.random.seed(42)  # For reproducible tests
        base_rri = 1000
        variability = 50
        n_beats = 300  # 5 minutes at 60 bpm

        self.realistic_rri = [
            base_rri + np.random.normal(0, variability) for _ in range(n_beats)
        ]

        self.bundle = Mock(spec=DataBundle)
        self.bundle.rri_ms = self.realistic_rri
        self.bundle.meta = {}
        self.bundle.preprocessing = None

    @patch("hrvlib.pipeline.preprocess_rri")
    @patch("hrvlib.pipeline.HRVTimeDomainAnalysis")
    @patch("hrvlib.pipeline.HRVFreqDomainAnalysis")
    @patch("hrvlib.pipeline.NonlinearHRVAnalysis")
    @patch("hrvlib.pipeline.analyze_respiratory_metrics")
    def test_realistic_data_processing(
        self, mock_resp, mock_nonlinear, mock_freq, mock_time, mock_preprocess
    ):
        """Test pipeline with realistic data volumes and configurations"""
        # Set up realistic preprocessing result
        preprocessing_result = Mock(spec=PreprocessingResult)
        preprocessing_result.corrected_rri = np.array(self.realistic_rri)
        preprocessing_result.stats = {
            "artifacts_detected": 5,
            "artifacts_corrected": 3,
            "artifact_percentage": 1.7,
        }
        preprocessing_result.noise_segments = []
        preprocessing_result.correction_method = "interpolation"
        preprocessing_result.quality_flags = {
            "poor_signal_quality": False,
            "excessive_artifacts": False,
        }
        mock_preprocess.return_value = preprocessing_result

        # Set up realistic analysis results
        mock_time_analyzer = Mock()
        mock_time_analyzer.full_analysis.return_value = {
            "sdnn": 45.2,
            "rmssd": 28.7,
            "pnn50": 12.3,
            "mean_hr": 72.5,
        }
        mock_time.return_value = mock_time_analyzer

        mock_freq_analyzer = Mock()
        mock_freq_analyzer.get_results.return_value = {
            "vlf_power": 150.0,
            "lf_power": 200.0,
            "hf_power": 180.0,
            "lf_hf_ratio": 1.11,
            "total_power": 530.0,
        }
        mock_freq.return_value = mock_freq_analyzer

        mock_nonlinear_analyzer = Mock()
        mock_nonlinear_analyzer.full_nonlinear_analysis.return_value = {
            "poincare": {"sd1": 20.3, "sd2": 64.1},
            "sample_entropy": 1.85,
            "dfa": {"alpha1": 1.02, "alpha2": 0.89},
        }
        mock_nonlinear.return_value = mock_nonlinear_analyzer

        mock_resp.return_value = {
            "respiratory_rate": 16.2,
            "confidence": 0.82,
            "warnings": [],
        }

        # Run complete analysis
        pipeline = UnifiedHRVPipeline(self.bundle)
        results = pipeline.run_all()

        # Verify comprehensive results
        self.assertIsNotNone(results.time_domain)
        self.assertIsNotNone(results.frequency_domain)
        self.assertIsNotNone(results.nonlinear)
        self.assertIsNotNone(results.respiratory)

        # Verify quality assessment
        self.assertEqual(results.quality_assessment["overall_quality"], "good")
        self.assertTrue(results.quality_assessment["data_length_adequate"])
        self.assertAlmostEqual(
            results.quality_assessment["duration_s"], 300.0, places=0
        )

        # Verify analysis metadata
        self.assertEqual(results.analysis_info["total_intervals"], 300)
        self.assertTrue(results.analysis_info["preprocessing_applied"])

        # Verify all modules were called
        mock_preprocess.assert_called_once()
        mock_time.assert_called_once()
        mock_freq.assert_called_once()
        mock_nonlinear.assert_called_once()
        mock_resp.assert_called_once()


if __name__ == "__main__":
    # Run with pytest for better output
    pytest.main([__file__, "-v", "--tb=short"])


# HRV-Analysis_TJU\tests\test_pipeline.py
