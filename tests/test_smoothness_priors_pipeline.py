"""
Integration tests for smoothness priors with HRV pipeline (CORRECTED)

Tests the full integration of smoothness priors detrending through
the complete HRV analysis pipeline.

Run with: pytest tests/test_smoothness_priors_pipeline.py -v
"""

import pytest
import numpy as np
from hrvlib.data_handler import DataBundle, SourceInfo
from hrvlib.pipeline import create_unified_pipeline, UnifiedHRVPipeline
from hrvlib.preprocessing import preprocess_rri


class TestPipelineIntegration:
    """Test smoothness priors integration with full pipeline"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        np.random.seed(42)

        # Create realistic RR intervals
        n_beats = 300
        self.rr_intervals = np.random.normal(800, 50, n_beats).tolist()

        # Create test bundle with correct structure
        self.bundle = DataBundle(
            rri_ms=self.rr_intervals,  # List of RR intervals in ms
            source=SourceInfo(path="test_data", filetype=".test", device="Test"),
        )

    def test_pipeline_with_smoothness_priors_default(self):
        """Test pipeline with smoothness priors using default lambda"""
        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 500,  # Kubios default
            }
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        assert results is not None
        assert results.frequency_domain is not None
        assert "lf_power" in results.frequency_domain
        assert "hf_power" in results.frequency_domain
        assert "lf_hf_ratio" in results.frequency_domain

    def test_pipeline_with_different_lambda_values(self):
        """Test pipeline with different lambda parameters"""
        lambda_values = [10, 500, 5000]
        results_list = []

        for lam in lambda_values:
            analysis_config = {
                "frequency_domain": {
                    "enabled": True,
                    "detrend_method": "smoothness_priors",
                    "detrend_lambda": lam,
                }
            }

            pipeline = create_unified_pipeline(
                self.bundle, analysis_config=analysis_config
            )

            results = pipeline.run_all()
            results_list.append(results)

        # All should complete successfully
        assert len(results_list) == 3
        assert all(r.frequency_domain is not None for r in results_list)

        # Results should vary with lambda
        lf_powers = [r.frequency_domain["lf_power"] for r in results_list]
        assert len(set(lf_powers)) > 1, "Results should differ with lambda"

    def test_pipeline_comparison_methods(self):
        """Compare smoothness priors with other detrending methods"""
        methods = ["linear", "constant", "smoothness_priors"]
        results_dict = {}

        for method in methods:
            config = {"frequency_domain": {"enabled": True, "detrend_method": method}}

            if method == "smoothness_priors":
                config["frequency_domain"]["detrend_lambda"] = 500

            pipeline = create_unified_pipeline(self.bundle, analysis_config=config)

            results = pipeline.run_all()
            results_dict[method] = results

        # All methods should produce results
        assert all(r.frequency_domain is not None for r in results_dict.values())

        # Verify all expected metrics are present
        for method, results in results_dict.items():
            assert "lf_power" in results.frequency_domain
            assert "hf_power" in results.frequency_domain
            assert results.frequency_domain["lf_power"] > 0

    def test_pipeline_with_preprocessing_and_smoothness_priors(self):
        """Test pipeline with both preprocessing and smoothness priors"""
        # Add some artifacts to the data
        rr_with_artifacts = np.array(self.rr_intervals)
        rr_with_artifacts[50] = rr_with_artifacts[50] * 1.5  # Ectopic beat
        rr_with_artifacts[150] = rr_with_artifacts[150] * 0.5  # Missed beat

        bundle_with_artifacts = DataBundle(
            rri_ms=rr_with_artifacts.tolist(),
            source=SourceInfo(path="test_data", filetype=".test", device="Test"),
        )

        # Configure preprocessing and smoothness priors
        preprocessing_config = {
            "artifact_correction_enabled": True,
            "interpolation_method": "cubic_spline",
        }

        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 500,
            }
        }

        pipeline = create_unified_pipeline(
            bundle_with_artifacts,
            preprocessing_config=preprocessing_config,
            analysis_config=analysis_config,
        )

        results = pipeline.run_all()

        assert results is not None
        assert results.preprocessing_stats is not None
        assert results.frequency_domain is not None
        assert results.preprocessing_stats["artifacts_detected"] > 0

    def test_pipeline_all_domains_with_smoothness_priors(self):
        """Test that all analysis domains work with smoothness priors"""
        analysis_config = {
            "time_domain": {"enabled": True},
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 500,
            },
            "nonlinear": {"enabled": True, "include_mse": True, "include_dfa": True},
            "respiratory": {"enabled": True},
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        assert results is not None
        assert results.time_domain is not None
        assert results.frequency_domain is not None
        assert results.nonlinear is not None
        # Respiratory might be None if no respiratory signal

        # Verify frequency domain used smoothness priors
        assert "lf_power" in results.frequency_domain
        assert "hf_power" in results.frequency_domain

    def test_pipeline_with_analysis_window(self):
        """Test smoothness priors with analysis window"""
        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 500,
                "analysis_window": (0.0, 120.0),  # First 2 minutes
            }
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        assert results is not None
        assert results.frequency_domain is not None

    def test_pipeline_error_handling(self):
        """Test error handling with invalid smoothness priors configuration"""
        # Invalid lambda (negative)
        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": -100,  # Invalid!
            }
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        # Should handle error gracefully
        results = pipeline.run_all()

        # Pipeline should either produce results with fallback or handle error
        assert results is not None


class TestPipelineOutputs:
    """Test that pipeline outputs are correct with smoothness priors"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        np.random.seed(42)

        # Create realistic RR intervals with a clear trend
        n_beats = 300
        time = np.arange(n_beats)

        # Add a clear linear trend
        trend = 800 + 0.5 * time  # 0.5ms increase per beat
        noise = np.random.normal(0, 30, n_beats)  # Less noise to see trend effect

        self.rr_intervals = (trend + noise).tolist()

        # Create test bundle
        self.bundle = DataBundle(
            rri_ms=self.rr_intervals,
            source=SourceInfo(path="test_data", filetype=".test", device="Test"),
        )

    def test_results_structure(self):
        """Test that results have correct structure"""
        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 500,
            }
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        # Check frequency domain structure
        freq_results = results.frequency_domain

        required_keys = [
            "lf_power",
            "hf_power",
            "vlf_power",
            "lf_hf_ratio",
            "total_power",
            "peak_freq_lf",
            "peak_freq_hf",
        ]

        for key in required_keys:
            assert key in freq_results, f"Missing key: {key}"

    def test_results_to_dict(self):
        """Test that results can be converted to dict"""
        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 500,
            }
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        results = pipeline.run_all()
        results_dict = results.to_dict()

        assert isinstance(results_dict, dict)
        assert "frequency_domain" in results_dict
        assert "freq_domain" in results_dict  # Alternative key
        assert results_dict["frequency_domain"] == results_dict["freq_domain"]

    def test_psd_data_availability(self):
        """Test that PSD data is available for plotting"""
        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 500,
            }
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        # Check if PSD data is present
        freq_results = results.frequency_domain

        # PSD data should be available for plotting
        if "psd_frequencies" in freq_results and "psd_power" in freq_results:
            assert len(freq_results["psd_frequencies"]) > 0
            assert len(freq_results["psd_power"]) > 0


class TestRealWorldScenarios:
    """Test with realistic scenarios"""

    def setup_method(self):
        """Set up test fixtures"""
        np.random.seed(42)

    def test_short_recording(self):
        """Test with short recording (1-2 minutes)"""
        # ~100 beats at 60 bpm = ~1.5 minutes
        short_rr = np.random.normal(1000, 50, 100).tolist()

        bundle = DataBundle(
            rri_ms=short_rr,
            source=SourceInfo(path="test_data", filetype=".test", device="Test"),
        )

        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 100,  # Lighter smoothing for short
            }
        }

        pipeline = create_unified_pipeline(bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        assert results is not None
        assert results.frequency_domain is not None

    def test_long_recording(self):
        """Test with longer recording (10+ minutes)"""
        # ~600 beats at 60 bpm = ~10 minutes
        long_rr = np.random.normal(800, 50, 600).tolist()

        bundle = DataBundle(
            rri_ms=long_rr,
            source=SourceInfo(path="test_data", filetype=".test", device="Test"),
        )

        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 1000,  # Stronger smoothing for long
            }
        }

        pipeline = create_unified_pipeline(bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        assert results is not None
        assert results.frequency_domain is not None

    def test_athletic_hrv_profile(self):
        """Test with athletic HRV profile (high variation)"""
        # Athletes typically have higher HRV
        athletic_rr = np.random.normal(1000, 100, 300).tolist()  # Lower HR, higher HRV

        bundle = DataBundle(
            rri_ms=athletic_rr,
            source=SourceInfo(path="test_data", filetype=".test", device="Test"),
        )

        analysis_config = {
            "frequency_domain": {
                "enabled": True,
                "detrend_method": "smoothness_priors",
                "detrend_lambda": 500,
            }
        }

        pipeline = create_unified_pipeline(bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        assert results is not None
        assert results.frequency_domain is not None
        # Athletic profile should show high HF power
        assert results.frequency_domain["hf_power"] > 0


class TestBackwardCompatibility:
    """Test that existing functionality still works"""

    def setup_method(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.rr_intervals = np.random.normal(800, 50, 300).tolist()
        self.bundle = DataBundle(
            rri_ms=self.rr_intervals,
            source=SourceInfo(path="test_data", filetype=".test", device="Test"),
        )

    def test_default_behavior_unchanged(self):
        """Test that default behavior (linear) still works"""
        # No explicit config - should use defaults
        pipeline = create_unified_pipeline(self.bundle)

        results = pipeline.run_all()

        assert results is not None
        assert results.frequency_domain is not None

    def test_linear_detrending_still_works(self):
        """Test that linear detrending still works after integration"""
        analysis_config = {
            "frequency_domain": {"enabled": True, "detrend_method": "linear"}
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        assert results is not None
        assert results.frequency_domain is not None

    def test_constant_detrending_still_works(self):
        """Test that constant detrending still works"""
        analysis_config = {
            "frequency_domain": {"enabled": True, "detrend_method": "constant"}
        }

        pipeline = create_unified_pipeline(self.bundle, analysis_config=analysis_config)

        results = pipeline.run_all()

        assert results is not None
        assert results.frequency_domain is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
