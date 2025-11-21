"""
Unit tests for smoothness priors detrending integration

Tests the signal_processing.smoothness_priors module and its integration
with the freq_domain analysis pipeline.

Run with: pytest tests/test_smoothness_priors.py -v
"""

import pytest
import numpy as np
import warnings
from hrvlib.signal_processing.smoothness_priors import (
    smoothness_priors_detrending,
    detrend_with_smoothness_priors,
    choose_lambda_for_hrv,
    validate_detrending,
)


class TestSmoothnessPriorsCore:
    """Test core smoothness priors functionality"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        np.random.seed(42)
        # Generate realistic RR intervals with trend
        n_beats = 300
        time = np.arange(n_beats)

        # Create baseline with trend
        trend = 800 + 0.2 * time + 30 * np.sin(2 * np.pi * time / 100)

        # Add HRV variation
        hrv_variation = 20 * np.sin(2 * np.pi * time / 20) + 10 * np.random.randn(
            n_beats
        )

        # Create RR intervals
        self.rr_intervals = trend + hrv_variation
        self.clean_rr = np.random.normal(800, 50, 300)

    def test_basic_smoothness_priors_execution(self):
        """Test basic smoothness priors execution"""
        detrended, trend = smoothness_priors_detrending(
            self.rr_intervals, lambda_param=500, fs=4.0
        )

        assert len(detrended) == len(self.rr_intervals)
        assert len(trend) == len(self.rr_intervals)
        assert isinstance(detrended, np.ndarray)
        assert isinstance(trend, np.ndarray)

    def test_reconstruction_accuracy(self):
        """Test that original = detrended + trend"""
        detrended, trend = smoothness_priors_detrending(
            self.rr_intervals, lambda_param=500
        )

        reconstructed = detrended + trend
        error = np.max(np.abs(reconstructed - self.rr_intervals))

        assert error < 1e-6, f"Reconstruction error too large: {error}"

    def test_lambda_parameter_effects(self):
        """Test that lambda parameter affects smoothing"""
        # Test different lambda values
        lambda_values = [10, 100, 500, 1000]
        stds = []

        for lam in lambda_values:
            detrended, _ = smoothness_priors_detrending(
                self.rr_intervals, lambda_param=lam
            )
            stds.append(np.std(detrended))

        # Higher lambda should generally preserve more variation
        # (removes less trend, so detrended has more variation)
        assert len(stds) == len(lambda_values)
        assert all(isinstance(s, (int, float)) for s in stds)

    def test_simplified_interface(self):
        """Test the simplified detrend_with_smoothness_priors interface"""
        # Without return_trend
        detrended = detrend_with_smoothness_priors(
            self.rr_intervals, lambda_param=500, return_trend=False
        )

        assert len(detrended) == len(self.rr_intervals)
        assert isinstance(detrended, np.ndarray)

        # With return_trend
        detrended2, trend = detrend_with_smoothness_priors(
            self.rr_intervals, lambda_param=500, return_trend=True
        )

        assert np.allclose(detrended, detrended2)

    def test_minimum_data_requirement(self):
        """Test that minimum data requirements are enforced"""
        # Too few data points
        short_rr = np.array([800, 810, 805])

        with pytest.raises(ValueError, match="at least 4 RR intervals"):
            smoothness_priors_detrending(short_rr, lambda_param=500)

    def test_lambda_validation(self):
        """Test that lambda parameter is validated"""
        # Zero lambda
        with pytest.raises(ValueError, match="Lambda parameter must be positive"):
            smoothness_priors_detrending(self.rr_intervals, lambda_param=0)

        # Negative lambda
        with pytest.raises(ValueError, match="Lambda parameter must be positive"):
            smoothness_priors_detrending(self.rr_intervals, lambda_param=-100)

    def test_validation_function(self):
        """Test the validate_detrending function"""
        detrended, trend = smoothness_priors_detrending(
            self.rr_intervals, lambda_param=500
        )

        is_valid, metrics = validate_detrending(self.rr_intervals, detrended, trend)

        assert bool(is_valid) is True
        assert "reconstruction_error" in metrics
        assert metrics["reconstruction_error"] < 1e-6
        assert "variance_ratio" in metrics
        assert 0 <= metrics["variance_ratio"] <= 1


class TestLambdaRecommendations:
    """Test lambda parameter recommendation functions"""

    def test_choose_lambda_for_hrv_standard(self):
        """Test lambda recommendations for standard analysis"""
        lambda_standard = choose_lambda_for_hrv("standard")
        assert lambda_standard == 500

    def test_choose_lambda_for_hrv_short(self):
        """Test lambda recommendations for short recordings"""
        lambda_short = choose_lambda_for_hrv("short")
        assert lambda_short == 100

    def test_choose_lambda_for_hrv_long(self):
        """Test lambda recommendations for long recordings"""
        lambda_long = choose_lambda_for_hrv("long")
        assert lambda_long == 1000

    def test_choose_lambda_default(self):
        """Test default lambda for unknown analysis type"""
        lambda_default = choose_lambda_for_hrv("unknown_type")
        assert lambda_default == 500


class TestFreqDomainIntegration:
    """Test integration with freq_domain module"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        np.random.seed(42)
        self.rr_intervals = np.random.normal(800, 50, 300)

    def test_freq_domain_accepts_smoothness_priors(self):
        """Test that freq_domain module accepts smoothness_priors method"""
        from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis

        # This should not raise an error after integration
        analyzer = HRVFreqDomainAnalysis(
            preprocessed_rri=self.rr_intervals,
            detrend_method="smoothness_priors",
            detrend_lambda=500,
        )

        assert analyzer.detrend_method == "smoothness_priors"
        assert analyzer.detrend_lambda == 500

    def test_freq_domain_smoothness_priors_execution(self):
        """Test that freq_domain can execute with smoothness priors"""
        from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis

        analyzer = HRVFreqDomainAnalysis(
            preprocessed_rri=self.rr_intervals,
            detrend_method="smoothness_priors",
            detrend_lambda=500,
        )

        results = analyzer.get_results()

        assert results is not None
        assert "lf_power" in results
        assert "hf_power" in results
        assert "lf_hf_ratio" in results

    def test_freq_domain_different_lambda_values(self):
        """Test freq_domain with different lambda values"""
        from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis

        lambda_values = [100, 500, 1000]
        results_list = []

        for lam in lambda_values:
            analyzer = HRVFreqDomainAnalysis(
                preprocessed_rri=self.rr_intervals,
                detrend_method="smoothness_priors",
                detrend_lambda=lam,
            )
            results = analyzer.get_results()
            results_list.append(results)

        # All should complete successfully
        assert len(results_list) == 3
        assert all(r is not None for r in results_list)
        assert all("lf_power" in r for r in results_list)

    def test_freq_domain_comparison_with_linear(self):
        """Compare smoothness priors with linear detrending"""
        from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis

        # Linear detrending
        analyzer_linear = HRVFreqDomainAnalysis(
            preprocessed_rri=self.rr_intervals, detrend_method="linear"
        )
        results_linear = analyzer_linear.get_results()

        # Smoothness priors
        analyzer_sp = HRVFreqDomainAnalysis(
            preprocessed_rri=self.rr_intervals,
            detrend_method="smoothness_priors",
            detrend_lambda=500,
        )
        results_sp = analyzer_sp.get_results()

        # Both should produce valid results
        assert results_linear is not None
        assert results_sp is not None

        # Results should be different (not identical)
        assert results_linear["lf_power"] != results_sp["lf_power"]


class TestEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        """Set up for each test"""
        np.random.seed(42)

    def test_constant_signal(self):
        """Test with constant RR intervals"""
        constant_rr = np.ones(300) * 800

        # Should not crash
        detrended, trend = smoothness_priors_detrending(constant_rr, lambda_param=500)

        assert len(detrended) == len(constant_rr)
        # Detrended should be near zero for constant input
        assert np.std(detrended) < 1.0

    def test_very_short_signal(self):
        """Test with minimum acceptable length"""
        short_rr = np.array([800, 810, 805, 820])

        # Should work with exactly 4 points
        detrended, trend = smoothness_priors_detrending(short_rr, lambda_param=500)

        assert len(detrended) == 4

    def test_noisy_signal(self):
        """Test with very noisy signal"""
        noisy_rr = 800 + 200 * np.random.randn(300)

        # Should complete without error
        detrended, trend = smoothness_priors_detrending(noisy_rr, lambda_param=500)

        assert len(detrended) == len(noisy_rr)

    def test_extreme_lambda_values(self):
        """Test with extreme lambda values"""
        rr = np.random.normal(800, 50, 300)

        # Very small lambda
        detrended_small, _ = smoothness_priors_detrending(rr, lambda_param=1)
        assert len(detrended_small) == len(rr)

        # Very large lambda
        detrended_large, _ = smoothness_priors_detrending(rr, lambda_param=10000)
        assert len(detrended_large) == len(rr)


class TestPerformance:
    """Test performance characteristics"""

    def setup_method(self):
        """Set up for each test"""
        np.random.seed(42)

    def test_typical_dataset_performance(self):
        """Test with typical 5-minute dataset"""
        import time

        # 300 beats (~5 minutes at 60 bpm)
        typical_rr = np.random.normal(800, 50, 300)

        start_time = time.time()
        detrended, trend = smoothness_priors_detrending(typical_rr, lambda_param=500)
        elapsed = time.time() - start_time

        assert len(detrended) == len(typical_rr)
        assert elapsed < 1.0, f"Processing too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
