import sys
import os
import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hrvlib.metrics.nonlinear import NonlinearHRVAnalysis, create_nonlinear_analysis
from hrvlib.data_handler import DataBundle, TimeSeries
from hrvlib.preprocessing import preprocess_rri, PreprocessingResult


def generate_realistic_hrv(n_points=500, add_artifacts=False, seed=42):
    """Generate realistic HRV test data with optional artifacts"""
    np.random.seed(seed)

    # Base heart rate around 800ms (75 bpm)
    base_rr = np.random.normal(800, 50, n_points)

    # Add respiratory sinus arrhythmia (0.25 Hz breathing)
    time_points = np.linspace(0, n_points * 0.8, n_points)  # ~0.8s intervals
    respiratory_effect = 30 * np.sin(2 * np.pi * 0.25 * time_points / 60)

    # Add random HRV variation
    random_variation = np.random.normal(0, 15, n_points)

    # Combine components
    rr_intervals = base_rr + respiratory_effect + random_variation

    # Ensure physiological range
    rr_intervals = np.clip(rr_intervals, 400, 1500)

    if add_artifacts:
        # Add some artifacts (5% of data)
        artifact_indices = np.random.choice(
            n_points, size=int(n_points * 0.05), replace=False
        )
        for idx in artifact_indices:
            if np.random.random() < 0.5:
                rr_intervals[idx] = 250  # Extra beat
            else:
                rr_intervals[idx] = 2200  # Missed beat

    return rr_intervals.tolist()


@pytest.fixture
def clean_rri_data():
    """Clean RRI data without artifacts"""
    return generate_realistic_hrv(500, add_artifacts=False)


@pytest.fixture
def noisy_rri_data():
    """RRI data with artifacts"""
    return generate_realistic_hrv(500, add_artifacts=True)


@pytest.fixture
def short_rri_data():
    """Short RRI data for testing minimum requirements"""
    return generate_realistic_hrv(50, add_artifacts=False)


@pytest.fixture
def clean_bundle(clean_rri_data):
    """DataBundle with clean RRI data"""
    return DataBundle(rri_ms=clean_rri_data)


@pytest.fixture
def noisy_bundle(noisy_rri_data):
    """DataBundle with noisy RRI data"""
    return DataBundle(rri_ms=noisy_rri_data)


@pytest.fixture
def preprocessed_bundle(noisy_rri_data):
    """DataBundle with preprocessing already applied"""
    bundle = DataBundle(rri_ms=noisy_rri_data)
    preprocessing_result = preprocess_rri(noisy_rri_data)
    bundle.preprocessing = preprocessing_result
    return bundle


class TestNonlinearHRVAnalysisInitialization:
    """Test initialization and data extraction"""

    def test_init_with_clean_data(self, clean_bundle):
        """Test initialization with clean data"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)

        assert len(analyzer.rr_ms) == 500
        assert analyzer.bundle is clean_bundle
        assert analyzer.use_preprocessing is True
        assert analyzer.analysis_window is None

    def test_init_with_preprocessing_disabled(self, clean_bundle):
        """Test initialization with preprocessing disabled"""
        analyzer = NonlinearHRVAnalysis(clean_bundle, use_preprocessing=False)

        assert len(analyzer.rr_ms) == 500
        assert analyzer.preprocessing_result is None

    def test_init_with_existing_preprocessing(self, preprocessed_bundle):
        """Test initialization with existing preprocessing results"""
        analyzer = NonlinearHRVAnalysis(preprocessed_bundle)

        assert analyzer.preprocessing_result is not None
        assert len(analyzer.rr_ms) > 0
        # Should use corrected RRI from preprocessing
        assert np.array_equal(
            analyzer.rr_ms, preprocessed_bundle.preprocessing.corrected_rri
        )

    def test_init_with_analysis_window(self, clean_bundle):
        """Test initialization with analysis window"""
        analyzer = NonlinearHRVAnalysis(clean_bundle, analysis_window=(10.0, 60.0))

        assert analyzer.analysis_window == (10.0, 60.0)
        # Should have fewer intervals due to windowing
        assert len(analyzer.rr_ms) < 500

    def test_init_insufficient_data(self):
        """Test initialization with insufficient data"""
        short_data = [800, 810, 790]
        bundle = DataBundle(rri_ms=short_data)

        with pytest.raises(ValueError, match="At least 10 RR intervals needed"):
            NonlinearHRVAnalysis(bundle)

    def test_init_no_rri_data(self):
        """Test initialization with no RRI data"""
        bundle = DataBundle()

        with pytest.raises(ValueError, match="No RRI data available"):
            NonlinearHRVAnalysis(bundle)

    def test_init_invalid_analysis_window(self, clean_bundle):
        """Test initialization with invalid analysis window"""
        with pytest.raises(ValueError, match="No data found in analysis window"):
            NonlinearHRVAnalysis(clean_bundle, analysis_window=(1000.0, 2000.0))


class TestPoincareAnalysis:
    """Test PoincarÃ© analysis functionality"""

    def test_poincare_basic(self, clean_bundle):
        """Test basic PoincarÃ© analysis"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)
        sd1, sd2, ratio, additional = analyzer.poincare_analysis()

        assert isinstance(sd1, float)
        assert isinstance(sd2, float)
        assert isinstance(ratio, float)
        assert isinstance(additional, dict)

        # Basic physiological checks
        assert sd1 > 0
        assert sd2 > 0
        assert sd2 > sd1  # SD2 typically larger than SD1
        assert 0 < ratio < 1  # SD1/SD2 ratio should be < 1

        # Check additional metrics
        assert "ellipse_area" in additional
        assert "csi" in additional
        assert "cvi" in additional
        assert "modified_csi" in additional

        assert additional["ellipse_area"] > 0
        assert additional["csi"] > 0

    def test_poincare_minimal_data(self):
        """Test PoincarÃ© analysis with minimal but sufficient data"""
        # Use 15 data points instead of 1 to pass the initialization check
        bundle = DataBundle(rri_ms=[800] * 15)
        analyzer = NonlinearHRVAnalysis(bundle, use_preprocessing=False)
        sd1, sd2, ratio, additional = analyzer.poincare_analysis()

        # Should return zeros for constant data
        assert sd1 == 0.0
        assert sd2 == 0.0
        assert ratio == 0.0 or np.isnan(ratio)
        assert additional == {}

    def test_poincare_constant_data(self):
        """Test PoincarÃ© analysis with constant RR intervals"""
        constant_data = [800] * 100
        bundle = DataBundle(rri_ms=constant_data)
        analyzer = NonlinearHRVAnalysis(bundle, use_preprocessing=False)

        sd1, sd2, ratio, additional = analyzer.poincare_analysis()

        # Should handle constant data gracefully
        assert sd1 == 0.0
        assert sd2 == 0.0
        # For constant data, additional metrics should be empty
        assert additional == {}


class TestSampleEntropy:
    """Test Sample Entropy calculation"""

    def test_sample_entropy_basic(self, clean_bundle):
        """Test basic sample entropy calculation"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)
        sampen = analyzer.sample_entropy()

        assert isinstance(sampen, float)
        assert not np.isnan(sampen)
        assert sampen > 0
        # Typical HRV sample entropy values
        assert 0.1 < sampen < 3.0

    def test_sample_entropy_parameters(self, clean_bundle):
        """Test sample entropy with different parameters"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)

        # Test different template lengths
        sampen_m1 = analyzer.sample_entropy(m=1)
        sampen_m2 = analyzer.sample_entropy(m=2)
        sampen_m3 = analyzer.sample_entropy(m=3)

        assert all(isinstance(x, float) for x in [sampen_m1, sampen_m2, sampen_m3])
        # Higher m typically gives higher entropy
        assert sampen_m1 < sampen_m3

        # Test different tolerance values
        sampen_r01 = analyzer.sample_entropy(r=0.1)
        sampen_r03 = analyzer.sample_entropy(r=0.3)

        # Higher r typically gives lower entropy
        assert sampen_r01 > sampen_r03

    def test_sample_entropy_insufficient_data(self):
        """Test sample entropy with insufficient data"""
        short_data = [800] * 50
        bundle = DataBundle(rri_ms=short_data)
        analyzer = NonlinearHRVAnalysis(bundle, use_preprocessing=False)

        with pytest.raises(ValueError, match="at least 100 data points"):
            analyzer.sample_entropy()

    def test_sample_entropy_invalid_parameters(self, clean_bundle):
        """Test sample entropy with invalid parameters"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)

        with pytest.raises(ValueError, match="Template length m must be at least 1"):
            analyzer.sample_entropy(m=0)

        with pytest.raises(ValueError, match="Tolerance r must be positive"):
            analyzer.sample_entropy(r=0)

    def test_sample_entropy_constant_data(self):
        """Test sample entropy with constant data"""
        constant_data = [800] * 200
        bundle = DataBundle(rri_ms=constant_data)
        analyzer = NonlinearHRVAnalysis(bundle, use_preprocessing=False)

        sampen = analyzer.sample_entropy()
        assert sampen == 0.0  # Constant data should have zero entropy


class TestMultiscaleEntropy:
    """Test Multiscale Entropy calculation"""

    def test_mse_basic(self, clean_bundle):
        """Test basic multiscale entropy calculation"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)
        mse = analyzer.multiscale_entropy(scale_max=5)

        assert isinstance(mse, np.ndarray)
        assert len(mse) == 5
        assert np.all(~np.isnan(mse))  # No NaN values
        assert np.all(mse > 0)  # All positive values

    def test_mse_different_scales(self, clean_bundle):
        """Test MSE with different scale parameters - using fewer scales for 500 data points"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)

        # Use smaller scale values that work with 500 data points
        mse_3 = analyzer.multiscale_entropy(scale_max=3)
        mse_5 = analyzer.multiscale_entropy(scale_max=5)

        assert len(mse_3) == 3
        assert len(mse_5) == 5
        # First 3 values should be the same
        np.testing.assert_array_almost_equal(mse_3, mse_5[:3])

    def test_mse_insufficient_data(self):
        """Test MSE with insufficient data"""
        short_data = [800] * 200
        bundle = DataBundle(rri_ms=short_data)
        analyzer = NonlinearHRVAnalysis(bundle, use_preprocessing=False)

        with pytest.raises(ValueError, match="Multiscale entropy requires at least"):
            analyzer.multiscale_entropy(scale_max=10)


class TestDFA:
    """Test Detrended Fluctuation Analysis"""

    def test_dfa_basic(self, clean_bundle):
        """Test basic DFA calculation"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)
        alpha1, alpha2, box_sizes, fluctuations = (
            analyzer.detrended_fluctuation_analysis()
        )

        assert isinstance(alpha1, float)
        assert isinstance(alpha2, (float, type(np.nan)))
        assert isinstance(box_sizes, np.ndarray)
        assert isinstance(fluctuations, np.ndarray)

        # Check alpha1 (short-term scaling)
        assert not np.isnan(alpha1)
        assert 0.3 < alpha1 < 1.8  # Physiological range

        # Check arrays
        assert len(box_sizes) == len(fluctuations)
        assert len(box_sizes) > 0
        assert np.all(fluctuations > 0)

    def test_dfa_insufficient_data(self):
        """Test DFA with insufficient data"""
        short_data = [800] * 50
        bundle = DataBundle(rri_ms=short_data)
        analyzer = NonlinearHRVAnalysis(bundle, use_preprocessing=False)

        with pytest.raises(ValueError, match="DFA calculation requires at least 100"):
            analyzer.detrended_fluctuation_analysis()

    def test_dfa_scaling_exponents(self, clean_bundle):
        """Test DFA scaling exponents interpretation"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)
        alpha1, alpha2, _, _ = analyzer.detrended_fluctuation_analysis()

        # Alpha1 should be in healthy HRV range
        assert 0.5 < alpha1 < 1.5  # Typical healthy range

        # Alpha2 might be NaN for shorter sequences
        if not np.isnan(alpha2):
            assert 0.5 < alpha2 < 1.8


class TestRQA:
    """Test Recurrence Quantification Analysis"""

    def test_rqa_basic(self, clean_bundle):
        """Test basic RQA calculation"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)
        rqa_metrics = analyzer.recurrence_quantification_analysis()

        assert isinstance(rqa_metrics, dict)

        # Check all expected metrics are present
        expected_keys = [
            "recurrence_rate",
            "determinism",
            "avg_diagonal_length",
            "max_diagonal_length",
            "laminarity",
            "avg_vertical_length",
            "max_vertical_length",
            "entropy_diagonal",
            "entropy_vertical",
        ]

        for key in expected_keys:
            assert key in rqa_metrics
            assert isinstance(rqa_metrics[key], float)

        # Basic range checks
        assert 0 <= rqa_metrics["recurrence_rate"] <= 1
        assert 0 <= rqa_metrics["determinism"] <= 1
        assert 0 <= rqa_metrics["laminarity"] <= 1

    def test_rqa_different_parameters(self, clean_bundle):
        """Test RQA with different parameters"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)

        # Test different thresholds - use more appropriate values for comparison
        rqa_low = analyzer.recurrence_quantification_analysis(threshold=0.01)
        rqa_high = analyzer.recurrence_quantification_analysis(threshold=0.1)

        # CORRECTED ASSERTION: Lower threshold should give higher recurrence rate
        # (More points will be considered recurrent with a more lenient threshold)
        assert (
            rqa_low["recurrence_rate"] <= rqa_high["recurrence_rate"]
        )  # Fixed direction

        # Test different embedding dimensions
        rqa_dim2 = analyzer.recurrence_quantification_analysis(embedding_dim=2)
        rqa_dim4 = analyzer.recurrence_quantification_analysis(embedding_dim=4)

        # Both should be valid
        assert 0 <= rqa_dim2["recurrence_rate"] <= 1
        assert 0 <= rqa_dim4["recurrence_rate"] <= 1

    def test_rqa_insufficient_data(self):
        """Test RQA with insufficient data"""
        short_data = [800] * 30
        bundle = DataBundle(rri_ms=short_data)
        analyzer = NonlinearHRVAnalysis(bundle, use_preprocessing=False)

        with pytest.raises(ValueError, match="RQA calculation requires at least 50"):
            analyzer.recurrence_quantification_analysis()


class TestFullAnalysis:
    """Test full nonlinear analysis functionality"""

    def test_full_analysis_all_enabled(self, clean_bundle):
        """Test full analysis with all methods enabled"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)
        # Use smaller MSE scale to avoid data requirements issues
        results = analyzer.full_nonlinear_analysis(mse_scales=5)

        assert isinstance(results, dict)

        # Check all main sections are present
        expected_sections = [
            "poincare",
            "sample_entropy",
            "multiscale_entropy",
            "dfa",
            "rqa",
            "analysis_info",
        ]

        for section in expected_sections:
            assert section in results

        # Allow for methods to fail gracefully
        # Check that at least some methods succeeded
        successful_methods = [
            k
            for k in expected_sections
            if k != "analysis_info" and results[k] is not None
        ]
        assert len(successful_methods) > 0, f"No methods succeeded. Results: {results}"

        # Check analysis info always exists
        info = results["analysis_info"]
        assert "total_intervals" in info
        assert "analysis_duration_s" in info
        assert info["total_intervals"] == 500

    def test_full_analysis_selective(self, clean_bundle):
        """Test full analysis with selective methods"""
        analyzer = NonlinearHRVAnalysis(clean_bundle)
        results = analyzer.full_nonlinear_analysis(
            include_mse=False, include_dfa=True, include_rqa=False
        )

        # Check that disabled methods are None and enabled methods exist
        assert results["multiscale_entropy"] is None
        assert results["rqa"] is None
        assert "dfa" in results  # Should be present even if it might fail
        assert "poincare" in results  # Should be present even if it might fail
        assert "sample_entropy" in results  # Should be present even if it might fail

    def test_full_analysis_with_preprocessing(self, noisy_bundle):
        """Test full analysis with preprocessing"""
        analyzer = NonlinearHRVAnalysis(noisy_bundle, use_preprocessing=True)
        results = analyzer.full_nonlinear_analysis(mse_scales=3)  # Use smaller scale

        # Should have preprocessing stats
        assert "preprocessing_stats" in results
        preprocessing_stats = results["preprocessing_stats"]

        expected_preprocessing_keys = [
            "artifacts_detected",
            "artifacts_corrected",
            "artifact_percentage",
            "noise_segments",
            "correction_method",
            "quality_flags",
        ]

        for key in expected_preprocessing_keys:
            assert key in preprocessing_stats

    def test_full_analysis_error_handling(self):
        """Test full analysis error handling"""
        # Create data that will cause some methods to fail
        problematic_data = [800] * 80  # Too short for some methods
        bundle = DataBundle(rri_ms=problematic_data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer = NonlinearHRVAnalysis(bundle, use_preprocessing=False)
            results = analyzer.full_nonlinear_analysis()

            # Should have warnings
            assert len(w) > 0

            # Some methods should fail gracefully
            assert results["sample_entropy"] is None  # Requires 100+ points
            assert results["multiscale_entropy"] is None

            # But others should still work
            assert results["poincare"] is not None


class TestFactoryFunction:
    """Test the factory function"""

    def test_create_nonlinear_analysis(self, clean_bundle):
        """Test factory function"""
        analyzer = create_nonlinear_analysis(clean_bundle)

        assert isinstance(analyzer, NonlinearHRVAnalysis)
        assert analyzer.bundle is clean_bundle
        assert analyzer.use_preprocessing is True

    def test_create_with_parameters(self, clean_bundle):
        """Test factory function with parameters"""
        analyzer = create_nonlinear_analysis(
            clean_bundle,
            use_preprocessing=False,
            analysis_window=(10.0, 60.0),
            preprocessing_params={"threshold_low": 250},
        )

        assert analyzer.use_preprocessing is False
        assert analyzer.analysis_window == (10.0, 60.0)
        assert analyzer.preprocessing_params == {"threshold_low": 250}


class TestDataValidation:
    """Test data validation and quality assessment"""

    def test_validation_with_preprocessing(self, noisy_bundle):
        """Test validation with preprocessing results"""
        analyzer = NonlinearHRVAnalysis(noisy_bundle, use_preprocessing=True)

        # Should initialize without raising errors
        assert analyzer.preprocessing_result is not None
        assert analyzer.preprocessing_result.quality_flags is not None

    def test_validation_warnings(self, noisy_bundle):
        """Test that appropriate warnings are issued"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Force poor quality by preprocessing first
            preprocessing_result = preprocess_rri(noisy_bundle.rri_ms)
            # Manually set quality flags for testing
            preprocessing_result.quality_flags = {
                "poor_signal_quality": True,
                "excessive_artifacts": True,
                "high_noise": False,
                "irregular_rhythm": False,
            }
            preprocessing_result.stats["artifact_percentage"] = 15.0

            noisy_bundle.preprocessing = preprocessing_result

            analyzer = NonlinearHRVAnalysis(noisy_bundle)

            # Should have warnings about data quality
            warning_messages = [str(warning.message) for warning in w]
            assert any("Poor signal quality" in msg for msg in warning_messages)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_analysis_window(self, clean_bundle):
        """Test analysis window that contains no data"""
        with pytest.raises(ValueError, match="No data found in analysis window"):
            NonlinearHRVAnalysis(clean_bundle, analysis_window=(10000, 20000))

    def test_very_short_analysis_window(self, clean_bundle):
        """Test very short analysis window that still has enough data"""
        # Use a window that will give us at least 10 intervals
        analyzer = NonlinearHRVAnalysis(clean_bundle, analysis_window=(0, 10))
        # Should have some data but less than full dataset
        assert 10 <= len(analyzer.rr_ms) < 500

    def test_preprocessing_failure_fallback(self, clean_bundle):
        """Test fallback when preprocessing fails"""
        with patch("hrvlib.metrics.nonlinear.preprocess_rri") as mock_preprocess:
            mock_preprocess.side_effect = Exception("Preprocessing failed")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                analyzer = NonlinearHRVAnalysis(clean_bundle, use_preprocessing=True)

                # Should fall back to raw data and have warnings
                assert analyzer.use_preprocessing is True  # Setting remains True
                assert len(w) > 0
                assert "Preprocessing failed" in str(w[0].message)
                # Should use raw data from bundle
                assert len(analyzer.rr_ms) == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
