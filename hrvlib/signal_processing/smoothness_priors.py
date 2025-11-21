"""
Smoothness Priors Detrending for HRV Analysis

This module implements the smoothness priors approach for detrending RR interval
time series, as used in Kubios HRV software. The method uses a regularization
parameter lambda (λ) to balance between fitting the data and enforcing smoothness.

References:
- Tarvainen et al. (2002). "An advanced detrending method with application to HRV analysis"
- Kubios HRV User's Guide

Author: HRV Analysis Implementation
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# Convenience function for integration with existing code
def detrend_with_smoothness_priors(
    rr_intervals, lambda_param=500, fs=4.0, return_trend=False
):
    """
    Simplified interface for smoothness priors detrending.

    Parameters
    ----------
    rr_intervals : array-like
        RR intervals in milliseconds
    lambda_param : float, optional (default=500)
        Smoothness parameter
    fs : float, optional (default=4.0)
        Sampling frequency in Hz
    return_trend : bool, optional (default=False)
        If True, return both detrended signal and trend

    Returns
    -------
    detrended : numpy.ndarray
        Detrended RR intervals
    trend : numpy.ndarray (only if return_trend=True)
        The estimated trend component
    """
    detrended, trend = smoothness_priors_detrending(rr_intervals, lambda_param, fs)

    if return_trend:
        return detrended, trend
    return detrended


def smoothness_priors_detrending(rr_intervals, lambda_param=500, fs=4.0):
    """
    Apply smoothness priors detrending to RR interval time series.

    This method removes low-frequency trends while preserving high-frequency
    variations that are physiologically relevant for HRV analysis.

    Parameters
    ----------
    rr_intervals : array-like
        RR intervals in milliseconds
    lambda_param : float, optional (default=500)
        Smoothness parameter (λ). Controls the degree of smoothing:
        - Lower values (10-100): Less smoothing, preserves more variation
        - Medium values (100-500): Moderate smoothing (recommended for HRV)
        - Higher values (500-1000+): Aggressive smoothing, removes more trends
        Common Kubios values: 10, 50, 100, 200, 500, 1000
    fs : float, optional (default=4.0)
        Sampling frequency in Hz for creating uniformly sampled signal
        (typically 4 Hz for HRV analysis, matching Kubios default)

    Returns
    -------
    detrended : numpy.ndarray
        Detrended RR intervals in milliseconds (same length as input)
    trend : numpy.ndarray
        The estimated trend component that was removed

    Examples
    --------
    >>> rr_intervals = np.array([800, 810, 805, 820, 815, 825])
    >>> detrended, trend = smoothness_priors_detrending(rr_intervals, lambda_param=500)
    >>> print(detrended)

    Notes
    -----
    The algorithm:
    1. Converts irregular RR intervals to uniformly sampled time series
    2. Constructs a penalty matrix for second-order differences
    3. Solves: (I + λ²D'D)z = y where z is the trend
    4. Returns detrended signal: y - z

    The lambda parameter determines smoothness:
    - λ → 0: No smoothing (trend follows data exactly)
    - λ → ∞: Maximum smoothing (trend becomes linear)
    """
    # Input validation
    rr_intervals = np.asarray(rr_intervals, dtype=float)

    if len(rr_intervals) < 4:
        raise ValueError(
            "Need at least 4 RR intervals for smoothness priors detrending"
        )

    if lambda_param <= 0:
        raise ValueError("Lambda parameter must be positive")

    # Step 1: Create time axis from cumulative sum of RR intervals
    # Convert RR intervals (ms) to time (seconds)
    rr_seconds = rr_intervals / 1000.0
    time_original = np.concatenate([[0], np.cumsum(rr_seconds[:-1])])

    # Step 2: Create uniformly sampled time series
    # This is necessary because smoothness priors work on evenly spaced data
    time_uniform = np.arange(0, time_original[-1], 1.0 / fs)

    # Interpolate RR intervals to uniform sampling
    rr_uniform = np.interp(time_uniform, time_original, rr_intervals)

    # Step 3: Apply smoothness priors algorithm
    N = len(rr_uniform)

    # Create second-order difference matrix (penalty on curvature)
    # This enforces smoothness by penalizing the second derivative
    identity = sparse.eye(N)
    D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N))

    # Solve the regularized least squares problem:
    # minimize: ||y - z||² + λ²||D²z||²
    # Solution: (I + λ²D'D)z = y
    H = identity + lambda_param**2 * D2.T @ D2

    # Convert to CSR format for efficient solving
    H = H.tocsr()

    # Solve for the trend
    trend_uniform = spsolve(H, rr_uniform)

    # Step 4: Interpolate trend back to original irregular sampling
    trend_original = np.interp(time_original, time_uniform, trend_uniform)

    # Step 5: Calculate detrended signal
    detrended = rr_intervals - trend_original

    return detrended, trend_original


def choose_lambda_for_hrv(analysis_type="standard"):
    """
    Get recommended lambda values for different HRV analysis scenarios.

    Parameters
    ----------
    analysis_type : str, optional (default="standard")
        Type of HRV analysis:
        - "standard": Standard 5-minute HRV analysis (λ=500)
        - "short": Short-term recordings <2 minutes (λ=100-200)
        - "long": Long-term recordings >30 minutes (λ=1000)
        - "minimal": Minimal detrending (λ=10-50)
        - "aggressive": Aggressive detrending (λ=1000-2000)

    Returns
    -------
    lambda_param : float
        Recommended lambda value

    Examples
    --------
    >>> lambda_val = choose_lambda_for_hrv("standard")
    >>> print(lambda_val)
    500
    """
    recommendations = {
        "standard": 500,  # Default Kubios value for 5-min recordings
        "short": 100,  # Less smoothing for short recordings
        "long": 1000,  # More smoothing for long recordings
        "minimal": 10,  # Preserve most variation
        "aggressive": 2000,  # Remove strong trends
    }

    return recommendations.get(analysis_type, 500)


def validate_detrending(rr_intervals, detrended, trend):
    """
    Validate that detrending was performed correctly.

    Parameters
    ----------
    rr_intervals : array-like
        Original RR intervals
    detrended : array-like
        Detrended RR intervals
    trend : array-like
        Estimated trend component

    Returns
    -------
    is_valid : bool
        True if detrending is valid
    metrics : dict
        Dictionary containing validation metrics
    """
    rr_intervals = np.asarray(rr_intervals)
    detrended = np.asarray(detrended)
    trend = np.asarray(trend)

    # Check reconstruction
    reconstructed = detrended + trend
    reconstruction_error = np.max(np.abs(reconstructed - rr_intervals))

    # Check mean removal
    original_mean = np.mean(rr_intervals)
    detrended_mean = np.mean(detrended)
    trend_mean = np.mean(trend)

    # Check variance
    original_var = np.var(rr_intervals)
    detrended_var = np.var(detrended)
    trend_var = np.var(trend)

    metrics = {
        "reconstruction_error": reconstruction_error,
        "original_mean": original_mean,
        "detrended_mean": detrended_mean,
        "trend_mean": trend_mean,
        "original_variance": original_var,
        "detrended_variance": detrended_var,
        "trend_variance": trend_var,
        "variance_ratio": detrended_var / original_var if original_var > 0 else 0,
    }

    is_valid = reconstruction_error < 1e-6  # Within numerical precision

    return is_valid, metrics


if __name__ == "__main__":
    # Example usage and testing
    print("Smoothness Priors Detrending - Example Usage\n")

    # Generate synthetic RR interval data with trend
    np.random.seed(42)
    n_beats = 300

    # Create baseline with trend
    time = np.arange(n_beats)
    trend_true = 800 + 0.2 * time + 30 * np.sin(2 * np.pi * time / 100)

    # Add HRV variation
    hrv_variation = 20 * np.sin(2 * np.pi * time / 20) + 10 * np.random.randn(n_beats)

    # Create RR intervals
    rr_synthetic = trend_true + hrv_variation

    # Test different lambda values
    lambda_values = [10, 100, 500, 1000]

    print("Testing different lambda values:\n")
    for lam in lambda_values:
        detrended, trend = smoothness_priors_detrending(rr_synthetic, lambda_param=lam)

        # Validate
        is_valid, metrics = validate_detrending(rr_synthetic, detrended, trend)

        print(f"Lambda = {lam}")
        print(f"  Valid: {is_valid}")
        print(f"  Reconstruction error: {metrics['reconstruction_error']:.6f}")
        print(f"  Variance ratio (detrended/original): {metrics['variance_ratio']:.3f}")
        print(f"  Original std: {np.sqrt(metrics['original_variance']):.2f} ms")
        print(f"  Detrended std: {np.sqrt(metrics['detrended_variance']):.2f} ms")
        print()

    print("\nRecommended lambda values by analysis type:")
    for analysis_type in ["minimal", "short", "standard", "long", "aggressive"]:
        lam = choose_lambda_for_hrv(analysis_type)
        print(f"  {analysis_type:12s}: λ = {lam}")
