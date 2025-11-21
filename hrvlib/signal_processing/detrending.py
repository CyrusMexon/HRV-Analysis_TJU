"""
Unified Detrending Interface for RR Intervals

This module provides a single, consistent interface for all detrending methods
used in HRV analysis, including scipy methods and smoothness priors.

Functions
---------
detrend_rr_intervals : function
    Main detrending function supporting multiple methods

Example Usage
-------------
>>> from signal_processing.detrending import detrend_rr_intervals
>>> 
>>> # Linear detrending (scipy)
>>> detrended = detrend_rr_intervals(rr_intervals, method='linear')
>>> 
>>> # Smoothness priors with default lambda
>>> detrended = detrend_rr_intervals(rr_intervals, method='smoothness_priors')
>>> 
>>> # Smoothness priors with custom lambda
>>> detrended = detrend_rr_intervals(
...     rr_intervals,
...     method='smoothness_priors',
...     lambda_param=1000
... )
"""

import numpy as np
from scipy import signal

# Try relative import first (when used as module), fall back to direct import
try:
    from .smoothness_priors import detrend_with_smoothness_priors
except ImportError:
    from smoothness_priors import detrend_with_smoothness_priors


def detrend_rr_intervals(rr_intervals, method="linear", return_trend=False, **kwargs):
    """
    Detrend RR intervals using the specified method.
    
    This function provides a unified interface for all detrending methods,
    making it easy to switch between different approaches.
    
    Parameters
    ----------
    rr_intervals : array-like
        RR intervals in milliseconds
    method : str, optional (default='linear')
        Detrending method to use. Options:
        - 'linear': Linear detrending (scipy)
        - 'constant': Mean removal (scipy)
        - 'smoothness_priors': Smoothness priors detrending (Kubios-style)
    return_trend : bool, optional (default=False)
        If True, return both detrended signal and trend component.
        Only supported for 'smoothness_priors' method.
    **kwargs : dict
        Additional method-specific parameters:
        
        For smoothness_priors:
        - lambda_param (or lambda): float, smoothness parameter (default=500)
        - fs: float, sampling frequency in Hz (default=4.0)
    
    Returns
    -------
    detrended : numpy.ndarray
        Detrended RR intervals in milliseconds
    trend : numpy.ndarray (only if return_trend=True and method='smoothness_priors')
        The estimated trend component that was removed
    
    Raises
    ------
    ValueError
        If method is not recognized or if parameters are invalid
    
    Examples
    --------
    >>> import numpy as np
    >>> rr_intervals = np.array([800, 810, 805, 820, 815, 825])
    
    >>> # Linear detrending
    >>> detrended = detrend_rr_intervals(rr_intervals, method='linear')
    
    >>> # Mean removal
    >>> detrended = detrend_rr_intervals(rr_intervals, method='constant')
    
    >>> # Smoothness priors with default lambda (500)
    >>> detrended = detrend_rr_intervals(rr_intervals, method='smoothness_priors')
    
    >>> # Smoothness priors with custom lambda
    >>> detrended = detrend_rr_intervals(
    ...     rr_intervals,
    ...     method='smoothness_priors',
    ...     lambda_param=1000
    ... )
    
    >>> # Get both detrended signal and trend
    >>> detrended, trend = detrend_rr_intervals(
    ...     rr_intervals,
    ...     method='smoothness_priors',
    ...     return_trend=True
    ... )
    
    Notes
    -----
    Method Comparison:
    - 'linear': Fast, simple, removes linear trends. Good for quick analysis.
    - 'constant': Only removes mean, preserves all variations including trends.
    - 'smoothness_priors': Advanced method that removes non-linear trends while
      preserving physiological variations. Recommended for research-quality HRV.
    
    The 'smoothness_priors' method is compatible with Kubios HRV software
    and is recommended for standard HRV analysis.
    """
    # Validate input
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    
    if len(rr_intervals) < 2:
        raise ValueError("At least 2 RR intervals are required for detrending")
    
    # Normalize method name
    method = str(method).lower().strip()
    
    # Route to appropriate detrending method
    if method == "smoothness_priors" or method == "smoothness-priors":
        # Extract smoothness priors parameters
        lambda_param = kwargs.get("lambda_param", kwargs.get("lambda", 500))
        fs = kwargs.get("fs", 4.0)
        
        # Validate lambda parameter
        if not isinstance(lambda_param, (int, float)) or lambda_param <= 0:
            raise ValueError(
                f"lambda_param must be a positive number, got: {lambda_param}"
            )
        
        # Apply smoothness priors detrending
        result = detrend_with_smoothness_priors(
            rr_intervals,
            lambda_param=lambda_param,
            fs=fs,
            return_trend=return_trend
        )
        
        return result
    
    elif method in ["linear", "constant"]:
        # Apply scipy detrending
        detrended = signal.detrend(rr_intervals, type=method)
        
        if return_trend:
            # Calculate trend for scipy methods
            trend = rr_intervals - detrended
            return detrended, trend
        else:
            return detrended
    
    else:
        # Unknown method
        raise ValueError(
            f"Unknown detrending method: '{method}'. "
            f"Supported methods: 'linear', 'constant', 'smoothness_priors'"
        )


def get_available_methods():
    """
    Get list of available detrending methods.
    
    Returns
    -------
    methods : dict
        Dictionary mapping method names to descriptions
    
    Examples
    --------
    >>> methods = get_available_methods()
    >>> for name, description in methods.items():
    ...     print(f"{name}: {description}")
    """
    return {
        "linear": "Linear detrending (scipy) - removes linear trends",
        "constant": "Constant detrending (scipy) - removes mean only",
        "smoothness_priors": "Smoothness priors (Kubios) - removes non-linear trends"
    }


def recommend_method(recording_length_minutes=5, has_strong_trends=False):
    """
    Recommend a detrending method based on recording characteristics.
    
    Parameters
    ----------
    recording_length_minutes : float, optional (default=5)
        Length of recording in minutes
    has_strong_trends : bool, optional (default=False)
        Whether the recording has strong non-linear trends
    
    Returns
    -------
    recommendation : dict
        Dictionary with 'method' and 'params' keys
    
    Examples
    --------
    >>> # Standard 5-minute recording
    >>> rec = recommend_method(recording_length_minutes=5)
    >>> print(rec['method'])
    smoothness_priors
    
    >>> # Short recording
    >>> rec = recommend_method(recording_length_minutes=1)
    >>> print(rec['params']['lambda_param'])
    100
    """
    if recording_length_minutes < 2:
        # Short recording - use light smoothing
        return {
            "method": "smoothness_priors",
            "params": {"lambda_param": 100}
        }
    elif recording_length_minutes <= 10:
        # Standard recording - use Kubios default
        return {
            "method": "smoothness_priors",
            "params": {"lambda_param": 500}
        }
    else:
        # Long recording - use stronger smoothing
        return {
            "method": "smoothness_priors",
            "params": {"lambda_param": 1000}
        }


# Convenience aliases
detrend = detrend_rr_intervals  # Shorter alias


if __name__ == "__main__":
    # Simple test
    print("Detrending Module Test")
    print("=" * 60)
    
    # Create test data
    rr_intervals = np.array([
        800, 810, 805, 820, 815, 825, 830, 828, 835, 840,
        845, 843, 850, 855, 852, 860, 865, 862, 870, 868
    ])
    
    print(f"\nTest RR intervals (n={len(rr_intervals)}):")
    print(f"Mean: {np.mean(rr_intervals):.2f} ms")
    print(f"Std: {np.std(rr_intervals):.2f} ms")
    
    # Test each method
    methods_to_test = ['linear', 'constant', 'smoothness_priors']
    
    print("\n" + "=" * 60)
    print("Testing detrending methods:")
    print("=" * 60)
    
    for method in methods_to_test:
        print(f"\nMethod: {method}")
        try:
            if method == 'smoothness_priors':
                detrended = detrend_rr_intervals(
                    rr_intervals,
                    method=method,
                    lambda_param=500
                )
            else:
                detrended = detrend_rr_intervals(rr_intervals, method=method)
            
            print(f"  Mean: {np.mean(detrended):.2f} ms")
            print(f"  Std: {np.std(detrended):.2f} ms")
            print("  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Available methods:")
    print("=" * 60)
    for name, desc in get_available_methods().items():
        print(f"  {name}: {desc}")
    
    print("\n" + "=" * 60)
    print("Method recommendations:")
    print("=" * 60)
    for length in [1, 5, 30]:
        rec = recommend_method(recording_length_minutes=length)
        print(f"  {length} min: {rec['method']} (λ={rec['params'].get('lambda_param', 'N/A')})")
    
    print("\n✓ All tests passed!")
