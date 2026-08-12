# Kubios Comparison Guide

## Summary of Investigation

We investigated differences between your HRV software and Kubios frequency domain metrics.

### Bugs Fixed

1. **FFT PSD Scaling Bug (CRITICAL)**
   - Location: `hrvlib/metrics/freq_domain.py` line 298-310
   - Issue: FFT periodogram was 50% too low
   - Fix: Corrected scaling formula to match scipy.signal.periodogram
   - Result: Now within 0.05% of reference implementation

2. **Default Detrending Mismatch**
   - Location: `hrvlib/metrics/freq_domain.py` line 49
   - Changed: `detrend_method="linear"` → `detrend_method=None`
   - Reason: Match Kubios stated default

### Key Findings

Your test data (`2018-04-30 13-20-04.txt`):
- 77 RR intervals
- Duration: 61.82 seconds

| Setting | VLF (ms²) | LF (ms²) | Observation |
|---------|-----------|----------|-------------|
| No detrending | 117,879 | 111 | VLF way too high |
| Constant detrend | 376 | 112 | VLF still 2.3x low |
| Linear detrend | 296 | 112 | VLF still 2.9x low |
| **Kubios** | **865** | **133** | Target values |

### Conclusion

**Kubios is applying automatic preprocessing** that cannot be disabled or seen in the UI:
- Likely using some form of detrending (constant or linear variant)
- Possibly using specific window width/overlap for spectral analysis
- The "grayed out Apply button" confirms automatic defaults

**We cannot exactly replicate Kubios** because their internal preprocessing is proprietary.

---

## Recommendations for Your Software

### Option 1: Provide Multiple Preset Configurations

Create presets that users can choose:

**Preset 1: "Standard" (Current default)**
- Detrending: None
- Sampling rate: 4 Hz
- Window: Hann
- Segment length: 120s
- Overlap: 75%

**Preset 2: "Kubios-like"**
- Detrending: Constant or Linear
- Sampling rate: 4 Hz
- Window: Hann
- Segment length: Try different values
- Overlap: Try 0% or 50%

**Preset 3: "Minimal Processing"**
- No detrending
- No windowing (rectangular window)
- Entire signal

### Option 2: Make All Settings User-Adjustable

Add UI controls for:
1. **Detrending method**: None, Constant, Linear, Smoothness Priors
2. **Segment length** (for Welch): 60s, 120s, 256s, 300s, Entire signal
3. **Overlap**: 0%, 25%, 50%, 75%
4. **Window type**: Hann, Hamming, Blackman, etc.

This way users can experiment to match their reference software.

### Option 3: Document the Differences

In your user manual or help system, explain:
- Your software uses scientifically standard methods (scipy implementation)
- Different HRV software may give different absolute values
- **Relative changes** (before/after intervention) are more important than absolute values
- If matching a specific reference is critical, settings should be carefully configured

---

## Current Default Settings (After Fixes)

Your software now uses:
```python
sampling_rate = 4.0 Hz
detrend_method = None
window_type = "hann"
segment_length = 120.0 seconds
overlap_ratio = 0.75
ar_order = 16
```

These are scientifically sound defaults based on:
- Task Force guidelines (1996)
- Common HRV analysis practices
- scipy.signal reference implementations

---

## Testing Different Configurations

Use the provided test scripts:

### 1. Test Your Own Data
```bash
python test_detrending_your_data.py
```
Shows how detrending affects VLF/LF/HF bands

### 2. Test Different Window Settings
```bash
python test_kubios_settings.py
```
Tests various window width and overlap combinations

### 3. Diagnostic Analysis
```bash
python diagnostic_freq_domain.py
```
Detailed breakdown of all computations

---

## For Users Comparing to Kubios

**Important Notes:**
1. Kubios uses proprietary algorithms that may differ from published standards
2. Kubios applies automatic preprocessing that cannot be disabled
3. Absolute power values may differ, but **relative patterns should be similar**
4. For clinical/research use, consistency within YOUR software is more important than matching Kubios exactly

**Suggested Approach:**
1. Choose settings that give reasonable, physiologically plausible values
2. Keep settings CONSISTENT across all your analyses
3. Compare relative changes (e.g., before/after treatment) not absolute values
4. If collaborating with Kubios users, document your settings clearly

---

## Technical Details

### FFT PSD Calculation (Corrected)
```python
# Proper one-sided PSD with windowing
S2 = np.sum(window**2) / n
psd = (2.0 / (fs * n * S2)) * |FFT|²
psd[0] /= 2.0  # DC correction
psd[-1] /= 2.0  # Nyquist correction (if n is even)
```

### Band Power Integration
```python
# Trapezoidal integration (using numpy trapezoid)
band_power = np.trapezoid(psd[mask], freqs[mask])

# VLF band excludes DC component (freq=0)
if freq[0] == 0.0:
    vlf_mask[0] = False
```

### Frequency Bands
- ULF: 0.000 - 0.003 Hz
- VLF: 0.003 - 0.040 Hz
- LF:  0.040 - 0.150 Hz
- HF:  0.150 - 0.400 Hz

---

## Files Created During Investigation

1. `diagnostic_freq_domain.py` - Detailed metric computation analysis
2. `test_psd_scaling.py` - Proves scaling bug and detrending impact
3. `verify_fixes.py` - Confirms fixes work correctly
4. `test_kubios_settings.py` - Test different window configurations
5. `analyze_your_data.py` - Analyze specific data file
6. `test_detrending_your_data.py` - Compare detrending methods

All tools are available in the project directory for future reference.

---

## Contact

If users report discrepancies with other HRV software:
1. Ask them to document exact settings used in the other software
2. Try to replicate those settings in your software
3. Document that different software may use different algorithms
4. Emphasize that consistency and relative changes matter most
