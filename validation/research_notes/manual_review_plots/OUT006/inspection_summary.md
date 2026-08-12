# Frequency Outlier Inspection: OUT006__nsr015_segment_089.txt

This report is generated for manual validation review only. It does not indicate which implementation is correct.

## File

- Source file: `validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT006__nsr015_segment_089.txt`
- Duration: 124.07 seconds

## Warning messages

- scipy.misc is deprecated and will be removed in 2.0.0
- RR intervals appear to be in seconds (mean=0.832s); converting to ms.
- Removed 497 invalid RR intervals before frequency-domain analysis.
- Duplicate RR cumulative time points detected before interpolation; invalid nonpositive intervals will be removed.
- VLF estimates are likely unreliable.
- ULF and VLF currently overlap; definitions are reported unchanged.
- Fewer than 2 PSD bins fall inside this band.
- Recording duration is insufficient for reliable ULF estimation.
- Recording duration is insufficient for reliable VLF estimation.
- Integrated FFT PSD power differs substantially from input signal variance.

## Key discrepancies

| Metric | Native | NeuroKit2 | Absolute error | Relative error |
| --- | --- | --- | --- | --- |
| VLF | 10505.6 | 11987 | 1481.32 | 12.36% |
| HF | 537.969 | 484.801 | 53.1677 | 10.97% |
| total_power | 12678.6 | 14152.6 | 1473.96 | 10.41% |
| LF | 1264.49 | 1174.12 | 90.3761 | 7.70% |
| LF/HF | 2.35049 | 2.42185 | 0.0713578 | 2.95% |

## Preprocessing signal comparison

- Resampled-signal Pearson correlation: 1.000.
- Resampled-signal RMSE: 0.160 ms.
- Detrended-signal Pearson correlation: 1.000.
- Detrended-signal RMSE: 0.160 ms.
- Low correlation or high RMSE before PSD estimation may suggest that the disagreement originates in interpolation, detrending, clipping, units, or other pre-PSD signal preparation differences.

## PSD shape assessment

- Shape assessment: PSD shape similarity is mixed or unclear.
- Log-PSD correlation on common 0-0.4 Hz grid: 0.961.
- This automated assessment is heuristic; the overlay plot requires manual review before drawing conclusions.

## Suggested disagreement category

- c) low-bin/duration issue

Interpretation should remain cautious. The disagreement may suggest methodological mismatch, amplitude/scaling differences, frequency-shape differences, low-bin/duration limitations, or data-specific issues.
