# Frequency Outlier Inspection: OUT005__nsr021_segment_071.txt

This report is generated for manual validation review only. It does not indicate which implementation is correct.

## File

- Source file: `validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT005__nsr021_segment_071.txt`
- Duration: 107.11 seconds

## Warning messages

- scipy.misc is deprecated and will be removed in 2.0.0
- RR intervals appear to be in seconds (mean=0.487s); converting to ms.
- Removed 9 invalid RR intervals before frequency-domain analysis.
- Duplicate RR cumulative time points detected before interpolation; invalid nonpositive intervals will be removed.
- LF estimates should be interpreted cautiously.
- VLF estimates are likely unreliable.
- ULF and VLF currently overlap; definitions are reported unchanged.
- Fewer than 2 PSD bins fall inside this band.
- Recording duration is insufficient for reliable ULF estimation.
- Recording duration is insufficient for reliable VLF estimation.
- Recording duration is insufficient for reliable LF estimation.
- Integrated FFT PSD power differs substantially from input signal variance.

## Key discrepancies

| Metric | Native | NeuroKit2 | Absolute error | Relative error |
| --- | --- | --- | --- | --- |
| HF_nu | 18.9742 | 2.29177 | 16.6824 | 727.93% |
| LF | 689776 | 2.19209e+10 | 2.19202e+10 | 100.00% |
| VLF | 244070 | 6.05838e+09 | 6.05813e+09 | 100.00% |
| total_power | 1.46897e+06 | 3.14142e+10 | 3.14127e+10 | 100.00% |
| HF | 161528 | 5.1416e+08 | 5.13998e+08 | 99.97% |

## Preprocessing signal comparison

- Resampled-signal Pearson correlation: 0.347.
- Resampled-signal RMSE: 211493.079 ms.
- Detrended-signal Pearson correlation: 0.347.
- Detrended-signal RMSE: 211493.079 ms.
- Low correlation or high RMSE before PSD estimation may suggest that the disagreement originates in interpolation, detrending, clipping, units, or other pre-PSD signal preparation differences.

## PSD shape assessment

- Shape assessment: PSD shapes appear broadly similar.
- Log-PSD correlation on common 0-0.4 Hz grid: 0.908.
- This automated assessment is heuristic; the overlay plot requires manual review before drawing conclusions.

## Suggested disagreement category

- c) low-bin/duration issue

Interpretation should remain cautious. The disagreement may suggest methodological mismatch, amplitude/scaling differences, frequency-shape differences, low-bin/duration limitations, or data-specific issues.
