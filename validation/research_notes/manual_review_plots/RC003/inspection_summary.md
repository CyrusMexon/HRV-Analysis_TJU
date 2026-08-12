# Frequency Outlier Inspection: RC003__nsr002_segment_069.txt

This report is generated for manual validation review only. It does not indicate which implementation is correct.

## File

- Source file: `validation\kubios_subset\input_ascii_rr\random_controls\RC003__nsr002_segment_069.txt`
- Duration: 25.20 seconds

## Warning messages

- scipy.misc is deprecated and will be removed in 2.0.0
- RR intervals mean (44.20) is unusual for ms; verify input units.
- Signal duration < 1 minute. Frequency domain results may be unreliable.
- Frequency-domain HRV metrics may be unreliable.
- LF estimates should be interpreted cautiously.
- VLF estimates are likely unreliable.
- ULF and VLF currently overlap; definitions are reported unchanged.
- Fewer than 2 PSD bins fall inside this band.
- Recording duration is insufficient for reliable ULF estimation.
- Recording duration is insufficient for reliable VLF estimation.
- Recording duration is insufficient for reliable LF estimation.
- Recording duration is insufficient for reliable HF estimation.
- Integrated FFT PSD power differs substantially from input signal variance.

## Key discrepancies

| Metric | Native | NeuroKit2 | Absolute error | Relative error |
| --- | --- | --- | --- | --- |
| VLF | 0 | 31.4978 | 31.4978 | 100.00% |
| LF | 0 | 74.9192 | 74.9192 | 100.00% |
| LF/HF | 0 | 0.451619 | 0.451619 | 100.00% |
| LF_nu | 0 | 31.1114 | 31.1114 | 100.00% |
| total_power | 1.3487 | 302.628 | 301.279 | 99.55% |

## Preprocessing signal comparison

- Resampled-signal Pearson correlation: 0.510.
- Resampled-signal RMSE: 144.025 ms.
- Detrended-signal Pearson correlation: 0.510.
- Detrended-signal RMSE: 144.025 ms.
- Low correlation or high RMSE before PSD estimation may suggest that the disagreement originates in interpolation, detrending, clipping, units, or other pre-PSD signal preparation differences.

## PSD shape assessment

- Shape assessment: PSD shapes do not appear closely aligned.
- Log-PSD correlation on common 0-0.4 Hz grid: -0.404.
- This automated assessment is heuristic; the overlay plot requires manual review before drawing conclusions.

## Suggested disagreement category

- c) low-bin/duration issue

Interpretation should remain cautious. The disagreement may suggest methodological mismatch, amplitude/scaling differences, frequency-shape differences, low-bin/duration limitations, or data-specific issues.
