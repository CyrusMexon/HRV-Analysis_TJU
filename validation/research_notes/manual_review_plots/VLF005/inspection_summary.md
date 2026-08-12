# Frequency Outlier Inspection: VLF005__nsr008_segment_046.txt

This report is generated for manual validation review only. It does not indicate which implementation is correct.

## File

- Source file: `validation\kubios_subset\input_ascii_rr\vlf_sensitive\VLF005__nsr008_segment_046.txt`
- Duration: 4.00 seconds

## Warning messages

- scipy.misc is deprecated and will be removed in 2.0.0
- RR intervals appear to be in seconds (mean=1.044s); converting to ms.
- Signal duration < 1 minute. Frequency domain results may be unreliable.
- Total power is zero or negative. Returning defaults.
- No frequency points found in lf band [0.04, 0.15] Hz
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
| total_power | 0 | 238.84 | 238.84 | 100.00% |

## Preprocessing signal comparison

- Resampled-signal Pearson correlation: 0.994.
- Resampled-signal RMSE: 8.438 ms.
- Detrended-signal Pearson correlation: 0.994.
- Detrended-signal RMSE: 8.438 ms.
- Low correlation or high RMSE before PSD estimation may suggest that the disagreement originates in interpolation, detrending, clipping, units, or other pre-PSD signal preparation differences.

## PSD shape assessment

- Shape assessment: unclear/manual review needed.
- Log-PSD correlation on common 0-0.4 Hz grid: not available.
- This automated assessment is heuristic; the overlay plot requires manual review before drawing conclusions.

## Suggested disagreement category

- c) low-bin/duration issue

Interpretation should remain cautious. The disagreement may suggest methodological mismatch, amplitude/scaling differences, frequency-shape differences, low-bin/duration limitations, or data-specific issues.
