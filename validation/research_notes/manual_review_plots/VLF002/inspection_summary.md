# Frequency Outlier Inspection: VLF002__nsr007_segment_062.txt

This report is generated for manual validation review only. It does not indicate which implementation is correct.

## File

- Source file: `validation\kubios_subset\input_ascii_rr\vlf_sensitive\VLF002__nsr007_segment_062.txt`
- Duration: 22.34 seconds

## Warning messages

- scipy.misc is deprecated and will be removed in 2.0.0
- RR intervals mean (42.72) is unusual for ms; verify input units.
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
| LF | 0 | 59.0189 | 59.0189 | 100.00% |
| LF/HF | 0 | 0.204905 | 0.204905 | 100.00% |
| LF_nu | 0 | 17.0059 | 17.0059 | 100.00% |
| HF | 0.895046 | 288.03 | 287.135 | 99.69% |
| total_power | 1.56862 | 407.838 | 406.27 | 99.62% |

## Preprocessing signal comparison

- Resampled-signal Pearson correlation: 0.369.
- Resampled-signal RMSE: 140.549 ms.
- Detrended-signal Pearson correlation: 0.369.
- Detrended-signal RMSE: 140.549 ms.
- Low correlation or high RMSE before PSD estimation may suggest that the disagreement originates in interpolation, detrending, clipping, units, or other pre-PSD signal preparation differences.

## PSD shape assessment

- Shape assessment: PSD shape similarity is mixed or unclear.
- Log-PSD correlation on common 0-0.4 Hz grid: 0.836.
- This automated assessment is heuristic; the overlay plot requires manual review before drawing conclusions.

## Suggested disagreement category

- c) low-bin/duration issue

Interpretation should remain cautious. The disagreement may suggest methodological mismatch, amplitude/scaling differences, frequency-shape differences, low-bin/duration limitations, or data-specific issues.
