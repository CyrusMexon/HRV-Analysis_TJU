# Focused VLF and Total-Power Discrepancy Analysis

File: `validation\kubios_subset\input_ascii_rr\vlf_sensitive\VLF002__nsr007_segment_062.txt`

This analysis is for validation review only. It does not claim that either implementation is correct.

## Method

- Recomputed the same native and NeuroKit2 Welch PSDs used by the manual inspection workflow.
- Detrending: `none`; interpolation rate: `4` Hz.
- Native Welch parameters: `nperseg=45`, `noverlap=33`, inferred `nfft=45`.
- NeuroKit2 Welch parameters: `nperseg=45`, `noverlap=33`, inferred `nfft=90`.
- Included the experimental native Welch `nfft x2` PSD for context; experimental `nfft=90`.
- The decomposition CSV uses a common 0.0-0.40 Hz frequency grid; values are linearly interpolated when a frequency exists in only one pipeline grid.

## PSD Bin-Level Observation

- Native has 1 bins in VLF 0.0-0.04 Hz; NeuroKit2 has 1 bins.
- Near-zero absolute PSD separation share within VLF: 100.00%.
- This near-zero share is heuristic and should be interpreted from the plots, not as a formal causal proof.

## Metric Snapshot

| Metric | Native | NeuroKit2 | Relative error |
| --- | ---: | ---: | ---: |
| VLF | 0.000 | 0.000 | not available% |
| LF | 0.000 | 59.019 | 100.00% |
| HF | 0.895 | 288.030 | 99.69% |
| total_power | 1.569 | 407.838 | 99.62% |
| LF/HF | 0.000 | 0.205 | 100.00% |

## Band-Specific Integrated Power

### Native HRV Studio PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 0.115 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 1 | 0.000 | 0.115 |
| VLF_0_0.04 | False | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | True | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | False | 0 | 0.000 | 0.000 |
| LF | True | 1 | 0.000 | 0.393 |
| LF | False | 1 | 0.000 | 0.393 |
| HF | True | 3 | 0.895 | 1.342 |
| HF | False | 3 | 0.895 | 1.342 |

### NeuroKit2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 11.261 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 1 | 0.000 | 11.261 |
| VLF_0_0.04 | False | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | True | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | False | 0 | 0.000 | 0.000 |
| LF | True | 3 | 59.019 | 90.028 |
| LF | False | 3 | 59.019 | 90.028 |
| HF | True | 6 | 288.030 | 362.919 |
| HF | False | 6 | 288.030 | 362.919 |

### Experimental native Welch nfft x2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 13097.663 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 1 | 0.000 | 13097.663 |
| VLF_0_0.04 | False | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | True | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | False | 0 | 0.000 | 0.000 |
| LF | True | 3 | 17028.237 | 27064.421 |
| LF | False | 3 | 17028.237 | 27064.421 |
| HF | True | 6 | 21.478 | 22.626 |
| HF | False | 6 | 21.478 | 22.626 |

## Cautious Interpretation

- Resampled-signal correlation: 0.369.
- Detrended-signal correlation: 0.369.
- Log-PSD correlation: 0.836.
- LF/HF remains aligned by the current threshold: no.
- Automated issue label: a) artifact/interpolation mismatch.

This label is heuristic. The plots and tables should be reviewed before drawing conclusions about either implementation.

## Generated Files

- `zoomed_vlf_psd.png`
- `cumulative_power_by_frequency.png`
- `vlf_total_power_decomposition.csv`
- `vlf_total_power_summary.md`
