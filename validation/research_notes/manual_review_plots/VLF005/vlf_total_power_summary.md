# Focused VLF and Total-Power Discrepancy Analysis

File: `validation\kubios_subset\input_ascii_rr\vlf_sensitive\VLF005__nsr008_segment_046.txt`

This analysis is for validation review only. It does not claim that either implementation is correct.

## Method

- Recomputed the same native and NeuroKit2 Welch PSDs used by the manual inspection workflow.
- Detrending: `none`; interpolation rate: `4` Hz.
- Native Welch parameters: `nperseg=8`, `noverlap=6`, inferred `nfft=8`.
- NeuroKit2 Welch parameters: `nperseg=8`, `noverlap=6`, inferred `nfft=16`.
- Included the experimental native Welch `nfft x2` PSD for context; experimental `nfft=16`.
- The decomposition CSV uses a common 0.0-0.40 Hz frequency grid; values are linearly interpolated when a frequency exists in only one pipeline grid.

## PSD Bin-Level Observation

- Native has 1 bins in VLF 0.0-0.04 Hz; NeuroKit2 has 1 bins.
- Near-zero absolute PSD separation share within VLF: 100.00%.
- This near-zero share is heuristic and should be interpreted from the plots, not as a formal causal proof.

## Metric Snapshot

| Metric | Native | NeuroKit2 | Relative error |
| --- | ---: | ---: | ---: |
| VLF | 0.000 | 0.000 | not available% |
| LF | 0.000 | 0.000 | not available% |
| HF | 0.000 | 0.000 | not available% |
| total_power | 0.000 | 238.840 | 100.00% |
| LF/HF | not available | not available | not available% |

## Band-Specific Integrated Power

### Native HRV Studio PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 461.170 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 1 | 0.000 | 461.170 |
| VLF_0_0.04 | False | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | True | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | False | 0 | 0.000 | 0.000 |
| LF | True | 0 | 0.000 | 0.000 |
| LF | False | 0 | 0.000 | 0.000 |
| HF | True | 0 | 0.000 | 0.000 |
| HF | False | 0 | 0.000 | 0.000 |

### NeuroKit2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 179.826 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 1 | 0.000 | 179.826 |
| VLF_0_0.04 | False | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | True | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | False | 0 | 0.000 | 0.000 |
| LF | True | 0 | 0.000 | 0.000 |
| LF | False | 0 | 0.000 | 0.000 |
| HF | True | 1 | 0.000 | 297.854 |
| HF | False | 1 | 0.000 | 297.854 |

### Experimental native Welch nfft x2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 291247.623 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 1 | 0.000 | 291247.623 |
| VLF_0_0.04 | False | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | True | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | False | 0 | 0.000 | 0.000 |
| LF | True | 0 | 0.000 | 0.000 |
| LF | False | 0 | 0.000 | 0.000 |
| HF | True | 1 | 0.000 | 453627.914 |
| HF | False | 1 | 0.000 | 453627.914 |

## Cautious Interpretation

- Resampled-signal correlation: 0.994.
- Detrended-signal correlation: 0.994.
- Log-PSD correlation: not available.
- LF/HF remains aligned by the current threshold: no.
- Automated issue label: e) unclear.

This label is heuristic. The plots and tables should be reviewed before drawing conclusions about either implementation.

## Generated Files

- `zoomed_vlf_psd.png`
- `cumulative_power_by_frequency.png`
- `vlf_total_power_decomposition.csv`
- `vlf_total_power_summary.md`
