# Focused VLF and Total-Power Discrepancy Analysis

File: `validation\kubios_subset\input_ascii_rr\random_controls\RC003__nsr002_segment_069.txt`

This analysis is for validation review only. It does not claim that either implementation is correct.

## Method

- Recomputed the same native and NeuroKit2 Welch PSDs used by the manual inspection workflow.
- Detrending: `none`; interpolation rate: `4` Hz.
- Native Welch parameters: `nperseg=50`, `noverlap=37`, inferred `nfft=50`.
- NeuroKit2 Welch parameters: `nperseg=50`, `noverlap=37`, inferred `nfft=100`.
- Included the experimental native Welch `nfft x2` PSD for context; experimental `nfft=100`.
- The decomposition CSV uses a common 0.0-0.40 Hz frequency grid; values are linearly interpolated when a frequency exists in only one pipeline grid.

## PSD Bin-Level Observation

- Native has 1 bins in VLF 0.0-0.04 Hz; NeuroKit2 has 2 bins.
- Near-zero absolute PSD separation share within VLF: 33.96%.
- This near-zero share is heuristic and should be interpreted from the plots, not as a formal causal proof.

## Metric Snapshot

| Metric | Native | NeuroKit2 | Relative error |
| --- | ---: | ---: | ---: |
| VLF | 0.000 | 31.498 | 100.00% |
| LF | 0.000 | 74.919 | 100.00% |
| HF | 0.943 | 165.890 | 99.43% |
| total_power | 1.349 | 302.628 | 99.55% |
| LF/HF | 0.000 | 0.452 | 100.00% |

## Band-Specific Integrated Power

### Native HRV Studio PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 0.074 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 1 | 0.000 | 0.074 |
| VLF_0_0.04 | False | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | True | 0 | 0.000 | 0.000 |
| VLF_0.003_0.04 | False | 0 | 0.000 | 0.000 |
| LF | True | 1 | 0.000 | 0.218 |
| LF | False | 1 | 0.000 | 0.218 |
| HF | True | 4 | 0.943 | 1.260 |
| HF | False | 4 | 0.943 | 1.260 |

### NeuroKit2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 21.396 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 2 | 31.498 | 62.996 |
| VLF_0_0.04 | False | 1 | 0.000 | 41.600 |
| VLF_0.003_0.04 | True | 1 | 0.000 | 41.600 |
| VLF_0.003_0.04 | False | 1 | 0.000 | 41.600 |
| LF | True | 3 | 74.919 | 112.042 |
| LF | False | 3 | 74.919 | 112.042 |
| HF | True | 7 | 165.890 | 198.855 |
| HF | False | 7 | 165.890 | 198.855 |

### Experimental native Welch nfft x2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 13124.007 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 2 | 16143.407 | 32286.815 |
| VLF_0_0.04 | False | 1 | 0.000 | 19162.808 |
| VLF_0.003_0.04 | True | 1 | 0.000 | 19162.808 |
| VLF_0.003_0.04 | False | 1 | 0.000 | 19162.808 |
| LF | True | 3 | 16995.567 | 27029.508 |
| LF | False | 3 | 16995.567 | 27029.508 |
| HF | True | 7 | 21.098 | 21.952 |
| HF | False | 7 | 21.098 | 21.952 |

## Cautious Interpretation

- Resampled-signal correlation: 0.510.
- Detrended-signal correlation: 0.510.
- Log-PSD correlation: -0.404.
- LF/HF remains aligned by the current threshold: no.
- Automated issue label: a) artifact/interpolation mismatch.

This label is heuristic. The plots and tables should be reviewed before drawing conclusions about either implementation.

## Generated Files

- `zoomed_vlf_psd.png`
- `cumulative_power_by_frequency.png`
- `vlf_total_power_decomposition.csv`
- `vlf_total_power_summary.md`
