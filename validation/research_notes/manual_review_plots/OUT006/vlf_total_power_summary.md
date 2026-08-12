# Focused VLF and Total-Power Discrepancy Analysis

File: `validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT006__nsr015_segment_089.txt`

This analysis is for validation review only. It does not claim that either implementation is correct.

## Method

- Recomputed the same native and NeuroKit2 Welch PSDs used by the manual inspection workflow.
- Detrending: `none`; interpolation rate: `4` Hz.
- Native Welch parameters: `nperseg=480`, `noverlap=360`, inferred `nfft=480`.
- NeuroKit2 Welch parameters: `nperseg=249`, `noverlap=248`, inferred `nfft=498`.
- Included the experimental native Welch `nfft x2` PSD for context; experimental `nfft=960`.
- The decomposition CSV uses a common 0.0-0.40 Hz frequency grid; values are linearly interpolated when a frequency exists in only one pipeline grid.

## PSD Bin-Level Observation

- Native has 5 bins in VLF 0.0-0.04 Hz; NeuroKit2 has 5 bins.
- Near-zero absolute PSD separation share within VLF: 53.35%.
- This near-zero share is heuristic and should be interpreted from the plots, not as a formal causal proof.

## Metric Snapshot

| Metric | Native | NeuroKit2 | Relative error |
| --- | ---: | ---: | ---: |
| VLF | 10505.635 | 11986.959 | 12.36% |
| LF | 1264.493 | 1174.117 | 7.70% |
| HF | 537.969 | 484.801 | 10.97% |
| total_power | 12678.635 | 14152.592 | 10.41% |
| LF/HF | 2.350 | 2.422 | 2.95% |

## Band-Specific Integrated Power

### Native HRV Studio PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 161.491 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 5 | 10505.635 | 10858.645 |
| VLF_0_0.04 | False | 4 | 6599.545 | 10697.154 |
| VLF_0.003_0.04 | True | 4 | 6599.545 | 10697.154 |
| VLF_0.003_0.04 | False | 4 | 6599.545 | 10697.154 |
| LF | True | 14 | 1264.493 | 1369.063 |
| LF | False | 14 | 1264.493 | 1369.063 |
| HF | True | 31 | 537.969 | 546.306 |
| HF | False | 31 | 537.969 | 546.306 |

### NeuroKit2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 2811.926 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 5 | 11986.959 | 13727.747 |
| VLF_0_0.04 | False | 4 | 8042.894 | 10915.821 |
| VLF_0.003_0.04 | True | 4 | 8042.894 | 10915.821 |
| VLF_0.003_0.04 | False | 4 | 8042.894 | 10915.821 |
| LF | True | 14 | 1174.117 | 1340.422 |
| LF | False | 14 | 1174.117 | 1340.422 |
| HF | True | 31 | 484.801 | 493.289 |
| HF | False | 31 | 484.801 | 493.289 |

### Experimental native Welch nfft x2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 249582.883 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 10 | 651929.451 | 776827.449 |
| VLF_0_0.04 | False | 9 | 343845.116 | 527244.566 |
| VLF_0.003_0.04 | True | 9 | 343845.116 | 527244.566 |
| VLF_0.003_0.04 | False | 9 | 343845.116 | 527244.566 |
| LF | True | 27 | 1242.123 | 1294.892 |
| LF | False | 27 | 1242.123 | 1294.892 |
| HF | True | 61 | 537.597 | 541.773 |
| HF | False | 61 | 537.597 | 541.773 |

## Cautious Interpretation

- Resampled-signal correlation: 1.000.
- Detrended-signal correlation: 1.000.
- Log-PSD correlation: 0.961.
- LF/HF remains aligned by the current threshold: yes.
- Automated issue label: e) unclear.

This label is heuristic. The plots and tables should be reviewed before drawing conclusions about either implementation.

## Generated Files

- `zoomed_vlf_psd.png`
- `cumulative_power_by_frequency.png`
- `vlf_total_power_decomposition.csv`
- `vlf_total_power_summary.md`
