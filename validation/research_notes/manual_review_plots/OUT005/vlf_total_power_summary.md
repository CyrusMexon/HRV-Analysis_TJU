# Focused VLF and Total-Power Discrepancy Analysis

File: `validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT005__nsr021_segment_071.txt`

This analysis is for validation review only. It does not claim that either implementation is correct.

## Method

- Recomputed the same native and NeuroKit2 Welch PSDs used by the manual inspection workflow.
- Detrending: `none`; interpolation rate: `4` Hz.
- Native Welch parameters: `nperseg=214`, `noverlap=160`, inferred `nfft=214`.
- NeuroKit2 Welch parameters: `nperseg=214`, `noverlap=160`, inferred `nfft=428`.
- Included the experimental native Welch `nfft x2` PSD for context; experimental `nfft=428`.
- The decomposition CSV uses a common 0.0-0.40 Hz frequency grid; values are linearly interpolated when a frequency exists in only one pipeline grid.

## PSD Bin-Level Observation

- Native has 3 bins in VLF 0.0-0.04 Hz; NeuroKit2 has 5 bins.
- Near-zero absolute PSD separation share within VLF: 25.70%.
- This near-zero share is heuristic and should be interpreted from the plots, not as a formal causal proof.

## Metric Snapshot

| Metric | Native | NeuroKit2 | Relative error |
| --- | ---: | ---: | ---: |
| VLF | 244070.014 | 6058376829.776 | 100.00% |
| LF | 689776.019 | 21920853357.237 | 100.00% |
| HF | 161528.159 | 514159752.376 | 99.97% |
| total_power | 1468970.278 | 31414178156.637 | 100.00% |
| LF/HF | 4.270 | 42.634 | 89.98% |

## Band-Specific Integrated Power

### Native HRV Studio PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 118834.598 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 3 | 244070.014 | 373593.169 |
| VLF_0_0.04 | False | 2 | 127379.286 | 254758.571 |
| VLF_0.003_0.04 | True | 2 | 127379.286 | 254758.571 |
| VLF_0.003_0.04 | False | 2 | 127379.286 | 254758.571 |
| LF | True | 6 | 689776.019 | 976744.218 |
| LF | False | 6 | 689776.019 | 976744.218 |
| HF | True | 13 | 161528.159 | 178701.206 |
| HF | False | 13 | 161528.159 | 178701.206 |

### NeuroKit2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 596372487.241 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 5 | 6058376829.776 | 7500168488.185 |
| VLF_0_0.04 | False | 4 | 5094635697.720 | 6903796000.944 |
| VLF_0.003_0.04 | True | 4 | 5094635697.720 | 6903796000.944 |
| VLF_0.003_0.04 | False | 4 | 5094635697.720 | 6903796000.944 |
| LF | True | 12 | 21920853357.237 | 23645776784.852 |
| LF | False | 12 | 21920853357.237 | 23645776784.852 |
| HF | True | 26 | 514159752.376 | 567558081.046 |
| HF | False | 26 | 514159752.376 | 567558081.046 |

### Experimental native Welch nfft x2 PSD

| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |
| --- | --- | ---: | ---: | ---: |
| ULF | True | 1 | 0.000 | 1503710.974 |
| ULF | False | 0 | 0.000 | 0.000 |
| VLF_0_0.04 | True | 5 | 3958825.148 | 4745748.511 |
| VLF_0_0.04 | False | 4 | 2104015.271 | 3242037.537 |
| VLF_0.003_0.04 | True | 4 | 2104015.271 | 3242037.537 |
| VLF_0.003_0.04 | False | 4 | 2104015.271 | 3242037.537 |
| LF | True | 12 | 931974.678 | 1028155.228 |
| LF | False | 12 | 931974.678 | 1028155.228 |
| HF | True | 26 | 177217.828 | 182861.927 |
| HF | False | 26 | 177217.828 | 182861.927 |

## Cautious Interpretation

- Resampled-signal correlation: 0.347.
- Detrended-signal correlation: 0.347.
- Log-PSD correlation: 0.908.
- LF/HF remains aligned by the current threshold: no.
- Automated issue label: a) artifact/interpolation mismatch.

This label is heuristic. The plots and tables should be reviewed before drawing conclusions about either implementation.

## Generated Files

- `zoomed_vlf_psd.png`
- `cumulative_power_by_frequency.png`
- `vlf_total_power_decomposition.csv`
- `vlf_total_power_summary.md`
