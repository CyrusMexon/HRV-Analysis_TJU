# Matched-Sequence Kubios Frequency-Domain Validation

## Scope

This rerun used the exact Kubios-selected `HRV.Data.RRs` interval sequences for the 44-recording benchmark while preserving the original Section 3.2 spectral comparator configuration.

## Settings

- interpolation frequency: 4.0 Hz
- detrending method: None
- PSD estimator: Welch / FFT comparator branch
- window length: 120.0 s requested
- overlap: 0.75
- window function: hann
- frequency bands: VLF 0.00-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz, total 0.00-0.40 Hz
- DC handling: unchanged HRV Studio comparator behavior; VLF band begins at 0.00 Hz
- PSD integration method: unchanged HRV Studio comparator behavior (`np.trapezoid`)
- native Welch nfft multiplier: 1.0
- Welch detrend mode: current

## Summary

| Metric      | Original median RE % | Matched-sequence median RE % | Absolute change in median RE | Relative improvement % | Original mean RE % | Matched-sequence mean RE % | Original Pearson r | Matched-sequence Pearson r |
| ----------- | -------------------- | ---------------------------- | ---------------------------- | ---------------------- | ------------------ | -------------------------- | ------------------ | -------------------------- |
| VLF         | 31.916               | 14.010                       | -17.906                      | 56.104                 | 137.125            | 17.065                     | 0.335              | 0.991                      |
| LF          | 20.417               | 3.495                        | -16.922                      | 82.884                 | 134.141            | 7.102                      | 0.835              | 0.998                      |
| HF          | 28.715               | 2.539                        | -26.176                      | 91.157                 | 172.439            | 6.168                      | 0.581              | 0.973                      |
| total_power | 19.908               | 2.514                        | -17.395                      | 87.374                 | 107.002            | 5.648                      | 0.518              | 0.994                      |
| LF/HF       | 17.565               | 5.549                        | -12.015                      | 68.405                 | 28.205             | 8.552                      | 0.895              | 0.982                      |
| LF_nu       | 7.447                | 1.459                        | -5.988                       | 80.403                 | 12.366             | 2.758                      | 0.860              | 0.990                      |
| HF_nu       | 11.222               | 3.792                        | -7.431                       | 66.213                 | 30.644             | 8.181                      | 0.862              | 0.990                      |

## Interpretation

Sequence mismatch was a major contributor for some absolute-power metrics, but residual spectral-processing differences remain.

This is a sequence-alignment correction only. Detrending, interpolation, Welch/FFT settings, band definitions, and integration behavior were kept unchanged from the original benchmark configuration.

## Outputs

- `validation\kubios_subset\frequency_domain_kubios_matched_sequence_summary.csv`
- `validation\kubios_subset\frequency_domain_kubios_matched_sequence_per_recording.csv`
- `validation\kubios_subset\frequency_domain_original_vs_matched_sequence_summary.csv`
