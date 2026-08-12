# Kubios Frequency-Domain Sequence Alignment Audit

## Scope

This audit used the 44 recordings in `cleaned_valid_only.csv` and preserved the original Section 3.2 comparator settings from `run_info.json`. No manuscript files or original benchmark outputs were modified.

## Phase 1: Export Structure

Kubios frequency metrics are stored in `Res.HRV.Frequency.Welch` in each MAT export. The FFT-column values printed in each Kubios TXT export match those MAT values within text precision. The same MAT export contains `Res.HRV.Data.RR` (complete imported vector), `Res.HRV.Data.RRs` (selected sample), and `Res.HRV.Data.RRi` (interpolated selected RR series).

Inference counts:

| inference                      | n  |
| ------------------------------ | -- |
| selected_RRs_confirmed         | 25 |
| selected_RRs_strongly_inferred | 19 |

Selection counts:

| selection_type | n  |
| -------------- | -- |
| shorter_prefix | 31 |
| full_sequence  | 13 |

**A. Can we establish that Kubios frequency-domain metrics were calculated from `HRV.Data.RRs`?**

Yes for this export set. Evidence is strongest for the 31 recordings where `HRV.Data.RRs` is a shorter prefix of `HRV.Data.RR`: `HRV.Data.RRi` length agrees with the selected-sequence duration at 4 Hz and not with the full imported duration, while printed FFT metrics match `Res.HRV.Frequency.Welch`. For the remaining 13 recordings, `HRV.Data.RRs` equals `HRV.Data.RR`, so selected and full sequences are indistinguishable.

**B. If yes, for how many of the 44 recordings?**

Established or directly indistinguishable for 44/44 recordings. In 31/44 recordings the selected sequence is shorter than the full input; in 13/44, selected equals full.

**C. Evidence supporting or contradicting this interpretation**

Supporting evidence:

- Printed Kubios FFT metrics match `Res.HRV.Frequency.Welch` values in the MAT exports.
- `Res.HRV.Data.RRi` has sample counts consistent with `sum(HRV.Data.RRs) * 4 Hz`.
- In shorter-prefix recordings, `Res.HRV.Data.RRi` is inconsistent with full-vector duration.
- `HRV.Data.RRs` is a prefix of `HRV.Data.RR`, matching the previously established time-domain selected-sample behavior.

Contradicting evidence: none found in the exported fields inspected. The exports do not include a separate explicit flag that says "frequency metrics calculated from HRV.Data.RRs"; the conclusion is based on MAT structure plus numerical/timing checks.

**D. Does the original HRV Studio frequency-domain benchmark use the full input sequence or the Kubios-selected sequence?**

The archived original benchmark values are not uniformly reproducible from the current Kubios MAT `HRV.Data.RR` vectors. Exact full-vector recomputation reproduced all metrics for 20/44 recordings and did not reproduce 24/44 recordings; the maximum absolute difference was `11568.697077443`. At least one current input file/export pair has 600 s in the MAT export while the archived validation row records a much shorter effective RR count/duration. Therefore this audit treats the original manuscript-facing values as archived provenance and does not claim that every archived HRV Studio row used the current `HRV.Data.RR` vector.

Original archived values were closer to these current recomputations by metric-row count:

| closer_to        | n   |
| ---------------- | --- |
| current_full     | 173 |
| tie              | 91  |
| current_selected | 44  |

**E. Was the original comparison sequence-aligned?**

No for the 31 shorter-prefix Kubios exports if the comparator is the current complete imported `HRV.Data.RR` sequence. Additionally, the provenance mismatch above means some archived original rows cannot be cleanly re-derived from the current full vectors, so the original comparison should not be interpreted as exact sequence-aligned evidence.

## Phase 2: Numerical Cross-Check

Settings recovered from the original run:

- interpolation frequency: 4.0 Hz
- detrending method: None
- PSD estimator: Welch / FFT comparator branch
- window length: 120.0 s requested
- overlap: 0.75
- window function: hann
- band convention: Kubios-compatible comparator mode
- frequency bands: VLF 0.00-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz, total 0.00-0.40 Hz
- DC handling: unchanged HRV Studio comparator behavior; VLF band begins at 0.00 Hz
- PSD integration method: unchanged HRV Studio comparator behavior (`np.trapezoid`)
- native Welch nfft multiplier: 1.0
- Welch detrend mode: current

Full-vs-selected comparison:

| condition | Metric      | n  | median_relative_error_pct | mean_relative_error_pct | pearson_r | median_absolute_error | mean_absolute_error |
| --------- | ----------- | -- | ------------------------- | ----------------------- | --------- | --------------------- | ------------------- |
| full      | VLF         | 44 | 21.359                    | 97.123                  | 0.837     | 264.069               | 438.814             |
| full      | LF          | 44 | 10.532                    | 22.543                  | 0.989     | 24.218                | 103.934             |
| full      | HF          | 44 | 7.342                     | 30.435                  | 0.826     | 15.675                | 135.121             |
| full      | total_power | 44 | 11.374                    | 40.017                  | 0.903     | 218.119               | 500.506             |
| full      | LF/HF       | 44 | 9.851                     | 14.473                  | 0.986     | 0.150                 | 0.363               |
| full      | LF_nu       | 44 | 2.347                     | 6.053                   | 0.973     | 1.682                 | 3.027               |
| full      | HF_nu       | 44 | 6.708                     | 9.393                   | 0.973     | 1.443                 | 3.105               |
| selected  | VLF         | 44 | 14.010                    | 17.065                  | 0.991     | 104.700               | 165.111             |
| selected  | LF          | 44 | 3.495                     | 7.102                   | 0.998     | 12.373                | 45.689              |
| selected  | HF          | 44 | 2.539                     | 6.168                   | 0.973     | 4.484                 | 44.360              |
| selected  | total_power | 44 | 2.514                     | 5.648                   | 0.994     | 43.683                | 107.140             |
| selected  | LF/HF       | 44 | 5.549                     | 8.552                   | 0.982     | 0.094                 | 0.329               |
| selected  | LF_nu       | 44 | 1.459                     | 2.758                   | 0.990     | 0.932                 | 1.729               |
| selected  | HF_nu       | 44 | 3.792                     | 8.181                   | 0.990     | 1.039                 | 1.977               |

Old-vs-new benchmark summary:

| Metric      | Original median RE % | Matched-sequence median RE % | Absolute change in median RE | Relative improvement % | Original mean RE % | Matched-sequence mean RE % | Original Pearson r | Matched-sequence Pearson r |
| ----------- | -------------------- | ---------------------------- | ---------------------------- | ---------------------- | ------------------ | -------------------------- | ------------------ | -------------------------- |
| VLF         | 31.916               | 14.010                       | -17.906                      | 56.104                 | 137.125            | 17.065                     | 0.335              | 0.991                      |
| LF          | 20.417               | 3.495                        | -16.922                      | 82.884                 | 134.141            | 7.102                      | 0.835              | 0.998                      |
| HF          | 28.715               | 2.539                        | -26.176                      | 91.157                 | 172.439            | 6.168                      | 0.581              | 0.973                      |
| total_power | 19.908               | 2.514                        | -17.395                      | 87.374                 | 107.002            | 5.648                      | 0.518              | 0.994                      |
| LF/HF       | 17.565               | 5.549                        | -12.015                      | 68.405                 | 28.205             | 8.552                      | 0.895              | 0.982                      |
| LF_nu       | 7.447                | 1.459                        | -5.988                       | 80.403                 | 12.366             | 2.758                      | 0.860              | 0.990                      |
| HF_nu       | 11.222               | 3.792                        | -7.431                       | 66.213                 | 30.644             | 8.181                      | 0.862              | 0.990                      |

Largest per-recording relative-error changes:

| pilot_id | Metric      | full_RE  | selected_RE | change    | selection_type |
| -------- | ----------- | -------- | ----------- | --------- | -------------- |
| OUT001   | VLF         | 1485.673 | 14.136      | -1471.536 | shorter_prefix |
| CH001    | VLF         | 1170.219 | 13.313      | -1156.905 | shorter_prefix |
| CH001    | total_power | 316.797  | 0.165       | -316.632  | shorter_prefix |
| OUT001   | total_power | 309.869  | 14.559      | -295.311  | shorter_prefix |
| OUT010   | VLF         | 291.967  | 16.663      | -275.305  | shorter_prefix |
| CH007    | HF          | 256.115  | 3.177       | -252.938  | shorter_prefix |
| OUT010   | total_power | 223.199  | 6.320       | -216.879  | shorter_prefix |
| OUT009   | VLF         | 189.108  | 6.278       | -182.830  | shorter_prefix |
| RC004    | HF          | 245.292  | 73.906      | -171.387  | shorter_prefix |
| OUT010   | HF          | 174.906  | 3.691       | -171.215  | shorter_prefix |
| OUT010   | LF          | 142.390  | 2.845       | -139.545  | shorter_prefix |
| CH007    | total_power | 127.834  | 10.694      | -117.140  | shorter_prefix |

## Phase 5 Interpretation

Sequence mismatch was a major contributor for some absolute-power metrics, but residual spectral-processing differences remain.

Metrics improved by median relative error: 7/7. Metrics worsened: 0/7.

LF/HF changed from `17.565%` to `5.549%` median relative error. VLF changed from `31.916%` to `14.010%`.

Normalized metrics remain among the strongest frequency-domain metrics after matching: True.

Remaining discrepancies after exact sequence matching should be interpreted as spectral-processing/convention differences under the preserved comparator configuration, not sequence-selection differences.

## Manuscript Consequences

The manuscript's existing Section 3.2 numbers, Figure 4B-D, Abstract Kubios statements, Discussion interpretation, and Limitations wording would need review before relying on the current frequency-domain Kubios benchmark. This audit does not edit those manuscript components.

## Commands Used

```powershell
python tools/frequency_domain_sequence_audit.py
```

## Outputs

- `validation\kubios_subset\frequency_domain_sequence_debug_band_corrected_20260811\frequency_domain_sequence_audit.csv`
- `validation\kubios_subset\frequency_domain_sequence_debug_band_corrected_20260811\full_vs_selected_sequence_metric_summary.csv`
- `validation\kubios_subset\frequency_domain_sequence_debug_band_corrected_20260811\full_vs_selected_sequence_per_recording.csv`
- `validation\kubios_subset\frequency_domain_kubios_matched_sequence_summary_band_corrected_20260811.csv`
- `validation\kubios_subset\frequency_domain_kubios_matched_sequence_per_recording_band_corrected_20260811.csv`
- `validation\kubios_subset\frequency_domain_original_vs_matched_sequence_summary_band_corrected_20260811.csv`
- `validation\research_notes\kubios_frequency_domain_sequence_alignment_audit_band_corrected_20260811.md`
- `validation\research_notes\frequency_domain_kubios_matched_sequence_validation_report_band_corrected_20260811.md`
