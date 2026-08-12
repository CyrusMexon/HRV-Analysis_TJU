# Time-Domain and Nonlinear Kubios Agreement Validation v2

## Status
This rerun compares HRV Studio metrics against Kubios printed metrics using the exact Kubios-selected NN interval sequences stored in `HRV.Data.RRs` in the Kubios `.mat` exports.

No manuscript files were modified.

## Changes Before Rerun
- HRV Studio SD2 was updated in `hrvlib/metrics/nonlinear.py` from `sqrt(2 * SDNN^2 - 0.5 * SD1^2)` to `sqrt(2 * SDNN^2 - SD1^2)`.
- A focused regression test was added in `tests/test_nonlinear.py` for the standard SD2 identity.

## Inputs
- Cohort: `validation\research_notes\manual_review_sensitivity_analysis\cleaned_valid_only.csv`
- Kubios exports: `validation\kubios_subset\kubios_exports_120s_75pct_none`
- HRV Studio matched-sequence input: Kubios `Res.HRV.Data.RRs` from each `.mat` file
- Recording count: 44
- Kubios-selected interval count: min=88, median=541.5, max=916
- Kubios-selected duration: min=61.431 s, median=450.559 s, max=600.704 s

## Matched-Sequence Kubios Results
| Metric | n | Median relative error (%) | Mean relative error (%) | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| RMSSD | 44 | 0.000843566940379 | 0.00567083131868 | 0.99999999609 |
| SDNN | 44 | 4.40262445099e-05 | 5.54005095117e-05 | 0.999999999999 |
| pNN50 | 38 | 0.000339208074022 | 0.0022219190083 | 0.999999999996 |
| SD1 | 44 | 0.000913535452176 | 0.00564143152102 | 0.999999996072 |
| SD2 | 44 | 0.0704908543979 | 0.300603769134 | 0.999866003753 |

## NeuroKit2 Rerun After SD2 Fix
| Metric | n | Median relative error (%) | Mean relative error (%) | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| RMSSD | 7598 | 0 | 0 | 1 |
| SDNN | 7598 | 0 | 0 | 1 |
| pNN50 | 7364 | 0.134952766532 | 0.13718550117 | 0.999999951422 |
| SD1 | 7598 | 0 | 7.27571474432e-15 | 1 |
| SD2 | 7598 | 0.0525414439098 | 0.104050893835 | 0.999979109063 |

## Interpretation
- RMSSD, SDNN, pNN50, and SD1 agree with Kubios to text-export precision when HRV Studio is run on Kubios `HRV.Data.RRs`.
- SD2 agreement improves after the formula correction and remains high, with median relative error 0.0705% against Kubios.
- The prior RMSSD/SDNN Kubios disagreement is attributable to sequence/segment mismatch, not to HRV Studio time-domain metric implementation.

## Files Generated
- Summary table: `validation\kubios_subset\time_domain_nonlinear_kubios_matched_sequence_summary.csv`
- Per-recording matched-sequence table: `validation\kubios_subset\time_domain_nonlinear_kubios_matched_sequence_per_recording.csv`
- NeuroKit2 summary rerun: `validation\runs\v05_physionet_10min_segment_linear\time_domain_nonlinear_neurokit2_agreement_summary.csv`
- NeuroKit2 per-recording rerun: `validation\runs\v05_physionet_10min_segment_linear\time_domain_nonlinear_neurokit2_agreement_per_recording.csv`
- NeuroKit2 rerun report: `validation\research_notes\time_domain_nonlinear_neurokit2_validation_report_v2.md`

## Verification
- Targeted SD2 regression test passed: `pytest tests\test_nonlinear.py::test_poincare_sd2_standard_identity`.
- Full `tests\test_nonlinear.py` was also run and still has pre-existing failures unrelated to SD2: missing/renamed RQA API expectations and one DFA insufficient-data expectation.
