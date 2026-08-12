# Time-Domain and Nonlinear NeuroKit2 Agreement Extension v2

## Status
This rerun was performed after changing HRV Studio SD2 to `sqrt(2 * SDNN^2 - SD1^2)`. It uses the same v05 input files and direct RR interval loading path as the prior extension. Manuscript files were not modified.

NeuroKit2 reference values were computed from the installed NeuroKit2 0.2.12 time-domain and Poincare definitions: SDNN/RMSSD/pNN50 from `hrv_time.py`, and SD1/SD2 as standard deviations of the Poincare axes used by `hrv_nonlinear.py`.

## Agreement Statistics
| Metric | n | Median relative error (%) | Mean relative error (%) | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| RMSSD | 7598 | 0 | 0 | 1 |
| SDNN | 7598 | 0 | 0 | 1 |
| pNN50 | 7364 | 0.134952766532 | 0.13718550117 | 0.999999951422 |
| SD1 | 7598 | 0 | 7.27571474432e-15 | 1 |
| SD2 | 7598 | 0.0525414439098 | 0.104050893835 | 0.999979109063 |

## Files Generated
- Summary table: `validation\runs\v05_physionet_10min_segment_linear\time_domain_nonlinear_neurokit2_agreement_summary.csv`.
- Per-recording comparison: `validation\runs\v05_physionet_10min_segment_linear\time_domain_nonlinear_neurokit2_agreement_per_recording.csv`.
- Report: `validation\research_notes\time_domain_nonlinear_neurokit2_validation_report_v2.md`.

Failed files: 0.
