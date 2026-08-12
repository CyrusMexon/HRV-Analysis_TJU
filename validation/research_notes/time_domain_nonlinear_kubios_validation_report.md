# Time-Domain and Nonlinear Kubios Agreement Extension

## Status
The target metrics were already present in the raw Kubios text exports used for the Section 3.2 benchmark, but they were not present in the existing parsed Kubios comparison CSVs. The existing parser `tools/parse_kubios_exports.py` extracts only frequency-domain metrics from these reports, so no duplicate parser was added and no existing validation code was modified. This report extends the existing benchmark by parsing the same raw exports and computing HRV Studio values from the same final-cleaned 44 validation input files.

No manuscript files were modified. No commits were made.

## Existing Pipeline Identified
- `tools/parse_kubios_exports.py`: Section 3.2 Kubios export parser and comparison workflow. It currently extracts VLF, LF, HF, total_power, LF/HF, LF_nu, and HF_nu from the frequency-domain section.
- `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/`: existing parsed frequency-domain benchmark outputs and QC report.
- `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`: final manually cleaned 44-recording benchmark subset used for manuscript-facing Kubios results.
- `hrvlib/metrics/time_domain.py`: `HRVTimeDomainAnalysis.full_analysis()` computes HRV Studio RMSSD, SDNN, and pNN50.
- `hrvlib/metrics/nonlinear.py`: `NonlinearHRVAnalysis.poincare_analysis()` computes HRV Studio SD1 and SD2.

## Dataset and Exports Used
- Final cohort source: `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`
- Exact cohort size: 44 recordings.
- Kubios exports: `validation/kubios_subset/kubios_exports_120s_75pct_none/`
- HRV Studio input files: the `validation_input_file` values recorded in the cleaned benchmark table.
- Manual-review exclusions inherited from the existing Section 3.2 workflow: `VLF002` and `RC003` removed from the 46-file valid-only table; `OUT005` and `VLF005` were already absent; `OUT006` retained.

## Preprocessing and Comparison Configuration
- Benchmark run context: `validation/runs/v07_kubios_subset_50_none_120_75_after_arm_a/freq_domain_neurokit2_validation.csv`.
- Existing comparator settings: segment length 120.0 seconds, overlap ratio 0.75, detrend method none, Welch detrend mode current, interpolation rate 4.0 Hz, window type hann.
- Kubios report configuration: detrending method none; FFT window width 256 seconds; FFT overlap 50%; AR order 16; AR factorization No.
- For the time-domain and nonlinear extension, no interpolation, spectral detrending, artifact correction, or new analysis windows were introduced. RR intervals were loaded through the existing validation loader, sanitized to finite positive intervals, and normalized to milliseconds using the existing HRV Studio frequency-analysis convention: mean RR < 10 indicates seconds and is multiplied by 1000.
- RR unit normalization observed in this final cohort:
- seconds_to_ms: 44 recordings
- Relative error used the existing Section 3.2/NeuroKit2 helper formula: `abs(HRV Studio - Kubios) / abs(Kubios) * 100`; zero Kubios denominators are excluded from relative-error summaries.

## Available Metrics
- RMSSD: available directly as `RMSSD (ms)` in the Kubios reports.
- SDNN: available as Kubios `STD RR (ms)`, which is the report's overall standard deviation of RR intervals. Kubios also reports `SDNN index (ms)`, but that segmented index was not used for manuscript SDNN.
- pNN50: available directly as `pNN50 (%)`.
- SD1: available directly as `SD1 (ms)` in the Poincare plot section.
- SD2: available directly as `SD2 (ms)` in the Poincare plot section.

## Unavailable Metrics
- None of the target metrics were unavailable in the final cleaned 44-export cohort.

No Kubios re-export is required for these five requested metrics.

## Agreement Statistics
| Metric | Domain | Number of recordings | Median relative error (%) | Mean relative error (%) | Pearson correlation coefficient (r) |
| --- | --- | --- | --- | --- | --- |
| RMSSD | Time-domain | 44 | 12.8889247561 | 47.6802795288 | 0.879564852648 |
| SDNN | Time-domain | 44 | 9.93261717344 | 42.0802948483 | 0.507425643805 |
| pNN50 | Time-domain | 38 | 26.239543689 | 535.707126151 | 0.662785173313 |
| SD1 | Nonlinear | 44 | 12.8928292364 | 47.6624324494 | 0.879235504335 |
| SD2 | Nonlinear | 44 | 15.1148388277 | 47.8331845682 | 0.400411935126 |

## Numerical Plausibility Checks
- RMSSD HRV Studio: min=12.7739, median=40.7248, max=215.715, outside_plausible_range=0
- RMSSD Kubios: min=9.6002, median=26.4257, max=128.012, outside_plausible_range=0
- SDNN HRV Studio: min=15.2033, median=60.8605, max=128.189, outside_plausible_range=0
- SDNN Kubios: min=13.1259, median=52.8897, max=99.5527, outside_plausible_range=0
- pNN50 HRV Studio: min=0, median=10.9542, max=38.7234, outside_plausible_range=0
- pNN50 Kubios: min=0, median=3.8947, max=33.7931, outside_plausible_range=0
- SD1 HRV Studio: min=9.03624, median=28.8123, max=152.859, outside_plausible_range=0
- SD1 Kubios: min=6.79285, median=18.7026, max=90.6101, outside_plausible_range=0
- SD2 HRV Studio: min=20.3489, median=82.9488, max=175.989, outside_plausible_range=0
- SD2 Kubios: min=17.2835, median=69.6385, max=138.561, outside_plausible_range=0

## Zero-Denominator Relative Error Checks
- RMSSD: 0 Kubios zero denominators.
- SDNN: 0 Kubios zero denominators.
- pNN50: 6 Kubios zero denominators.
- SD1: 0 Kubios zero denominators.
- SD2: 0 Kubios zero denominators.

## Output Files
- Summary table: `validation/kubios_subset/time_domain_nonlinear_kubios_agreement_summary.csv`
- Per-recording comparison: `validation/kubios_subset/time_domain_nonlinear_kubios_agreement_per_recording.csv`
- Report: `validation/research_notes/time_domain_nonlinear_kubios_validation_report.md`

## Limitations
- The existing Kubios parser was not modified because the requested output can be generated from the already exported raw reports without changing the Section 3.2 frequency-domain workflow.
- SDNN is mapped to Kubios `STD RR (ms)` because the report does not use a literal `SDNN (ms)` label for overall SDNN. Kubios `SDNN index (ms)` is a different metric and was not used.
- pNN50 shows larger and more skewed relative errors when Kubios pNN50 is zero or near zero. HRV Studio divides by successive RR differences; Kubios report conventions can differ by denominator definition.
