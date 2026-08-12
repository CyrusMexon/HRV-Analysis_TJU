# Frequency-Band Corrected Validation Rerun Report

Generated on 2026-08-11. Manuscript files were not modified, legacy validation outputs were not overwritten, and no commit was made.

## Band Conventions Used

- Section 3.1 NeuroKit2 agreement: `standard` (`0 < f <= 0.40` total, VLF `0.003 <= f < 0.04`).
- Section 3.2 Kubios sequence-harmonized benchmark: `kubios_compatible` to match exported Kubios VLF/total conventions.
- Section 3.3 Smoothness Priors sensitivity: `kubios_compatible`.
- Alternative spectral-method comparison, duration sensitivity, synthetic robustness, and MIT-BIH/QC: `standard`.

## Integrity Checks

- Section 3.1 run metadata records `band_convention = standard` for HRV Studio and NeuroKit2 comparator masks.
- Section 3.2 used `HRV.Data.RRs` selected NN sequences from the established 44-recording audit and `kubios_compatible` mode.
- Section 3.3 used the Smoothness Priors export subset and `kubios_compatible` mode.
- No manuscript file was edited.
- New outputs were written under `v12_band_corrected_*` run directories or `_band_corrected_20260811` suffixed Kubios paths.
- A partial failed MIT-BIH retry folder exists at `validation/runs/v12_band_corrected_mitbih_arrhythmia_robustness_standard_20260811`; the completed rerun is the `_complete` folder.

## Section 3.1: Large-Scale NeuroKit2 Agreement

Old 5-minute output: `validation/runs/v03_physionet_5min_neurokit2/freq_domain_neurokit2_validation.csv`  
New 5-minute output: `validation/runs/v12_band_corrected_physionet_5min_neurokit2_standard_20260811/comparison.csv`  
New 5-minute summary: `validation/runs/v12_band_corrected_physionet_5min_neurokit2_standard_20260811/metric_summary.csv`

Corrected 5-minute headline values:

| Metric | n | Median relative error % | Mean relative error % | Pearson r |
| --- | --- | --- | --- | --- |
| VLF | 15179 | 37.79 | 41.00 | 0.90 |
| LF | 15179 | 1.35 | 2.40 | 0.99 |
| HF | 15179 | 0.18 | 1.17 | 1.00 |
| Total power | 15179 | 13.57 | 19.38 | 0.98 |
| LF/HF | 15179 | 1.41 | 2.63 | 1.00 |
| LFnu | 15179 | 0.42 | 1.10 | 1.00 |
| HFnu | 15179 | 0.81 | 1.51 | 1.00 |

Old-vs-new 5-minute comparison: `validation/research_notes/frequency_band_corrected_validation_rerun_tables/section3_1_5min_old_vs_new.csv`.

Old 10-minute segment-linear output: `validation/runs/v05_physionet_10min_segment_linear/freq_domain_neurokit2_validation.csv`  
New 10-minute segment-linear output: `validation/runs/v12_band_corrected_physionet_10min_segment_linear_standard_20260811/comparison.csv`  
New 10-minute summary: `validation/runs/v12_band_corrected_physionet_10min_segment_linear_standard_20260811/metric_summary.csv`

Corrected 10-minute segment-linear headline values:

| Metric | n | Median relative error % | Mean relative error % | Pearson r |
| --- | --- | --- | --- | --- |
| VLF | 7598 | 22.72 | 22.96 | 1.00 |
| LF | 7598 | 1.21 | 1.61 | 1.00 |
| HF | 7598 | 0.11 | 0.45 | 1.00 |
| Total power | 7598 | 6.03 | 6.82 | 1.00 |
| LF/HF | 7598 | 1.25 | 1.68 | 1.00 |
| LFnu | 7598 | 0.37 | 0.74 | 1.00 |
| HFnu | 7598 | 0.72 | 0.95 | 1.00 |

Old-vs-new 10-minute comparison: `validation/research_notes/frequency_band_corrected_validation_rerun_tables/section3_1_10min_segment_linear_old_vs_new.csv`.

Manuscript consequence: numerical frequency-domain values in Section 3.1 need updating. The qualitative pattern remains metric-dependent agreement: LF and normalized/ratio metrics are stronger than VLF/total-power metrics.

## Section 3.2: Sequence-Harmonized Kubios Benchmark

Old sequence-harmonized output: `validation/kubios_subset/frequency_domain_kubios_matched_sequence_summary.csv`  
New output: `validation/kubios_subset/frequency_domain_kubios_matched_sequence_summary_band_corrected_20260811.csv`  
New per-recording output: `validation/kubios_subset/frequency_domain_kubios_matched_sequence_per_recording_band_corrected_20260811.csv`

Corrected Kubios values:

| Metric | n | Median relative error % | Mean relative error % | Pearson r |
| --- | --- | --- | --- | --- |
| VLF | 44 | 14.01 | 17.06 | 0.99 |
| LF | 44 | 3.49 | 7.10 | 1.00 |
| HF | 44 | 2.54 | 6.17 | 0.97 |
| Total power | 44 | 2.51 | 5.65 | 0.99 |
| LF/HF | 44 | 5.55 | 8.55 | 0.98 |
| LFnu | 44 | 1.46 | 2.76 | 0.99 |
| HFnu | 44 | 3.79 | 8.18 | 0.99 |

Unexpected change after production refactor: no. Maximum median-RE absolute change from the prior sequence-harmonized Kubios-compatible output was 0.0000 percentage points.

Manuscript consequence: Section 3.2 should continue using the already corrected sequence-harmonized Kubios values. No additional numerical change was introduced by the production refactor when `kubios_compatible` mode is used.

## Section 3.3: Smoothness Priors Sensitivity

Old output directory: `validation/kubios_subset/smoothness_priors_pilot`  
New output directory: `validation/kubios_subset/smoothness_priors_pilot_band_corrected_20260811`  
New report: `validation/research_notes/kubios_smoothness_priors_pilot_report_band_corrected_20260811.md`

Old-vs-new Smoothness Priors summary:

| Method | Metric | old_n | new_n | old_median_RE_pct | new_median_RE_pct | median_RE_abs_change_pct | old_Pearson_r | new_Pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FFT | VLF | 10 | 10 | 17.44 | 17.44 | 0.00 | 0.59 | 0.59 |
| FFT | LF | 10 | 10 | 18.41 | 18.41 | 0.00 | 0.91 | 0.91 |
| FFT | HF | 10 | 10 | 6.62 | 6.62 | 0.00 | 0.99 | 0.99 |
| FFT | Total power | 10 | 10 | 15.78 | 15.78 | 0.00 | 0.92 | 0.92 |
| FFT | LF/HF | 10 | 10 | 14.83 | 14.83 | 0.00 | 0.99 | 0.99 |
| FFT | LFnu | 10 | 10 | 4.10 | 4.10 | 0.00 | 0.96 | 0.96 |
| FFT | HFnu | 10 | 10 | 11.48 | 11.48 | 0.00 | 0.96 | 0.96 |
| AR | VLF | 10 | 10 | 1.40 | 1.40 | 0.00 | 1.00 | 1.00 |
| AR | LF | 10 | 10 | 1.39 | 1.39 | 0.00 | 1.00 | 1.00 |
| AR | HF | 10 | 10 | 3.18 | 3.18 | 0.00 | 0.99 | 0.99 |
| AR | Total power | 10 | 10 | 1.74 | 1.74 | 0.00 | 1.00 | 1.00 |
| AR | LF/HF | 10 | 10 | 4.38 | 4.38 | 0.00 | 1.00 | 1.00 |
| AR | LFnu | 10 | 10 | 1.16 | 1.16 | 0.00 | 0.99 | 0.99 |
| AR | HFnu | 10 | 10 | 2.14 | 2.14 | 0.00 | 0.99 | 0.99 |

Headline verification: FFT VLF median RE 17.44%, FFT total power 15.78%; AR VLF 1.40%, AR total power 1.74%.

Manuscript consequence: no numerical update is required for the reported Smoothness Priors headline values. The interpretation remains supported: AR agreement is substantially closer than FFT under matched Smoothness Priors, while FFT retains larger residual differences.

## Alternative Spectral-Method Comparison

Old output: `validation/runs/v08_fft_ar_validation/fft_ar_comparison.csv`  
New output: `validation/runs/v12_band_corrected_fft_ar_validation_standard_20260811/fft_ar_comparison.csv`

Corrected method-comparison summary:

| Metric | n | FFT median RE vs Welch % | AR median RE vs Welch % | Welch median RE vs NeuroKit2 % | FFT Pearson r vs Welch | AR Pearson r vs Welch |
| --- | --- | --- | --- | --- | --- | --- |
| VLF | 100 | 37.69 | 106.65 | 40.68 | 0.85 | 0.92 |
| LF | 100 | 19.03 | 11.22 | 0.92 | 0.97 | 0.92 |
| HF | 100 | 19.58 | 9.59 | 15.40 | 0.98 | 1.00 |
| Total power | 100 | 5332.19 | 95.11 | 24.20 | 0.51 | 0.85 |
| LF/HF | 100 | 16.44 | 11.24 | 13.34 | 0.85 | 0.90 |
| LFnu | 100 | 4.63 | 3.13 | 3.24 | 0.96 | 0.93 |
| HFnu | 100 | 10.94 | 7.35 | 11.35 | 0.96 | 0.93 |
| FFT instability flags | 100 | 100.00 | NA | NA | NA | NA |
| AR instability flags | 100 | NA | 3.00 | NA | NA | NA |

Main findings after correction: FFT still shows severe total-power/PSD-area sensitivity under the no-detrend baseline; AR remains more variance-consistent but method- and order-sensitive. The qualitative conclusion does not change.

## Duration Sensitivity

Old output: `validation/runs/v10_duration_sensitivity_validation/duration_metric_table.csv`  
New output: `validation/runs/v12_band_corrected_duration_sensitivity_standard_20260811/duration_metric_table.csv`  
New frequency-only summary: `validation/runs/v12_band_corrected_duration_sensitivity_standard_20260811/frequency_metric_summary.csv`

Corrected duration headline values:

| duration_label | metric | n | median_relative_error_vs_neurokit2_pct | mean_relative_error_vs_neurokit2_pct | pearson_correlation_vs_neurokit2 | median_abs_change_vs_10min_hrvstudio_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 30s | VLF | 50 | NA | NA | NA | 100.00 |
| 30s | LF | 50 | 6.85 | 11.88 | 0.97 | 63.86 |
| 30s | HF | 50 | 33.51 | 34.28 | 1.00 | 80.60 |
| 30s | total_power | 50 | 28.64 | 29.45 | 0.99 | 80.36 |
| 30s | LF/HF | 50 | 60.29 | 68.86 | 0.94 | 66.98 |
| 30s | LF_nu | 50 | 9.82 | 11.22 | 0.99 | 20.08 |
| 30s | HF_nu | 50 | 33.60 | 29.53 | 0.99 | 59.39 |
| 60s | VLF | 50 | 100.00 | 100.00 | NA | 100.00 |
| 60s | LF | 50 | 32.37 | 34.23 | 0.99 | 61.69 |
| 60s | HF | 50 | 8.48 | 12.24 | 1.00 | 67.89 |
| 60s | total_power | 50 | 24.06 | 23.61 | 0.98 | 71.79 |
| 60s | LF/HF | 50 | 27.66 | 29.99 | 0.95 | 52.47 |
| 60s | LF_nu | 50 | 7.94 | 11.45 | 0.97 | 17.84 |
| 60s | HF_nu | 50 | 25.40 | 31.31 | 0.97 | 53.48 |
| 2min | VLF | 50 | 41.86 | 40.29 | 0.99 | 69.32 |
| 2min | LF | 50 | 19.67 | 20.73 | 0.99 | 56.92 |
| 2min | HF | 50 | 10.03 | 16.35 | 1.00 | 50.96 |
| 2min | total_power | 50 | 17.58 | 18.78 | 0.95 | 51.36 |
| 2min | LF/HF | 50 | 28.61 | 30.30 | 1.00 | 45.47 |
| 2min | LF_nu | 50 | 8.05 | 11.85 | 0.99 | 14.50 |
| 2min | HF_nu | 50 | 24.20 | 27.55 | 0.99 | 35.68 |
| 3min | VLF | 50 | 32.25 | 30.09 | 0.98 | 53.48 |
| 3min | LF | 50 | 12.85 | 16.58 | 0.98 | 51.03 |
| 3min | HF | 50 | 21.27 | 34.58 | 1.00 | 41.11 |
| 3min | total_power | 50 | 19.76 | 21.27 | 0.91 | 49.04 |
| 3min | LF/HF | 50 | 19.20 | 23.38 | 0.94 | 31.20 |
| 3min | LF_nu | 50 | 4.11 | 8.80 | 0.94 | 7.94 |
| 3min | HF_nu | 50 | 14.53 | 17.89 | 0.94 | 22.99 |
| 5min | VLF | 50 | 33.71 | 34.01 | 0.99 | 36.97 |
| 5min | LF | 50 | 1.05 | 1.85 | 1.00 | 29.96 |
| 5min | HF | 50 | 14.53 | 24.56 | 1.00 | 25.02 |
| 5min | total_power | 50 | 17.69 | 18.60 | 0.92 | 29.79 |
| 5min | LF/HF | 50 | 11.98 | 16.75 | 0.99 | 32.89 |
| 5min | LF_nu | 50 | 3.08 | 6.64 | 0.98 | 7.50 |
| 5min | HF_nu | 50 | 9.43 | 13.73 | 0.98 | 22.63 |
| 10min | VLF | 50 | 38.75 | 38.57 | 0.98 | 0.00 |
| 10min | LF | 50 | 0.71 | 1.35 | 1.00 | 0.00 |
| 10min | HF | 50 | 13.55 | 23.89 | 1.00 | 0.00 |
| 10min | total_power | 50 | 22.83 | 22.42 | 0.90 | 0.00 |
| 10min | LF/HF | 50 | 12.57 | 16.63 | 0.99 | 0.00 |
| 10min | LF_nu | 50 | 3.06 | 6.23 | 0.98 | 0.00 |
| 10min | HF_nu | 50 | 10.13 | 13.78 | 0.98 | 0.00 |

Corrected 5-minute VLF median RE vs NeuroKit2: 33.71%. Corrected 10-minute VLF median RE vs NeuroKit2: 38.75%.

Manuscript consequence: duration-sensitivity frequency values and Figure 7 require regeneration. The conclusions remain supported: short recordings show greater low-frequency instability; 5 minutes remains the practical short-term standard; 10 minutes gives the lowest within-HRV Studio numerical change because it is the reference duration in this analysis. This does not imply that 10 minutes is clinically required.

## Synthetic Robustness

Old output: `validation/runs/v09b_robustness_10base_signal_conditions/robustness_results.csv`  
New output: `validation/runs/v12_band_corrected_robustness_10base_signal_conditions_standard_20260811/robustness_results.csv`

Headline finite/warning impact:

| version | rows | finite_rate_pct | warning_rows_pct |
| --- | --- | --- | --- |
| old | 120 | 100.00 | 100.00 |
| new | 120 | 100.00 | 100.00 |

Condition-level corrected summary:

| condition | cases | finite_after_count | finite_after_rate_pct | artifact_detection_rate_pct | artifact_correction_rate_pct | median_artifacts_detected | median_artifacts_corrected | warning_rate_pct | median_abs_after_vs_clean_change_pct | median_abs_after_vs_before_change_pct | max_abs_after_vs_clean_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_jitter_noise | 10 | 10 | 100.00 | 60.00 | 60.00 | 1.00 | 1.00 | 100.00 | 31.68 | 1.48 | 596.49 |
| isolated_extreme_rr_artifacts | 10 | 10 | 100.00 | 100.00 | 100.00 | 4.50 | 4.50 | 100.00 | 3.22 | 91.94 | 51.70 |
| ectopic_short_long_pairs | 10 | 10 | 100.00 | 100.00 | 100.00 | 19.50 | 19.50 | 100.00 | 1.54 | 78.21 | 98.69 |
| short_dropout_sections | 10 | 10 | 100.00 | 50.00 | 50.00 | 0.50 | 0.50 | 100.00 | 1.01 | 0.17 | 13.47 |
| random_missed_beats | 10 | 10 | 100.00 | 100.00 | 100.00 | 13.00 | 13.00 | 100.00 | 0.71 | 84.42 | 20.40 |
| clean | 10 | 10 | 100.00 | 50.00 | 50.00 | 0.50 | 0.50 | 100.00 | 0.00 | 0.14 | 0.00 |

Manuscript consequence: finite-output rates and QC-warning conclusions are unchanged. Frequency-domain robustness values involving total power should be numerically refreshed if reported or plotted.

## MIT-BIH QC / Engineering Stress Test

Old output: `validation/runs/v11_mitbih_arrhythmia_robustness/mitbih_robustness_results.csv`  
New completed output: `validation/runs/v12_band_corrected_mitbih_arrhythmia_robustness_standard_20260811_complete/mitbih_robustness_results.csv`

Headline finite/warning impact:

| version | rows | segments | finite_rate_pct | warning_rows_pct |
| --- | --- | --- | --- | --- |
| old | 24 | 12 | 100.00 | 100.00 |
| new | 24 | 12 | 100.00 | 100.00 |

Manuscript consequence: the engineering stress-test interpretation is unchanged, but Figure 8/source values should be regenerated from the completed corrected run if frequency-domain values or warning labels are displayed.

## Manuscript Component Classification

| Manuscript component | Classification | Reason |
| --- | --- | --- |
| Section 3.1 NeuroKit2 frequency values | NUMERICAL UPDATE ONLY | Standard non-overlapping bands changed VLF/total definitions; qualitative agreement pattern remains metric-dependent. |
| Section 3.2 Kubios frequency values | UNCHANGED | Kubios-compatible mode reproduces sequence-harmonized comparator convention to rounding. |
| Section 3.3 Smoothness Priors values | UNCHANGED | Kubios-compatible mode reproduces the prior Kubios convention; FFT/AR headline values unchanged to rounding. |
| Alternative spectral-method table/text | NUMERICAL UPDATE ONLY | Standard bands change VLF/total magnitudes; method-dependence conclusion remains. |
| Duration sensitivity values and Figure 7 | FIGURE REGENERATION REQUIRED | Duration frequency metrics changed under standard bands; new 180 s row was generated as requested. |
| Synthetic robustness values and Figure 6 | NUMERICAL UPDATE ONLY | Finite/warning conclusions unchanged; total-power values change under standard bands. |
| MIT-BIH/QC values and Figure 8 | FIGURE REGENERATION REQUIRED | Finite/QC interpretation unchanged, but warning labels now include ULF few-bin diagnostics and spectral values changed. |
| Abstract/discussion broad claims | NUMERICAL UPDATE ONLY | No major scientific conclusion changed, but frequency-domain numbers should be refreshed. |

## Commands Used

- `python -m py_compile tools\validate_freq_domain_neurokit2.py tools\validate_fft_ar_methods.py tools\duration_sensitivity_validation.py tools\robustness_signal_condition_study.py tools\mitbih_arrhythmia_robustness_study.py tools\frequency_domain_sequence_audit.py tools\run_kubios_smoothness_priors_pilot.py`
- `python tools\validate_freq_domain_neurokit2.py validation\processed_data\physionet_nsr_rr_5min --run-name v12_band_corrected_physionet_5min_neurokit2_standard_20260811 --output comparison.csv --purpose "Band-corrected Section 3.1 5-minute NeuroKit2 validation using standard non-overlapping frequency bands." --interpolation-rate 4 --window-type hann --segment-length 120 --overlap-ratio 0.75 --detrend-method linear --detrend-lambda 500 --ar-order 16 --neurokit-interpolation-method cubic --enable-diagnostics`
- `python tools\validate_freq_domain_neurokit2.py validation\processed_data\physionet_nsr_rr_10min --run-name v12_band_corrected_physionet_10min_segment_linear_standard_20260811 --output comparison.csv --purpose "Band-corrected Section 3.1 10-minute segment-linear NeuroKit2 validation using standard non-overlapping frequency bands." --interpolation-rate 4 --window-type hann --segment-length 120 --overlap-ratio 0.75 --detrend-method linear --detrend-lambda 500 --ar-order 16 --neurokit-interpolation-method cubic --welch-detrend-mode segment_linear --enable-diagnostics`
- `$env:HRV_FREQ_BAND_RERUN_SUFFIX='_band_corrected_20260811'; python tools\frequency_domain_sequence_audit.py`
- `$env:HRV_SMOOTHNESS_RERUN_SUFFIX='_band_corrected_20260811'; python tools\run_kubios_smoothness_priors_pilot.py`
- `python tools\validate_fft_ar_methods.py --run-dir validation\runs\v12_band_corrected_fft_ar_validation_standard_20260811 --input-dir validation\processed_data\physionet_nsr_rr_10min --subset-size 100 --min-duration-s 590 --max-invalid-removed-pct 1.0 --interpolation-rate 4 --segment-length 120 --overlap-ratio 0.75 --window-type hann --detrend-method none --neurokit-interpolation-method monotone_cubic`
- `python tools\duration_sensitivity_validation.py --run-dir validation\runs\v12_band_corrected_duration_sensitivity_standard_20260811 --input-dir validation\processed_data\physionet_nsr_rr_10min --sample-size 50 --min-full-duration-s 590 --max-invalid-removed-pct 1.0 --interpolation-rate 4 --segment-length 120 --overlap-ratio 0.75 --window-type hann --detrend-method none --neurokit-interpolation-method monotone_cubic --kubios-comparison validation\kubios_subset\parsed_results_50_none_120s_75pct_after_arm_a\comparison_valid_only.csv`
- `python tools\robustness_signal_condition_study.py --run-dir validation\runs\v12_band_corrected_robustness_10base_signal_conditions_standard_20260811 --input-dir validation\processed_data\physionet_nsr_rr_10min --base-file-count 10 --min-duration-s 590 --max-invalid-removed-pct 1.0 --seed 20260609 --interpolation-rate 4 --segment-length 120 --overlap-ratio 0.75 --window-type hann --detrend-method none`
- `python tools\mitbih_arrhythmia_robustness_study.py --run-name v12_band_corrected_mitbih_arrhythmia_robustness_standard_20260811_complete --output-root . --archive-dir validation\raw_data\archives --zip-pattern "mit-bih-arrhythmia-database*.zip" --segment-duration-s 600 --window-step-s 300 --min-segment-duration-s 540 --min-rr-count 250 --max-segments 12 --interpolation-rate 4 --segment-length 120 --overlap-ratio 0.75 --window-type hann --detrend-method none`

## Generated Summary Tables

- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\alternative_spectral_methods_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\duration_sensitivity_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\manuscript_component_update_classification.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\mitbih_qc_headline_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\mitbih_warning_summary_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\section3_1_10min_segment_linear_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\section3_1_5min_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\section3_2_kubios_matched_sequence_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\section3_3_smoothness_priors_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\synthetic_robustness_headline_old_vs_new.csv`
- `validation\research_notes\frequency_band_corrected_validation_rerun_tables\synthetic_robustness_old_vs_new.csv`
