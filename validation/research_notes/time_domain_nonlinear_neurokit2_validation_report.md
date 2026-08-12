# Time-Domain and Nonlinear NeuroKit2 Agreement Extension

## Status
The target metrics were not available in the existing large-scale Section 3.1 NeuroKit2 agreement outputs. Existing v05 artifacts contained only frequency-domain metrics: VLF, LF, HF, total_power, LF/HF, LF_nu, and HF_nu. SDNN and RMSSD existed in smaller duration/robustness studies, but not for the full large-scale v05 dataset and not with pNN50, SD1, or SD2.

No manuscript files were modified. No validation pipeline code was modified. This extension reused the existing v05 run metadata and the same agreement error function from tools/validate_freq_domain_neurokit2.py.

## Existing Pipeline Identified
- tools/validate_freq_domain_neurokit2.py: Section 3.1 HRV Studio vs NeuroKit2 agreement runner; key functions reused or mirrored were resolve_inputs(), load_rr_intervals_ms(), relative_error(), and safe_abs_error().
- tools/analyze_validation_run.py: summary generator for row-level NeuroKit2 agreement CSVs.
- hrvlib/metrics/time_domain.py: HRVTimeDomainAnalysis.full_analysis() computes HRV Studio RMSSD, SDNN, and pNN50.
- hrvlib/metrics/nonlinear.py: NonlinearHRVAnalysis.poincare_analysis() computes HRV Studio SD1 and SD2.

## Dataset and Run Used
- Run: v05_physionet_10min_segment_linear
- Run metadata: validation\runs\v05_physionet_10min_segment_linear\run_info.json
- Validation script recorded in run metadata: tools\validate_freq_domain_neurokit2.py
- Input segments: 7598 ten-minute PhysioNet NSR RR segment files from validation/processed_data/physionet_nsr_rr_10min/.
- Failed files in this extension: 0.

## Preprocessing and Comparison Configuration
- RR loading followed the existing load_rr_intervals_ms() path, which calls load_rr_file() without auto_preprocess.
- Invalid, non-finite, and nonpositive RR intervals were removed before metric calculation, matching the existing frequency analyzer's sanitization basis.
- No artifact correction, interpolation, spectral detrending, or new analysis windows were introduced for these time-domain and Poincare metrics.
- Existing frequency-domain settings retained as run context: {'sampling_rate': 4.0, 'window_type': 'hann', 'segment_length': 120.0, 'overlap_ratio': 0.75, 'detrend_method': 'linear', 'detrend_lambda': 500.0, 'ar_order': 16, 'enable_diagnostics': True, 'experimental_native_welch_nfft_multiplier': 1.0, 'welch_detrend_mode': 'segment_linear'}
- Existing NeuroKit2 settings retained as run context: {'interpolation_rate': 4.0, 'interpolation_method': 'cubic', 'normalize': False, 'welch_detrend_mode': 'segment_linear'}
- NeuroKit2 time-domain values were computed with neurokit2.hrv_time() using direct RRI / RRI_Time input.
- NeuroKit2 SD1 and SD2 were computed with NeuroKit2's Poincare helper used internally by hrv_nonlinear() to avoid computing unrelated entropy/fractal endpoints.
- Relative error used the existing Section 3.1 formula: abs(HRV Studio - NeuroKit2) / abs(NeuroKit2) * 100; zero NeuroKit2 denominators are excluded from relative-error summaries.

## Agreement Statistics
| Metric | Domain | Number of recordings | Median relative error (%) | Mean relative error (%) | Pearson correlation coefficient (r) |
| --- | --- | --- | --- | --- | --- |
| RMSSD | Time-domain | 7598 | 0 | 0 | 1 |
| SDNN | Time-domain | 7598 | 0 | 0 | 1 |
| pNN50 | Time-domain | 7364 | 0.134953 | 0.137186 | 1 |
| SD1 | Nonlinear | 7598 | 0 | 7.27571e-15 | 1 |
| SD2 | Nonlinear | 7598 | 2.72874 | 4.72376 | 0.988248 |

## Output Files
- Summary table: validation\runs\v05_physionet_10min_segment_linear\time_domain_nonlinear_neurokit2_agreement_summary.csv
- Per-recording comparison: validation\runs\v05_physionet_10min_segment_linear\time_domain_nonlinear_neurokit2_agreement_per_recording.csv
- Report: validation\research_notes\time_domain_nonlinear_neurokit2_validation_report.md

## Numerical Plausibility Checks
- RMSSD HRV Studio value: min=7.71219, median=32.5609, max=483.382, outside_plausible_range=0
- RMSSD NeuroKit2 value: min=7.71219, median=32.5609, max=483.382, outside_plausible_range=0
- SDNN HRV Studio value: min=8.12018, median=54.6922, max=323.577, outside_plausible_range=0
- SDNN NeuroKit2 value: min=8.12018, median=54.6922, max=323.577, outside_plausible_range=0
- pNN50 HRV Studio value: min=0, median=3.83554, max=90.9639, outside_plausible_range=0
- pNN50 NeuroKit2 value: min=0, median=3.83025, max=90.7816, outside_plausible_range=0
- SD1 HRV Studio value: min=5.45599, median=23.0405, max=342.132, outside_plausible_range=0
- SD1 NeuroKit2 value: min=5.45599, median=23.0405, max=342.132, outside_plausible_range=0
- SD2 HRV Studio value: min=9.90062, median=74.6779, max=409.891, outside_plausible_range=0
- SD2 NeuroKit2 value: min=8.01604, median=71.6099, max=393.467, outside_plausible_range=0

## Limitations
- pNN50 shows small systematic disagreement because HRV Studio divides by the number of successive RR differences, while NeuroKit2 divides by the number of NN intervals.
- HRV Studio SD2 differs systematically from NeuroKit2 for many records because the current HRV Studio Poincare implementation uses sqrt(2 * SDNN^2 - 0.5 * SD1^2), whereas NeuroKit2 computes the standard deviation along the line of identity directly from lagged RR pairs.
- These outputs extend the large-scale NeuroKit2 validation numerically; they do not change production analysis code or manuscript text.
