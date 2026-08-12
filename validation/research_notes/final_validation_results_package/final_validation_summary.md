# Final Validation Results Package

Generated from frozen validation artifacts after the Arm A no-detrend fix. Production HRV code was not modified and full validation was not rerun.

## Source Artifacts

- Cleaned Kubios comparison: `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`
- Pre-cleaning Kubios valid-only comparison: `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_valid_only.csv`
- Exclusion records: `validation/research_notes/manual_review_sensitivity_analysis/excluded_files_log.csv` and `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_excluded_files.csv`
- Large-scale summaries: `validation/runs/v03_physionet_5min_neurokit2/diagnostic_summary.md` and `validation/runs/v05_physionet_10min_segment_linear/diagnostic_summary.md`

## Large-Scale NeuroKit2 Validation Summary

### 5-min PhysioNet Validation

- Analyzed 106,253 metric rows across 15,179 files.
- Rows exceeding 20% relative error: 15,089; rows exceeding 80% relative error: 177.
- Metric-level median relative errors were low for LF (0.56%), HF (0.18%), LF/HF (0.66%), LF_nu (0.20%), and HF_nu (0.39%).
- VLF and total_power were the main higher-discrepancy metrics: VLF mean/median relative error 31.45%/28.23%; total_power mean/median relative error 15.15%/9.26%.
- The diagnostic report identified 9,899 files with an absolute-power mismatch pattern while preserving relative spectral distribution.

### 10-min PhysioNet Validation

- Analyzed 53,186 metric rows across 7,598 files using segment-wise linear Welch detrending.
- Rows exceeding 20% relative error: 271; rows exceeding 80% relative error: 2.
- Metric-level median relative errors were low for LF (0.39%), HF (0.11%), total_power (0.87%), LF/HF (0.44%), LF_nu (0.14%), and HF_nu (0.26%).
- VLF remained the largest p90 relative-error metric, with mean/median relative error 9.95%/9.18% and p90 14.81%.
- The 10-min analysis reduced the broad absolute-power discrepancy pattern to 197 files.

### Detrending Convention Finding

The large-scale runs indicate that Welch detrending convention materially affects VLF and total_power. Segment-wise linear detrending in the 10-min run markedly reduced discrepancies relative to the 5-min diagnostic pattern, while the Kubios subset was intentionally evaluated with no detrending to match the Kubios export convention after the Arm A fix.

### VLF/Total-Power Issue Explanation

VLF and total_power are especially sensitive because low-frequency bins, DC removal, interpolation, windowing, segment length, and bin-boundary integration can change the absolute spectral area without necessarily changing LF/HF or normalized-unit structure. This supports interpreting absolute-power disagreement as a methodological-convention issue unless raw-signal review indicates an implementation or input failure.

## Kubios Subset Summary

- Kubios exports parsed: 50/50.
- Matched validation rows: 48 files.
- Valid after automatic QC: 46 files.
- Retained after manual-review exclusions: 44 files.
- Final cleaned HRV Studio overall mean relative error: 88.85%.
- Final cleaned HRV Studio overall median relative error: 16.68%.

### Metric-Level HRV Studio Agreement

| Metric | Mean rel err % | Median rel err % | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- |
| VLF | 137.12 | 31.92 | 0.33 | 0.51 |
| LF | 134.14 | 20.42 | 0.83 | 0.84 |
| HF | 172.44 | 28.72 | 0.58 | 0.85 |
| total_power | 107.00 | 19.91 | 0.52 | 0.72 |
| LF/HF | 28.21 | 17.56 | 0.90 | 0.83 |
| LF_nu | 12.37 | 7.45 | 0.86 | 0.82 |
| HF_nu | 30.64 | 11.22 | 0.86 | 0.83 |

### Category-Level HRV Studio Agreement

| Category | Files | Mean rel err % | Median rel err % |
| --- | --- | --- | --- |
| clean_high_agreement | 18 | 57.54 | 17.79 |
| vlf_sensitive | 8 | 22.89 | 10.40 |
| remaining_outliers | 9 | 240.82 | 47.18 |
| short_or_adjusted | 5 | 8.00 | 2.34 |
| random_controls | 4 | 120.78 | 26.85 |

## Exclusion Summary

| Subset | Category | Stage | Reason class | Primary reason |
| --- | --- | --- | --- | --- |
| CH004 | clean_high_agreement | automatic_qc | validation input failure | No RRI/PPI intervals found; validation rows missing |
| CH012 | clean_high_agreement | automatic_qc | validation input failure | No RRI/PPI intervals found; validation rows missing |
| OUT005 | remaining_outliers | automatic_qc | preprocessing instability | Native power >100x Kubios; adjusted Welch overlap event |
| VLF005 | vlf_sensitive | automatic_qc | invalid/short signal; preprocessing instability | Nonfinite metrics, all native powers zero, native total power nonpositive; adjusted Welch overlap event |
| VLF002 | vlf_sensitive | manual_review | invalid/short signal; preprocessing instability | Manual-review exclusion after zero/near-zero HRV Studio powers and adjusted Welch overlap event |
| RC003 | random_controls | manual_review | invalid/short signal; preprocessing instability | Manual-review exclusion after zero HRV Studio powers/ratios and adjusted Welch overlap event |

## Main Interpretation

- Agreement is strongest for normalized and ratio metrics, especially LF_nu, LF/HF, and HF_nu by median relative error.
- Absolute spectral powers remain more sensitive to methodological conventions and retained pathological records.
- VLF and total_power are particularly convention-sensitive because they depend heavily on low-frequency treatment, detrending, segment duration, and frequency-bin integration.
- Pathological or unstable files require QC before being used for agreement claims; all-matched means are not appropriate as headline results.

## Paper-Ready Cautious Conclusions

The validation evidence supports cautious use of HRV Studio frequency-domain outputs when preprocessing settings and quality-control criteria are explicitly reported. The strongest support is for LF/HF and normalized-unit metrics, where the relative spectral distribution is often preserved. The evidence does not support claiming full equivalence with Kubios, particularly for absolute spectral powers and VLF-derived quantities, which remain sensitive to methodological conventions and input quality.

## Remaining Work Before Manuscript

- Review the retained high-error files, especially `OUT006`, `OUT003`, `OUT001`, `CH001`, `CH005`, `RC001`, `RC004`, and `OUT004`.
- Confirm Kubios report sample-limit and segment-selection fields for the subset before presenting absolute-power comparisons as definitive.
- Decide whether the manuscript should report medians as the primary Kubios subset statistic, with means treated as outlier-sensitive secondary statistics.
- Add a methods paragraph documenting detrending, Welch segment length, overlap, interpolation rate, window type, and QC exclusions.

## MIT-BIH Arrhythmia Robustness Update (v11)

- v11 run path: `validation/runs/v11_mitbih_arrhythmia_robustness/`
- Dataset path: `validation/raw_data/mit_bih_arrhythmia/`
- Script path: `tools/mitbih_arrhythmia_robustness_study.py`
- Segments analyzed: 12.
- Finite metrics before correction: 12/12.
- Finite metrics after correction: 12/12.
- After-correction warnings: 12/12.
- Corrected intervals present: 9/12.
- Metric sensitivity: LF, LF/HF, and RMSSD were the most sensitive to correction.
- Mostly normal rhythm correction effect: approximately 0%.
- Interpretation: arrhythmic records should be excluded from primary agreement validation and used only as robustness/QC stress tests.

Paper limitation wording: MIT-BIH arrhythmia recordings are robustness/QC stress tests, not clinical validation or diagnostic-accuracy evidence.
