# Figure 4 Kubios Benchmark Source Notes

Generated on 2026-06-15 from existing validation artifacts in the HRV-Analysis_TJU project. No validation experiments were rerun, no raw Kubios exports were parsed directly, and no original validation outputs were modified.

## Output Files

- `validation/research_notes/paper_figures/figure4_kubios_benchmark.png`
- `validation/research_notes/paper_figures/figure4_kubios_benchmark.pdf`
- `validation/research_notes/paper_figures/figure4_kubios_benchmark.svg`

## Exact Input Files Used

- Final metric summary for Panels B and cross-checks: `validation/research_notes/final_validation_results_package/final_kubios_metric_table.csv`
- Final validation summary for filtering counts and interpretation: `validation/research_notes/final_validation_results_package/final_validation_summary.md`
- Final cleaned row-level Kubios comparison for Panels C and D: `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`
- Parsed-results table used to count parsed exports: `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/kubios_parsed_results.csv`
- All comparison rows used to count matched files: `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_all.csv`
- Automatic-QC valid-only comparison used to count 46 retained files: `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_valid_only.csv`
- Automatic-QC exclusion records reviewed for the filtering flow: `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_excluded_files.csv`
- Master numbers table used as cross-check: `validation/research_notes/final_master_validation_report/master_validation_numbers_table.csv`
- Master validation report reviewed as contextual source: `validation/research_notes/final_master_validation_report/master_validation_report.md`

## Panel A: Kubios Subset Filtering

- Kubios exports parsed: 50
- Matched to validation rows: 48
- Retained after automatic QC: 46
- Retained after manual-review exclusions: 44
- Final cleaned row count: 308 metric rows across 7 metrics.
- Panel regenerated as a compact flow diagram; no existing plot was reused.

## Panel B: Median Relative Error by Metric

| Metric | Median relative error (%) | Mean relative error (%) | Pearson r | n files |
|---|---:|---:|---:|---:|
| LF_nu | 7.44717 | 12.3662 | 0.860384 | 44 |
| HF_nu | 11.2221 | 30.6441 | 0.86181 | 44 |
| LF/HF | 17.5646 | 28.2055 | 0.895023 | 44 |
| total_power | 19.9085 | 107.002 | 0.517846 | 44 |
| LF | 20.4165 | 134.141 | 0.834944 | 44 |
| HF | 28.7151 | 172.439 | 0.581347 | 44 |
| VLF | 31.9164 | 137.125 | 0.334506 | 44 |

- Source: `validation/research_notes/final_validation_results_package/final_kubios_metric_table.csv`.
- Metric order follows the final cleaned median relative error ranking requested for the figure.
- Panel regenerated from summary CSV; no existing plot was reused.

## Panel C: LF/HF Agreement

- Source rows selected: `metric == "LF/HF"` from `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`.
- Rows before finite/positive filtering: 44
- Rows plotted and analyzed after filtering: 44
- Unique files represented: 44
- Filtering applied: non-finite values were removed and LF/HF pairs were required to be positive. No rows were removed by this filter.
- Pearson correlation: 0.895022967
- Median relative error: 17.564579010%
- Panel regenerated from cleaned row-level CSV; no existing plot was reused.

## Panel D: Mean Versus Median Error

| Category | Mean relative error (%) | Median relative error (%) | Rows |
|---|---:|---:|---:|
| Overall | 88.8461 | 16.6806 | 308 |
| VLF | 137.125 | 31.9164 | 44 |
| HF | 172.439 | 28.7151 | 44 |
| total_power | 107.002 | 19.9085 | 44 |

- Source: `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`.
- Overall uses all finite HRV Studio relative-error rows in the final cleaned 44-file dataset.
- Metric-specific bars use final cleaned rows for VLF, HF, and total_power only.
- The panel is intended to show skew and outlier influence in the relative-error distribution. It does not imply that the mean is invalid; it supports using the median as the more representative headline statistic for this cleaned subset.
- Panel regenerated from cleaned row-level CSV; no existing plot was reused.

## Assumptions

- `kubios_value` is the benchmark value and `hrvstudio_native_value` is the HRV Studio value in the cleaned comparison CSV.
- `hrvstudio_relative_error_pct` is the manuscript-facing HRV Studio relative-error field for Kubios comparison.
- The manually reviewed `cleaned_valid_only.csv` file is the final cleaned 44-file Kubios subset specified for this figure.
- No raw Kubios export files were read; all values came from parsed comparison CSVs and validation summary artifacts.
- Figure contains no overall title and no embedded caption; manuscript caption should be supplied separately.
