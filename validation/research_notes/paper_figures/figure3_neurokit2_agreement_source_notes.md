# Figure 3 NeuroKit2 Agreement Source Notes

Generated on 2026-06-15 from existing validation artifacts in the HRV-Analysis_TJU project. No validation experiments were rerun, and no original validation outputs were modified.

## Output Files

- `validation/research_notes/paper_figures/figure3_neurokit2_agreement.png`
- `validation/research_notes/paper_figures/figure3_neurokit2_agreement.pdf`
- `validation/research_notes/paper_figures/figure3_neurokit2_agreement.svg`

## Exact Input Files Used

- Row-level 10-min comparison data for Panels A and B: `validation/runs/v05_physionet_10min_segment_linear/freq_domain_neurokit2_validation.csv`
- 10-min diagnostic summary for Panel C and Panel D 10-min bars: `validation/runs/v05_physionet_10min_segment_linear/diagnostic_summary.csv`
- 5-min diagnostic summary for Panel D 5-min bars: `validation/runs/v03_physionet_5min_neurokit2/diagnostic_summary.csv`
- Master numbers table used as cross-check: `validation/research_notes/final_master_validation_report/master_validation_numbers_table.csv`
- Master report reviewed as contextual source: `validation/research_notes/final_master_validation_report/master_validation_report.md`

## Panel A: LF/HF Agreement Scatter

- Source rows selected: `metric == "LF/HF"` from `validation/runs/v05_physionet_10min_segment_linear/freq_domain_neurokit2_validation.csv`.
- Rows before finite/positive filtering: 7598
- Rows plotted and analyzed after filtering: 7598
- Unique input files represented: 7598
- Filtering applied: non-finite values were removed and LF/HF pairs were required to be positive for log-scaled scatter axes. No rows were removed by this filter.
- Pearson correlation: 0.999417270
- Median relative error: 0.441998057%
- Axes: log-scaled x and y axes to preserve the full positive LF/HF range while retaining the identity line.
- Panel regenerated from CSV; no existing plot was reused.

## Panel B: LF/HF Bland-Altman

- Source rows selected: same 7598 finite positive LF/HF pairs as Panel A.
- Bias, HRV Studio minus NeuroKit2: 0.004141458
- SD of differences: 0.087910900
- Lower 1.96 SD limit: -0.168163906
- Upper 1.96 SD limit: 0.176446822
- Y-axis display range: -0.8 to 0.8 LF/HF units, chosen to show the bias and limits clearly. 8 extreme points fall outside this visual range; they were retained in all statistics above.
- Panel regenerated from CSV; no existing plot was reused.

## Panel C: 10-Min Median Relative Error Values

| Metric | Median relative error (%) |
|---|---:|
| LF | 0.388869 |
| HF | 0.112051 |
| total_power | 0.865509 |
| LF/HF | 0.441998 |
| LF_nu | 0.137526 |
| HF_nu | 0.262845 |
| VLF | 9.18477 |

- Source rows selected: `section == "metric_level_agreement"` from `validation/runs/v05_physionet_10min_segment_linear/diagnostic_summary.csv`.
- Number of files/rows represented per metric: 7598.
- Panel regenerated from diagnostic summary CSV; no existing plot was reused.

## Panel D: Five-Minute vs Ten-Minute Median Relative Error Values

| Metric | 5-min median relative error (%) | 10-min median relative error (%) | 5-min rows | 10-min rows |
|---|---:|---:|---:|---:|
| VLF | 28.2285 | 9.18477 | 15179 | 7598 |
| LF | 0.564672 | 0.388869 | 15179 | 7598 |
| HF | 0.182703 | 0.112051 | 15179 | 7598 |
| total_power | 9.26252 | 0.865509 | 15179 | 7598 |
| LF/HF | 0.657749 | 0.441998 | 15179 | 7598 |
| LF_nu | 0.195247 | 0.137526 | 15179 | 7598 |
| HF_nu | 0.386611 | 0.262845 | 15179 | 7598 |

- Source rows selected: `section == "metric_level_agreement"` from `validation/runs/v03_physionet_5min_neurokit2/diagnostic_summary.csv` and `validation/runs/v05_physionet_10min_segment_linear/diagnostic_summary.csv`.
- 5-min run represented: 15179 files / 106253 metric rows overall.
- 10-min run represented: 7598 files / 53186 metric rows overall.
- Panel regenerated from diagnostic summary CSVs; no existing plot was reused.

## Assumptions

- `native_value` in the NeuroKit2 comparison CSV represents HRV Studio output and `neurokit2_value` represents NeuroKit2 output, consistent with the validation summaries.
- `relative_error_pct` values were taken directly from validation artifacts rather than recomputed for Panels C and D.
- The 10-min segment-linear run (`v05_physionet_10min_segment_linear`) is the matched preprocessing/spectral-setting run emphasized in the master validation package.
- Figure contains no overall title and no embedded caption; manuscript caption should be supplied separately.
