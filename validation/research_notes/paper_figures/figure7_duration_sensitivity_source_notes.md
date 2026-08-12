# Figure 7 Duration Sensitivity Source Notes

Generated on 2026-06-15 from existing v10 duration-sensitivity validation artifacts in the HRV-Analysis_TJU project. Revised on 2026-06-16 to move annotation labels away from plotted lines and bar-label text. No validation experiments were rerun, no production HRV code was modified, and no original validation outputs were overwritten.

## Output Files

- `validation/research_notes/paper_figures/figure7_duration_sensitivity.png`
- `validation/research_notes/paper_figures/figure7_duration_sensitivity.pdf`
- `validation/research_notes/paper_figures/figure7_duration_sensitivity.svg`
- `validation/research_notes/paper_figures/generate_figure7_duration_sensitivity.py`

## Exact Input Files Used

- Duration metric summary: `validation/runs/v10_duration_sensitivity_validation/duration_metric_table.csv`
- Row-level duration results: `validation/runs/v10_duration_sensitivity_validation/duration_results.csv`
- VLF instability summary: `validation/runs/v10_duration_sensitivity_validation/vlf_instability_table.csv`
- Duration summary markdown: `validation/runs/v10_duration_sensitivity_validation/duration_summary.md`
- Run metadata: `validation/runs/v10_duration_sensitivity_validation/run_info.json`
- Existing duration error plot reviewed but not reused: `validation/runs/v10_duration_sensitivity_validation/duration_vs_error_plot.png`
- Existing duration correlation plot reviewed but not reused: `validation/runs/v10_duration_sensitivity_validation/duration_vs_correlation_plot.png`
- Master numbers table reviewed as cross-check: `validation/research_notes/final_master_validation_report/master_validation_numbers_table.csv`
- Master validation report reviewed as contextual source: `validation/research_notes/final_master_validation_report/master_validation_report.md`

## Panel A Values

Panel A plots median relative error versus NeuroKit2 from `validation/runs/v10_duration_sensitivity_validation/duration_metric_table.csv` for the requested metrics.

| Duration | LF | HF | total_power | LF/HF | LF_nu | HF_nu | VLF |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 6.85396 | 33.5059 | 8.56412 | 60.2931 | 9.82291 | 33.5999 | 100 |
| 60 s | 32.3725 | 8.4849 | 6.69999 | 27.6555 | 7.93516 | 25.3951 | 20.525 |
| 2 min | 19.6655 | 10.0305 | 4.32332 | 28.605 | 8.05482 | 24.2044 | 9.95271 |
| 5 min | 1.27397 | 14.5314 | 4.35951 | 11.0599 | 2.77348 | 8.17799 | 12.3308 |
| 10 min | 1.1389 | 13.5509 | 6.17465 | 11.0544 | 2.48251 | 9.15317 | 12.9595 |

## Panel B Values

Panel B uses row-level VLF relative error versus NeuroKit2 from `validation/runs/v10_duration_sensitivity_validation/duration_results.csv`.

| Duration | n | VLF median relative error (%) | VLF p90 relative error (%) |
|---|---:|---:|---:|
| 30 s | 50 | 100 | 100 |
| 60 s | 50 | 20.525 | 27.1022 |
| 2 min | 50 | 9.95271 | 14.2685 |
| 5 min | 50 | 12.3308 | 14.8554 |
| 10 min | 50 | 12.9595 | 16.8073 |

## Panel C Values

Finite-output and warning-visible rates were calculated from all row-level metric records in `validation/runs/v10_duration_sensitivity_validation/duration_results.csv`. A row was counted as finite when `finite_hrvstudio == True`. A row was counted as warning-visible when `warning_count > 0`. Panel C also overlays the recording-level duration-warning rate to distinguish numerical computability from short-duration interpretability warnings.

| Duration | Files | Metric rows | Finite-output rate (%) | Warning-visible rate (%) | Duration-warning rate (%) |
|---|---:|---:|---:|---:|---:|
| 30 s | 50 | 450 | 100 | 100 | 100 |
| 60 s | 50 | 450 | 100 | 100 | 100 |
| 2 min | 50 | 450 | 100 | 100 | 100 |
| 5 min | 50 | 450 | 100 | 100 | 100 |
| 10 min | 50 | 450 | 100 | 100 | 0 |

## Metrics Included

- LF
- HF
- total_power
- LF/HF
- LF_nu
- HF_nu
- VLF

## Filtering Applied

- Panel A used rows in `duration_metric_table.csv` matching the seven requested metrics and five requested duration labels.
- Panel B used only `metric == "VLF"` rows with finite `relative_error_vs_neurokit2_pct` values.
- Panel C used all row-level metric rows in `duration_results.csv`; no duration or metric rows were excluded.

## Regeneration/Re-use Status

- Existing v10 PNG outputs were inspected but not reused because the requested figure combines metric agreement, VLF-specific behavior, and finite/warning behavior in one publication layout.
- All panels were regenerated from existing CSV outputs only.
- The 2026-06-16 revision used `generate_figure7_duration_sensitivity.py` to refresh the PNG, PDF, and SVG outputs after repositioning three annotations:
  - Panel A `<5 min: caution for spectral metrics` moved below the x-axis.
  - Panel B `VLF remains convention-sensitive` moved below the x-axis.
  - Panel C `Finite output does not imply interpretability` moved downward into the bar region so it does not cover the `50 files` labels above the bars.

## Assumptions

- Median relative error versus NeuroKit2 is the agreement statistic requested for Panels A and B.
- `warning_count > 0` is interpreted as warning visibility.
- Finite output is interpreted as numerical computability, not physiological interpretability.
- The shaded area in Panels A and B marks durations shorter than 5 minutes, reinforcing caution rather than invalidating short-duration output.
- Figure supports recommending at least 5 minutes for primary frequency-domain reporting, with strongest support at 10 minutes in this validation framework.
- Figure contains no overall title and no embedded caption; manuscript caption should be supplied separately.

## No Experiment Rerun Confirmation

No duration-sensitivity validation experiment was rerun. The figure was generated only from stored v10 CSV/markdown artifacts.
