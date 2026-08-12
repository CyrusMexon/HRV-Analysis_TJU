# Figure 4 Sequence Harmonization Update

## Original Figure 4 Source

- Existing rendered figure: `validation/research_notes/paper_figures/figure4_kubios_benchmark.png`
- Existing source notes: `validation/research_notes/paper_figures/figure4_kubios_benchmark_source_notes.md`
- Original generator script: not found in the repository; the source notes indicate Figure 4 was regenerated from stored validation CSV artifacts with Matplotlib.
- Plotting library used for this update: Matplotlib.

## Original Input Files

- `validation/research_notes/final_validation_results_package/final_kubios_metric_table.csv`
- `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`
- `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/kubios_parsed_results.csv`
- `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_all.csv`
- `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_valid_only.csv`
- `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_excluded_files.csv`

## New Input Files

- `validation/kubios_subset/frequency_domain_kubios_matched_sequence_summary.csv`
- `validation/kubios_subset/frequency_domain_kubios_matched_sequence_per_recording.csv`
- `validation/kubios_subset/frequency_domain_original_vs_matched_sequence_summary.csv`

## Output Files

- `validation/research_notes/paper_figures/figure4_kubios_matched_sequence.png`
- `validation/research_notes/paper_figures/figure4_kubios_matched_sequence.pdf`
- `validation/research_notes/paper_figures/figure4_kubios_matched_sequence.svg`

## Exact Changes

- Panel A was kept unchanged: n=50 parsed exports, n=48 matched rows, n=46 automatic QC retained, n=44 manual-review retained.
- Panel B now uses matched-sequence median relative errors from `validation/kubios_subset/frequency_domain_kubios_matched_sequence_summary.csv`.
- Panel C now uses `kubios_value` versus `selected_hrvstudio_value` for LF/HF from `validation/kubios_subset/frequency_domain_kubios_matched_sequence_per_recording.csv`. The annotation is Pearson r = 0.982 and median error = 5.55%.
- Panel D was changed from mean-versus-median error to paired original-versus-sequence-matched median relative error bars using `validation/kubios_subset/frequency_domain_original_vs_matched_sequence_summary.csv`.

## Visual Style Preservation

The update preserves the original 2x2 A-D layout, DejaVu Sans typography, panel-label format, manuscript-scale dimensions, bar/scatter styling, grid treatment, and compact source-data-only design. The substantive changes are limited to the data sources and the requested Panel D comparison.

## Row Counts

- Matched summary rows: 7
- Matched per-recording rows: 308
- LF/HF rows plotted in Panel C before finite/positive filtering: 44
- Original-versus-matched summary rows: 7
