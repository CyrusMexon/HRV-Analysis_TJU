# Figure 6 Synthetic Robustness Source Notes

Generated on 2026-06-15 from existing validation artifacts in the HRV-Analysis_TJU project. No full robustness experiment was rerun, no production HRV code was modified, and no original validation outputs were overwritten.

## Output Files

- `validation/research_notes/paper_figures/figure6_synthetic_robustness.png`
- `validation/research_notes/paper_figures/figure6_synthetic_robustness.pdf`
- `validation/research_notes/paper_figures/figure6_synthetic_robustness.svg`

## Exact Input Files Used

- Ten-base robustness results: `validation/runs/v09b_robustness_10base_signal_conditions/robustness_results.csv`
- Artifact effect table: `validation/runs/v09b_robustness_10base_signal_conditions/artifact_effect_table.csv`
- Warning summary: `validation/runs/v09b_robustness_10base_signal_conditions/warning_summary.csv`
- Condition-level stability table: `validation/runs/v09b_robustness_10base_signal_conditions/condition_level_stability_table.csv`
- Robustness summary: `validation/runs/v09b_robustness_10base_signal_conditions/robustness_summary.md`
- Run metadata and random seed: `validation/runs/v09b_robustness_10base_signal_conditions/run_info.json`
- Stored representative corrupted tachogram reviewed: `validation/runs/v09b_robustness_10base_signal_conditions/tachograms/nsr001_segment_000_ectopic_short_long_pairs_tachogram.png`
- Base RR file used for deterministic reconstruction of Panels A and B: `validation/processed_data/physionet_nsr_rr_10min/nsr001_segment_000.csv`
- Original two-base robustness run inspected as a supporting artifact: `validation/runs/v09_robustness_signal_conditions/robustness_summary.md`

## Selected Corruption Scenario

- Base file: `nsr001_segment_000.csv`
- Condition: `ectopic_short_long_pairs`
- Corruption note from validation logic: Inserted 11 short-long ectopic-like interval pairs.
- Reason selected: ectopic beat corruption was the highest-priority scenario requested by the figure specification and gives a visually clear short-long RR pattern with documented correction behavior.
- Stored v09b result row for this example: artifacts detected = 22, artifacts corrected = 22, extra beats removed = 8, intervals interpolated = 14.
- Reconstructed tachogram lengths: corrupted input = 927 intervals; corrected output = 919 intervals.
- Panels A and B use identical x/y axis limits for direct visual comparison.

## Panel C Values

- Metric rows finite before correction: 60/60.
- Metric rows finite after correction: 60/60.
- After-correction cases with at least one warning/diagnostic label: 60/60.

| Condition | Cases | Finite after correction (%) | Warning visibility (%) | Artifact detection (%) | Artifact correction (%) |
|---|---:|---:|---:|---:|---:|
| Clean | 10 | 100 | 100 | 50 | 50 |
| Ectopy | 10 | 100 | 100 | 100 | 100 |
| Missed | 10 | 100 | 100 | 100 | 100 |
| Jitter | 10 | 100 | 100 | 60 | 60 |
| Dropout | 10 | 100 | 100 | 50 | 50 |
| Extreme | 10 | 100 | 100 | 100 | 100 |

## Panel D Values

Panel D uses the selected `ectopic_short_long_pairs` example from `nsr001_segment_000.csv` and normalizes each metric to the same-file clean after-correction baseline. This avoids mixing units across RMSSD, spectral powers, and LF/HF.

| Metric | Clean value | Corrupted value | Corrected value | Corrupted (% clean) | Corrected (% clean) |
|---|---:|---:|---:|---:|---:|
| RMSSD | 18.6623 | 95.9513 | 18.6601 | 514.144 | 99.988 |
| LF | 571.015 | 740.367 | 560.929 | 129.658 | 98.2336 |
| HF | 68.6508 | 1274.96 | 69.9562 | 1857.16 | 101.901 |
| LF/HF | 8.31767 | 0.5807 | 8.01829 | 6.98152 | 96.4006 |

## Aggregation and Filtering

- Panel C aggregates the ten-base v09b robustness run at the condition level using the existing `condition_level_stability_table.csv` and `robustness_results.csv` files.
- Panel D is not aggregated; it shows one representative ectopic-beat example to match Panels A and B.
- No files or rows were excluded from the v09b Panel C summary.
- Panel A marks artifact indices detected by the same preprocessing pass used to reconstruct the corrected RR series; Panel B plots the corresponding corrected RR output.

## Regeneration/Re-use Status

- Existing tachogram PNGs were inspected but not directly reused because they contain the corrupted input series only and not a matched corrected RR series.
- Panels A and B were regenerated from the stored base RR file, recorded seed, and validation-only corruption/correction logic to reproduce the selected v09b example.
- Panels C and D were regenerated from existing CSV outputs.

## Assumptions

- `metrics_finite == True` is interpreted as numerical output stability, not physiological correctness.
- Any non-empty `warning_labels` field is counted as warning visibility.
- The selected corrected RR series demonstrates preprocessing behavior for a detector-compatible synthetic ectopic condition; it does not imply perfect recovery for all degraded signals.
- Robustness stress testing is interpreted as QC-aware numerical robustness, not clinical validation.
- Figure contains no overall title and no embedded caption; manuscript caption should be supplied separately.
