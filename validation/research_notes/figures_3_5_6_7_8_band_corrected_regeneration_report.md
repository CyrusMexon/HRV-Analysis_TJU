# Figures 3, 5, 6, 7, and 8 Band-Corrected Regeneration Report

Generated from stored v12 band-corrected validation outputs. Manuscript files and legacy figure files were not modified.

Band convention: standard: ULF 0 < f < 0.003 Hz; VLF 0.003 <= f < 0.04 Hz; LF 0.04 <= f < 0.15 Hz; HF 0.15 <= f <= 0.40 Hz; total 0 < f <= 0.40 Hz.

## Phase 1 Inventory

| Figure | Original script | Legacy input paths | Legacy output paths | Data level | Hard-coded numerical annotations |
|---|---|---|---|---|---|
| Figure 3 | No retained generator; source notes only | v05 row-level + v03/v05 diagnostic summaries | figure3_neurokit2_agreement.{png,pdf,svg} | row-level for A/B; summary for C/D | source notes only |
| Figure 5 | No retained generator; source notes only | v08 fft_ar_comparison + instability + ar_order_sensitivity | figure5_spectral_methods.{png,pdf,svg} | row-level/method tables | source notes only |
| Figure 6 | No retained generator; source notes only | v09b robustness CSVs + base RR file | figure6_synthetic_robustness.{png,pdf,svg} | raw reconstruction for A/B; CSV for C/D | source notes only |
| Figure 7 | validation/research_notes/paper_figures/generate_figure7_duration_sensitivity.py | v10 duration_metric_table + duration_results | figure7_duration_sensitivity.{png,pdf,svg} | summary + row-level | script constants |
| Figure 8 | No retained generator; source notes only | v11 MIT-BIH CSVs + local annotations | figure8_mitbih_arrhythmia_robustness.{png,pdf,svg} | raw reconstruction for A-C; CSV for D | source notes only |

## Regenerated Outputs

### Figure 3

- Original script: `validation/research_notes/paper_figures/generate_band_corrected_figures_3_5_6_7_8.py`; no retained legacy Figure 3 generator was found.
- New input files: `validation/runs/v12_band_corrected_physionet_10min_segment_linear_standard_20260811/comparison.csv`, `validation/runs/v12_band_corrected_physionet_10min_segment_linear_standard_20260811/metric_summary.csv`, `validation/runs/v12_band_corrected_physionet_5min_neurokit2_standard_20260811/metric_summary.csv`
- New output files: `validation/figures/Figure3_neurokit2_band_corrected.png`, `validation/figures/Figure3_neurokit2_band_corrected.pdf`, `validation/figures/Figure3_neurokit2_band_corrected.svg`
- Panels regenerated: A-D
- Old headline values: LF/HF r 0.999417; median RE 0.441998%; VLF 10-min median RE 9.18477%.
- New headline values: LF/HF r 0.999; median RE 1.25%; BA bias -0.0241; LOA -0.2025 to 0.1543.
- Layout changed: No panel arrangement change; Panel D uses log-scaled horizontal paired bars to keep VLF and sub-2% metrics readable.
- Logic changes: Inputs changed to v12 standard-band outputs; annotations are recomputed.
- Expected-value discrepancy: None; generated LF/HF values match the corrected run summary within rounding.

### Figure 5

- Original script: `validation/research_notes/paper_figures/generate_band_corrected_figures_3_5_6_7_8.py`; no retained legacy Figure 5 generator was found.
- New input files: `validation/runs/v12_band_corrected_fft_ar_validation_standard_20260811/fft_ar_comparison.csv`, `validation/runs/v12_band_corrected_fft_ar_validation_standard_20260811/ar_order_sensitivity.csv`
- New output files: `validation/figures/Figure5_spectral_methods_band_corrected.png`, `validation/figures/Figure5_spectral_methods_band_corrected.pdf`, `validation/figures/Figure5_spectral_methods_band_corrected.svg`
- Panels regenerated: A-C
- Old headline values: FFT instability 100/100; AR instability 2/100; AR order-sensitive 25/100.
- New headline values: FFT instability 100/100; AR instability 3/100; order-sensitive 25/100.
- Layout changed: No.
- Logic changes: Inputs changed to v12 standard-band spectral-method outputs; AR instability annotation updates from old 2/100 to corrected 3/100.
- Expected-value discrepancy: None.

### Figure 6

- Original script: `validation/research_notes/paper_figures/generate_band_corrected_figures_3_5_6_7_8.py`; no retained legacy Figure 6 generator was found.
- New input files: `validation/runs/v12_band_corrected_robustness_10base_signal_conditions_standard_20260811/robustness_results.csv`, `validation/runs/v12_band_corrected_robustness_10base_signal_conditions_standard_20260811/condition_level_stability_table.csv`, `validation/runs/v12_band_corrected_robustness_10base_signal_conditions_standard_20260811/artifact_effect_table.csv`, `validation/processed_data/physionet_nsr_rr_10min/nsr001_segment_000.csv`
- New output files: `validation/figures/Figure6_synthetic_robustness_band_corrected.png`, `validation/figures/Figure6_synthetic_robustness_band_corrected.pdf`, `validation/figures/Figure6_synthetic_robustness_band_corrected.svg`
- Panels regenerated: A-D
- Old headline values: Finite after 60/60; warnings after 60/60; Panel D used RMSSD, LF, HF, LF/HF from v09b.
- New headline values: Finite after correction 60/60; warning-visible 60/60.
- Layout changed: No.
- Logic changes: Inputs changed to corrected v12 synthetic robustness outputs; representative trace reconstructed with the same validation corruption/preprocessing logic.
- Expected-value discrepancy: None.

### Figure 7

- Original script: `validation/research_notes/paper_figures/generate_figure7_duration_sensitivity.py` was used as the visual template; corrected generation is in `generate_band_corrected_figures_3_5_6_7_8.py`.
- New input files: `validation/runs/v12_band_corrected_duration_sensitivity_standard_20260811/duration_metric_table.csv`, `validation/runs/v12_band_corrected_duration_sensitivity_standard_20260811/duration_results.csv`, `validation/runs/v12_band_corrected_duration_sensitivity_standard_20260811/frequency_metric_summary.csv`
- New output files: `validation/figures/Figure7_duration_sensitivity_band_corrected.png`, `validation/figures/Figure7_duration_sensitivity_band_corrected.pdf`, `validation/figures/Figure7_duration_sensitivity_band_corrected.svg`
- Panels regenerated: A-C
- Old headline values: Durations 30 s, 60 s, 2 min, 5 min, 10 min; 5-min VLF 12.3308%; 10-min VLF 12.9595%.
- New headline values: 5-min VLF 33.71%, LF/HF 11.98%; 10-min VLF 38.75%, LF/HF 12.57%.
- Layout changed: No panel arrangement change; duration axis now includes 3 min and 30-s VLF is explicitly marked NA.
- Logic changes: Removed monotonic VLF implication; shaded <5 min region now spans 30 s through 3 min.
- Expected-value discrepancy: None.

### Figure 8

- Original script: `validation/research_notes/paper_figures/generate_band_corrected_figures_3_5_6_7_8.py`; no retained legacy Figure 8 generator was found.
- New input files: `validation/runs/v12_band_corrected_mitbih_arrhythmia_robustness_standard_20260811_complete/mitbih_robustness_results.csv`, `validation/runs/v12_band_corrected_mitbih_arrhythmia_robustness_standard_20260811_complete/mitbih_segment_manifest.csv`, local MIT-BIH annotations under `validation/raw_data/mit_bih_arrhythmia/mit-bih-arrhythmia-database-1.0.0`
- New output files: `validation/figures/Figure8_mitbih_qc_band_corrected.png`, `validation/figures/Figure8_mitbih_qc_band_corrected.pdf`, `validation/figures/Figure8_mitbih_qc_band_corrected.svg`
- Panels regenerated: A-D
- Old headline values: Finite after 12/12; warnings after 12/12; correction applied 9/12.
- New headline values: Finite after correction 12/12; warning-visible 12/12; correction applied 9/12.
- Layout changed: No.
- Logic changes: Inputs changed to the completed corrected v12 MIT-BIH run; Panel D preserves the original manuscript-facing warning taxonomy.
- Expected-value discrepancy: None.

## Quality Control

- All numerical annotations in regenerated figures are computed from v12 corrected CSVs or reconstructed from the same v12 source rows.
- Legacy figure files in `validation/research_notes/paper_figures` were not overwritten.
- New figures were written to `validation/figures` as PNG, PDF, and SVG.
- Figure 7 includes the new 3-minute duration and treats 30-second VLF as unavailable rather than zero.
- Figure 8 preserves the manuscript-facing warning taxonomy and does not add ULF few-bin diagnostics as a new Panel D bar.
