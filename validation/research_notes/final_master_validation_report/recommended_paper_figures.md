# Recommended Paper Figures

Use a small main-figure set that tells the validation story without repeating every diagnostic artifact. The strongest manuscript structure is: large-scale agreement, Kubios benchmark, duration limits, QC robustness, and spectral-method caveats.

## Keep in Main Paper

1. NeuroKit2 ten-minute agreement summary
   - Source: `validation/runs/v05_physionet_10min_segment_linear/diagnostic_summary.md`
   - Recommended form: compact metric-level bar/forest plot of median relative error and p90 relative error.
   - Reason: strongest large-scale agreement evidence; shows low median errors for LF, HF, total_power, LF/HF, LF_nu, and HF_nu while preserving the VLF caveat.

2. Kubios final 44-file agreement figure
   - Source: `validation/research_notes/final_validation_results_package/final_kubios_metric_table.csv`
   - Candidate existing figures: `scatter_LFHF.png`, `scatter_LF.png`, `scatter_HF.png`, `bland_altman_LF.png`, `bland_altman_HF.png`, `bland_altman_total_power.png`, `relative_error_boxplot_by_metric.png`
   - Recommended form: one metric-level relative-error panel plus one LF/HF or normalized-unit agreement panel.
   - Reason: directly addresses the external commercial benchmark while avoiding overclaiming equivalence.

3. Duration sensitivity figure
   - Source: `validation/runs/v10_duration_sensitivity_validation/duration_vs_error_plot.png` and `duration_vs_correlation_plot.png`
   - Recommended form: keep the error plot as main; use correlation plot as supplement if space is limited.
   - Reason: supports the practical software warning recommendation that primary frequency-domain reporting should use 5 minutes or longer.

4. Synthetic robustness correction plot
   - Source: `validation/runs/v09b_robustness_10base_signal_conditions/before_after_metric_change_boxplots.png`
   - Recommended form: main paper or compact supplement depending on figure count.
   - Reason: shows finite output, correction effects, and QC transparency under controlled corruptions.

5. FFT/AR comparison
   - Source: `validation/runs/v08_fft_ar_validation/psd_plots/*_psd_comparison.png` and `fft_ar_summary.md`
   - Recommended form: one representative PSD comparison plus a small table of LF/HF correlations and PSD caveats.
   - Reason: justifies Welch as the primary frequency-domain method and frames FFT/AR as secondary.

## Optional Supplement

- Five-minute NeuroKit2 diagnostic summary focused on VLF and absolute-power mismatch.
- Additional Kubios Bland-Altman figures for LF, HF, and total_power.
- `before_after_qc_error_distribution.png` from the final validation package.
- MIT-BIH tachogram examples from `validation/runs/v11_mitbih_arrhythmia_robustness/tachograms/`.
- Synthetic tachogram examples from `validation/runs/v09b_robustness_10base_signal_conditions/tachograms/`.
- AR order-sensitivity table from `validation/runs/v08_fft_ar_validation/ar_order_sensitivity.csv`.
- VLF instability table from `validation/runs/v10_duration_sensitivity_validation/vlf_instability_table.csv`.

## Remove or Avoid From Main Paper

- Redundant scatter plots for every metric.
- All-matched Kubios mean tables as headline evidence.
- Any figure that combines arrhythmia stress-test recordings with normal-rhythm primary agreement results.
- Large raw warning-summary tables.
- The initial two-base robustness figure if the ten-base study is shown.
- Diagnostic-only exploratory Syl_Vain plots unless needed to explain historical method development.

## Suggested Main Figure Set

Figure 1: HRV Studio validation workflow and QC pipeline.

Figure 2: Ten-minute NeuroKit2 metric-level agreement.

Figure 3: Final Kubios benchmark agreement after QC/manual review.

Figure 4: Duration sensitivity and recommended frequency-domain duration threshold.

Figure 5: Synthetic robustness before/after correction behavior.

Figure 6 or supplement: FFT/AR method comparison and Welch-primary rationale.

Supplementary Figure: MIT-BIH arrhythmia tachogram examples, clearly labeled as robustness/QC stress tests.

