# Figure 5 Spectral Methods Source Notes

Generated on 2026-06-15 from existing validation artifacts in the HRV-Analysis_TJU project. No validation experiments were rerun, and no original validation outputs were modified.

## Output Files

- `validation/research_notes/paper_figures/figure5_spectral_methods.png`
- `validation/research_notes/paper_figures/figure5_spectral_methods.pdf`
- `validation/research_notes/paper_figures/figure5_spectral_methods.svg`

## Exact Input Files Used

- FFT/AR row-level comparison: `validation/runs/v08_fft_ar_validation/fft_ar_comparison.csv`
- FFT/AR validation summary: `validation/runs/v08_fft_ar_validation/fft_ar_summary.md`
- FFT/AR instability records: `validation/runs/v08_fft_ar_validation/instability_cases.csv`
- AR order sensitivity table: `validation/runs/v08_fft_ar_validation/ar_order_sensitivity.csv`
- Representative PSD plot directory reviewed but not reused: `validation/runs/v08_fft_ar_validation/psd_plots/`
- Master numbers table used as cross-check: `validation/research_notes/final_master_validation_report/master_validation_numbers_table.csv`
- Master validation report reviewed as contextual source: `validation/research_notes/final_master_validation_report/master_validation_report.md`

## Panel A: Welch Agreement With NeuroKit2

- Source rows selected: finite positive `welch_LF_HF` and `neurokit2_welch_LF_HF` pairs from `validation/runs/v08_fft_ar_validation/fft_ar_comparison.csv`.
- Rows/files plotted and analyzed: 100.
- Pearson correlation: 0.987007965.
- Median relative difference, HRV Studio Welch versus NeuroKit2 Welch LF/HF: 12.701778718%.
- Cross-check method correlations from the same table: FFT LF/HF versus Welch r = 0.852648972; AR LF/HF versus Welch r = 0.900869715.
- Panel regenerated from CSV; no existing plot was reused.

## Panel B: PSD-Area Consistency

| Method | n | Median PSD/variance | Mean | P25 | P75 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Welch | 100 | 0.828173 | 0.801561 | 0.709939 | 0.918238 | 0.221364 | 1.10334 |
| FFT | 100 | 96.7572 | 113.125 | 63.9362 | 140.502 | 21.8777 | 499.102 |
| AR | 100 | 1 | 1 | 1 | 1 | 1 | 1 |

- Welch ratio was computed as `welch_psd_full_area / fft_variance_ms2` because the table provides the Welch integrated PSD area and the same signal variance field used by FFT.
- FFT ratio used the provided `fft_psd_variance_ratio` field.
- AR ratio used the provided `ar_psd_variance_ratio` field.
- FFT PSD/variance warnings: 100/100 files.
- FFT instability flags: 100/100 files.
- AR instability flags: 2/100 files.
- AR PSD-positive files: 100/100; AR files with nonfinite PSD values: 0/100; AR fallbacks at order 16: 0/100.
- The log-scaled y-axis is used to show Welch/AR ratios near 1 together with FFT ratios that are much larger under the no-detrend/DC convention.
- Panel regenerated from CSV; no existing plot was reused.

## Panel C: AR Order Sensitivity

| Comparison | n | Median relative change (%) | P75 (%) | P90 (%) | Max (%) |
|---|---:|---:|---:|---:|---:|
| AR(8) vs AR(16) | 100 | 21.9561 | 41.7841 | 68.0251 | 240.937 |
| AR(24) vs AR(16) | 100 | 6.70045 | 11.3841 | 17.6232 | 38.5797 |
| AR order spread across 8/16/24 | 100 | 30.5246 | 49.3277 | 68.0251 | 258.528 |

- Source rows selected: AR(8), AR(16), and AR(24) LF/HF values from `validation/runs/v08_fft_ar_validation/ar_order_sensitivity.csv`.
- Files with complete AR(8/16/24) LF/HF values: 100.
- Order-sensitive files, using the validation flag `ar_order_sensitive`: 25/100.
- Threshold visualized: 50% LF/HF order spread.
- Numerical stability annotation comes from v08 summary fields: no AR fallbacks at order 16 and no nonfinite AR PSD values in the comparison table.
- Panel regenerated from CSV; no existing plot was reused.

## Filtering Applied

- Panel A required finite, positive LF/HF values for log-scaled scatter axes. No files were removed beyond finite/positive checks.
- Panel B removed non-finite PSD/variance ratios before plotting. All three plotted methods retained n = 100 values.
- Panel C required complete AR(8), AR(16), and AR(24) LF/HF values. All 100 files had complete values.

## Assumptions

- `welch_LF_HF` represents HRV Studio Welch LF/HF and `neurokit2_welch_LF_HF` represents NeuroKit2 Welch LF/HF.
- The FFT PSD/variance mismatch is interpreted as convention sensitivity under the no-detrend/DC handling described in `validation/runs/v08_fft_ar_validation/fft_ar_summary.md`, not as evidence that FFT is unusable.
- AR is interpreted as numerically stable in this run but method/order-sensitive; the figure avoids presenting AR as invalid.
- Figure contains no overall title and no embedded caption; manuscript caption should be supplied separately.
