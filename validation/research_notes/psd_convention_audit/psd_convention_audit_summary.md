# PSD Convention Audit Summary

This validation-only audit recomputed diagnostic Welch PSD variants for the remaining worst Kubios agreement outliers. It does not establish Kubios as ground truth and does not change production behavior.

## Inputs

- Cleaned comparison CSV used: `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`
- Cleaned CSV source: `fallback_manual_review_sensitivity_analysis`
- Target files: OUT006, OUT003, OUT001, CH001, CH005
- Settings: no detrending convention with one global mean removed, 4 Hz interpolation, 120 s segment length, 75% overlap, Hann window, VLF 0-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz.

## Best Variant By File

| Subset | Signal source | Best variant | Native baseline mean err % | Best mean err % | Improvement % |
| --- | --- | --- | --- | --- | --- |
| OUT006 | neurokit2_resampled | hann_coherent_multiply | 994.14 | 173.17 | 82.58 |
| OUT003 | native_resampled | hann_coherent_multiply | 363.07 | 65.02 | 82.09 |
| OUT001 | native_resampled | hann_coherent_multiply | 258.32 | 68.05 | 73.66 |
| CH001 | neurokit2_resampled | hann_coherent_multiply | 249.20 | 47.08 | 81.11 |
| CH005 | native_resampled | hann_coherent_multiply | 228.68 | 41.25 | 81.96 |

## Best Aggregate Variants

| Signal source | Variant | Mean file-level err % | Files |
| --- | --- | --- | --- |
| neurokit2_resampled | hann_coherent_multiply | 78.92 | 5 |
| native_resampled | hann_coherent_multiply | 86.58 | 5 |
| neurokit2_resampled | hann_energy_multiply | 127.49 | 5 |
| native_resampled | hann_energy_multiply | 138.50 | 5 |
| neurokit2_resampled | one_sided_half_positive | 205.09 | 5 |
| native_resampled | one_sided_half_positive | 210.28 | 5 |
| neurokit2_resampled | exclude_dc | 281.47 | 5 |
| neurokit2_resampled | vlf_low_0_0033 | 315.64 | 5 |
| native_resampled | exclude_dc | 325.15 | 5 |
| native_resampled | vlf_low_0_0033 | 351.70 | 5 |

## Best Variants By Metric

| Metric | Rank | Signal source | Variant | Mean err % | Median err % |
| --- | --- | --- | --- | --- | --- |
| VLF | 1 | neurokit2_resampled | hann_coherent_multiply | 153.20 | 217.64 |
| VLF | 2 | native_resampled | hann_coherent_multiply | 153.63 | 217.65 |
| VLF | 3 | neurokit2_resampled | hann_energy_multiply | 259.40 | 376.47 |
| LF | 1 | neurokit2_resampled | hann_coherent_multiply | 110.16 | 47.50 |
| LF | 2 | native_resampled | hann_coherent_multiply | 143.76 | 47.50 |
| LF | 3 | neurokit2_resampled | hann_energy_multiply | 175.24 | 78.73 |
| HF | 1 | neurokit2_resampled | hann_coherent_multiply | 159.02 | 108.93 |
| HF | 2 | native_resampled | hann_coherent_multiply | 172.08 | 108.93 |
| HF | 3 | neurokit2_resampled | hann_energy_multiply | 248.53 | 213.40 |
| total_power | 1 | native_resampled | hann_coherent_multiply | 61.52 | 9.66 |
| total_power | 2 | neurokit2_resampled | hann_coherent_multiply | 61.78 | 9.66 |
| total_power | 3 | native_resampled | hann_energy_multiply | 140.62 | 64.49 |
| LF/HF | 1 | neurokit2_resampled | simpson | 26.04 | 11.57 |
| LF/HF | 2 | neurokit2_resampled | hann_energy_divide | 28.09 | 11.85 |
| LF/HF | 3 | neurokit2_resampled | hann_energy_multiply | 28.09 | 11.85 |
| LF_nu | 1 | neurokit2_resampled | simpson | 13.45 | 1.78 |
| LF_nu | 2 | neurokit2_resampled | hann_energy_divide | 14.08 | 3.77 |
| LF_nu | 3 | neurokit2_resampled | baseline_trapz_closed_dc | 14.08 | 3.77 |
| HF_nu | 1 | neurokit2_resampled | simpson | 25.01 | 11.06 |
| HF_nu | 2 | neurokit2_resampled | hann_energy_multiply | 26.08 | 11.36 |
| HF_nu | 3 | neurokit2_resampled | hann_energy_divide | 26.08 | 11.36 |

## Interpretation

- Native baseline mean file-level error across the five files was 418.68%; the per-file best diagnostic variants averaged 78.92%. This shows some convention sensitivity, but the best setting is selected post hoc per file.
- Best variants repeated by file: native_resampled / hann_coherent_multiply: 3, neurokit2_resampled / hann_coherent_multiply: 2.
- No single PSD convention should be treated as a production fix unless it improves most files and most metrics without relying on file-specific tuning.
- If the best aggregate variants are mostly nfft/grid or boundary variants, the residual mismatch is more likely a convention/documentation issue. If window or one-sided scaling variants dominate, that would suggest a scale convention, but those variants are diagnostic only.
- Mismatches should be read cautiously: OUT006 remains a retained pathological edge case, while CH001/CH005/OUT001/OUT003 may still reflect Kubios selection, interpolation, or band-grid differences.
- Recommended next step is documentation or a validation-only Kubios-compatible mode rather than changing production code, unless the same convention wins consistently on a larger curated set.

## Per-File Plots

- `OUT006`:
  - `validation/research_notes/psd_convention_audit/OUT006/baseline_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/OUT006/best_variant_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/OUT006/cumulative_power_by_frequency.png`
  - `validation/research_notes/psd_convention_audit/OUT006/band_integration_comparison.png`
- `OUT003`:
  - `validation/research_notes/psd_convention_audit/OUT003/baseline_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/OUT003/best_variant_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/OUT003/cumulative_power_by_frequency.png`
  - `validation/research_notes/psd_convention_audit/OUT003/band_integration_comparison.png`
- `OUT001`:
  - `validation/research_notes/psd_convention_audit/OUT001/baseline_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/OUT001/best_variant_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/OUT001/cumulative_power_by_frequency.png`
  - `validation/research_notes/psd_convention_audit/OUT001/band_integration_comparison.png`
- `CH001`:
  - `validation/research_notes/psd_convention_audit/CH001/baseline_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/CH001/best_variant_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/CH001/cumulative_power_by_frequency.png`
  - `validation/research_notes/psd_convention_audit/CH001/band_integration_comparison.png`
- `CH005`:
  - `validation/research_notes/psd_convention_audit/CH005/baseline_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/CH005/best_variant_psd_overlay.png`
  - `validation/research_notes/psd_convention_audit/CH005/cumulative_power_by_frequency.png`
  - `validation/research_notes/psd_convention_audit/CH005/band_integration_comparison.png`
