# Full Cleaned Hann Coherent-Gain PSD Diagnostic Audit

This validation-only analysis applies the leading diagnostic variant from the focused PSD-convention audit, `hann_coherent_multiply`, across the full cleaned Kubios subset. It does not change production behavior and does not assert that Kubios is ground truth.

## Counts

- Files analyzed: 44
- Metric rows analyzed: 308
- Overcorrection rows: 83

## Overall Result

- Overall mean relative error: 88.85% before, 53.91% after (39.33% improvement).
- Overall median relative error: 16.68% before, 53.29% after (-219.50% improvement).

## By Metric

| Metric | Before mean % | After mean % | Mean improvement % | Before median % | After median % | Median improvement % | Overcorrection rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VLF | 137.12 | 82.76 | 39.65 | 31.92 | 78.30 | -145.34 | 17 |
| LF | 134.14 | 78.83 | 41.23 | 20.42 | 75.00 | -267.35 | 17 |
| HF | 172.44 | 76.26 | 55.77 | 28.72 | 72.78 | -153.46 | 28 |
| total_power | 107.00 | 68.30 | 36.17 | 19.91 | 72.64 | -264.85 | 21 |
| LF/HF | 28.21 | 28.19 | 0.04 | 17.56 | 17.56 | 0.04 | 0 |
| LF_nu | 12.37 | 12.36 | 0.07 | 7.45 | 7.47 | -0.34 | 0 |
| HF_nu | 30.64 | 30.64 | 0.03 | 11.22 | 11.31 | -0.78 | 0 |
| OVERALL | 88.85 | 53.91 | 39.33 | 16.68 | 53.29 | -219.50 | 83 |

## By Category

| Category | Files | Before mean % | After mean % | Mean improvement % | Before median % | After median % |
| --- | --- | --- | --- | --- | --- | --- |
| clean_high_agreement | 18 | 57.54 | 49.85 | 13.35 | 17.79 | 51.60 |
| random_controls | 4 | 120.78 | 41.00 | 66.06 | 26.85 | 37.87 |
| remaining_outliers | 9 | 240.82 | 72.76 | 69.79 | 47.18 | 44.97 |
| short_or_adjusted | 5 | 8.00 | 46.05 | -475.96 | 2.34 | 70.85 |
| vlf_sensitive | 8 | 22.89 | 53.19 | -132.42 | 10.40 | 71.14 |

## Remaining Outliers After Hann Scale

| Rank | Subset | Category | After mean % | Before mean % | Worst after metrics |
| --- | --- | --- | --- | --- | --- |
| 1 | OUT006 | remaining_outliers | 211.51 | 995.00 | LF=548.32%; HF=400.49%; total_power=255.38% |
| 2 | OUT007 | remaining_outliers | 90.15 | 82.13 | HF_nu=232.35%; LF=89.35%; LF/HF=82.55% |
| 3 | VLF009 | vlf_sensitive | 88.57 | 70.69 | HF_nu=169.01%; VLF=95.98%; LF=93.19% |
| 4 | CH016 | clean_high_agreement | 86.29 | 124.24 | HF_nu=228.63%; VLF=82.06%; LF/HF=80.61% |
| 5 | OUT001 | remaining_outliers | 68.05 | 258.50 | VLF=278.82%; LF=85.82%; HF=83.91% |
| 6 | OUT003 | remaining_outliers | 65.02 | 362.90 | HF=224.39%; HF_nu=72.82%; LF/HF=57.89% |
| 7 | CH017 | clean_high_agreement | 63.96 | 86.35 | VLF=98.69%; total_power=92.86%; LF=70.34% |
| 8 | CH008 | clean_high_agreement | 58.88 | 29.25 | VLF=82.41%; LF=80.61%; total_power=79.19% |
| 9 | VLF008 | vlf_sensitive | 58.88 | 32.54 | VLF=89.64%; LF=78.50%; total_power=76.28% |
| 10 | OUT008 | remaining_outliers | 58.64 | 50.27 | VLF=88.47%; LF=75.41%; total_power=70.58% |

## Interpretation

- Absolute power metrics improved broadly on mean error: VLF 39.65%, LF 41.23%, HF 55.77%, total_power 36.17%.
- Ratio/normalized metrics are mostly stable by construction or change less directly with a uniform PSD scale: LF/HF 0.04%, LF_nu 0.07%, HF_nu 0.03%.
- Category-level results should be interpreted cautiously because the category sizes are small and remaining edge cases can dominate means.
- The diagnostic scale still overcorrects some rows (83 metric rows), so it is not cleanly safe as a production-wide change.
- The current evidence supports treating `hann_coherent_multiply` as either an optional Kubios-compatible validation mode or a documentation finding, not as an immediate production default change.
- A production change would require confirming the same scaling convention against a larger curated set and checking whether it worsens agreement with non-Kubios references or established PSD variance behavior.

## Recommendation

- Recommended classification: **b) optional Kubios-compatible mode** for validation/reporting experiments, with accompanying documentation. A documentation-only note is also defensible if the project wants to avoid any alternate computation mode.
- Not recommended at this point: making this a production default.
