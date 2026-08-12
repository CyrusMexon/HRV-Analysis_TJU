# Manual-Review Sensitivity Analysis

This validation-only analysis recomputes Kubios agreement after the manual-review exclusions `OUT005`, `VLF005`, `VLF002`, and `RC003`. `OUT006` is intentionally retained as a pathological edge case.

## File Counts

- Files before cleaning: 46
- Requested exclusions: OUT005, VLF005, VLF002, RC003
- Requested exclusions present in input CSV: 2
- Files after cleaning: 44
- `OUT006` retained after cleaning: yes

| Subset | Category | Input status | After status | Notes |
| --- | --- | --- | --- | --- |
| OUT005 | remaining_outliers | already_absent | not_present | Not present in comparison_valid_only.csv before this step. |
| VLF005 | vlf_sensitive | already_absent | not_present | Not present in comparison_valid_only.csv before this step. |
| VLF002 | vlf_sensitive | present | removed | Removed by manual-review sensitivity analysis. |
| RC003 | random_controls | present | removed | Removed by manual-review sensitivity analysis. |

## Overall Agreement

- HRV Studio overall mean relative error: 95.02% before, 88.85% after (6.50% improvement).
- HRV Studio overall median relative error: 18.55% before, 16.68% after (10.10% improvement).
- NeuroKit2 overall mean relative error: 103.47% before, 96.10% after (7.12% improvement).
- NeuroKit2 overall median relative error: 21.29% before, 18.97% after (10.89% improvement).

## HRV Studio Before vs After

| Metric | Before mean % | After mean % | Mean change | Mean improvement % | Before median % | After median % | Median improvement % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VLF | 135.51 | 137.12 | 1.61 | -1.19 | 34.63 | 31.92 | 7.83 |
| LF | 132.66 | 134.14 | 1.48 | -1.12 | 22.41 | 20.42 | 8.88 |
| HF | 169.16 | 172.44 | 3.28 | -1.94 | 30.32 | 28.72 | 5.30 |
| total_power | 106.69 | 107.00 | 0.31 | -0.29 | 21.62 | 19.91 | 7.92 |
| LF/HF | 31.33 | 28.21 | -3.12 | 9.96 | 20.16 | 17.56 | 12.87 |
| LF_nu | 16.18 | 12.37 | -3.81 | 23.55 | 8.52 | 7.45 | 12.56 |
| HF_nu | 73.64 | 30.64 | -43.00 | 58.39 | 11.35 | 11.22 | 1.16 |
| OVERALL | 95.02 | 88.85 | -6.18 | 6.50 | 18.55 | 16.68 | 10.10 |

## NeuroKit2 Comparator Before vs After

| Metric | Before mean % | After mean % | Mean improvement % | Before median % | After median % |
| --- | --- | --- | --- | --- | --- |
| VLF | 163.43 | 166.37 | -1.79 | 31.25 | 27.47 |
| LF | 130.52 | 132.43 | -1.46 | 22.42 | 22.14 |
| HF | 190.56 | 175.42 | 7.95 | 32.86 | 31.97 |
| total_power | 122.18 | 124.36 | -1.78 | 25.96 | 25.61 |
| LF/HF | 32.30 | 29.33 | 9.22 | 23.31 | 21.14 |
| LF_nu | 16.27 | 13.10 | 19.45 | 8.87 | 8.30 |
| HF_nu | 69.04 | 31.73 | 54.04 | 12.93 | 11.67 |
| OVERALL | 103.47 | 96.10 | 7.12 | 21.29 | 18.97 |

## Worst Remaining Outliers

| Rank | Subset | Category | HRV mean % | HRV median % | Worst HRV metrics |
| --- | --- | --- | --- | --- | --- |
| 1 | OUT006 | remaining_outliers | 995.00 | 1185.65 | LF=2501.86%; HF=1902.34%; total_power=1320.04% |
| 2 | OUT003 | remaining_outliers | 362.90 | 297.49 | HF=1197.04%; LF=445.57%; total_power=442.07% |
| 3 | OUT001 | remaining_outliers | 258.50 | 35.91 | VLF=1416.42%; total_power=289.10%; LF=43.39% |
| 4 | CH001 | clean_high_agreement | 249.10 | 109.97 | VLF=1170.22%; total_power=316.80%; HF=129.26% |
| 5 | CH005 | clean_high_agreement | 228.65 | 103.75 | HF=735.57%; total_power=338.60%; LF=302.08% |
| 6 | RC001 | random_controls | 217.63 | 139.25 | LF=513.41%; HF=478.69%; total_power=380.27% |
| 7 | RC004 | random_controls | 215.01 | 41.80 | HF=634.33%; total_power=433.79%; LF=327.37% |
| 8 | OUT004 | remaining_outliers | 201.78 | 73.98 | LF=684.90%; HF=479.79%; total_power=106.73% |
| 9 | CH016 | clean_high_agreement | 124.24 | 38.20 | HF=450.83%; HF_nu=228.67%; LF/HF=80.62% |
| 10 | OUT010 | remaining_outliers | 122.47 | 142.39 | VLF=291.97%; total_power=223.20%; HF=174.91% |

## Interpretation For Paper Drafting

- The cleaned analysis improves the aggregate HRV Studio agreement metrics, which supports the cautious interpretation that part of the original disagreement was driven by pathological recordings or problematic input selections.
- Agreement does not become uniformly close after cleaning. Remaining mean errors are still influenced by retained edge cases, including `OUT006`, and by broader method differences in VLF, LF, HF, and total power.
- The median relative errors are more stable than the means and are the more defensible high-level summary for this manually reviewed subset.
- The remaining disagreement appears mixed: some residual differences are isolated/pathological, while the absolute-power bands still show systematic sensitivity to preprocessing, interpolation, and Welch/bin conventions.
- Additional manual review is still warranted before using the cleaned 50-file subset as a definitive agreement claim. The remaining-outlier shortlist identifies the next files to inspect.

## Distribution Plot

- `validation\research_notes\manual_review_sensitivity_analysis\relative_error_before_after_boxplot.png`
