# Kubios Manual-Review Shortlist

Inputs reviewed:
- `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_valid_only.csv`
- `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_excluded_files.csv`
- `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/kubios_comparison_qc.md`
- Existing v07 diagnostics/run metadata only; validation was not regenerated.

This shortlist is ranked by severity and expected usefulness for manual inspection. It intentionally separates likely data-quality exclusions from files that may reveal a real method disagreement.

## Recommended Scripts

- `tools/inspect_freq_outlier.py`: first-pass per-file inspection. Use it to generate native/NeuroKit2 PSD overlays, tachogram/resampled-signal plots, band tables, and diagnostics.
- `tools/focused_vlf_total_power_decomposition.py`: second-pass inspection for VLF/total-power cases. Use it after narrowing the list to compare cumulative power and VLF bin contributions across pipelines.
- `tools/analyze_validation_run.py`: run-level summary only. It is useful for context but is not the primary file-level inspection tool.

Use the same validation settings when manually generating plots:

```powershell
--run-name v07_kubios_subset_50_none_120_75_after_arm_a --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

## Priority 1: Must Inspect

### 1. OUT005

- Category: `remaining_outliers`
- Why ranked here: excluded from valid-only results because native powers exceed 100x Kubios; NeuroKit2 powers are even more extreme.
- Worst metrics:
  - LF: HRV Studio 491377.94%, NeuroKit2 8429172188.59%
  - total_power: HRV Studio 217331.28%, NeuroKit2 2461524894.60%
  - HF: HRV Studio 82419.01%, NeuroKit2 112577138.53%
  - VLF: HRV Studio 71890.60%, NeuroKit2 942412124.74%
- `adjusted_noverlap_for_short_signal`: yes, effective `nperseg=215`, `noverlap=214`, `n_samples=430`.
- Invalid RR cleanup: yes, 9 invalid RR intervals removed; duplicate time points detected.
- Kubios sample limit/range: `0-451`, data length `00:10:01`; suspicious because the effective validation signal was short/adjusted and the Kubios range does not obviously explain the huge PSD scale.
- Likely failure mode hypothesis: invalid RR data plus short-duration Welch issue, with possible interpolation mismatch. Treat as the most important pathological PSD case.
- Command:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT005__nsr021_segment_071.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\OUT005" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

### 2. VLF005

- Category: `vlf_sensitive`
- Why ranked here: excluded from valid-only results because HRV Studio/NeuroKit2 powers are zero or non-finite while Kubios has valid nonzero powers.
- Worst metrics:
  - VLF/LF/HF/LF_nu/HF_nu: HRV Studio 100.00%, NeuroKit2 100.00% for most zero-power metrics
  - LF/HF: non-finite for both HRV Studio and NeuroKit2
  - total_power: HRV Studio 100.00%, NeuroKit2 83.92%
- `adjusted_noverlap_for_short_signal`: yes, effective `nperseg=8`, `noverlap=7`, `n_samples=17`.
- Invalid RR cleanup: no invalid RR removals reported; duplicate time points not reported.
- Kubios sample limit/range: `0-601`, data length `00:10:01`; not suspicious by itself, but the effective validation signal is only 17 resampled samples.
- Likely failure mode hypothesis: short-duration Welch issue. This is likely a data/window-length exclusion unless the raw tachogram shows that the validator should have received a much longer signal.
- Command:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\vlf_sensitive\VLF005__nsr008_segment_046.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\VLF005" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

### 3. OUT006

- Category: `remaining_outliers`
- Why ranked here: valid-only file with very large absolute-power errors and heavy invalid-RR cleanup.
- Worst metrics:
  - LF: HRV Studio 2501.86%, NeuroKit2 2315.90%
  - HF: HRV Studio 1902.34%, NeuroKit2 1704.45%
  - total_power: HRV Studio 1320.04%, NeuroKit2 1485.13%
  - VLF: HRV Studio 1185.65%, NeuroKit2 1366.93%
- `adjusted_noverlap_for_short_signal`: yes, effective `nperseg=249`, `noverlap=248`, `n_samples=498`.
- Invalid RR cleanup: yes, 497 invalid RR intervals removed; duplicate time points detected.
- Kubios sample limit/range: `0-451`, data length `00:10:01`; suspicious in combination with extensive invalid cleanup and adjusted Welch parameters.
- Likely failure mode hypothesis: invalid RR data plus short-duration Welch issue. Useful for checking whether cleanup leaves a physiologically meaningful series.
- Command:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT006__nsr015_segment_089.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\OUT006" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

### 4. VLF002

- Category: `vlf_sensitive`
- Why ranked here: valid-only file, but HRV Studio and NeuroKit2 both return zero VLF while Kubios reports substantial VLF.
- Worst metrics:
  - HF_nu: HRV Studio 436.24%, NeuroKit2 364.20%
  - HF: HRV Studio 98.60%, NeuroKit2 379.94%
  - VLF: HRV Studio 100.00%, NeuroKit2 100.00%
  - LF: HRV Studio 100.00%, NeuroKit2 82.88%
- `adjusted_noverlap_for_short_signal`: yes, effective `nperseg=45`, `noverlap=44`, `n_samples=91`.
- Invalid RR cleanup: none reported; duplicate time points not reported.
- Kubios sample limit/range: `0-601`, data length `00:10:01`; not suspicious by itself, but validation only had 91 resampled samples.
- Likely failure mode hypothesis: short-duration Welch issue, possibly aggravated by VLF bin coverage. Inspect whether the tachogram duration is truly too short for 0.00-0.04 Hz power.
- Command:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\vlf_sensitive\VLF002__nsr007_segment_062.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\VLF002" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

### 5. RC003

- Category: `random_controls`
- Why ranked here: valid-only file with zero/near-zero native powers and severe normalized-power distortion.
- Worst metrics:
  - HF_nu: HRV Studio 1602.97%, NeuroKit2 1415.54%
  - HF: HRV Studio 95.36%, NeuroKit2 667.51%
  - VLF: HRV Studio 100.00%, NeuroKit2 97.83%
  - LF: HRV Studio 100.00%, NeuroKit2 94.07%
- `adjusted_noverlap_for_short_signal`: yes, effective `nperseg=51`, `noverlap=50`, `n_samples=102`.
- Invalid RR cleanup: none reported; duplicate time points not reported.
- Kubios sample limit/range: `0-451`, data length `00:10:01`; suspicious because validation used a very short resampled signal.
- Likely failure mode hypothesis: short-duration Welch issue with unstable normalized powers. Likely exclude if visual inspection confirms too few cycles/bins.
- Command:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\random_controls\RC003__nsr002_segment_069.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\RC003" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

## Priority 2: Inspect If Needed

### 6. OUT001

- Category: `remaining_outliers`
- Why ranked here: large VLF-only disagreement after the DC/no-detrend fix; no invalid cleanup or adjusted Welch event.
- Worst metrics:
  - VLF: HRV Studio 1416.42%, NeuroKit2 1742.39%
  - total_power: HRV Studio 289.10%, NeuroKit2 359.55%
  - LF: HRV Studio 43.39%, NeuroKit2 42.99%
- `adjusted_noverlap_for_short_signal`: no.
- Invalid RR cleanup: none reported; duplicate time points not reported.
- Kubios sample limit/range: `0-62`, data length `00:10:01`; highly suspicious and should be checked in Kubios against the actual selected sample range.
- Likely failure mode hypothesis: Kubios export/sample-range issue or genuine VLF algorithm disagreement. Useful because the non-VLF bands are much closer.
- Command:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT001__nsr009_segment_012.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\OUT001" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

### 7. CH001

- Category: `clean_high_agreement`
- Why ranked here: nominally clean file, but Kubios is far lower than both HRV Studio and NeuroKit2 for VLF/total power.
- Worst metrics:
  - VLF: HRV Studio 1170.22%, NeuroKit2 1342.75%
  - total_power: HRV Studio 316.80%, NeuroKit2 339.44%
  - HF: HRV Studio 129.26%, NeuroKit2 130.30%
- `adjusted_noverlap_for_short_signal`: no.
- Invalid RR cleanup: none reported; duplicate time points not reported.
- Kubios sample limit/range: `0-62`, data length `00:10:01`; highly suspicious, especially because this was selected as clean/high-agreement before Kubios comparison.
- Likely failure mode hypothesis: Kubios export/sample-range issue or sample-limit mismatch. Inspect alongside OUT001.
- Command:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\clean_high_agreement\CH001__nsr020_segment_000.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\CH001" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

## Priority 3: Likely Exclude Unless Recovering Missing Validation Rows

### 8. CH004

- Category: `clean_high_agreement`
- Why ranked here: Kubios parsed successfully, but validation produced no rows.
- Worst metrics:
  - Kubios-only values exist: VLF 1700.72, LF 500.99, HF 147.50, total_power 2350.10.
  - HRV Studio/NeuroKit2 values are missing because validation failed.
- `adjusted_noverlap_for_short_signal`: no event recorded.
- Invalid RR cleanup: unknown; validation did not reach diagnostics.
- Kubios sample limit/range: `0-451`, data length `00:10:01`; mildly suspicious, but the immediate issue is validation input loading.
- Likely failure mode hypothesis: invalid RR data or validation loader/input-format issue. Likely exclude from agreement statistics unless the validation input failure is resolved.
- Command, only if investigating missing-row recovery:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\clean_high_agreement\CH004__nsr006_segment_000.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\CH004" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

### 9. CH012

- Category: `clean_high_agreement`
- Why ranked here: same pattern as CH004: Kubios parsed successfully, validation produced no rows.
- Worst metrics:
  - Kubios-only values exist: VLF 767.40, LF 584.87, HF 51.09, total_power 1403.74.
  - HRV Studio/NeuroKit2 values are missing because validation failed.
- `adjusted_noverlap_for_short_signal`: no event recorded.
- Invalid RR cleanup: unknown; validation did not reach diagnostics.
- Kubios sample limit/range: `0-451`, data length `00:10:01`; mildly suspicious, but secondary to validation input failure.
- Likely failure mode hypothesis: invalid RR data or validation loader/input-format issue. Likely exclude unless missing-row recovery becomes important.
- Command, only if investigating missing-row recovery:

```powershell
python -B tools\inspect_freq_outlier.py "validation\kubios_subset\input_ascii_rr\clean_high_agreement\CH012__nsr028_segment_025.txt" --run-name v07_kubios_subset_50_none_120_75_after_arm_a --output-dir "validation\research_notes\manual_review_plots\CH012" --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics
```

## Batch Follow-Up Command

After running the per-file inspection for the Priority 1 files, run a focused VLF/total-power decomposition on the same set:

```powershell
python -B tools\focused_vlf_total_power_decomposition.py --run-name v07_kubios_subset_50_none_120_75_after_arm_a --detrend-method none --segment-length 120 --overlap-ratio 0.75 --enable-diagnostics "validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT005__nsr021_segment_071.txt" "validation\kubios_subset\input_ascii_rr\vlf_sensitive\VLF005__nsr008_segment_046.txt" "validation\kubios_subset\input_ascii_rr\remaining_outliers\OUT006__nsr015_segment_089.txt" "validation\kubios_subset\input_ascii_rr\vlf_sensitive\VLF002__nsr007_segment_062.txt" "validation\kubios_subset\input_ascii_rr\random_controls\RC003__nsr002_segment_069.txt"
```

## What To Look For

- Tachogram/raw RR: abrupt zeros, negative or non-finite intervals, isolated huge intervals, long flat regions, duplicated cumulative time points, or a much shorter effective duration than Kubios reports.
- Resampled signal: whether interpolation creates edge ringing, sharp excursions, or a near-constant signal that makes normalized powers unstable.
- Native and NeuroKit2 PSD overlays: whether the disagreement is broad-band, VLF-only, or caused by one near-zero-frequency bin.
- Zoomed VLF plot: whether the first non-DC bin dominates 0.00-0.04 Hz, and whether Kubios appears to be integrating a different frequency grid or a different selected range.
- Cumulative power: where the HRV Studio/NeuroKit2 curve separates from Kubios-scale expectations; early separation suggests VLF/sample-range issues, while broad separation suggests interpolation or algorithm disagreement.
- Welch metadata: effective `nperseg`, `noverlap`, number of segments, and resampled sample count. Files with fewer than one or two stable 120 s windows should be treated as questionable for VLF.

## Classification Summary

| Priority | Files | Intended decision |
| --- | --- | --- |
| Priority 1 | OUT005, VLF005, OUT006, VLF002, RC003 | Must inspect before using or excluding as representative validation evidence. |
| Priority 2 | OUT001, CH001 | Inspect if the goal is to understand VLF/sample-range disagreement in otherwise usable files. |
| Priority 3 | CH004, CH012 | Likely exclude from comparison unless the missing validation rows are specifically being recovered. |

