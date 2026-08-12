# Priority 1 Kubios Manual Review Summary

Generated from per-file manual inspection artifacts under `validation/research_notes/manual_review_plots/` and existing v07 comparison diagnostics. Full validation was not rerun.

## Summary Table

| Subset | Effective duration | RR count | Resampled samples | Native Welch nperseg/noverlap | Adjusted overlap | Invalid RR removed | Duplicate times | Nonfinite PSD | Preliminary verdict |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| OUT005 | 107.25 s | 228 | 429 | 214/160 | yes | 9 | yes | native no, Welch no | exclude |
| VLF005 | 4.25 s | 5 | 17 | 8/6 | yes | 0 | no | native no, Welch no | re-export/check Kubios selection |
| OUT006 | 124.25 s | 150 | 497 | 480/360 | yes | 497 | yes | native no, Welch no | exclude |
| VLF002 | 22.50 s | 525 | 90 | 45/33 | yes | 0 | no | native no, Welch no | re-export/check Kubios selection |
| RC003 | 25.25 s | 571 | 101 | 50/37 | yes | 0 | no | native no, Welch no | re-export/check Kubios selection |

## Per-File Findings

## OUT005

- Source file path: `validation/kubios_subset/input_ascii_rr/remaining_outliers/OUT005__nsr021_segment_071.txt`
- Output folder: `validation/research_notes/manual_review_plots/OUT005/`
- Effective duration: 107.25 s
- RR count: 228
- Invalid RR removed count: 9
- Duplicate time points detected: yes
- Number of resampled samples: 429
- Effective Welch nperseg/noverlap: native diagnostics `214/160`; run event: effective nperseg/noverlap 215/214, n_samples 430
- Overlap adjusted: yes
- PSD contains NaN/non-finite values: native diagnostic `no`, Welch diagnostic `no`
- Resampling comparison: Pearson correlation 0.347, RMSE 211493.08 ms
- PSD bin coverage: native VLF/LF/HF bins 3/6/13; NeuroKit2 5/12/26
- Largest disagreement metrics vs Kubios:
  - LF: Kubios 140.347, HRV err 491377.94%, NeuroKit2 err 8429172188.59%
  - total_power: Kubios 675.602, HRV err 217331.28%, NeuroKit2 err 2461524894.60%
  - VLF: Kubios 339.030, HRV err 71890.60%, NeuroKit2 err 942412124.74%
  - HF: Kubios 195.747, HRV err 82419.01%, NeuroKit2 err 112577138.53%
- Apparent cause classification: a) caused by invalid RR cleanup; b) caused by too-short effective signal; c) interpolation/resampling mismatch likely contributes; d) low-bin placement also matters; f) not a clean algorithm-disagreement case.
- Recommended preliminary verdict: **exclude**
- Rationale: Invalid RR intervals and duplicate cumulative times are present, preprocessing correlation is poor, and NeuroKit2 PSD power explodes relative to both native and Kubios.

Minimum review artifacts:
- `validation/research_notes/manual_review_plots/OUT005/inspection_summary.md`
- `validation/research_notes/manual_review_plots/OUT005/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/OUT005/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/OUT005/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/OUT005/zoomed_vlf_psd.png`

## VLF005

- Source file path: `validation/kubios_subset/input_ascii_rr/vlf_sensitive/VLF005__nsr008_segment_046.txt`
- Output folder: `validation/research_notes/manual_review_plots/VLF005/`
- Effective duration: 4.25 s
- RR count: 5
- Invalid RR removed count: 0
- Duplicate time points detected: no
- Number of resampled samples: 17
- Effective Welch nperseg/noverlap: native diagnostics `8/6`; run event: effective nperseg/noverlap 8/7, n_samples 17
- Overlap adjusted: yes
- PSD contains NaN/non-finite values: native diagnostic `no`, Welch diagnostic `no`
- Resampling comparison: Pearson correlation 0.994, RMSE 8.44 ms
- PSD bin coverage: native VLF/LF/HF bins 1/0/0; NeuroKit2 1/0/1
- Largest disagreement metrics vs Kubios:
  - VLF: Kubios 1088.549, HRV err 100.00%, NeuroKit2 err 100.00%
  - LF: Kubios 309.884, HRV err 100.00%, NeuroKit2 err 100.00%
  - HF: Kubios 187.580, HRV err 100.00%, NeuroKit2 err 100.00%
  - total_power: Kubios 1587.055, HRV err 100.00%, NeuroKit2 err 83.92%
- Apparent cause classification: b) caused by too-short effective signal; d) VLF/DC/bin placement dominates because only 17 resampled samples and too few PSD bins are available; f) Kubios selection/input interpretation remains unclear.
- Recommended preliminary verdict: **re-export/check Kubios selection**
- Rationale: The inspection input contains only 5 RR intervals, while the Kubios export reports a 10-minute data length and valid powers.

Minimum review artifacts:
- `validation/research_notes/manual_review_plots/VLF005/inspection_summary.md`
- `validation/research_notes/manual_review_plots/VLF005/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/VLF005/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/VLF005/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/VLF005/zoomed_vlf_psd.png`

## OUT006

- Source file path: `validation/kubios_subset/input_ascii_rr/remaining_outliers/OUT006__nsr015_segment_089.txt`
- Output folder: `validation/research_notes/manual_review_plots/OUT006/`
- Effective duration: 124.25 s
- RR count: 150
- Invalid RR removed count: 497
- Duplicate time points detected: yes
- Number of resampled samples: 497
- Effective Welch nperseg/noverlap: native diagnostics `480/360`; run event: effective nperseg/noverlap 249/248, n_samples 498
- Overlap adjusted: yes
- PSD contains NaN/non-finite values: native diagnostic `no`, Welch diagnostic `no`
- Resampling comparison: Pearson correlation 1.000, RMSE 0.16 ms
- PSD bin coverage: native VLF/LF/HF bins 5/14/31; NeuroKit2 5/14/31
- Largest disagreement metrics vs Kubios:
  - LF: Kubios 48.599, HRV err 2501.86%, NeuroKit2 err 2315.90%
  - HF: Kubios 26.867, HRV err 1902.34%, NeuroKit2 err 1704.45%
  - total_power: Kubios 892.834, HRV err 1320.04%, NeuroKit2 err 1485.13%
  - VLF: Kubios 817.144, HRV err 1185.65%, NeuroKit2 err 1366.93%
- Apparent cause classification: a) caused by invalid RR cleanup; b) shortened/adjusted comparator signal is present; c) interpolation mismatch does not appear primary because native and NeuroKit2 resampled signals align closely; f) not a clean algorithm-disagreement case.
- Recommended preliminary verdict: **exclude**
- Rationale: 497 invalid intervals were removed and duplicate time points were detected; the remaining 150-interval signal is not representative of the original file.

Minimum review artifacts:
- `validation/research_notes/manual_review_plots/OUT006/inspection_summary.md`
- `validation/research_notes/manual_review_plots/OUT006/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/OUT006/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/OUT006/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/OUT006/zoomed_vlf_psd.png`

## VLF002

- Source file path: `validation/kubios_subset/input_ascii_rr/vlf_sensitive/VLF002__nsr007_segment_062.txt`
- Output folder: `validation/research_notes/manual_review_plots/VLF002/`
- Effective duration: 22.50 s
- RR count: 525
- Invalid RR removed count: 0
- Duplicate time points detected: no
- Number of resampled samples: 90
- Effective Welch nperseg/noverlap: native diagnostics `45/33`; run event: effective nperseg/noverlap 45/44, n_samples 91
- Overlap adjusted: yes
- PSD contains NaN/non-finite values: native diagnostic `no`, Welch diagnostic `no`
- Resampling comparison: Pearson correlation 0.369, RMSE 140.55 ms
- PSD bin coverage: native VLF/LF/HF bins 1/1/3; NeuroKit2 1/3/6
- Largest disagreement metrics vs Kubios:
  - HF_nu: Kubios 18.648, HRV err 436.24%, NeuroKit2 err 364.20%
  - HF: Kubios 63.976, HRV err 98.60%, NeuroKit2 err 379.94%
  - VLF: Kubios 838.318, HRV err 100.00%, NeuroKit2 err 100.00%
  - LF: Kubios 278.331, HRV err 100.00%, NeuroKit2 err 82.88%
- Apparent cause classification: b) caused by too-short effective signal; c) interpolation/resampling mismatch likely contributes; d) VLF/bin placement is unstable; f) input units or Kubios selection remain unclear.
- Recommended preliminary verdict: **re-export/check Kubios selection**
- Rationale: The inspected RR mean is 42.72 ms with 90 native resampled samples, which is physiologically suspicious and inconsistent with a stable 120 s Welch comparison.

Minimum review artifacts:
- `validation/research_notes/manual_review_plots/VLF002/inspection_summary.md`
- `validation/research_notes/manual_review_plots/VLF002/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/VLF002/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/VLF002/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/VLF002/zoomed_vlf_psd.png`

## RC003

- Source file path: `validation/kubios_subset/input_ascii_rr/random_controls/RC003__nsr002_segment_069.txt`
- Output folder: `validation/research_notes/manual_review_plots/RC003/`
- Effective duration: 25.25 s
- RR count: 571
- Invalid RR removed count: 0
- Duplicate time points detected: no
- Number of resampled samples: 101
- Effective Welch nperseg/noverlap: native diagnostics `50/37`; run event: effective nperseg/noverlap 51/50, n_samples 102
- Overlap adjusted: yes
- PSD contains NaN/non-finite values: native diagnostic `no`, Welch diagnostic `no`
- Resampling comparison: Pearson correlation 0.510, RMSE 144.03 ms
- PSD bin coverage: native VLF/LF/HF bins 1/1/4; NeuroKit2 2/3/7
- Largest disagreement metrics vs Kubios:
  - HF_nu: Kubios 5.872, HRV err 1602.97%, NeuroKit2 err 1415.54%
  - HF: Kubios 20.323, HRV err 95.36%, NeuroKit2 err 667.51%
  - VLF: Kubios 1044.252, HRV err 100.00%, NeuroKit2 err 97.83%
  - LF: Kubios 325.355, HRV err 100.00%, NeuroKit2 err 94.07%
- Apparent cause classification: b) caused by too-short effective signal; c) interpolation/resampling mismatch likely contributes; d) VLF/bin placement is unstable; f) input units or Kubios selection remain unclear.
- Recommended preliminary verdict: **re-export/check Kubios selection**
- Rationale: The inspected RR mean is 44.20 ms with 101 native resampled samples and poor native/NeuroKit2 preprocessing correlation.

Minimum review artifacts:
- `validation/research_notes/manual_review_plots/RC003/inspection_summary.md`
- `validation/research_notes/manual_review_plots/RC003/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/RC003/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/RC003/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/RC003/zoomed_vlf_psd.png`

## Minimum Files To Upload For Review

- `validation/research_notes/manual_review_plots/priority1_manual_review_summary.md`
- `validation/research_notes/manual_review_plots/OUT005/inspection_summary.md`
- `validation/research_notes/manual_review_plots/OUT005/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/OUT005/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/OUT005/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/OUT005/zoomed_vlf_psd.png`
- `validation/research_notes/manual_review_plots/VLF005/inspection_summary.md`
- `validation/research_notes/manual_review_plots/VLF005/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/VLF005/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/VLF005/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/VLF005/zoomed_vlf_psd.png`
- `validation/research_notes/manual_review_plots/OUT006/inspection_summary.md`
- `validation/research_notes/manual_review_plots/OUT006/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/OUT006/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/OUT006/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/OUT006/zoomed_vlf_psd.png`
- `validation/research_notes/manual_review_plots/VLF002/inspection_summary.md`
- `validation/research_notes/manual_review_plots/VLF002/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/VLF002/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/VLF002/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/VLF002/zoomed_vlf_psd.png`
- `validation/research_notes/manual_review_plots/RC003/inspection_summary.md`
- `validation/research_notes/manual_review_plots/RC003/preprocessing_summary.csv`
- `validation/research_notes/manual_review_plots/RC003/vlf_total_power_summary.md`
- `validation/research_notes/manual_review_plots/RC003/psd_native_vs_neurokit2.png`
- `validation/research_notes/manual_review_plots/RC003/zoomed_vlf_psd.png`

## Notes

- `vlf_total_power_summary.md` and `vlf_total_power_decomposition.csv` were generated by `tools/focused_vlf_total_power_decomposition.py` and copied into each requested per-file folder.
- The focused decomposition script also keeps its default run-level copies under `validation/runs/v07_kubios_subset_50_none_120_75_after_arm_a/manual_inspection/`.
- These verdicts are preliminary and intended to guide human review; they do not establish Kubios, HRV Studio, or NeuroKit2 as ground truth.
