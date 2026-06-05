# Exploratory Validation Findings: Syl_Vain RR Files

Internal research summary and decision log for later paper writing. This is not the final validation paper and does not establish that either HRV Studio or NeuroKit2 is the correct implementation.

Source runs:

- `validation/runs/v01_pre_diagnostics/`
- `validation/runs/v02_with_diagnostics/`
- `validation/runs/v02b_nfft_alignment_experiment/`

Primary artifact types used:

- `notes.md`
- `run_info.json`
- `diagnostic_summary.md`
- `nfft_alignment_comparison.csv`
- `manual_inspection/long_outlier_inspection_summary.md`
- `manual_inspection/*/inspection_summary.md`
- `manual_inspection/2023-12-17_10-14-05.txt/vlf_total_power_summary.md`

Note: `v01_pre_diagnostics` is CSV-only. It has no `notes.md`, `run_info.json`, or `diagnostic_summary.md`.

## 1. Objective of Exploratory Phase

The Syl_Vain RR files were used as a pre-PhysioNet exploratory validation set. They were already available in the project and provided practical RR interval examples for early software-to-software comparison between HRV Studio and NeuroKit2.

The goal was not to perform final external validation. The goal was to identify implementation-sensitive behavior before committing to a larger, more systematic validation design.

The main questions were:

1. Whether HRV Studio frequency-domain results were broadly comparable to NeuroKit2 under matched high-level settings.
2. Whether discrepancies were uniform across VLF, LF, HF, total power, LF/HF, LF_nu, and HF_nu.
3. Whether short recording duration and sparse frequency-bin placement explained the largest disagreements.
4. Whether NeuroKit2-style Welch `nfft=2*nperseg` explained the observed native-vs-NeuroKit2 mismatch.
5. Whether high-error cases represented one common failure mode or multiple distinct discrepancy classes.

The overall purpose was to decide what the next validation phase needed to control: recording duration, RR quality, interpolation, Welch grid density, band-edge definitions, and independent comparator software.

## 2. Experimental Timeline

| Stage | Run folder | Purpose | Key artifacts | Main contribution |
| --- | --- | --- | --- | --- |
| Initial pilot | `v01_pre_diagnostics` | Small HRV Studio vs NeuroKit2 check on selected Syl_Vain files | Four CSV files | Identified early metric-specific discrepancies and bin-count differences |
| Diagnostics expansion | `v02_with_diagnostics` | Full Syl_Vain comparison with diagnostic capture | `notes.md`, `run_info.json`, `diagnostic_summary.md`, `diagnostics.csv/json`, metric CSV | Quantified corpus-wide error patterns, duration effects, bin-count mismatch, and warning patterns |
| nfft and outlier review | `v02b_nfft_alignment_experiment` | Test whether `nfft=2*nperseg` explains mismatch; inspect long-duration outliers | `notes.md`, `run_info.json`, `diagnostic_summary.md`, `nfft_alignment_comparison.csv`, manual inspection summaries | Showed that nfft alignment greatly reduces bin-count mismatch but does not fully explain absolute-power disagreement |

### v01 Pilot

The pilot used three unique short recordings. `freq_validation.csv` duplicated `freq_validation_2018__04_30_13_20_04.csv`, so it should not be double counted.

| Recording | RR count | Native duration | NeuroKit2 duration |
| --- | ---: | ---: | ---: |
| `2018-04-30 13-20-04.txt` | 77 | 61.087 s | 61.25 s |
| `2018-04-30 21-03-36.txt` | 60 | 58.811 s | 59.00 s |
| `2020-09-20 16-49-51.txt` | 65 | 65.810 s | 66.00 s |

Across 21 unique metric comparisons, 12 rows exceeded 20% relative error, 4 exceeded 50%, and 1 exceeded 80%.

### v02 Diagnostics Expansion

`run_info.json` records:

| Field | Value |
| --- | --- |
| Run name | `v02_with_diagnostics` |
| Timestamp | `2026-06-01T08:46:02.574151+00:00` |
| Script | `tools\validate_freq_domain_neurokit2.py` |
| Git commit | `0abf2ade1da3fc132c43d3853787ff4c3c995fad` |
| Git state | Dirty worktree |
| Input files listed | 1,934 |
| Failures recorded | 43 |
| Experimental nfft multiplier | 1.0 |
| Software versions | Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, NeuroKit2 0.2.12 |

`notes.md` records the human-readable run ledger: purpose, validation type, code-state flags, input files, settings, outputs, auto-generated observations, and an empty researcher-interpretation placeholder.

### v02b nfft and Manual Review

`run_info.json` records the same script, commit, software versions, input count, and failure count as v02. The key planned difference was `experimental_native_welch_nfft_multiplier = 2.0`.

The v02b phase added:

- `nfft_alignment_comparison.csv`
- `long_duration_outliers.csv`
- `manual_inspection/long_outlier_inspection_summary.md`
- per-file `inspection_summary.md` reports
- focused `vlf_total_power_summary.md` for `2023-12-17 10-14-05.txt`

## 3. Main Findings

### Confirmed Findings

#### Short-duration limitations were substantial

The v02 diagnostic summary showed that short recordings were a major source of instability.

| Duration bucket | Rows | Mean relative error % | Median relative error % | P90 relative error % | Rows >80% |
| --- | ---: | ---: | ---: | ---: | ---: |
| <60 s | 5,103 | 166.34 | 97.06 | 184.54 | 2,825 |
| 60-120 s | 7,623 | 27.33 | 25.22 | 56.63 | 223 |
| 120-300 s | 112 | 35.85 | 3.67 | 97.85 | 16 |
| >=300 s | 399 | 32.22 | 9.03 | 82.52 | 58 |

This supports the decision that Syl_Vain recordings are not sufficient as the primary validation substrate. They are useful for exploratory stress testing, but many are too short for stable frequency-domain HRV, especially VLF.

#### VLF was unstable and often the weakest band

In the v01 pilot, VLF had the highest mean relative error among the three unique recordings:

| Metric | Mean relative error % | Range % |
| --- | ---: | ---: |
| VLF | 73.40 | 68.44-82.35 |
| LF | 31.15 | 20.12-37.70 |
| HF | 6.71 | 2.01-12.69 |
| total_power | 32.87 | 18.40-51.00 |
| LF/HF | 26.46 | 8.52-38.92 |
| LF_nu | 11.12 | 2.35-16.48 |
| HF_nu | 23.02 | 6.74-36.75 |

The later focused inspection of `2023-12-17 10-14-05.txt` confirmed a localized low-frequency discrepancy: LF, HF, and LF/HF were close, while VLF and total power differed strongly.

#### Duration was not a sufficient explanation

Duration explained many high-error short-recording cases, but not all important discrepancies. The v02b diagnostic summary reported 122 metric rows from long-duration recordings, exported to `long_duration_outliers.csv`, where duration was at least 300 s and relative error exceeded 50%. These represented 45 source files requiring manual review.

Manual examples:

| File | Duration | Finding |
| --- | ---: | --- |
| `2018-06-02 17-41-25.txt` | 373.53 s | High errors with artifact/interpolation mismatch pattern |
| `2024-07-15 16-41-40.txt` | 1,591.37 s | High errors with artifact/interpolation mismatch pattern |
| `2024-12-08 16-21-12.txt` | 1,843.08 s | Broad PSD shape alignment but amplitude/power disagreement |
| `2023-12-17 10-14-05.txt` | 2,797.15 s | LF/HF aligned; VLF and total power discrepant |

This means the next validation phase cannot rely on duration alone. It also needs quality control and method-specific diagnostics.

#### nfft mismatch was important but not dominant as a complete explanation

The v02b `nfft=2*nperseg` experiment nearly eliminated bin-count mismatch:

| Check | Before | After |
| --- | ---: | ---: |
| Total bin mismatch | 12,508 / 12,508 (100.00%) | 63 / 12,508 (0.50%) |
| Metric-band bin mismatch | 12,508 / 12,508 (100.00%) | 60 / 12,508 (0.48%) |

It also improved several median relative errors:

| Metric | Median before % | Median after % | Median change |
| --- | ---: | ---: | ---: |
| LF | 36.57 | 8.94 | -27.63 |
| HF | 8.11 | 2.48 | -5.63 |
| LF/HF | 35.41 | 7.26 | -28.15 |
| LF_nu | 15.19 | 2.24 | -12.95 |
| HF_nu | 23.79 | 3.64 | -20.15 |
| VLF | 71.31 | 66.37 | -4.94 |
| total_power | 30.12 | 28.63 | -1.49 |

However, the configured absolute-power mismatch pattern increased from 100 files before nfft alignment to 882 files after nfft alignment. This indicates that nfft/bin-density mismatch is a real contributor but cannot be the sole explanation for absolute-power differences.

#### Artifact/interpolation mismatch was a distinct discrepancy class

Manual inspection separated cases where disagreement likely began before PSD integration.

`2018-06-02 17-41-25.txt`:

- 21.18% of raw RR values were below 300 ms.
- Resampled-signal correlation was 0.747.
- Resampled RMSE was 3835.401 ms.
- Log-PSD correlation was -0.439.
- Largest errors included LF/HF 787.39%, LF_nu 390.18%, HF 97.13%, total power 93.85%.

`2024-07-15 16-41-40.txt`:

- 20.62% of raw RR values were below 300 ms.
- Resampled-signal correlation was 0.749.
- Resampled RMSE was 3383.328 ms.
- Log-PSD correlation was -0.373.
- Largest errors included LF/HF 215.40%, LF_nu 125.68%, HF 96.05%, total power 93.69%.

These patterns suggest that some disagreements originate in interpolation, clipping, artifact handling, or other pre-PSD signal preparation differences. This does not identify which implementation is correct.

#### Localized VLF discrepancy was observed in a relatively clean long recording

`2023-12-17 10-14-05.txt` was important because the high disagreement was localized:

| Metric | Native | NeuroKit2 | Relative error |
| --- | ---: | ---: | ---: |
| VLF | 4801.5 | 27669.2 | 82.65% |
| total_power | 10605.7 | 33170.5 | 68.03% |
| HF | 1120.45 | 1118.34 | 0.19% |
| LF | 4158.2 | 4153.94 | 0.10% |
| LF/HF | 3.71118 | 3.71438 | 0.09% |

Signal and PSD-shape agreement were strong:

- Resampled-signal correlation: 0.995.
- Detrended-signal correlation: 0.995.
- Log-PSD correlation: 0.958.

The focused VLF analysis found:

- Native VLF bins in 0.0-0.04 Hz: 5.
- NeuroKit2 VLF bins in 0.0-0.04 Hz: 10.
- Native nfft x2 VLF changed only from 4801.5 to 5194.72.
- LF and HF stayed close after alignment.
- NeuroKit2/native total-power ratio remained about 3.13 to 3.27 depending on definition.

The discrepancy appears concentrated at DC and the first few nonzero VLF bins, not across the whole spectrum.

### Ruled-Out Hypotheses

#### Catastrophic Welch implementation bug

The exploratory evidence does not support a global or catastrophic Welch failure. Some long cases showed close LF, HF, and LF/HF agreement, and diagnostics reported that enabling diagnostics changed metric outputs in 0 of 1,891 files. This does not prove correctness, but it argues against a simple catastrophic implementation bug.

#### Whole-spectrum scaling issue

The `2023-12-17 10-14-05.txt` inspection argues against a uniform whole-spectrum scaling explanation. LF, HF, and LF/HF were nearly identical while VLF and total power diverged. A single multiplicative scale factor would not produce that localized pattern.

#### nfft-only explanation

The nfft alignment experiment strongly reduced bin-count mismatch and improved several median errors. But VLF and total-power errors remained, and the focused VLF analysis showed that native `nfft x2` did not approach NeuroKit2 VLF for `2023-12-17 10-14-05.txt`. Therefore, nfft mismatch is not sufficient as a complete explanation.

### Remaining Unresolved Questions

#### Near-zero-frequency VLF behavior

The most important unresolved issue is how the two pipelines handle the PSD around DC and the first few nonzero VLF bins. In the focused VLF analysis, excluding DC reduced VLF power for both pipelines but did not remove the large VLF gap.

This needs targeted follow-up with controlled synthetic signals and explicit near-zero-frequency integration checks.

#### NeuroKit2 low-frequency handling

The available artifacts infer that NeuroKit2 uses denser frequency grids in these comparisons. They do not fully explain how NeuroKit2 handles low-frequency interpolation, edge behavior, or very-low-frequency PSD amplitude. This remains a black-box comparator issue unless the NeuroKit2 implementation path is audited directly.

#### Artifact and interpolation sensitivity

For high-artifact files, the disagreement may occur before PSD estimation. The current manual review points to this, but it does not quantify how much is due to artifact handling, interpolation method, clipping, extrapolation, or input cleaning.

#### External reference behavior

NeuroKit2 is useful but not a ground truth. A Kubios subset remains necessary to understand how a widely used HRV package behaves under comparable settings, especially for VLF and total power.

## 4. Supporting Evidence

### Run-Level Statistics

From `v02_with_diagnostics/diagnostic_summary.md` and `v02b_nfft_alignment_experiment/diagnostic_summary.md`:

| Statistic | Value |
| --- | ---: |
| Metric rows analyzed | 13,237 |
| Files with analyzable metric rows | 1,891 |
| Rows >20% relative error | 8,395 |
| Rows >80% relative error | 3,122 |
| Largest p90 relative-error metric | LF/HF, 300.27% |
| Metric-band bin mismatch, all rows | 95.27% |
| Metric-band bin mismatch, rows >20% error | 100.00% |
| Absolute-power mismatch with preserved relative distribution | 100 files |
| Diagnostics changed metric outputs | 0 / 1,891 |
| AR fallbacks | 71 / 1,891 |
| Variance consistency warnings | 1,022 / 1,891 |

### Metric-Level Agreement

| Metric | Rows | Mean relative error % | Median relative error % | P90 relative error % | Rows >80% |
| --- | ---: | ---: | ---: | ---: | ---: |
| VLF | 1,891 | 70.71 | 71.31 | 100.00 | 658 |
| LF | 1,891 | 50.95 | 36.57 | 99.16 | 527 |
| HF | 1,891 | 36.24 | 8.11 | 98.83 | 561 |
| total_power | 1,891 | 45.00 | 30.12 | 98.44 | 574 |
| LF/HF | 1,891 | 174.53 | 35.41 | 300.27 | 386 |
| LF_nu | 1,891 | 101.31 | 15.19 | 215.57 | 360 |
| HF_nu | 1,891 | 55.64 | 23.79 | 46.83 | 56 |

### Representative Files

| File | Why it matters |
| --- | --- |
| `2018-04-30 13-20-04.txt` | v01 one-minute pilot; VLF 82.35% error, HF 5.43% error |
| `2018-04-30 21-03-36.txt` | v01 one-minute pilot; VLF 69.42% error, LF_nu 2.35% error |
| `2020-09-20 16-49-51.txt` | v01 one-minute pilot; VLF 68.44% error, HF 2.01% error |
| `2018-06-02 17-41-25.txt` | Long outlier with artifact/interpolation mismatch pattern |
| `2024-07-15 16-41-40.txt` | Long outlier with artifact/interpolation mismatch pattern |
| `2024-12-08 16-21-12.txt` | Broad PSD shape alignment but large amplitude/power disagreement |
| `2023-12-17 10-14-05.txt` | Cleanest localized VLF/total-power discrepancy |

### Key Warnings Observed

Common warning patterns included:

- Frequency-domain HRV metrics may be unreliable.
- LF estimates should be interpreted cautiously.
- VLF estimates are likely unreliable.
- ULF and VLF currently overlap; definitions are reported unchanged.
- Fewer than 2 PSD bins fall inside this band.
- Recording duration is insufficient for reliable ULF/VLF/LF/HF estimation.
- Integrated FFT PSD power differs substantially from input signal variance.
- AR fallback warnings occurred in a minority of files.

These warnings support cautious interpretation, especially for short recordings and low-frequency bands.

## Confirmed finding: Welch segment detrending drives VLF discrepancy

After the PhysioNet 10-minute manual inspections, a validation-only A/B experiment isolated the effect of Welch segment detrending on the near-zero-frequency discrepancy. The representative files were:

- `nsr023_segment_131.csv`
- `nsr037_segment_131.csv`
- `nsr043_segment_135.csv`
- `nsr022_segment_022.csv`

The experiment kept preprocessing and Welch settings fixed: RR intervals were interpolated to 4 Hz, one global linear detrend was applied to the full uniformly sampled RR signal, and Welch used a Hann window with `nperseg=480`, `noverlap=360`, `nfft=960`, `scaling="density"`, and `average="mean"`. Only the Welch detrending argument changed:

- Arm A: globally detrended RR signal plus `scipy.signal.welch(detrend=False)`.
- Arm B: globally detrended RR signal plus `scipy.signal.welch(detrend="linear")`.

The key result was consistent across the representative files. Arm A reproduced NeuroKit2-like VLF values. Arm B collapsed VLF toward HRV Studio-like values. LF/HF remained nearly unchanged across both arms, with less than 0.1% relative change in the representative files.

This confirms that VLF and total_power are highly sensitive to the Welch segment-detrending convention. LF, HF, LF/HF, LFnu, and HFnu were highly stable under the same A/B change, which supports the conclusion that HRV Studio's frequency-domain implementation is not broadly wrong. The main observed difference is methodological near-zero-frequency handling, concentrated at DC and the first VLF bins, rather than a broad PSD scaling or shape bug.

This does not prove which convention is correct. It shows that cross-software VLF and total_power validation requires explicit documentation of whether detrending is applied globally before Welch, within each Welch segment, or both. A Kubios subset remains important for deciding which convention is the most appropriate external reference for final reporting.

## 5. Lessons for Next Validation Phase

### Why PhysioNet NSR is Needed

The Syl_Vain phase showed that many available recordings were too short or too artifact-sensitive for final validation. A main validation set needs:

- longer controlled segments, especially 5-minute and 10-minute windows;
- systematic segment generation;
- clearer inclusion/exclusion rules;
- enough samples to stratify by duration and quality;
- repeated observations across many subjects/records.

The prepared PhysioNet NSR dataset addresses these needs better than the exploratory Syl_Vain files. It provides a more appropriate substrate for manuscript-grade validation, especially for frequency-domain metrics.

### Why a Kubios Subset Remains Necessary

NeuroKit2 is an open-source comparator, but it is not a ground-truth standard. The Syl_Vain phase identified implementation-sensitive behavior that may also differ across established HRV packages.

A smaller Kubios subset remains necessary because:

- Kubios is widely used in HRV research;
- it can provide a pragmatic external-software benchmark;
- it may expose whether VLF and total-power behavior is NeuroKit2-specific or more general;
- it helps frame software-to-software differences for readers familiar with HRV validation practice.

The Kubios comparison should remain limited and carefully documented because proprietary preprocessing and hidden defaults can make exact replication impossible.

## 6. Implications for Validation Paper

### Potential Methods Material

The following elements may later become methods text:

- A staged validation approach: exploratory Syl_Vain comparison followed by main PhysioNet NSR validation.
- Matched high-level spectral settings: 4 Hz interpolation, Hann window, 120 s nominal segment length, 75% overlap, linear detrending, cubic NeuroKit2 interpolation.
- Use of diagnostics to capture effective Welch parameters, bin counts, duration warnings, AR fallback, variance consistency warnings, and high-error rows.
- Sensitivity analysis using native Welch `nfft=2*nperseg`.
- Manual inspection workflow for selected long-duration outliers.

### Potential Discussion Material

The following points may later support discussion:

- Cross-software frequency-domain HRV agreement depends on implementation details that are often underreported.
- Short recordings are unsuitable for robust VLF validation.
- Frequency-bin placement and nfft influence agreement but do not fully determine absolute-power behavior.
- Some discrepancies originate before PSD estimation through artifact and interpolation behavior.
- Some cleaner cases show localized VLF and total-power disagreement despite close LF, HF, and LF/HF agreement.

### Potential Limitations Material

The following limitations should remain visible:

- The Syl_Vain phase was exploratory and used a dirty git worktree for v02/v02b.
- v01 lacks formal run metadata and diagnostics.
- NeuroKit2 is a comparator, not a ground truth.
- Many Syl_Vain recordings were short and not designed for frequency-domain validation.
- Manual inspection covered only selected outliers.
- The near-zero-frequency VLF behavior remains unresolved.
- Kubios comparison will be needed for external interpretability but may be limited by proprietary defaults.

### Decision Log Summary

The Syl_Vain exploratory phase justified moving to the main PhysioNet NSR validation because it showed that:

- short recordings confound frequency-domain comparisons;
- VLF and total power need special handling;
- duration alone does not explain all discrepancies;
- nfft/bin-grid alignment improves agreement but is not sufficient;
- artifact/interpolation mismatch and localized VLF discrepancy are distinct cases;
- final validation needs longer, cleaner, reproducibly segmented data and at least one additional external-software subset.
