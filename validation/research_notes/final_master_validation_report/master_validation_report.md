# HRV Studio Master Validation Report

Generated: 2026-06-15

Source of truth: `D:\Team_Zhao_projects\HRV\HRV-Analysis_TJU\`

This consolidated report uses only validation artifacts in the HRV-Analysis_TJU project. It does not use the old Glucose-HRV project, does not rerun experiments, and does not modify production HRV code.

## A. Executive Summary

HRV Studio has now been evaluated across large-scale NeuroKit2 agreement runs, a targeted Kubios benchmark subset, FFT/AR spectral method checks, synthetic signal-condition robustness studies, duration-sensitivity analyses, and a migrated MIT-BIH arrhythmia robustness/QC stress test. The evidence base covers normal-sinus PhysioNet RR segments, manually exported Kubios benchmark files, controlled synthetic corruptions, duration-truncated clean records, and selected arrhythmic MIT-BIH RR segments used only as stress-test material.

Across the completed artifacts, the package includes approximately 23,261 file, segment, or condition-level analysis units, not counting duplicated metric rows as independent recordings. The largest validation evidence comes from 15,179 five-minute PhysioNet segments and 7,598 ten-minute PhysioNet segments. Metric-row coverage includes 106,253 rows in the five-minute NeuroKit2 run, 53,186 rows in the ten-minute NeuroKit2 run, 308 rows in the final 44-file Kubios manual-review dataset, 2,250 rows in the duration-sensitivity study, and additional robustness rows for synthetic and MIT-BIH stress tests.

The strongest finding is that HRV Studio shows strong agreement with NeuroKit2 for LF, HF, LF/HF, LF_nu, and HF_nu under matched, clean, large-scale conditions. The ten-minute segment-linear PhysioNet run is the primary numerical agreement result: median relative errors were 0.39% for LF, 0.11% for HF, 0.87% for total_power, 0.44% for LF/HF, 0.14% for LF_nu, and 0.26% for HF_nu. VLF remained the most convention-sensitive metric, with a 9.18% median and 14.81% p90 relative error in that run.

The Kubios subset supports a more cautious benchmark claim. The workflow parsed 50/50 Kubios exports, matched 48 files to validation rows, retained 46 after automatic quality control, and used 44 files in the final manual-review dataset. In the final cleaned dataset, agreement distributions remained skewed because of retained pathological cases and the known sensitivity of absolute spectral power metrics to preprocessing and PSD conventions. Consequently, robust statistics were prioritized for interpretation. HRV Studio achieved an overall **median relative error of 16.68%** versus Kubios, while the overall mean relative error (**88.85%**) remained disproportionately influenced by a small number of high-error outlier recordings despite quality-control filtering and manual review. The strongest Kubios-facing metrics were **LF_nu**, **HF_nu**, and **LF/HF**, showing the strongest median agreement and correlation structure, whereas absolute powers—particularly **VLF**, **HF**, and **total_power**—remained more sensitive to retained edge cases and methodological conventions.


The FFT/AR validation supports Welch as the primary publishable spectral method. HRV Studio Welch LF/HF correlated strongly with NeuroKit2 Welch (r = 0.987). FFT and AR were numerically finite in 100/100 files, but FFT showed pervasive PSD-area/variance flags because of DC and no-detrend convention differences, and AR showed order sensitivity in 25/100 files.

The robustness and duration studies support operational stability and transparent QC rather than unconditional physiological validity. Synthetic corruptions remained finite before and after correction, and the larger ten-base robustness study reported 60/60 finite outputs with 60/60 warning visibility. Duration testing showed finite outputs at 30 s, 60 s, 2 min, 5 min, and 10 min, but frequency-domain interpretation became defensible only with longer recordings; 5 minutes or longer should be recommended for primary frequency-domain reporting, with 10 minutes strongest in this validation design. MIT-BIH arrhythmia data remained numerically finite in 12/12 selected segments after correction, but all arrhythmia findings must be framed as robustness/QC stress tests, not clinical validation.

Estimated validation completeness: 90%. The remaining 10% is mostly manuscript-facing risk reduction: final figure selection, wording discipline, sample-limit confirmation for Kubios exports, and optional targeted review of retained high-error Kubios files.

Estimated manuscript readiness: 75%. The evidence is mature enough to begin manuscript drafting now, but the paper still needs figure cleanup, a methods table, and strict limitation wording.

Final verdict: HRV Studio is publishable and defensible as a transparent, QC-aware HRV analysis platform when claims are metric-specific and tied to clearly reported preprocessing settings. The evidence does not support broad claims of Kubios equivalence, clinical validation, diagnostic accuracy, or unrestricted interpretation of short-duration, VLF, absolute-power, or arrhythmic outputs.

## B. Phase-by-Phase Validation Summary

### 1. Large-Scale NeuroKit2 Validation

The five-minute PhysioNet NeuroKit2 validation analyzed 106,253 metric rows across 15,179 files. Median relative errors were low for LF (0.56%), HF (0.18%), LF/HF (0.66%), LF_nu (0.20%), and HF_nu (0.39%). VLF and total_power were the main higher-discrepancy metrics, with VLF mean/median relative error of 31.45%/28.23% and total_power mean/median relative error of 15.15%/9.26%. The diagnostic run identified 9,899 files with an absolute-power mismatch pattern while relative spectral distribution was preserved.

The ten-minute PhysioNet segment-linear run analyzed 53,186 metric rows across 7,598 files. This is the strongest large-scale agreement result. Rows exceeding 20% relative error decreased to 271, and rows exceeding 80% decreased to 2. Median relative errors were below 1% for LF, HF, total_power, LF/HF, LF_nu, and HF_nu. VLF remained the largest p90 relative-error metric but improved materially relative to the five-minute diagnostic pattern.

The major methodological discovery was that Welch detrending and DC handling strongly affect VLF and total_power. Segment-wise linear detrending reduced the broad absolute-power discrepancy pattern in the ten-minute NeuroKit2 run. In contrast, the Kubios subset was intentionally evaluated with no detrending to match the Kubios export convention after the Arm A fix. This means the paper should describe detrending as a planned methodological convention, not a minor implementation detail.

The DC handling fix and detrending review support the interpretation that much of the VLF and total-power disagreement is convention-sensitive. Low-frequency bins, DC removal, interpolation, segment length, overlap, windowing, and bin-boundary integration can change absolute spectral area without changing LF/HF or normalized-unit structure.

Main interpretation: HRV Studio shows strong large-scale agreement with NeuroKit2 for standard LF/HF and normalized frequency-domain metrics under matched settings. VLF and absolute powers require explicit methods reporting and cautious language.

### 2. Kubios Benchmark Subset

The Kubios benchmark subset used 50 manually exported Kubios files. Parser success was complete: 50/50 exports were discovered and parsed. Forty-eight files matched validation rows. After automatic QC, 46 files remained. After manual-review exclusions, the final manuscript-facing dataset contained 44 files.

In the 44-file final dataset, due to retained pathological cases and absolute-power sensitivity, Kubios agreement showed skewed error distributions; therefore median and robust statistics were prioritized as primary headline statistics because means remain outlier-sensitive even after manual review.

Metric-level results were heterogeneous. Median relative errors were strongest for LF_nu (7.45%), HF_nu (11.22%), LF/HF (17.56%), total_power (19.91%), LF (20.42%), HF (28.72%), and VLF (31.92%). Pearson correlations were strongest for LF/HF (0.895), HF_nu (0.862), LF_nu (0.860), and LF (0.835), with weaker correlations for VLF, HF, and total_power.

The weakest metrics were VLF and absolute spectral powers, especially when retained pathological or convention-sensitive cases were included. The strongest interpretation is not Kubios equivalence. It is that HRV Studio shows moderate, metric-dependent benchmark agreement with Kubios after QC, with best support for normalized and ratio metrics.

### 3. FFT / AR Validation

The FFT/AR validation processed 100 clean ten-minute PhysioNet files. The primary anchor remained HRV Studio Welch. HRV Studio Welch LF/HF correlated strongly with NeuroKit2 Welch (r = 0.987), supporting the publishable choice of Welch as the primary frequency-domain method.

FFT LF/HF correlation versus HRV Studio Welch was 0.853, with a median LF/HF relative difference of 16.87%. AR LF/HF correlation versus Welch was 0.901, with a median LF/HF relative difference of 10.92%. FFT and AR PSD outputs were positive in 100/100 files and had no non-finite PSD values.

The caveat is PSD-area consistency. FFT median integrated PSD / variance ratio was 96.757, while AR was 1.000. FFT instability flags occurred in 100/100 files, mostly from PSD/variance mismatch driven by no-detrend DC handling. AR instability flags occurred in 2/100 files, and 25/100 files had AR LF/HF order spread greater than 50%.

Publication recommendation: report Welch as the primary spectral method. FFT and AR can be described as secondary method-specific analyses, with explicit disclosure of PSD-area, fallback, and order-sensitivity checks.

### 4. Synthetic Robustness Study

The synthetic robustness work used controlled corruption types: random missed beats, ectopic short-long pairs, Gaussian jitter noise, short dropout sections, isolated extreme RR artifacts, and clean controls. The initial two-base study covered 12 file/condition cases; the stronger ten-base study covered 60 cases.

In the ten-base robustness study, metric rows remained finite before correction in 60/60 cases and after correction in 60/60 cases. Warning or diagnostic labels appeared after correction in 60/60 cases. Artifact detection and correction were present in 46/60 cases.

After correction, the most stable metrics versus the clean same-file baseline were SDNN, RMSSD, and total_power. The more sensitive metrics were HF, LF/HF, and LF. Artifact correction changed some metrics materially, especially HF, RMSSD, total_power, LF/HF, and LF, so before/after correction status must be reported explicitly.

Warning behavior was useful as a QC screen: it exposed noise, artifact burden, runtime warnings, and Welch low-frequency limitations. It should not be framed as proof that finite outputs are physiologically valid.

### 5. Duration Sensitivity Study

The duration study evaluated 50 clean PhysioNet recordings at 30 s, 60 s, 2 min, 5 min, and 10 min, producing 250 file-duration combinations and 2,250 metric rows. Finite output rate was 100% at every duration, and warning frequency was also 100% at every duration.

Duration had a strong effect on frequency-domain interpretability. LF/HF median relative error versus NeuroKit2 was 60.29% at 30 s, 27.66% at 60 s, 28.61% at 2 min, 11.06% at 5 min, and 11.05% at 10 min. The median absolute change versus HRV Studio's own 10-minute value remained high at short durations: 65.20% at 30 s, 53.03% at 60 s, 44.46% at 2 min, 32.29% at 5 min, and 0.00% at 10 min.

VLF was the most duration-limited metric. VLF instability was 100% at 30 s, 60 s, 2 min, and 5 min, and only reached 0% at 10 min. LF instability was 100% at 30 s, 60 s, and 2 min, but 0% at 5 min and 10 min.

Minimum defensible duration: use 5 minutes or longer for primary frequency-domain reporting, with 10 minutes strongest in this validation design. For recordings shorter than 5 minutes, VLF and total_power should be marked unreliable or convention-sensitive. For 30-60 seconds, frequency-domain metrics should be labeled exploratory or screening-only.

### 6. MIT-BIH Arrhythmia Robustness

The migrated MIT-BIH v11 robustness run analyzed 12 selected arrhythmia segments. The mix included PVC/ectopic-heavy rhythm, mostly normal rhythm, irregular rhythm, and noisy/problematic rhythm.

Numerical stability was good as a stress-test result: finite metrics were present before correction in 12/12 segments and after correction in 12/12 segments. Warning behavior was also universal: after-correction warnings appeared in 12/12 segments. Corrected intervals were present in 9/12 segments.

The largest median correction effects were observed for LF (97.50%), LF/HF (81.42%), and RMSSD (64.19%). Mostly normal rhythm segments had approximately 0% category-level median correction effect, while irregular and noisy/problematic rhythms changed substantially.

Interpretation: arrhythmic records should be excluded from primary agreement validation and used only as robustness/QC stress tests. Required limitation wording: MIT-BIH arrhythmia recordings are robustness/QC stress tests, not clinical validation or diagnostic-accuracy evidence.

## C. Final Claims

### Strong Claims

- HRV Studio shows strong agreement with NeuroKit2 for LF, HF, LF/HF, LF_nu, and HF_nu under matched clean-data settings.
- The ten-minute segment-linear PhysioNet validation is the strongest primary agreement result.
- Welch is the best-supported primary frequency-domain method in the current evidence package.
- HRV Studio provides transparent QC artifacts, including warnings, diagnostic labels, artifact correction counts, and before/after correction comparisons.
- HRV Studio remains numerically operational across completed synthetic, duration, FFT/AR, and MIT-BIH robustness checks.
- Five minutes or longer is the defensible minimum recommendation for primary frequency-domain reporting, with ten-minute windows strongest in this package.

### Moderate Claims

- HRV Studio shows moderate, metric-dependent benchmark agreement with Kubios after QC and manual-review exclusions.
- Kubios subset medians are defensible headline statistics; means should be treated as outlier-sensitive secondary statistics.
- LF/HF and normalized-unit metrics are better supported than VLF and absolute spectral powers.
- Artifact correction can improve QC transparency and stabilize detector-compatible artifacts, but it can materially change metrics.
- FFT and AR analyses are useful secondary method checks, but they are convention- and order-sensitive.
- Short-duration outputs can be finite and useful descriptively, but 30-60 s frequency-domain results should be treated as exploratory.
- MIT-BIH arrhythmia results support robustness/QC visibility under non-normal rhythm stress tests.

### Claims to Avoid

- Do not claim full Kubios equivalence.
- Do not claim HRV Studio is clinically validated.
- Do not claim diagnostic accuracy for arrhythmia detection or classification.
- Do not claim arrhythmic MIT-BIH recordings validate standard HRV metrics under arrhythmia.
- Do not claim VLF or total_power are robust to all detrending, DC, interpolation, and PSD-integration conventions.
- Do not claim 30-60 s frequency-domain metrics are interchangeable with 5-10 min metrics.
- Do not pool arrhythmia stress tests with normal-rhythm primary agreement validation.
- Do not imply that finite output alone establishes physiological validity.

## D. Final Limitations

- The Kubios benchmark subset is small and manually curated. It is appropriate for targeted benchmarking, not for broad commercial-software equivalence claims.
- Kubios export parsing depended on manual report exports and the first frequency-domain result column. Kubios sample-limit and segment-selection fields should be confirmed before making strong absolute-power comparisons.
- Primary agreement results emphasize clean or QC-filtered RR interval data. Results may not generalize to all device noise patterns, missing-data mechanisms, or rhythm abnormalities.
- VLF and total_power remain sensitive to detrending, DC handling, interpolation, Welch segment length, overlap, windowing, low-frequency bin treatment, and integration conventions.
- Short-duration recordings can produce finite outputs, but frequency-domain agreement is duration-dependent and unstable for VLF and some ratio metrics at 30-120 seconds.
- MIT-BIH arrhythmia recordings are robustness/QC stress tests, not clinical validation or diagnostic-accuracy evidence.
- FFT and AR outputs are method-dependent. FFT showed PSD-area inconsistency under the current no-detrend convention, and AR showed non-trivial order sensitivity.
- PSD convention sensitivity should be framed as a methodological limitation rather than hidden implementation failure unless file-level review shows otherwise.

## E. Recommended Manuscript Structure

1. Introduction: Explain the need for transparent, reproducible HRV software and the risk of hidden preprocessing differences. Include no results figures; use this section to motivate QC-aware validation.
2. Related Work: Summarize HRV frequency-domain conventions, NeuroKit2, Kubios, Welch/FFT/AR spectral methods, duration recommendations, and arrhythmia limitations. Include a concise table of validation comparators if space allows.
3. HRV Studio System Design: Describe RR ingestion, artifact handling, frequency-domain settings, warnings, diagnostic outputs, and report generation. Include a workflow schematic or QC pipeline figure if available.
4. Validation Methodology: Describe each validation phase, datasets, inclusion/exclusion criteria, settings, metrics, QC filters, and statistical summaries. Include a validation-phase table and the final numbers table.
5. Results: Present large-scale NeuroKit2 results first, then Kubios subset, duration sensitivity, robustness/QC, FFT/AR, and MIT-BIH stress test. Use only the strongest non-redundant figures.
6. Discussion: Interpret metric-specific agreement, VLF/absolute-power sensitivity, duration effects, QC warnings, artifact correction, and why equivalence/clinical claims are not supported.
7. Limitations: State Kubios subset size, manual export constraints, clean-data emphasis, short-duration limits, arrhythmia limits, FFT/AR caveats, and PSD convention sensitivity.
8. Conclusion: Position HRV Studio as a defensible, QC-aware HRV platform for research workflows under explicit preprocessing and spectral settings.

## F. Recommended Paper Figures

Keep only the strongest figures in the main paper:

- NeuroKit2 ten-minute summary table or compact forest/bar plot of median relative errors by metric.
- Kubios agreement figure focused on final 44-file manual-review dataset, preferably LF/HF, LF_nu, HF_nu, and selected absolute-power panels.
- Duration sensitivity plot showing LF/HF error/correlation by duration and a clear VLF limitation marker.
- Synthetic robustness before/after correction plot from the ten-base robustness run.
- One FFT/AR comparison figure showing Welch as primary and FFT/AR as secondary method-dependent checks.

Use as optional supplement:

- Five-minute NeuroKit2 diagnostic summary and VLF-focused detail.
- Additional Kubios Bland-Altman panels for LF, HF, total_power.
- MIT-BIH tachogram examples as stress-test evidence, not validation evidence.
- Robustness tachogram examples for each synthetic corruption type.
- AR order-sensitivity table.

Remove or avoid from the main paper:

- Redundant scatter plots for every metric.
- All-matched Kubios mean tables as headline evidence.
- Any figure that mixes arrhythmia stress tests with normal-rhythm agreement validation.
- Large raw warning tables unless moved to supplement.

## G. Final Numbers Table

See `master_validation_numbers_table.csv` in this folder.

## H. Validation Readiness Checklist

See `validation_readiness_checklist.md` in this folder.

