# Manuscript Outline

## 1. Introduction

Content:

- Motivate HRV analysis as a common but convention-sensitive research workflow.
- Explain why reproducibility requires explicit preprocessing, spectral settings, and QC reporting.
- Position HRV Studio as a transparent HRV analysis platform rather than a diagnostic system.
- State the validation objective: evaluate agreement, robustness, duration sensitivity, and QC behavior across external comparators and stress tests.

Figures/tables:

- No required results figure.
- Optional high-level workflow schematic if a polished HRV Studio pipeline figure is available.

## 2. Related Work

Content:

- Briefly summarize time-domain and frequency-domain HRV metrics.
- Discuss Welch, FFT, and AR spectral estimation and why method conventions matter.
- Summarize the role of NeuroKit2 and Kubios as comparators.
- Explain duration recommendations and why VLF is sensitive to short recordings.
- State why arrhythmia recordings are not primary validation material for standard HRV agreement.

Figures/tables:

- Optional comparator table covering HRV Studio, NeuroKit2, Kubios, Welch, FFT, and AR conventions.

## 3. HRV Studio System Design

Content:

- Describe RR interval input handling.
- Describe artifact detection/correction and before/after reporting.
- Describe frequency-domain defaults used in validation: 4 Hz interpolation, 120 s Welch segment, 75% overlap, Hann window, and specified detrending convention.
- Describe warnings, diagnostics, finite-output checks, and QC transparency.

Figures/tables:

- HRV Studio processing and QC pipeline figure.
- Table of primary software settings used in validation.

## 4. Validation Methodology

Content:

- Present validation phases in order: NeuroKit2 large-scale, Kubios subset, FFT/AR, synthetic robustness, duration sensitivity, MIT-BIH arrhythmia robustness.
- Define primary endpoints: relative error, correlation, finite-output rate, warning rate, correction behavior, PSD consistency, and duration stability.
- Define inclusion/exclusion criteria and manual-review handling for Kubios.
- Clarify that MIT-BIH arrhythmia recordings were stress tests, not clinical validation evidence.

Figures/tables:

- Validation phase summary table.
- `master_validation_numbers_table.csv` converted to a compact manuscript table.
- Flow diagram for Kubios 50 -> 48 -> 46 -> 44 filtering.

## 5. Results

Content:

- Start with large-scale NeuroKit2 results. Emphasize ten-minute segment-linear findings and the improvement over the five-minute diagnostic pattern.
- Present Kubios final 44-file results using medians as the primary statistic and means as outlier-sensitive secondary statistics.
- Present duration sensitivity, emphasizing 5-minute minimum and 10-minute strongest support.
- Present synthetic robustness results, focusing on finite output, warning behavior, and correction effects.
- Present FFT/AR results as method checks supporting Welch-first reporting.
- Present MIT-BIH arrhythmia robustness only as QC/numerical-stability stress-test evidence.

Figures/tables:

- Figure: ten-minute NeuroKit2 metric-level agreement.
- Figure: Kubios final 44-file agreement.
- Figure: duration sensitivity.
- Figure: synthetic robustness before/after correction.
- Optional figure or supplement: FFT/AR representative PSD comparison.
- Optional supplement: MIT-BIH tachogram examples.

## 6. Discussion

Content:

- Interpret HRV Studio as defensible for QC-aware research workflows under explicit settings.
- Explain why LF/HF and normalized metrics are better supported than VLF and absolute powers.
- Discuss detrending, DC handling, and PSD integration as convention-sensitive sources of disagreement.
- Discuss QC warnings as screening indicators rather than guarantees of physiological validity.
- Explain that Kubios benchmark results are useful but do not establish equivalence.
- Explain that arrhythmia data support robustness only, not clinical validation.

Figures/tables:

- No new required figure.
- Optional summary table mapping claims to supporting evidence and limitations.

## 7. Limitations

Content:

- Kubios subset size and manual export limitations.
- Clean-data and QC-filtered emphasis.
- VLF and absolute-power sensitivity.
- Short-duration instability.
- MIT-BIH arrhythmia robustness is not clinical validation or diagnostic-accuracy evidence.
- FFT and AR method dependence.
- PSD convention sensitivity and sample-limit uncertainty.

Figures/tables:

- Optional limitations table with mitigation and manuscript wording.

## 8. Conclusion

Content:

- State that HRV Studio is defensible as a transparent, reproducible, QC-aware HRV platform.
- Emphasize metric-specific agreement under reported settings.
- Avoid equivalence and clinical claims.
- Recommend future expansion with larger Kubios exports, prospective device data, and clinical cohorts only if the paper wants to move beyond software validation.

Figures/tables:

- None required.

