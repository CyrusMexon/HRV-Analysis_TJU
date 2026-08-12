# Final Claims and Limitations

This file is the manuscript-facing claims guide for the completed HRV Studio validation package. Use cautious, metric-specific wording throughout the paper.

## Strong Claims

- HRV Studio shows strong agreement with NeuroKit2 for LF, HF, LF/HF, LF_nu, and HF_nu under matched clean-data settings.
- The ten-minute segment-linear NeuroKit2 validation is the strongest primary agreement result: 53,186 metric rows across 7,598 segments, with median relative errors below 1% for LF, HF, total_power, LF/HF, LF_nu, and HF_nu.
- Welch is the best-supported primary frequency-domain method in the current validation evidence.
- HRV Studio provides transparent QC outputs, including warnings, diagnostic labels, artifact-correction counts, and before/after correction comparisons.
- HRV Studio remained numerically operational across completed duration, synthetic robustness, FFT/AR, and MIT-BIH stress-test artifacts.
- Primary frequency-domain reporting should use recordings of 5 minutes or longer, with ten-minute windows strongest in this validation package.

## Moderate Claims

- HRV Studio shows moderate, metric-dependent benchmark agreement with Kubios after QC and manual-review exclusions.
- Kubios subset medians are more appropriate headline statistics than means because retained outliers strongly affect mean relative error.
- LF/HF and normalized-unit metrics are better supported than VLF and absolute spectral powers.
- Artifact correction can improve QC transparency and stabilize detector-compatible artifacts, but correction can materially change HRV metrics.
- FFT and AR are useful secondary spectral analyses, but they are method-dependent and should not replace Welch as the primary reported method without separate validation.
- Short-duration outputs can be finite, but 30-60 s frequency-domain outputs should be treated as exploratory or screening-only.
- MIT-BIH arrhythmia results support numerical robustness and QC visibility under non-normal rhythm stress tests.

## Claims to Avoid

- Do not claim full equivalence with Kubios.
- Do not claim clinical validation.
- Do not claim diagnostic accuracy for arrhythmia detection, arrhythmia classification, or clinical decision-making.
- Do not claim that MIT-BIH arrhythmia recordings validate standard HRV metrics during arrhythmia.
- Do not claim that VLF or absolute spectral powers are robust to all preprocessing, detrending, DC, interpolation, and PSD-integration conventions.
- Do not claim that 30-60 s frequency-domain metrics are interchangeable with 5-10 min metrics.
- Do not pool arrhythmia stress-test recordings with normal-rhythm primary agreement validation.
- Do not imply that finite output alone establishes physiological validity.

## Required Limitation Statements

Primary agreement analyses were performed mainly on clean or QC-filtered RR interval data. Results may not generalize to all device noise patterns, missing-data mechanisms, rhythm abnormalities, or clinical populations.

The Kubios comparison used a manually curated benchmark subset. It is useful for targeted benchmarking but is not large enough to support broad commercial-software equivalence claims.

Kubios results depended on manual report exports and selected frequency-domain columns. Kubios sample-limit and segment-selection fields should be verified before absolute-power comparisons are presented as definitive.

VLF and absolute spectral power estimates are sensitive to detrending, DC handling, interpolation, Welch segment length and overlap, windowing, low-frequency bin treatment, and integration conventions. These metrics should be interpreted only with explicit methods reporting.

Short-duration recordings can produce finite HRV Studio outputs, but frequency-domain agreement is duration-dependent. Recordings shorter than 5 minutes should not be used for primary frequency-domain agreement claims.

MIT-BIH arrhythmia recordings were used only for robustness and quality-control stress testing under non-normal rhythm conditions. These analyses evaluate numerical stability, warning behavior, and sensitivity to artifact correction, but they do not establish clinical equivalence, diagnostic accuracy, or validity of standard HRV metrics during arrhythmia.

FFT and AR outputs are method-dependent. Welch should remain the primary reported spectral method unless FFT or AR conventions are separately specified and validated.

## Suggested Results Claim

HRV Studio demonstrated strong agreement with NeuroKit2 for most standard frequency-domain HRV metrics under matched clean-data settings, with the strongest support for LF, HF, LF/HF, and normalized-unit metrics. A smaller cleaned Kubios benchmark subset showed moderate, metric-dependent agreement, particularly for LF/HF and normalized units. Across duration, synthetic artifact, FFT/AR, and MIT-BIH arrhythmia stress tests, HRV Studio remained operational and produced QC-visible outputs, but VLF, absolute spectral powers, short-duration frequency-domain estimates, and arrhythmic recordings require cautious interpretation.

## Suggested Conclusion Wording

These validation results support HRV Studio as a transparent and reproducible HRV analysis platform for QC-aware research workflows. The evidence supports metric-specific agreement claims under clearly reported preprocessing and spectral settings, but it does not support unconditional equivalence with Kubios or clinical validity claims in arrhythmic recordings.

