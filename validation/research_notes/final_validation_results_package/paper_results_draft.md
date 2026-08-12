# Paper Results Draft

## Results: Large-Scale NeuroKit2 Validation

Frequency-domain outputs were evaluated against NeuroKit2 in two large PhysioNet validation runs. In the 5-min run, 106,253 metric rows from 15,179 files were analyzed. Median relative errors were low for LF (0.56%), HF (0.18%), LF/HF (0.66%), LF_nu (0.20%), and HF_nu (0.39%). VLF and total_power showed larger discrepancies, with VLF mean/median relative error of 31.45%/28.23% and total_power mean/median relative error of 15.15%/9.26%.

In the 10-min segment-wise linear detrending run, 53,186 metric rows from 7,598 files were analyzed. Discrepancies were substantially reduced for most metrics: median relative errors were 0.39% for LF, 0.11% for HF, 0.87% for total_power, 0.44% for LF/HF, 0.14% for LF_nu, and 0.26% for HF_nu. VLF remained the largest p90 relative-error metric, with mean/median relative error of 9.95%/9.18%. These findings indicate strong numerical agreement for most LF/HF and normalized metrics under matched large-scale validation conditions, while VLF remains more sensitive to the spectral estimation convention.

## Results: Kubios Subset Validation

The Kubios subset workflow parsed all 50 available Kubios exports. Forty-eight files matched validation rows, 46 files remained after automatic QC, and 44 files remained after manual-review exclusions. In the final cleaned 44-file subset, HRV Studio had an overall mean relative error of 88.85% and an overall median relative error of 16.68% versus Kubios.

Metric-level results were heterogeneous. Median relative errors were 7.45% for LF_nu, 11.22% for HF_nu, and 17.56% for LF/HF. Absolute spectral powers showed larger median relative errors: 31.92% for VLF, 20.42% for LF, 28.72% for HF, and 19.91% for total_power. These results support reporting medians as the primary subset statistic because means remain strongly influenced by retained edge cases.

## Discussion: Interpretation of Frequency-Domain Discrepancies

The validation results do not justify a claim that HRV Studio is fully equivalent to Kubios. Instead, they indicate that agreement is strongest for LF/HF and normalized metrics, where relative spectral distribution is less affected by absolute power scaling and low-frequency conventions. Absolute power metrics, especially VLF and total_power, remain sensitive to methodological choices including detrending, interpolation, Welch segment length and overlap, windowing, and frequency-bin integration.

The manual-review and QC results also show that pathological files can dominate mean relative error. Files with validation input failures, nonfinite outputs, zero or near-zero native powers, extreme power ratios, short/adjusted Welch windows, or preprocessing instability should not be pooled with clean files for headline agreement claims. A cautious manuscript interpretation is that HRV Studio frequency-domain metrics show good agreement for ratio and normalized measures under matched settings, while absolute spectral powers require explicit reporting of preprocessing conventions and continued QC-based review.
