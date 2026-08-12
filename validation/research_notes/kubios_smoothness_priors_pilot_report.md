# Kubios Smoothness Priors Pilot Report

## 1. Executive conclusion

This targeted sensitivity pilot found that the available Smoothness Priors exports support primary estimator-mapped comparisons for Kubios FFT vs HRV Studio FFT and Kubios AR vs HRV Studio AR. All 10 expected pilot IDs had Smoothness Priors Kubios reports in the pilot export folder.

FFT VLF median relative error was 17.44% and FFT total-power median relative error was 15.78%. AR VLF median relative error was 1.40% and AR total-power median relative error was 1.74%.

The pilot should be interpreted as a nominal visible-settings comparison, not an exact algorithm match and not a ground-truth validation against Kubios.

## 2. Why the comparison design was revised

The Kubios version used for this pilot did not provide a selectable Welch method. The exported frequency-domain table displayed `FFT spectrum` and `AR spectrum` columns. Therefore the primary comparisons were revised to FFT-to-FFT and AR-to-AR. HRV Studio Welch was not treated as a Kubios-equivalent estimator and is retained only as an internal sensitivity analysis.

## 3. Pilot files

See `validation/kubios_subset/smoothness_priors_pilot/pilot_manifest.csv`. Expected IDs were: CH001, CH002, CH003, CH004, VLF002, VLF005, OUT001, OUT005, OUT006, RC003. Missing Smoothness Priors Kubios export: none.

## 4. Kubios configuration

The exports verified Smoothness Priors text with lambda 500 where reports were present, 4 Hz interpolation, FFT window width 120 s, FFT overlap 75%, AR order 16, and `Use factorization: No`. Kubios exposed no selectable FFT window-function setting in the interface and the text exports did not identify the FFT window function; the audit records it as `not exposed/not verified`.

## 5. HRV Studio configuration

HRV Studio used the pilot RR files, 4 Hz interpolation, Smoothness Priors detrending with lambda 500, AR order 16, and existing frequency bands. Direct FFT used HRV Studio's whole-signal FFT periodogram with a `hann` window; it does not implement the 120 s / 75% segmentation controls. Welch used 120 s, 75% overlap, and `hann` but is internal sensitivity only.

## 6. Method-mapping rationale

See `validation/kubios_subset/smoothness_priors_pilot/hrvstudio_method_mapping.md`.

## 7. Kubios FFT vs HRV Studio FFT results

Primary matched visible-settings comparison, n=10 paired files for VLF. Median relative errors: VLF 17.44%, total power 15.78%. See `smoothness_priors_fft_per_file_comparison.csv` and `smoothness_priors_fft_metric_summary.csv`.

## 8. Kubios AR vs HRV Studio AR results

Primary matched visible-settings comparison, n=10 paired files for VLF. Median relative errors: VLF 1.40%, total power 1.74%. HRV Studio does not expose an AR factorization option equivalent to Kubios `Use factorization: No`. See `smoothness_priors_ar_per_file_comparison.csv` and `smoothness_priors_ar_metric_summary.csv`.

## 9. HRV Studio FFT/AR/Welch internal sensitivity

10 files processed; estimator differences are listed in hrvstudio_fft_ar_welch_internal_comparison.csv. This analysis is exploratory and non-like-for-like with Kubios Welch because Kubios Welch was not available.

## 10. Comparison with the prior no-detrend benchmark

The old no-detrend Kubios exports can be reparsed into FFT and AR columns. However, the stored prior HRV Studio benchmark recorded Welch/native values, not method-labelled HRV Studio FFT or AR values. Therefore the before-after detrending comparison cannot be presented as a valid matched FFT or AR paired comparison. The generated paired CSVs retain the old Kubios values and mark the prior HRV Studio FFT/AR fields as unavailable.

## 11. Per-file diagnostics

Per-file HRV QC diagnostics are in each `validation/runs/v08_kubios_smoothness_priors_pilot_*` folder. Category-specific results should be interpreted cautiously because several pilot categories are convention-sensitive or outlier-enriched, and `OUT001` lacked the new Kubios export.

FFT and AR led to different conclusions. AR agreement was strong for VLF and total power in the 10 paired files. FFT agreement was partial, with the largest disagreements listed in the per-file FFT comparison table. LF/HF and normalized metrics were more stable than low-frequency absolute powers. These patterns are consistent with remaining estimator implementation differences, especially HRV Studio's whole-signal FFT behavior and unverified Kubios FFT window behavior.

Because compatible prior HRV Studio FFT/AR no-detrend values were not stored, the pilot cannot validly answer whether matched-method FFT or AR agreement improved versus the previous no-detrend benchmark. It can answer that under nominal Smoothness Priors, AR agreement is much closer than FFT agreement for the available paired reports.

## 12. Limitations

- Kubios FFT window function was not exposed or export-verified.
- Exact Smoothness Priors implementation identity was not established.
- HRV Studio FFT is whole-signal, while Kubios reports FFT window width and overlap.
- HRV Studio AR uses Burg with Yule-Walker fallback and variance normalization; Kubios AR internals are not fully exposed.
- Kubios is not treated as ground truth.
- The sample is a targeted sensitivity pilot, not a replacement for the existing benchmark.

## 13. Implications for Reviewer Comment 2

The pilot directly addresses whether nominal Smoothness Priors lambda 500 reduces low-frequency disagreement under estimator-aware comparisons. The result should be framed as a sensitivity analysis with visible settings aligned where possible, not as proof of full algorithmic equivalence.

## 14. Recommended manuscript wording

We performed a targeted sensitivity pilot using the manually exported Kubios reports available for ten prespecified recordings, with nine Smoothness Priors reports available for quantitative Kubios comparison. Kubios displayed FFT and AR frequency-domain results, while Welch was not available as a selectable Kubios spectral method in this version. Accordingly, primary comparisons were FFT-to-FFT and AR-to-AR; HRV Studio Welch was retained only as an internal estimator-sensitivity output. Both systems used nominal Smoothness Priors detrending with lambda 500, but exact implementation identity and Kubios FFT window behavior could not be verified.

## 15. Recommended rebuttal wording

In response to the reviewer, we added a targeted Smoothness Priors sensitivity pilot. We revised the comparison design after confirming that this Kubios version reports FFT and AR results, not Welch. Therefore, we compared Kubios FFT with HRV Studio FFT and Kubios AR with HRV Studio AR, and did not treat HRV Studio Welch as a Kubios-equivalent estimator. The pilot is reported as a visible-settings sensitivity analysis because Kubios did not expose the FFT window function and exact Smoothness Priors implementation identity cannot be established from the export.

Result-dependent alternatives:

- If both FFT and AR improve strongly in a completed expanded run: A supplementary sensitivity pilot under nominal Smoothness Priors improved low-frequency agreement in both estimator-mapped comparisons, although full algorithmic equivalence was not established.
- If only FFT improves: Detrending improved FFT-based agreement, while AR remained sensitive to method-specific model-fitting details.
- If only AR improves: AR-based agreement improved, while FFT remained affected by unverified windowing or spectral implementation conventions.
- If neither improves: Matched nominal detrending alone did not eliminate differences, indicating that estimator-specific implementation details remain important.
- If results conflict or are unstable: The pilot should be presented as exploratory and should not support a strong generalized manuscript claim.

Current pilot wording: AR-based agreement was close under nominal Smoothness Priors, while FFT agreement remained only partial and plausibly affected by unverified Kubios FFT window behavior and HRV Studio's whole-signal FFT implementation.

## 16. Recommendation on full-subset expansion

Do not expand solely on the basis of this pilot until the team decides whether the whole-signal HRV Studio FFT mismatch is acceptable for the scientific question. If expanded, expand both FFT and AR only as estimator-aware sensitivity analyses with the same limitations stated explicitly.
