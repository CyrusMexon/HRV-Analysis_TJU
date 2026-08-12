# Smoothness Priors Sequence Alignment Audit

## Executive Conclusion

Section 3.3 sequence alignment status: **A. confirmed sequence-aligned**.

The existing Smoothness Priors sensitivity analysis checked 10 recordings. In all 10, the source ASCII RR sequence supplied to HRV Studio matched both Kubios `HRV.Data.RR` and Kubios selected `HRV.Data.RRs`. No sequence-length or duration mismatch was found, so a matched-sequence rerun was **not required**.

## Artifacts Located

Scripts:

- `tools/run_kubios_smoothness_priors_pilot.py`
- `tools/parse_kubios_exports.py`

Configurations and manifests:

- `validation/kubios_subset/smoothness_priors_pilot/pilot_manifest.csv`
- `validation/kubios_subset/smoothness_priors_pilot/kubios_export_settings_audit.csv`
- `validation/runs/v08_kubios_smoothness_priors_pilot_fft/run_info.json`
- `validation/runs/v08_kubios_smoothness_priors_pilot_ar/run_info.json`
- `validation/runs/v08_kubios_smoothness_priors_pilot_welch/run_info.json`
- `validation/kubios_subset/smoothness_priors_pilot/pilot_run_info.json`

Kubios exports:

- `validation/kubios_subset/smoothness_priors_pilot/kubios_exports` with 10 TXT/MAT/PDF export sets.

HRV Studio outputs:

- `validation/runs/v08_kubios_smoothness_priors_pilot_fft/hrvstudio_fft_results.csv`
- `validation/runs/v08_kubios_smoothness_priors_pilot_ar/hrvstudio_ar_results.csv`
- `validation/runs/v08_kubios_smoothness_priors_pilot_welch/hrvstudio_welch_results.csv`
- `validation/kubios_subset/smoothness_priors_pilot/smoothness_priors_fft_per_file_comparison.csv`
- `validation/kubios_subset/smoothness_priors_pilot/smoothness_priors_ar_per_file_comparison.csv`
- `validation/kubios_subset/smoothness_priors_pilot/smoothness_priors_fft_metric_summary.csv`
- `validation/kubios_subset/smoothness_priors_pilot/smoothness_priors_ar_metric_summary.csv`

Reports:

- `validation/research_notes/kubios_smoothness_priors_pilot_report.md`
- `validation/kubios_subset/smoothness_priors_pilot/hrvstudio_method_mapping.md`
- `validation/kubios_subset/smoothness_priors_pilot/parser_assessment.md`

## Recordings Included

Number of recordings: **10**.

| pilot_id | source n | Kubios RRs n | source s | Kubios RRs s | status                     |
| -------- | -------- | ------------ | -------- | ------------ | -------------------------- |
| CH001    | 922      | 922          | 600.651  | 600.651      | confirmed sequence-aligned |
| CH002    | 745      | 745          | 600.776  | 600.776      | confirmed sequence-aligned |
| CH003    | 631      | 631          | 600.971  | 600.971      | confirmed sequence-aligned |
| CH004    | 784      | 784          | 600.569  | 600.569      | confirmed sequence-aligned |
| VLF002   | 703      | 703          | 600.477  | 600.477      | confirmed sequence-aligned |
| VLF005   | 716      | 716          | 600.405  | 600.405      | confirmed sequence-aligned |
| OUT001   | 934      | 934          | 600.381  | 600.381      | confirmed sequence-aligned |
| OUT005   | 602      | 602          | 600.440  | 600.440      | confirmed sequence-aligned |
| OUT006   | 648      | 648          | 600.529  | 600.529      | confirmed sequence-aligned |
| RC003    | 709      | 709          | 600.186  | 600.186      | confirmed sequence-aligned |

## Sequence Evidence

- For each MAT export, `HRV.Data.RR` and `HRV.Data.RRs` had identical length and values.
- For each pilot source RR file listed in `pilot_manifest.csv`, the source RR vector matched `HRV.Data.RR` and `HRV.Data.RRs` exactly after seconds-to-ms unit normalization.
- Kubios text sample limits covered the full imported interval duration in every case.
- Affected by mismatch: **0/10**.

Detailed row-level evidence is saved in `validation/kubios_subset/smoothness_priors_pilot/smoothness_priors_sequence_alignment_audit.csv`.

## Spectral Settings

Kubios-visible settings from `kubios_export_settings_audit.csv`:

- Smoothness Priors lambda: 500, verified in 10/10 exports.
- Interpolation frequency: 4 Hz.
- FFT configuration: 120 s window width, 75% overlap; FFT window function not exposed/not verified in Kubios export.
- AR configuration: order 16, `Use factorization: No`.
- Frequency bands: VLF 0.0-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.4 Hz.

HRV Studio settings from v08 run metadata:

- Detrending: `smoothness_priors`, lambda 500.
- Interpolation: 4 Hz.
- Frequency bands: `{'vlf': [0.0, 0.04], 'lf': [0.04, 0.15], 'hf': [0.15, 0.4], 'total_power': [0.0, 0.4]}`.
- FFT path: `HRVFreqDomainAnalysis._compute_fft_psd`; window `hann`; whole-signal FFT, so Kubios FFT segmentation remains an estimator/configuration limitation rather than a sequence issue.
- AR path: `HRVFreqDomainAnalysis._compute_ar_psd`; requested order 16; algorithm `Burg first, Yule-Walker fallback`.
- Welch: `internal sensitivity only; no Kubios Welch output`.

## Existing Section 3.3 Results

FFT summary:

| metric      | n  | median RE % | mean RE % | Pearson r |
| ----------- | -- | ----------- | --------- | --------- |
| VLF         | 10 | 17.444      | 30.975    | 0.586     |
| LF          | 10 | 18.412      | 22.011    | 0.906     |
| HF          | 10 | 6.620       | 14.523    | 0.988     |
| total_power | 10 | 15.776      | 17.941    | 0.921     |
| LF/HF       | 10 | 14.829      | 17.489    | 0.989     |
| LF_nu       | 10 | 4.104       | 6.226     | 0.960     |
| HF_nu       | 10 | 11.483      | 13.125    | 0.960     |

AR summary:

| metric      | n  | median RE % | mean RE % | Pearson r |
| ----------- | -- | ----------- | --------- | --------- |
| VLF         | 10 | 1.397       | 2.551     | 0.997     |
| LF          | 10 | 1.388       | 1.965     | 0.999     |
| HF          | 10 | 3.184       | 9.600     | 0.989     |
| total_power | 10 | 1.735       | 3.725     | 0.996     |
| LF/HF       | 10 | 4.376       | 7.511     | 0.996     |
| LF_nu       | 10 | 1.164       | 3.043     | 0.989     |
| HF_nu       | 10 | 2.140       | 5.746     | 0.989     |

## Rerun Decision

Rerun required: **No**.

Because the sequences were confirmed aligned, no `_matched_sequence` Smoothness Priors outputs were generated. Existing FFT and AR agreement summaries remain the relevant Section 3.3 outputs.

## Manuscript Consequences

No Section 3.3 values or figures need updating due to NN-sequence harmonization. The primary Section 3.2 correction does not carry over to this Smoothness Priors pilot because Kubios selected the full imported sequence in all 10 Smoothness Priors exports.

The current interpretation remains supported: AR agreement improves substantially under nominal matched Smoothness Priors, while FFT retains greater residual differences. The residual FFT limitations remain those already documented: Kubios FFT window function is not exposed/export-verified, and HRV Studio direct FFT uses a whole-signal periodogram rather than Kubios' reported 120 s / 75% FFT windowing controls.

## Commands Used

```powershell
python tools/smoothness_priors_sequence_alignment_audit.py
```
