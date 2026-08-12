# Frequency Band Definition Audit

## Executive Summary

This is a diagnostic-only audit. Production code, manuscript files, and existing validation outputs were not modified.

Current HRV Studio code defines overlapping low-frequency bands: ULF is `0.0-0.003 Hz`, VLF is `0.0-0.04 Hz`, LF is `0.04-0.15 Hz`, HF is `0.15-0.40 Hz`, and total power is `0.0-0.40 Hz`. Band masks are inclusive at both ends in the production spectral metric function, and integration uses `np.trapezoid`.

The main empirical effect of non-overlapping candidate definitions is concentrated in ULF, VLF, VLF percent-of-total, and total power when the DC bin is excluded. LF/HF, LFnu, and HFnu can also change when exact boundary bins such as `0.15 Hz` are present, because production LF currently includes the `0.15 Hz` bin and HF also includes it.

## Task 1: Confirmed Implementation

Authoritative implementation: `hrvlib/metrics/freq_domain.py`.

- Band definitions: `HRVFreqDomainAnalysis.DEFAULT_FREQ_BANDS`.
- PSD generation: `_compute_welch_psd()`, `_compute_fft_psd()`, `_compute_ar_psd()`.
- Spectral metric integration: `_compute_spectral_metrics(use_ar=False, use_fft=False)`.
- Diagnostics use the same band definitions in `_compute_band_diagnostics()`.

Current masks in `_compute_spectral_metrics()`:

- ULF: `(freqs >= 0.0) & (freqs <= 0.003)`.
- VLF: `(freqs >= 0.0) & (freqs <= 0.04)`.
- LF: `(freqs >= 0.04) & (freqs <= 0.15)`.
- HF: `(freqs >= 0.15) & (freqs <= 0.4)`.
- Total: `(freqs >= 0.0) & (freqs <= 0.4)`, implemented as the lower bound of VLF through the upper bound of HF.

Integration method: `np.trapezoid(psd[mask], freqs[mask])`, clamped to non-negative for individual bands. The 0-Hz/DC bin is included in ULF, VLF, and total power when present. Exact boundary bins are included by both adjacent band masks. The exact boundary point itself has no finite width, but it participates as an endpoint in trapezoids on both sides and is double-counted in bin counts and any simple sum of band powers.

Welch, FFT, and AR all use `_compute_spectral_metrics()`:

- Welch: `self.spectral_metrics = self._compute_spectral_metrics()`.
- FFT: `self.fft_spectral_metrics = self._compute_spectral_metrics(use_fft=True)`.
- AR: `self.ar_spectral_metrics = self._compute_spectral_metrics(use_ar=True)`.

## Task 2: Derived Metrics

Production formulas from `_compute_spectral_metrics()`:

- `ulf_power`, `vlf_power`, `lf_power`, `hf_power`: trapezoidal band integration over the masks above.
- `total_power`: trapezoidal integration over `0.0 <= f <= 0.4`.
- `ulf_power_nu`, `vlf_power_nu`, `lf_power_nu`, `hf_power_nu`: each band power divided by `total_power` and multiplied by 100. These are percent-of-total outputs, despite the `_nu` suffix.
- `lf_hf_ratio`: `lf_power / hf_power`.
- `lf_nu`, `hf_nu`: `LF/(LF+HF)*100` and `HF/(LF+HF)*100`.
- `relative_lf_power`, `relative_hf_power`: same formulas as `lf_nu` and `hf_nu`.
- `peak_freq_vlf`, `peak_freq_lf`, `peak_freq_hf`: peak PSD location inside the same inclusive band masks.

Metrics expected to change under non-overlapping Candidate A:

- Always definition-affected: `ulf_power`, `vlf_power`, `ulf_power_nu`, `vlf_power_nu`, `peak_freq_vlf`.
- Potentially affected by exact boundary bins: `lf_power`, `lf_power_nu`, `lf_hf_ratio`, `lf_nu`, `hf_nu`, `relative_lf_power`, `relative_hf_power`, `peak_freq_lf`.
- Usually unaffected unless boundary bins exist: `hf_power`, `hf_power_nu`.
- `total_power` is unchanged in Candidate A as defined here because it remains direct integration over `0.0 <= f <= 0.4`.

Metrics expected to change under Candidate B:

- All Candidate A changes.
- `total_power` and all percent-of-total metrics can change because Candidate B integrates total power over `0.0 < f <= 0.4`, excluding the DC/first segment contribution.

## Task 3: Project Documentation and Intended Convention

Repository evidence does not show an intentional cumulative VLF definition beyond documenting current implementation:

- `validation/research_notes/final_master_validation_report/hrv_studio_system_description.md` documents ULF `0.0-0.003` and VLF `0.0-0.04`, and states total power is `0.0-0.4`.
- Manual inspection reports explicitly warn: `ULF and VLF currently overlap; definitions are reported unchanged.`
- `validation/README_validation.md` and multiple research notes identify DC / first VLF bins as convention-sensitive for VLF and total power.
- `tests/test_freq_domain.py` contains tests expecting the current result structure and default bands, including ULF and VLF outputs. It does not establish a physiological rationale for cumulative VLF.
- Local validation scripts for NeuroKit2 and duration sensitivity import `HRVFreqDomainAnalysis.DEFAULT_FREQ_BANDS`, so they mirror the production overlap rather than serving as independent convention references.

The project convention currently documented is therefore descriptive of implementation, not clear evidence that overlapping ULF/VLF was intentionally chosen as the target physiological convention.

## Task 4: Isolated Candidate Definitions

Candidate A:

- ULF: `0 <= f < 0.003`.
- VLF: `0.003 <= f < 0.04`.
- LF: `0.04 <= f < 0.15`.
- HF: `0.15 <= f <= 0.40`.
- Total: direct trapezoidal integration over `0 <= f <= 0.40`; this preserves current total-power convention while removing low-band overlap.

Candidate B:

- ULF: `0 < f < 0.003`.
- VLF/LF/HF: same as Candidate A.
- Total: direct trapezoidal integration over `0 < f <= 0.40`; this excludes the exact DC bin from physiological power calculations.

Candidate calculations are isolated in `tools/frequency_band_definition_audit.py` and do not modify production behavior.

## Task 5: Targeted Empirical Impact

Sample composition:

| sample_group                  | n_samples |
| ----------------------------- | --------- |
| dc_vlf_sensitive_none         | 7         |
| duration_truncated_linear     | 5         |
| kubios_44_selected_none       | 44        |
| normal_physionet_10min_linear | 8         |
| smoothness_priors_10          | 10        |

Impact summary across Welch, FFT, and AR PSDs:

| candidate   | metric       | n   | median_abs_change | median_relative_change_pct | max_abs_change | max_relative_change_pct | materially_affected_n | materially_affected_pct |
| ----------- | ------------ | --- | ----------------- | -------------------------- | -------------- | ----------------------- | --------------------- | ----------------------- |
| candidate_a | ulf_power    | 222 | 0.000             | 0.000                      | 0.000          | 0.000                   | 0                     | 0.000                   |
| candidate_a | vlf_power    | 222 | 202.922           | 26.169                     | 655766.574     | 100.000                 | 207                   | 93.243                  |
| candidate_a | lf_power     | 222 | 0.000             | 0.000                      | 152.289        | 11.914                  | 59                    | 26.577                  |
| candidate_a | hf_power     | 222 | 0.000             | 0.000                      | 0.000          | 0.000                   | 0                     | 0.000                   |
| candidate_a | total_power  | 222 | 0.000             | 0.000                      | 0.000          | 0.000                   | 0                     | 0.000                   |
| candidate_a | lf_hf_ratio  | 222 | 0.000             | 0.000                      | 0.312          | 11.914                  | 59                    | 26.577                  |
| candidate_a | lf_nu        | 222 | 0.000             | 0.000                      | 2.922          | 7.797                   | 23                    | 10.360                  |
| candidate_a | hf_nu        | 222 | 0.000             | 0.000                      | 2.922          | 8.966                   | 43                    | 19.369                  |
| candidate_a | vlf_power_nu | 222 | 13.460            | 26.169                     | 99.981         | 100.000                 | 207                   | 93.243                  |
| candidate_a | lf_power_nu  | 222 | 0.000             | 0.000                      | 5.467          | 11.914                  | 59                    | 26.577                  |
| candidate_a | hf_power_nu  | 222 | 0.000             | 0.000                      | 0.000          | 0.000                   | 0                     | 0.000                   |
| candidate_b | ulf_power    | 222 | 2.204             | 17.797                     | 491766.989     | 100.000                 | 129                   | 58.108                  |
| candidate_b | vlf_power    | 222 | 202.922           | 26.169                     | 655766.574     | 100.000                 | 207                   | 93.243                  |
| candidate_b | lf_power     | 222 | 0.000             | 0.000                      | 152.289        | 11.914                  | 59                    | 26.577                  |
| candidate_b | hf_power     | 222 | 0.000             | 0.000                      | 0.000          | 0.000                   | 0                     | 0.000                   |
| candidate_b | total_power  | 222 | 116.596           | 6.901                      | 491766.989     | 75.339                  | 159                   | 71.622                  |
| candidate_b | lf_hf_ratio  | 222 | 0.000             | 0.000                      | 0.312          | 11.914                  | 59                    | 26.577                  |
| candidate_b | lf_nu        | 222 | 0.000             | 0.000                      | 2.922          | 7.797                   | 23                    | 10.360                  |
| candidate_b | hf_nu        | 222 | 0.000             | 0.000                      | 2.922          | 8.966                   | 43                    | 19.369                  |
| candidate_b | vlf_power_nu | 222 | 7.378             | 15.263                     | 99.937         | 100.000                 | 198                   | 89.189                  |
| candidate_b | lf_power_nu  | 222 | 0.389             | 6.192                      | 14.211         | 305.503                 | 165                   | 74.324                  |
| candidate_b | hf_power_nu  | 222 | 0.227             | 7.412                      | 13.919         | 305.503                 | 159                   | 71.622                  |

No-detrend material-effect fractions for VLF/total-power groups:

| candidate   | sample_group            | metric      | materially_affected |
| ----------- | ----------------------- | ----------- | ------------------- |
| candidate_a | dc_vlf_sensitive_none   | total_power | 0.000               |
| candidate_a | dc_vlf_sensitive_none   | vlf_power   | 1.000               |
| candidate_a | kubios_44_selected_none | total_power | 0.000               |
| candidate_a | kubios_44_selected_none | vlf_power   | 1.000               |
| candidate_b | dc_vlf_sensitive_none   | total_power | 1.000               |
| candidate_b | dc_vlf_sensitive_none   | vlf_power   | 1.000               |
| candidate_b | kubios_44_selected_none | total_power | 0.803               |
| candidate_b | kubios_44_selected_none | vlf_power   | 1.000               |

Detailed row-level impact is saved in `validation/research_notes/frequency_band_definition_impact.csv`.

Interpretation of observed changes:

- Candidate A materially changes VLF-related outputs because VLF no longer includes the DC/ULF region.
- Candidate A leaves direct total power unchanged by definition.
- Candidate B materially changes total power in records where the DC-to-first-positive-frequency trapezoid contributes appreciable area.
- LF/HF, LFnu, and HFnu are not affected by VLF removal, but they can change when exact LF/HF boundary bins are present, especially the `0.15 Hz` Welch bin.
- No-detrend and short-duration/VLF-sensitive cases are the most likely to show large VLF or total-power changes.

## Task 6: Manuscript Experiment Impact if Corrected

If production adopts non-overlapping bands and/or DC exclusion, affected experiments:

| Experiment | Classification | Reason |
| ---------- | -------------- | ------ |
| Section 3.1 large-scale NeuroKit2 agreement | must rerun | Frequency metrics and the local NeuroKit2 comparator integration use the current overlapping bands; VLF/total and boundary-sensitive LF metrics would change. |
| Section 3.2 sequence-harmonized Kubios benchmark | must rerun | The benchmark reports VLF, LF, HF, total power, LF/HF, LFnu, HFnu; corrected HRV Studio values and comparator-definition interpretation must be regenerated. |
| Section 3.3 Smoothness Priors sensitivity | must rerun if manuscript reports frequency values after code correction | FFT and AR summaries use the shared integration function; AR/FFT VLF, total, and boundary-sensitive LF-derived metrics could change. |
| Alternative spectral-method comparison | must rerun | It directly compares Welch/FFT/AR frequency metrics from the shared integration function. |
| Synthetic robustness testing | should rerun | Frequency-domain robustness metrics and figures include LF, HF, total power, LF/HF, and related values. |
| Duration sensitivity analysis | must rerun | The script imports current band definitions and VLF/duration conclusions depend on low-frequency integration. |
| MIT-BIH engineering/QC stress test | should rerun | Any frequency-domain QC/stress outputs would change; time-domain-only results are unaffected. |
| Figures/tables containing frequency-domain results | must regenerate | Figures 3-8/source tables with frequency-domain metrics, final validation tables, Kubios Figure 4, duration Figure 7, spectral-method Figure 5, and any VLF/total-power tables need review/regeneration. |

Time-domain and nonlinear-only validations are unaffected.

## Task 7: Comparator Conventions from Local Evidence

NeuroKit2 local validation:

- The project does not primarily use NeuroKit2's own `hrv_frequency()` band outputs in the large-scale comparator scripts.
- `tools/validate_freq_domain_neurokit2.py`, `tools/duration_sensitivity_validation.py`, and `tools/validate_fft_ar_methods.py` compute NeuroKit2 PSDs and then integrate them with `HRVFreqDomainAnalysis.DEFAULT_FREQ_BANDS`.
- Therefore local NeuroKit2 agreement numbers use the same overlapping VLF definition and inclusive masks as HRV Studio. They cannot independently validate the correctness of the band definitions.

Kubios local exports:

- Kubios text exports visibly report bands as VLF `0-0.04 Hz`, LF `0.04-0.15 Hz`, HF `0.15-0.4 Hz`; no ULF output is exposed in these exports.
- The MAT PSD cross-check supports inclusive VLF/total more strongly than DC-excluded or non-overlapping alternatives for Kubios exports. Median relative difference for inclusive VLF was `0.847%`; median relative difference for inclusive total power was `0.068%`.
- Excluding DC or using non-overlapping VLF did not reproduce Kubios VLF/total as closely in the checked exports: median relative difference for non-overlapping VLF was `13.874%`, and median relative difference for DC-excluded total power was `6.411%`.
- The exported PSD arrays did not exactly reproduce every stored Kubios band power with simple trapezoidal masks, especially LF in some records, so this is evidence about the low-boundary/DC convention rather than proof of Kubios' complete internal integration implementation.

Kubios evidence therefore supports that the exported Kubios VLF power corresponds to `0-0.04 Hz` integration of its exported PSD, including the lowest/DC endpoint as represented in that PSD. It does not establish that HRV Studio should expose overlapping ULF and VLF simultaneously.

Kubios PSD convention details are saved in `validation/research_notes/frequency_band_definition_kubios_psd_check.csv`.

## Task 8: Engineering Recommendation

Current VLF definition is demonstrably overlapping with ULF and is likely incorrect if HRV Studio intends standard mutually exclusive physiological bands. It is, however, partially aligned with Kubios' exported VLF label of `0-0.04 Hz`, so comparator-specific reporting must be explicit.

Recommended production direction, pending team approval:

- Keep ULF as a separate output only for sufficiently long recordings, with duration warnings; do not include ULF inside VLF when both are reported as separate bands.
- Use non-overlapping masks:
  - ULF: `0 < f < 0.003 Hz` or `0 <= f < 0.003 Hz` only if DC is intentionally retained for diagnostic total power.
  - VLF: `0.003 <= f < 0.04 Hz`.
  - LF: `0.04 <= f < 0.15 Hz`.
  - HF: `0.15 <= f <= 0.40 Hz`.
- Exclude the exact DC bin from physiological band-power reporting unless a comparator-specific mode explicitly requires including it.
- Define total physiological power as direct integration over `0 < f <= 0.40 Hz` for short-term HRV frequency metrics, or as the sum/integral over the selected non-overlapping physiological bands. Use one convention consistently and document it.
- Preserve a comparator-specific diagnostic mode if Kubios `0-0.04` VLF reproduction remains required.

Derived metrics affected by this recommendation:

- Directly: ULF, VLF, total power, band percent-of-total metrics, VLF peak frequency.
- Potentially by boundary handling: LF, LF/HF, LFnu, HFnu, relative LF/HF power.
- Usually not affected by VLF-only correction: HF absolute power, unless boundary conventions change.

Validation rerun scope would be broad: every manuscript frequency-domain validation table/figure and comparator summary should be regenerated after any production change. Time-domain and nonlinear validations do not need rerun for this issue.

## Generated Files

- `validation/research_notes/frequency_band_definition_audit.md`
- `validation/research_notes/frequency_band_definition_impact.csv`
- `validation/research_notes/frequency_band_definition_kubios_psd_check.csv`

## Commands Used

```powershell
python tools/frequency_band_definition_audit.py
```
