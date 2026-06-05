# v06 none/current Native Frequency-Domain Investigation

Scope: validation-only investigation of HRV Studio native frequency-domain computation under `detrend_method=none` and `welch_detrend_mode=current`. No production HRV code was modified.

## Short Answer

The current evidence points to two issues rather than one uniform PSD scaling problem.

- CH001 and OUT001: the native Welch computation appears to include very large DC / first-near-zero content when `detrend_method=None` maps to `scipy.signal.welch(detrend=False)`. This inflates VLF and total power while LF, HF, LF/HF, and normalized powers remain comparatively close to Kubios/NeuroKit2.
- VLF001: native Welch PSD contains non-finite values, so `total_power` becomes non-finite and `_compute_spectral_metrics()` returns zero/default band powers for VLF/LF/HF. This is a separate robustness problem in the no-detrend path or its preprocessing inputs.
- Cautious classification: likely a real HRV Studio behavior/bug under `detrend_method=none`, centered on mean/DC handling and non-finite PSD handling. It does not look like a simple global PSD scaling or one-sided FFT normalization error for the main CH001/OUT001 symptom.

## Key Metrics

| File | Kubios VLF | Native VLF | NeuroKit2 VLF | Native mean-removed VLF | Native VLF err % | Native total | NeuroKit2 total | Native DC PSD | Native PSD NaNs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH001 | 1105.886 | 286060.289 | 1082.275 | 953.127 | 25767.07 | 286860.140 | 1847.858 | 34188677.24 | 0 |
| OUT001 | 2918.690 | 272254.148 | 2751.499 | 2262.981 | 9227.96 | 272593.109 | 3075.015 | 32562994.41 | 0 |
| VLF001 | 1946.109 | 0.000 | 476.759 | 0.000 | 100.00 | not available | 2894.022 | not available | 241 |

## Checks Against Requested Hypotheses

- PSD scaling / FFT normalization / one-sided PSD scaling: not the leading explanation for CH001 and OUT001. LF/HF shape and non-VLF bands are close enough that a uniform scaling error is unlikely. The native FFT diagnostic does show variance inconsistency when no detrending is used, but the reported validation metric is Welch, not FFT.
- Integration of DC bin: strongly implicated for CH001 and OUT001. The native VLF band starts at 0.0 Hz and the no-detrend Welch call preserves the large mean RR level, producing extreme DC/near-zero density.
- VLF band mask: native VLF is `(freqs >= 0.0) & (freqs <= 0.04)`, so DC is included. ULF is also defined as `0.0-0.003`, which overlaps VLF, but the immediate v06 symptom is VLF/total inflation rather than ULF reporting.
- total_power integration: native total power uses the same lower bound as VLF, `0.0-0.4 Hz`, so any DC/near-zero inflation is carried directly into total power.
- `detrend_method=none` preprocessing: the native path bypasses both global and segment detrending. For Welch it passes `detrend=False`, which is expected from the code but probably not comparable to Kubios/NeuroKit2 output when the mean RR level is retained.
- Extreme DC inflation: present for CH001 and OUT001. The plots and first-bin tables show the native PSD dominated by the first bins.
- VLF001 zeros: the native PSD contains non-finite values. `_compute_spectral_metrics()` does not reject non-finite total power before continuing; NaN band integrations are then clamped/defaulted to zeros for VLF/LF/HF and normalized powers.
- `current` mode: in v06, `current` preserves NeuroKit2's existing `signal_psd()` behavior. It does not mimic HRV Studio native `detrend=False`; therefore `current` is expected to behave differently from native none-mode near DC.

## Per-File Details

### CH001

- Plots: `validation\research_notes\v06_none_mode_investigation_plots\CH001\psd_overlay.png`, `validation\research_notes\v06_none_mode_investigation_plots\CH001\zoomed_vlf.png`, `validation\research_notes\v06_none_mode_investigation_plots\CH001\cumulative_power_by_frequency.png`
- RR count: 922; native resampled samples: 2401; native time-domain NaN/non-finite count: 0.
- RR minimum: 531.000 ms; nonpositive RR count: 0; duplicate RR start-time count: 0.
- Native time-domain mean/std: 654.500 / 44.671 ms.
- Welch parameters native/NeuroKit2: nperseg `480` / `480`, noverlap `360` / `360`.
- Native VLF with DC: 286060.289; excluding DC: 71954.842; mean-removed diagnostic: 953.127; constant-detrend diagnostic: 582.516.
- Native total with DC: 286860.140; excluding DC: 72754.693; mean-removed diagnostic: 1753.050.

First native VLF bins:

| Index | Frequency | Native PSD | NeuroKit2 PSD | Mean-removed native PSD |
| --- | --- | --- | --- | --- |
| 0 | 0.000000 | 34188677.245 | 45513.651 | 45514.327 |
| 1 | 0.008333 | 17196630.071 | 80526.749 | 55353.490 |
| 2 | 0.016667 | 18435.376 | 55352.695 | 18408.631 |
| 3 | 0.025000 | 13688.708 | 31034.330 | 13723.431 |
| 4 | 0.033333 | 8283.822 | 18408.631 | 8265.067 |
| 5 | 0.041667 | 7352.780 | 15032.326 | 7357.201 |

### OUT001

- Plots: `validation\research_notes\v06_none_mode_investigation_plots\OUT001\psd_overlay.png`, `validation\research_notes\v06_none_mode_investigation_plots\OUT001\zoomed_vlf.png`, `validation\research_notes\v06_none_mode_investigation_plots\OUT001\cumulative_power_by_frequency.png`
- RR count: 931; native resampled samples: 2384; native time-domain NaN/non-finite count: 0.
- RR minimum: 320.000 ms; nonpositive RR count: 0; duplicate RR start-time count: 0.
- Native time-domain mean/std: 646.369 / 61.057 ms.
- Welch parameters native/NeuroKit2: nperseg `480` / `480`, noverlap `360` / `360`.
- Native VLF with DC: 272254.148; excluding DC: 68330.680; mean-removed diagnostic: 2262.981; constant-detrend diagnostic: 327.298.
- Native total with DC: 272593.109; excluding DC: 68669.642; mean-removed diagnostic: 2602.646.

First native VLF bins:

| Index | Frequency | Native PSD | NeuroKit2 PSD | Mean-removed native PSD |
| --- | --- | --- | --- | --- |
| 0 | 0.000000 | 32562994.408 | 226692.779 | 226681.711 |
| 1 | 0.008333 | 16378637.812 | 345305.741 | 147869.452 |
| 2 | 0.016667 | 4714.040 | 147875.411 | 4676.656 |
| 3 | 0.025000 | 3957.848 | 32508.165 | 3986.540 |
| 4 | 0.033333 | 3381.603 | 4676.656 | 3368.319 |
| 5 | 0.041667 | 6055.622 | 3347.364 | 6063.906 |

### VLF001

- Plots: `validation\research_notes\v06_none_mode_investigation_plots\VLF001\psd_overlay.png`, `validation\research_notes\v06_none_mode_investigation_plots\VLF001\zoomed_vlf.png`, `validation\research_notes\v06_none_mode_investigation_plots\VLF001\cumulative_power_by_frequency.png`
- RR count: 674; native resampled samples: 1064; native time-domain NaN/non-finite count: 1.
- RR minimum: 0.000 ms; nonpositive RR count: 245; duplicate RR start-time count: 245.
- Native time-domain mean/std: 615.269 / 74.581 ms.
- Welch parameters native/NeuroKit2: nperseg `480` / `480`, noverlap `360` / `360`.
- Native VLF with DC: not available; excluding DC: not available; mean-removed diagnostic: not available; constant-detrend diagnostic: not available.
- Native total with DC: not available; excluding DC: not available; mean-removed diagnostic: not available.

First native VLF bins:

| Index | Frequency | Native PSD | NeuroKit2 PSD | Mean-removed native PSD |
| --- | --- | --- | --- | --- |
| 0 |  |  | 20216.758 |  |
| 1 |  |  | 35245.222 |  |
| 2 |  |  | 24895.375 |  |
| 3 |  |  | 16106.475 |  |
| 4 |  |  | 9637.827 |  |
| 5 |  |  | 5781.804 |  |

## Cautious Conclusion

This looks more like `a) a real HRV Studio no-detrend behavior/bug` plus `c) a DC integration problem` than a pure settings mismatch. The settings are intentionally matched at a high level, but the implementations do not handle the mean/DC component the same way under `none/current`.

The strongest evidence is that removing only the mean from the native Welch input collapses CH001/OUT001 VLF and total power toward Kubios/NeuroKit2-like magnitudes, while LF/HF is already close. VLF001 should be treated separately: the native PSD becomes non-finite, so zeros are a downstream defaulting artifact rather than valid zero physiological power.

Recommended next code-review target before any fix: native `HRVFreqDomainAnalysis._compute_welch_psd()` and `_compute_spectral_metrics()` handling of `detrend_method=None`, finite-signal/finite-PSD validation, and whether total/VLF should include the DC bin for HRV band-power reporting.
