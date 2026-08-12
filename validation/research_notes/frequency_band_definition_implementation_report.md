# Frequency Band Definition Implementation Report

## Scope

Implemented the approved production correction for HRV Studio frequency-band integration. Manuscript files were not modified, existing validation outputs were not regenerated, and no commit was made.

## Files Changed

- `hrvlib/metrics/freq_domain.py`
- `tests/test_freq_domain.py`
- `tools/frequency_domain_sequence_audit.py`
- `tools/run_kubios_smoothness_priors_pilot.py`
- `tools/validate_freq_domain_neurokit2.py`
- `tools/validate_fft_ar_methods.py`
- `tools/duration_sensitivity_validation.py`

## Production Convention

Default convention: `standard`.

Standard bands:

- ULF: `0 < f < 0.003 Hz`
- VLF: `0.003 <= f < 0.04 Hz`
- LF: `0.04 <= f < 0.15 Hz`
- HF: `0.15 <= f <= 0.40 Hz`
- Total physiological power: `0 < f <= 0.40 Hz`

Boundary behavior:

- DC / 0 Hz is excluded from standard ULF, VLF, and total power.
- `0.003 Hz` belongs to VLF, not ULF.
- `0.04 Hz` belongs to LF, not VLF.
- `0.15 Hz` belongs to HF, not LF.
- Adjacent standard bands are mutually exclusive.

## Kubios-Compatible Comparator Convention

Explicit convention: `kubios_compatible`.

Kubios-compatible bands:

- ULF: `0 <= f <= 0.003 Hz` retained only as a schema-compatible diagnostic output.
- VLF: `0 <= f <= 0.04 Hz`
- LF: `0.04 <= f <= 0.15 Hz`
- HF: `0.15 <= f <= 0.40 Hz`
- Total power: `0 <= f <= 0.40 Hz`

This mode is documented in diagnostics as comparator-only. ULF overlaps VLF in this mode and should not be interpreted as a separate disjoint physiological band.

## Implementation Details

- Added `band_convention` to `HRVFreqDomainAnalysis`, defaulting to `standard`.
- Added centralized convention specifications and mask helpers.
- Routed Welch, FFT, and AR metric calculations through the same selected-convention masks in `_compute_spectral_metrics()`.
- Routed peak-frequency lookup and frequency diagnostics through the same masks.
- Added convention metadata to `analysis_info` and `frequency_diagnostics`.
- Added public class mask helpers for validation utilities that integrate comparator PSD arrays outside the analyzer.
- Updated Kubios-specific validation runners to request `band_convention="kubios_compatible"` explicitly.
- Updated NeuroKit2, FFT/AR, and duration validation utilities to use the same convention-aware masks for independent PSD integrations.
- Added explicit short-recording ULF duration warning text.

## Derived Metrics

Derived formulas were preserved:

- `lf_hf_ratio = LF / HF`
- `lf_nu = LF / (LF + HF) * 100`
- `hf_nu = HF / (LF + HF) * 100`
- `relative_lf_power` and `relative_hf_power` remain identical to `lf_nu` and `hf_nu`.
- Band `_power_nu` outputs remain percent-of-total outputs and now use the selected convention's total-power definition.

## Tests Added

Added focused tests covering:

- Standard ULF/VLF/LF/HF non-overlap.
- Standard DC exclusion.
- `0.04 Hz` assigned only to LF in standard mode.
- `0.15 Hz` assigned only to HF in standard mode.
- No adjacent standard-mode boundary double counting.
- Standard total power excludes DC.
- Kubios-compatible VLF and total retain DC.
- Welch, FFT, and AR all respect the selected convention.
- LF/HF, LFnu, HFnu, and relative LF/HF identities.
- Existing output schema plus added convention metadata.

## Verification

Commands run:

```powershell
python -m pytest tests\test_freq_domain.py -q
python -m pytest tests\test_freq_domain.py tests\test_smoothness_priors.py -q
python -m pytest tests\test_parse_kubios_exports_methods.py -q
python -m py_compile hrvlib\metrics\freq_domain.py tools\frequency_domain_sequence_audit.py tools\run_kubios_smoothness_priors_pilot.py tools\validate_freq_domain_neurokit2.py tools\validate_fft_ar_methods.py tools\duration_sensitivity_validation.py
python -m pytest tests\test_freq_domain.py tests\test_smoothness_priors.py tests\test_parse_kubios_exports_methods.py -q
```

Results:

- `tests/test_freq_domain.py`: 40 passed.
- `tests/test_freq_domain.py tests/test_smoothness_priors.py`: 60 passed.
- `tests/test_parse_kubios_exports_methods.py`: 1 passed.
- Final combined focused test run: 61 passed.
- `py_compile`: passed.

Warnings observed were expected existing short-recording / no-bin warnings, including standard-mode ULF warnings when no PSD bins fall in `0 < f < 0.003 Hz`.

## Smoke Test

A synthetic RR sequence with LF and HF modulation was analyzed under both `standard` and `kubios_compatible` modes. Welch, FFT, and AR all returned finite total power and finite LF/HF, and `analysis_info.band_convention` matched the requested convention.

Boundary mask smoke check on exact `[0, 0.003, 0.04, 0.15, 0.40] Hz` frequencies:

- Standard VLF mask: `[0, 1, 0, 0, 0]`
- Standard LF mask: `[0, 0, 1, 0, 0]`
- Standard HF mask: `[0, 0, 0, 1, 1]`
- Standard total mask: `[0, 1, 1, 1, 1]`
- Kubios-compatible VLF mask: `[1, 1, 1, 0, 0]`
- Kubios-compatible total mask: `[1, 1, 1, 1, 1]`

## Backward Compatibility

Existing stored validation artifacts were not changed. New normal HRV Studio analyses will use standard mutually exclusive bands by default, so frequency-domain outputs involving ULF, VLF, total power, and percent-of-total values will differ from legacy outputs.

Direct Kubios comparator reruns remain supported by explicit `band_convention="kubios_compatible"`. The Kubios frequency sequence audit and Smoothness Priors pilot scripts were updated to use that mode explicitly.

`DEFAULT_FREQ_BANDS` remains as broad low/high tuple metadata for compatibility, but tuple bounds cannot encode inclusive/exclusive endpoint rules. New code should use the convention mask helpers instead of reconstructing masks from tuples.

## Unexpected Behavior

No unexpected production behavior was observed in focused tests or smoke checks. The intended default change is behavior-changing for standard HRV Studio frequency powers.

## Validation Rerun Readiness

It is safe to proceed to validation reruns from an implementation standpoint. Reruns should be planned as new outputs because default standard-mode values will differ from prior stored artifacts. Kubios direct-comparator reruns should continue to use `kubios_compatible` mode and should label that convention explicitly.
