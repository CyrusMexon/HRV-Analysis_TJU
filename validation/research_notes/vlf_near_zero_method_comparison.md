# VLF Near-Zero Method Comparison

This note compares the near-zero-frequency and VLF behavior observed in representative files from `v04_physionet_10min_neurokit2`.

Files reviewed:

- `validation/processed_data/physionet_nsr_rr_10min/nsr023_segment_131.csv`
- `validation/processed_data/physionet_nsr_rr_10min/nsr037_segment_131.csv`
- `validation/processed_data/physionet_nsr_rr_10min/nsr043_segment_135.csv`
- `validation/processed_data/physionet_nsr_rr_10min/nsr022_segment_022.csv`

This is a validation-method note only. It does not propose a production HRV code change.

## Short Conclusion

The most likely implementation-level cause is different detrending at the Welch segment level.

HRV Studio computes Welch PSD with SciPy `signal.welch(..., detrend="linear")` for the `--detrend-method linear` run. SciPy applies that detrending independently inside each Welch segment. In contrast, the NeuroKit2 validation path globally detrends the full interpolated RR series first, then calls NeuroKit2 `signal_psd()`, which removes the global mean and calls SciPy Welch with `detrend=False`.

That difference is highly concentrated at DC and the first few VLF bins. It raises VLF and total_power strongly while leaving LF, HF, and LF/HF nearly unchanged.

## Code-Level Comparison

### HRV Studio Welch Path

In `hrvlib/metrics/freq_domain.py`, HRV Studio:

- interpolates RR intervals to a 4 Hz uniformly sampled RR signal in seconds;
- clips the resampled RR signal to the configured physiologic range;
- for `detrend_method="linear"`, passes the original resampled signal into SciPy Welch with `detrend="linear"`;
- uses `scaling="density"` and `average="mean"`;
- does not explicitly set `nfft`, so SciPy uses `nfft=nperseg`;
- converts PSD from seconds squared per Hz to milliseconds squared per Hz by multiplying by `1e6`;
- computes VLF with `freqs >= 0.0` and `freqs <= 0.04`;
- computes total_power over `0.0 <= f <= 0.4`.

Relevant local code:

- `hrvlib/metrics/freq_domain.py`: `_compute_welch_psd()`
- `hrvlib/metrics/freq_domain.py`: `_compute_spectral_metrics()`

### NeuroKit2 PSD Path

In installed NeuroKit2 `0.2.12`:

- `neurokit2.signal.signal_psd()` first subtracts the global mean: `signal = signal - np.mean(signal)`;
- `_signal_psd_welch()` sets `nfft = int(nperseg * 2)` when `nperseg` is provided;
- `_signal_psd_welch()` calls SciPy Welch with `scaling="density"`, `detrend=False`, `average="mean"`, `nperseg=nperseg`, and `window=window_type`;
- one-sided PSD handling is SciPy's default behavior, not a custom NeuroKit2 normalization step;
- the validation comparator calls `signal_psd(..., normalize=False)`, so NeuroKit2's optional max-power normalization is disabled.

Relevant local installed source:

- `.venv/Lib/site-packages/neurokit2/signal/signal_psd.py`
- `.venv/Lib/site-packages/neurokit2/hrv/hrv_frequency.py`
- `.venv/Lib/site-packages/neurokit2/signal/signal_power.py`

### NeuroKit2 HRV Band Defaults

Official `neurokit2.hrv.hrv_frequency()` defaults differ from the current validation comparator:

- ULF: `0.0-0.0033 Hz`
- VLF: `0.0033-0.04 Hz`
- LF: `0.04-0.15 Hz`
- HF: `0.15-0.4 Hz`
- VHF: `0.4-0.5 Hz`
- TP is the sum of returned band powers.

The current validation comparator intentionally uses HRV Studio's band definition for comparison:

- VLF: `0.0-0.04 Hz`
- total_power: trapezoidal integration over `0.0-0.4 Hz`

Changing VLF from `0.0-0.04` to `0.0033-0.04` reduces NeuroKit2 VLF, but it does not remove the discrepancy. The first nonzero VLF bins remain much larger under the NeuroKit2-style Welch path.

## Representative Evidence

The manual inspection summary found the same pattern in all four files:

| File | VLF native / NeuroKit2 | LF native / NeuroKit2 | HF native / NeuroKit2 | LF/HF native / NeuroKit2 | Near-zero concentration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nsr023_segment_131.csv` | 322.985 / 6503.400 | 107.288 / 104.919 | 66.400 / 64.476 | 1.616 / 1.627 | 99.7% |
| `nsr037_segment_131.csv` | 91.849 / 1780.004 | 39.481 / 39.644 | 23.794 / 23.842 | 1.659 / 1.663 | 99.6% |
| `nsr043_segment_135.csv` | 56.250 / 1237.959 | 112.235 / 112.086 | 222.217 / 221.954 | 0.505 / 0.505 | 99.6% |
| `nsr022_segment_022.csv` | 869.708 / 12714.802 | 225.399 / 225.338 | 256.421 / 256.832 | 0.879 / 0.877 | 99.5% |

For each file, resampled-signal correlation and detrended-signal correlation were effectively `1.000`. This argues against interpolation mismatch as the dominant explanation for these representative cases.

## Targeted Variant Check

I ran a validation-only variant table using the same representative files. The key variants were:

- `hrv_current_segment_linear`: HRV Studio-style SciPy Welch, `nfft=nperseg`, `detrend="linear"`.
- `hrv_segment_linear_nfft2`: same as HRV Studio, but with `nfft=2*nperseg`.
- `global_linear_then_no_segment_detrend_nfft2`: globally linearly detrend the full signal once, then SciPy Welch with `detrend=False` and `nfft=2*nperseg`.
- `neurokit2_validation_path`: current validation NeuroKit2 PSD path.

### VLF Results

| File | HRV current | HRV `nfft x2` only | Global linear + no segment detrend + `nfft x2` | NeuroKit2 validation |
| --- | ---: | ---: | ---: | ---: |
| `nsr023_segment_131.csv` | 322.876 | 346.385 | 6758.860 | 6503.400 |
| `nsr037_segment_131.csv` | 91.795 | 100.996 | 1779.960 | 1780.004 |
| `nsr043_segment_135.csv` | 56.230 | 63.186 | 1237.550 | 1237.959 |
| `nsr022_segment_022.csv` | 869.097 | 960.119 | 12714.400 | 12714.802 |

This is the strongest evidence in the investigation. Doubling `nfft` alone barely changes VLF. Switching from segment-wise linear detrending to global linear detrending with no segment detrending reproduces NeuroKit2-level VLF.

### LF/HF Stability In The Same Variants

LF/HF remains close in the matching variant:

| File | HRV current LF/HF | Global linear + no segment detrend LF/HF | NeuroKit2 validation LF/HF |
| --- | ---: | ---: | ---: |
| `nsr023_segment_131.csv` | 1.616 | 1.616 | 1.627 |
| `nsr037_segment_131.csv` | 1.659 | 1.663 | 1.663 |
| `nsr043_segment_135.csv` | 0.505 | 0.505 | 0.505 |
| `nsr022_segment_022.csv` | 0.879 | 0.877 | 0.877 |

This supports the interpretation that the difference is localized to near-zero/VLF leakage rather than a broad PSD scaling problem.

## Candidate Causes Assessed

### Detrending Behavior

This is the leading cause.

HRV Studio's `detrend="linear"` is applied by SciPy inside each Welch segment. NeuroKit2-style validation detrending is applied once to the full interpolated RR signal, after which Welch receives `detrend=False`.

Segment-wise linear detrending removes local offsets and slopes from every 120-second Welch window. That directly suppresses DC and the first very-low-frequency bins. Global detrending does not remove each segment's local offset/slope, so those slow segment-level components remain and leak into DC and the first nonzero VLF bins.

### DC / Mean Removal

NeuroKit2 `signal_psd()` subtracts the global mean before Welch. The validation comparator also globally linearly detrends before calling NeuroKit2, so the additional mean subtraction should be small after global linear detrending.

However, global mean removal is not equivalent to per-segment detrending. Segment-wise residual means and slopes can remain after a full-record detrend and are enough to inflate the near-zero bins.

### Welch PSD Settings

Both paths use:

- SciPy Welch;
- Hann window;
- `nperseg=480`;
- `noverlap=360`;
- `scaling="density"`;
- mean averaging;
- one-sided PSD by SciPy default.

The material Welch difference is `detrend="linear"` per segment versus `detrend=False` after full-record detrending.

### nfft

NeuroKit2 uses `nfft=2*nperseg` when `nperseg` is provided. HRV Studio currently leaves `nfft` unset, effectively `nfft=nperseg`.

This changes the frequency grid and VLF bin count:

- HRV Studio: 5 bins in `0.0-0.04 Hz`.
- NeuroKit2: 10 bins in `0.0-0.04 Hz`.

But the variant check shows that `nfft x2` alone does not explain the VLF gap. Native `nfft x2` VLF remains near HRV Studio values, not NeuroKit2 values.

### Scaling And One-Sided PSD Normalization

A global scaling mismatch is unlikely.

Both sides use SciPy Welch `scaling="density"` and default one-sided output. LF and HF are nearly identical, which would not happen under a broad multiplicative scaling error affecting the full spectrum.

### DC Bin Inclusion / Exclusion

The PSD includes a DC bin in both paths. With trapezoidal integration, a band containing only the DC bin contributes zero area by itself. But the interval from DC to the first nonzero frequency bin can strongly affect VLF when the VLF band starts at `0.0 Hz`.

Excluding DC or starting VLF at `0.0033 Hz` reduces VLF but does not eliminate the discrepancy. For example, NeuroKit2 VLF in the representative files remains large under `0.0033-0.04 Hz`:

| File | NeuroKit2 VLF `0.0-0.04` | NeuroKit2 VLF `0.0033-0.04` |
| --- | ---: | ---: |
| `nsr023_segment_131.csv` | 6503.400 | 3685.840 |
| `nsr037_segment_131.csv` | 1780.004 | 1028.770 |
| `nsr043_segment_135.csv` | 1237.959 | 710.973 |
| `nsr022_segment_022.csv` | 12714.802 | 7590.112 |

So DC-adjacent integration contributes substantially, but the larger issue is the PSD amplitude in the first VLF bins.

### VLF Lower Bound

There are two definitions in play:

- HRV Studio and current validation comparator: VLF starts at `0.0 Hz`.
- Official NeuroKit2 HRV defaults: VLF starts at `0.0033 Hz`, with ULF covering `0.0-0.0033 Hz`.

This boundary difference is important for reporting and comparability, but it is not sufficient to explain the observed discrepancy. The first nonzero VLF bins remain elevated in the NeuroKit2-style PSD.

### Total Power

HRV Studio and the current validation comparator compute total_power as integrated PSD over `0.0-0.4 Hz`.

Official NeuroKit2 `hrv_frequency()` computes TP as the sum of the returned band powers. Its default bands include ULF, VLF, LF, HF, and VHF. In this validation work, `max_frequency` is limited to `0.4 Hz`, and the comparator's total_power is not official NeuroKit2 TP; it is a matched `0.0-0.4 Hz` integration from the NeuroKit2 PSD.

The total_power discrepancy is therefore mostly VLF-driven in these files.

## Most Likely Cause

The most likely implementation-level cause is that HRV Studio removes a linear trend independently from each Welch segment, while the NeuroKit2 validation path removes a trend only at the full-record level and then disables segment detrending inside Welch.

This explains all major observations:

- VLF and total_power differ strongly.
- LF, HF, and LF/HF remain nearly identical.
- Resampled and detrended signal correlations are effectively perfect.
- `nfft x2` alone does not reproduce NeuroKit2 VLF.
- Global-linear detrend followed by `detrend=False` does reproduce NeuroKit2 VLF.
- The discrepancy is concentrated at DC and the first VLF bins, where segment offsets/slopes have the largest spectral effect.

## Proposed Validation-Only Experiment

Run a focused A/B comparator on these files, using the exact same interpolated RR signal and exact same Welch parameters, and change only the segment detrending mode:

- Arm A: `signal.welch(x=global_linear_detrended_rr, nperseg=480, noverlap=360, nfft=960, detrend=False, scaling="density", average="mean")`
- Arm B: `signal.welch(x=global_linear_detrended_rr, nperseg=480, noverlap=360, nfft=960, detrend="linear", scaling="density", average="mean")`

Expected result:

- Arm A should reproduce the NeuroKit2 validation VLF values.
- Arm B should collapse VLF toward HRV Studio / native `nfft x2` values.
- LF, HF, and LF/HF should remain nearly unchanged.

This experiment should be implemented only in validation tooling or a notebook/script under `tools/` or `validation/`, not in production HRV code. If confirmed, the next methodological question is not "which output is correct" but which detrending convention HRV Studio intends to validate against: per-segment Welch detrending, NeuroKit2-style global detrending, or an explicitly documented third convention.
