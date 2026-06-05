# None-Detrend Convention A/B

Validation-only experiment after the DC fix. Production code was not modified by this experiment.

## Arms

- Arm A: remove one global mean from the full interpolated RR signal, then run Welch with `detrend=False`.
- Arm B: run Welch with `detrend="constant"` per segment.
- Shared settings: `fs=4 Hz`, `nperseg=480`, `noverlap=360`, `window=hann`, `nfft=480`, `scaling=density`, `average=mean`.
- Bands: VLF `0.0-0.04`, LF `0.04-0.15`, HF `0.15-0.40`, total `0.0-0.40` Hz.

## Results

| File | Metric | Kubios | NeuroKit2 | Arm A global mean | Arm B segment constant | Arm A err % | Arm B err % | Closer to Kubios |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH001 | VLF | 1105.886 | 1082.275 | 952.849 | 582.115 | 13.84 | 47.36 | Arm A |
| CH001 | LF | 622.207 | 609.959 | 609.944 | 609.944 | 1.97 | 1.97 | Arm B |
| CH001 | HF | 124.494 | 125.343 | 124.773 | 124.773 | 0.22 | 0.22 | Arm A |
| CH001 | total_power | 1852.739 | 1847.858 | 1752.657 | 1381.923 | 5.40 | 25.41 | Arm A |
| CH001 | LF/HF | 4.998 | 4.866 | 4.888 | 4.888 | 2.19 | 2.19 | Arm B |
| OUT001 | VLF | 2918.690 | 2751.499 | 2264.679 | 326.535 | 22.41 | 88.81 | Arm A |
| OUT001 | LF | 292.129 | 250.597 | 248.814 | 248.812 | 14.83 | 14.83 | Arm A |
| OUT001 | HF | 49.115 | 50.870 | 50.810 | 50.810 | 3.45 | 3.45 | Arm B |
| OUT001 | total_power | 3260.106 | 3075.015 | 2603.582 | 665.437 | 20.14 | 79.59 | Arm A |
| OUT001 | LF/HF | 5.948 | 4.926 | 4.897 | 4.897 | 17.67 | 17.67 | Arm A |
| VLF001 | VLF | 1946.109 | 395.057 | 350.834 | 315.495 | 81.97 | 83.79 | Arm A |
| VLF001 | LF | 262.958 | 300.849 | 300.569 | 300.570 | 14.30 | 14.30 | Arm A |
| VLF001 | HF | 458.060 | 619.919 | 619.239 | 619.239 | 35.19 | 35.19 | Arm A |
| VLF001 | total_power | 2671.874 | 1333.814 | 1303.966 | 1268.625 | 51.20 | 52.52 | Arm A |
| VLF001 | LF/HF | 0.574 | 0.485 | 0.485 | 0.485 | 15.45 | 15.45 | Arm B |

## Mean Relative Error vs Kubios

- Arm A overall mean relative error: 20.02%
- Arm B overall mean relative error: 32.18%

| Metric | Arm A mean err % | Arm B mean err % | Lower mean error |
| --- | --- | --- | --- |
| VLF | 39.41 | 73.32 | Arm A |
| LF | 10.37 | 10.37 | Arm A |
| HF | 12.95 | 12.95 | Arm B |
| total_power | 25.58 | 52.51 | Arm A |
| LF/HF | 11.77 | 11.77 | Arm A |

## Input Diagnostics

| File | Invalid RR removed | Duplicate time points | Mean removed diagnostic flag | Nonfinite PSD |
| --- | --- | --- | --- | --- |
| CH001 | 0 | False | True | False |
| OUT001 | 0 | False | True | False |
| VLF001 | 245 | True | True | False |

## Cautious Recommendation

Arm A is modestly closer to Kubios overall in this n=3 pilot. It better preserves the low-frequency magnitude seen in Kubios/NeuroKit2 for CH001 and OUT001 than per-segment constant detrending.

This should be treated as a convention-finding result, not a general accuracy claim. The pilot is only three files, and VLF001 still has invalid RR intervals that require cleanup before either arm can be compared fairly. If Kubios-compatible no-detrend behavior is the target, this evidence favors documenting and considering Arm A as the closer convention for `detrend_method=None`, then validating on a larger manual-export subset before changing production behavior again.
