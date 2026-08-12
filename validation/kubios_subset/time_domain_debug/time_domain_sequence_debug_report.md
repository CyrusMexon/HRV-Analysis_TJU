# Kubios Time-Domain Sequence Debug

## Scope
- Cohort: `validation\research_notes\manual_review_sensitivity_analysis\cleaned_valid_only.csv`.
- Recording count: 44.
- HRV Studio interval inputs: `validation/kubios_subset/input_ascii_rr/...`, via the `validation_input_file` column in the cleaned benchmark table.
- Kubios exports: `validation\kubios_subset\kubios_exports_120s_75pct_none`. Summary metrics were read from `*_hrv.txt`; interval arrays were read from `*_hrv.mat`.
- Manuscript files were not modified.

## Files Generated
- `per_recording_sequence_comparison.csv`
- `kubios_selected_sequence_metric_crosscheck.csv`

## Finding
The HRV Studio and Kubios time-domain/nonlinear metrics were not computed from identical analyzed NN interval sequences for the full cohort.

For every one of the 44 recordings, Kubios imported the full RR vector and the full imported vector matches the HRV Studio input (`HRV.Data.RR`) exactly at the checked precision. However, Kubios' printed time-domain and nonlinear statistics are computed from `HRV.Data.RRs`, a selected sample. In 31/44 recordings, that selected sample is a shorter prefix of the full input; in 13/44 recordings, it covers the full input.

Evidence:
- Full Kubios import matched HRV Studio input for 44/44 recordings.
- Kubios selected sequence was a prefix of the HRV Studio input for 44/44 recordings.
- Kubios selected sequence was shorter than the HRV Studio input for 31/44 recordings.
- Kubios selected sequence covered the full HRV Studio input for 13/44 recordings.
- HRV Studio analyzed interval count: min=337, median=664.0, max=1219.
- Kubios selected/analyzed interval count: min=88, median=541.5, max=916.
- Count difference (HRV Studio minus Kubios): min=0, median=173.5, max=846.
- HRV Studio duration: min=147.534s, median=600.492s, max=600.971s.
- Kubios selected duration: min=61.431s, median=450.558s, max=600.704s.
- Duration difference (HRV Studio minus Kubios): min=0.000s, median=149.799s, max=539.220s.
- Kubios text reports list sample limits including: 0-148, 0-209, 0-221, 0-227, 0-232, 0-243, 0-451, 0-600, 0-601, 0-62.

Example CH001:
- HRV Studio input: 922 intervals, 600.651s, SDNN 45.2200 ms, RMSSD 20.9853 ms.
- Kubios selected sequence: 94 intervals, 61.431s, printed SDNN 34.6949 ms, printed RMSSD 18.7144 ms.
- Kubios report `Data length` is 00:10:01, but `Sample limits (s)` is 0-62.

## Preprocessing
No artifact-correction or correction-count fields were found in the Kubios text reports or in the inspected `.mat` field names. The debug CSV therefore records `artifact_difference=not_available`.

The `.mat` files show no evidence that Kubios changed the imported full RR vector: `HRV.Data.RR` matches the HRV Studio input for every recording. The disagreement appears after import, at Kubios sample selection (`HRV.Data.RRs`), not from RR-value preprocessing in the full imported vector.

## Segment Alignment
Segment alignment differs for 31/44 recordings. HRV Studio used the full benchmark input sequence for the time-domain/nonlinear extension, while Kubios' printed summary metrics used the selected sample stored as `HRV.Data.RRs`. In all 44 records, that selected sequence is a prefix of the HRV Studio input; in 31 records it is shorter than the full sequence.

## Metric Implementation
The available evidence does not support metric implementation as the primary source of the RMSSD/SDNN disagreement. Recomputing metrics from the Kubios selected sequence reproduces the printed Kubios time-domain metrics closely:
- Mean NN median absolute difference: 2.57139e-05 ms; max 4.94024e-05 ms.
- SDNN median absolute difference: 2.64733e-05 ms; max 4.96024e-05 ms.
- RMSSD median absolute difference: 0.000261925 ms; max 0.0151408 ms.

For SD2, there is evidence of an additional implementation/formula difference. The current HRV Studio nonlinear implementation uses `sd2_squared = 2 * sdnn**2 - 0.5 * sd1**2` in `hrvlib/metrics/nonlinear.py`. Applying that formula to the Kubios selected sequences leaves a median absolute SD2 difference of 1.3333 ms and a maximum difference of 24.3858 ms versus Kubios. Applying `sqrt(2 * SDNN^2 - SD1^2)` to Kubios' printed SDNN and SD1 leaves a smaller median absolute difference of 0.0497 ms and maximum difference of 2.1816 ms.

Conclusion by metric:
- RMSSD and SDNN: disagreement is explained by segment/sequence mismatch, not by metric implementation in this evidence set.
- pNN50 and SD1: selected-sequence recalculation also reproduces Kubios closely.
- SD2: disagreement has both a segment/sequence component and an HRV Studio formula component.
