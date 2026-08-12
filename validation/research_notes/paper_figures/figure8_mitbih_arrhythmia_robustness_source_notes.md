# Figure 8 MIT-BIH Arrhythmia Robustness Source Notes

Generated on 2026-06-15 from existing migrated v11 validation artifacts in the HRV-Analysis_TJU project. Production HRV code was not modified, original validation outputs were not overwritten, and the full v11 experiment was not rerun.

## Output Files

- `validation/research_notes/paper_figures/figure8_mitbih_arrhythmia_robustness.png`
- `validation/research_notes/paper_figures/figure8_mitbih_arrhythmia_robustness.pdf`
- `validation/research_notes/paper_figures/figure8_mitbih_arrhythmia_robustness.svg`

## Exact Input Files Used

- v11 robustness results: `validation/runs/v11_mitbih_arrhythmia_robustness/mitbih_robustness_results.csv`
- v11 artifact effect table: `validation/runs/v11_mitbih_arrhythmia_robustness/mitbih_artifact_effect_table.csv`
- v11 segment manifest: `validation/runs/v11_mitbih_arrhythmia_robustness/mitbih_segment_manifest.csv`
- v11 warning summary: `validation/runs/v11_mitbih_arrhythmia_robustness/mitbih_warning_summary.csv`
- v11 summary markdown: `validation/runs/v11_mitbih_arrhythmia_robustness/mitbih_robustness_summary.md`
- v11 run metadata: `validation/runs/v11_mitbih_arrhythmia_robustness/run_info.json`
- Migrated MIT-BIH annotation/data root used for selected visual reconstruction: `validation/raw_data/mit_bih_arrhythmia/mit-bih-arrhythmia-database-1.0.0`
- Existing representative tachograms reviewed but not directly reused: `validation/runs/v11_mitbih_arrhythmia_robustness/tachograms/`

## MIT-BIH Records and Segments Used

| Panel | Segment | Record | Category | Window (s) | RR before | RR after | Correction count | Why selected |
|---|---|---:|---|---:|---:|---:|---:|---|
| A | mitbih_103_0000_0600 | 103 | mostly normal rhythm | 0-600 | 702 | 702 | 0 | stable mostly normal record with 100% normal/conducted beats and no correction applied |
| B | mitbih_208_0300_0900 | 208 | PVC/ectopic-heavy rhythm | 300-900 | 984 | 984 | 241 | PVC-heavy segment with 57% ventricular ectopic/fusion beats and visible irregularity |
| C | mitbih_207_1200_1800 | 207 | irregular rhythm | 1200-1800 | 552 | 552 | 10 | highly irregular/nonstationary segment with very high RR CV and mixed rhythm annotations |

## Why These Examples Were Selected

- Record 103 was selected for Panel A because it is a stable mostly normal rhythm segment in the v11 manifest and matches the preferred near-normal record requested.
- Record 208 was selected for Panel B because it is the preferred PVC-heavy example and has a high ventricular ectopic/fusion fraction in the v11 manifest.
- Record 207 was selected for Panel C because it is the preferred highly irregular example, with mixed annotations, very high RR CV, and irregular-rhythm QC flags.

## Panel D Values

| Summary item | Count |
|---|---:|
| Finite outputs before correction | 12/12 |
| Finite outputs after correction | 12/12 |
| Warning visibility after correction | 12/12 |
| Correction applied after correction | 9/12 |
| Excess artifacts after correction | 6/12 |
| High noise after correction | 6/12 |
| Irregular flag after correction | 3/12 |
| Poor quality after correction | 6/12 |

Additional v11 summary values:

- Segments analyzed: 12.
- After-correction warning/diagnostic labels: 12/12.
- After-correction segments with at least one corrected interval: 9/12.
- Primary interpretation: MIT-BIH arrhythmia recordings are robustness/QC stress tests, not clinical validation or diagnostic-accuracy evidence.

## Filtering and Regeneration

- Panels A-C use the exact segment windows in `validation/runs/v11_mitbih_arrhythmia_robustness/mitbih_segment_manifest.csv`.
- No v11 segment was excluded from Panel D.
- Existing tachogram PNGs were reviewed but not reused directly because the requested layout uses stacked before/after traces within each panel.
- Panels A-C were regenerated only for the three selected visual examples from the migrated MIT-BIH annotations and the recorded v11 segment windows. This was not a full validation rerun.
- Panel D was regenerated from existing v11 CSV outputs.

## Reconstruction Checks

- `mitbih_103_0000_0600`: reconstructed correction count 0; CSV correction count 0.
- `mitbih_208_0300_0900`: reconstructed correction count 241; CSV correction count 241.
- `mitbih_207_1200_1800`: reconstructed correction count 10; CSV correction count 10.

## Assumptions

- `metrics_finite == True` is interpreted as numerical output stability only, not physiological correctness.
- Any non-empty after-correction `warning_labels` field is counted as warning visibility.
- `correction_count > 0` is counted as correction applied.
- The before/after traces show preprocessing behavior and retained rhythm irregularity; they do not imply successful physiological correction.
- The figure is intended to support robustness and transparent QC under arrhythmia stress testing, not clinical validation, diagnostic performance, or validity of HRV metrics during arrhythmia.
- Figure contains no overall title and no embedded caption; manuscript caption should be supplied separately.
