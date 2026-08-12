# MIT-BIH v11 Migration Audit

Generated: 2026-06-13T08:24:59

Source project preserved for provenance: external Glucose-HRV project workspace (local path omitted)
Target project: repository root

No files had been copied when this audit was created.

## Source and Target Inventory

### MIT-BIH extracted dataset
Source: external Glucose-HRV project workspace `validation/raw_data/mit_bih_arrhythmia`
Target: `validation/raw_data/mit_bih_arrhythmia`
Source exists: True
Source file count / size: 705 / 109,337,739 bytes (104.27 MiB)
Target exists before migration: False
Pre-copy hash comparison: target folder missing, so all source files are missing in target; no differing target files found.

### v11 MIT-BIH robustness run
Source: external Glucose-HRV project workspace `validation/runs/v11_mitbih_arrhythmia_robustness`
Target: `validation/runs/v11_mitbih_arrhythmia_robustness`
Source exists: True
Source file count / size: 12 / 896,770 bytes (0.86 MiB)
Target exists before migration: False
Pre-copy hash comparison: target folder missing, so all source files are missing in target; no differing target files found.

### MIT-BIH robustness script
Source: external Glucose-HRV project workspace `tools/mitbih_arrhythmia_robustness_study.py`
Target: `tools/mitbih_arrhythmia_robustness_study.py`
Source exists: True
Source size / SHA256: 41,314 bytes (0.04 MiB) / `068682C073A71D4280D8606E081F29A29AE17952940293C21E21B3C1478D81F5`
Target exists before migration: False
Pre-copy comparison: target file missing; no differing target file found.

### MIT-BIH-specific research note
Source: external Glucose-HRV project workspace `validation/research_notes/final_master_validation_report/final_claims_and_limitations.md`
Target: `validation/research_notes/final_master_validation_report/final_claims_and_limitations.md`
Source exists: True
Source size / SHA256: 4,672 bytes (0.00 MiB) / `798C618B0028AA7F68056DEBFC55AE46815DEEBFE041F86F5A0D8838F180058E`
Target exists before migration: False
Pre-copy comparison: target file missing; no differing target file found.

## Partial v11 Output Check
Target v11 folder does not exist before migration; no partial v11 outputs found there.

## Existing MIT-BIH Archives in Correct Project
- `validation/raw_data/archives/mit-bih-arrhythmia-database-1.0.0.zip` | 77,030,320 bytes (73.46 MiB) | SHA256 `47B26926927C11BD9174154D367C811AFC1B186F650F8A205931CA8C520F0A87`

## Conflict Handling Plan
For copy operations, existing identical files will be skipped. Existing differing target files will be preserved, and the source file will be copied beside them with `_from_GlucoseHRV_migrated` before the extension.

## Post-Copy Result
Recorded after the safe copy phase: 2026-06-13T08:27:53
  - Copied files: 719
 - Skipped identical files: 0
 - Conflicts: 0
 - Missing sources: 0
 - Copy log: `validation/research_notes/migration_mitbih_v11_copy_log.json`
 - No target files were overwritten.
