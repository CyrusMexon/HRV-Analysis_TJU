# HRV Validation Workspace

This folder contains local validation inputs, generated validation runs, and lightweight documentation for frequency-domain validation work. It is intended to support reproducible validation development without committing raw datasets or large generated outputs to GitHub.

## Folder Layout

- `validation/raw_data/`: local raw datasets, including Syl_Vain files and downloaded PhysioNet archives or extracted records.
- `validation/processed_data/`: derived RR interval files prepared for validation, including PhysioNet NSR full-record and fixed-length segment exports.
- `validation/runs/`: generated validation outputs such as comparison CSV files, diagnostics, plots, summaries, and run metadata.
- `validation/research_notes/`: lightweight internal notes and decision logs that may be reused later when writing validation methods, discussion, and limitations.

Raw data, processed data, and generated runs are intentionally ignored by Git. These files can be large, may be redistributable only under their source dataset terms, and are expected to differ across local validation experiments.

## Running NeuroKit2 Validation

Use `tools/validate_freq_domain_neurokit2.py` to compare HRV Studio frequency-domain metrics with NeuroKit2.

Example:

```powershell
python tools/validate_freq_domain_neurokit2.py validation/processed_data/physionet_nsr_rr_5min --recursive --run-name v03_physionet_5min_neurokit2 --enable-diagnostics
```

The runner writes outputs under `validation/runs/<run-name>/`, including `run_info.json`, `notes.md`, comparison rows, and optional diagnostics. Avoid passing manifest files such as `5min_manifest.csv` as RR input files.

Known methodological issue: VLF / total_power comparison depends strongly on Welch detrending convention.

HRV Studio currently uses segment-wise Welch detrending for linear mode. The NeuroKit2 validation path historically uses global detrending followed by Welch with `detrend=False`. This mainly affects DC / first VLF bins, so VLF and total_power can differ substantially even when LF, HF, and LF/HF remain closely aligned. A Kubios subset is still needed to decide which reference convention is preferred for final reporting.

## Analyzing a Run

Use `tools/analyze_validation_run.py` after a validation run finishes.

```powershell
python tools/analyze_validation_run.py --run-name v03_physionet_5min_neurokit2
```

This generates or updates diagnostic summaries for the run and appends notes unless `--no-append-notes` is used.

## Inspecting Outliers

Use `tools/inspect_freq_outlier.py` for a representative RR file or validation outlier.

```powershell
python tools/inspect_freq_outlier.py validation/processed_data/physionet_nsr_rr_5min/nsr001_segment_001.csv --run-name v03_physionet_5min_neurokit2 --enable-diagnostics
```

Inspection outputs are written under the selected run folder and are ignored by Git.

## Preparing PhysioNet NSR RR Data

Use `tools/prepare_physionet_nsr_rr.py` to prepare local PhysioNet Normal Sinus Rhythm RR Interval Database files.

```powershell
python tools/prepare_physionet_nsr_rr.py --input-dir validation/raw_data/normal-sinus-rhythm-rr-interval-database-1.0.0 --output-dir validation/processed_data --segment-lengths 300 600
```

Use `--overwrite` only when intentionally regenerating processed files.

## Recommended Run Names

- `v01_pre_diagnostics`
- `v02_with_diagnostics`
- `v02b_nfft_alignment_experiment`
- `v03_physionet_5min_neurokit2`

Run names should encode the major validation condition or methodological change. Keep generated run outputs local and summarize important conclusions in `validation/research_notes/`.

## Kubios Validation Note

Kubios validation is expected to remain a smaller manual subset because the current Kubios version does not provide a practical batch-processing workflow for the full validation corpus. The Kubios subset should be used as an additional reference point rather than as the only large-scale validation comparator.
