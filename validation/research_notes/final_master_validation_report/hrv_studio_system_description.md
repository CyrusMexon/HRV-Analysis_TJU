# HRV Studio System Description and Architecture

Generated: 2026-06-15

Repository inspected: repository root

This document describes the implemented HRV Studio repository for manuscript Methods/System Design writing. It is based on repository inspection and does not rerun validation, modify production code, or use artifacts outside the HRV-Analysis_TJU project.

## 1. Software Overview

HRV Studio is a Python-based desktop application and analysis library for heart rate variability analysis. The main application entry point is `main.py`, which launches the PyQt6 GUI defined in `hrvlib/ui/app.py`. The repository is organized around a reusable analysis library (`hrvlib/`) and separate validation scripts (`tools/`, `validation/`).

The intended users are biomedical engineering researchers, HRV analysts, and investigators who need transparent RR-interval processing, metric calculation, visualization, export, and validation traceability. The strongest validated use case is downstream HRV analysis from RR/NN interval data under explicit preprocessing and spectral settings. The repository also implements waveform ingestion and ECG/PPG-derived interval extraction, but the completed validation package is strongest for interval-based HRV processing rather than independent R-peak detector validation.

The software is positioned as transparent, reproducible, and QC-aware. This positioning is supported by explicit modules for data handling, preprocessing, metric computation, warnings/diagnostics, GUI display, export, manual-edit audit trails, and validation run metadata.

## 2. System Architecture

The implementation follows a layered architecture:

- Data input layer: `hrvlib/data_handler.py` loads files into `DataBundle`, `SourceInfo`, and `TimeSeries` structures.
- Preprocessing layer: `hrvlib/preprocessing.py` validates and corrects RR intervals, detects noise/artifacts, and reports quality flags.
- HRV computation layer: `hrvlib/pipeline.py` orchestrates time-domain, frequency-domain, nonlinear, and respiratory modules.
- Time-domain metrics: `hrvlib/metrics/time_domain.py`.
- Frequency-domain metrics: `hrvlib/metrics/freq_domain.py`.
- Nonlinear metrics: `hrvlib/metrics/nonlinear.py`.
- Respiratory metrics: `hrvlib/metrics/respiratory.py`.
- Visualization layer: `hrvlib/ui/plots.py` and `hrvlib/ui/widgets.py`.
- GUI workflow layer: `hrvlib/ui/app.py` and `hrvlib/ui/workers.py`.
- Export/report layer: `hrvlib/export_system.py`.
- Manual editing layer: `hrvlib/beat_editor.py` and `SignalViewerWidget` in `hrvlib/ui/widgets.py`.
- Validation/diagnostics layer: `validation/README_validation.md`, `tools/validate_freq_domain_neurokit2.py`, `tools/analyze_validation_run.py`, `tools/parse_kubios_exports.py`, `tools/validate_fft_ar_methods.py`, `tools/robustness_signal_condition_study.py`, `tools/duration_sensitivity_validation.py`, and `tools/mitbih_arrhythmia_robustness_study.py`.

The central runtime path is: file load -> `DataBundle` -> optional interval extraction -> preprocessing -> `UnifiedHRVPipeline.run_all()` -> metric modules -> quality assessment -> GUI display and export.

## 3. Input Data Support

Confirmed implemented input formats in `hrvlib/data_handler.py`:

- `.csv`: explicit `rri`, `rr`, or `nn` columns for RR/NN intervals; `ppi` for pulse intervals; `ecg`, `ppg`, and `resp` waveform columns with global or channel-specific sampling-rate fields where available. If no recognized columns are found, the first column is treated as RR intervals.
- `.txt`: parsed via pandas when possible or manual numeric parsing; plain numeric TXT is treated as RR intervals.
- `.edf`: waveform import through `pyedflib`; ECG, PPG, RESP channels are inferred from channel labels; possible RR interval channels are heuristically detected.
- `.hrm`: Polar HRM text files, with RR data read from HRData-style sections.
- `.fit`: Garmin FIT files through `fitparse`, using `rr_interval` records converted from seconds to milliseconds.
- `.sml`: Suunto XML/SML files with heuristic RR/PPI extraction.
- `.json`: Movesense-style JSON files with RR/RRI/PPI keys or nested samples.
- `.acq`: Biopac/AcqKnowledge files through `bioread`, with ECG/PPG/RESP channel detection.

The GUI open-file filter confirms `.csv`, `.txt`, `.edf`, `.hrm`, `.fit`, `.sml`, and `.json`. Although `.acq` is implemented in the data handler, it is not listed in the GUI file filter inspected in `hrvlib/ui/app.py`.

For ECG and PPG waveforms, `data_handler.py` can derive intervals using NeuroKit2 (`nk.ecg_process` and `nk.ppg_process`) when waveforms are present and intervals are missing. This feature is implemented, but R-peak or PPG peak detector validation should be treated as separate from the completed HRV metric validation unless specifically documented elsewhere.

## 4. Preprocessing Pipeline

Preprocessing is implemented in `hrvlib/preprocessing.py` and invoked centrally by `UnifiedHRVPipeline._run_preprocessing()`.

Confirmed preprocessing behavior:

- RR unit handling: intervals with mean below 10 are treated as seconds and converted to milliseconds; intervals with mean above 10,000 are treated as microseconds and converted to milliseconds.
- Invalid interval handling: NaN and infinite values are removed before processing; validation flags non-positive, extremely short, extremely long, and too-few intervals.
- Artifact detection: missed beats are intervals above `threshold_high` (default 2000 ms), extra beats are intervals below `threshold_low` (default 300 ms), and ectopic beats are detected by relative deviation from a local mean (`ectopic_threshold`, default 0.3 in preprocessing; GUI worker maps to 0.20).
- Extra-beat correction: extra intervals are merged with an adjacent interval.
- Missed/ectopic correction: cubic spline interpolation is used when `correction_method` is `cubic_spline`; interpolation values are clipped to physiological limits.
- Noise detection: local variability and jump metrics detect noisy segments using a default z-score-like threshold of 2.0, window size 10, and minimum segment length 3.
- Quality flags: high noise (>15% noisy samples), excessive artifacts (>5% detected artifacts), poor signal quality, and irregular rhythm (coefficient of variation >0.45).
- Correction logging: `PreprocessingResult` stores original and corrected RR arrays, artifact indices/types, interpolation indices, correction details, noise segments, quality flags, and summary stats.

Implementation caveat: `PipelineWorker._convert_gui_params_to_preprocessing_config()` returns `artifact_correction_enabled`, but `UnifiedHRVPipeline._run_preprocessing()` filters valid preprocessing parameters and does not include that key. Therefore, the GUI checkbox intended to disable automatic artifact correction may not currently disable correction in the pipeline. This should be resolved or described carefully before claiming configurable correction disablement.

## 5. Time-Domain Metrics

`hrvlib/metrics/time_domain.py` implements:

- SDNN
- RMSSD
- pNN50
- pNN20
- NN50
- NN20
- mean RR
- median RR
- mean HR
- standard deviation of HR
- CVNN
- CVSD
- HRV triangular index
- TINN

The module also reports analysis metadata and preprocessing statistics when available. It includes data-quality warnings for poor signal quality, excessive artifacts, high artifact percentages, short duration, and insufficient interval counts.

## 6. Frequency-Domain Metrics

`hrvlib/metrics/freq_domain.py` implements Welch PSD, whole-signal FFT periodogram PSD, and adaptive autoregressive PSD estimation. The default pipeline settings are 4.0 Hz interpolation, 120 s Welch segment length, 0.75 overlap, Hann window, AR order 16, and linear detrending in the default pipeline configuration. The GUI worker maps the GUI detrending value `"none"` to Python `None` and passes `linear`, `constant`, `smoothness_priors`, or `None`.

Confirmed spectral methods:

- Welch PSD through `scipy.signal.welch`.
- FFT PSD through `numpy.fft.rfft` with window normalization.
- AR PSD with Burg estimation attempted first and Yule-Walker fallback; if AR fails, Welch PSD is used as fallback.

Implemented frequency bands:

- ULF: 0.0-0.003 Hz
- VLF: 0.0-0.04 Hz
- LF: 0.04-0.15 Hz
- HF: 0.15-0.4 Hz

Implemented frequency-domain metrics:

- ULF, VLF, LF, HF power
- ULF, VLF, LF, HF percent of total power
- total power integrated over 0.0-0.4 Hz
- LF/HF ratio
- LF_nu and HF_nu as LF and HF normalized to LF+HF
- relative LF and HF power
- VLF, LF, and HF peak frequencies
- method-prefixed Welch, FFT, and AR versions of spectral metrics

Diagnostics include effective Welch segment length/overlap, frequency resolution, number of segments, resampled duration, PSD nonfinite checks, duration warnings, per-band bin counts, per-band duration warnings, FFT/AR integrated PSD-to-variance ratios, AR estimator/fallback details, and invalid-input flags.

Arm A / no-detrend handling: when `detrend_method is None`, the Welch path removes one global mean before Welch and disables segment detrending. The diagnostics label this as `global_mean_then_none`. This is relevant to the Kubios-oriented validation convention.

## 7. Nonlinear Metrics

`hrvlib/metrics/nonlinear.py` implements:

- Poincare SD1
- Poincare SD2
- SD1/SD2 ratio
- Poincare ellipse area
- CSI
- CVI
- modified CSI
- Sample Entropy
- Approximate Entropy
- Multiscale Entropy with area under the MSE curve
- DFA alpha1 and alpha2, with box sizes and fluctuation values

RQA or recurrence quantification analysis was not confirmed in repository inspection. It should not be claimed as implemented unless added or separately verified.

## 8. Visualization and GUI

The desktop GUI is implemented with PyQt6 in `hrvlib/ui/app.py` and supporting widgets in `hrvlib/ui/widgets.py`. The GUI includes:

- File loading and metadata display.
- Analysis parameter controls for analysis window, artifact correction setting, interpolation method, quality-warning threshold, detrending method, detrending lambda, AR model order, and Welch window function.
- Threaded analysis through `PipelineWorker` to avoid blocking the UI.
- Results panel with time-domain, frequency-domain, nonlinear, quality, and warning widgets.
- Signal viewer with analysis overview and beat-editing modes.
- Manual beat editing controls: select, delete, move, interpolate, insert, undo, reset, and reanalyze.
- Session save/load support through `SessionManager`.
- Export dialog for PDF, CSV, SPSS-compatible CSV, and audit-trail JSON.

Implemented visualizations include:

- RR/tachogram over time.
- Heart-rate distribution histogram.
- PSD plot with LF and HF band shading.
- Frequency band power bar plot.
- Poincare plot.
- DFA log-log plot.
- Quality assessment bar plot.
- Editing-oriented RR interval view.

Segment or analysis-window selection is implemented through GUI parameters (`start_sec`, `duration_sec`) and is applied in the pipeline and metric modules. A specialized segment browser beyond the analysis-window/editing view was not confirmed.

## 9. Export and Reproducibility

`hrvlib/export_system.py` implements:

- PDF report export with summary figures, metric tables, quality assessment, warnings, and parameters.
- CSV metrics export with flattened time, frequency, nonlinear, and quality metrics.
- SPSS-compatible CSV export with selected SPSS-style variable names.
- JSON audit trail export with source info, analysis parameters, preprocessing log, manual editing log, and analysis summary.

The GUI adds timestamped filenames by default and can append an edited-data suffix when manual edits exist. Manual-edit information is tracked through `edit_history` and `detailed_audit_trail` when available. The validation tools separately produce run-level reproducibility artifacts, especially `run_info.json`, `notes.md`, comparison CSV files, diagnostics JSON/CSV, and generated summaries under `validation/runs/<run-name>/`.

## 10. Validation-Support Features

Validation support is implemented both inside the frequency-domain module and in external tools.

Core diagnostics in `freq_domain.py` include:

- duration warnings
- per-band duration warnings
- per-band PSD bin counts
- ULF/VLF overlap warning
- invalid RR removal count
- duplicate cumulative time-point detection
- nonfinite interpolation/PSD flags
- Welch, FFT, and AR method diagnostics
- FFT and AR PSD area-to-variance checks
- AR fallback and estimator metadata

Validation scripts implement:

- HRV Studio versus NeuroKit2 frequency-domain validation (`tools/validate_freq_domain_neurokit2.py`).
- Post-run diagnostic summaries (`tools/analyze_validation_run.py`).
- Kubios export parsing and comparison (`tools/parse_kubios_exports.py`).
- FFT/AR method validation (`tools/validate_fft_ar_methods.py`).
- Synthetic robustness/signal-condition stress tests (`tools/robustness_signal_condition_study.py`).
- Duration-sensitivity analysis (`tools/duration_sensitivity_validation.py`).
- MIT-BIH arrhythmia robustness/QC stress testing (`tools/mitbih_arrhythmia_robustness_study.py`).

`validation/README_validation.md` documents the local validation workspace, run layout, NeuroKit2 validation runner, diagnostic analysis, outlier inspection, PhysioNet preparation, and the role of Kubios as a smaller manual subset.

## 11. Design Choices and Transparency

HRV Studio is transparent because analysis stages are implemented as separate modules with explicit parameters and intermediate diagnostics. The data structures preserve source metadata, preprocessing statistics, quality flags, and analysis settings. The frequency-domain module exposes method-specific diagnostics rather than hiding PSD convention issues.

The system is reproducible because validation scripts write run folders with `run_info.json`, notes, comparison tables, diagnostics, summaries, and generated plots. GUI exports can include analysis parameters and an audit trail. Batch and validation scripts are command-line driven and designed to avoid overwriting run meaning without explicit run names.

The system is QC-aware because it does not treat finite output as sufficient evidence of reliability. It reports artifact percentages, noise segments, quality flags, rhythm irregularity indicators, duration warnings, band-bin warnings, PSD variance warnings, and manual edit logs.

## 12. Limitations

- HRV Studio should not be described as a medical diagnostic tool.
- The completed validation evidence is strongest for downstream RR/NN interval processing and frequency-domain agreement under explicit settings.
- ECG/PPG peak extraction is implemented through NeuroKit2 or fallback methods, but independent R-peak or PPG peak detector validation is not established by the completed master validation package.
- VLF and absolute spectral powers are sensitive to detrending, DC handling, interpolation, segment length, overlap, windowing, low-frequency bin treatment, and PSD integration conventions.
- Short-duration frequency-domain outputs can be finite but should be labeled exploratory when below 5 minutes, especially at 30-60 seconds.
- FFT and AR outputs are implemented but remain method-dependent; Welch should remain the primary reported method unless FFT/AR are separately specified and validated for primary use.
- Arrhythmic MIT-BIH outputs require caution and should be framed as robustness/QC stress tests, not clinical validation or diagnostic-accuracy evidence.
- GUI support for disabling automatic artifact correction may be inconsistent with pipeline parameter filtering and should not be overclaimed until fixed or verified.

## 13. Paper-Ready Methods/System Description Draft

HRV Studio was implemented as a Python-based desktop application and analysis library for transparent heart rate variability (HRV) analysis. The system is organized as a modular pipeline with distinct layers for data ingestion, preprocessing, metric computation, visualization, export, and validation support. The graphical user interface is implemented in PyQt6 and provides file loading, parameter selection, threaded analysis execution, visualization of signals and metrics, manual beat-editing controls, session management, and export of results. The main application launches from `main.py`, while the core analysis functions are implemented in the `hrvlib` package.

Input data are represented internally using a `DataBundle` structure that stores RR intervals, pulse-to-pulse intervals, ECG, PPG, respiratory waveforms, source metadata, and preprocessing information. Implemented file readers support CSV and TXT interval files, EDF waveforms, Polar HRM, Garmin FIT, Suunto SML, Movesense-style JSON, and Biopac ACQ files where optional dependencies are available. CSV and TXT files may contain explicit RR/NN, PPI, ECG, PPG, or respiratory columns; plain numeric files are treated as RR interval sequences. For waveform inputs, the repository implements ECG/PPG-derived interval extraction using NeuroKit2 when intervals are not already available. The present validation evidence, however, is strongest for downstream RR/NN interval processing rather than independent validation of peak detection from raw waveforms.

RR interval preprocessing is centralized before metric computation. The preprocessing module removes nonfinite intervals, detects likely unit scale, identifies noisy segments, and detects artifacts using interval thresholds and local relative-change criteria. Default artifact thresholds identify extra beats below 300 ms, missed beats above 2000 ms, and ectopic intervals based on local deviation. Extra beats are corrected by interval merging, while missed and ectopic intervals can be corrected by cubic spline interpolation. The preprocessing result records original and corrected intervals, artifact indices and types, interpolated indices, correction details, noisy segments, quality flags, and summary statistics. Quality flags include high noise, excessive artifacts, poor signal quality, and irregular rhythm. These outputs are propagated to the analysis results and export layer.

The unified analysis pipeline computes time-domain, frequency-domain, nonlinear, and respiratory-derived measures from the preprocessed intervals. Time-domain metrics include SDNN, RMSSD, pNN50, pNN20, NN50, NN20, mean and median RR, mean and standard deviation of heart rate, CVNN, CVSD, HRV triangular index, and TINN. Frequency-domain analysis resamples the RR series, by default at 4 Hz, and computes Welch PSD, whole-signal FFT periodogram PSD, and adaptive autoregressive PSD. The primary validation-supported spectral method is Welch. The implemented bands are ULF (0.0-0.003 Hz), VLF (0.0-0.04 Hz), LF (0.04-0.15 Hz), and HF (0.15-0.4 Hz). Outputs include band powers, total power integrated over 0.0-0.4 Hz, LF/HF ratio, LF and HF normalized units, relative LF/HF powers, peak band frequencies, and method-prefixed Welch, FFT, and AR metrics. When no detrending is selected for Welch, the implementation removes a global mean before Welch integration and disables segment detrending, matching the convention used in the Kubios-oriented validation path. Frequency diagnostics report duration warnings, per-band bin counts, effective Welch parameters, PSD nonfinite flags, PSD area-to-variance consistency for FFT/AR, AR fallback information, and band-specific reliability warnings.

Nonlinear analyses include Poincare SD1/SD2 and related ellipse indices, sample entropy, approximate entropy, multiscale entropy, and detrended fluctuation analysis alpha1/alpha2. Respiratory support includes ECG-derived respiration by amplitude modulation, direct respiratory channel analysis, RSA coherence, and LF/HF boundary-overlap annotation when sufficient respiratory information is available.

Visualization and reporting are integrated into the desktop workflow. The GUI displays RR tachograms, heart-rate distributions, PSD curves, frequency-band power summaries, Poincare plots, DFA plots, quality summaries, warnings, and metric panels. The export system can generate PDF reports, standard CSV metric files, SPSS-compatible CSV files, and JSON audit trails containing source information, parameters, preprocessing logs, quality assessment, and manual-editing details. Separate validation scripts generate reproducible run folders containing `run_info.json`, notes, comparison CSV files, diagnostics, plots, and summaries for NeuroKit2, Kubios, FFT/AR, duration-sensitivity, robustness, and MIT-BIH stress-test analyses. These design choices support the use of HRV Studio as a transparent, reproducible, QC-aware research platform, while avoiding claims of clinical diagnostic validity or unrestricted equivalence to external software.

## Files Inspected

- `main.py`
- `batch_run_hrv_studio.py`
- `hrvlib/data_handler.py`
- `hrvlib/preprocessing.py`
- `hrvlib/pipeline.py`
- `hrvlib/beat_editor.py`
- `hrvlib/export_system.py`
- `hrvlib/metrics/time_domain.py`
- `hrvlib/metrics/freq_domain.py`
- `hrvlib/metrics/nonlinear.py`
- `hrvlib/metrics/respiratory.py`
- `hrvlib/signal_processing/detrending.py`
- `hrvlib/signal_processing/smoothness_priors.py`
- `hrvlib/ui/app.py`
- `hrvlib/ui/widgets.py`
- `hrvlib/ui/plots.py`
- `hrvlib/ui/workers.py`
- `validation/README_validation.md`
- `tests/README.md`
- `tools/validate_freq_domain_neurokit2.py`
- `tools/analyze_validation_run.py`
- `tools/parse_kubios_exports.py`
- `tools/validate_fft_ar_methods.py`
- `tools/robustness_signal_condition_study.py`
- `tools/duration_sensitivity_validation.py`
- `tools/mitbih_arrhythmia_robustness_study.py`

## Confirmed Features

- PyQt6 desktop GUI.
- CSV/TXT/EDF/HRM/FIT/SML/JSON/ACQ data-handler support, with ACQ not listed in the inspected GUI file filter.
- RR, PPI, ECG, PPG, and RESP data structures.
- ECG/PPG-derived interval extraction via NeuroKit2 when available.
- Centralized RR preprocessing with artifact detection/correction, noise detection, quality flags, and correction stats.
- Time-domain, frequency-domain, nonlinear, and respiratory modules.
- Welch, FFT, and AR spectral outputs.
- Frequency diagnostics for duration, bin count, PSD variance consistency, invalid inputs, and AR fallback.
- Manual beat editing and audit-trail support.
- PDF, CSV, SPSS-compatible CSV, and JSON audit-trail export.
- Validation scripts with reproducible run folders and diagnostics.

## Features Not Confirmed

- RQA / recurrence quantification analysis.
- Independent validation of ECG R-peak or PPG peak detection accuracy.
- Medical diagnostic or clinical decision-support functionality.
- A dedicated segment browser beyond analysis-window selection and signal-view editing.
- GUI-listed support for `.acq` files, despite implementation in the data handler.

## Implementation/Documentation Inconsistencies Found

- The GUI exposes an artifact-correction checkbox, but pipeline parameter filtering appears to drop `artifact_correction_enabled` before calling `preprocess_rri`. This may prevent the checkbox from disabling correction.
- The GUI file-open filter omits `.acq`, although `data_handler.py` implements a Biopac ACQ loader.
- Some source comments and UI strings contain mojibake or garbled Unicode characters. This does not necessarily affect computation, but it can reduce documentation clarity.
