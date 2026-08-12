"""MIT-BIH Arrhythmia robustness/QC stress-test study for HRV Studio.

This validation-only utility evaluates HRV Studio behavior on real
arrhythmic and non-ideal RR signals derived from MIT-BIH Arrhythmia
Database WFDB annotations. It does not modify production HRV code.

Example:
    python tools/mitbih_arrhythmia_robustness_study.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_DIR = SCRIPT_PROJECT_ROOT / "validation" / "raw_data" / "archives"
DEFAULT_HRV_PROJECT_ROOT = SCRIPT_PROJECT_ROOT
RUN_NAME = "v11_mitbih_arrhythmia_robustness"
BAND_CONVENTION = "standard"
DATASET_DIR_NAME = "mit_bih_arrhythmia"
METRICS = ["SDNN", "RMSSD", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu"]

BEAT_SYMBOLS = {
    "N",
    "L",
    "R",
    "B",
    "A",
    "a",
    "J",
    "S",
    "V",
    "r",
    "F",
    "e",
    "j",
    "n",
    "E",
    "/",
    "f",
    "Q",
    "?",
}
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
VENTRICULAR_SYMBOLS = {"V", "E", "F"}
SUPRAVENTRICULAR_SYMBOLS = {"A", "a", "J", "S"}
PACED_SYMBOLS = {"/", "f"}
NOISE_SYMBOLS = {"~", "|", "x"}

PREFERRED_RECORDS: Dict[str, str] = {
    "100": "mostly normal rhythm",
    "101": "mostly normal rhythm",
    "103": "mostly normal rhythm",
    "112": "mostly normal rhythm",
    "106": "PVC/ectopic-heavy rhythm",
    "119": "PVC/ectopic-heavy rhythm",
    "200": "PVC/ectopic-heavy rhythm",
    "208": "PVC/ectopic-heavy rhythm",
    "221": "PVC/ectopic-heavy rhythm",
    "201": "irregular rhythm",
    "202": "irregular rhythm",
    "203": "irregular rhythm",
    "207": "irregular rhythm",
    "222": "irregular rhythm",
    "223": "irregular rhythm",
    "105": "noisy/problematic rhythm",
    "108": "noisy/problematic rhythm",
    "210": "noisy/problematic rhythm",
    "228": "noisy/problematic rhythm",
}
TARGET_CATEGORY_COUNTS = {
    "mostly normal rhythm": 3,
    "PVC/ectopic-heavy rhythm": 4,
    "irregular rhythm": 3,
    "noisy/problematic rhythm": 2,
}


@dataclass
class RecordAnnotations:
    record_id: str
    record_path: Path
    fs: float
    beat_samples: np.ndarray
    beat_symbols: np.ndarray
    all_samples: np.ndarray
    all_symbols: np.ndarray
    aux_notes: List[str]

    @property
    def beat_times_s(self) -> np.ndarray:
        return self.beat_samples.astype(float) / self.fs

    @property
    def duration_s(self) -> float:
        if self.all_samples.size:
            return float(np.max(self.all_samples) / self.fs)
        if self.beat_samples.size:
            return float(np.max(self.beat_samples) / self.fs)
        return 0.0


@dataclass
class Segment:
    segment_id: str
    record_id: str
    intended_category: str
    start_s: float
    end_s: float
    duration_s: float
    rr_ms: np.ndarray
    rr_times_s: np.ndarray
    beat_symbols: List[str]
    rhythm_summary: str
    stats: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-only MIT-BIH Arrhythmia Database robustness/QC "
            "stress test for HRV Studio metrics."
        )
    )
    parser.add_argument("--hrv-project-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_PROJECT_ROOT,
        help=(
            "Root for validation/raw_data and validation/runs outputs. "
            "Defaults to the project containing this script."
        ),
    )
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--zip-pattern", default="mit-bih-arrhythmia-database*.zip")
    parser.add_argument("--segment-duration-s", type=float, default=600.0)
    parser.add_argument("--window-step-s", type=float, default=300.0)
    parser.add_argument("--min-segment-duration-s", type=float, default=540.0)
    parser.add_argument("--min-rr-count", type=int, default=250)
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--interpolation-rate", type=float, default=4.0)
    parser.add_argument("--segment-length", type=float, default=120.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.75)
    parser.add_argument("--window-type", default="hann")
    parser.add_argument(
        "--detrend-method",
        choices=["none", "linear", "constant", "smoothness_priors"],
        default="none",
    )
    return parser.parse_args()


def discover_hrv_project_root(cli_value: Optional[Path]) -> Path:
    candidates = []
    if cli_value is not None:
        candidates.append(cli_value)
    env_value = os.environ.get("HRV_STUDIO_PROJECT_ROOT")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend([SCRIPT_PROJECT_ROOT, Path.cwd(), DEFAULT_HRV_PROJECT_ROOT])

    for candidate in candidates:
        root = candidate.resolve()
        if (root / "hrvlib").is_dir() and (root / "validation").is_dir():
            return root
    raise SystemExit(
        "Could not locate the HRV Studio project root containing hrvlib/. "
        "Use --hrv-project-root or set HRV_STUDIO_PROJECT_ROOT."
    )


def import_hrv_studio(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        import hrvlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Could not import HRV Studio hrvlib from {project_root}: {exc}")


def require_wfdb() -> Any:
    try:
        import wfdb  # type: ignore
    except ImportError:
        raise SystemExit(
            "WFDB is required to parse MIT-BIH .atr annotations but is not installed.\n"
            "Install it in the active environment, then rerun:\n"
            "  python -m pip install wfdb"
        )
    return wfdb


def normalize_detrend(value: str) -> Optional[str]:
    return None if value == "none" else value


def prepare_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Output folder already exists and is non-empty: {run_dir}. "
            "This script will not overwrite previous validation outputs."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tachograms").mkdir(parents=True, exist_ok=True)


def find_mitbih_zip(archive_dir: Path, pattern: str) -> Path:
    matches = sorted(archive_dir.glob(pattern))
    if not matches:
        raise SystemExit(f"No MIT-BIH archive matching {pattern!r} found in {archive_dir}")
    return matches[-1]


def extract_dataset_if_needed(zip_path: Path, dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if list(dataset_dir.rglob("*.atr")) and list(dataset_dir.rglob("*.hea")):
        return find_dataset_record_root(dataset_dir)

    if any(dataset_dir.iterdir()):
        raise SystemExit(
            f"Dataset directory exists but does not contain MIT-BIH .atr/.hea files: {dataset_dir}. "
            "Inspect it manually before rerunning."
        )

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir)
    return find_dataset_record_root(dataset_dir)


def find_dataset_record_root(dataset_dir: Path) -> Path:
    candidates = [dataset_dir] + [p for p in dataset_dir.rglob("*") if p.is_dir()]
    for candidate in candidates:
        if (candidate / "RECORDS").exists() and list(candidate.glob("*.atr")):
            return candidate
    for candidate in candidates:
        if list(candidate.glob("*.atr")) and list(candidate.glob("*.hea")):
            return candidate
    raise SystemExit(f"Could not find extracted MIT-BIH record files under {dataset_dir}")


def read_records(record_root: Path) -> List[str]:
    records_path = record_root / "RECORDS"
    if records_path.exists():
        records = []
        for line in records_path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                records.append(text)
        if records:
            return records
    return sorted(path.stem for path in record_root.glob("*.atr"))


def load_record_annotations(wfdb: Any, record_root: Path, record_id: str) -> Optional[RecordAnnotations]:
    record_path = record_root / record_id
    try:
        header = wfdb.rdheader(str(record_path))
        ann = wfdb.rdann(str(record_path), "atr")
    except Exception as exc:
        print(f"Skipping record {record_id}: {exc}")
        return None

    fs = float(getattr(header, "fs", 360.0) or 360.0)
    samples = np.asarray(ann.sample, dtype=int)
    symbols = np.asarray(ann.symbol, dtype=object)
    aux_notes = [str(x).strip() for x in getattr(ann, "aux_note", []) if str(x).strip()]
    beat_mask = np.asarray([str(symbol) in BEAT_SYMBOLS for symbol in symbols], dtype=bool)

    if np.count_nonzero(beat_mask) < 3:
        return None
    return RecordAnnotations(
        record_id=record_id,
        record_path=record_path,
        fs=fs,
        beat_samples=samples[beat_mask],
        beat_symbols=symbols[beat_mask],
        all_samples=samples,
        all_symbols=symbols,
        aux_notes=aux_notes,
    )


def symbol_fraction(symbols: Sequence[str], symbol_set: set[str]) -> float:
    if not symbols:
        return 0.0
    return sum(1 for symbol in symbols if symbol in symbol_set) / len(symbols)


def summarize_segment(
    record: RecordAnnotations,
    start_s: float,
    end_s: float,
    intended_category: str,
) -> Optional[Segment]:
    beat_times = record.beat_times_s
    beat_mask = (beat_times >= start_s) & (beat_times <= end_s)
    beat_indices = np.where(beat_mask)[0]
    if beat_indices.size < 3:
        return None

    selected_samples = record.beat_samples[beat_indices]
    selected_symbols = [str(symbol) for symbol in record.beat_symbols[beat_indices]]
    rr_ms = np.diff(selected_samples).astype(float) / record.fs * 1000.0
    rr_times_s = selected_samples[1:].astype(float) / record.fs
    rr_symbols = selected_symbols[1:]
    finite_rr = rr_ms[np.isfinite(rr_ms)]

    if finite_rr.size < 3:
        return None
    duration_s = float(rr_times_s[-1] - rr_times_s[0]) if rr_times_s.size > 1 else float(np.sum(rr_ms) / 1000.0)

    all_times = record.all_samples.astype(float) / record.fs
    all_mask = (all_times >= start_s) & (all_times <= end_s)
    all_symbols = [str(symbol) for symbol in record.all_symbols[all_mask]]
    noise_events = sum(1 for symbol in all_symbols if symbol in NOISE_SYMBOLS)
    rhythm_notes = sorted(
        {
            note
            for note in record.aux_notes
            if any(token in note.upper() for token in ["AFIB", "AFL", "VT", "VFL", "SVTA", "BII"])
        }
    )

    normal_fraction = symbol_fraction(rr_symbols, NORMAL_SYMBOLS)
    ventricular_fraction = symbol_fraction(rr_symbols, VENTRICULAR_SYMBOLS)
    supraventricular_fraction = symbol_fraction(rr_symbols, SUPRAVENTRICULAR_SYMBOLS)
    paced_fraction = symbol_fraction(rr_symbols, PACED_SYMBOLS)
    ectopic_fraction = ventricular_fraction + supraventricular_fraction
    invalid_rr_count = int(
        np.count_nonzero(
            (~np.isfinite(rr_ms)) | (rr_ms <= 0) | (rr_ms < 300.0) | (rr_ms > 2000.0)
        )
    )
    rr_cv = float(np.std(finite_rr) / np.mean(finite_rr)) if np.mean(finite_rr) > 0 else math.nan
    rr_diff_median = float(np.median(np.abs(np.diff(finite_rr)))) if finite_rr.size > 1 else math.nan
    symbol_counts = pd.Series(rr_symbols, dtype=object).value_counts().to_dict()
    symbol_diversity = int(len(symbol_counts))
    artifact_like_fraction = invalid_rr_count / len(rr_ms) if len(rr_ms) else 0.0

    stats = {
        "normal_fraction": normal_fraction,
        "ventricular_fraction": ventricular_fraction,
        "supraventricular_fraction": supraventricular_fraction,
        "ectopic_fraction": ectopic_fraction,
        "paced_fraction": paced_fraction,
        "rr_cv": rr_cv,
        "rr_diff_median_ms": rr_diff_median,
        "symbol_diversity": symbol_diversity,
        "noise_event_count": noise_events,
        "invalid_rr_count": invalid_rr_count,
        "artifact_like_fraction": artifact_like_fraction,
        "symbol_counts": symbol_counts,
        "rhythm_notes": "; ".join(rhythm_notes[:6]),
    }
    rhythm_summary = (
        f"{int(normal_fraction * 100)}% normal/conducted beats; "
        f"{int(ventricular_fraction * 100)}% ventricular ectopic/fusion; "
        f"{int(supraventricular_fraction * 100)}% supraventricular ectopic; "
        f"RR CV {rr_cv:.3f}; invalid/extreme RR {invalid_rr_count}; noise events {noise_events}"
    )
    segment_id = f"mitbih_{record.record_id}_{int(round(start_s)):04d}_{int(round(end_s)):04d}"
    return Segment(
        segment_id=segment_id,
        record_id=record.record_id,
        intended_category=intended_category,
        start_s=float(start_s),
        end_s=float(end_s),
        duration_s=duration_s,
        rr_ms=rr_ms,
        rr_times_s=rr_times_s,
        beat_symbols=rr_symbols,
        rhythm_summary=rhythm_summary,
        stats=stats,
    )


def candidate_score(segment: Segment, category: str) -> float:
    stats = segment.stats
    if category == "mostly normal rhythm":
        return (
            stats["normal_fraction"] * 4.0
            - stats["ectopic_fraction"] * 6.0
            - stats["artifact_like_fraction"] * 3.0
            - stats["noise_event_count"] * 0.2
            - abs(stats["rr_cv"] - 0.08)
        )
    if category == "PVC/ectopic-heavy rhythm":
        return (
            stats["ventricular_fraction"] * 6.0
            + stats["ectopic_fraction"] * 3.0
            + stats["symbol_diversity"] * 0.05
            + stats["rr_cv"]
        )
    if category == "irregular rhythm":
        return (
            stats["rr_cv"] * 4.0
            + stats["symbol_diversity"] * 0.12
            + stats["ectopic_fraction"]
            - stats["noise_event_count"] * 0.03
        )
    if category == "noisy/problematic rhythm":
        return (
            stats["noise_event_count"] * 2.0
            + stats["artifact_like_fraction"] * 10.0
            + stats["rr_cv"] * 2.0
            + stats["symbol_diversity"] * 0.05
        )
    return 0.0


def build_candidates(
    records: Sequence[RecordAnnotations],
    segment_duration_s: float,
    step_s: float,
    min_duration_s: float,
    min_rr_count: int,
) -> List[Segment]:
    candidates: List[Segment] = []
    for record in records:
        intended_category = PREFERRED_RECORDS.get(record.record_id, "irregular rhythm")
        max_start = max(0.0, record.duration_s - segment_duration_s)
        starts = np.arange(0.0, max_start + 1.0, step_s)
        if starts.size == 0:
            starts = np.array([0.0])
        for start_s in starts:
            segment = summarize_segment(
                record,
                start_s=float(start_s),
                end_s=float(start_s + segment_duration_s),
                intended_category=intended_category,
            )
            if segment is None:
                continue
            if segment.duration_s < min_duration_s or len(segment.rr_ms) < min_rr_count:
                continue
            candidates.append(segment)
    return candidates


def select_segments(candidates: Sequence[Segment], max_segments: int) -> List[Segment]:
    selected: List[Segment] = []
    used_records: set[str] = set()
    used_ids: set[str] = set()

    for category, count in TARGET_CATEGORY_COUNTS.items():
        category_candidates = sorted(
            candidates,
            key=lambda item: candidate_score(item, category),
            reverse=True,
        )
        chosen = 0
        for segment in category_candidates:
            if segment.segment_id in used_ids:
                continue
            if segment.intended_category != category and chosen < max(1, count - 1):
                continue
            if segment.record_id in used_records and chosen < count - 1:
                continue
            selected.append(segment)
            used_records.add(segment.record_id)
            used_ids.add(segment.segment_id)
            chosen += 1
            if chosen >= count or len(selected) >= max_segments:
                break
        if len(selected) >= max_segments:
            break

    if len(selected) < min(max_segments, 10):
        filler = sorted(
            candidates,
            key=lambda item: (
                item.record_id in used_records,
                -candidate_score(item, item.intended_category),
            ),
        )
        for segment in filler:
            if segment.segment_id in used_ids:
                continue
            selected.append(segment)
            used_ids.add(segment.segment_id)
            used_records.add(segment.record_id)
            if len(selected) >= max_segments:
                break

    return selected[:max_segments]


def flatten_warning_messages(recorded: Iterable[warnings.WarningMessage]) -> List[str]:
    messages: List[str] = []
    seen: set[str] = set()
    for item in recorded:
        text = str(item.message)
        if text and text not in seen:
            seen.add(text)
            messages.append(text)
    return messages


def frequency_warning_labels(freq_results: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    diagnostics = freq_results.get("frequency_diagnostics", {}) if freq_results else {}
    if diagnostics.get("duration_warnings"):
        labels.append("duration_warning")
    if diagnostics.get("nonfinite_psd"):
        labels.append("nonfinite_psd")
    if diagnostics.get("invalid_rr_removed_count", 0):
        labels.append("invalid_rr_removed")
    for method in ["welch", "fft", "ar"]:
        method_diag = diagnostics.get(method, {})
        if method_diag.get("warning"):
            labels.append(f"{method}_warning")
        if method_diag.get("fallback_reason"):
            labels.append(f"{method}_fallback")
        for band, band_diag in method_diag.get("band_diagnostics", {}).items():
            for warning_text in band_diag.get("warnings", []):
                text = str(warning_text).lower()
                if "fewer than 2 psd bins" in text:
                    labels.append(f"{method}_{band}_few_bins")
                if "insufficient" in text:
                    labels.append(f"{method}_{band}_duration")
    return sorted(set(labels))


def finite_metric_values(values: Dict[str, float]) -> bool:
    return all(np.isfinite(value) for value in values.values())


def run_single_analysis(
    rr_input: np.ndarray,
    correction_enabled: bool,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], np.ndarray]:
    from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
    from hrvlib.metrics.time_domain import HRVTimeDomainAnalysis
    from hrvlib.preprocessing import preprocess_rri

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                preprocessing = preprocess_rri(
                    rr_input.tolist(),
                    correction_method="cubic_spline",
                    noise_detection=True,
                    artifact_correction_enabled=correction_enabled,
                )
            rr_for_metrics = preprocessing.corrected_rri
            time_results = HRVTimeDomainAnalysis(
                rr_for_metrics, preprocessing_result=preprocessing
            ).full_analysis()
            freq = HRVFreqDomainAnalysis(
                rr_for_metrics,
                preprocessing_result=preprocessing,
                sampling_rate=args.interpolation_rate,
                detrend_method=normalize_detrend(args.detrend_method),
                window_type=args.window_type,
                segment_length=args.segment_length,
                overlap_ratio=args.overlap_ratio,
                enable_diagnostics=True,
            )
            freq_results = freq.get_results()
            analysis_error = ""
        except Exception as exc:
            preprocessing = None
            rr_for_metrics = np.asarray([], dtype=float)
            time_results = {}
            freq_results = {}
            analysis_error = f"{type(exc).__name__}: {exc}"

    metric_values = {
        "SDNN": float(time_results.get("sdnn", math.nan)),
        "RMSSD": float(time_results.get("rmssd", math.nan)),
        "LF": float(freq_results.get("welch_lf_power", math.nan)),
        "HF": float(freq_results.get("welch_hf_power", math.nan)),
        "total_power": float(freq_results.get("welch_total_power", math.nan)),
        "LF/HF": float(freq_results.get("welch_lf_hf_ratio", math.nan)),
        "LF_nu": float(freq_results.get("welch_lf_nu", math.nan)),
        "HF_nu": float(freq_results.get("welch_hf_nu", math.nan)),
    }

    warning_messages = flatten_warning_messages(captured)
    warning_labels = frequency_warning_labels(freq_results)
    input_invalid_count = int(
        np.count_nonzero(
            (~np.isfinite(rr_input)) | (rr_input <= 0) | (rr_input < 300.0) | (rr_input > 2000.0)
        )
    )
    if preprocessing is not None:
        quality_flags = preprocessing.quality_flags or {}
        if quality_flags.get("poor_signal_quality"):
            warning_labels.append("poor_signal_quality")
        if quality_flags.get("excessive_artifacts"):
            warning_labels.append("excessive_artifacts")
        if quality_flags.get("high_noise"):
            warning_labels.append("high_noise")
        if quality_flags.get("irregular_rhythm"):
            warning_labels.append("irregular_rhythm")
        if preprocessing.noise_segments:
            warning_labels.append("noise_segments_detected")
        stats = preprocessing.stats
        artifact_indices = preprocessing.artifact_indices
    else:
        quality_flags = {}
        stats = {}
        artifact_indices = []
        if analysis_error:
            warning_labels.append("analysis_error")
            warning_messages.append(analysis_error)

    if warning_messages:
        warning_labels.append("runtime_warnings")

    row: Dict[str, Any] = {
        "phase": "after_correction" if correction_enabled else "before_correction",
        "correction_enabled": bool(correction_enabled),
        "input_count": int(len(rr_input)),
        "input_invalid_rr_count": input_invalid_count,
        "preprocessed_count": int(len(rr_for_metrics)),
        "correction_count": int(stats.get("artifacts_corrected", 0)),
        "artifacts_detected": int(stats.get("artifacts_detected", 0)),
        "artifacts_corrected": int(stats.get("artifacts_corrected", 0)),
        "artifact_percentage": float(stats.get("artifact_percentage", 0.0)),
        "extra_beats_removed": int(stats.get("extra_beats_removed", 0)),
        "intervals_interpolated": int(stats.get("intervals_interpolated", 0)),
        "noise_segments": int(len(getattr(preprocessing, "noise_segments", []) or [])),
        "noise_percentage": float(stats.get("noise_percentage", 0.0)),
        "poor_signal_quality": bool(quality_flags.get("poor_signal_quality", False)),
        "excessive_artifacts": bool(quality_flags.get("excessive_artifacts", False)),
        "high_noise": bool(quality_flags.get("high_noise", False)),
        "irregular_rhythm_flag": bool(quality_flags.get("irregular_rhythm", False)),
        "metrics_finite": finite_metric_values(metric_values),
        "finite_output_status": "finite" if finite_metric_values(metric_values) else "nonfinite_or_failed",
        "warning_labels": "; ".join(sorted(set(warning_labels))),
        "warning_messages": " | ".join(warning_messages),
        "artifact_indices_sample": ";".join(str(i) for i in list(artifact_indices)[:25]),
    }
    row.update(metric_values)
    return row, rr_for_metrics


def relative_change(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before) or abs(before) < 1e-12:
        return math.nan
    return float((after - before) / abs(before) * 100.0)


def write_manifest(segments: Sequence[Segment], run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for segment in segments:
        row = {
            "segment_id": segment.segment_id,
            "record_id": segment.record_id,
            "intended_category": segment.intended_category,
            "start_s": segment.start_s,
            "end_s": segment.end_s,
            "duration_s": segment.duration_s,
            "rr_count": int(len(segment.rr_ms)),
            "rhythm_summary": segment.rhythm_summary,
        }
        row.update({k: v for k, v in segment.stats.items() if k != "symbol_counts"})
        row["symbol_counts_json"] = json.dumps(segment.stats.get("symbol_counts", {}), sort_keys=True)
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(run_dir / "mitbih_segment_manifest.csv", index=False)
    return manifest


def build_artifact_effect_table(results: pd.DataFrame) -> pd.DataFrame:
    effect_rows: List[Dict[str, Any]] = []
    after = results[results["phase"] == "after_correction"].copy()
    before = results[results["phase"] == "before_correction"].copy()
    for _, after_row in after.iterrows():
        matches = before[before["segment_id"] == after_row["segment_id"]]
        if matches.empty:
            continue
        before_row = matches.iloc[0]
        for metric in METRICS:
            change = relative_change(float(after_row[metric]), float(before_row[metric]))
            effect_rows.append(
                {
                    "segment_id": after_row["segment_id"],
                    "record_id": after_row["record_id"],
                    "intended_category": after_row["intended_category"],
                    "metric": metric,
                    "before_value": before_row[metric],
                    "after_value": after_row[metric],
                    "after_vs_before_change_pct": change,
                    "abs_after_vs_before_change_pct": abs(change) if np.isfinite(change) else math.nan,
                    "correction_count": after_row["correction_count"],
                    "artifacts_detected": after_row["artifacts_detected"],
                    "artifact_percentage": after_row["artifact_percentage"],
                }
            )
    return pd.DataFrame(effect_rows)


def build_warning_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (category, phase), group in results.groupby(["intended_category", "phase"], sort=True):
        labels: Dict[str, int] = {}
        for text in group["warning_labels"].fillna(""):
            for label in [part.strip() for part in text.split(";") if part.strip()]:
                labels[label] = labels.get(label, 0) + 1
        if not labels:
            rows.append(
                {
                    "intended_category": category,
                    "phase": phase,
                    "warning_label": "none",
                    "segment_count": int(len(group)),
                    "segments_with_label": 0,
                }
            )
        else:
            for label, count in sorted(labels.items()):
                rows.append(
                    {
                        "intended_category": category,
                        "phase": phase,
                        "warning_label": label,
                        "segment_count": int(len(group)),
                        "segments_with_label": int(count),
                    }
                )
    return pd.DataFrame(rows)


def plot_tachograms(
    segments: Sequence[Segment],
    corrected_rr_by_segment: Dict[str, np.ndarray],
    run_dir: Path,
    max_plots: int = 6,
) -> List[str]:
    selected: List[Segment] = []
    for category in TARGET_CATEGORY_COUNTS:
        matches = [segment for segment in segments if segment.intended_category == category]
        if matches:
            selected.append(matches[0])
    for segment in segments:
        if len(selected) >= max_plots:
            break
        if segment.segment_id not in {item.segment_id for item in selected}:
            selected.append(segment)

    output_files: List[str] = []
    for segment in selected[:max_plots]:
        corrected = corrected_rr_by_segment.get(segment.segment_id, np.asarray([], dtype=float))
        fig, ax = plt.subplots(figsize=(9.0, 3.6))
        ax.plot(
            np.arange(len(segment.rr_ms)),
            segment.rr_ms,
            linewidth=0.85,
            color="#245f73",
            label="Before correction",
        )
        if corrected.size:
            ax.plot(
                np.arange(len(corrected)),
                corrected,
                linewidth=0.75,
                color="#b45f2a",
                alpha=0.82,
                label="After correction",
            )
        ax.set_title(f"{segment.record_id}: {segment.intended_category}")
        ax.set_xlabel("RR interval index")
        ax.set_ylabel("RR interval (ms)")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        safe_name = f"{segment.segment_id}_tachogram.png"
        fig.savefig(run_dir / "tachograms" / safe_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_files.append(f"tachograms/{safe_name}")
    return output_files


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if pd.isna(value) else f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(
    run_dir: Path,
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    warning_summary: pd.DataFrame,
    effect_table: pd.DataFrame,
    tachogram_files: Sequence[str],
    args: argparse.Namespace,
    dataset_zip: Path,
    record_root: Path,
) -> None:
    before = results[results["phase"] == "before_correction"]
    after = results[results["phase"] == "after_correction"]
    before_finite_count = int(before["metrics_finite"].sum())
    after_finite_count = int(after["metrics_finite"].sum())
    before_rate = before_finite_count / len(before) * 100.0 if len(before) else 0.0
    after_rate = after_finite_count / len(after) * 100.0 if len(after) else 0.0
    warning_rate = after["warning_labels"].fillna("").ne("").mean() * 100.0 if len(after) else 0.0
    corrected_rate = (after["correction_count"] > 0).mean() * 100.0 if len(after) else 0.0

    sensitivity = (
        effect_table.groupby("metric")["abs_after_vs_before_change_pct"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"abs_after_vs_before_change_pct": "median_abs_correction_change_pct"})
    )
    category_effect = (
        effect_table.groupby("intended_category")["abs_after_vs_before_change_pct"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"abs_after_vs_before_change_pct": "median_abs_correction_change_pct"})
    )
    sensitive_metrics = ", ".join(sensitivity.head(3)["metric"].tolist()) if not sensitivity.empty else "not determined"
    stable_metrics = ", ".join(sensitivity.tail(3)["metric"].tolist()) if not sensitivity.empty else "not determined"
    categories = manifest["intended_category"].value_counts().rename_axis("category").reset_index(name="segments")

    summary = f"""# MIT-BIH Arrhythmia Robustness/QC Stress Test

This study is a robustness/QC stress test of HRV Studio behavior under non-normal rhythm conditions. It is not a clinical validation, diagnostic-accuracy study, or equivalence study.

## Run Configuration

- Dataset archive: `{dataset_zip}`
- Extracted record root: `{record_root}`
- Output directory: `{run_dir}`
- Segment target: {args.segment_duration_s:.0f} seconds when possible
- Segments analyzed: {len(after)}
- Frequency-domain settings: {args.interpolation_rate:.1f} Hz interpolation, {args.segment_length:.0f}s Welch segments, {args.overlap_ratio:.2f} overlap, `{args.window_type}` window, detrend `{args.detrend_method}`, band convention `{BAND_CONVENTION}`

## Segment Mix

{md_table(categories)}

## Headline Results

- Finite metric rows before correction: {before_finite_count}/{len(before)} ({before_rate:.1f}%).
- Finite metric rows after correction: {after_finite_count}/{len(after)} ({after_rate:.1f}%).
- After-correction segments with warning/diagnostic labels: {int(after["warning_labels"].fillna("").ne("").sum())}/{len(after)} ({warning_rate:.1f}%).
- After-correction segments with at least one corrected interval: {int((after["correction_count"] > 0).sum())}/{len(after)} ({corrected_rate:.1f}%).

## Metric Sensitivity to Correction

Median absolute before/after correction change:

{md_table(sensitivity)}

Category-level median absolute correction change:

{md_table(category_effect)}

## Warning Summary

{md_table(warning_summary)}

## Answers to Study Questions

### 1. Does HRV Studio remain numerically stable on arrhythmic RR data?

In this run, HRV Studio remained operational on the selected MIT-BIH arrhythmic RR segments, with finite requested metrics in {after_finite_count}/{len(after)} after-correction rows. Any non-finite rows should be interpreted as stress-test failures or edge cases, not clinical findings.

### 2. Do warnings/diagnostics appropriately flag problematic rhythm segments?

Warnings and diagnostics flagged {int(after["warning_labels"].fillna("").ne("").sum())}/{len(after)} after-correction segments. The labels mainly reflect artifact burden, irregular rhythm flags, noise/noisy-segment detection, runtime warnings, and frequency-domain diagnostics. These are useful QC indicators, but they do not prove that a finite metric is physiologically valid.

### 3. Which metrics are most sensitive to arrhythmia and correction?

The largest median absolute correction effects were observed for: {sensitive_metrics}. The smallest median correction effects were observed for: {stable_metrics}. Frequency-domain powers and LF/HF-derived quantities should be interpreted especially cautiously because ectopic intervals, interpolation, and irregular timing can materially change spectral area.

### 4. Does artifact correction meaningfully stabilize outputs?

Artifact correction made HRV Studio outputs more QC-transparent by exposing correction counts and before/after metric shifts. It did not make arrhythmic recordings equivalent to normal sinus rhythm. The `mitbih_artifact_effect_table.csv` file should be used to describe the magnitude and direction of correction effects.

### 5. Should arrhythmic recordings be excluded from primary validation?

Yes. Arrhythmic MIT-BIH recordings should be excluded from primary agreement/validation analyses that aim to evaluate normal-rhythm HRV metric agreement. They are appropriate as a separate robustness/QC stress-test cohort because arrhythmia changes the physiological meaning of standard HRV metrics and can inflate sensitivity to preprocessing choices.

### 6. What limitation statement belongs in the paper?

Suggested wording: "MIT-BIH arrhythmia recordings were used only for robustness and quality-control stress testing under non-normal rhythm conditions. These analyses evaluate numerical stability, warning behavior, and sensitivity to artifact correction, but they do not establish clinical equivalence, diagnostic accuracy, or validity of standard HRV metrics during arrhythmia. Primary validation results should therefore be interpreted for the intended rhythm and data-quality conditions, with arrhythmic recordings reported separately or excluded from agreement claims."

## Representative Tachograms

{chr(10).join(f"- `{path}`" for path in tachogram_files)}

## Output Files

- `mitbih_robustness_results.csv`
- `mitbih_segment_manifest.csv`
- `mitbih_warning_summary.csv`
- `mitbih_artifact_effect_table.csv`
- `tachograms/*_tachogram.png`
- `mitbih_robustness_summary.md`
"""
    (run_dir / "mitbih_robustness_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = discover_hrv_project_root(args.hrv_project_root)
    import_hrv_studio(project_root)
    wfdb = require_wfdb()

    args.archive_dir = args.archive_dir.resolve()
    output_root = args.output_root.resolve()
    run_dir = output_root / "validation" / "runs" / args.run_name
    dataset_dir = output_root / "validation" / "raw_data" / DATASET_DIR_NAME
    prepare_run_dir(run_dir)

    dataset_zip = find_mitbih_zip(args.archive_dir, args.zip_pattern)
    record_root = extract_dataset_if_needed(dataset_zip, dataset_dir)
    record_ids = read_records(record_root)
    preferred = [record_id for record_id in PREFERRED_RECORDS if record_id in record_ids]
    remaining = [record_id for record_id in record_ids if record_id not in preferred]
    load_ids = preferred + remaining

    loaded_records = [
        record
        for record in (
            load_record_annotations(wfdb, record_root, record_id) for record_id in load_ids
        )
        if record is not None
    ]
    if not loaded_records:
        raise SystemExit(f"No MIT-BIH annotation records could be loaded from {record_root}")

    candidates = build_candidates(
        loaded_records,
        segment_duration_s=args.segment_duration_s,
        step_s=args.window_step_s,
        min_duration_s=args.min_segment_duration_s,
        min_rr_count=args.min_rr_count,
    )
    if len(candidates) < 10:
        raise SystemExit(
            f"Only {len(candidates)} eligible MIT-BIH segments found; expected at least 10."
        )
    segments = select_segments(candidates, max_segments=args.max_segments)
    if not 10 <= len(segments) <= 15:
        raise SystemExit(f"Selected {len(segments)} segments; expected 10-15.")

    manifest = write_manifest(segments, run_dir)

    rows: List[Dict[str, Any]] = []
    corrected_rr_by_segment: Dict[str, np.ndarray] = {}
    for segment in segments:
        for correction_enabled in [False, True]:
            row, rr_for_metrics = run_single_analysis(segment.rr_ms, correction_enabled, args)
            row.update(
                {
                    "segment_id": segment.segment_id,
                    "record_id": segment.record_id,
                    "intended_category": segment.intended_category,
                    "rhythm_summary": segment.rhythm_summary,
                    "rr_count": int(len(segment.rr_ms)),
                    "duration_s": segment.duration_s,
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                }
            )
            rows.append(row)
            if correction_enabled:
                corrected_rr_by_segment[segment.segment_id] = rr_for_metrics

    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "mitbih_robustness_results.csv", index=False)

    warning_summary = build_warning_summary(results)
    warning_summary.to_csv(run_dir / "mitbih_warning_summary.csv", index=False)

    effect_table = build_artifact_effect_table(results)
    effect_table.to_csv(run_dir / "mitbih_artifact_effect_table.csv", index=False)

    tachogram_files = plot_tachograms(segments, corrected_rr_by_segment, run_dir, max_plots=6)

    run_info = {
        "script": "tools/mitbih_arrhythmia_robustness_study.py",
        "validation_only": True,
        "production_code_modified": False,
        "dataset_zip": str(dataset_zip),
        "record_root": str(record_root),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "segments": int(len(segments)),
        "settings": {
            "band_convention": BAND_CONVENTION,
            "segment_duration_s": args.segment_duration_s,
            "window_step_s": args.window_step_s,
            "interpolation_rate": args.interpolation_rate,
            "segment_length": args.segment_length,
            "overlap_ratio": args.overlap_ratio,
            "window_type": args.window_type,
            "detrend_method": args.detrend_method,
        },
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    write_summary(
        run_dir,
        results,
        manifest,
        warning_summary,
        effect_table,
        tachogram_files,
        args,
        dataset_zip,
        record_root,
    )

    print(f"Wrote MIT-BIH robustness study outputs to {run_dir}")
    print(f"Segments: {len(segments)}")
    print(f"Rows in mitbih_robustness_results.csv: {len(results)}")


if __name__ == "__main__":
    main()
