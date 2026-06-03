"""Prepare PhysioNet NSR RR validation data.

This validation utility converts the downloaded PhysioNet Normal Sinus Rhythm
RR Interval Database into standardized RR CSV files and non-overlapping segment
sets for frequency-domain validation. It does not modify production HRV Studio
analysis code or raw PhysioNet files.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "validation"
    / "raw_data"
    / "normal-sinus-rhythm-rr-interval-database-1.0.0"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "validation" / "processed_data"

RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0
MIN_SEGMENT_DURATION_RATIO = 0.90
MIN_RR_PER_MINUTE = 30
RR_DIRECT_SUFFIXES = {".csv", ".txt", ".rr", ".rri", ".nn"}
WFDB_META_SUFFIXES = {".hea"}


@dataclass
class DatasetInspection:
    input_dir: Path
    file_type_counts: Dict[str, int]
    records: List[str]
    annotation_extensions: List[str]
    direct_rr_candidates: List[Path]
    header_count: int
    signal_count_estimate: Optional[int]
    available_signals: List[str]
    rr_directly_available: bool
    annotation_conversion_needed: bool
    assumptions: List[str]


@dataclass
class ExtractionResult:
    record_id: str
    source_file: Path
    raw_rr_ms: np.ndarray
    raw_start_s: np.ndarray
    clean_rr_ms: np.ndarray
    clean_start_s: np.ndarray
    removed_invalid: int
    removal_reasons: Dict[str, int]
    extraction_method: str
    annotation_extension: str = ""

    @property
    def duration_seconds(self) -> float:
        if len(self.clean_rr_ms) == 0:
            return 0.0
        return float(self.clean_start_s[-1] + self.clean_rr_ms[-1] / 1000.0 - self.clean_start_s[0])

    @property
    def percent_removed(self) -> float:
        total = len(self.raw_rr_ms)
        return (self.removed_invalid / total * 100.0) if total else 0.0


@dataclass
class SegmentResult:
    record_id: str
    segment_id: str
    source_record: Path
    output_file: Path
    duration_seconds: float
    rr_ms: np.ndarray
    start_s: np.ndarray
    percent_invalid_removed: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the PhysioNet Normal Sinus Rhythm RR Interval Database for "
            "HRV Studio validation and later Kubios comparison."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--segment-lengths", type=int, nargs="+", default=[300, 600])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def require_input_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Input directory does not exist: {path}")


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    targets = [
        path / "physionet_nsr_rr",
        path / "physionet_nsr_rr_5min",
        path / "physionet_nsr_rr_10min",
    ]
    existing = [target for target in targets if target.exists() and any(target.iterdir())]
    if existing and not overwrite:
        formatted = "\n".join(str(target) for target in existing)
        raise SystemExit(
            "Prepared output folders already contain files. Use --overwrite to regenerate:\n"
            f"{formatted}"
        )
    if overwrite:
        for target in targets:
            if target.exists():
                for child in target.glob("*.csv"):
                    child.unlink()


def safe_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def format_float(value: object, digits: int = 6) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def read_records_file(input_dir: Path) -> List[str]:
    records_path = input_dir / "RECORDS"
    if not records_path.exists():
        return []
    records: List[str] = []
    for line in records_path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            records.append(text)
    return records


def read_annotators(input_dir: Path) -> List[str]:
    annotators_path = input_dir / "ANNOTATORS"
    if not annotators_path.exists():
        return []
    extensions: List[str] = []
    for line in annotators_path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        extensions.append(text.split()[0])
    return extensions


def inspect_header_signal_count(header_path: Path) -> Optional[int]:
    try:
        first_line = header_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (IndexError, OSError):
        return None
    parts = first_line.split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def discover_direct_rr_candidates(input_dir: Path, records: Sequence[str]) -> List[Path]:
    record_names = set(records)
    candidates: List[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in RR_DIRECT_SUFFIXES:
            continue
        if path.name.upper() in {"RECORDS", "ANNOTATORS"} or path.name.upper().startswith("SHA256"):
            continue
        if path.stem in record_names or suffix in {".rr", ".rri", ".nn", ".csv"}:
            candidates.append(path)
    return candidates


def inspect_dataset(input_dir: Path) -> DatasetInspection:
    files = [path for path in input_dir.iterdir() if path.is_file()]
    file_type_counts: Dict[str, int] = {}
    for path in files:
        key = path.suffix.lower() if path.suffix else "(none)"
        file_type_counts[key] = file_type_counts.get(key, 0) + 1

    records = read_records_file(input_dir)
    if not records:
        records = sorted(path.stem for path in input_dir.glob("*.hea"))

    annotation_extensions = read_annotators(input_dir)
    direct_rr_candidates = discover_direct_rr_candidates(input_dir, records)
    headers = sorted(input_dir.glob("*.hea"))
    signal_counts = [
        count
        for count in (inspect_header_signal_count(path) for path in headers)
        if count is not None
    ]
    signal_count_estimate = signal_counts[0] if signal_counts else None
    available_signals = []
    if signal_count_estimate and signal_count_estimate > 0:
        available_signals.append("waveform signals declared in .hea headers")
    if annotation_extensions:
        available_signals.append("WFDB beat annotations: " + ", ".join(annotation_extensions))
    if not available_signals:
        available_signals.append("not determined")

    rr_directly_available = bool(direct_rr_candidates)
    annotation_conversion_needed = not rr_directly_available and bool(annotation_extensions)
    assumptions = [
        "RR intervals are measured in milliseconds in prepared CSV outputs.",
        "Intervals with RR <= 0, RR < 300 ms, or RR > 2000 ms are excluded from prepared validation files and counted in manifests.",
        "Prepared cumulative_time_seconds values preserve original annotation timing for full-record outputs.",
        "Segment files reset cumulative_time_seconds to the segment-local first retained interval while preserving RR interval order.",
    ]
    if annotation_conversion_needed:
        assumptions.append(
            "No direct RR interval files were detected, so RR intervals are computed from consecutive WFDB annotation sample positions."
        )
    if signal_count_estimate == 0:
        assumptions.append(
            "Headers declare zero waveform signals; this is treated as an annotation-only PhysioNet RR dataset."
        )

    return DatasetInspection(
        input_dir=input_dir,
        file_type_counts=file_type_counts,
        records=records,
        annotation_extensions=annotation_extensions,
        direct_rr_candidates=direct_rr_candidates,
        header_count=len(headers),
        signal_count_estimate=signal_count_estimate,
        available_signals=available_signals,
        rr_directly_available=rr_directly_available,
        annotation_conversion_needed=annotation_conversion_needed,
        assumptions=assumptions,
    )


def load_direct_rr(path: Path) -> Tuple[np.ndarray, str]:
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python")
    lower_cols = {str(col).lower(): col for col in df.columns}
    column = None
    for candidate in ["rr_ms", "rri_ms", "rr", "rri", "nn", "ibi", "interval"]:
        if candidate in lower_cols:
            column = lower_cols[candidate]
            break
    if column is None:
        column = df.columns[0]
    values = [safe_float(value) for value in df[column].tolist()]
    rr = np.asarray([value for value in values if value is not None], dtype=float)
    return rr, f"direct_rr:{path.name}"


def load_wfdb_annotation(input_dir: Path, record_id: str, annotation_extensions: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, str, str]:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("wfdb is required to extract RR intervals from PhysioNet annotations.") from exc

    errors: List[str] = []
    for extension in annotation_extensions or ["ecg"]:
        try:
            annotation = wfdb.rdann(str(input_dir / record_id), extension)
        except Exception as exc:  # pragma: no cover - exact wfdb errors vary by version
            errors.append(f"{extension}: {exc}")
            continue

        samples = np.asarray(annotation.sample, dtype=float)
        fs = safe_float(getattr(annotation, "fs", None))
        if fs is None or fs <= 0:
            fs = infer_fs_from_header(input_dir / f"{record_id}.hea")
        if fs is None or fs <= 0:
            raise RuntimeError(f"Cannot determine sampling frequency for {record_id}.{extension}")
        if len(samples) < 2:
            raise RuntimeError(f"Fewer than two annotations for {record_id}.{extension}")

        rr_ms = np.diff(samples) / fs * 1000.0
        starts_s = samples[:-1] / fs
        return rr_ms.astype(float), starts_s.astype(float), f"{record_id}.{extension}", extension

    joined = "; ".join(errors) if errors else "no annotation extensions available"
    raise RuntimeError(f"Could not read WFDB annotation for {record_id}: {joined}")


def infer_fs_from_header(header_path: Path) -> Optional[float]:
    try:
        first_line = header_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (IndexError, OSError):
        return None
    parts = first_line.split()
    if len(parts) < 3:
        return None
    return safe_float(parts[2])


def clean_rr(rr_ms: np.ndarray, starts_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    reasons = {
        "rr_non_finite": int(np.count_nonzero(~np.isfinite(rr_ms))),
        "rr_le_0": int(np.count_nonzero(np.isfinite(rr_ms) & (rr_ms <= 0))),
        "rr_lt_300_ms": int(np.count_nonzero(np.isfinite(rr_ms) & (rr_ms > 0) & (rr_ms < RR_MIN_MS))),
        "rr_gt_2000_ms": int(np.count_nonzero(np.isfinite(rr_ms) & (rr_ms > RR_MAX_MS))),
    }
    valid = np.isfinite(rr_ms) & (rr_ms > 0) & (rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)
    return rr_ms[valid], starts_s[valid], reasons


def extract_record(input_dir: Path, record_id: str, inspection: DatasetInspection) -> ExtractionResult:
    direct_by_stem = {path.stem: path for path in inspection.direct_rr_candidates}
    if record_id in direct_by_stem:
        rr_ms, method = load_direct_rr(direct_by_stem[record_id])
        starts_s = np.concatenate([[0.0], np.cumsum(rr_ms[:-1]) / 1000.0]) if len(rr_ms) else np.array([])
        source_file = direct_by_stem[record_id]
        annotation_extension = ""
    else:
        rr_ms, starts_s, source_name, annotation_extension = load_wfdb_annotation(
            input_dir,
            record_id,
            inspection.annotation_extensions,
        )
        source_file = input_dir / source_name
        method = "wfdb_annotation"

    clean_values, clean_starts, reasons = clean_rr(rr_ms, starts_s)
    removed = int(len(rr_ms) - len(clean_values))
    return ExtractionResult(
        record_id=record_id,
        source_file=source_file,
        raw_rr_ms=rr_ms,
        raw_start_s=starts_s,
        clean_rr_ms=clean_values,
        clean_start_s=clean_starts,
        removed_invalid=removed,
        removal_reasons=reasons,
        extraction_method=method,
        annotation_extension=annotation_extension,
    )


def cumulative_from_rr(rr_ms: np.ndarray) -> np.ndarray:
    if len(rr_ms) == 0:
        return np.array([], dtype=float)
    return np.concatenate([[0.0], np.cumsum(rr_ms[:-1]) / 1000.0])


def write_rr_csv(path: Path, rr_ms: np.ndarray, cumulative_time_s: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rr_ms", "cumulative_time_seconds"])
        writer.writeheader()
        for rr, time_s in zip(rr_ms, cumulative_time_s):
            writer.writerow(
                {
                    "rr_ms": format_float(rr, 6),
                    "cumulative_time_seconds": format_float(time_s, 6),
                }
            )


def stat_or_blank(values: np.ndarray, fn) -> str:
    if len(values) == 0:
        return ""
    return format_float(fn(values), 6)


def manifest_row(result: ExtractionResult, output_file: Path) -> Dict[str, object]:
    return {
        "record_id": result.record_id,
        "source_file": str(result.source_file),
        "output_file": str(output_file),
        "extraction_method": result.extraction_method,
        "annotation_extension": result.annotation_extension,
        "number_raw_intervals": len(result.raw_rr_ms),
        "number_removed_invalid": result.removed_invalid,
        "percent_removed": format_float(result.percent_removed, 6),
        "mean_rr": stat_or_blank(result.clean_rr_ms, np.mean),
        "median_rr": stat_or_blank(result.clean_rr_ms, np.median),
        "min_rr": stat_or_blank(result.clean_rr_ms, np.min),
        "max_rr": stat_or_blank(result.clean_rr_ms, np.max),
        "duration_seconds": format_float(result.duration_seconds, 6),
        **result.removal_reasons,
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def segment_record(
    result: ExtractionResult,
    source_record: Path,
    output_dir: Path,
    segment_length_s: int,
) -> Tuple[List[SegmentResult], List[Dict[str, object]]]:
    segments: List[SegmentResult] = []
    exclusions: List[Dict[str, object]] = []
    if len(result.clean_rr_ms) == 0:
        exclusions.append(
            {
                "record_id": result.record_id,
                "segment_length_seconds": segment_length_s,
                "reason": "no clean RR intervals",
            }
        )
        return segments, exclusions

    min_duration = segment_length_s * MIN_SEGMENT_DURATION_RATIO
    min_rr_count = int((segment_length_s / 60.0) * MIN_RR_PER_MINUTE)
    recording_start = float(result.clean_start_s[0])
    recording_end = float(result.clean_start_s[-1] + result.clean_rr_ms[-1] / 1000.0)
    window_start = recording_start
    index = 0

    while window_start < recording_end:
        window_end = window_start + segment_length_s
        mask = (result.clean_start_s >= window_start) & (result.clean_start_s < window_end)
        rr = result.clean_rr_ms[mask]
        starts = result.clean_start_s[mask]
        if len(rr) == 0:
            exclusions.append(
                {
                    "record_id": result.record_id,
                    "segment_length_seconds": segment_length_s,
                    "window_start_seconds": format_float(window_start, 6),
                    "reason": "empty segment window",
                }
            )
            window_start = window_end
            continue

        duration = float(starts[-1] + rr[-1] / 1000.0 - starts[0])
        if duration < min_duration:
            exclusions.append(
                {
                    "record_id": result.record_id,
                    "segment_length_seconds": segment_length_s,
                    "window_start_seconds": format_float(window_start, 6),
                    "duration_seconds": format_float(duration, 6),
                    "number_rr_intervals": len(rr),
                    "reason": "duration below 90 percent of target",
                }
            )
            window_start = window_end
            continue
        if len(rr) < min_rr_count:
            exclusions.append(
                {
                    "record_id": result.record_id,
                    "segment_length_seconds": segment_length_s,
                    "window_start_seconds": format_float(window_start, 6),
                    "duration_seconds": format_float(duration, 6),
                    "number_rr_intervals": len(rr),
                    "reason": "too few RR intervals",
                }
            )
            window_start = window_end
            continue

        segment_id = f"segment_{index:03d}"
        output_file = output_dir / f"{result.record_id}_{segment_id}.csv"
        local_starts = starts - starts[0]
        write_rr_csv(output_file, rr, local_starts)
        segments.append(
            SegmentResult(
                record_id=result.record_id,
                segment_id=segment_id,
                source_record=source_record,
                output_file=output_file,
                duration_seconds=duration,
                rr_ms=rr,
                start_s=local_starts,
                percent_invalid_removed=result.percent_removed,
            )
        )
        index += 1
        window_start = window_end

    return segments, exclusions


def segment_manifest_row(segment: SegmentResult) -> Dict[str, object]:
    return {
        "record_id": segment.record_id,
        "segment_id": segment.segment_id,
        "source_record": str(segment.source_record),
        "output_file": str(segment.output_file),
        "duration_seconds": format_float(segment.duration_seconds, 6),
        "number_rr_intervals": len(segment.rr_ms),
        "mean_rr_ms": stat_or_blank(segment.rr_ms, np.mean),
        "std_rr_ms": stat_or_blank(segment.rr_ms, np.std),
        "min_rr_ms": stat_or_blank(segment.rr_ms, np.min),
        "max_rr_ms": stat_or_blank(segment.rr_ms, np.max),
        "percent_invalid_removed": format_float(segment.percent_invalid_removed, 6),
    }


def write_dataset_summary(path: Path, inspection: DatasetInspection, extraction_rows: List[Dict[str, object]]) -> None:
    direct = "yes" if inspection.rr_directly_available else "no"
    conversion = "yes" if inspection.annotation_conversion_needed else "no"
    lines = [
        "# PhysioNet NSR Dataset Preparation Summary",
        "",
        "This summary is generated for validation data preparation only. It does not modify raw PhysioNet files or production HRV Studio analysis code.",
        "",
        "## Dataset Inspection",
        "",
        f"- Input directory: `{inspection.input_dir}`",
        f"- Number of recordings: {len(inspection.records)}",
        f"- Header files detected: {inspection.header_count}",
        f"- Estimated signal count from headers: {inspection.signal_count_estimate if inspection.signal_count_estimate is not None else 'not determined'}",
        f"- RR intervals directly available: {direct}",
        f"- Conversion from beat annotations needed: {conversion}",
        "",
        "## File Types",
        "",
    ]
    for suffix, count in sorted(inspection.file_type_counts.items()):
        lines.append(f"- `{suffix}`: {count}")
    lines.extend(["", "## Available Signals or Annotation Information", ""])
    for item in inspection.available_signals:
        lines.append(f"- {item}")
    lines.extend(["", "## Direct RR Candidates", ""])
    if inspection.direct_rr_candidates:
        for path_item in inspection.direct_rr_candidates:
            lines.append(f"- `{path_item.name}`")
    else:
        lines.append("- None detected.")
    lines.extend(["", "## Assumptions Made", ""])
    for assumption in inspection.assumptions:
        lines.append(f"- {assumption}")
    if extraction_rows:
        total_raw = sum(int(row["number_raw_intervals"]) for row in extraction_rows)
        total_removed = sum(int(row["number_removed_invalid"]) for row in extraction_rows)
        lines.extend(
            [
                "",
                "## Extraction Overview",
                "",
                f"- Records extracted: {len(extraction_rows)}",
                f"- Raw RR intervals: {total_raw}",
                f"- Invalid intervals removed: {total_removed}",
                f"- Overall percent removed: {(total_removed / total_raw * 100.0) if total_raw else 0.0:.3f}%",
            ]
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=float), q))


def write_quality_report(
    path: Path,
    extraction_results: List[ExtractionResult],
    segment_rows_by_length: Dict[int, List[Dict[str, object]]],
    exclusions: List[Dict[str, object]],
) -> None:
    durations = [result.duration_seconds for result in extraction_results if result.duration_seconds > 0]
    rr_values = np.concatenate([result.clean_rr_ms for result in extraction_results if len(result.clean_rr_ms)]) if extraction_results else np.array([])
    excessive_invalid = [result for result in extraction_results if result.percent_removed >= 5.0]
    suspicious = [
        result
        for result in extraction_results
        if result.percent_removed >= 5.0
        or result.duration_seconds < 300.0
        or len(result.clean_rr_ms) < 100
    ]
    total_segments = sum(len(rows) for rows in segment_rows_by_length.values())

    lines = [
        "# PhysioNet NSR Quality Report",
        "",
        "This report is descriptive and intended for validation review only. It does not claim that excluded or flagged files are scientifically unusable.",
        "",
        "## Dataset-Level Statistics",
        "",
        f"- Number of recordings extracted: {len(extraction_results)}",
        f"- Total segments generated: {total_segments}",
    ]
    for length_s, rows in sorted(segment_rows_by_length.items()):
        lines.append(f"- {length_s // 60 if length_s % 60 == 0 else length_s}-minute segment count: {len(rows)}")
    lines.extend(
        [
            f"- Recording duration seconds: min {format_float(min(durations), 3) if durations else 'not available'}, median {format_float(statistics.median(durations), 3) if durations else 'not available'}, max {format_float(max(durations), 3) if durations else 'not available'}",
            f"- RR distribution ms: min {format_float(np.min(rr_values), 3) if len(rr_values) else 'not available'}, median {format_float(np.median(rr_values), 3) if len(rr_values) else 'not available'}, max {format_float(np.max(rr_values), 3) if len(rr_values) else 'not available'}",
            f"- RR distribution ms p05/p95: {format_float(percentile(rr_values.tolist(), 5), 3) if len(rr_values) else 'not available'} / {format_float(percentile(rr_values.tolist(), 95), 3) if len(rr_values) else 'not available'}",
            "",
            "## Files Excluded or Segment Windows Skipped",
            "",
        ]
    )
    if exclusions:
        reason_counts: Dict[str, int] = {}
        for row in exclusions:
            reason = str(row.get("reason", "unknown"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- No segment-window exclusions were logged.")

    lines.extend(["", "## Suspicious or Review-Worthy Files", ""])
    if suspicious:
        for result in suspicious:
            lines.append(
                f"- `{result.record_id}`: removed {result.removed_invalid}/{len(result.raw_rr_ms)} "
                f"({result.percent_removed:.2f}%), duration {result.duration_seconds:.1f}s"
            )
    else:
        lines.append("- No files crossed the simple review thresholds used by this script.")

    lines.extend(["", "## Excessive Invalid RR Removal Check", ""])
    if excessive_invalid:
        for result in excessive_invalid:
            lines.append(f"- `{result.record_id}`: {result.percent_removed:.2f}% invalid intervals removed")
    else:
        lines.append("- No recordings had 5% or more invalid intervals removed.")

    lines.extend(["", "## Unusual Duration Issues", ""])
    if exclusions:
        examples = exclusions[:20]
        for row in examples:
            lines.append(
                "- "
                + ", ".join(f"{key}={value}" for key, value in row.items())
            )
        if len(exclusions) > len(examples):
            lines.append(f"- Additional skipped windows not shown: {len(exclusions) - len(examples)}")
    else:
        lines.append("- No duration-related segment exclusions were logged.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def folder_for_segment_length(output_dir: Path, segment_length_s: int) -> Path:
    if segment_length_s == 300:
        return output_dir / "physionet_nsr_rr_5min"
    if segment_length_s == 600:
        return output_dir / "physionet_nsr_rr_10min"
    label = f"{segment_length_s}s"
    if segment_length_s % 60 == 0:
        label = f"{segment_length_s // 60}min"
    return output_dir / f"physionet_nsr_rr_{label}"


def manifest_name_for_segment_length(segment_length_s: int) -> str:
    if segment_length_s == 300:
        return "5min_manifest.csv"
    if segment_length_s == 600:
        return "10min_manifest.csv"
    return f"{segment_length_s}s_manifest.csv"


def print_validation_commands() -> None:
    print("")
    print("READY FOR VALIDATION")
    print("")
    print("Example commands:")
    print(
        "python tools\\validate_freq_domain_neurokit2.py "
        "validation\\processed_data\\physionet_nsr_rr_5min --recursive "
        "--run-name v03_physionet_5min_neurokit2 --enable-diagnostics --detrend-method linear"
    )
    print(
        "python tools\\validate_freq_domain_neurokit2.py "
        "validation\\processed_data\\physionet_nsr_rr_10min --recursive "
        "--run-name v03_physionet_10min_neurokit2 --enable-diagnostics --detrend-method linear"
    )


def run(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    require_input_dir(input_dir)
    ensure_output_dir(output_dir, args.overwrite)

    inspection = inspect_dataset(input_dir)
    if not inspection.records:
        raise SystemExit("No recordings could be identified from RECORDS or .hea files.")

    rr_dir = output_dir / "physionet_nsr_rr"
    extraction_results: List[ExtractionResult] = []
    extraction_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for index, record_id in enumerate(inspection.records, start=1):
        if args.verbose:
            print(f"[{index}/{len(inspection.records)}] Extracting {record_id}")
        else:
            print(f"Extracting {record_id}")
        try:
            result = extract_record(input_dir, record_id, inspection)
        except Exception as exc:  # pragma: no cover - depends on local dataset state
            failures.append({"record_id": record_id, "reason": str(exc)})
            print(f"  skipped: {exc}")
            continue

        output_file = rr_dir / f"{record_id}_rr.csv"
        write_rr_csv(output_file, result.clean_rr_ms, result.clean_start_s)
        extraction_results.append(result)
        extraction_rows.append(manifest_row(result, output_file))

    write_csv(
        rr_dir / "rr_extraction_manifest.csv",
        extraction_rows,
        [
            "record_id",
            "source_file",
            "output_file",
            "extraction_method",
            "annotation_extension",
            "number_raw_intervals",
            "number_removed_invalid",
            "percent_removed",
            "mean_rr",
            "median_rr",
            "min_rr",
            "max_rr",
            "duration_seconds",
            "rr_non_finite",
            "rr_le_0",
            "rr_lt_300_ms",
            "rr_gt_2000_ms",
        ],
    )
    write_dataset_summary(
        output_dir / "physionet_nsr_dataset_summary.md",
        inspection,
        extraction_rows,
    )

    segment_rows_by_length: Dict[int, List[Dict[str, object]]] = {}
    all_exclusions: List[Dict[str, object]] = []
    for segment_length_s in args.segment_lengths:
        segment_dir = folder_for_segment_length(output_dir, int(segment_length_s))
        rows: List[Dict[str, object]] = []
        for result in extraction_results:
            source_record = rr_dir / f"{result.record_id}_rr.csv"
            segments, exclusions = segment_record(
                result,
                source_record,
                segment_dir,
                int(segment_length_s),
            )
            rows.extend(segment_manifest_row(segment) for segment in segments)
            all_exclusions.extend(exclusions)
        segment_rows_by_length[int(segment_length_s)] = rows
        write_csv(segment_dir / manifest_name_for_segment_length(int(segment_length_s)), rows)
        print(f"Wrote {len(rows)} segments to {segment_dir}")

    if failures:
        all_exclusions.extend({"segment_length_seconds": "", **failure} for failure in failures)
        write_csv(output_dir / "physionet_nsr_extraction_failures.csv", failures)

    write_csv(output_dir / "physionet_nsr_segment_exclusions.csv", all_exclusions)
    write_quality_report(
        output_dir / "physionet_nsr_quality_report.md",
        extraction_results,
        segment_rows_by_length,
        all_exclusions,
    )

    print(f"Wrote RR extraction manifest to {rr_dir / 'rr_extraction_manifest.csv'}")
    print(f"Wrote dataset summary to {output_dir / 'physionet_nsr_dataset_summary.md'}")
    print(f"Wrote quality report to {output_dir / 'physionet_nsr_quality_report.md'}")
    print_validation_commands()
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
