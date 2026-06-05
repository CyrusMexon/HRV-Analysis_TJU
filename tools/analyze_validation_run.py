"""
Analyze a frequency-domain NeuroKit2 validation run.

This utility reads validation/runs/<run_name>/ outputs and writes a run-level
diagnostic summary. It does not import or modify HRV analysis code.

Example:
    python tools/analyze_validation_run.py --run-name v02_with_diagnostics
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "validation" / "runs"

METRICS = ["VLF", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu"]
ABSOLUTE_POWER_METRICS = {"VLF", "LF", "HF", "total_power"}
NORMALIZED_METRICS = {"LF/HF", "LF_nu", "HF_nu"}
OUTLIER_THRESHOLDS = (20.0, 50.0, 80.0)
DURATION_BUCKETS = (
    ("<60s", None, 60.0),
    ("60-120s", 60.0, 120.0),
    ("120-300s", 120.0, 300.0),
    (">=300s", 300.0, None),
)
WARNING_KEYWORDS = {
    "insufficient duration": ("insufficient", "duration"),
    "fewer than 2 PSD bins": ("fewer than 2 psd bins",),
    "ULF/VLF overlap": ("ulf", "vlf", "overlap"),
    "FFT variance mismatch": ("fft", "variance"),
    "AR fallback": ("ar fallback", "fallback"),
}
LONG_DURATION_OUTLIER_COLUMNS = [
    "source_file",
    "metric",
    "duration_seconds",
    "native_value",
    "neurokit2_value",
    "absolute_error",
    "relative_error_pct",
    "native_bins_total",
    "neurokit2_bins_total",
    "native_metric_band_bins",
    "neurokit2_metric_band_bins",
    "detrend_method",
    "welch_detrend_mode",
    "warnings",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create diagnostic_summary.md/csv for a validation run."
    )
    parser.add_argument("--run-name", required=True, help="Run folder under validation/runs/.")
    parser.add_argument(
        "--runs-root",
        default=str(RUNS_ROOT),
        help="Directory containing validation run folders.",
    )
    parser.add_argument(
        "--top-outliers",
        type=int,
        default=25,
        help="Maximum outlier/manual-review rows to include in Markdown tables.",
    )
    parser.add_argument(
        "--absolute-power-high-threshold",
        type=float,
        default=20.0,
        help="Relative-error threshold used to flag high absolute-power mismatch.",
    )
    parser.add_argument(
        "--normalized-low-threshold",
        type=float,
        default=10.0,
        help="Relative-error threshold used to flag aligned normalized metrics.",
    )
    parser.add_argument(
        "--no-append-notes",
        action="store_true",
        help="Write summary files but do not append the short summary to notes.md.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key)) for key in fieldnames})


def write_csv_selected(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key)) for key in fieldnames})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)
    return "" if value is None else value


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else math.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return math.nan
    try:
        value_float = float(text)
    except ValueError:
        return math.nan
    return value_float if math.isfinite(value_float) else math.nan


def clean_metric(metric: Any) -> str:
    text = str(metric).strip()
    aliases = {
        "vlf": "VLF",
        "lf": "LF",
        "hf": "HF",
        "lf_hf": "LF/HF",
        "lf/hf": "LF/HF",
        "lf_nu": "LF_nu",
        "hf_nu": "HF_nu",
        "total": "total_power",
        "total_power": "total_power",
    }
    return aliases.get(text.lower(), text)


def format_number(value: Any, digits: int = 2) -> str:
    value_float = to_float(value)
    if math.isnan(value_float):
        return ""
    return f"{value_float:.{digits}f}"


def format_pct(value: Any, digits: int = 2) -> str:
    text = format_number(value, digits)
    return f"{text}%" if text else ""


def basename(path_like: Any) -> str:
    text = str(path_like or "")
    if not text:
        return ""
    return Path(text).name


def percentile(values: Sequence[float], pct: float) -> float:
    cleaned = sorted(value for value in values if math.isfinite(value))
    if not cleaned:
        return math.nan
    if len(cleaned) == 1:
        return cleaned[0]
    position = (len(cleaned) - 1) * (pct / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return cleaned[int(position)]
    fraction = position - lower
    return cleaned[lower] + (cleaned[upper] - cleaned[lower]) * fraction


def mean(values: Sequence[float]) -> float:
    cleaned = [value for value in values if math.isfinite(value)]
    if not cleaned:
        return math.nan
    return sum(cleaned) / len(cleaned)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def parse_list_like(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"[]", "nan", "None"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except Exception:
                    continue
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
        return [text]
    return [str(value).strip()]


def extract_warnings_from_obj(obj: Any) -> List[str]:
    warnings: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lower_key = key.lower()
            if lower_key.endswith("warnings") or lower_key.endswith("warning"):
                warnings.extend(parse_list_like(value))
            elif key in {"fallback_reason", "ar_fallback_reason"} and value:
                warnings.append(f"AR fallback: {value}")
            warnings.extend(extract_warnings_from_obj(value))
    elif isinstance(obj, list):
        for item in obj:
            warnings.extend(extract_warnings_from_obj(item))
    return dedupe(warnings)


def dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalize_file_key(value: Any) -> str:
    return str(value or "").strip()


def load_diagnostics(run_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], str]:
    json_path = run_dir / "diagnostics.json"
    csv_path = run_dir / "diagnostics.csv"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        diagnostics: Dict[str, Dict[str, Any]] = {}
        for item in payload if isinstance(payload, list) else []:
            file_key = normalize_file_key(item.get("input_file"))
            if not file_key:
                continue
            freq = item.get("frequency_diagnostics", {})
            diagnostics[file_key] = {
                "duration_seconds": to_float(freq.get("duration_seconds")),
                "warnings": extract_warnings_from_obj(freq),
                "source": "diagnostics.json",
            }
        return diagnostics, "diagnostics.json"

    if csv_path.exists():
        diagnostics = {}
        for row in read_csv(csv_path):
            file_key = normalize_file_key(row.get("input_file"))
            if not file_key:
                continue
            warnings: List[str] = []
            for key, value in row.items():
                lower_key = key.lower()
                if lower_key.endswith("warnings") or lower_key.endswith("warning"):
                    warnings.extend(parse_list_like(value))
                elif "fallback_reason" in lower_key and value:
                    warnings.append(f"AR fallback: {value}")
            diagnostics[file_key] = {
                "duration_seconds": to_float(
                    row.get("frequency_diagnostics_duration_seconds")
                ),
                "warnings": dedupe(warnings),
                "source": "diagnostics.csv",
            }
        return diagnostics, "diagnostics.csv"

    return {}, "none"


def enrich_rows(
    validation_rows: List[Dict[str, str]], diagnostics: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in validation_rows:
        metric = clean_metric(row.get("metric"))
        if metric not in METRICS:
            continue
        file_key = normalize_file_key(row.get("input_file"))
        diagnostic = diagnostics.get(file_key, {})
        warnings = diagnostic.get("warnings", [])
        duration = first_finite(
            row.get("duration_seconds"),
            row.get("native_recording_duration_s"),
            row.get("neurokit2_recording_duration_s"),
            diagnostic.get("duration_seconds"),
        )
        enriched = dict(row)
        enriched.update(
            {
                "metric": metric,
                "file_name": basename(file_key),
                "duration_seconds": duration,
                "native_value": to_float(row.get("native_value")),
                "neurokit2_value": to_float(row.get("neurokit2_value")),
                "absolute_error": to_float(row.get("absolute_error")),
                "relative_error_pct": to_float(row.get("relative_error_pct")),
                "native_bins_total": to_float(row.get("native_bins_total")),
                "neurokit2_bins_total": to_float(row.get("neurokit2_bins_total")),
                "native_metric_band_bins": to_float(row.get("native_metric_band_bins")),
                "neurokit2_metric_band_bins": to_float(row.get("neurokit2_metric_band_bins")),
                "detrend_method": row.get("detrend_method", ""),
                "welch_detrend_mode": row.get("welch_detrend_mode", ""),
                "warnings": warnings,
                "experimental_native_value": to_float(row.get("experimental_native_value")),
                "experimental_absolute_error": to_float(row.get("experimental_absolute_error")),
                "experimental_relative_error_pct": to_float(
                    row.get("experimental_relative_error_pct")
                ),
                "experimental_native_metric_band_bins": to_float(
                    row.get("experimental_native_metric_band_bins")
                ),
                "experimental_native_bins_total": to_float(
                    row.get("experimental_native_bins_total")
                ),
                "experimental_native_welch_nfft": to_float(
                    row.get("experimental_native_welch_nfft")
                ),
                "experimental_native_welch_nfft_multiplier": to_float(
                    row.get("experimental_native_welch_nfft_multiplier")
                ),
            }
        )
        rows.append(enriched)
    return rows


def first_finite(*values: Any) -> float:
    for value in values:
        value_float = to_float(value)
        if math.isfinite(value_float):
            return value_float
    return math.nan


def metric_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_metric: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_metric[row["metric"]].append(row)

    summaries: List[Dict[str, Any]] = []
    for metric in METRICS:
        metric_rows = by_metric.get(metric, [])
        rel = [row["relative_error_pct"] for row in metric_rows]
        abs_err = [row["absolute_error"] for row in metric_rows]
        summaries.append(
            {
                "section": "metric_level_agreement",
                "metric": metric,
                "rows": len(metric_rows),
                "mean_relative_error_pct": mean(rel),
                "median_relative_error_pct": median(rel),
                "p75_relative_error_pct": percentile(rel, 75.0),
                "p90_relative_error_pct": percentile(rel, 90.0),
                "max_relative_error_pct": percentile(rel, 100.0),
                "mean_absolute_error": mean(abs_err),
                "median_absolute_error": median(abs_err),
                "outliers_gt_20_pct": count_gt(rel, 20.0),
                "outliers_gt_50_pct": count_gt(rel, 50.0),
                "outliers_gt_80_pct": count_gt(rel, 80.0),
            }
        )
    return summaries


def welch_mode_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter = Counter(
        str(row.get("welch_detrend_mode") or "not recorded")
        for row in rows
    )
    return [
        {
            "section": "welch_detrend_mode",
            "welch_detrend_mode": mode,
            "rows": count,
            "files": len(
                {
                    normalize_file_key(row.get("input_file"))
                    for row in rows
                    if str(row.get("welch_detrend_mode") or "not recorded") == mode
                }
            ),
        }
        for mode, count in counter.most_common()
    ]


def has_experimental_columns(rows: List[Dict[str, Any]]) -> bool:
    return any(math.isfinite(row.get("experimental_relative_error_pct", math.nan)) for row in rows)


def nfft_alignment_metric_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_metric: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if math.isfinite(row.get("experimental_relative_error_pct", math.nan)):
            by_metric[row["metric"]].append(row)

    summaries: List[Dict[str, Any]] = []
    for metric in METRICS:
        metric_rows = by_metric.get(metric, [])
        baseline = [row["relative_error_pct"] for row in metric_rows]
        experimental = [row["experimental_relative_error_pct"] for row in metric_rows]
        long_rows = [
            row
            for row in metric_rows
            if math.isfinite(row["duration_seconds"]) and row["duration_seconds"] >= 300.0
        ]
        long_baseline = [row["relative_error_pct"] for row in long_rows]
        long_experimental = [row["experimental_relative_error_pct"] for row in long_rows]
        summaries.append(
            {
                "section": "nfft_alignment_metric",
                "metric": metric,
                "rows": len(metric_rows),
                "mean_relative_error_pct_before": mean(baseline),
                "mean_relative_error_pct_after": mean(experimental),
                "mean_relative_error_pct_change": mean(experimental) - mean(baseline),
                "median_relative_error_pct_before": median(baseline),
                "median_relative_error_pct_after": median(experimental),
                "median_relative_error_pct_change": median(experimental) - median(baseline),
                "long_duration_rows": len(long_rows),
                "long_duration_mean_relative_error_pct_before": mean(long_baseline),
                "long_duration_mean_relative_error_pct_after": mean(long_experimental),
                "long_duration_mean_relative_error_pct_change": (
                    mean(long_experimental) - mean(long_baseline)
                ),
                "long_duration_median_relative_error_pct_before": median(long_baseline),
                "long_duration_median_relative_error_pct_after": median(long_experimental),
                "long_duration_median_relative_error_pct_change": (
                    median(long_experimental) - median(long_baseline)
                ),
            }
        )
    return summaries


def nfft_alignment_bin_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    experimental_rows = [
        row for row in rows if math.isfinite(row.get("experimental_relative_error_pct", math.nan))
    ]
    total_rows = len(experimental_rows)
    baseline_total = [
        row
        for row in experimental_rows
        if comparable(row["native_bins_total"], row["neurokit2_bins_total"])
        and row["native_bins_total"] != row["neurokit2_bins_total"]
    ]
    experimental_total = [
        row
        for row in experimental_rows
        if comparable(row["experimental_native_bins_total"], row["neurokit2_bins_total"])
        and row["experimental_native_bins_total"] != row["neurokit2_bins_total"]
    ]
    baseline_band = [
        row
        for row in experimental_rows
        if comparable(row["native_metric_band_bins"], row["neurokit2_metric_band_bins"])
        and row["native_metric_band_bins"] != row["neurokit2_metric_band_bins"]
    ]
    experimental_band = [
        row
        for row in experimental_rows
        if comparable(row["experimental_native_metric_band_bins"], row["neurokit2_metric_band_bins"])
        and row["experimental_native_metric_band_bins"] != row["neurokit2_metric_band_bins"]
    ]
    return [
        {
            "section": "nfft_alignment_bin_count",
            "row_type": "total_bin_mismatch_before",
            "rows": len(baseline_total),
            "denominator": total_rows,
            "rate_pct": safe_rate(len(baseline_total), total_rows),
        },
        {
            "section": "nfft_alignment_bin_count",
            "row_type": "total_bin_mismatch_after",
            "rows": len(experimental_total),
            "denominator": total_rows,
            "rate_pct": safe_rate(len(experimental_total), total_rows),
        },
        {
            "section": "nfft_alignment_bin_count",
            "row_type": "metric_band_bin_mismatch_before",
            "rows": len(baseline_band),
            "denominator": total_rows,
            "rate_pct": safe_rate(len(baseline_band), total_rows),
        },
        {
            "section": "nfft_alignment_bin_count",
            "row_type": "metric_band_bin_mismatch_after",
            "rows": len(experimental_band),
            "denominator": total_rows,
            "rate_pct": safe_rate(len(experimental_band), total_rows),
        },
    ]


def normalized_metric_stability(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for metric in METRICS:
        metric_rows = [
            row
            for row in rows
            if row["metric"] == metric
            and math.isfinite(row.get("experimental_native_value", math.nan))
            and math.isfinite(row.get("native_value", math.nan))
        ]
        value_delta = [
            abs(row["experimental_native_value"] - row["native_value"])
            for row in metric_rows
        ]
        relative_value_delta = [
            relative_error(row["experimental_native_value"], row["native_value"]) * 100.0
            for row in metric_rows
        ]
        result.append(
            {
                "section": "nfft_alignment_value_stability",
                "metric": metric,
                "metric_group": (
                    "normalized_or_ratio" if metric in NORMALIZED_METRICS else "absolute_power"
                ),
                "rows": len(metric_rows),
                "mean_native_value_abs_change": mean(value_delta),
                "median_native_value_abs_change": median(value_delta),
                "mean_native_value_relative_change_pct": mean(relative_value_delta),
                "median_native_value_relative_change_pct": median(relative_value_delta),
            }
        )
    return result


def relative_error(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference) or abs(reference) <= 1e-12:
        return math.nan
    return abs(value - reference) / abs(reference)


def count_gt(values: Sequence[float], threshold: float) -> int:
    return sum(1 for value in values if math.isfinite(value) and value > threshold)


def outlier_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in rows:
        rel = row["relative_error_pct"]
        if not math.isfinite(rel) or rel <= min(OUTLIER_THRESHOLDS):
            continue
        threshold = max(threshold for threshold in OUTLIER_THRESHOLDS if rel > threshold)
        result.append(
            {
                "section": "outlier",
                "threshold": f">{threshold:.0f}%",
                "file_name": row["file_name"],
                "input_file": row.get("input_file", ""),
                "duration_seconds": row["duration_seconds"],
                "metric": row["metric"],
                "native_value": row["native_value"],
                "neurokit2_value": row["neurokit2_value"],
                "absolute_error": row["absolute_error"],
                "relative_error_pct": rel,
                "native_bins_total": row["native_bins_total"],
                "neurokit2_bins_total": row["neurokit2_bins_total"],
                "native_metric_band_bins": row["native_metric_band_bins"],
                "neurokit2_metric_band_bins": row["neurokit2_metric_band_bins"],
                "detrend_method": row["detrend_method"],
                "welch_detrend_mode": row["welch_detrend_mode"],
                "warnings": row["warnings"],
            }
        )
    result.sort(key=lambda item: to_float(item["relative_error_pct"]), reverse=True)
    return result


def duration_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for label, low, high in DURATION_BUCKETS:
        bucket_rows = [
            row
            for row in rows
            if math.isfinite(row["duration_seconds"])
            and (low is None or row["duration_seconds"] >= low)
            and (high is None or row["duration_seconds"] < high)
        ]
        rel = [row["relative_error_pct"] for row in bucket_rows]
        summaries.append(
            {
                "section": "duration_effect",
                "duration_bucket": label,
                "rows": len(bucket_rows),
                "mean_relative_error_pct": mean(rel),
                "median_relative_error_pct": median(rel),
                "p75_relative_error_pct": percentile(rel, 75.0),
                "p90_relative_error_pct": percentile(rel, 90.0),
                "max_relative_error_pct": percentile(rel, 100.0),
                "outliers_gt_20_pct": count_gt(rel, 20.0),
                "outliers_gt_50_pct": count_gt(rel, 50.0),
                "outliers_gt_80_pct": count_gt(rel, 80.0),
            }
        )
    return summaries


def long_duration_outlier_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outliers: List[Dict[str, Any]] = []
    for row in rows:
        if not (
            math.isfinite(row["duration_seconds"])
            and row["duration_seconds"] >= 300.0
            and math.isfinite(row["relative_error_pct"])
            and row["relative_error_pct"] > 50.0
        ):
            continue
        outliers.append(
            {
                "source_file": row.get("input_file", ""),
                "file_name": row["file_name"],
                "metric": row["metric"],
                "duration_seconds": row["duration_seconds"],
                "native_value": row["native_value"],
                "neurokit2_value": row["neurokit2_value"],
                "absolute_error": row["absolute_error"],
                "relative_error_pct": row["relative_error_pct"],
                "native_bins_total": row["native_bins_total"],
                "neurokit2_bins_total": row["neurokit2_bins_total"],
                "native_metric_band_bins": row["native_metric_band_bins"],
                "neurokit2_metric_band_bins": row["neurokit2_metric_band_bins"],
                "detrend_method": row["detrend_method"],
                "welch_detrend_mode": row["welch_detrend_mode"],
                "warnings": row["warnings"],
            }
        )
    outliers.sort(key=lambda item: to_float(item["relative_error_pct"]), reverse=True)
    return outliers


def long_duration_outlier_metric_counts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter = Counter(row["metric"] for row in rows)
    return [
        {"metric": metric, "rows": count}
        for metric, count in counter.most_common()
    ]


def bin_mismatch_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total_rows = len(rows)
    total_mismatch = [
        row
        for row in rows
        if comparable(row["native_bins_total"], row["neurokit2_bins_total"])
        and row["native_bins_total"] != row["neurokit2_bins_total"]
    ]
    band_mismatch = [
        row
        for row in rows
        if comparable(row["native_metric_band_bins"], row["neurokit2_metric_band_bins"])
        and row["native_metric_band_bins"] != row["neurokit2_metric_band_bins"]
    ]
    high_error = [
        row
        for row in rows
        if math.isfinite(row["relative_error_pct"]) and row["relative_error_pct"] > 20.0
    ]
    not_high_error = [
        row
        for row in rows
        if math.isfinite(row["relative_error_pct"]) and row["relative_error_pct"] <= 20.0
    ]
    high_with_total_mismatch = [
        row
        for row in high_error
        if comparable(row["native_bins_total"], row["neurokit2_bins_total"])
        and row["native_bins_total"] != row["neurokit2_bins_total"]
    ]
    high_with_band_mismatch = [
        row
        for row in high_error
        if comparable(row["native_metric_band_bins"], row["neurokit2_metric_band_bins"])
        and row["native_metric_band_bins"] != row["neurokit2_metric_band_bins"]
    ]
    low_with_band_mismatch = [
        row
        for row in not_high_error
        if comparable(row["native_metric_band_bins"], row["neurokit2_metric_band_bins"])
        and row["native_metric_band_bins"] != row["neurokit2_metric_band_bins"]
    ]
    return [
        {
            "section": "bin_count_effect",
            "row_type": "all_rows_total_bin_mismatch",
            "rows": len(total_mismatch),
            "denominator": total_rows,
            "rate_pct": safe_rate(len(total_mismatch), total_rows),
        },
        {
            "section": "bin_count_effect",
            "row_type": "all_rows_metric_band_bin_mismatch",
            "rows": len(band_mismatch),
            "denominator": total_rows,
            "rate_pct": safe_rate(len(band_mismatch), total_rows),
        },
        {
            "section": "bin_count_effect",
            "row_type": "high_error_rows_total_bin_mismatch",
            "rows": len(high_with_total_mismatch),
            "denominator": len(high_error),
            "rate_pct": safe_rate(len(high_with_total_mismatch), len(high_error)),
        },
        {
            "section": "bin_count_effect",
            "row_type": "high_error_rows_metric_band_bin_mismatch",
            "rows": len(high_with_band_mismatch),
            "denominator": len(high_error),
            "rate_pct": safe_rate(len(high_with_band_mismatch), len(high_error)),
        },
        {
            "section": "bin_count_effect",
            "row_type": "low_error_rows_metric_band_bin_mismatch",
            "rows": len(low_with_band_mismatch),
            "denominator": len(not_high_error),
            "rate_pct": safe_rate(len(low_with_band_mismatch), len(not_high_error)),
        },
    ]


def comparable(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right)


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan
    return 100.0 * numerator / denominator


def absolute_vs_normalized_flags(
    rows: List[Dict[str, Any]], high_threshold: float, low_threshold: float
) -> List[Dict[str, Any]]:
    return absolute_vs_normalized_flags_for(
        rows,
        high_threshold,
        low_threshold,
        rel_error_key="relative_error_pct",
        section="absolute_power_vs_normalized",
    )


def absolute_vs_normalized_flags_for(
    rows: List[Dict[str, Any]],
    high_threshold: float,
    low_threshold: float,
    rel_error_key: str,
    section: str,
) -> List[Dict[str, Any]]:
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[normalize_file_key(row.get("input_file"))].append(row)

    flags: List[Dict[str, Any]] = []
    for file_key, file_rows in by_file.items():
        by_metric = {row["metric"]: row for row in file_rows}
        abs_errors = [
            by_metric[metric].get(rel_error_key, math.nan)
            for metric in ABSOLUTE_POWER_METRICS
            if metric in by_metric and math.isfinite(by_metric[metric].get(rel_error_key, math.nan))
        ]
        normalized_errors = [
            by_metric[metric].get(rel_error_key, math.nan)
            for metric in NORMALIZED_METRICS
            if metric in by_metric and math.isfinite(by_metric[metric].get(rel_error_key, math.nan))
        ]
        if not abs_errors or len(normalized_errors) < len(NORMALIZED_METRICS):
            continue
        max_abs = max(abs_errors)
        max_norm = max(normalized_errors)
        if max_abs > high_threshold and max_norm <= low_threshold:
            representative = by_metric.get("LF") or file_rows[0]
            flags.append(
                {
                    "section": section,
                    "file_name": basename(file_key),
                    "input_file": file_key,
                    "duration_seconds": representative["duration_seconds"],
                    "max_absolute_power_relative_error_pct": max_abs,
                    "max_normalized_relative_error_pct": max_norm,
                    "mean_absolute_power_relative_error_pct": mean(abs_errors),
                    "mean_normalized_relative_error_pct": mean(normalized_errors),
                    "interpretation": (
                        "absolute power mismatch with preserved relative spectral distribution"
                    ),
                    "warnings": representative["warnings"],
                }
            )
    flags.sort(
        key=lambda item: (
            to_float(item["max_absolute_power_relative_error_pct"]),
            -to_float(item["max_normalized_relative_error_pct"]),
        ),
        reverse=True,
    )
    return flags


def warning_summary(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    by_file: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        file_key = normalize_file_key(row.get("input_file"))
        by_file[file_key].extend(row.get("warnings", []))

    warning_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    for warnings in by_file.values():
        for warning in dedupe(warnings):
            warning_counter[warning] += 1
            lower = warning.lower()
            for category, required_terms in WARNING_KEYWORDS.items():
                if all(term in lower for term in required_terms):
                    category_counter[category] += 1

    summary_rows: List[Dict[str, Any]] = []
    for category in WARNING_KEYWORDS:
        summary_rows.append(
            {
                "section": "warning_category",
                "warning": category,
                "files": category_counter[category],
            }
        )
    for warning, count in warning_counter.most_common():
        summary_rows.append({"section": "warning", "warning": warning, "files": count})
    return summary_rows, warning_counter


def manual_review_rows(
    outliers: List[Dict[str, Any]],
    absolute_vs_normalized: List[Dict[str, Any]],
    top_outliers: int,
) -> List[Dict[str, Any]]:
    review: Dict[str, Dict[str, Any]] = {}
    for outlier in outliers:
        file_key = normalize_file_key(outlier.get("input_file"))
        current = review.setdefault(
            file_key,
            {
                "section": "manual_review",
                "file_name": outlier["file_name"],
                "input_file": file_key,
                "duration_seconds": outlier["duration_seconds"],
                "reasons": [],
                "max_relative_error_pct": 0.0,
                "warnings": outlier["warnings"],
            },
        )
        current["max_relative_error_pct"] = max(
            to_float(current["max_relative_error_pct"]),
            to_float(outlier["relative_error_pct"]),
        )
        if to_float(outlier["relative_error_pct"]) > 80.0:
            current["reasons"].append(f"{outlier['metric']} relative error >80%")
        elif to_float(outlier["relative_error_pct"]) > 50.0:
            current["reasons"].append(f"{outlier['metric']} relative error >50%")
    for flag in absolute_vs_normalized:
        file_key = normalize_file_key(flag.get("input_file"))
        current = review.setdefault(
            file_key,
            {
                "section": "manual_review",
                "file_name": flag["file_name"],
                "input_file": file_key,
                "duration_seconds": flag["duration_seconds"],
                "reasons": [],
                "max_relative_error_pct": flag["max_absolute_power_relative_error_pct"],
                "warnings": flag["warnings"],
            },
        )
        current["reasons"].append("absolute power mismatch with preserved relative distribution")

    rows = list(review.values())
    for row in rows:
        row["reasons"] = dedupe(row["reasons"])
    rows.sort(key=lambda item: to_float(item["max_relative_error_pct"]), reverse=True)
    return rows[:top_outliers]


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[Tuple[str, str]], limit: Optional[int] = None) -> str:
    selected = rows[:limit] if limit is not None else rows
    if not selected:
        return "_None found._\n"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in selected:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = format_number(value)
            elif isinstance(value, list):
                value = "; ".join(value)
            text = str(value).replace("\n", " ").replace("|", "/")
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_markdown(
    run_name: str,
    diagnostics_source: str,
    rows: List[Dict[str, Any]],
    welch_mode_rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    outliers: List[Dict[str, Any]],
    duration_rows: List[Dict[str, Any]],
    long_duration_outliers: List[Dict[str, Any]],
    bin_rows: List[Dict[str, Any]],
    abs_norm_rows: List[Dict[str, Any]],
    warning_rows: List[Dict[str, Any]],
    manual_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    nfft_metric_rows: Optional[List[Dict[str, Any]]] = None,
    nfft_bin_rows: Optional[List[Dict[str, Any]]] = None,
    nfft_stability_rows: Optional[List[Dict[str, Any]]] = None,
    nfft_abs_norm_before: Optional[List[Dict[str, Any]]] = None,
    nfft_abs_norm_after: Optional[List[Dict[str, Any]]] = None,
) -> str:
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    total_files = len({normalize_file_key(row.get("input_file")) for row in rows})
    high_error_rows = sum(
        1 for row in rows if math.isfinite(row["relative_error_pct"]) and row["relative_error_pct"] > 20.0
    )
    very_high_error_rows = sum(
        1 for row in rows if math.isfinite(row["relative_error_pct"]) and row["relative_error_pct"] > 80.0
    )
    abs_norm_count = len(abs_norm_rows)
    bin_band_row = next(
        row for row in bin_rows if row.get("row_type") == "all_rows_metric_band_bin_mismatch"
    )
    bin_high_row = next(
        row for row in bin_rows if row.get("row_type") == "high_error_rows_metric_band_bin_mismatch"
    )
    worst_metric = max(
        summaries,
        key=lambda row: to_float(row.get("p90_relative_error_pct")),
    )
    nfft_metric_rows = nfft_metric_rows or []
    nfft_bin_rows = nfft_bin_rows or []
    nfft_stability_rows = nfft_stability_rows or []
    nfft_abs_norm_before = nfft_abs_norm_before or []
    nfft_abs_norm_after = nfft_abs_norm_after or []
    nfft_experiment_enabled = bool(nfft_metric_rows)

    lines = [
        f"# Diagnostic Summary: {run_name}",
        "",
        f"Generated: {generated_at}",
        "",
        "## Executive summary",
        "",
        (
            f"- Analyzed {len(rows)} metric rows across {total_files} files using "
            f"{diagnostics_source} for warning detail."
        ),
        (
            "- Welch detrending mode(s): "
            + ", ".join(
                f"{row['welch_detrend_mode']} ({row['files']} files)"
                for row in welch_mode_rows
            )
        ),
        (
            f"- {high_error_rows} rows exceed 20% relative error and "
            f"{very_high_error_rows} rows exceed 80% relative error."
        ),
        (
            f"- The largest p90 relative-error metric is {worst_metric['metric']} "
            f"({format_pct(worst_metric['p90_relative_error_pct'])}), which suggests "
            "that metric should receive focused manual review."
        ),
        (
            f"- Metric-band bin-count mismatch occurs in {format_pct(bin_band_row['rate_pct'])} "
            f"of rows overall and {format_pct(bin_high_row['rate_pct'])} of rows above 20% "
            "relative error, which is consistent with bin placement contributing to some discrepancies."
        ),
        (
            f"- {abs_norm_count} files match the configured pattern for absolute power mismatch "
            "with preserved relative spectral distribution."
        ),
    ]
    if nfft_experiment_enabled:
        largest_mean_improvement = min(
            nfft_metric_rows,
            key=lambda row: to_float(row.get("mean_relative_error_pct_change")),
        )
        lines.extend(
            [
                (
                    f"- The nfft alignment experiment compares current native Welch against "
                    f"native Welch with nfft=2*nperseg. The largest mean relative-error improvement "
                    f"is for {largest_mean_improvement['metric']} "
                    f"({format_pct(largest_mean_improvement['mean_relative_error_pct_change'])} change)."
                ),
                (
                    f"- Absolute-power mismatch files changed from {len(nfft_abs_norm_before)} "
                    f"to {len(nfft_abs_norm_after)} under the nfft=2 experiment."
                ),
            ]
        )
    lines.extend(
        [
        "",
        "## Metric-level agreement",
        "",
        markdown_table(
            summaries,
            [
                ("metric", "Metric"),
                ("rows", "Rows"),
                ("mean_relative_error_pct", "Mean rel err"),
                ("median_relative_error_pct", "Median rel err"),
                ("p75_relative_error_pct", "P75 rel err"),
                ("p90_relative_error_pct", "P90 rel err"),
                ("max_relative_error_pct", "Max rel err"),
                ("mean_absolute_error", "Mean abs err"),
                ("median_absolute_error", "Median abs err"),
                ("outliers_gt_20_pct", ">20%"),
                ("outliers_gt_50_pct", ">50%"),
                ("outliers_gt_80_pct", ">80%"),
            ],
        ),
        "## Welch detrending mode",
        "",
        (
            "This records the validation comparator mode when present in the validation CSV. "
            "`current` means the historical NeuroKit2 validation path was preserved."
        ),
        "",
        markdown_table(
            welch_mode_rows,
            [
                ("welch_detrend_mode", "Mode"),
                ("rows", "Rows"),
                ("files", "Files"),
            ],
        ),
        ]
    )

    if nfft_experiment_enabled:
        lines.extend(
            [
                "## NeuroKit2-style nfft alignment experiment",
                "",
                (
                    "This validation-only experiment recomputes native Welch values with "
                    "`nfft = 2 * nperseg` and compares those experimental values against NeuroKit2. "
                    "Negative changes indicate lower relative error after the nfft change; positive changes indicate higher relative error."
                ),
                "",
                markdown_table(
                    nfft_metric_rows,
                    [
                        ("metric", "Metric"),
                        ("rows", "Rows"),
                        ("mean_relative_error_pct_before", "Mean before"),
                        ("mean_relative_error_pct_after", "Mean after"),
                        ("mean_relative_error_pct_change", "Mean change"),
                        ("median_relative_error_pct_before", "Median before"),
                        ("median_relative_error_pct_after", "Median after"),
                        ("median_relative_error_pct_change", "Median change"),
                        ("long_duration_rows", "Long rows"),
                        ("long_duration_mean_relative_error_pct_before", "Long mean before"),
                        ("long_duration_mean_relative_error_pct_after", "Long mean after"),
                        ("long_duration_mean_relative_error_pct_change", "Long mean change"),
                        ("long_duration_median_relative_error_pct_before", "Long median before"),
                        ("long_duration_median_relative_error_pct_after", "Long median after"),
                        ("long_duration_median_relative_error_pct_change", "Long median change"),
                    ],
                ),
                "### nfft bin-count alignment",
                "",
                markdown_table(
                    nfft_bin_rows,
                    [
                        ("row_type", "Check"),
                        ("rows", "Rows"),
                        ("denominator", "Denominator"),
                        ("rate_pct", "Rate"),
                    ],
                ),
                "### Normalized-metric stability",
                "",
                (
                    "Small native-value changes for LF/HF, LF_nu, and HF_nu would suggest that the "
                    "relative spectral distribution is stable under the denser PSD grid."
                ),
                "",
                markdown_table(
                    nfft_stability_rows,
                    [
                        ("metric", "Metric"),
                        ("metric_group", "Group"),
                        ("rows", "Rows"),
                        ("mean_native_value_abs_change", "Mean value change"),
                        ("median_native_value_abs_change", "Median value change"),
                        ("mean_native_value_relative_change_pct", "Mean relative value change"),
                        ("median_native_value_relative_change_pct", "Median relative value change"),
                    ],
                ),
                "### Absolute-power mismatch file counts",
                "",
                markdown_table(
                    [
                        {
                            "stage": "before_nfft_alignment",
                            "files": len(nfft_abs_norm_before),
                            "interpretation": (
                                "absolute power mismatch with preserved relative spectral distribution"
                            ),
                        },
                        {
                            "stage": "after_nfft_alignment",
                            "files": len(nfft_abs_norm_after),
                            "interpretation": (
                                "absolute power mismatch with preserved relative spectral distribution"
                            ),
                        },
                    ],
                    [
                        ("stage", "Stage"),
                        ("files", "Files"),
                        ("interpretation", "Interpretation"),
                    ],
                ),
            ]
        )

    lines.extend(
        [
        "## Outlier analysis",
        "",
        (
            "Rows are flagged when `relative_error_pct` is greater than 20%, 50%, or 80%. "
            "These rows require manual review; the pattern alone does not identify either implementation as correct or wrong."
        ),
        "",
        markdown_table(
            outliers,
            [
                ("threshold", "Threshold"),
                ("file_name", "File"),
                ("duration_seconds", "Duration"),
                ("metric", "Metric"),
                ("native_value", "Native"),
                ("neurokit2_value", "NeuroKit2"),
                ("absolute_error", "Abs err"),
                ("relative_error_pct", "Rel err"),
                ("native_bins_total", "Native bins"),
                ("neurokit2_bins_total", "NK2 bins"),
                ("native_metric_band_bins", "Native band bins"),
                ("neurokit2_metric_band_bins", "NK2 band bins"),
                ("detrend_method", "Detrend"),
                ("welch_detrend_mode", "Welch detrend mode"),
                ("warnings", "Warnings"),
            ],
            args.top_outliers,
        ),
        "## Duration effects",
        "",
        (
            "Shorter recordings can make frequency-band estimates less stable. "
            "The bucketed distributions below are consistent with duration being a possible explanatory factor when high-error buckets align with duration warnings."
        ),
        "",
        markdown_table(
            duration_rows,
            [
                ("duration_bucket", "Duration"),
                ("rows", "Rows"),
                ("mean_relative_error_pct", "Mean rel err"),
                ("median_relative_error_pct", "Median rel err"),
                ("p75_relative_error_pct", "P75 rel err"),
                ("p90_relative_error_pct", "P90 rel err"),
                ("max_relative_error_pct", "Max rel err"),
                ("outliers_gt_20_pct", ">20%"),
                ("outliers_gt_50_pct", ">50%"),
                ("outliers_gt_80_pct", ">80%"),
            ],
        ),
        "## Long-duration outliers",
        "",
        (
            f"- {len({normalize_file_key(row.get('source_file')) for row in long_duration_outliers})} "
            "source files have at least one long-duration outlier."
        ),
        (
            f"- "
            f"{len({normalize_file_key(row.get('source_file')) for row in long_duration_outliers})} "
            "unique recordings require manual review."
        ),
        (
            f"- {len(long_duration_outliers)} metric rows were exported to "
            "`long_duration_outliers.csv` because `duration_seconds >= 300` and "
            "`relative_error_pct > 50`."
        ),
        (
            "- High discrepancies in this duration range are unexpected for recording duration "
            "and may suggest methodological mismatch, bin placement effects, scaling differences, "
            "or data-specific issues."
        ),
        "- Most affected metrics:",
        markdown_table(
            long_duration_outlier_metric_counts(long_duration_outliers),
            [
                ("metric", "Metric"),
                ("rows", "Rows"),
            ],
            10,
        ),
        "Top 10 largest discrepancies:",
        "",
        markdown_table(
            long_duration_outliers,
            [
                ("source_file", "Source file"),
                ("metric", "Metric"),
                ("duration_seconds", "Duration"),
                ("native_value", "Native"),
                ("neurokit2_value", "NeuroKit2"),
                ("absolute_error", "Abs err"),
                ("relative_error_pct", "Rel err"),
                ("native_bins_total", "Native bins"),
                ("neurokit2_bins_total", "NK2 bins"),
                ("native_metric_band_bins", "Native band bins"),
                ("neurokit2_metric_band_bins", "NK2 band bins"),
                ("detrend_method", "Detrend"),
                ("welch_detrend_mode", "Welch detrend mode"),
                ("warnings", "Warnings"),
            ],
            10,
        ),
        "## Bin-count effects",
        "",
        (
            "Bin-count differences suggest that the two methods may be integrating different discrete frequency bins. "
            "This is especially relevant when the metric-band bin mismatch rate is higher among high-error rows."
        ),
        "",
        markdown_table(
            bin_rows,
            [
                ("row_type", "Check"),
                ("rows", "Rows"),
                ("denominator", "Denominator"),
                ("rate_pct", "Rate"),
            ],
        ),
        "## Absolute power vs normalized metrics",
        "",
        (
            f"Configured rule: at least one VLF/LF/HF/total_power error > "
            f"{args.absolute_power_high_threshold:.1f}% while LF/HF, LF_nu, and HF_nu are all <= "
            f"{args.normalized_low_threshold:.1f}%. These cases are interpreted cautiously as "
            '"absolute power mismatch with preserved relative spectral distribution."'
        ),
        "",
        markdown_table(
            abs_norm_rows,
            [
                ("file_name", "File"),
                ("duration_seconds", "Duration"),
                ("max_absolute_power_relative_error_pct", "Max abs-power rel err"),
                ("max_normalized_relative_error_pct", "Max normalized rel err"),
                ("mean_absolute_power_relative_error_pct", "Mean abs-power rel err"),
                ("mean_normalized_relative_error_pct", "Mean normalized rel err"),
                ("interpretation", "Interpretation"),
            ],
            args.top_outliers,
        ),
        "## Warning summary",
        "",
        markdown_table(
            warning_rows,
            [
                ("warning", "Warning"),
                ("files", "Files"),
            ],
        ),
        "## Recommended interpretation",
        "",
        (
            "- Welch detrending mode should be considered when interpreting VLF and total_power agreement. "
            "`global_then_none` and segment-wise linear detrending can differ mainly at DC and the first VLF bins."
        ),
        (
            "- Large absolute-power discrepancies with aligned LF/HF and normalized-unit metrics are consistent with a scaling, integration, or total-power normalization difference rather than a wholesale change in relative spectral distribution."
        ),
        (
            "- High VLF/LF/HF errors in short recordings or rows with few PSD bins are likely explained by duration limits and discrete bin placement, but require manual review before drawing implementation conclusions."
        ),
        (
            "- Rows with FFT variance mismatch or AR fallback warnings suggest method-specific numerical conditions that should be reviewed alongside the raw PSD diagnostics."
        ),
        (
            "- This report is diagnostic only; it does not establish whether HRV Studio or NeuroKit2 is correct for any individual row."
        ),
        "",
        "## Files requiring manual review",
        "",
        markdown_table(
            manual_rows,
            [
                ("file_name", "File"),
                ("duration_seconds", "Duration"),
                ("max_relative_error_pct", "Max rel err"),
                ("reasons", "Reasons"),
                ("warnings", "Warnings"),
            ],
            args.top_outliers,
        ),
    ]
    )
    return "\n".join(lines)


def append_notes(
    run_dir: Path,
    run_name: str,
    welch_mode_rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    outliers: List[Dict[str, Any]],
    abs_norm_rows: List[Dict[str, Any]],
    nfft_metric_rows: Optional[List[Dict[str, Any]]] = None,
    nfft_abs_norm_before: Optional[List[Dict[str, Any]]] = None,
    nfft_abs_norm_after: Optional[List[Dict[str, Any]]] = None,
) -> None:
    notes_path = run_dir / "notes.md"
    worst_metric = max(
        summaries,
        key=lambda row: to_float(row.get("p90_relative_error_pct")),
    )
    summary = [
        "",
        "## Auto-generated diagnostic summary",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
        "",
        (
            f"- Diagnostic summary files were generated for `{run_name}`: "
            "`diagnostic_summary.md` and `diagnostic_summary.csv`."
        ),
        (
            "- Welch detrending mode(s): "
            + ", ".join(
                f"{row['welch_detrend_mode']} ({row['files']} files)"
                for row in welch_mode_rows
            )
        ),
        (
            f"- {len(outliers)} metric rows exceed 20% relative error; "
            f"{sum(1 for row in outliers if to_float(row['relative_error_pct']) > 80.0)} exceed 80%."
        ),
        (
            f"- The highest p90 relative-error metric is `{worst_metric['metric']}` "
            f"({format_pct(worst_metric['p90_relative_error_pct'])})."
        ),
        (
            f"- {len(abs_norm_rows)} files are consistent with absolute power mismatch "
            "while relative spectral distribution is preserved."
        ),
        (
            "- Interpretation remains cautious: flagged rows suggest possible explanatory factors "
            "and require manual review."
        ),
    ]
    if nfft_metric_rows:
        largest_mean_improvement = min(
            nfft_metric_rows,
            key=lambda row: to_float(row.get("mean_relative_error_pct_change")),
        )
        summary.extend(
            [
                (
                    f"- Nfft alignment comparison was generated in "
                    "`nfft_alignment_comparison.csv`; the largest mean relative-error change "
                    f"is `{largest_mean_improvement['metric']}` "
                    f"({format_pct(largest_mean_improvement['mean_relative_error_pct_change'])})."
                ),
                (
                    f"- Absolute-power mismatch files changed from "
                    f"{len(nfft_abs_norm_before or [])} to {len(nfft_abs_norm_after or [])} "
                    "under the nfft=2 experiment."
                ),
            ]
        )
    summary.append("")
    with notes_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(summary))


def main() -> int:
    args = parse_args()
    run_dir = Path(args.runs_root) / args.run_name
    validation_csv = run_dir / "freq_domain_neurokit2_validation.csv"
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if not validation_csv.exists():
        raise SystemExit(f"Missing validation CSV: {validation_csv}")

    diagnostics, diagnostics_source = load_diagnostics(run_dir)
    validation_rows = read_csv(validation_csv)
    rows = enrich_rows(validation_rows, diagnostics)
    if not rows:
        raise SystemExit("No supported metric rows found in validation CSV.")

    welch_mode_rows = welch_mode_summary(rows)
    summaries = metric_summary(rows)
    outliers = outlier_rows(rows)
    duration_rows = duration_summary(rows)
    long_outliers = long_duration_outlier_rows(rows)
    bin_rows = bin_mismatch_summary(rows)
    abs_norm_rows = absolute_vs_normalized_flags(
        rows,
        args.absolute_power_high_threshold,
        args.normalized_low_threshold,
    )
    warning_rows, _warning_counter = warning_summary(rows)
    manual_rows = manual_review_rows(outliers, abs_norm_rows, args.top_outliers)
    nfft_metric_rows: List[Dict[str, Any]] = []
    nfft_bin_rows: List[Dict[str, Any]] = []
    nfft_stability_rows: List[Dict[str, Any]] = []
    nfft_abs_norm_before: List[Dict[str, Any]] = []
    nfft_abs_norm_after: List[Dict[str, Any]] = []
    nfft_comparison_rows: List[Dict[str, Any]] = []
    if has_experimental_columns(rows):
        nfft_metric_rows = nfft_alignment_metric_summary(rows)
        nfft_bin_rows = nfft_alignment_bin_summary(rows)
        nfft_stability_rows = normalized_metric_stability(rows)
        nfft_abs_norm_before = absolute_vs_normalized_flags_for(
            rows,
            args.absolute_power_high_threshold,
            args.normalized_low_threshold,
            rel_error_key="relative_error_pct",
            section="nfft_alignment_absolute_power_mismatch_before",
        )
        nfft_abs_norm_after = absolute_vs_normalized_flags_for(
            rows,
            args.absolute_power_high_threshold,
            args.normalized_low_threshold,
            rel_error_key="experimental_relative_error_pct",
            section="nfft_alignment_absolute_power_mismatch_after",
        )
        nfft_comparison_rows.extend(nfft_metric_rows)
        nfft_comparison_rows.extend(nfft_bin_rows)
        nfft_comparison_rows.extend(nfft_stability_rows)
        nfft_comparison_rows.extend(
            [
                {
                    "section": "nfft_alignment_absolute_power_mismatch_count",
                    "stage": "before_nfft_alignment",
                    "files": len(nfft_abs_norm_before),
                    "absolute_power_high_threshold_pct": args.absolute_power_high_threshold,
                    "normalized_low_threshold_pct": args.normalized_low_threshold,
                },
                {
                    "section": "nfft_alignment_absolute_power_mismatch_count",
                    "stage": "after_nfft_alignment",
                    "files": len(nfft_abs_norm_after),
                    "absolute_power_high_threshold_pct": args.absolute_power_high_threshold,
                    "normalized_low_threshold_pct": args.normalized_low_threshold,
                },
            ]
        )

    csv_rows: List[Dict[str, Any]] = []
    csv_rows.extend(welch_mode_rows)
    csv_rows.extend(summaries)
    csv_rows.extend(nfft_comparison_rows)
    csv_rows.extend(duration_rows)
    csv_rows.extend(bin_rows)
    csv_rows.extend(abs_norm_rows)
    csv_rows.extend(warning_rows)
    csv_rows.extend(outliers)
    csv_rows.extend(manual_rows)

    md = build_markdown(
        args.run_name,
        diagnostics_source,
        rows,
        welch_mode_rows,
        summaries,
        outliers,
        duration_rows,
        long_outliers,
        bin_rows,
        abs_norm_rows,
        warning_rows,
        manual_rows,
        args,
        nfft_metric_rows,
        nfft_bin_rows,
        nfft_stability_rows,
        nfft_abs_norm_before,
        nfft_abs_norm_after,
    )

    md_path = run_dir / "diagnostic_summary.md"
    csv_path = run_dir / "diagnostic_summary.csv"
    long_outliers_path = run_dir / "long_duration_outliers.csv"
    nfft_csv_path = run_dir / "nfft_alignment_comparison.csv"
    md_path.write_text(md, encoding="utf-8")
    write_csv(csv_path, csv_rows)
    write_csv_selected(long_outliers_path, long_outliers, LONG_DURATION_OUTLIER_COLUMNS)
    if nfft_comparison_rows:
        write_csv(nfft_csv_path, nfft_comparison_rows)
    if not args.no_append_notes:
        append_notes(
            run_dir,
            args.run_name,
            welch_mode_rows,
            summaries,
            outliers,
            abs_norm_rows,
            nfft_metric_rows,
            nfft_abs_norm_before,
            nfft_abs_norm_after,
        )

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {long_outliers_path}")
    if nfft_comparison_rows:
        print(f"Wrote {nfft_csv_path}")
    if not args.no_append_notes:
        print(f"Appended auto-generated summary to {run_dir / 'notes.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
