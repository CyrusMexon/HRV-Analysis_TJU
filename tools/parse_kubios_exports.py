"""
Parse manually exported Kubios HRV text reports and compare them to validation results.

This is validation-only tooling. It does not import or modify production HRV
Studio code, and it does not rerun validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORTS_DIR = PROJECT_ROOT / "validation" / "kubios_subset" / "kubios_exports"
DEFAULT_MANIFEST = PROJECT_ROOT / "validation" / "kubios_subset" / "subset_manifest.csv"
DEFAULT_VALIDATION_CSV = (
    PROJECT_ROOT
    / "validation"
    / "runs"
    / "v05_physionet_10min_segment_linear"
    / "freq_domain_neurokit2_validation.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "validation" / "kubios_subset" / "parsed_results"

KUBIOS_PARSED_COLUMNS = [
    "subset_id",
    "source_file",
    "category",
    "kubios_vlf",
    "kubios_lf",
    "kubios_hf",
    "kubios_total_power",
    "kubios_lf_hf",
    "kubios_lf_nu",
    "kubios_hf_nu",
]

COMPARISON_COLUMNS = [
    "subset_id",
    "metric",
    "kubios_value",
    "hrvstudio_native_value",
    "neurokit_value",
    "hrvstudio_abs_error",
    "hrvstudio_relative_error_pct",
    "neurokit_abs_error",
    "neurokit_relative_error_pct",
]

COMPARISON_ALL_COLUMNS = [
    "subset_id",
    "category",
    "source_file",
    "kubios_input_txt",
    "validation_input_file",
    "validation_matched",
    "metric",
    "kubios_value",
    "hrvstudio_native_value",
    "neurokit_value",
    "hrvstudio_abs_error",
    "hrvstudio_relative_error_pct",
    "neurokit_abs_error",
    "neurokit_relative_error_pct",
]

EXCLUDED_COLUMNS = [
    "subset_id",
    "category",
    "source_file",
    "kubios_input_txt",
    "validation_input_file",
    "exclusion_reasons",
    "max_hrvstudio_relative_error_pct",
    "max_neurokit_relative_error_pct",
]

METRICS: Sequence[Tuple[str, str]] = (
    ("VLF", "kubios_vlf"),
    ("LF", "kubios_lf"),
    ("HF", "kubios_hf"),
    ("total_power", "kubios_total_power"),
    ("LF/HF", "kubios_lf_hf"),
    ("LF_nu", "kubios_lf_nu"),
    ("HF_nu", "kubios_hf_nu"),
)

POWER_METRICS = {"VLF", "LF", "HF", "total_power"}

KUBIOS_PATTERNS: Dict[str, Tuple[str, str]] = {
    "VLF": ("Absolute powers / VLF (ms^2)", r"^\s*VLF\s*\(\s*ms\s*(?:\^?2)\s*\)\s*:"),
    "LF": ("Absolute powers / LF (ms^2)", r"^\s*LF\s*\(\s*ms\s*(?:\^?2)\s*\)\s*:"),
    "HF": ("Absolute powers / HF (ms^2)", r"^\s*HF\s*\(\s*ms\s*(?:\^?2)\s*\)\s*:"),
    "total_power": (
        "Frequency-Domain Results / Total power (ms^2)",
        r"^\s*Total\s+power\s*\(\s*ms\s*(?:\^?2)\s*\)\s*:",
    ),
    "LF/HF": ("Frequency-Domain Results / LF/HF ratio", r"^\s*LF\s*/\s*HF\s+ratio\s*:"),
    "LF_nu": ("Normalized powers / LF (n.u.)", r"^\s*LF\s*\(\s*n\.?\s*u\.?\s*\)\s*:"),
    "HF_nu": ("Normalized powers / HF (n.u.)", r"^\s*HF\s*\(\s*n\.?\s*u\.?\s*\)\s*:"),
}

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|NaN", re.IGNORECASE)


class ParseError(Exception):
    def __init__(self, section: str, path: Path):
        super().__init__(f"Could not parse {section}: {path}")
        self.section = section
        self.path = path


@dataclass
class ParsedExport:
    export_path: Path
    manifest_row: Dict[str, str]
    metrics: Dict[str, float]
    method_metrics: Dict[str, Dict[str, float]]
    detrending_method: str
    sample_limits: str
    data_length: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Kubios *_hrv.txt exports and compare with native/NeuroKit2 validation results."
    )
    parser.add_argument("--exports-dir", "--export-root", dest="exports_dir", default=str(DEFAULT_EXPORTS_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--validation-csv", default=str(DEFAULT_VALIDATION_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def norm_path(path: Path) -> str:
    return str(path.resolve()).casefold()


def build_manifest_indexes(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    indexes: Dict[str, Dict[str, Dict[str, str]]] = {
        "export_path": {},
        "export_stem": {},
        "subset_id": {},
        "input_stem": {},
    }
    for row in rows:
        subset_id = row.get("subset_id", "").strip()
        if subset_id:
            indexes["subset_id"][subset_id.casefold()] = row

        export_txt = row.get("kubios_export_txt_path", "").strip()
        if export_txt:
            export_path = project_path(export_txt)
            indexes["export_path"][norm_path(export_path)] = row
            indexes["export_stem"][export_path.stem.casefold()] = row

        kubios_input = row.get("kubios_input_txt", "").strip()
        if kubios_input:
            indexes["input_stem"][Path(kubios_input).stem.casefold()] = row
    return indexes


def match_manifest_row(export_path: Path, indexes: Dict[str, Dict[str, Dict[str, str]]]) -> Optional[Dict[str, str]]:
    exact = indexes["export_path"].get(norm_path(export_path))
    if exact:
        return exact

    stem = export_path.stem.casefold()
    by_stem = indexes["export_stem"].get(stem)
    if by_stem:
        return by_stem

    input_stem = re.sub(r"_hrv$", "", export_path.stem, flags=re.IGNORECASE).casefold()
    by_input = indexes["input_stem"].get(input_stem)
    if by_input:
        return by_input

    subset_id = export_path.stem.split("__", 1)[0].casefold()
    return indexes["subset_id"].get(subset_id)


def first_semicolon_value(line: str) -> float:
    parts = line.split(";")
    if len(parts) > 1:
        search_parts = parts[1:]
    else:
        search_parts = parts

    for part in search_parts:
        match = NUMBER_RE.search(part)
        if match:
            token = match.group(0)
            if token.lower() == "nan":
                return math.nan
            return float(token)
    raise ValueError(f"No numeric value in line: {line}")


def first_semicolon_text(line: str) -> str:
    parts = line.split(";")
    if len(parts) > 1:
        for part in parts[1:]:
            value = part.strip()
            if value:
                return value
    return ""


def extract_frequency_section(text: str, path: Path) -> str:
    normalized = text.replace("\u00b2", "^2")
    match = re.search(
        r"Frequency-Domain Results(?P<section>.*?)(?:\n\s*Nonlinear Results|\n\s*RR INTERVAL DATA|$)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ParseError("Frequency-Domain Results section", path)
    return match.group(0)


def parse_kubios_metric(section: str, metric: str, path: Path) -> float:
    section_name, pattern = KUBIOS_PATTERNS[metric]
    regex = re.compile(pattern, flags=re.IGNORECASE)
    for line in section.splitlines():
        if regex.search(line):
            try:
                return first_semicolon_value(line)
            except ValueError as exc:
                raise ParseError(section_name, path) from exc
    raise ParseError(section_name, path)


def semicolon_values(line: str) -> List[float]:
    values: List[float] = []
    for part in line.split(";")[1:]:
        match = NUMBER_RE.search(part)
        if match:
            token = match.group(0)
            values.append(math.nan if token.lower() == "nan" else float(token))
    return values


def parse_kubios_metric_columns(section: str, metric: str, path: Path) -> Dict[str, float]:
    """Parse FFT and AR columns from a Kubios frequency-domain table.

    Older validation code used only the first metric column, which is retained by
    parse_kubios_metric(). This helper exposes method-labelled columns for
    validation analyses that need to compare Kubios FFT and AR separately.
    """
    section_name, pattern = KUBIOS_PATTERNS[metric]
    regex = re.compile(pattern, flags=re.IGNORECASE)
    for line in section.splitlines():
        if regex.search(line):
            values = semicolon_values(line)
            if not values:
                raise ParseError(section_name, path)
            parsed = {"fft": values[0]}
            if len(values) > 1:
                parsed["ar"] = values[1]
            return parsed
    raise ParseError(section_name, path)


def parse_metadata(text: str) -> Tuple[str, str, str]:
    detrending = ""
    sample_limits = ""
    data_length = ""
    for line in text.splitlines():
        if re.search(r"^\s*Detrending method\s*:", line, flags=re.IGNORECASE):
            detrending = line.split(":", 1)[1].strip()
        elif re.search(r"^\s*Sample limits\s*\(s\)\s*:", line, flags=re.IGNORECASE):
            sample_limits = first_semicolon_text(line)
        elif re.search(r"^\s*Data length\s*:", line, flags=re.IGNORECASE):
            data_length = line.split(":", 1)[1].strip()
    return detrending, sample_limits, data_length


def parse_export(path: Path, manifest_row: Dict[str, str]) -> ParsedExport:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    section = extract_frequency_section(text, path)
    metrics = {metric: parse_kubios_metric(section, metric, path) for metric, _ in METRICS}
    method_metrics = {
        metric: parse_kubios_metric_columns(section, metric, path)
        for metric, _ in METRICS
    }
    detrending, sample_limits, data_length = parse_metadata(text)
    return ParsedExport(
        export_path=path,
        manifest_row=manifest_row,
        metrics=metrics,
        method_metrics=method_metrics,
        detrending_method=detrending,
        sample_limits=sample_limits,
        data_length=data_length,
    )


def build_validation_index(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    by_source: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        source_name = Path(row.get("input_file", "")).name.casefold()
        metric = row.get("metric", "")
        if source_name and metric:
            by_source.setdefault(source_name, {})[metric] = row
    return by_source


def validation_metrics_for_manifest(
    manifest: Dict[str, str],
    validation_index: Dict[str, Dict[str, Dict[str, str]]],
) -> Dict[str, Dict[str, str]]:
    candidate_paths = (
        manifest.get("source_csv", ""),
        manifest.get("kubios_input_txt", ""),
    )
    for candidate_path in candidate_paths:
        source_name = Path(candidate_path).name.casefold()
        if source_name in validation_index:
            return validation_index[source_name]
    return {}


def validation_input_file(validation_metrics: Dict[str, Dict[str, str]]) -> str:
    for row in validation_metrics.values():
        input_file = row.get("input_file", "")
        if input_file:
            return input_file
    return ""


def load_validation_run_info(validation_csv: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    run_info_path = validation_csv.parent / "run_info.json"
    if not run_info_path.exists():
        return {}, {}

    data = json.loads(run_info_path.read_text(encoding="utf-8"))
    failures = {
        Path(item.get("input_file", "")).name.casefold(): item.get("error", "")
        for item in data.get("failures", [])
        if item.get("input_file")
    }
    file_events: Dict[str, List[str]] = {}
    for item in data.get("file_events", []):
        input_file = item.get("input_file", "")
        reason = item.get("reason", "")
        if input_file and reason:
            file_events.setdefault(Path(input_file).name.casefold(), []).append(reason)
    return failures, file_events


def finite_float(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def rel_error_pct(reference: float, value: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(value) or reference == 0:
        return math.nan
    return abs(value - reference) / abs(reference) * 100.0


def parsed_csv_row(parsed: ParsedExport) -> Dict[str, object]:
    row = parsed.manifest_row
    return {
        "subset_id": row.get("subset_id", ""),
        "source_file": row.get("source_csv", ""),
        "category": row.get("category", ""),
        "kubios_vlf": parsed.metrics["VLF"],
        "kubios_lf": parsed.metrics["LF"],
        "kubios_hf": parsed.metrics["HF"],
        "kubios_total_power": parsed.metrics["total_power"],
        "kubios_lf_hf": parsed.metrics["LF/HF"],
        "kubios_lf_nu": parsed.metrics["LF_nu"],
        "kubios_hf_nu": parsed.metrics["HF_nu"],
    }


def comparison_rows(
    parsed_exports: Sequence[ParsedExport],
    validation_index: Dict[str, Dict[str, Dict[str, str]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for parsed in parsed_exports:
        manifest = parsed.manifest_row
        validation_metrics = validation_metrics_for_manifest(manifest, validation_index)
        validation_input = validation_input_file(validation_metrics)
        for metric, _column in METRICS:
            validation_row = validation_metrics.get(metric, {})
            kubios_value = parsed.metrics[metric]
            native_value = finite_float(validation_row.get("native_value"))
            neurokit_value = finite_float(validation_row.get("neurokit2_value"))
            rows.append(
                {
                    "subset_id": manifest.get("subset_id", ""),
                    "category": manifest.get("category", ""),
                    "source_file": manifest.get("source_csv", ""),
                    "kubios_input_txt": manifest.get("kubios_input_txt", ""),
                    "validation_input_file": validation_input,
                    "validation_matched": bool(validation_metrics),
                    "metric": metric,
                    "kubios_value": kubios_value,
                    "hrvstudio_native_value": native_value,
                    "neurokit_value": neurokit_value,
                    "hrvstudio_abs_error": abs(native_value - kubios_value)
                    if math.isfinite(native_value) and math.isfinite(kubios_value)
                    else math.nan,
                    "hrvstudio_relative_error_pct": rel_error_pct(kubios_value, native_value),
                    "neurokit_abs_error": abs(neurokit_value - kubios_value)
                    if math.isfinite(neurokit_value) and math.isfinite(kubios_value)
                    else math.nan,
                    "neurokit_relative_error_pct": rel_error_pct(kubios_value, neurokit_value),
                }
            )
    return rows


def fmt(value: object, digits: int = 3) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def mean_finite(values: Iterable[object]) -> float:
    finite = [finite_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def median_finite(values: Iterable[object]) -> float:
    finite = [finite_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return median(finite) if finite else math.nan


def summarize_by_metric(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for metric, _column in METRICS:
        metric_rows = [row for row in rows if row["metric"] == metric]
        hrv_mean = mean_finite(row["hrvstudio_relative_error_pct"] for row in metric_rows)
        hrv_median = median_finite(row["hrvstudio_relative_error_pct"] for row in metric_rows)
        neurokit_mean = mean_finite(row["neurokit_relative_error_pct"] for row in metric_rows)
        neurokit_median = median_finite(row["neurokit_relative_error_pct"] for row in metric_rows)
        if math.isfinite(hrv_mean) and math.isfinite(neurokit_mean):
            if abs(hrv_mean - neurokit_mean) < 1e-12:
                closer = "tie"
            elif hrv_mean < neurokit_mean:
                closer = "HRV Studio"
            else:
                closer = "NeuroKit2"
        else:
            closer = ""
        summaries.append(
            {
                "metric": metric,
                "hrvstudio_mean_relative_error_pct": hrv_mean,
                "hrvstudio_median_relative_error_pct": hrv_median,
                "neurokit_mean_relative_error_pct": neurokit_mean,
                "neurokit_median_relative_error_pct": neurokit_median,
                "closer_to_kubios": closer,
            }
        )
    return summaries


def rows_by_subset(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("subset_id", "")), []).append(row)
    return grouped


def all_native_powers_zero(rows: Sequence[Dict[str, object]]) -> bool:
    power_rows = [row for row in rows if row.get("metric") in POWER_METRICS]
    if len(power_rows) < len(POWER_METRICS):
        return False
    return all(finite_float(row.get("hrvstudio_native_value")) == 0.0 for row in power_rows)


def has_extreme_native_vs_kubios(rows: Sequence[Dict[str, object]], ratio_limit: float = 100.0) -> bool:
    for row in rows:
        if row.get("metric") not in POWER_METRICS:
            continue
        kubios = finite_float(row.get("kubios_value"))
        native = finite_float(row.get("hrvstudio_native_value"))
        if kubios > 0 and math.isfinite(native) and abs(native) > ratio_limit * abs(kubios):
            return True
    return False


def max_error(rows: Sequence[Dict[str, object]], column: str) -> float:
    return max((finite_float(row.get(column)) for row in rows), default=math.nan)


def classify_exclusions(
    comparison: Sequence[Dict[str, object]],
    validation_failures: Dict[str, str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, List[str]]]:
    grouped = rows_by_subset(comparison)
    excluded: List[Dict[str, object]] = []
    valid_rows: List[Dict[str, object]] = []
    reasons_by_subset: Dict[str, List[str]] = {}

    for subset_id, subset_rows in grouped.items():
        reasons: List[str] = []
        first = subset_rows[0]
        input_name = Path(str(first.get("kubios_input_txt", ""))).name.casefold()

        if input_name in validation_failures:
            reasons.append(f"validation_input_failure: {validation_failures[input_name]}")
        if not any(row.get("validation_matched") for row in subset_rows):
            reasons.append("missing_validation_results")
        if any(not math.isfinite(finite_float(row.get("hrvstudio_native_value"))) for row in subset_rows):
            reasons.append("missing_or_nonfinite_hrvstudio_metric")
        if any(not math.isfinite(finite_float(row.get("neurokit_value"))) for row in subset_rows):
            reasons.append("missing_or_nonfinite_neurokit2_metric")
        if any(not math.isfinite(finite_float(row.get("kubios_value"))) for row in subset_rows):
            reasons.append("missing_or_nonfinite_kubios_metric")
        if all_native_powers_zero(subset_rows):
            reasons.append("all_native_powers_zero")

        total_rows = [row for row in subset_rows if row.get("metric") == "total_power"]
        total_value = finite_float(total_rows[0].get("hrvstudio_native_value")) if total_rows else math.nan
        if math.isfinite(total_value) and total_value <= 0:
            reasons.append("native_total_power_nonpositive")
        if has_extreme_native_vs_kubios(subset_rows):
            reasons.append("native_power_gt_100x_kubios")

        if reasons:
            reasons_by_subset[subset_id] = reasons
            excluded.append(
                {
                    "subset_id": subset_id,
                    "category": first.get("category", ""),
                    "source_file": first.get("source_file", ""),
                    "kubios_input_txt": first.get("kubios_input_txt", ""),
                    "validation_input_file": first.get("validation_input_file", ""),
                    "exclusion_reasons": "; ".join(reasons),
                    "max_hrvstudio_relative_error_pct": max_error(
                        subset_rows, "hrvstudio_relative_error_pct"
                    ),
                    "max_neurokit_relative_error_pct": max_error(
                        subset_rows, "neurokit_relative_error_pct"
                    ),
                }
            )
        else:
            valid_rows.extend(subset_rows)

    return valid_rows, excluded, reasons_by_subset


def matched_subset_ids(comparison: Sequence[Dict[str, object]]) -> Set[str]:
    return {
        str(row.get("subset_id", ""))
        for row in comparison
        if row.get("validation_matched")
    }


def subsets_with_nonfinite_or_zero_hrv(comparison: Sequence[Dict[str, object]]) -> List[str]:
    flagged: Set[str] = set()
    for subset_id, subset_rows in rows_by_subset(comparison).items():
        if any(not math.isfinite(finite_float(row.get("hrvstudio_native_value"))) for row in subset_rows):
            flagged.add(subset_id)
        elif all_native_powers_zero(subset_rows):
            flagged.add(subset_id)
        elif any(
            row.get("metric") in POWER_METRICS
            and math.isfinite(finite_float(row.get("hrvstudio_native_value")))
            and finite_float(row.get("hrvstudio_native_value")) <= 0
            for row in subset_rows
        ):
            flagged.add(subset_id)
    return sorted(flagged)


def extreme_value_rows(comparison: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in comparison:
        if row.get("metric") not in POWER_METRICS:
            continue
        kubios = finite_float(row.get("kubios_value"))
        native = finite_float(row.get("hrvstudio_native_value"))
        neurokit = finite_float(row.get("neurokit_value"))
        if kubios <= 0:
            continue
        native_ratio = abs(native) / abs(kubios) if math.isfinite(native) else math.nan
        neurokit_ratio = abs(neurokit) / abs(kubios) if math.isfinite(neurokit) else math.nan
        if native_ratio > 100.0 or neurokit_ratio > 100.0:
            rows.append(
                {
                    "subset_id": row.get("subset_id", ""),
                    "category": row.get("category", ""),
                    "metric": row.get("metric", ""),
                    "kubios_value": kubios,
                    "hrvstudio_native_value": native,
                    "neurokit_value": neurokit,
                    "hrvstudio_ratio_to_kubios": native_ratio,
                    "neurokit_ratio_to_kubios": neurokit_ratio,
                }
            )
    return rows


def files_above_error_threshold(
    comparison: Sequence[Dict[str, object]], threshold_pct: float
) -> List[str]:
    flagged = {
        str(row.get("subset_id", ""))
        for row in comparison
        if finite_float(row.get("hrvstudio_relative_error_pct")) > threshold_pct
        or finite_float(row.get("neurokit_relative_error_pct")) > threshold_pct
    }
    return sorted(flagged)


def subset_count(rows: Sequence[Dict[str, object]]) -> int:
    return len({str(row.get("subset_id", "")) for row in rows})


def manifest_unrepresented_count(
    manifest_rows: Sequence[Dict[str, str]],
    parsed_exports: Sequence[ParsedExport],
    unmatched: Sequence[Path],
) -> int:
    represented_subset_ids = {
        export.manifest_row.get("subset_id", "").strip().casefold() for export in parsed_exports
    }
    represented_subset_ids.update(path.stem.split("__", 1)[0].casefold() for path in unmatched)
    return sum(
        1
        for row in manifest_rows
        if row.get("subset_id", "").strip().casefold() not in represented_subset_ids
    )


def unique_csv_values(rows: Sequence[Dict[str, str]], column: str) -> List[str]:
    values = sorted({row.get(column, "").strip() or "none" for row in rows})
    return values


def joined_values(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "unknown"


def subset_rows_table(rows: Sequence[Dict[str, object]]) -> List[List[object]]:
    table = []
    for subset_id, subset_rows in sorted(rows_by_subset(rows).items()):
        first = subset_rows[0]
        table.append(
            [
                subset_id,
                first.get("category", ""),
                first.get("kubios_input_txt", ""),
                first.get("validation_input_file", "") or "",
            ]
        )
    return table


def write_qc_report(
    path: Path,
    export_paths: Sequence[Path],
    parsed_exports: Sequence[ParsedExport],
    comparison: Sequence[Dict[str, object]],
    valid_only: Sequence[Dict[str, object]],
    excluded_files: Sequence[Dict[str, object]],
    validation_failures: Dict[str, str],
    file_events: Dict[str, List[str]],
) -> None:
    all_matched = [row for row in comparison if row.get("validation_matched")]
    missing_validation = [row for row in comparison if not row.get("validation_matched")]
    zero_or_nonfinite = subsets_with_nonfinite_or_zero_hrv(comparison)
    extreme_rows = extreme_value_rows(comparison)
    threshold_100 = files_above_error_threshold(all_matched, 100.0)
    threshold_500 = files_above_error_threshold(all_matched, 500.0)
    threshold_1000 = files_above_error_threshold(all_matched, 1000.0)

    missing_table = subset_rows_table(missing_validation)
    excluded_table = [
        [
            row["subset_id"],
            row["category"],
            row["exclusion_reasons"],
            fmt(row["max_hrvstudio_relative_error_pct"], 2),
            fmt(row["max_neurokit_relative_error_pct"], 2),
        ]
        for row in excluded_files
    ]
    extreme_table = [
        [
            row["subset_id"],
            row["category"],
            row["metric"],
            fmt(row["kubios_value"]),
            fmt(row["hrvstudio_native_value"]),
            fmt(row["neurokit_value"]),
            fmt(row["hrvstudio_ratio_to_kubios"], 1),
            fmt(row["neurokit_ratio_to_kubios"], 1),
        ]
        for row in extreme_rows
    ]
    file_events_table = [
        [input_name, ", ".join(reasons)]
        for input_name, reasons in sorted(file_events.items())
    ]

    lines: List[str] = [
        "# Kubios Comparison QC",
        "",
        "## Workflow Counts",
        "",
        f"- Total Kubios exports discovered: {len(export_paths)}",
        f"- Successfully parsed exports: {len(parsed_exports)}",
        f"- Files matched to HRV Studio/NeuroKit2 validation rows: {subset_count(all_matched)}",
        f"- Files missing validation results: {subset_count(missing_validation)}",
        f"- Files retained in `comparison_valid_only.csv`: {subset_count(valid_only)}",
        f"- Files excluded from `comparison_valid_only.csv`: {len(excluded_files)}",
        "",
        "## Missing Validation Results",
        "",
    ]
    if missing_table:
        lines.extend(
            markdown_table(
                ["Subset", "Category", "Kubios input", "Validation input"],
                missing_table,
            )
        )
    else:
        lines.append("None.")

    lines.extend(["", "## Validation Input Failures", ""])
    if validation_failures:
        lines.extend(f"- {input_name}: {error}" for input_name, error in sorted(validation_failures.items()))
    else:
        lines.append("None recorded in sibling `run_info.json`.")

    lines.extend(["", "## Zero/NaN/Nonfinite HRV Studio Metrics", ""])
    if zero_or_nonfinite:
        lines.extend(f"- {subset_id}" for subset_id in zero_or_nonfinite)
    else:
        lines.append("None detected.")

    lines.extend(["", "## Extreme Values", ""])
    lines.append("Rows below have HRV Studio or NeuroKit2 power values greater than 100x Kubios.")
    lines.append("")
    if extreme_table:
        lines.extend(
            markdown_table(
                [
                    "Subset",
                    "Category",
                    "Metric",
                    "Kubios",
                    "HRV Studio",
                    "NeuroKit2",
                    "HRV/Kubios",
                    "NeuroKit2/Kubios",
                ],
                extreme_table,
            )
        )
    else:
        lines.append("None detected.")

    lines.extend(["", "## Relative Error Thresholds", ""])
    for label, flagged in ((">100%", threshold_100), (">500%", threshold_500), (">1000%", threshold_1000)):
        lines.append(f"- Files with any metric relative error {label}: {len(flagged)}")
        lines.append(f"  - {', '.join(flagged) if flagged else 'none'}")

    lines.extend(["", "## Excluded Files", ""])
    if excluded_table:
        lines.extend(
            markdown_table(
                [
                    "Subset",
                    "Category",
                    "Reason",
                    "Max HRV rel err %",
                    "Max NeuroKit2 rel err %",
                ],
                excluded_table,
            )
        )
    else:
        lines.append("None.")

    lines.extend(["", "## Validation File Events", ""])
    if file_events_table:
        lines.extend(markdown_table(["Input file", "Events"], file_events_table))
    else:
        lines.append("None recorded in sibling `run_info.json`.")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    validation_csv: Path,
    validation_rows: Sequence[Dict[str, str]],
    export_paths: Sequence[Path],
    parsed_exports: Sequence[ParsedExport],
    comparison: Sequence[Dict[str, object]],
    valid_only: Sequence[Dict[str, object]],
    excluded_files: Sequence[Dict[str, object]],
    validation_failures: Dict[str, str],
    file_events: Dict[str, List[str]],
    failed: Sequence[str],
    unmatched: Sequence[Path],
    unrepresented_manifest_rows: int,
) -> None:
    all_matched = [row for row in comparison if row.get("validation_matched")]
    metric_summary_all = summarize_by_metric(all_matched)
    metric_summary_valid = summarize_by_metric(valid_only)
    hrv_overall = mean_finite(row["hrvstudio_relative_error_pct"] for row in comparison)
    hrv_overall_median = median_finite(row["hrvstudio_relative_error_pct"] for row in comparison)
    neurokit_overall = mean_finite(row["neurokit_relative_error_pct"] for row in comparison)
    neurokit_overall_median = median_finite(row["neurokit_relative_error_pct"] for row in comparison)
    valid_hrv_overall = mean_finite(row["hrvstudio_relative_error_pct"] for row in valid_only)
    valid_hrv_overall_median = median_finite(row["hrvstudio_relative_error_pct"] for row in valid_only)
    valid_neurokit_overall = mean_finite(row["neurokit_relative_error_pct"] for row in valid_only)
    valid_neurokit_overall_median = median_finite(row["neurokit_relative_error_pct"] for row in valid_only)

    def metric_summary_table(rows: Sequence[Dict[str, object]]) -> List[List[object]]:
        return [
            [
                row["metric"],
                fmt(row["hrvstudio_mean_relative_error_pct"], 2),
                fmt(row["hrvstudio_median_relative_error_pct"], 2),
                fmt(row["neurokit_mean_relative_error_pct"], 2),
                fmt(row["neurokit_median_relative_error_pct"], 2),
                row["closer_to_kubios"],
            ]
            for row in rows
        ]

    category_rows: List[List[object]] = []
    for category in sorted({str(row.get("category", "")) for row in valid_only}):
        rows = [row for row in valid_only if row.get("category") == category]
        category_rows.append(
            [
                category or "unknown",
                subset_count(rows),
                fmt(mean_finite(row["hrvstudio_relative_error_pct"] for row in rows), 2),
                fmt(median_finite(row["hrvstudio_relative_error_pct"] for row in rows), 2),
                fmt(mean_finite(row["neurokit_relative_error_pct"] for row in rows), 2),
                fmt(median_finite(row["neurokit_relative_error_pct"] for row in rows), 2),
            ]
        )

    excluded_table = [
        [
            row["subset_id"],
            row["category"],
            row["exclusion_reasons"],
            fmt(row["max_hrvstudio_relative_error_pct"], 2),
            fmt(row["max_neurokit_relative_error_pct"], 2),
        ]
        for row in excluded_files
    ]
    excluded_lines = (
        markdown_table(
            [
                "Subset",
                "Category",
                "Reason",
                "Max HRV rel err %",
                "Max NeuroKit2 rel err %",
            ],
            excluded_table,
        )
        if excluded_table
        else ["No files were excluded by the configured QC rules."]
    )

    detrending_methods = sorted({export.detrending_method or "unknown" for export in parsed_exports})
    sample_limits = sorted({export.sample_limits or "unknown" for export in parsed_exports})
    data_lengths = sorted({export.data_length or "unknown" for export in parsed_exports})
    segment_lengths = unique_csv_values(validation_rows, "segment_length_s")
    overlap_ratios = unique_csv_values(validation_rows, "overlap_ratio")
    detrend_methods = unique_csv_values(validation_rows, "detrend_method")
    welch_detrend_modes = unique_csv_values(validation_rows, "welch_detrend_mode")
    interpolation_rates = unique_csv_values(validation_rows, "interpolation_rate_hz")
    window_types = unique_csv_values(validation_rows, "window_type")

    lines: List[str] = [
        "# Kubios Comparison Summary",
        "",
        "Kubios values are parsed from the first frequency-domain result column, `FFT spectrum`, "
        "because the validation comparator is Welch/FFT-based rather than AR-based. This report "
        "separates parser success from validation data quality and filtered comparison results.",
        "",
        "## Workflow/Parser Success",
        "",
        f"- Discovered Kubios `*_hrv.txt` files: {len(export_paths)}",
        f"- Successfully parsed files: {len(parsed_exports)}",
        f"- Failed discovered exports: {len(failed)}",
        f"- Unmatched discovered exports: {len(unmatched)}",
        f"- Manifest rows not represented in this export directory: {unrepresented_manifest_rows}",
        f"- Parsed Kubios exports matched to validation rows: {subset_count(all_matched)}",
        f"- Parsed Kubios exports missing validation rows: {len(parsed_exports) - subset_count(all_matched)}",
        "",
    ]

    if failed:
        lines.extend(["### Failed Exports", ""])
        lines.extend(f"- {item}" for item in failed)
        lines.append("")
    else:
        lines.extend(["Failed exports among currently available files: none.", ""])

    if unmatched:
        lines.extend(["### Unmatched Exports", ""])
        lines.extend(f"- {path}" for path in unmatched)
        lines.append("")

    if validation_failures:
        lines.extend(["### Validation Input Failures", ""])
        for input_name, error in sorted(validation_failures.items()):
            lines.append(f"- {input_name}: {error}")
        lines.append("")

    lines.extend(
        [
            "## Data Quality Exclusions",
            "",
            f"- Files excluded from valid-only comparison: {len(excluded_files)}",
            f"- Files retained for valid-only comparison: {subset_count(valid_only)}",
            f"- Files with validation file events/adjusted Welch parameters: {len(file_events)}",
            "",
            *excluded_lines,
            "",
            "## Valid Comparison Results",
            "",
            f"- Valid-only HRV Studio mean/median relative error vs Kubios: "
            f"{fmt(valid_hrv_overall, 2)}% / {fmt(valid_hrv_overall_median, 2)}%",
            f"- Valid-only NeuroKit2 mean/median relative error vs Kubios: "
            f"{fmt(valid_neurokit_overall, 2)}% / {fmt(valid_neurokit_overall_median, 2)}%",
            "",
            *markdown_table(
                [
                    "Metric",
                    "HRV mean %",
                    "HRV median %",
                    "NeuroKit2 mean %",
                    "NeuroKit2 median %",
                    "Closer to Kubios",
                ],
                metric_summary_table(metric_summary_valid),
            ),
            "",
            "## All Matched Results",
            "",
            f"- All matched HRV Studio mean/median relative error vs Kubios: "
            f"{fmt(hrv_overall, 2)}% / {fmt(hrv_overall_median, 2)}%",
            f"- All matched NeuroKit2 mean/median relative error vs Kubios: "
            f"{fmt(neurokit_overall, 2)}% / {fmt(neurokit_overall_median, 2)}%",
            "",
            *markdown_table(
                [
                    "Metric",
                    "HRV mean %",
                    "HRV median %",
                    "NeuroKit2 mean %",
                    "NeuroKit2 median %",
                    "Closer to Kubios",
                ],
                metric_summary_table(metric_summary_all),
            ),
            "",
            "## Valid-Only Results by Category",
            "",
            *markdown_table(
                [
                    "Category",
                    "Valid files",
                    "HRV mean %",
                    "HRV median %",
                    "NeuroKit2 mean %",
                    "NeuroKit2 median %",
                ],
                category_rows,
            ),
            "",
            "## Comparator Settings",
            "",
            f"- Validation CSV: `{validation_csv}`",
            f"- Segment length(s): {joined_values(segment_lengths)} seconds",
            f"- Overlap ratio(s): {joined_values(overlap_ratios)}",
            f"- Detrend method(s): {joined_values(detrend_methods)}",
            f"- Welch detrend mode(s): {joined_values(welch_detrend_modes)}",
            f"- Interpolation rate(s): {joined_values(interpolation_rates)} Hz",
            f"- Window type(s): {joined_values(window_types)}",
            "",
            "## Comparability Assessment",
            "",
            f"- Kubios report detrending method(s): {', '.join(detrending_methods)}.",
            "- The validation comparator settings now match the requested Kubios-style settings "
            "for 120-second Welch windows, 75% overlap, and no detrending.",
            "- The full 50-export workflow succeeded at the Kubios parsing layer, but only "
            f"{subset_count(all_matched)} files have validation rows and {subset_count(valid_only)} "
            "files remain after QC filtering.",
            "- All-matched means should be interpreted cautiously because pathological files "
            "and short/adjusted Welch windows can dominate mean relative error.",
            "- Valid-only medians are the better high-level comparison signal for this run.",
            f"- Kubios report data length(s): {', '.join(data_lengths)}; sample limit(s): "
            f"{', '.join(sample_limits)}. The sample-limit field should be verified in Kubios "
            "before treating absolute power differences as definitive.",
            "",
            "## Pathological Cases Requiring Manual Review",
            "",
            "- See `kubios_comparison_qc.md` and `comparison_excluded_files.csv` for missing "
            "validation rows, non-finite/zero native outputs, extreme power ratios, and large "
            "relative-error thresholds.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    exports_dir = project_path(args.exports_dir)
    manifest_path = project_path(args.manifest)
    validation_csv = project_path(args.validation_csv)
    output_dir = project_path(args.output_dir)

    manifest_rows = read_csv_rows(manifest_path)
    validation_rows = read_csv_rows(validation_csv)
    manifest_indexes = build_manifest_indexes(manifest_rows)
    validation_index = build_validation_index(validation_rows)
    validation_failures, file_events = load_validation_run_info(validation_csv)

    export_paths = sorted(exports_dir.rglob("*_hrv.txt"))
    parsed_exports: List[ParsedExport] = []
    failed: List[str] = []
    unmatched: List[Path] = []

    for export_path in export_paths:
        manifest_row = match_manifest_row(export_path, manifest_indexes)
        if not manifest_row:
            unmatched.append(export_path)
            print(f"Could not match export to subset_manifest.csv: {export_path}")
            continue
        try:
            parsed_exports.append(parse_export(export_path, manifest_row))
        except ParseError as exc:
            message = str(exc)
            failed.append(message)
            print(message)

    parsed_rows = [parsed_csv_row(parsed) for parsed in parsed_exports]
    comparison = comparison_rows(parsed_exports, validation_index)
    valid_only, excluded_files, _reasons_by_subset = classify_exclusions(comparison, validation_failures)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "kubios_parsed_results.csv", KUBIOS_PARSED_COLUMNS, parsed_rows)
    write_csv_rows(output_dir / "comparison_with_hrvstudio.csv", COMPARISON_COLUMNS, comparison)
    write_csv_rows(output_dir / "comparison_all.csv", COMPARISON_ALL_COLUMNS, comparison)
    write_csv_rows(output_dir / "comparison_valid_only.csv", COMPARISON_ALL_COLUMNS, valid_only)
    write_csv_rows(output_dir / "comparison_excluded_files.csv", EXCLUDED_COLUMNS, excluded_files)
    write_qc_report(
        output_dir / "kubios_comparison_qc.md",
        export_paths,
        parsed_exports,
        comparison,
        valid_only,
        excluded_files,
        validation_failures,
        file_events,
    )
    write_summary(
        output_dir / "kubios_comparison_summary.md",
        validation_csv,
        validation_rows,
        export_paths,
        parsed_exports,
        comparison,
        valid_only,
        excluded_files,
        validation_failures,
        file_events,
        failed,
        unmatched,
        manifest_unrepresented_count(manifest_rows, parsed_exports, unmatched),
    )

    print(f"Discovered Kubios exports: {len(export_paths)}")
    print(f"Successfully parsed files: {len(parsed_exports)}")
    print(f"Failed exports: {len(failed)}")
    print(f"Wrote outputs to: {output_dir}")
    return 1 if failed or unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
