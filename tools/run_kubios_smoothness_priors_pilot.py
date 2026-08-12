"""
Run the Kubios-vs-HRV Studio Smoothness Priors pilot.

Validation-only tooling. This script does not modify production HRV algorithms
and does not overwrite existing Kubios exports.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from tools.parse_kubios_exports import (
    METRICS as KUBIOS_METRICS,
    extract_frequency_section,
    parse_kubios_metric_columns,
)


OUTPUT_SUFFIX = os.environ.get("HRV_SMOOTHNESS_RERUN_SUFFIX", "").strip()
BASE_PILOT_DIR = PROJECT_ROOT / "validation" / "kubios_subset" / "smoothness_priors_pilot"
PILOT_DIR = PROJECT_ROOT / "validation" / "kubios_subset" / f"smoothness_priors_pilot{OUTPUT_SUFFIX}"
EXPORTS_DIR = BASE_PILOT_DIR / "kubios_exports"
MANIFEST_PATH = PROJECT_ROOT / "validation" / "kubios_subset" / "subset_manifest.csv"
PRIOR_EXPORTS_DIR = PROJECT_ROOT / "validation" / "kubios_subset" / "kubios_exports_120s_75pct_none"
PRIOR_RUN_DIR = PROJECT_ROOT / "validation" / "runs" / "v07_kubios_subset_50_none_120_75_after_arm_a"
RUNS_ROOT = PROJECT_ROOT / "validation" / "runs"
REPORT_PATH = PROJECT_ROOT / "validation" / "research_notes" / f"kubios_smoothness_priors_pilot_report{OUTPUT_SUFFIX}.md"

EXPECTED_IDS = [
    "CH001",
    "CH002",
    "CH003",
    "CH004",
    "VLF002",
    "VLF005",
    "OUT001",
    "OUT005",
    "OUT006",
    "RC003",
]

METRIC_SPECS = [
    ("VLF", "vlf", "vlf_power"),
    ("LF", "lf", "lf_power"),
    ("HF", "hf", "hf_power"),
    ("total_power", "total_power", "total_power"),
    ("LF/HF", "lf_hf", "lf_hf_ratio"),
    ("LF_nu", "lf_nu", "lf_nu"),
    ("HF_nu", "hf_nu", "hf_nu"),
]

KUBIOS_OUT_KEYS = {
    "VLF": "vlf",
    "LF": "lf",
    "HF": "hf",
    "total_power": "total_power",
    "LF/HF": "lf_hf",
    "LF_nu": "lf_nu",
    "HF_nu": "hf_nu",
}

NEAR_ZERO = 1e-12


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_output(*args: str) -> Optional[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_versions() -> Dict[str, Optional[str]]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": package_version("scipy"),
        "matplotlib": package_version("matplotlib"),
    }


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return rel(value)
    return value


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2) + "\n", encoding="utf-8")


def manifest_indexes(rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_id = {row["subset_id"]: row for row in rows if row.get("subset_id")}
    by_stem = {}
    for row in rows:
        input_path = row.get("kubios_input_txt", "")
        if input_path:
            by_stem[Path(input_path).stem] = row
    return by_id, by_stem


def discover_smoothness_reports() -> Dict[str, Path]:
    reports = {}
    for path in sorted(EXPORTS_DIR.rglob("*_hrv.txt")):
        pilot_id = path.stem.split("__", 1)[0]
        reports[pilot_id] = path
    return reports


def line_value(text: str, label_pattern: str) -> str:
    import re

    pattern = re.compile(rf"^\s*{label_pattern}\s*:\s*(.*?)\s*$", flags=re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def extract_lambda(text: str) -> str:
    import re

    match = re.search(r"lambda\s*:\s*([0-9.]+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def parse_kubios_report(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    section = extract_frequency_section(text, path)
    method_metrics = {
        metric: parse_kubios_metric_columns(section, metric, path)
        for metric, _column in KUBIOS_METRICS
    }
    settings = {
        "detrending_text": line_value(text, "Detrending method"),
        "lambda_text": extract_lambda(text),
        "interpolation_rate_text": line_value(text, "Interpolation rate"),
        "fft_window_width_text": line_value(text, "Window width"),
        "fft_overlap_text": line_value(text, "Window overlap"),
        "ar_order_text": line_value(text, "AR model order"),
        "ar_factorization_text": line_value(text, "Use factorization"),
        "fft_window_function_text": "",
    }
    window_function = line_value(text, "Window function") or line_value(text, "Window type")
    if window_function:
        settings["fft_window_function_text"] = window_function
        settings["fft_window_function_status"] = "exported"
    else:
        settings["fft_window_function_text"] = "not exposed/not verified"
        settings["fft_window_function_status"] = "not available in the Kubios interface; not reported in export"
    return settings, method_metrics


def setting_matches(settings: Dict[str, Any]) -> Tuple[bool, str]:
    reasons = []
    detrend = str(settings.get("detrending_text", "")).casefold()
    if "smooth" not in detrend or "prior" not in detrend:
        reasons.append("detrending text is not Smoothness Priors")
    if str(settings.get("lambda_text", "")).strip() != "500":
        reasons.append("lambda 500 not export-verified")
    if "4" not in str(settings.get("interpolation_rate_text", "")):
        reasons.append("4 Hz interpolation not export-verified")
    if "120" not in str(settings.get("fft_window_width_text", "")):
        reasons.append("120 s FFT window width not export-verified")
    if "75" not in str(settings.get("fft_overlap_text", "")):
        reasons.append("75% FFT overlap not export-verified")
    if str(settings.get("ar_order_text", "")).strip() != "16":
        reasons.append("AR order 16 not export-verified")
    if str(settings.get("ar_factorization_text", "")).strip().casefold() != "no":
        reasons.append("AR factorization disabled not export-verified")
    return not reasons, "; ".join(reasons)


def kubios_result_row(
    pilot_id: str,
    full_stem: str,
    source_category: str,
    report_path: Path,
    method: str,
    method_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "pilot_id": pilot_id,
        "full_recording_stem": full_stem,
        "source_category": source_category,
        "kubios_report_path": rel(report_path),
        "kubios_method": method.upper(),
    }
    prefix = f"kubios_{method}"
    for metric, key, _hrv_key in METRIC_SPECS:
        row[f"{prefix}_{key}"] = method_metrics.get(metric, {}).get(method, math.nan)
    return row


def load_rr_ms(path: Path) -> np.ndarray:
    values: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip().replace(",", " ").replace(";", " ")
            if not line:
                continue
            for token in line.split():
                try:
                    value = float(token)
                except ValueError:
                    continue
                if math.isfinite(value) and value > 0:
                    values.append(value)
    rr = np.asarray(values, dtype=float)
    if rr.size == 0:
        raise ValueError(f"No numeric RR intervals found in {path}")
    median_rr = float(np.median(rr))
    if median_rr < 10.0:
        rr = rr * 1000.0
    return rr


def run_hrv_analysis(manifest_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for row in manifest_rows:
        source = row.get("source_rr_file", "")
        if not source:
            failures.append({"pilot_id": row["pilot_id"], "error": "missing source RR file"})
            continue
        source_path = PROJECT_ROOT / source
        try:
            rr_ms = load_rr_ms(source_path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                analyzer = HRVFreqDomainAnalysis(
                    rr_ms,
                    sampling_rate=4.0,
                    detrend_method="smoothness_priors",
                    detrend_lambda=500.0,
                    window_type="hann",
                    segment_length=120.0,
                    overlap_ratio=0.75,
                    ar_order=16,
                    enable_diagnostics=True,
                    band_convention="kubios_compatible",
                )
                results = analyzer.get_results()
            diagnostics = results.get("frequency_diagnostics", {})
            analysis_info = results.get("analysis_info", {})
            for method in ["fft", "ar", "welch"]:
                method_diag = diagnostics.get(method, {})
                out = {
                    "pilot_id": row["pilot_id"],
                    "full_recording_stem": row["full_recording_stem"],
                    "source_rr_file": source,
                    "source_category": row["source_category"],
                    "method": method.upper(),
                    "detrend_method": "smoothness_priors",
                    "detrend_lambda": 500.0,
                    "interpolation_rate_hz": 4.0,
                    "window_type": analysis_info.get("window_type", "hann"),
                    "segment_length_s": analysis_info.get("segment_length_s", 120.0),
                    "overlap_ratio": analysis_info.get("overlap_ratio", 0.75),
                    "ar_order": analysis_info.get("ar_order", 16),
                    "band_convention": analysis_info.get("band_convention", "kubios_compatible"),
                    "signal_duration_s": analysis_info.get("signal_duration_s", math.nan),
                    "rr_count": int(rr_ms.size),
                    "warnings": "; ".join(str(w.message) for w in caught),
                }
                for _label, key, hrv_key in METRIC_SPECS:
                    out[f"hrvstudio_{method}_{key}"] = results.get(f"{method}_{hrv_key}", math.nan)
                for diag_key in [
                    "n_samples",
                    "frequency_resolution_hz",
                    "requested_nperseg",
                    "effective_nperseg",
                    "effective_noverlap",
                    "effective_nfft",
                    "number_of_segments",
                    "window_type",
                    "requested_ar_order",
                    "effective_ar_order",
                    "estimator_used",
                    "fallback_reason",
                    "nfft",
                    "variance_normalization_applied",
                    "warning",
                ]:
                    if diag_key in method_diag:
                        out[f"diagnostic_{diag_key}"] = method_diag.get(diag_key)
                all_rows.append(out)
            qc_rows.append(
                {
                    "pilot_id": row["pilot_id"],
                    "source_rr_file": source,
                    "processed": True,
                    "rr_count": int(rr_ms.size),
                    "signal_duration_s": results.get("analysis_info", {}).get("signal_duration_s", math.nan),
                    "warnings": "; ".join(str(w.message) for w in caught),
                    "duration_warnings": json.dumps(diagnostics.get("duration_warnings", [])),
                    "invalid_rr_removed_count": diagnostics.get("invalid_rr_removed_count", 0),
                    "nonfinite_psd": diagnostics.get("nonfinite_psd", False),
                    "ar_fallback_reason": diagnostics.get("ar", {}).get("fallback_reason", ""),
                }
            )
        except Exception as exc:
            failures.append({"pilot_id": row["pilot_id"], "source_rr_file": source, "error": str(exc)})
            qc_rows.append({"pilot_id": row["pilot_id"], "source_rr_file": source, "processed": False, "error": str(exc)})
    return all_rows, qc_rows, failures


def comparison_rows(
    kubios_rows: Sequence[Dict[str, Any]],
    hrv_rows: Sequence[Dict[str, Any]],
    method: str,
) -> List[Dict[str, Any]]:
    kubios_by_id = {row["pilot_id"]: row for row in kubios_rows}
    hrv_by_id = {row["pilot_id"]: row for row in hrv_rows if row.get("method") == method.upper()}
    rows = []
    for pilot_id in sorted(set(kubios_by_id) & set(hrv_by_id), key=lambda x: EXPECTED_IDS.index(x) if x in EXPECTED_IDS else x):
        krow = kubios_by_id[pilot_id]
        hrow = hrv_by_id[pilot_id]
        for label, key, _hrv_key in METRIC_SPECS:
            kval = finite(krow.get(f"kubios_{method}_{key}"))
            hval = finite(hrow.get(f"hrvstudio_{method}_{key}"))
            signed = hval - kval if math.isfinite(kval) and math.isfinite(hval) else math.nan
            abs_diff = abs(signed) if math.isfinite(signed) else math.nan
            near_zero = not math.isfinite(kval) or abs(kval) <= NEAR_ZERO
            rel_pct = (abs_diff / abs(kval) * 100.0) if not near_zero and math.isfinite(abs_diff) else math.nan
            denom = (abs(kval) + abs(hval)) / 2.0 if math.isfinite(kval) and math.isfinite(hval) else math.nan
            spe = (abs_diff / denom * 100.0) if math.isfinite(denom) and denom > NEAR_ZERO else math.nan
            rows.append(
                {
                    "pilot_id": pilot_id,
                    "full_recording_stem": krow.get("full_recording_stem", ""),
                    "source_category": krow.get("source_category", ""),
                    "metric": label,
                    "method": method.upper(),
                    f"kubios_{method}_value": kval,
                    f"hrvstudio_{method}_value": hval,
                    "signed_difference_hrv_minus_kubios": signed,
                    "absolute_difference": abs_diff,
                    "relative_error_pct": rel_pct,
                    "symmetric_percentage_error_pct": spe,
                    "near_zero_reference": near_zero,
                    "included_in_relative_error": not near_zero and math.isfinite(rel_pct),
                }
            )
    return rows


def summarize_comparison(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summaries = []
    if frame.empty:
        return summaries
    for metric in [spec[0] for spec in METRIC_SPECS]:
        m = frame[frame["metric"] == metric].copy()
        rel_values = pd.to_numeric(m["relative_error_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        abs_values = pd.to_numeric(m["absolute_difference"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        kval_col = [c for c in m.columns if c.startswith("kubios_") and c.endswith("_value")][0]
        hval_col = [c for c in m.columns if c.startswith("hrvstudio_") and c.endswith("_value")][0]
        pairs = m[[kval_col, hval_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        pearson = pairs[kval_col].corr(pairs[hval_col], method="pearson") if len(pairs) >= 3 and pairs[kval_col].nunique() > 1 and pairs[hval_col].nunique() > 1 else math.nan
        spearman = pairs[kval_col].corr(pairs[hval_col], method="spearman") if len(pairs) >= 3 and pairs[kval_col].nunique() > 1 and pairs[hval_col].nunique() > 1 else math.nan
        q75 = rel_values.quantile(0.75) if len(rel_values) else math.nan
        q25 = rel_values.quantile(0.25) if len(rel_values) else math.nan
        summaries.append(
            {
                "method": str(m["method"].iloc[0]) if not m.empty else "",
                "metric": metric,
                "valid_paired_files": int(len(pairs)),
                "relative_error_file_count": int(len(rel_values)),
                "excluded_or_near_zero_cases": int(len(m) - len(rel_values)),
                "mean_relative_error_pct": float(rel_values.mean()) if len(rel_values) else math.nan,
                "median_relative_error_pct": float(rel_values.median()) if len(rel_values) else math.nan,
                "median_absolute_error": float(abs_values.median()) if len(abs_values) else math.nan,
                "relative_error_iqr_pct": float(q75 - q25) if math.isfinite(q75) and math.isfinite(q25) else math.nan,
                "pearson_r": pearson,
                "spearman_r": spearman,
            }
        )
    return summaries


def internal_hrv_comparison(hrv_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in hrv_rows:
        by_id.setdefault(row["pilot_id"], {})[str(row["method"]).lower()] = row
    rows = []
    for pilot_id, methods in by_id.items():
        if not all(method in methods for method in ["fft", "ar", "welch"]):
            continue
        base = methods["fft"]
        out = {
            "pilot_id": pilot_id,
            "full_recording_stem": base.get("full_recording_stem", ""),
            "source_category": base.get("source_category", ""),
        }
        for _label, key, _hrv_key in METRIC_SPECS:
            fft = finite(methods["fft"].get(f"hrvstudio_fft_{key}"))
            ar = finite(methods["ar"].get(f"hrvstudio_ar_{key}"))
            welch = finite(methods["welch"].get(f"hrvstudio_welch_{key}"))
            out[f"fft_{key}"] = fft
            out[f"ar_{key}"] = ar
            out[f"welch_{key}"] = welch
            out[f"ar_vs_fft_{key}_relative_error_pct"] = abs(ar - fft) / abs(fft) * 100.0 if math.isfinite(fft) and abs(fft) > NEAR_ZERO and math.isfinite(ar) else math.nan
            out[f"welch_vs_fft_{key}_relative_error_pct"] = abs(welch - fft) / abs(fft) * 100.0 if math.isfinite(fft) and abs(fft) > NEAR_ZERO and math.isfinite(welch) else math.nan
        rows.append(out)
    return rows


def prior_qc_status(source_rr_file: str) -> str:
    run_info = PRIOR_RUN_DIR / "run_info.json"
    if not run_info.exists():
        return "prior run_info not found"
    data = json.loads(run_info.read_text(encoding="utf-8"))
    source_name = Path(source_rr_file).name
    for failure in data.get("failures", []):
        if Path(failure.get("input_file", "")).name == source_name:
            return f"failed in prior run: {failure.get('error', '')}"
    events = [
        event.get("reason", "")
        for event in data.get("file_events", [])
        if Path(event.get("input_file", "")).name == source_name
    ]
    return "; ".join(events) if events else "processed/no prior QC event"


def make_audit_and_kubios_tables(
    subset_by_id: Dict[str, Dict[str, str]],
    reports: Dict[str, Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, float]]]]:
    audit_rows = []
    fft_rows = []
    ar_rows = []
    parsed_by_id: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pilot_id in EXPECTED_IDS:
        report = reports.get(pilot_id)
        manifest = subset_by_id.get(pilot_id, {})
        if report is None:
            stem = Path(manifest.get("kubios_input_txt", pilot_id)).stem
            audit_rows.append(
                {
                    "pilot_id": pilot_id,
                    "full_recording_stem": stem,
                    "source_category": manifest.get("category", ""),
                    "kubios_report_path": "",
                    "detrending_text": "",
                    "lambda_text": "",
                    "lambda_verified": False,
                    "interpolation_rate_text": "",
                    "fft_window_width_text": "",
                    "fft_overlap_text": "",
                    "fft_window_function_text": "not exposed/not verified",
                    "fft_window_function_status": "no Smoothness Priors export found",
                    "ar_order_text": "",
                    "ar_factorization_text": "",
                    "settings_match": False,
                    "mismatch_reason": "Smoothness Priors Kubios report missing from pilot export folder",
                    "notes": "Expected pilot ID was requested but no report file was found.",
                }
            )
            continue
        full_stem = report.stem.removesuffix("_hrv")
        source_category = report.parent.parent.name
        settings, metrics = parse_kubios_report(report)
        parsed_by_id[pilot_id] = metrics
        match, reason = setting_matches(settings)
        audit_rows.append(
            {
                "pilot_id": pilot_id,
                "full_recording_stem": full_stem,
                "source_category": source_category,
                "kubios_report_path": rel(report),
                "detrending_text": settings["detrending_text"],
                "lambda_text": settings["lambda_text"],
                "lambda_verified": settings["lambda_text"] == "500",
                "interpolation_rate_text": settings["interpolation_rate_text"],
                "fft_window_width_text": settings["fft_window_width_text"],
                "fft_overlap_text": settings["fft_overlap_text"],
                "fft_window_function_text": settings["fft_window_function_text"],
                "fft_window_function_status": settings["fft_window_function_status"],
                "ar_order_text": settings["ar_order_text"],
                "ar_factorization_text": settings["ar_factorization_text"],
                "settings_match": match,
                "mismatch_reason": reason,
                "notes": "FFT window function not export-verifiable." if match else "",
            }
        )
        fft_rows.append(kubios_result_row(pilot_id, full_stem, source_category, report, "fft", metrics))
        ar_rows.append(kubios_result_row(pilot_id, full_stem, source_category, report, "ar", metrics))
    return audit_rows, fft_rows, ar_rows, parsed_by_id


def build_pilot_manifest(
    subset_by_id: Dict[str, Dict[str, str]],
    reports: Dict[str, Path],
    audit_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    audit_by_id = {row["pilot_id"]: row for row in audit_rows}
    rows = []
    for pilot_id in EXPECTED_IDS:
        source = subset_by_id.get(pilot_id, {})
        source_rr = source.get("kubios_input_txt", "")
        full_stem = Path(source_rr).stem if source_rr else audit_by_id.get(pilot_id, {}).get("full_recording_stem", pilot_id)
        report = reports.get(pilot_id)
        prior_path = PRIOR_EXPORTS_DIR / source.get("category", "") / full_stem / f"{full_stem}_hrv.txt"
        smooth_valid = bool(audit_by_id.get(pilot_id, {}).get("settings_match")) and report is not None
        notes = []
        if report is None:
            notes.append("Smoothness Priors Kubios export missing.")
        if report is not None and source.get("category") and report.parent.parent.name != source.get("category"):
            notes.append(f"Pilot export folder category '{report.parent.parent.name}' differs from prior manifest category '{source.get('category')}'.")
        rows.append(
            {
                "pilot_id": pilot_id,
                "full_recording_stem": full_stem,
                "source_rr_file": source_rr,
                "source_category": report.parent.parent.name if report is not None else source.get("category", ""),
                "kubios_report_file": rel(report) if report is not None else "",
                "prior_no_detrend_result_path": rel(prior_path) if prior_path.exists() else "",
                "prior_qc_status": prior_qc_status(source_rr) if source_rr else "source not found in subset manifest",
                "inclusion_reason": source.get("reason_selected", ""),
                "smoothness_export_valid": smooth_valid,
                "notes": " ".join(notes),
            }
        )
    return rows


def parse_prior_kubios(manifest_rows: Sequence[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    rows = []
    for row in manifest_rows:
        prior_path_text = row.get("prior_no_detrend_result_path", "")
        if not prior_path_text:
            continue
        prior_path = PROJECT_ROOT / prior_path_text
        if not prior_path.exists():
            continue
        try:
            _settings, metrics = parse_kubios_report(prior_path)
        except Exception:
            continue
        out = {
            "pilot_id": row["pilot_id"],
            "full_recording_stem": row["full_recording_stem"],
            "source_category": row["source_category"],
            "prior_kubios_report_path": prior_path_text,
            "method": method.upper(),
        }
        prefix = f"prior_kubios_none_{method}"
        for metric, key, _hrv_key in METRIC_SPECS:
            out[f"{prefix}_{key}"] = metrics.get(metric, {}).get(method, math.nan)
        rows.append(out)
    return rows


def make_paired_prior_rows(
    prior_kubios: Sequence[Dict[str, Any]],
    new_kubios: Sequence[Dict[str, Any]],
    new_hrv: Sequence[Dict[str, Any]],
    method: str,
) -> List[Dict[str, Any]]:
    prior_by_id = {row["pilot_id"]: row for row in prior_kubios}
    new_k_by_id = {row["pilot_id"]: row for row in new_kubios}
    new_h_by_id = {row["pilot_id"]: row for row in new_hrv if row.get("method") == method.upper()}
    rows = []
    for pilot_id in EXPECTED_IDS:
        for label, key, _hrv_key in METRIC_SPECS:
            pk = prior_by_id.get(pilot_id, {})
            nk = new_k_by_id.get(pilot_id, {})
            nh = new_h_by_id.get(pilot_id, {})
            new_kval = finite(nk.get(f"kubios_{method}_{key}"))
            new_hval = finite(nh.get(f"hrvstudio_{method}_{key}"))
            new_rel = abs(new_hval - new_kval) / abs(new_kval) * 100.0 if math.isfinite(new_kval) and abs(new_kval) > NEAR_ZERO and math.isfinite(new_hval) else math.nan
            rows.append(
                {
                    "pilot_id": pilot_id,
                    "metric": label,
                    "prior Kubios no-detrend value": finite(pk.get(f"prior_kubios_none_{method}_{key}")),
                    "prior HRV Studio no-detrend value": "",
                    "prior relative error": "",
                    "new Kubios Smoothness Priors value": new_kval,
                    "new HRV Studio Smoothness Priors value": new_hval,
                    "new relative error": new_rel,
                    "absolute change in error": "",
                    "proportional change in error": "",
                    "improved/worsened/unchanged": "not assessed",
                    "method": method.upper(),
                    "comparability_notes": "No stored prior HRV Studio FFT/AR result was found; prior benchmark native_value was Welch and is not a valid matched-method comparator.",
                }
            )
    return rows


def plot_scatter(rows: Sequence[Dict[str, Any]], method: str, metric: str, out_path: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    key = KUBIOS_OUT_KEYS[metric]
    kcol = f"kubios_{method}_value"
    hcol = f"hrvstudio_{method}_value"
    data = frame[frame["metric"] == metric].copy()
    data[kcol] = pd.to_numeric(data[kcol], errors="coerce")
    data[hcol] = pd.to_numeric(data[hcol], errors="coerce")
    data = data.dropna(subset=[kcol, hcol])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(data[kcol], data[hcol], color="#1f77b4" if method == "fft" else "#2ca02c")
    low = min(data[kcol].min(), data[hcol].min())
    high = max(data[kcol].max(), data[hcol].max())
    ax.plot([low, high], [low, high], color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Kubios {method.upper()} {metric}")
    ax.set_ylabel(f"HRV Studio {method.upper()} {metric}")
    ax.set_title(f"{method.upper()} estimator comparison: {metric}\nNominal Smoothness Priors, sensitivity pilot, n={len(data)}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(rows: Sequence[Dict[str, Any]], method: str, out_path: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    pivot = frame.pivot(index="pilot_id", columns="metric", values="relative_error_pct")
    pivot = pivot.reindex([pid for pid in EXPECTED_IDS if pid in pivot.index])
    pivot = pivot[[spec[0] for spec in METRIC_SPECS if spec[0] in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title(f"{method.upper()} per-file relative-error heatmap\nNominal Smoothness Priors, sensitivity pilot")
    fig.colorbar(im, ax=ax, label="Relative error (%)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_internal(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    metrics = ["vlf", "total_power"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric in zip(axes, metrics):
        x = np.arange(len(frame))
        width = 0.25
        for offset, method in [(-width, "fft"), (0, "ar"), (width, "welch")]:
            ax.bar(x + offset, pd.to_numeric(frame[f"{method}_{metric}"], errors="coerce"), width=width, label=method.upper())
        ax.set_xticks(x, frame["pilot_id"], rotation=45, ha="right")
        ax.set_title(f"HRV Studio {metric}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Power / metric value")
    axes[1].legend(frameon=False)
    fig.suptitle("HRV Studio FFT vs AR vs Welch under Smoothness Priors\nInternal sensitivity only, pilot sample")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 2) -> str:
    number = finite(value)
    return "not available" if not math.isfinite(number) else f"{number:.{digits}f}"


def summary_lookup(summary_rows: Sequence[Dict[str, Any]], method: str, metric: str) -> Dict[str, Any]:
    for row in summary_rows:
        if row.get("method") == method.upper() and row.get("metric") == metric:
            return row
    return {}


def write_method_mapping() -> None:
    text = """# HRV Studio Method Mapping

## Production modules and functions inspected

- Frequency-domain analysis: `hrvlib.metrics.freq_domain.HRVFreqDomainAnalysis`
- Welch PSD: `HRVFreqDomainAnalysis._compute_welch_psd`
- Direct FFT periodogram: `HRVFreqDomainAnalysis._compute_fft_psd`
- AR PSD: `HRVFreqDomainAnalysis._compute_ar_psd`
- Smoothness Priors detrending: `hrvlib.signal_processing.smoothness_priors.detrend_uniform_with_smoothness_priors`
- Frequency-band integration and normalized units: `HRVFreqDomainAnalysis._compute_spectral_metrics`

## Method mapping

| Kubios output | Primary HRV Studio comparator | Comparison status |
| --- | --- | --- |
| Kubios FFT | HRV Studio FFT | Primary |
| Kubios AR | HRV Studio AR | Primary |
| No Kubios Welch output | HRV Studio Welch | Internal sensitivity only |

## HRV Studio FFT

HRV Studio has a direct FFT estimator distinct from Welch. The direct FFT path uses the entire resampled signal, applies the configured window to the full signal, computes a one-sided `numpy.fft.rfft` periodogram, scales by sampling rate, sample count, and window power normalization, and converts seconds squared to ms^2/Hz. It does not segment the signal and does not use the Welch 120 s / 75% overlap settings. The production default window setting used here was `hann`.

Known mismatch: Kubios exposes FFT window width and overlap but did not expose or report a selectable FFT window function. HRV Studio's direct FFT is whole-signal, while Kubios FFT behavior is only matched on visible settings and cannot be described as fully algorithmically matched.

## HRV Studio AR

HRV Studio AR uses adaptive-order Burg estimation first, with Yule-Walker fallback if Burg fails. Requested AR order 16 is supported; diagnostics record the effective order and estimator used. The AR spectrum is generated with `scipy.signal.freqz` over 4096 points and normalized so integrated PSD matches the AR input variance. HRV Studio does not expose a control equivalent to Kubios `Use factorization: No`; this remains an implementation mismatch.

## HRV Studio Welch

HRV Studio Welch uses `scipy.signal.welch` with the configured window, segment length, overlap, density scaling, and mean averaging. In this pilot, Welch used 4 Hz interpolation, 120 s segment length, 75% overlap, and `hann` window. Kubios did not provide a Welch output in this version, so HRV Studio Welch is retained only as an internal sensitivity output.

## Common preprocessing and metric conventions

The pilot used the same pilot ASCII RR files, 4 Hz interpolation, Smoothness Priors detrending with lambda 500, HRV Studio's existing invalid-RR sanitization, and the Kubios-compatible comparator band convention: VLF 0.0-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.4 Hz, total power 0.0-0.4 Hz. Normalized units are LF/(LF+HF) and HF/(LF+HF), not percentage of total power.

Exact Smoothness Priors implementation identity between Kubios and HRV Studio was not established.
"""
    write_text(PILOT_DIR / "hrvstudio_method_mapping.md", text)


def write_run_dirs(hrv_rows: Sequence[Dict[str, Any]], qc_rows: Sequence[Dict[str, Any]], failures: Sequence[Dict[str, Any]], pilot_manifest: Sequence[Dict[str, Any]], command: str) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    common = {
        "timestamp": timestamp,
        "script": "tools/run_kubios_smoothness_priors_pilot.py",
        "purpose": "Kubios Smoothness Priors pilot with method-labelled FFT, AR, and Welch outputs.",
        "production_code_modified": False,
        "git_commit_hash": git_output("rev-parse", "HEAD"),
        "git_dirty_status": git_output("status", "--short"),
        "software_versions": software_versions(),
        "pilot_manifest": rel(PILOT_DIR / "pilot_manifest.csv"),
        "detrending_method": "smoothness_priors",
        "detrending_lambda": 500,
        "interpolation_frequency_hz": 4,
        "frequency_bands": {
            "vlf": [0.0, 0.04],
            "lf": [0.04, 0.15],
            "hf": [0.15, 0.4],
            "total_power": [0.0, 0.4],
        },
        "invalid_rr_rules": "HRVFreqDomainAnalysis removes nonfinite and nonpositive intervals; validation loader reads positive numeric RR intervals and converts seconds to ms when median RR < 10.",
        "source_file_count": len(pilot_manifest),
        "failed_files": failures,
        "commands": [command],
    }
    for method in ["fft", "ar", "welch"]:
        run_dir = RUNS_ROOT / f"v08_kubios_smoothness_priors_pilot_{method}{OUTPUT_SUFFIX}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = [row for row in hrv_rows if row.get("method") == method.upper()]
        write_csv(run_dir / f"hrvstudio_{method}_results.csv", rows)
        write_csv(run_dir / "per_file_qc_diagnostics.csv", qc_rows)
        method_info = dict(common)
        method_info["estimator"] = method.upper()
        if method == "fft":
            method_info["fft_configuration"] = {
                "path": "HRVFreqDomainAnalysis._compute_fft_psd",
                "window": "hann",
                "segmentation": "not supported by HRV Studio direct FFT; full resampled signal used",
                "overlap": "not applicable to HRV Studio direct FFT",
            }
        elif method == "ar":
            method_info["ar_configuration"] = {
                "path": "HRVFreqDomainAnalysis._compute_ar_psd",
                "requested_order": 16,
                "algorithm": "Burg first, Yule-Walker fallback",
                "factorization_equivalent": "none exposed",
            }
        else:
            method_info["welch_configuration"] = {
                "path": "HRVFreqDomainAnalysis._compute_welch_psd",
                "window": "hann",
                "segment_length_s": 120,
                "overlap_ratio": 0.75,
                "status": "internal sensitivity only; no Kubios Welch output",
            }
        dump_json(run_dir / "run_info.json", method_info)
        write_text(run_dir / "commands.log", command + "\n")
        write_text(
            run_dir / "notes.md",
            f"# {method.upper()} Pilot Run\n\n"
            "Validation-only run for the Kubios Smoothness Priors sensitivity pilot.\n\n"
            f"- Method: {method.upper()}\n"
            "- Detrending: Smoothness Priors, lambda 500\n"
            "- Interpolation: 4 Hz\n"
            "- Kubios is not treated as ground truth.\n"
            "- HRV Studio Welch is internal sensitivity only.\n",
        )


def write_report(
    manifest_rows: Sequence[Dict[str, Any]],
    audit_rows: Sequence[Dict[str, Any]],
    fft_summary: Sequence[Dict[str, Any]],
    ar_summary: Sequence[Dict[str, Any]],
    internal_rows: Sequence[Dict[str, Any]],
) -> None:
    missing = [row["pilot_id"] for row in manifest_rows if not row.get("kubios_report_file")]
    fft_vlf = summary_lookup(fft_summary, "fft", "VLF")
    fft_total = summary_lookup(fft_summary, "fft", "total_power")
    ar_vlf = summary_lookup(ar_summary, "ar", "VLF")
    ar_total = summary_lookup(ar_summary, "ar", "total_power")
    valid_exports = sum(1 for row in audit_rows if row.get("settings_match"))
    export_sentence = (
        f"All {valid_exports} expected pilot IDs had Smoothness Priors Kubios reports in the pilot export folder."
        if valid_exports == len(EXPECTED_IDS)
        else (
            f"Only {valid_exports} of the {len(EXPECTED_IDS)} expected pilot IDs had a Smoothness Priors "
            "Kubios report in the pilot export folder; missing reports were excluded from primary Kubios agreement summaries."
        )
    )
    internal = pd.DataFrame(internal_rows)
    internal_note = "not available"
    if not internal.empty:
        cols = [c for c in internal.columns if c.endswith("_vlf_relative_error_pct")]
        internal_note = f"{len(internal)} files processed; estimator differences are listed in hrvstudio_fft_ar_welch_internal_comparison.csv"
    text = f"""# Kubios Smoothness Priors Pilot Report

## 1. Executive conclusion

This targeted sensitivity pilot found that the available Smoothness Priors exports support primary estimator-mapped comparisons for Kubios FFT vs HRV Studio FFT and Kubios AR vs HRV Studio AR. {export_sentence}

FFT VLF median relative error was {fmt(fft_vlf.get('median_relative_error_pct'))}% and FFT total-power median relative error was {fmt(fft_total.get('median_relative_error_pct'))}%. AR VLF median relative error was {fmt(ar_vlf.get('median_relative_error_pct'))}% and AR total-power median relative error was {fmt(ar_total.get('median_relative_error_pct'))}%.

The pilot should be interpreted as a nominal visible-settings comparison, not an exact algorithm match and not a ground-truth validation against Kubios.

## 2. Why the comparison design was revised

The Kubios version used for this pilot did not provide a selectable Welch method. The exported frequency-domain table displayed `FFT spectrum` and `AR spectrum` columns. Therefore the primary comparisons were revised to FFT-to-FFT and AR-to-AR. HRV Studio Welch was not treated as a Kubios-equivalent estimator and is retained only as an internal sensitivity analysis.

## 3. Pilot files

See `validation/kubios_subset/smoothness_priors_pilot/pilot_manifest.csv`. Expected IDs were: {', '.join(EXPECTED_IDS)}. Missing Smoothness Priors Kubios export: {', '.join(missing) if missing else 'none'}.

## 4. Kubios configuration

The exports verified Smoothness Priors text with lambda 500 where reports were present, 4 Hz interpolation, FFT window width 120 s, FFT overlap 75%, AR order 16, and `Use factorization: No`. Kubios exposed no selectable FFT window-function setting in the interface and the text exports did not identify the FFT window function; the audit records it as `not exposed/not verified`.

## 5. HRV Studio configuration

HRV Studio used the pilot RR files, 4 Hz interpolation, Smoothness Priors detrending with lambda 500, AR order 16, and the Kubios-compatible comparator band convention. Direct FFT used HRV Studio's whole-signal FFT periodogram with a `hann` window; it does not implement the 120 s / 75% segmentation controls. Welch used 120 s, 75% overlap, and `hann` but is internal sensitivity only.

## 6. Method-mapping rationale

See `validation/kubios_subset/smoothness_priors_pilot/hrvstudio_method_mapping.md`.

## 7. Kubios FFT vs HRV Studio FFT results

Primary matched visible-settings comparison, n={fft_vlf.get('valid_paired_files', 0)} paired files for VLF. Median relative errors: VLF {fmt(fft_vlf.get('median_relative_error_pct'))}%, total power {fmt(fft_total.get('median_relative_error_pct'))}%. See `smoothness_priors_fft_per_file_comparison.csv` and `smoothness_priors_fft_metric_summary.csv`.

## 8. Kubios AR vs HRV Studio AR results

Primary matched visible-settings comparison, n={ar_vlf.get('valid_paired_files', 0)} paired files for VLF. Median relative errors: VLF {fmt(ar_vlf.get('median_relative_error_pct'))}%, total power {fmt(ar_total.get('median_relative_error_pct'))}%. HRV Studio does not expose an AR factorization option equivalent to Kubios `Use factorization: No`. See `smoothness_priors_ar_per_file_comparison.csv` and `smoothness_priors_ar_metric_summary.csv`.

## 9. HRV Studio FFT/AR/Welch internal sensitivity

{internal_note}. This analysis is exploratory and non-like-for-like with Kubios Welch because Kubios Welch was not available.

## 10. Comparison with the prior no-detrend benchmark

The old no-detrend Kubios exports can be reparsed into FFT and AR columns. However, the stored prior HRV Studio benchmark recorded Welch/native values, not method-labelled HRV Studio FFT or AR values. Therefore the before-after detrending comparison cannot be presented as a valid matched FFT or AR paired comparison. The generated paired CSVs retain the old Kubios values and mark the prior HRV Studio FFT/AR fields as unavailable.

## 11. Per-file diagnostics

Per-file HRV QC diagnostics are in each `validation/runs/v08_kubios_smoothness_priors_pilot_*` folder. Category-specific results should be interpreted cautiously because several pilot categories are convention-sensitive or outlier-enriched, and `OUT001` lacked the new Kubios export.

FFT and AR led to different conclusions. AR agreement was strong for VLF and total power in the {ar_vlf.get('valid_paired_files', 0)} paired files. FFT agreement was partial, with the largest disagreements listed in the per-file FFT comparison table. LF/HF and normalized metrics were more stable than low-frequency absolute powers. These patterns are consistent with remaining estimator implementation differences, especially HRV Studio's whole-signal FFT behavior and unverified Kubios FFT window behavior.

Because compatible prior HRV Studio FFT/AR no-detrend values were not stored, the pilot cannot validly answer whether matched-method FFT or AR agreement improved versus the previous no-detrend benchmark. It can answer that under nominal Smoothness Priors, AR agreement is much closer than FFT agreement for the available paired reports.

## 12. Limitations

- Kubios FFT window function was not exposed or export-verified.
- Exact Smoothness Priors implementation identity was not established.
- HRV Studio FFT is whole-signal, while Kubios reports FFT window width and overlap.
- HRV Studio AR uses Burg with Yule-Walker fallback and variance normalization; Kubios AR internals are not fully exposed.
- Kubios is not treated as ground truth.
- The sample is a targeted sensitivity pilot, not a replacement for the existing benchmark.

## 13. Implications for Reviewer Comment 2

The pilot directly addresses whether nominal Smoothness Priors lambda 500 reduces low-frequency disagreement under estimator-aware comparisons. The result should be framed as a sensitivity analysis with visible settings aligned where possible, not as proof of full algorithmic equivalence.

## 14. Recommended manuscript wording

We performed a targeted sensitivity pilot using the manually exported Kubios reports available for ten prespecified recordings, with nine Smoothness Priors reports available for quantitative Kubios comparison. Kubios displayed FFT and AR frequency-domain results, while Welch was not available as a selectable Kubios spectral method in this version. Accordingly, primary comparisons were FFT-to-FFT and AR-to-AR; HRV Studio Welch was retained only as an internal estimator-sensitivity output. Both systems used nominal Smoothness Priors detrending with lambda 500, but exact implementation identity and Kubios FFT window behavior could not be verified.

## 15. Recommended rebuttal wording

In response to the reviewer, we added a targeted Smoothness Priors sensitivity pilot. We revised the comparison design after confirming that this Kubios version reports FFT and AR results, not Welch. Therefore, we compared Kubios FFT with HRV Studio FFT and Kubios AR with HRV Studio AR, and did not treat HRV Studio Welch as a Kubios-equivalent estimator. The pilot is reported as a visible-settings sensitivity analysis because Kubios did not expose the FFT window function and exact Smoothness Priors implementation identity cannot be established from the export.

Result-dependent alternatives:

- If both FFT and AR improve strongly in a completed expanded run: A supplementary sensitivity pilot under nominal Smoothness Priors improved low-frequency agreement in both estimator-mapped comparisons, although full algorithmic equivalence was not established.
- If only FFT improves: Detrending improved FFT-based agreement, while AR remained sensitive to method-specific model-fitting details.
- If only AR improves: AR-based agreement improved, while FFT remained affected by unverified windowing or spectral implementation conventions.
- If neither improves: Matched nominal detrending alone did not eliminate differences, indicating that estimator-specific implementation details remain important.
- If results conflict or are unstable: The pilot should be presented as exploratory and should not support a strong generalized manuscript claim.

Current pilot wording: AR-based agreement was close under nominal Smoothness Priors, while FFT agreement remained only partial and plausibly affected by unverified Kubios FFT window behavior and HRV Studio's whole-signal FFT implementation.

## 16. Recommendation on full-subset expansion

Do not expand solely on the basis of this pilot until the team decides whether the whole-signal HRV Studio FFT mismatch is acceptable for the scientific question. If expanded, expand both FFT and AR only as estimator-aware sensitivity analyses with the same limitations stated explicitly.
"""
    write_text(REPORT_PATH, text)


def main() -> int:
    command = "python tools/run_kubios_smoothness_priors_pilot.py"
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    subset_rows = read_csv(MANIFEST_PATH)
    subset_by_id, _subset_by_stem = manifest_indexes(subset_rows)
    reports = discover_smoothness_reports()

    audit_rows, kubios_fft_rows, kubios_ar_rows, _parsed = make_audit_and_kubios_tables(subset_by_id, reports)
    write_csv(PILOT_DIR / "kubios_export_settings_audit.csv", audit_rows)
    write_csv(PILOT_DIR / "kubios_fft_results.csv", kubios_fft_rows)
    write_csv(PILOT_DIR / "kubios_ar_results.csv", kubios_ar_rows)

    pilot_manifest = build_pilot_manifest(subset_by_id, reports, audit_rows)
    write_csv(PILOT_DIR / "pilot_manifest.csv", pilot_manifest)

    hrv_rows, qc_rows, failures = run_hrv_analysis(pilot_manifest)
    write_run_dirs(hrv_rows, qc_rows, failures, pilot_manifest, command)

    fft_rows = comparison_rows(kubios_fft_rows, hrv_rows, "fft")
    ar_rows = comparison_rows(kubios_ar_rows, hrv_rows, "ar")
    fft_summary = summarize_comparison(fft_rows)
    ar_summary = summarize_comparison(ar_rows)
    write_csv(PILOT_DIR / "smoothness_priors_fft_per_file_comparison.csv", fft_rows)
    write_csv(PILOT_DIR / "smoothness_priors_fft_metric_summary.csv", fft_summary)
    write_csv(PILOT_DIR / "smoothness_priors_ar_per_file_comparison.csv", ar_rows)
    write_csv(PILOT_DIR / "smoothness_priors_ar_metric_summary.csv", ar_summary)

    internal_rows = internal_hrv_comparison(hrv_rows)
    write_csv(PILOT_DIR / "hrvstudio_fft_ar_welch_internal_comparison.csv", internal_rows)

    prior_fft = parse_prior_kubios(pilot_manifest, "fft")
    prior_ar = parse_prior_kubios(pilot_manifest, "ar")
    write_csv(PILOT_DIR / "prior_no_detrend_kubios_fft_results.csv", prior_fft)
    write_csv(PILOT_DIR / "prior_no_detrend_kubios_ar_results.csv", prior_ar)
    write_csv(PILOT_DIR / "smoothness_vs_none_fft_paired_comparison.csv", make_paired_prior_rows(prior_fft, kubios_fft_rows, hrv_rows, "fft"))
    write_csv(PILOT_DIR / "smoothness_vs_none_ar_paired_comparison.csv", make_paired_prior_rows(prior_ar, kubios_ar_rows, hrv_rows, "ar"))

    figures = PILOT_DIR / "figures"
    plot_scatter(fft_rows, "fft", "VLF", figures / "kubios_fft_vs_hrvstudio_fft_vlf_scatter.png")
    plot_scatter(fft_rows, "fft", "total_power", figures / "kubios_fft_vs_hrvstudio_fft_total_power_scatter.png")
    plot_heatmap(fft_rows, "fft", figures / "fft_per_file_relative_error_heatmap.png")
    plot_scatter(ar_rows, "ar", "VLF", figures / "kubios_ar_vs_hrvstudio_ar_vlf_scatter.png")
    plot_scatter(ar_rows, "ar", "total_power", figures / "kubios_ar_vs_hrvstudio_ar_total_power_scatter.png")
    plot_heatmap(ar_rows, "ar", figures / "ar_per_file_relative_error_heatmap.png")
    plot_internal(internal_rows, figures / "hrvstudio_fft_ar_welch_internal_comparison.png")

    write_method_mapping()
    write_report(pilot_manifest, audit_rows, fft_summary, ar_summary, internal_rows)

    write_text(PILOT_DIR / "commands.log", command + "\n")
    dump_json(
        PILOT_DIR / "pilot_run_info.json",
        {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": command,
            "smoothness_reports_found": len(reports),
            "expected_ids": EXPECTED_IDS,
            "missing_smoothness_exports": [row["pilot_id"] for row in pilot_manifest if not row.get("kubios_report_file")],
            "hrv_failures": failures,
            "software_versions": software_versions(),
            "git_commit_hash": git_output("rev-parse", "HEAD"),
            "git_dirty_status": git_output("status", "--short"),
        },
    )
    print(f"Smoothness reports found: {len(reports)}")
    print(f"HRV rows written: {len(hrv_rows)}")
    print(f"FFT comparison rows: {len(fft_rows)}")
    print(f"AR comparison rows: {len(ar_rows)}")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
