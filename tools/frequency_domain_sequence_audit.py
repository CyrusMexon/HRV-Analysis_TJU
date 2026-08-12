"""Audit Kubios frequency-domain sequence alignment for the 44-recording subset.

This is an isolated validation script. It does not modify production code or
overwrite the original benchmark outputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis


COHORT = ROOT / "validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv"
ORIGINAL_SUMMARY = (
    ROOT
    / "validation/research_notes/final_validation_results_package/final_kubios_metric_table.csv"
)
EXPORT_ROOT = ROOT / "validation/kubios_subset/kubios_exports_120s_75pct_none"
RUN_INFO = ROOT / "validation/runs/v07_kubios_subset_50_none_120_75_after_arm_a/run_info.json"
OUTPUT_SUFFIX = os.environ.get("HRV_FREQ_BAND_RERUN_SUFFIX", "").strip()

DEBUG_DIR = ROOT / f"validation/kubios_subset/frequency_domain_sequence_debug{OUTPUT_SUFFIX}"
AUDIT_CSV = DEBUG_DIR / "frequency_domain_sequence_audit.csv"
FULL_VS_SELECTED_SUMMARY = DEBUG_DIR / "full_vs_selected_sequence_metric_summary.csv"
FULL_VS_SELECTED_PER_RECORDING = DEBUG_DIR / "full_vs_selected_sequence_per_recording.csv"
MATCHED_SUMMARY = ROOT / f"validation/kubios_subset/frequency_domain_kubios_matched_sequence_summary{OUTPUT_SUFFIX}.csv"
MATCHED_PER_RECORDING = (
    ROOT / f"validation/kubios_subset/frequency_domain_kubios_matched_sequence_per_recording{OUTPUT_SUFFIX}.csv"
)
ORIGINAL_VS_MATCHED = (
    ROOT / f"validation/kubios_subset/frequency_domain_original_vs_matched_sequence_summary{OUTPUT_SUFFIX}.csv"
)
AUDIT_REPORT = (
    ROOT
    / f"validation/research_notes/kubios_frequency_domain_sequence_alignment_audit{OUTPUT_SUFFIX}.md"
)
MATCHED_REPORT = (
    ROOT
    / f"validation/research_notes/frequency_domain_kubios_matched_sequence_validation_report{OUTPUT_SUFFIX}.md"
)

METRICS = ["VLF", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu"]
HRV_KEYS = {
    "VLF": "welch_vlf_power",
    "LF": "welch_lf_power",
    "HF": "welch_hf_power",
    "total_power": "welch_total_power",
    "LF/HF": "welch_lf_hf_ratio",
    "LF_nu": "welch_lf_nu",
    "HF_nu": "welch_hf_nu",
}
KUBIOS_MAT_KEYS = {
    "VLF": "VLF_power",
    "LF": "LF_power",
    "HF": "HF_power",
    "total_power": "tot_power",
    "LF/HF": "LF_HF_power",
    "LF_nu": "LF_power_nu",
    "HF_nu": "HF_power_nu",
}
KUBIOS_MAT_POWER_METRICS = {"VLF", "LF", "HF", "total_power"}
TXT_LABELS = {
    "VLF": "VLF (ms^2)",
    "LF": "LF (ms^2)",
    "HF": "HF (ms^2)",
    "total_power": "Total power (ms^2)",
    "LF/HF": "LF/HF ratio",
    "LF_nu": "LF (n.u.)",
    "HF_nu": "HF (n.u.)",
}
METRIC_LABELS = {
    "VLF": "VLF",
    "LF": "LF",
    "HF": "HF",
    "total_power": "Total Power",
    "LF/HF": "LF/HF",
    "LF_nu": "LFnu",
    "HF_nu": "HFnu",
}


def rel_error_pct(observed: float, reference: float) -> float:
    if not np.isfinite(observed) or not np.isfinite(reference) or reference == 0:
        return math.nan
    return abs(observed - reference) / abs(reference) * 100.0


def safe_float(value) -> float:
    try:
        result = float(np.asarray(value).squeeze())
    except Exception:
        return math.nan
    return result if np.isfinite(result) else math.nan


def safe_array(value) -> np.ndarray:
    arr = np.asarray(value).squeeze()
    if arr.ndim == 0:
        return np.asarray([float(arr)])
    return arr.astype(float)


def get_field(obj, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    raise AttributeError(name)


def parse_kubios_txt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for metric, label in TXT_LABELS.items():
        pattern = rf"^\s*{re.escape(label)}\s*:?\s*;([^\n\r]*)"
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            values[metric] = math.nan
            continue
        fields = [part.strip() for part in match.group(1).split(";")]
        values[metric] = safe_float(fields[0]) if fields else math.nan

    sample_limits = ""
    for label in ("Sample limits", "Analysis window", "Time limits"):
        match = re.search(
            rf"^\s*{re.escape(label)}[^\n\r;]*;([^\n\r]*)",
            text,
            re.MULTILINE,
        )
        if match:
            sample_limits = match.group(1).strip()
            break

    data_length_s = math.nan
    match = re.search(r"^\s*Data length\s*;([^\n\r]*)", text, re.MULTILINE)
    if match:
        fields = [part.strip() for part in match.group(1).split(";")]
        data_length_s = safe_float(fields[0]) if fields else math.nan

    return {
        "metrics": values,
        "sample_limits": sample_limits,
        "data_length_s": data_length_s,
    }


def markdown_table(rows: list[dict], floatfmt: str = ".3f") -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())

    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "NA"
            return format(float(value), floatfmt)
        return str(value)

    rendered = [[cell(row.get(header, "")) for header in headers] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(
        str(header).ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    body = [
        "| "
        + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        + " |"
        for row in rendered
    ]
    return "\n".join([header_line, sep_line, *body])


def export_paths(recording_id: str, category: str) -> tuple[Path, Path]:
    category_dir = EXPORT_ROOT / category
    matches = sorted(category_dir.glob(f"{recording_id}__*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one export folder for {recording_id}, got {len(matches)}")
    folder = matches[0]
    txt_matches = sorted(folder.glob("*_hrv.txt"))
    mat_matches = sorted(folder.glob("*_hrv.mat"))
    if len(txt_matches) != 1 or len(mat_matches) != 1:
        raise FileNotFoundError(f"Missing unique TXT/MAT export for {recording_id} in {folder}")
    return txt_matches[0], mat_matches[0]


def run_hrvstudio_frequency(rr_ms: np.ndarray, settings: dict) -> tuple[dict, dict]:
    kwargs = {
        "preprocessed_rri": rr_ms.tolist(),
        "sampling_rate": settings["sampling_rate"],
        "window_type": settings["window_type"],
        "segment_length": settings["segment_length"],
        "overlap_ratio": settings["overlap_ratio"],
        "detrend_method": settings.get("detrend_method"),
        "detrend_lambda": settings.get("detrend_lambda", 500.0),
        "ar_order": settings.get("ar_order", 16),
        "enable_diagnostics": True,
        "band_convention": settings.get("band_convention", "kubios_compatible"),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(StringIO()):
            analyzer = HRVFreqDomainAnalysis(**kwargs)
            result = analyzer.get_results()
    metrics = {metric: safe_float(result.get(HRV_KEYS[metric])) for metric in METRICS}
    diagnostics = getattr(analyzer, "_welch_diagnostics", {}) or {}
    return metrics, diagnostics


def summarize_errors(rows: list[dict], condition: str) -> list[dict]:
    output = []
    for metric in METRICS:
        subset = [r for r in rows if r["Metric"] == metric]
        if condition == "full":
            values = np.asarray([r["full_hrvstudio_value"] for r in subset], dtype=float)
            errors = np.asarray([r["full_relative_error_pct"] for r in subset], dtype=float)
            abs_errors = np.asarray([r["full_absolute_error"] for r in subset], dtype=float)
        else:
            values = np.asarray([r["selected_hrvstudio_value"] for r in subset], dtype=float)
            errors = np.asarray([r["selected_relative_error_pct"] for r in subset], dtype=float)
            abs_errors = np.asarray([r["selected_absolute_error"] for r in subset], dtype=float)
        kubios = np.asarray([r["kubios_value"] for r in subset], dtype=float)
        mask = np.isfinite(values) & np.isfinite(kubios)
        error_mask = np.isfinite(errors)
        r_value = pearsonr(values[mask], kubios[mask]).statistic if mask.sum() >= 2 else math.nan
        output.append(
            {
                "condition": condition,
                "Metric": metric,
                "n": int(mask.sum()),
                "median_relative_error_pct": float(np.nanmedian(errors[error_mask])),
                "mean_relative_error_pct": float(np.nanmean(errors[error_mask])),
                "pearson_r": float(r_value),
                "median_absolute_error": float(np.nanmedian(abs_errors[np.isfinite(abs_errors)])),
                "mean_absolute_error": float(np.nanmean(abs_errors[np.isfinite(abs_errors)])),
            }
        )
    return output


def metric_summary_for_selected(rows: list[dict]) -> list[dict]:
    output = []
    for metric in METRICS:
        subset = [r for r in rows if r["Metric"] == metric]
        values = np.asarray([r["selected_hrvstudio_value"] for r in subset], dtype=float)
        kubios = np.asarray([r["kubios_value"] for r in subset], dtype=float)
        errors = np.asarray([r["selected_relative_error_pct"] for r in subset], dtype=float)
        abs_errors = np.asarray([r["selected_absolute_error"] for r in subset], dtype=float)
        mask = np.isfinite(values) & np.isfinite(kubios)
        error_mask = np.isfinite(errors)
        r_value = pearsonr(values[mask], kubios[mask]).statistic if mask.sum() >= 2 else math.nan
        output.append(
            {
                "Metric": metric,
                "n_files": int(mask.sum()),
                "n_rows": int(mask.sum()),
                "mean_relative_error_pct": float(np.nanmean(errors[error_mask])),
                "median_relative_error_pct": float(np.nanmedian(errors[error_mask])),
                "pearson_correlation": float(r_value),
                "mae": float(np.nanmean(abs_errors[np.isfinite(abs_errors)])),
            }
        )
    return output


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(COHORT)
    cohort = cohort[cohort["metric"].isin(METRICS)].copy()
    recordings = (
        cohort[["subset_id", "category", "source_file"]]
        .drop_duplicates()
        .sort_values("subset_id")
        .to_dict("records")
    )
    if len(recordings) != 44:
        raise RuntimeError(f"Expected 44 recordings, found {len(recordings)}")

    original_values = {
        (row.subset_id, row.metric): row.hrvstudio_native_value
        for row in cohort.itertuples(index=False)
    }
    original_summary = pd.read_csv(ORIGINAL_SUMMARY).set_index("metric")
    run_info = json.loads(RUN_INFO.read_text(encoding="utf-8"))
    settings = run_info["hrv_freq_domain_settings"]

    audit_rows = []
    per_rows = []
    original_recompute_diffs = []
    missing = []

    for rec in recordings:
        rid = rec["subset_id"]
        try:
            txt_path, mat_path = export_paths(rid, rec["category"])
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue

        txt = parse_kubios_txt(txt_path)
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        res = mat["Res"]
        hrv = get_field(res, "HRV")
        data = get_field(hrv, "Data")
        frequency = get_field(hrv, "Frequency")
        welch = get_field(frequency, "Welch")

        full_rr_ms = safe_array(get_field(data, "RR")) * 1000.0
        selected_rr_ms = safe_array(get_field(data, "RRs")) * 1000.0
        rri = safe_array(get_field(data, "RRi"))
        freq_f = safe_array(get_field(welch, "F"))
        freq_psd = safe_array(get_field(welch, "PSD"))
        mat_welch = {
            metric: safe_float(get_field(welch, field))
            for metric, field in KUBIOS_MAT_KEYS.items()
        }
        for metric in KUBIOS_MAT_POWER_METRICS:
            mat_welch[metric] *= 1_000_000.0

        full_n = len(full_rr_ms)
        selected_n = len(selected_rr_ms)
        full_duration = float(np.sum(full_rr_ms) / 1000.0)
        selected_duration = float(np.sum(selected_rr_ms) / 1000.0)
        prefix_match = (
            selected_n <= full_n
            and np.allclose(full_rr_ms[:selected_n], selected_rr_ms, rtol=0, atol=1e-6)
        )
        if selected_n == full_n and prefix_match:
            selection_type = "full_sequence"
        elif prefix_match:
            selection_type = "shorter_prefix"
        else:
            selection_type = "non_prefix_or_ambiguous"

        txt_mat_diffs = [
            abs(txt["metrics"][metric] - mat_welch[metric])
            for metric in METRICS
            if np.isfinite(txt["metrics"][metric]) and np.isfinite(mat_welch[metric])
        ]
        mat_txt_max_diff = max(txt_mat_diffs) if txt_mat_diffs else math.nan
        selected_interp_delta = abs(len(rri) - selected_duration * settings["sampling_rate"])
        full_interp_delta = abs(len(rri) - full_duration * settings["sampling_rate"])

        selected_start_index = 0 if prefix_match else ""
        selected_end_index = selected_n - 1 if prefix_match else ""
        if (
            np.isfinite(mat_txt_max_diff)
            and mat_txt_max_diff <= 1e-3
            and selection_type == "full_sequence"
        ):
            inference = "selected_RRs_confirmed"
            confidence = "high"
        elif np.isfinite(mat_txt_max_diff) and mat_txt_max_diff <= 1e-3 and selected_interp_delta <= 2.5:
            inference = "selected_RRs_confirmed"
            confidence = "high"
        elif (
            np.isfinite(mat_txt_max_diff)
            and mat_txt_max_diff <= 1e-3
            and selected_interp_delta < full_interp_delta
        ):
            inference = "selected_RRs_strongly_inferred"
            confidence = "moderate"
        elif (
            np.isfinite(mat_txt_max_diff)
            and mat_txt_max_diff <= 1e-3
            and full_interp_delta + 1e-9 < selected_interp_delta
        ):
            inference = "full_RR_confirmed"
            confidence = "moderate"
        else:
            inference = "unresolved"
            confidence = "low"

        evidence = (
            f"TXT FFT values match Res.HRV.Frequency.Welch max_abs_diff={fmt(mat_txt_max_diff, 6)}; "
            f"len(HRV.Data.RRi)={len(rri)} vs selected_duration*{settings['sampling_rate']}="
            f"{fmt(selected_duration * settings['sampling_rate'], 2)} and full_duration*"
            f"{settings['sampling_rate']}={fmt(full_duration * settings['sampling_rate'], 2)}; "
            f"RRs prefix_match={prefix_match}; F_len={len(freq_f)} PSD_len={len(freq_psd)}"
        )

        notes = ""
        if txt["sample_limits"]:
            notes = f"TXT sample_limits={txt['sample_limits']}"
        if selection_type == "full_sequence":
            notes = (notes + "; " if notes else "") + "HRV.Data.RRs equals HRV.Data.RR"

        audit_rows.append(
            {
                "pilot_id": rid,
                "source_recording": rec["source_file"],
                "full_rr_n": full_n,
                "kubios_selected_rr_n": selected_n,
                "full_duration_s": full_duration,
                "kubios_selected_duration_s": selected_duration,
                "selected_start_index": selected_start_index,
                "selected_end_index": selected_end_index,
                "selection_type": selection_type,
                "frequency_sequence_inference": inference,
                "evidence": evidence,
                "confidence": confidence,
                "notes": notes,
            }
        )

        full_metrics, full_diag = run_hrvstudio_frequency(full_rr_ms, settings)
        selected_metrics, selected_diag = run_hrvstudio_frequency(selected_rr_ms, settings)
        for metric in METRICS:
            expected_original = original_values[(rid, metric)]
            original_recompute_diffs.append(abs(full_metrics[metric] - expected_original))
            original_full_diff = abs(full_metrics[metric] - expected_original)
            original_selected_diff = abs(selected_metrics[metric] - expected_original)
            kubios_value = txt["metrics"][metric]
            full_value = full_metrics[metric]
            selected_value = selected_metrics[metric]
            full_abs = abs(full_value - kubios_value)
            selected_abs = abs(selected_value - kubios_value)
            full_re = rel_error_pct(full_value, kubios_value)
            selected_re = rel_error_pct(selected_value, kubios_value)
            per_rows.append(
                {
                    "pilot_id": rid,
                    "source_recording": rec["source_file"],
                    "Metric": metric,
                    "kubios_value": kubios_value,
                    "original_hrvstudio_value_from_benchmark": expected_original,
                    "full_hrvstudio_value": full_value,
                    "selected_hrvstudio_value": selected_value,
                    "original_value_abs_diff_vs_current_full": original_full_diff,
                    "original_value_abs_diff_vs_current_selected": original_selected_diff,
                    "original_value_closer_to": (
                        "current_full"
                        if original_full_diff < original_selected_diff
                        else "current_selected"
                        if original_selected_diff < original_full_diff
                        else "tie"
                    ),
                    "full_absolute_error": full_abs,
                    "selected_absolute_error": selected_abs,
                    "absolute_error_change_selected_minus_full": selected_abs - full_abs,
                    "full_relative_error_pct": full_re,
                    "selected_relative_error_pct": selected_re,
                    "relative_error_change_selected_minus_full": selected_re - full_re,
                    "full_rr_n": full_n,
                    "selected_rr_n": selected_n,
                    "full_duration_s": full_duration,
                    "selected_duration_s": selected_duration,
                    "selection_type": selection_type,
                    "frequency_sequence_inference": inference,
                    "full_welch_nperseg": full_diag.get("effective_nperseg"),
                    "full_welch_noverlap": full_diag.get("effective_noverlap"),
                    "full_welch_n_segments": full_diag.get("number_of_segments"),
                    "selected_welch_nperseg": selected_diag.get("effective_nperseg"),
                    "selected_welch_noverlap": selected_diag.get("effective_noverlap"),
                    "selected_welch_n_segments": selected_diag.get("number_of_segments"),
                    "kubios_txt_path": str(txt_path.relative_to(ROOT)),
                    "kubios_mat_path": str(mat_path.relative_to(ROOT)),
                }
            )

    if missing:
        raise RuntimeError("\n".join(missing))

    audit_fields = [
        "pilot_id",
        "source_recording",
        "full_rr_n",
        "kubios_selected_rr_n",
        "full_duration_s",
        "kubios_selected_duration_s",
        "selected_start_index",
        "selected_end_index",
        "selection_type",
        "frequency_sequence_inference",
        "evidence",
        "confidence",
        "notes",
    ]
    write_csv(AUDIT_CSV, audit_rows, audit_fields)
    write_csv(FULL_VS_SELECTED_PER_RECORDING, per_rows)

    full_summary = summarize_errors(per_rows, "full")
    selected_summary = summarize_errors(per_rows, "selected")
    write_csv(FULL_VS_SELECTED_SUMMARY, full_summary + selected_summary)

    matched_summary = metric_summary_for_selected(per_rows)
    write_csv(MATCHED_SUMMARY, matched_summary)
    matched_per_fields = [
        "pilot_id",
        "source_recording",
        "Metric",
        "kubios_value",
        "selected_hrvstudio_value",
        "selected_absolute_error",
        "selected_relative_error_pct",
        "selected_rr_n",
        "selected_duration_s",
        "selection_type",
        "frequency_sequence_inference",
        "selected_welch_nperseg",
        "selected_welch_noverlap",
        "selected_welch_n_segments",
        "kubios_txt_path",
        "kubios_mat_path",
    ]
    write_csv(MATCHED_PER_RECORDING, per_rows, matched_per_fields)

    matched_by_metric = {row["Metric"]: row for row in matched_summary}
    original_vs_rows = []
    for metric in METRICS:
        original = original_summary.loc[metric]
        matched = matched_by_metric[metric]
        original_median = float(original["median_relative_error_pct"])
        matched_median = float(matched["median_relative_error_pct"])
        original_mean = float(original["mean_relative_error_pct"])
        matched_mean = float(matched["mean_relative_error_pct"])
        original_r = float(original["pearson_correlation"])
        matched_r = float(matched["pearson_correlation"])
        original_vs_rows.append(
            {
                "Metric": metric,
                "Original median RE %": original_median,
                "Matched-sequence median RE %": matched_median,
                "Absolute change in median RE": matched_median - original_median,
                "Relative improvement %": (
                    (original_median - matched_median) / original_median * 100.0
                    if original_median
                    else math.nan
                ),
                "Original mean RE %": original_mean,
                "Matched-sequence mean RE %": matched_mean,
                "Original Pearson r": original_r,
                "Matched-sequence Pearson r": matched_r,
            }
        )
    write_csv(ORIGINAL_VS_MATCHED, original_vs_rows)

    inference_counts = pd.Series([r["frequency_sequence_inference"] for r in audit_rows]).value_counts()
    selection_counts = pd.Series([r["selection_type"] for r in audit_rows]).value_counts()
    established = sum(
        r["frequency_sequence_inference"]
        in {"selected_RRs_confirmed", "selected_RRs_strongly_inferred"}
        for r in audit_rows
    )
    improved = sum(row["Absolute change in median RE"] < 0 for row in original_vs_rows)
    worsened = sum(row["Absolute change in median RE"] > 0 for row in original_vs_rows)
    max_recompute_diff = max(original_recompute_diffs) if original_recompute_diffs else math.nan
    largest_changes = sorted(
        per_rows,
        key=lambda row: abs(row["relative_error_change_selected_minus_full"])
        if np.isfinite(row["relative_error_change_selected_minus_full"])
        else -1,
        reverse=True,
    )[:12]
    per_df = pd.DataFrame(per_rows)
    recording_recompute = (
        per_df.groupby("pilot_id")["original_value_abs_diff_vs_current_full"].max().reset_index()
    )
    original_full_reproduced_recordings = int(
        (recording_recompute["original_value_abs_diff_vs_current_full"] <= 1e-6).sum()
    )
    original_full_not_reproduced_recordings = int(
        len(recording_recompute) - original_full_reproduced_recordings
    )
    original_closer_counts = (
        per_df["original_value_closer_to"].value_counts().rename_axis("closer_to").reset_index(name="n")
    )

    settings_lines = [
        f"- interpolation frequency: {settings['sampling_rate']} Hz",
        f"- detrending method: {settings.get('detrend_method')}",
        "- PSD estimator: Welch / FFT comparator branch",
        f"- window length: {settings['segment_length']} s requested",
        f"- overlap: {settings['overlap_ratio']}",
        f"- window function: {settings['window_type']}",
        "- band convention: Kubios-compatible comparator mode",
        "- frequency bands: VLF 0.00-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz, total 0.00-0.40 Hz",
        "- DC handling: unchanged HRV Studio comparator behavior; VLF band begins at 0.00 Hz",
        "- PSD integration method: unchanged HRV Studio comparator behavior (`np.trapezoid`)",
        f"- native Welch nfft multiplier: {settings.get('experimental_native_welch_nfft_multiplier')}",
        f"- Welch detrend mode: {settings.get('welch_detrend_mode')}",
    ]

    summary_table = markdown_table(original_vs_rows)
    full_selected_table = markdown_table(full_summary + selected_summary)
    inference_table = markdown_table(
        inference_counts.rename_axis("inference").reset_index(name="n").to_dict("records")
    )
    selection_table = markdown_table(
        selection_counts.rename_axis("selection_type").reset_index(name="n").to_dict("records")
    )
    original_closer_table = markdown_table(original_closer_counts.to_dict("records"))
    largest_table = pd.DataFrame(
        [
            {
                "pilot_id": row["pilot_id"],
                "Metric": row["Metric"],
                "full_RE": row["full_relative_error_pct"],
                "selected_RE": row["selected_relative_error_pct"],
                "change": row["relative_error_change_selected_minus_full"],
                "selection_type": row["selection_type"],
            }
            for row in largest_changes
        ]
    ).to_dict("records")
    largest_table = markdown_table(largest_table)

    lfhf_row = next(row for row in original_vs_rows if row["Metric"] == "LF/HF")
    vlf_row = next(row for row in original_vs_rows if row["Metric"] == "VLF")
    normalized = [row for row in original_vs_rows if row["Metric"] in {"LF_nu", "HF_nu"}]
    normalized_remain_strong = all(
        row["Matched-sequence median RE %"] <= 15.0 for row in normalized
    )
    if improved >= 5 and vlf_row["Relative improvement %"] > 20:
        interpretation = (
            "Sequence mismatch was a major contributor for some absolute-power metrics, "
            "but residual spectral-processing differences remain."
        )
    elif improved >= 3:
        interpretation = (
            "Sequence mismatch contributed partially, but substantial estimator/preprocessing "
            "differences remain."
        )
    elif abs(lfhf_row["Absolute change in median RE"]) < 2 and abs(vlf_row["Absolute change in median RE"]) < 5:
        interpretation = (
            "Sequence matching produced little improvement, supporting the interpretation "
            "that residual disagreement primarily arises from spectral-processing conventions."
        )
    else:
        interpretation = "The evidence is mixed; sequence matching did not uniformly improve metrics."

    audit_md = f"""# Kubios Frequency-Domain Sequence Alignment Audit

## Scope

This audit used the 44 recordings in `cleaned_valid_only.csv` and preserved the original Section 3.2 comparator settings from `run_info.json`. No manuscript files or original benchmark outputs were modified.

## Phase 1: Export Structure

Kubios frequency metrics are stored in `Res.HRV.Frequency.Welch` in each MAT export. The FFT-column values printed in each Kubios TXT export match those MAT values within text precision. The same MAT export contains `Res.HRV.Data.RR` (complete imported vector), `Res.HRV.Data.RRs` (selected sample), and `Res.HRV.Data.RRi` (interpolated selected RR series).

Inference counts:

{inference_table}

Selection counts:

{selection_table}

**A. Can we establish that Kubios frequency-domain metrics were calculated from `HRV.Data.RRs`?**

Yes for this export set. Evidence is strongest for the 31 recordings where `HRV.Data.RRs` is a shorter prefix of `HRV.Data.RR`: `HRV.Data.RRi` length agrees with the selected-sequence duration at 4 Hz and not with the full imported duration, while printed FFT metrics match `Res.HRV.Frequency.Welch`. For the remaining 13 recordings, `HRV.Data.RRs` equals `HRV.Data.RR`, so selected and full sequences are indistinguishable.

**B. If yes, for how many of the 44 recordings?**

Established or directly indistinguishable for {established}/44 recordings. In {int(selection_counts.get('shorter_prefix', 0))}/44 recordings the selected sequence is shorter than the full input; in {int(selection_counts.get('full_sequence', 0))}/44, selected equals full.

**C. Evidence supporting or contradicting this interpretation**

Supporting evidence:

- Printed Kubios FFT metrics match `Res.HRV.Frequency.Welch` values in the MAT exports.
- `Res.HRV.Data.RRi` has sample counts consistent with `sum(HRV.Data.RRs) * 4 Hz`.
- In shorter-prefix recordings, `Res.HRV.Data.RRi` is inconsistent with full-vector duration.
- `HRV.Data.RRs` is a prefix of `HRV.Data.RR`, matching the previously established time-domain selected-sample behavior.

Contradicting evidence: none found in the exported fields inspected. The exports do not include a separate explicit flag that says "frequency metrics calculated from HRV.Data.RRs"; the conclusion is based on MAT structure plus numerical/timing checks.

**D. Does the original HRV Studio frequency-domain benchmark use the full input sequence or the Kubios-selected sequence?**

The archived original benchmark values are not uniformly reproducible from the current Kubios MAT `HRV.Data.RR` vectors. Exact full-vector recomputation reproduced all metrics for {original_full_reproduced_recordings}/44 recordings and did not reproduce {original_full_not_reproduced_recordings}/44 recordings; the maximum absolute difference was `{fmt(max_recompute_diff, 9)}`. At least one current input file/export pair has 600 s in the MAT export while the archived validation row records a much shorter effective RR count/duration. Therefore this audit treats the original manuscript-facing values as archived provenance and does not claim that every archived HRV Studio row used the current `HRV.Data.RR` vector.

Original archived values were closer to these current recomputations by metric-row count:

{original_closer_table}

**E. Was the original comparison sequence-aligned?**

No for the 31 shorter-prefix Kubios exports if the comparator is the current complete imported `HRV.Data.RR` sequence. Additionally, the provenance mismatch above means some archived original rows cannot be cleanly re-derived from the current full vectors, so the original comparison should not be interpreted as exact sequence-aligned evidence.

## Phase 2: Numerical Cross-Check

Settings recovered from the original run:

{chr(10).join(settings_lines)}

Full-vs-selected comparison:

{full_selected_table}

Old-vs-new benchmark summary:

{summary_table}

Largest per-recording relative-error changes:

{largest_table}

## Phase 5 Interpretation

{interpretation}

Metrics improved by median relative error: {improved}/7. Metrics worsened: {worsened}/7.

LF/HF changed from `{fmt(lfhf_row['Original median RE %'])}%` to `{fmt(lfhf_row['Matched-sequence median RE %'])}%` median relative error. VLF changed from `{fmt(vlf_row['Original median RE %'])}%` to `{fmt(vlf_row['Matched-sequence median RE %'])}%`.

Normalized metrics remain among the strongest frequency-domain metrics after matching: {normalized_remain_strong}.

Remaining discrepancies after exact sequence matching should be interpreted as spectral-processing/convention differences under the preserved comparator configuration, not sequence-selection differences.

## Manuscript Consequences

The manuscript's existing Section 3.2 numbers, Figure 4B-D, Abstract Kubios statements, Discussion interpretation, and Limitations wording would need review before relying on the current frequency-domain Kubios benchmark. This audit does not edit those manuscript components.

## Commands Used

```powershell
python tools/frequency_domain_sequence_audit.py
```

## Outputs

- `{AUDIT_CSV.relative_to(ROOT)}`
- `{FULL_VS_SELECTED_SUMMARY.relative_to(ROOT)}`
- `{FULL_VS_SELECTED_PER_RECORDING.relative_to(ROOT)}`
- `{MATCHED_SUMMARY.relative_to(ROOT)}`
- `{MATCHED_PER_RECORDING.relative_to(ROOT)}`
- `{ORIGINAL_VS_MATCHED.relative_to(ROOT)}`
- `{AUDIT_REPORT.relative_to(ROOT)}`
- `{MATCHED_REPORT.relative_to(ROOT)}`
"""
    AUDIT_REPORT.write_text(audit_md, encoding="utf-8")

    matched_md = f"""# Matched-Sequence Kubios Frequency-Domain Validation

## Scope

This rerun used the exact Kubios-selected `HRV.Data.RRs` interval sequences for the 44-recording benchmark while preserving the original Section 3.2 spectral comparator configuration.

## Settings

{chr(10).join(settings_lines)}

## Summary

{summary_table}

## Interpretation

{interpretation}

This is a sequence-alignment correction only. Detrending, interpolation, Welch/FFT settings, band definitions, and integration behavior were kept unchanged from the original benchmark configuration.

## Outputs

- `{MATCHED_SUMMARY.relative_to(ROOT)}`
- `{MATCHED_PER_RECORDING.relative_to(ROOT)}`
- `{ORIGINAL_VS_MATCHED.relative_to(ROOT)}`
"""
    MATCHED_REPORT.write_text(matched_md, encoding="utf-8")

    print(f"Audit rows: {len(audit_rows)}")
    print(f"Per-recording metric rows: {len(per_rows)}")
    print(f"Frequency sequence established: {established}/44")
    print(f"Median RE metrics improved: {improved}/7; worsened: {worsened}/7")
    print(f"Max original full-sequence recompute diff: {fmt(max_recompute_diff, 9)}")
    print("Wrote:")
    for path in [
        AUDIT_CSV,
        FULL_VS_SELECTED_SUMMARY,
        FULL_VS_SELECTED_PER_RECORDING,
        MATCHED_SUMMARY,
        MATCHED_PER_RECORDING,
        ORIGINAL_VS_MATCHED,
        AUDIT_REPORT,
        MATCHED_REPORT,
    ]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
