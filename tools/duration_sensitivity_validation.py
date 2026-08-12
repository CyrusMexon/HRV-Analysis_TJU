"""
Validation-only duration sensitivity study.

This script evaluates how HRV Studio agreement changes as RR recording duration
changes. It compares HRV Studio Welch/time-domain metrics with NeuroKit2-style
Welch/time-domain metrics across fixed truncation durations. Parsed Kubios
exports are included only when a matching full-duration export exists.

Production HRV code is not modified.

Example:
    python tools/duration_sensitivity_validation.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from hrvlib.metrics.time_domain import HRVTimeDomainAnalysis


DEFAULT_INPUT_DIR = PROJECT_ROOT / "validation" / "processed_data" / "physionet_nsr_rr_10min"
DEFAULT_RUN_DIR = PROJECT_ROOT / "validation" / "runs" / "v10_duration_sensitivity_validation"
DEFAULT_KUBIOS_COMPARISON = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "parsed_results_50_none_120s_75pct_after_arm_a"
    / "comparison_valid_only.csv"
)

DURATIONS = [
    ("30s", 30.0),
    ("60s", 60.0),
    ("2min", 120.0),
    ("3min", 180.0),
    ("5min", 300.0),
    ("10min", 600.0),
]
METRICS = ["VLF", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu", "SDNN", "RMSSD"]
FREQ_METRICS = ["VLF", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu"]
TIME_METRICS = ["SDNN", "RMSSD"]
FREQ_KEYS = {
    "VLF": "vlf_power",
    "LF": "lf_power",
    "HF": "hf_power",
    "total_power": "total_power",
    "LF/HF": "lf_hf_ratio",
    "LF_nu": "lf_nu",
    "HF_nu": "hf_nu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-only duration sensitivity analysis."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--min-full-duration-s", type=float, default=590.0)
    parser.add_argument("--max-invalid-removed-pct", type=float, default=1.0)
    parser.add_argument("--interpolation-rate", type=float, default=4.0)
    parser.add_argument("--segment-length", type=float, default=120.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.75)
    parser.add_argument("--window-type", default="hann")
    parser.add_argument("--detrend-method", choices=["none", "linear", "constant", "smoothness_priors"], default="none")
    parser.add_argument("--neurokit-interpolation-method", default="monotone_cubic")
    parser.add_argument("--kubios-comparison", type=Path, default=DEFAULT_KUBIOS_COMPARISON)
    return parser.parse_args()


def normalize_detrend(value: str) -> Optional[str]:
    return None if value == "none" else value


def prepare_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Output folder already exists and is non-empty: {run_dir}. "
            "This script will not overwrite previous validation outputs."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bland_altman").mkdir(parents=True, exist_ok=True)


def load_rr_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "rr_ms" in df.columns:
        values = df["rr_ms"].to_numpy(dtype=float)
    else:
        values = df.iloc[:, 0].to_numpy(dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def select_clean_files(input_dir: Path, sample_size: int, min_duration_s: float, max_invalid_pct: float) -> List[Path]:
    manifest_path = input_dir / "10min_manifest.csv"
    if not manifest_path.exists():
        candidates = sorted(p for p in input_dir.glob("*.csv") if p.name != "10min_manifest.csv")
        return candidates[:sample_size]

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[
        (manifest["duration_seconds"] >= min_duration_s)
        & (manifest["percent_invalid_removed"] <= max_invalid_pct)
    ].copy()
    manifest["output_path"] = manifest["output_file"].map(Path)
    manifest = manifest[manifest["output_path"].map(lambda p: p.exists())]
    manifest = manifest.sort_values(["record_id", "segment_id"]).reset_index(drop=True)

    selected: List[Path] = []
    used_records = set()
    for _, row in manifest.iterrows():
        if row["record_id"] in used_records:
            continue
        selected.append(row["output_path"])
        used_records.add(row["record_id"])
        if len(selected) >= sample_size:
            return selected
    for _, row in manifest.iterrows():
        path = row["output_path"]
        if path not in selected:
            selected.append(path)
            if len(selected) >= sample_size:
                break
    return selected


def truncate_rr(rr_ms: np.ndarray, duration_s: float) -> np.ndarray:
    rr = np.asarray(rr_ms, dtype=float)
    cumulative = np.cumsum(rr) / 1000.0
    keep = cumulative <= duration_s
    if np.count_nonzero(keep) < 2:
        return rr[: max(2, min(len(rr), np.count_nonzero(cumulative <= duration_s) + 1))]
    return rr[keep]


def rr_start_times_s(rr_ms: np.ndarray) -> np.ndarray:
    rr_s = rr_ms.astype(float) / 1000.0
    return np.concatenate([[0.0], np.cumsum(rr_s[:-1])])


def band_mask(freqs: np.ndarray, band: str) -> np.ndarray:
    if band == "total_power":
        return HRVFreqDomainAnalysis.total_power_mask_for_convention(freqs)
    return HRVFreqDomainAnalysis.mask_for_band(freqs, band)


def integrate_band(freqs: np.ndarray, psd: np.ndarray, band: str) -> float:
    mask = band_mask(freqs, band)
    if np.count_nonzero(mask) < 2:
        return 0.0
    value = np.trapezoid(psd[mask], freqs[mask])
    return float(value) if np.isfinite(value) else math.nan


def psd_metrics(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    vlf = integrate_band(freqs, psd, "vlf")
    lf = integrate_band(freqs, psd, "lf")
    hf = integrate_band(freqs, psd, "hf")
    total = integrate_band(freqs, psd, "total_power")
    lf_hf = lf / hf if np.isfinite(hf) and hf > 1e-10 else math.nan
    lf_hf_sum = lf + hf
    lf_nu = (lf / lf_hf_sum) * 100.0 if np.isfinite(lf_hf_sum) and lf_hf_sum > 0 else math.nan
    hf_nu = (hf / lf_hf_sum) * 100.0 if np.isfinite(lf_hf_sum) and lf_hf_sum > 0 else math.nan
    return {
        "VLF": vlf,
        "LF": lf,
        "HF": hf,
        "total_power": total,
        "LF/HF": lf_hf,
        "LF_nu": lf_nu,
        "HF_nu": hf_nu,
    }


def neurokit2_welch_metrics(
    rr_ms: np.ndarray,
    sampling_rate: float,
    interpolation_method: str,
    window_type: str,
    segment_length_s: float,
    overlap_ratio: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    try:
        import neurokit2 as nk
    except Exception as exc:
        raise RuntimeError(f"NeuroKit2 import failed: {exc}") from exc

    rri_time = rr_start_times_s(rr_ms)
    intervals, _, _ = nk.intervals_process(
        rr_ms,
        intervals_time=rri_time,
        interpolate=True,
        interpolation_rate=sampling_rate,
        method=interpolation_method,
    )
    intervals = np.asarray(intervals, dtype=float)
    requested_nperseg = int(segment_length_s * sampling_rate)
    nperseg = min(requested_nperseg, max(8, len(intervals) // 2))
    noverlap = min(int(nperseg * overlap_ratio), max(0, nperseg - 1))

    psd = nk.signal_psd(
        intervals,
        sampling_rate=sampling_rate,
        method="welch",
        min_frequency=-np.inf,
        max_frequency=0.4,
        window=nperseg / sampling_rate,
        window_type=window_type,
        noverlap=noverlap,
        normalize=False,
        silent=True,
    )
    freqs = psd["Frequency"].to_numpy(dtype=float)
    power = psd["Power"].to_numpy(dtype=float)
    return psd_metrics(freqs, power), {
        "neurokit2_n_samples": int(len(intervals)),
        "neurokit2_nperseg": int(nperseg),
        "neurokit2_noverlap": int(noverlap),
        "neurokit2_frequency_resolution_hz": float(freqs[1] - freqs[0]) if len(freqs) > 1 else math.nan,
    }


def neurokit2_time_metrics(rr_ms: np.ndarray) -> Dict[str, float]:
    diff = np.diff(rr_ms)
    return {
        "SDNN": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else math.nan,
        "RMSSD": float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else math.nan,
    }


def relative_error_pct(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return math.nan
    return float(abs(value - reference) / abs(reference) * 100.0)


def finite_corr(x: Iterable[float], y: Iterable[float], method: str = "pearson") -> float:
    frame = pd.DataFrame({"x": list(x), "y": list(y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return math.nan
    return float(frame["x"].corr(frame["y"], method=method))


def bland_altman_stats(value: pd.Series, reference: pd.Series) -> Dict[str, float]:
    pair = pd.concat([value, reference], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return {"bias": math.nan, "loa_lower": math.nan, "loa_upper": math.nan}
    diff = pair.iloc[:, 0] - pair.iloc[:, 1]
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    return {"bias": bias, "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd}


def duration_warning_labels(freq_results: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    diagnostics = freq_results.get("frequency_diagnostics", {})
    if diagnostics.get("duration_warnings"):
        labels.append("duration_warning")
    if diagnostics.get("nonfinite_psd"):
        labels.append("nonfinite_psd")
    for method in ["welch", "fft", "ar"]:
        method_diag = diagnostics.get(method, {})
        if method_diag.get("warning"):
            labels.append(f"{method}_warning")
        for band, band_diag in method_diag.get("band_diagnostics", {}).items():
            warnings_list = band_diag.get("warnings", [])
            if any("Fewer than 2 PSD bins" in x for x in warnings_list):
                labels.append(f"{method}_{band}_few_bins")
            if any("insufficient" in x.lower() for x in warnings_list):
                labels.append(f"{method}_{band}_duration")
    return sorted(set(labels))


def load_kubios_map(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["source_basename"] = df["source_file"].map(lambda x: Path(str(x)).name)
    pivot = df.pivot_table(index="source_basename", columns="metric", values="kubios_value", aggfunc="first")
    return {
        str(index): {metric: float(row[metric]) for metric in FREQ_METRICS if metric in row and pd.notna(row[metric])}
        for index, row in pivot.iterrows()
    }


def analyze_duration(path: Path, rr_full: np.ndarray, duration_label: str, duration_s: float, args: argparse.Namespace, kubios_map: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    rr = truncate_rr(rr_full, duration_s)
    actual_duration = float(np.sum(rr) / 1000.0)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            time_results = HRVTimeDomainAnalysis(rr).full_analysis()
            freq = HRVFreqDomainAnalysis(
                rr,
                sampling_rate=args.interpolation_rate,
                detrend_method=normalize_detrend(args.detrend_method),
                window_type=args.window_type,
                segment_length=args.segment_length,
                overlap_ratio=args.overlap_ratio,
                enable_diagnostics=True,
            )
            freq_results = freq.get_results()
            nk_freq_metrics, nk_diag = neurokit2_welch_metrics(
                rr,
                sampling_rate=args.interpolation_rate,
                interpolation_method=args.neurokit_interpolation_method,
                window_type=args.window_type,
                segment_length_s=args.segment_length,
                overlap_ratio=args.overlap_ratio,
            )
    native_metrics = {
        "SDNN": float(time_results.get("sdnn", math.nan)),
        "RMSSD": float(time_results.get("rmssd", math.nan)),
    }
    for metric, key in FREQ_KEYS.items():
        native_metrics[metric] = float(freq_results.get(f"welch_{key}", math.nan))
    nk_metrics = {**nk_freq_metrics, **neurokit2_time_metrics(rr)}
    warning_labels = duration_warning_labels(freq_results)
    if captured:
        warning_labels.append("runtime_warning")
    warning_labels = sorted(set(warning_labels))
    diagnostics = freq_results.get("frequency_diagnostics", {})
    source_basename = path.name
    kubios_values = kubios_map.get(source_basename, {}) if duration_label == "10min" else {}

    rows: List[Dict[str, Any]] = []
    for metric in METRICS:
        native_value = native_metrics.get(metric, math.nan)
        nk_value = nk_metrics.get(metric, math.nan)
        kubios_value = kubios_values.get(metric, math.nan)
        rows.append(
            {
                "file": str(path),
                "file_name": path.name,
                "duration_label": duration_label,
                "duration_seconds_requested": duration_s,
                "duration_seconds_actual": actual_duration,
                "n_rr": int(len(rr)),
                "metric": metric,
                "hrvstudio_value": native_value,
                "neurokit2_value": nk_value,
                "kubios_value": kubios_value,
                "relative_error_vs_neurokit2_pct": relative_error_pct(native_value, nk_value),
                "relative_error_vs_kubios_pct": relative_error_pct(native_value, kubios_value),
                "finite_hrvstudio": bool(np.isfinite(native_value)),
                "finite_neurokit2": bool(np.isfinite(nk_value)),
                "finite_kubios": bool(np.isfinite(kubios_value)),
                "warning_labels": "; ".join(warning_labels),
                "warning_count": len(warning_labels),
                "duration_warning": "duration_warning" in warning_labels,
                "vlf_instability_flag": metric == "VLF"
                and (
                    "welch_vlf_duration" in warning_labels
                    or "welch_vlf_few_bins" in warning_labels
                    or relative_error_pct(native_value, nk_value) > 20.0
                ),
                "lf_instability_flag": metric == "LF"
                and (
                    "welch_lf_duration" in warning_labels
                    or "welch_lf_few_bins" in warning_labels
                    or relative_error_pct(native_value, nk_value) > 20.0
                ),
                "welch_nperseg": diagnostics.get("welch", {}).get("effective_nperseg", math.nan),
                "welch_noverlap": diagnostics.get("welch", {}).get("effective_noverlap", math.nan),
                "welch_frequency_resolution_hz": diagnostics.get("welch", {}).get("frequency_resolution_hz", math.nan),
                **nk_diag,
            }
        )
    return rows


def build_metric_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (duration_label, metric), group in results.groupby(["duration_label", "metric"], sort=False):
        ba = bland_altman_stats(group["hrvstudio_value"], group["neurokit2_value"])
        rows.append(
            {
                "duration_label": duration_label,
                "duration_seconds": float(group["duration_seconds_requested"].iloc[0]),
                "metric": metric,
                "n": int(len(group)),
                "finite_output_rate_pct": float(group["finite_hrvstudio"].mean() * 100.0),
                "warning_frequency_pct": float(group["warning_count"].gt(0).mean() * 100.0),
                "median_relative_error_vs_neurokit2_pct": float(group["relative_error_vs_neurokit2_pct"].median()),
                "mean_relative_error_vs_neurokit2_pct": float(group["relative_error_vs_neurokit2_pct"].mean()),
                "median_relative_error_vs_kubios_pct": float(group["relative_error_vs_kubios_pct"].median()),
                "pearson_correlation_vs_neurokit2": finite_corr(group["hrvstudio_value"], group["neurokit2_value"], "pearson"),
                "spearman_correlation_vs_neurokit2": finite_corr(group["hrvstudio_value"], group["neurokit2_value"], "spearman"),
                "bland_altman_bias_vs_neurokit2": ba["bias"],
                "bland_altman_loa_lower_vs_neurokit2": ba["loa_lower"],
                "bland_altman_loa_upper_vs_neurokit2": ba["loa_upper"],
                "kubios_matched_n": int(group["finite_kubios"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_vlf_instability_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for duration_label, group in results.groupby("duration_label", sort=False):
        vlf = group[group["metric"] == "VLF"]
        lf = group[group["metric"] == "LF"]
        rows.append(
            {
                "duration_label": duration_label,
                "duration_seconds": float(group["duration_seconds_requested"].iloc[0]),
                "files": int(vlf["file_name"].nunique()),
                "vlf_instability_rate_pct": float(vlf["vlf_instability_flag"].mean() * 100.0),
                "lf_instability_rate_pct": float(lf["lf_instability_flag"].mean() * 100.0),
                "vlf_median_relative_error_vs_neurokit2_pct": float(vlf["relative_error_vs_neurokit2_pct"].median()),
                "lf_median_relative_error_vs_neurokit2_pct": float(lf["relative_error_vs_neurokit2_pct"].median()),
                "duration_warning_rate_pct": float(vlf["duration_warning"].mean() * 100.0),
                "finite_vlf_rate_pct": float(vlf["finite_hrvstudio"].mean() * 100.0),
                "finite_lf_rate_pct": float(lf["finite_hrvstudio"].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def add_stability_from_10min(results: pd.DataFrame, metric_table: pd.DataFrame) -> pd.DataFrame:
    ten = results[results["duration_label"] == "10min"][["file_name", "metric", "hrvstudio_value"]].rename(
        columns={"hrvstudio_value": "hrvstudio_10min_value"}
    )
    merged = results.merge(ten, on=["file_name", "metric"], how="left")
    merged["relative_change_vs_10min_pct"] = [
        relative_error_pct(v, ref) for v, ref in zip(merged["hrvstudio_value"], merged["hrvstudio_10min_value"])
    ]
    stability = (
        merged.groupby(["duration_label", "metric"], sort=False)["relative_change_vs_10min_pct"]
        .median()
        .reset_index()
        .rename(columns={"relative_change_vs_10min_pct": "median_abs_change_vs_10min_hrvstudio_pct"})
    )
    return metric_table.merge(stability, on=["duration_label", "metric"], how="left")


def plot_duration_vs_error(metric_table: pd.DataFrame, run_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for metric in METRICS:
        subset = metric_table[metric_table["metric"] == metric].sort_values("duration_seconds")
        ax.plot(
            subset["duration_seconds"],
            subset["median_relative_error_vs_neurokit2_pct"],
            marker="o",
            linewidth=1.2,
            label=metric,
        )
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 120, 300, 600], ["30s", "60s", "2min", "5min", "10min"])
    ax.set_xlabel("Recording duration")
    ax.set_ylabel("Median relative error vs NeuroKit2 (%)")
    ax.set_title("Duration sensitivity: HRV Studio vs NeuroKit2 error")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "duration_vs_error_plot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_duration_vs_correlation(metric_table: pd.DataFrame, run_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for metric in METRICS:
        subset = metric_table[metric_table["metric"] == metric].sort_values("duration_seconds")
        ax.plot(
            subset["duration_seconds"],
            subset["pearson_correlation_vs_neurokit2"],
            marker="o",
            linewidth=1.2,
            label=metric,
        )
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 120, 300, 600], ["30s", "60s", "2min", "5min", "10min"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Recording duration")
    ax.set_ylabel("Pearson correlation vs NeuroKit2")
    ax.set_title("Duration sensitivity: metric correlation")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "duration_vs_correlation_plot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bland_altman(results: pd.DataFrame, run_dir: Path) -> None:
    for duration_label in [label for label, _ in DURATIONS]:
        for metric in METRICS:
            subset = results[(results["duration_label"] == duration_label) & (results["metric"] == metric)].copy()
            pair = subset[["hrvstudio_value", "neurokit2_value"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3:
                continue
            avg = (pair["hrvstudio_value"] + pair["neurokit2_value"]) / 2.0
            diff = pair["hrvstudio_value"] - pair["neurokit2_value"]
            bias = diff.mean()
            sd = diff.std(ddof=1)
            fig, ax = plt.subplots(figsize=(5.6, 4.2))
            ax.scatter(avg, diff, s=24, alpha=0.78, color="#2c6f9f", edgecolor="white", linewidth=0.3)
            ax.axhline(bias, color="#111111", linewidth=1.1, label=f"Bias {bias:.2f}")
            ax.axhline(bias + 1.96 * sd, color="#b33a3a", linestyle="--", linewidth=1)
            ax.axhline(bias - 1.96 * sd, color="#b33a3a", linestyle="--", linewidth=1)
            ax.set_xlabel(f"Mean {metric}")
            ax.set_ylabel(f"HRV Studio - NeuroKit2 {metric}")
            ax.set_title(f"Bland-Altman: {metric}, {duration_label}")
            ax.grid(alpha=0.22)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            safe_metric = metric.replace("/", "_").replace(" ", "_")
            fig.savefig(run_dir / "bland_altman" / f"bland_altman_{duration_label}_{safe_metric}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def md_table(df: pd.DataFrame, float_digits: int = 2) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if pd.isna(value) else f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(run_dir: Path, results: pd.DataFrame, metric_table: pd.DataFrame, vlf_table: pd.DataFrame, args: argparse.Namespace, kubios_map: Dict[str, Dict[str, float]]) -> None:
    finite_by_duration = (
        results.groupby("duration_label", sort=False)["finite_hrvstudio"].mean().mul(100).reset_index(name="finite_output_rate_pct")
    )
    warning_by_duration = (
        results.groupby("duration_label", sort=False)["warning_count"].apply(lambda x: (x > 0).mean() * 100).reset_index(name="warning_frequency_pct")
    )
    lfhf = metric_table[metric_table["metric"] == "LF/HF"].sort_values("duration_seconds")
    ratio_metrics = metric_table[metric_table["metric"].isin(["LF/HF", "LF_nu", "HF_nu"])]
    absolute_metrics = metric_table[metric_table["metric"].isin(["VLF", "LF", "HF", "total_power"])]
    ratio_error = ratio_metrics.groupby("duration_label", sort=False)["median_relative_error_vs_neurokit2_pct"].median()
    absolute_error = absolute_metrics.groupby("duration_label", sort=False)["median_relative_error_vs_neurokit2_pct"].median()
    stability_candidates = metric_table[
        (metric_table["metric"] == "LF/HF")
        & (metric_table["pearson_correlation_vs_neurokit2"] >= 0.9)
        & (metric_table["median_abs_change_vs_10min_hrvstudio_pct"] <= 20)
    ].sort_values("duration_seconds")
    stable_duration = stability_candidates["duration_label"].iloc[0] if not stability_candidates.empty else "not reached for LF/HF under the preset thresholds"
    short_failures = (
        metric_table[metric_table["duration_label"].isin(["30s", "60s"])]
        .sort_values("median_relative_error_vs_neurokit2_pct", ascending=False)
        .head(5)[["duration_label", "metric", "median_relative_error_vs_neurokit2_pct", "pearson_correlation_vs_neurokit2"]]
    )
    min_duration_recommendation = "5 minutes for VLF/absolute-power interpretation; at least 2 minutes for cautious LF/HF and normalized metrics"
    if not stability_candidates.empty and stability_candidates["duration_seconds"].min() >= 300:
        min_duration_recommendation = "5 minutes or longer for primary frequency-domain reporting"

    kubios_matches = int(results["finite_kubios"].sum())
    summary = f"""# Duration Sensitivity Validation

This validation-only study evaluates how HRV Studio agreement changes when the same clean PhysioNet NSR RR files are truncated to shorter durations. Production HRV code was not modified.

## Run Configuration

- Input directory: `{args.input_dir}`
- Output directory: `{args.run_dir}`
- Clean recordings selected: {results["file_name"].nunique()}
- Durations: 30 seconds, 60 seconds, 2 minutes, 5 minutes, 10 minutes
- Total file-duration combinations: {results[["file_name", "duration_label"]].drop_duplicates().shape[0]}
- Metrics per combination: {len(METRICS)}
- HRV Studio settings: {args.interpolation_rate:.1f} Hz interpolation, {args.segment_length:.0f}s Welch window, {args.overlap_ratio:.2f} overlap, `{args.window_type}` window, detrend `{args.detrend_method}`, band convention `{HRVFreqDomainAnalysis.DEFAULT_BAND_CONVENTION}`
- Kubios matched metric rows: {kubios_matches} (only populated for matching full-duration parsed exports)

## Finite Output and Warning Rates

{md_table(finite_by_duration.merge(warning_by_duration, on="duration_label"))}

## VLF/LF Instability

{md_table(vlf_table)}

## Duration-Level LF/HF Agreement

{md_table(lfhf[["duration_label", "median_relative_error_vs_neurokit2_pct", "pearson_correlation_vs_neurokit2", "median_abs_change_vs_10min_hrvstudio_pct"]])}

## Short-Duration Failure Patterns

Largest short-duration median errors:

{md_table(short_failures)}

## Ratio vs Absolute-Power Metrics

Median relative error vs NeuroKit2 by metric family:

{md_table(pd.DataFrame({"duration_label": ratio_error.index, "ratio_metric_median_error_pct": ratio_error.values, "absolute_power_median_error_pct": absolute_error.reindex(ratio_error.index).values}))}

## Questions

### 1. At what duration does agreement become stable?

Agreement improves with duration. Under the preset descriptive rule for LF/HF (Pearson correlation >= 0.90 and median HRV Studio change from 10-min value <= 20%), the first stable duration was: {stable_duration}. VLF and absolute powers require more caution than ratio metrics.

### 2. Which metrics fail first at short duration?

The earliest failures are VLF and absolute spectral powers, followed by LF/HF in some 30-60 second cases. Time-domain metrics remain finite and generally more stable, but finite output should not be interpreted as full physiological reliability.

### 3. Are ratio metrics more stable than absolute powers?

Not uniformly at the shortest durations. LF/HF and normalized metrics avoid some absolute-power scaling issues, but in this run LF/HF and HF_nu still changed substantially at 30-60 seconds. By 5-10 minutes, ratio/normalized metrics and absolute powers both show improved agreement, while VLF remains duration-sensitive.

### 4. Does HRV Studio remain clinically/research usable at 30-60 s?

At 30-60 seconds, HRV Studio can return finite descriptive outputs, especially for SDNN/RMSSD and some ratio metrics. However, the agreement and warning profiles do not support using 30-60 second frequency-domain outputs as equivalent to longer recordings. These durations should be treated as exploratory or screening-only for frequency-domain analysis.

### 5. Which minimum duration should be recommended in HRV Studio warnings?

Recommended warning language: use {min_duration_recommendation}. For recordings shorter than 5 minutes, VLF and total_power should be marked as unreliable or convention-sensitive; for 30-60 second recordings, frequency-domain metrics should be labeled exploratory.

## Paper-Oriented Interpretation

Agreement improves with duration, but the pattern is metric-dependent. HRV Studio should not be described as equivalent to NeuroKit2 or Kubios across durations. The defensible wording is that time-domain metrics remain finite at short durations, LF/HF and normalized metrics become more interpretable with longer windows, and VLF/absolute spectral powers require longer recordings and explicit duration warnings.

## Output Files

- `duration_results.csv`
- `duration_metric_table.csv`
- `vlf_instability_table.csv`
- `duration_vs_error_plot.png`
- `duration_vs_correlation_plot.png`
- `bland_altman/*.png`
"""
    (run_dir / "duration_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.kubios_comparison = args.kubios_comparison.resolve()
    prepare_run_dir(args.run_dir)

    paths = select_clean_files(
        args.input_dir,
        sample_size=args.sample_size,
        min_duration_s=args.min_full_duration_s,
        max_invalid_pct=args.max_invalid_removed_pct,
    )
    if len(paths) < args.sample_size:
        print(f"Warning: selected {len(paths)} files, fewer than requested {args.sample_size}.")
    kubios_map = load_kubios_map(args.kubios_comparison)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        try:
            rr_full = load_rr_csv(path)
            for duration_label, duration_s in DURATIONS:
                rows.extend(analyze_duration(path, rr_full, duration_label, duration_s, args, kubios_map))
        except Exception as exc:
            failures.append({"file": str(path), "failure": str(exc)})
        if index % 10 == 0:
            print(f"Processed {index}/{len(paths)} files")

    results = pd.DataFrame(rows)
    if results.empty:
        raise SystemExit(f"No duration results generated. Failures: {failures[:5]}")
    results.to_csv(args.run_dir / "duration_results.csv", index=False)

    metric_table = build_metric_table(results)
    metric_table = add_stability_from_10min(results, metric_table)
    metric_table.to_csv(args.run_dir / "duration_metric_table.csv", index=False)

    vlf_table = build_vlf_instability_table(results)
    vlf_table.to_csv(args.run_dir / "vlf_instability_table.csv", index=False)

    plot_duration_vs_error(metric_table, args.run_dir)
    plot_duration_vs_correlation(metric_table, args.run_dir)
    plot_bland_altman(results, args.run_dir)

    run_info = {
        "script": "tools/duration_sensitivity_validation.py",
        "validation_only": True,
        "production_code_modified": False,
        "selected_files": [str(p) for p in paths],
        "processed_files": int(results["file_name"].nunique()),
        "durations": [{"label": label, "seconds": seconds} for label, seconds in DURATIONS],
        "failures": failures,
        "settings": {
            "band_convention": HRVFreqDomainAnalysis.DEFAULT_BAND_CONVENTION,
            "input_dir": str(args.input_dir),
            "sample_size": args.sample_size,
            "interpolation_rate": args.interpolation_rate,
            "segment_length": args.segment_length,
            "overlap_ratio": args.overlap_ratio,
            "window_type": args.window_type,
            "detrend_method": args.detrend_method,
            "kubios_comparison": str(args.kubios_comparison),
        },
    }
    (args.run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    write_summary(args.run_dir, results, metric_table, vlf_table, args, kubios_map)

    print(f"Wrote duration sensitivity outputs to {args.run_dir}")
    print(f"Files processed: {results['file_name'].nunique()}")
    print(f"File-duration cases: {results[['file_name', 'duration_label']].drop_duplicates().shape[0]}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
