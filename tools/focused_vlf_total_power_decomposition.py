"""
Focused VLF / total-power decomposition for manual validation inspections.

This is a validation/debugging utility only. It does not modify HRV Studio
production analysis behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "validation" / "runs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from inspect_freq_outlier import (
    common_grid_compare,
    detrend_signal_ms,
    native_resampled_rr_ms,
    neurokit2_resampled_rr_ms,
    safe_file_name,
    shape_similarity,
)
from validate_freq_domain_neurokit2 import (
    METRICS,
    band_bin_counts,
    band_ranges,
    effective_welch_params,
    experimental_native_welch_psd,
    load_rr_intervals_ms,
    metrics_from_psd,
    neurokit2_welch_psd,
    normalize_detrend_method,
    relative_error,
    rr_start_times_s,
    safe_abs_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate focused VLF/total-power decomposition artifacts."
    )
    parser.add_argument("input_files", nargs="+", help="RR files to inspect.")
    parser.add_argument("--run-name", required=True, help="Run folder under validation/runs/.")
    parser.add_argument("--detrend-method", choices=["none", "linear", "constant", "smoothness_priors"], default="none")
    parser.add_argument("--interpolation-rate", type=float, default=4.0)
    parser.add_argument("--segment-length", type=float, default=120.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.75)
    parser.add_argument("--window-type", default="hann")
    parser.add_argument("--ar-order", type=int, default=16)
    parser.add_argument("--detrend-lambda", type=float, default=500.0)
    parser.add_argument("--neurokit-interpolation-method", default="cubic")
    parser.add_argument("--enable-diagnostics", action="store_true")
    parser.add_argument("--experimental-native-welch-nfft-multiplier", type=float, default=2.0)
    return parser.parse_args()


def finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def fmt(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "not available"
    return f"{float(value):.{digits}f}"


def csv_cell(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)
    return "" if value is None else value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})


def integrate_band(freqs: np.ndarray, psd: np.ndarray, low: float, high: float, include_dc: bool = True) -> float:
    if len(freqs) < 2 or len(psd) < 2:
        return math.nan
    mask = (freqs >= low) & (freqs <= high) & np.isfinite(freqs) & np.isfinite(psd)
    if not include_dc:
        mask &= freqs > 0.0
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(max(0.0, np.trapezoid(psd[mask], freqs[mask])))


def rectangular_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float, include_dc: bool = True) -> float:
    if len(freqs) < 2 or len(psd) < 2:
        return math.nan
    mask = (freqs >= low) & (freqs <= high) & np.isfinite(freqs) & np.isfinite(psd)
    if not include_dc:
        mask &= freqs > 0.0
    if np.count_nonzero(mask) == 0:
        return 0.0
    spacing = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 0.0
    return float(max(0.0, np.sum(psd[mask]) * spacing))


def count_band_bins(freqs: np.ndarray, low: float, high: float, include_dc: bool = True) -> int:
    if len(freqs) == 0:
        return 0
    mask = (freqs >= low) & (freqs <= high)
    if not include_dc:
        mask &= freqs > 0.0
    return int(np.count_nonzero(mask))


def band_label(freq: float) -> str:
    if freq <= 0.003:
        return "ULF+VLF"
    if freq <= 0.04:
        return "VLF"
    if freq < 0.15:
        return "LF"
    if freq <= 0.4:
        return "HF"
    return "outside_total"


def common_frequency_rows(
    native_freqs: np.ndarray,
    native_psd: np.ndarray,
    nk_freqs: np.ndarray,
    nk_psd: np.ndarray,
    experimental_freqs: np.ndarray,
    experimental_psd: np.ndarray,
) -> List[Dict[str, Any]]:
    grids = [freqs[(freqs >= 0.0) & (freqs <= 0.4)] for freqs in (native_freqs, nk_freqs, experimental_freqs) if len(freqs)]
    if not grids:
        return []
    common = np.array(sorted({round(float(freq), 12) for grid in grids for freq in grid}), dtype=float)
    native_interp = np.interp(common, native_freqs, native_psd) if len(native_freqs) else np.full_like(common, np.nan)
    nk_interp = np.interp(common, nk_freqs, nk_psd) if len(nk_freqs) else np.full_like(common, np.nan)
    exp_interp = np.interp(common, experimental_freqs, experimental_psd) if len(experimental_freqs) else np.full_like(common, np.nan)

    native_cumulative = cumulative_power(common, native_interp)
    nk_cumulative = cumulative_power(common, nk_interp)
    exp_cumulative = cumulative_power(common, exp_interp)

    rows: List[Dict[str, Any]] = []
    for i, freq in enumerate(common):
        native_value = float(native_interp[i])
        nk_value = float(nk_interp[i])
        ratio = nk_value / native_value if finite(native_value) and abs(native_value) > 1e-12 else math.nan
        rows.append(
            {
                "frequency": float(freq),
                "native_psd": native_value,
                "neurokit2_psd": nk_value,
                "experimental_native_nfft_x2_psd": float(exp_interp[i]) if len(experimental_freqs) else math.nan,
                "absolute_difference": abs(nk_value - native_value) if finite(native_value) and finite(nk_value) else math.nan,
                "ratio_neurokit2_native": ratio,
                "native_cumulative_power": float(native_cumulative[i]),
                "neurokit2_cumulative_power": float(nk_cumulative[i]),
                "experimental_native_nfft_x2_cumulative_power": float(exp_cumulative[i]) if len(experimental_freqs) else math.nan,
                "band_label": band_label(float(freq)),
            }
        )
    return rows


def cumulative_power(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    if len(freqs) == 0:
        return np.array([])
    values = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(len(freqs), dtype=float)
    if len(freqs) > 1:
        out[1:] = np.cumsum((values[:-1] + values[1:]) * 0.5 * np.diff(freqs))
    return out


def positive_limits(*arrays: np.ndarray) -> Tuple[float, float]:
    values = np.concatenate([arr[np.isfinite(arr) & (arr > 0)] for arr in arrays if len(arr)])
    if len(values) == 0:
        return 1e-6, 1.0
    return max(float(np.min(values)) * 0.5, 1e-12), float(np.max(values)) * 2.0


def plot_zoomed_vlf(
    native_freqs: np.ndarray,
    native_psd: np.ndarray,
    nk_freqs: np.ndarray,
    nk_psd: np.ndarray,
    experimental_freqs: np.ndarray,
    experimental_psd: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axvspan(0.0, 0.003, color="#e8e8e8", alpha=0.45, label="ULF overlap")
    ax.axvspan(0.003, 0.04, color="#e8eef7", alpha=0.55, label="VLF")
    ax.axvline(0.04, color="#555555", linewidth=0.9)
    ax.plot(native_freqs, native_psd, marker="o", markersize=3, linewidth=1.5, label="HRV Studio native Welch")
    ax.plot(nk_freqs, nk_psd, marker="o", markersize=3, linewidth=1.5, label="NeuroKit2 Welch")
    if len(experimental_freqs):
        ax.plot(experimental_freqs, experimental_psd, linestyle="--", linewidth=1.2, label="Native Welch nfft x2")
    ax.set_xlim(0.0, 0.06)
    y_min, y_max = positive_limits(native_psd[native_freqs <= 0.06], nk_psd[nk_freqs <= 0.06], experimental_psd[experimental_freqs <= 0.06])
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD power (ms^2/Hz)")
    ax.set_title("Zoomed VLF PSD")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cumulative(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    freqs = np.asarray([row["frequency"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axvspan(0.0, 0.04, color="#e8eef7", alpha=0.45)
    ax.axvspan(0.04, 0.15, color="#edf7ed", alpha=0.45)
    ax.axvspan(0.15, 0.4, color="#fff1df", alpha=0.45)
    ax.plot(freqs, [row["native_cumulative_power"] for row in rows], label="HRV Studio native Welch", linewidth=1.7)
    ax.plot(freqs, [row["neurokit2_cumulative_power"] for row in rows], label="NeuroKit2 Welch", linewidth=1.7)
    if any(finite(row.get("experimental_native_nfft_x2_cumulative_power")) for row in rows):
        ax.plot(
            freqs,
            [row["experimental_native_nfft_x2_cumulative_power"] for row in rows],
            label="Native Welch nfft x2",
            linewidth=1.2,
            linestyle="--",
        )
    ax.set_xlim(0.0, 0.4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cumulative integrated power (ms^2)")
    ax.set_title("Cumulative power by frequency")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def metric_rows(native_metrics: Dict[str, float], nk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    for label, key in METRICS.items():
        native_value = float(native_metrics.get(key, math.nan))
        nk_value = float(nk_metrics.get(key, math.nan))
        rows.append(
            {
                "metric": label,
                "native_value": native_value,
                "neurokit2_value": nk_value,
                "absolute_error": safe_abs_error(native_value, nk_value),
                "relative_error_pct": relative_error(native_value, nk_value) * 100.0,
            }
        )
    return rows


def band_power_rows(
    native_freqs: np.ndarray,
    native_psd: np.ndarray,
    nk_freqs: np.ndarray,
    nk_psd: np.ndarray,
    experimental_freqs: np.ndarray,
    experimental_psd: np.ndarray,
) -> List[Dict[str, Any]]:
    definitions = [
        ("ULF", 0.0, 0.003),
        ("VLF_0_0.04", 0.0, 0.04),
        ("VLF_0.003_0.04", 0.003, 0.04),
        ("LF", 0.04, 0.15),
        ("HF", 0.15, 0.4),
    ]
    rows: List[Dict[str, Any]] = []
    for pipeline, freqs, psd in [
        ("native_hrv_studio", native_freqs, native_psd),
        ("neurokit2", nk_freqs, nk_psd),
        ("experimental_native_nfft_x2", experimental_freqs, experimental_psd),
    ]:
        if len(freqs) == 0:
            continue
        for band, low, high in definitions:
            for include_dc in [True, False]:
                rows.append(
                    {
                        "pipeline": pipeline,
                        "band_definition": band,
                        "include_dc": include_dc,
                        "bins": count_band_bins(freqs, low, high, include_dc),
                        "trapz_power": integrate_band(freqs, psd, low, high, include_dc),
                        "rectangular_bin_sum_power": rectangular_power(freqs, psd, low, high, include_dc),
                    }
                )
    return rows


def near_zero_concentration(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"near_zero_difference_share": math.nan, "concentrated": False}
    vlf_diff = sum(float(row["absolute_difference"]) for row in rows if row["frequency"] <= 0.04 and finite(row["absolute_difference"]))
    near_zero_diff = sum(float(row["absolute_difference"]) for row in rows if row["frequency"] <= 0.0125 and finite(row["absolute_difference"]))
    share = near_zero_diff / vlf_diff if vlf_diff > 0 else math.nan
    return {
        "near_zero_difference_share": share,
        "concentrated": finite(share) and share >= 0.65,
    }


def classify_issue(
    rows: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    before_corr: float,
    after_corr: float,
    psd_corr: float,
) -> str:
    rel = {row["metric"]: float(row["relative_error_pct"]) for row in metrics if finite(row["relative_error_pct"])}
    lf_hf_aligned = rel.get("LF/HF", math.inf) <= 10.0
    lf_hf_powers_aligned = rel.get("LF", math.inf) <= 20.0 and rel.get("HF", math.inf) <= 20.0
    near_zero = near_zero_concentration(rows)["concentrated"]
    high_vlf_total = rel.get("VLF", 0.0) >= 20.0 and rel.get("total_power", 0.0) >= 20.0
    all_power_errors = [rel.get(name, math.nan) for name in ["VLF", "LF", "HF", "total_power"]]
    finite_power_errors = [value for value in all_power_errors if finite(value)]

    if finite(before_corr) and finite(after_corr) and (before_corr < 0.97 or after_corr < 0.97):
        return "a) artifact/interpolation mismatch"
    if finite(psd_corr) and psd_corr < 0.60:
        return "b) broad PSD shape mismatch"
    if high_vlf_total and lf_hf_aligned and lf_hf_powers_aligned and near_zero:
        return "c) localized VLF / near-zero-frequency mismatch"
    if len(finite_power_errors) >= 3 and max(finite_power_errors) - min(finite_power_errors) <= 15.0 and min(finite_power_errors) >= 20.0:
        return "d) global scaling mismatch"
    return "e) unclear"


def top_error_text(metrics: Sequence[Dict[str, Any]], n: int = 3) -> str:
    top = sorted(
        [row for row in metrics if finite(row.get("relative_error_pct"))],
        key=lambda row: float(row["relative_error_pct"]),
        reverse=True,
    )[:n]
    return "; ".join(f"{row['metric']} {float(row['relative_error_pct']):.2f}%" for row in top)


def metric_lookup(metrics: Sequence[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for row in metrics:
        if row["metric"] == name:
            return row
    return {"native_value": math.nan, "neurokit2_value": math.nan, "relative_error_pct": math.nan}


def inspect_file(args: argparse.Namespace, input_path: Path) -> Dict[str, Any]:
    rr_ms = load_rr_intervals_ms(input_path)
    detrend_method = normalize_detrend_method(args.detrend_method)
    analyzer = HRVFreqDomainAnalysis(
        rr_ms,
        sampling_rate=args.interpolation_rate,
        detrend_method=detrend_method,
        detrend_lambda=args.detrend_lambda,
        window_type=args.window_type,
        segment_length=args.segment_length,
        overlap_ratio=args.overlap_ratio,
        ar_order=args.ar_order,
        enable_diagnostics=args.enable_diagnostics,
    )
    native_results = analyzer.get_results()
    native_nperseg, native_noverlap = effective_welch_params(analyzer)
    if native_nperseg is None or native_noverlap is None:
        raise RuntimeError(f"Cannot determine Welch parameters for {input_path}")

    nk_freqs, nk_psd, nk_nperseg, nk_noverlap, nk_duration_s, _adjustment = neurokit2_welch_psd(
        path=input_path,
        rr_ms=analyzer.rr_intervals_ms,
        sampling_rate=args.interpolation_rate,
        interpolation_method=args.neurokit_interpolation_method,
        window_type=args.window_type,
        nperseg=native_nperseg,
        noverlap=native_noverlap,
        detrend_method=detrend_method,
        detrend_lambda=args.detrend_lambda,
    )
    experimental_freqs, experimental_psd, experimental_nfft = experimental_native_welch_psd(
        analyzer,
        native_nperseg,
        native_noverlap,
        args.experimental_native_welch_nfft_multiplier,
    )

    native_time_s, native_resampled_ms = native_resampled_rr_ms(analyzer)
    nk_time_s, nk_resampled_ms = neurokit2_resampled_rr_ms(
        analyzer.rr_intervals_ms,
        args.interpolation_rate,
        args.neurokit_interpolation_method,
    )
    native_detrended_ms = detrend_signal_ms(native_resampled_ms, detrend_method, args.detrend_lambda)
    nk_detrended_ms = detrend_signal_ms(nk_resampled_ms, detrend_method, args.detrend_lambda)
    before_compare = common_grid_compare(native_time_s, native_resampled_ms, nk_time_s, nk_resampled_ms, args.interpolation_rate)
    after_compare = common_grid_compare(native_time_s, native_detrended_ms, nk_time_s, nk_detrended_ms, args.interpolation_rate)

    native_metrics = {
        key: float(native_results.get(f"welch_{key}", native_results.get(key, math.nan)))
        for key in METRICS.values()
    }
    nk_metrics = metrics_from_psd(nk_freqs, nk_psd)
    metrics = metric_rows(native_metrics, nk_metrics)
    rows = common_frequency_rows(analyzer.freqs, analyzer.psd, nk_freqs, nk_psd, experimental_freqs, experimental_psd)
    power_rows = band_power_rows(analyzer.freqs, analyzer.psd, nk_freqs, nk_psd, experimental_freqs, experimental_psd)
    similarity = shape_similarity(analyzer.freqs, analyzer.psd, nk_freqs, nk_psd)
    concentration = near_zero_concentration(rows)
    issue_type = classify_issue(
        rows,
        metrics,
        before_compare["pearson_correlation"],
        after_compare["pearson_correlation"],
        similarity.get("correlation", math.nan),
    )

    output_dir = RUNS_ROOT / args.run_name / "manual_inspection" / safe_file_name(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_zoomed_vlf(analyzer.freqs, analyzer.psd, nk_freqs, nk_psd, experimental_freqs, experimental_psd, output_dir / "zoomed_vlf_psd.png")
    plot_cumulative(rows, output_dir / "cumulative_power_by_frequency.png")
    write_csv(
        output_dir / "vlf_total_power_decomposition.csv",
        rows,
        [
            "frequency",
            "native_psd",
            "neurokit2_psd",
            "experimental_native_nfft_x2_psd",
            "absolute_difference",
            "ratio_neurokit2_native",
            "native_cumulative_power",
            "neurokit2_cumulative_power",
            "experimental_native_nfft_x2_cumulative_power",
            "band_label",
        ],
    )

    duration_s = float(rr_start_times_s(analyzer.rr_intervals_ms)[-1]) if len(analyzer.rr_intervals_ms) else 0.0
    native_counts = band_bin_counts(analyzer.freqs)
    nk_counts = band_bin_counts(nk_freqs)
    summary = {
        "file": input_path.name,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "duration_s": duration_s,
        "rr_count": len(analyzer.rr_intervals_ms),
        "native_nperseg": native_nperseg,
        "native_noverlap": native_noverlap,
        "native_nfft": estimate_nfft(analyzer.freqs, args.interpolation_rate, native_nperseg),
        "neurokit2_nperseg": nk_nperseg,
        "neurokit2_noverlap": nk_noverlap,
        "neurokit2_nfft": estimate_nfft(nk_freqs, args.interpolation_rate, None),
        "experimental_native_nfft": experimental_nfft,
        "native_bins_vlf": native_counts.get("vlf", 0),
        "neurokit2_bins_vlf": nk_counts.get("vlf", 0),
        "largest_error_metrics": top_error_text(metrics),
        "metrics": metrics,
        "resampled_signal_correlation": before_compare["pearson_correlation"],
        "detrended_signal_correlation": after_compare["pearson_correlation"],
        "psd_log_correlation": similarity.get("correlation", math.nan),
        "near_zero_difference_share": concentration["near_zero_difference_share"],
        "near_zero_concentrated": concentration["concentrated"],
        "lf_hf_aligned": float(metric_lookup(metrics, "LF/HF")["relative_error_pct"]) <= 10.0 if finite(metric_lookup(metrics, "LF/HF")["relative_error_pct"]) else False,
        "issue_type": issue_type,
        "band_power_rows": power_rows,
    }
    write_vlf_summary(output_dir / "vlf_total_power_summary.md", summary, args)
    return summary


def estimate_nfft(freqs: np.ndarray, sampling_rate: float, fallback: Optional[int]) -> Optional[int]:
    if len(freqs) > 1:
        spacing = float(freqs[1] - freqs[0])
        if spacing > 0:
            return int(round(sampling_rate / spacing))
    return fallback


def write_power_table(lines: List[str], title: str, rows: Sequence[Dict[str, Any]], pipeline: str) -> None:
    lines.extend([f"### {title}", "", "| Band definition | Include DC | Bins | Trapz power | Rectangular/bin-sum power |", "| --- | --- | ---: | ---: | ---: |"])
    for row in rows:
        if row["pipeline"] != pipeline:
            continue
        lines.append(
            f"| {row['band_definition']} | {row['include_dc']} | {row['bins']} | "
            f"{fmt(row['trapz_power'], 3)} | {fmt(row['rectangular_bin_sum_power'], 3)} |"
        )
    lines.append("")


def write_vlf_summary(path: Path, summary: Dict[str, Any], args: argparse.Namespace) -> None:
    vlf = metric_lookup(summary["metrics"], "VLF")
    total = metric_lookup(summary["metrics"], "total_power")
    lf = metric_lookup(summary["metrics"], "LF")
    hf = metric_lookup(summary["metrics"], "HF")
    lf_hf = metric_lookup(summary["metrics"], "LF/HF")
    lines = [
        "# Focused VLF and Total-Power Discrepancy Analysis",
        "",
        f"File: `{summary['input_path']}`",
        "",
        "This analysis is for validation review only. It does not claim that either implementation is correct.",
        "",
        "## Method",
        "",
        "- Recomputed the same native and NeuroKit2 Welch PSDs used by the manual inspection workflow.",
        f"- Detrending: `{args.detrend_method}`; interpolation rate: `{args.interpolation_rate:g}` Hz.",
        f"- Native Welch parameters: `nperseg={summary['native_nperseg']}`, `noverlap={summary['native_noverlap']}`, inferred `nfft={summary['native_nfft']}`.",
        f"- NeuroKit2 Welch parameters: `nperseg={summary['neurokit2_nperseg']}`, `noverlap={summary['neurokit2_noverlap']}`, inferred `nfft={summary['neurokit2_nfft']}`.",
        f"- Included the experimental native Welch `nfft x2` PSD for context; experimental `nfft={summary['experimental_native_nfft']}`.",
        "- The decomposition CSV uses a common 0.0-0.40 Hz frequency grid; values are linearly interpolated when a frequency exists in only one pipeline grid.",
        "",
        "## PSD Bin-Level Observation",
        "",
        f"- Native has {summary['native_bins_vlf']} bins in VLF 0.0-0.04 Hz; NeuroKit2 has {summary['neurokit2_bins_vlf']} bins.",
        f"- Near-zero absolute PSD separation share within VLF: {fmt(float(summary['near_zero_difference_share']) * 100.0 if finite(summary['near_zero_difference_share']) else math.nan, 2)}%.",
        "- This near-zero share is heuristic and should be interpreted from the plots, not as a formal causal proof.",
        "",
        "## Metric Snapshot",
        "",
        "| Metric | Native | NeuroKit2 | Relative error |",
        "| --- | ---: | ---: | ---: |",
        f"| VLF | {fmt(vlf['native_value'], 3)} | {fmt(vlf['neurokit2_value'], 3)} | {fmt(vlf['relative_error_pct'], 2)}% |",
        f"| LF | {fmt(lf['native_value'], 3)} | {fmt(lf['neurokit2_value'], 3)} | {fmt(lf['relative_error_pct'], 2)}% |",
        f"| HF | {fmt(hf['native_value'], 3)} | {fmt(hf['neurokit2_value'], 3)} | {fmt(hf['relative_error_pct'], 2)}% |",
        f"| total_power | {fmt(total['native_value'], 3)} | {fmt(total['neurokit2_value'], 3)} | {fmt(total['relative_error_pct'], 2)}% |",
        f"| LF/HF | {fmt(lf_hf['native_value'], 3)} | {fmt(lf_hf['neurokit2_value'], 3)} | {fmt(lf_hf['relative_error_pct'], 2)}% |",
        "",
        "## Band-Specific Integrated Power",
        "",
    ]
    write_power_table(lines, "Native HRV Studio PSD", summary["band_power_rows"], "native_hrv_studio")
    write_power_table(lines, "NeuroKit2 PSD", summary["band_power_rows"], "neurokit2")
    write_power_table(lines, "Experimental native Welch nfft x2 PSD", summary["band_power_rows"], "experimental_native_nfft_x2")
    lines.extend(
        [
            "## Cautious Interpretation",
            "",
            f"- Resampled-signal correlation: {fmt(summary['resampled_signal_correlation'], 3)}.",
            f"- Detrended-signal correlation: {fmt(summary['detrended_signal_correlation'], 3)}.",
            f"- Log-PSD correlation: {fmt(summary['psd_log_correlation'], 3)}.",
            f"- LF/HF remains aligned by the current threshold: {'yes' if summary['lf_hf_aligned'] else 'no'}.",
            f"- Automated issue label: {summary['issue_type']}.",
            "",
            "This label is heuristic. The plots and tables should be reviewed before drawing conclusions about either implementation.",
            "",
            "## Generated Files",
            "",
            "- `zoomed_vlf_psd.png`",
            "- `cumulative_power_by_frequency.png`",
            "- `vlf_total_power_decomposition.csv`",
            "- `vlf_total_power_summary.md`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_pair(summary: Dict[str, Any], metric: str) -> str:
    row = metric_lookup(summary["metrics"], metric)
    return f"{fmt(row['native_value'], 3)} / {fmt(row['neurokit2_value'], 3)}"


def write_combined_summary(run_name: str, summaries: Sequence[Dict[str, Any]]) -> Path:
    out_path = RUNS_ROOT / run_name / "manual_inspection" / "physionet_10min_outlier_inspection_summary.md"
    lines = [
        "# PhysioNet 10-Minute Outlier Manual Inspection Summary",
        "",
        "This summary compares a small representative manual-inspection subset from `v04_physionet_10min_neurokit2`. It uses cautious, heuristic labels and does not identify either implementation as ground truth.",
        "",
        "| File | Duration (s) | Largest error metrics | VLF native / NeuroKit2 | LF native / NeuroKit2 | HF native / NeuroKit2 | total_power native / NeuroKit2 | LF/HF native / NeuroKit2 | Resampled corr | Detrended corr | PSD log-corr | DC / first VLF concentration | LF/HF aligned | Issue label |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| {file} | {duration} | {top} | {vlf} | {lf} | {hf} | {total} | {lfhf} | {resamp} | {detrended} | {psd} | {near_zero} | {lfhf_aligned} | {issue} |".format(
                file=f"`{summary['file']}`",
                duration=fmt(summary["duration_s"], 2),
                top=summary["largest_error_metrics"],
                vlf=metric_pair(summary, "VLF"),
                lf=metric_pair(summary, "LF"),
                hf=metric_pair(summary, "HF"),
                total=metric_pair(summary, "total_power"),
                lfhf=metric_pair(summary, "LF/HF"),
                resamp=fmt(summary["resampled_signal_correlation"], 3),
                detrended=fmt(summary["detrended_signal_correlation"], 3),
                psd=fmt(summary["psd_log_correlation"], 3),
                near_zero=(
                    "yes"
                    if summary["near_zero_concentrated"]
                    else "no"
                )
                + f" ({fmt(float(summary['near_zero_difference_share']) * 100.0 if finite(summary['near_zero_difference_share']) else math.nan, 1)}%)",
                lfhf_aligned="yes" if summary["lf_hf_aligned"] else "no",
                issue=summary["issue_type"],
            )
        )

    lines.extend(
        [
            "",
            "## Cautious Comparative Notes",
            "",
            "- `a) artifact/interpolation mismatch` is used when the resampled or detrended signals are not closely correlated before PSD estimation.",
            "- `b) broad PSD shape mismatch` is used when the common-grid log-PSD correlation is low.",
            "- `c) localized VLF / near-zero-frequency mismatch` is used when VLF and total_power are discrepant while LF, HF, LF/HF, and broader PSD shape remain comparatively aligned and the VLF difference is concentrated near DC / first VLF bins.",
            "- `d) global scaling mismatch` is reserved for similarly large errors across most absolute-power bands.",
            "- `e) unclear` means this heuristic did not cleanly identify one of the above patterns.",
            "",
            "## Generated Per-File Artifacts",
            "",
        ]
    )
    for summary in summaries:
        rel_dir = Path(summary["output_dir"]).relative_to(PROJECT_ROOT)
        lines.append(f"- `{summary['file']}`: `{rel_dir}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    args = parse_args()
    summaries = []
    for input_file in args.input_files:
        path = Path(input_file)
        if not path.exists():
            raise SystemExit(f"Input file does not exist: {path}")
        summaries.append(inspect_file(args, path))
    out_path = write_combined_summary(args.run_name, summaries)
    print(f"Wrote combined summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
