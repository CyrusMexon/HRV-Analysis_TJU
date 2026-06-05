"""
Manual inspection helper for frequency-domain validation outliers.

This is a validation/debugging tool only. It does not modify HRV Studio
production analysis behavior.

Example:
    python tools/inspect_freq_outlier.py "validation/raw_data/Syl_Vain/2018-06-02 17-41-25.txt" --run-name v02b_nfft_alignment_experiment --detrend-method linear --enable-diagnostics
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import warnings
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
from validate_freq_domain_neurokit2 import (
    BAND_FOR_METRIC,
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
        description="Generate plots and tables for one frequency-domain validation outlier."
    )
    parser.add_argument("input_file", help="Path to one RR interval file.")
    parser.add_argument("--run-name", required=True, help="Run folder under validation/runs/.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional output directory. Defaults to "
            "validation/runs/<run_name>/manual_inspection/<safe_file_name>/."
        ),
    )
    parser.add_argument(
        "--detrend-method",
        choices=["none", "linear", "constant", "smoothness_priors"],
        default="none",
    )
    parser.add_argument("--interpolation-rate", type=float, default=4.0)
    parser.add_argument("--segment-length", type=float, default=120.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.75)
    parser.add_argument("--window-type", default="hann")
    parser.add_argument("--ar-order", type=int, default=16)
    parser.add_argument("--detrend-lambda", type=float, default=500.0)
    parser.add_argument("--neurokit-interpolation-method", default="cubic")
    parser.add_argument("--enable-diagnostics", action="store_true")
    parser.add_argument(
        "--experimental-native-welch-nfft-multiplier",
        type=float,
        default=1.0,
        help="Validation-only experiment: compute native Welch with nfft = multiplier * nperseg.",
    )
    return parser.parse_args()


def safe_file_name(path: Path) -> str:
    stem = path.name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "inspection"


def resolve_output_dir(args: argparse.Namespace, input_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return RUNS_ROOT / args.run_name / "manual_inspection" / safe_file_name(input_path)


def finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key)) for key in fieldnames})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.8g}"
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)
    return "" if value is None else value


def collect_warnings(obj: Any) -> List[str]:
    result: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lower = key.lower()
            if lower.endswith("warning") or lower.endswith("warnings"):
                if isinstance(value, list):
                    result.extend(str(item) for item in value if str(item).strip())
                elif value:
                    result.append(str(value))
            elif "fallback_reason" in lower and value:
                result.append(f"AR fallback: {value}")
            result.extend(collect_warnings(value))
    elif isinstance(obj, list):
        for item in obj:
            result.extend(collect_warnings(item))
    return dedupe(result)


def dedupe(values: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def plot_rr_tachogram(rr_ms: np.ndarray, input_path: Path, output_path: Path) -> None:
    times = rr_start_times_s(rr_ms)
    duration = float(times[-1]) if len(times) else 0.0
    x = times / 60.0 if duration >= 600 else times
    xlabel = "Cumulative time (minutes)" if duration >= 600 else "Cumulative time (seconds)"

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(x, rr_ms, linewidth=1.0, color="#1f77b4")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RR interval (ms)")
    ax.set_title(f"RR tachogram: {input_path.name} ({duration:.1f}s)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def neurokit2_resampled_rr_ms(
    rr_ms: np.ndarray,
    interpolation_rate: float,
    interpolation_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    import neurokit2 as nk

    intervals, intervals_time, _rate = nk.intervals_process(
        rr_ms,
        intervals_time=rr_start_times_s(rr_ms),
        interpolate=True,
        interpolation_rate=interpolation_rate,
        method=interpolation_method,
    )
    return np.asarray(intervals_time, dtype=float), np.asarray(intervals, dtype=float)


def native_resampled_rr_ms(analyzer: HRVFreqDomainAnalysis) -> Tuple[np.ndarray, np.ndarray]:
    values_ms = np.asarray(analyzer.time_domain_s, dtype=float) * 1000.0
    times_s = np.arange(len(values_ms), dtype=float) / analyzer.sampling_rate
    return times_s, values_ms


def detrend_signal_ms(
    values_ms: np.ndarray,
    detrend_method: Optional[str],
    detrend_lambda: float,
) -> np.ndarray:
    if len(values_ms) == 0:
        return np.array([])
    if detrend_method == "linear":
        return signal.detrend(values_ms, type="linear")
    if detrend_method == "constant":
        return signal.detrend(values_ms, type="constant")
    if detrend_method == "smoothness_priors":
        from hrvlib.signal_processing.smoothness_priors import (
            detrend_uniform_with_smoothness_priors,
        )

        return (
            detrend_uniform_with_smoothness_priors(
                values_ms / 1000.0,
                lambda_param=detrend_lambda,
                return_trend=False,
            )
            * 1000.0
        )
    return values_ms.copy()


def plot_signal_overlay(
    native_time_s: np.ndarray,
    native_values_ms: np.ndarray,
    nk_time_s: np.ndarray,
    nk_values_ms: np.ndarray,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if len(native_time_s):
        ax.plot(native_time_s, native_values_ms, label="Native resampled RR", linewidth=1.2)
    if len(nk_time_s):
        ax.plot(nk_time_s, nk_values_ms, label="NeuroKit2 resampled RR", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def common_grid_compare(
    native_time_s: np.ndarray,
    native_values_ms: np.ndarray,
    nk_time_s: np.ndarray,
    nk_values_ms: np.ndarray,
    sampling_rate: float,
) -> Dict[str, float]:
    if len(native_time_s) < 2 or len(nk_time_s) < 2:
        return {"pearson_correlation": math.nan, "rmse_ms": math.nan, "samples": 0}

    start = max(float(native_time_s[0]), float(nk_time_s[0]))
    stop = min(float(native_time_s[-1]), float(nk_time_s[-1]))
    if stop <= start:
        return {"pearson_correlation": math.nan, "rmse_ms": math.nan, "samples": 0}

    step = 1.0 / sampling_rate
    grid = np.arange(start, stop + step / 2.0, step)
    native_interp = np.interp(grid, native_time_s, native_values_ms)
    nk_interp = np.interp(grid, nk_time_s, nk_values_ms)
    valid = np.isfinite(native_interp) & np.isfinite(nk_interp)
    if np.count_nonzero(valid) < 2:
        return {"pearson_correlation": math.nan, "rmse_ms": math.nan, "samples": int(np.count_nonzero(valid))}

    native_valid = native_interp[valid]
    nk_valid = nk_interp[valid]
    if np.std(native_valid) <= 1e-12 or np.std(nk_valid) <= 1e-12:
        corr = math.nan
    else:
        corr = float(np.corrcoef(native_valid, nk_valid)[0, 1])
    rmse = float(np.sqrt(np.mean((native_valid - nk_valid) ** 2)))
    return {
        "pearson_correlation": corr,
        "rmse_ms": rmse,
        "samples": int(np.count_nonzero(valid)),
    }


def signal_variance_ms2(values_ms: np.ndarray) -> float:
    if len(values_ms) == 0:
        return math.nan
    return float(np.var(values_ms))


def preprocessing_summary_rows(
    rr_ms: np.ndarray,
    native_values_ms: np.ndarray,
    nk_values_ms: np.ndarray,
    native_detrended_ms: np.ndarray,
    nk_detrended_ms: np.ndarray,
    args: argparse.Namespace,
    before_compare: Dict[str, float],
    after_compare: Dict[str, float],
) -> List[Dict[str, Any]]:
    common = {
        "rr_interval_count": int(len(rr_ms)),
        "rr_mean_ms": float(np.mean(rr_ms)) if len(rr_ms) else math.nan,
        "rr_std_ms": float(np.std(rr_ms)) if len(rr_ms) else math.nan,
        "rr_min_ms": float(np.min(rr_ms)) if len(rr_ms) else math.nan,
        "rr_max_ms": float(np.max(rr_ms)) if len(rr_ms) else math.nan,
        "interpolation_rate_hz": args.interpolation_rate,
        "detrending_method": args.detrend_method,
        "pearson_correlation_resampled": before_compare["pearson_correlation"],
        "pearson_correlation_detrended": after_compare["pearson_correlation"],
        "rmse_resampled_ms": before_compare["rmse_ms"],
        "rmse_detrended_ms": after_compare["rmse_ms"],
        "correlation_samples_resampled": before_compare["samples"],
        "correlation_samples_detrended": after_compare["samples"],
    }
    return [
        {
            **common,
            "pipeline": "native_hrv_studio",
            "interpolation_method": "native cubic spline",
            "signal_length": int(len(native_values_ms)),
            "signal_variance_before_detrending_ms2": signal_variance_ms2(native_values_ms),
            "signal_variance_after_detrending_ms2": signal_variance_ms2(native_detrended_ms),
        },
        {
            **common,
            "pipeline": "neurokit2",
            "interpolation_method": args.neurokit_interpolation_method,
            "signal_length": int(len(nk_values_ms)),
            "signal_variance_before_detrending_ms2": signal_variance_ms2(nk_values_ms),
            "signal_variance_after_detrending_ms2": signal_variance_ms2(nk_detrended_ms),
        },
    ]


def positive_psd_limits(*psds: np.ndarray) -> Tuple[float, float]:
    values = np.concatenate([psd[np.isfinite(psd) & (psd > 0)] for psd in psds if len(psd)])
    if len(values) == 0:
        return 1e-6, 1.0
    return max(float(np.nanmin(values)) * 0.5, 1e-12), float(np.nanmax(values)) * 2.0


def plot_psd(
    native_freqs: np.ndarray,
    native_psd: np.ndarray,
    nk_freqs: np.ndarray,
    nk_psd: np.ndarray,
    experimental_freqs: np.ndarray,
    experimental_psd: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    shade_bands(ax)
    if len(native_freqs):
        ax.plot(native_freqs, native_psd, label="HRV Studio native Welch", linewidth=1.7)
    if len(nk_freqs):
        ax.plot(nk_freqs, nk_psd, label="NeuroKit2 Welch", linewidth=1.7)
    if len(experimental_freqs):
        ax.plot(
            experimental_freqs,
            experimental_psd,
            label="Native Welch nfft experiment",
            linewidth=1.2,
            linestyle="--",
        )
    ax.set_xlim(0.0, 0.4)
    y_min, y_max = positive_psd_limits(native_psd, nk_psd, experimental_psd)
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD power (ms^2/Hz)")
    ax.set_title("Native vs NeuroKit2 Welch PSD")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def shade_bands(ax: Any) -> None:
    bands = [("VLF", 0.0, 0.04), ("LF", 0.04, 0.15), ("HF", 0.15, 0.4)]
    colors = ["#e8eef7", "#edf7ed", "#fff1df"]
    for (label, low, high), color in zip(bands, colors):
        ax.axvspan(low, high, color=color, alpha=0.55, zorder=0)
        ax.axvline(low, color="#555555", linewidth=0.8, alpha=0.65)
        ax.text((low + high) / 2, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.axvline(0.4, color="#555555", linewidth=0.8, alpha=0.65)


def integrated_power(freqs: np.ndarray, psd: np.ndarray) -> float:
    if len(freqs) < 2 or len(psd) < 2:
        return math.nan
    mask = (freqs >= 0.0) & (freqs <= 0.4) & np.isfinite(freqs) & np.isfinite(psd)
    if np.count_nonzero(mask) < 2:
        return math.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def variance_ms2(signal_seconds: np.ndarray) -> float:
    if len(signal_seconds) == 0:
        return math.nan
    return float(np.var(signal_seconds) * 1e6)


def psd_ratio(power: float, variance: float) -> float:
    if not finite(power) or not finite(variance) or variance <= 0:
        return math.nan
    return float(power / variance)


def estimate_nfft(freqs: np.ndarray, sampling_rate: float, fallback: Optional[int]) -> Optional[int]:
    if len(freqs) > 1:
        spacing = float(freqs[1] - freqs[0])
        if spacing > 0:
            return int(round(sampling_rate / spacing))
    return fallback


def peak_frequency(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    if len(freqs) == 0 or len(psd) == 0:
        return math.nan
    mask = (freqs >= low) & (freqs <= high) & np.isfinite(psd)
    if not np.any(mask):
        return math.nan
    local_freqs = freqs[mask]
    local_psd = psd[mask]
    return float(local_freqs[int(np.argmax(local_psd))])


def shape_similarity(native_freqs: np.ndarray, native_psd: np.ndarray, nk_freqs: np.ndarray, nk_psd: np.ndarray) -> Dict[str, Any]:
    if len(native_freqs) < 3 or len(nk_freqs) < 3:
        return {"correlation": math.nan, "label": "unclear/manual review needed"}

    low = max(float(np.nanmin(native_freqs)), float(np.nanmin(nk_freqs)), 0.0)
    high = min(float(np.nanmax(native_freqs)), float(np.nanmax(nk_freqs)), 0.4)
    if high <= low:
        return {"correlation": math.nan, "label": "unclear/manual review needed"}

    grid = np.linspace(low, high, 400)
    native_interp = np.interp(grid, native_freqs, np.maximum(native_psd, 1e-12))
    nk_interp = np.interp(grid, nk_freqs, np.maximum(nk_psd, 1e-12))
    native_log = np.log10(native_interp)
    nk_log = np.log10(nk_interp)
    if np.std(native_log) <= 1e-12 or np.std(nk_log) <= 1e-12:
        corr = math.nan
    else:
        corr = float(np.corrcoef(native_log, nk_log)[0, 1])

    native_peaks = {
        band: peak_frequency(native_freqs, native_psd, *limits)
        for band, limits in band_ranges().items()
        if band in {"vlf", "lf", "hf"}
    }
    nk_peaks = {
        band: peak_frequency(nk_freqs, nk_psd, *limits)
        for band, limits in band_ranges().items()
        if band in {"vlf", "lf", "hf"}
    }
    peak_deltas = [
        abs(native_peaks[band] - nk_peaks[band])
        for band in native_peaks
        if finite(native_peaks[band]) and finite(nk_peaks[band])
    ]
    max_peak_delta = max(peak_deltas) if peak_deltas else math.nan

    if finite(corr) and corr >= 0.85 and (not finite(max_peak_delta) or max_peak_delta <= 0.03):
        label = "PSD shapes appear broadly similar"
    elif finite(corr) and corr < 0.6:
        label = "PSD shapes do not appear closely aligned"
    else:
        label = "PSD shape similarity is mixed or unclear"
    return {
        "correlation": corr,
        "label": label,
        "native_peaks": native_peaks,
        "neurokit2_peaks": nk_peaks,
        "max_peak_delta_hz": max_peak_delta,
    }


def likely_disagreement_category(
    duration_s: float,
    native_counts: Dict[str, int],
    nk_counts: Dict[str, int],
    similarity: Dict[str, Any],
    metric_rows: List[Dict[str, Any]],
) -> str:
    low_bin = any(native_counts.get(band, 0) < 2 or nk_counts.get(band, 0) < 2 for band in ("vlf", "lf", "hf"))
    if duration_s < 300.0 or low_bin:
        return "c) low-bin/duration issue"

    abs_rows = [row for row in metric_rows if row["metric"] in {"VLF", "LF", "HF", "total_power"}]
    norm_rows = [row for row in metric_rows if row["metric"] in {"LF/HF", "LF_nu", "HF_nu"}]
    high_abs = any(row["relative_error_pct"] > 50.0 for row in abs_rows if finite(row["relative_error_pct"]))
    low_norm = all(row["relative_error_pct"] <= 20.0 for row in norm_rows if finite(row["relative_error_pct"]))
    corr = similarity.get("correlation", math.nan)
    if high_abs and low_norm and finite(corr) and corr >= 0.75:
        return "a) scaling/amplitude mismatch"
    if finite(corr) and corr < 0.6:
        return "b) peak/frequency-shape mismatch"
    return "d) unclear/manual review needed"


def metric_table_rows(
    native_metrics: Dict[str, float],
    nk_metrics: Dict[str, float],
    experimental_metrics: Dict[str, float],
    experimental_enabled: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, key in METRICS.items():
        native_value = float(native_metrics.get(key, math.nan))
        nk_value = float(nk_metrics.get(key, math.nan))
        experimental_value = float(experimental_metrics.get(key, math.nan))
        row = {
            "metric": label,
            "native_value": native_value,
            "neurokit2_value": nk_value,
            "absolute_error": safe_abs_error(native_value, nk_value),
            "relative_error_pct": relative_error(native_value, nk_value) * 100.0,
            "experimental_native_value": experimental_value if experimental_enabled else math.nan,
            "experimental_relative_error_pct": (
                relative_error(experimental_value, nk_value) * 100.0
                if experimental_enabled
                else math.nan
            ),
        }
        rows.append(row)
    return rows


def diagnostics_row(
    analyzer: HRVFreqDomainAnalysis,
    native_freqs: np.ndarray,
    native_psd: np.ndarray,
    nk_freqs: np.ndarray,
    nk_psd: np.ndarray,
    native_nperseg: Optional[int],
    nk_nperseg: Optional[int],
    experimental_freqs: np.ndarray,
    experimental_psd: np.ndarray,
    experimental_nfft: Optional[int],
) -> Dict[str, Any]:
    native_counts = band_bin_counts(native_freqs)
    nk_counts = band_bin_counts(nk_freqs)
    native_power = integrated_power(native_freqs, native_psd)
    nk_power = integrated_power(nk_freqs, nk_psd)
    input_var = variance_ms2(analyzer.time_domain_s)
    row: Dict[str, Any] = {
        "native_nperseg": native_nperseg,
        "neurokit2_nperseg": nk_nperseg,
        "native_nfft": estimate_nfft(native_freqs, analyzer.sampling_rate, native_nperseg),
        "neurokit2_nfft": estimate_nfft(nk_freqs, analyzer.sampling_rate, None),
        "native_bins_total": native_counts.get("total", 0),
        "neurokit2_bins_total": nk_counts.get("total", 0),
        "native_band_bins_vlf": native_counts.get("vlf", 0),
        "native_band_bins_lf": native_counts.get("lf", 0),
        "native_band_bins_hf": native_counts.get("hf", 0),
        "neurokit2_band_bins_vlf": nk_counts.get("vlf", 0),
        "neurokit2_band_bins_lf": nk_counts.get("lf", 0),
        "neurokit2_band_bins_hf": nk_counts.get("hf", 0),
        "native_integrated_psd_power": native_power,
        "neurokit2_integrated_psd_power": nk_power,
        "input_signal_variance": input_var,
        "native_integrated_psd_power_over_variance_ratio": psd_ratio(native_power, input_var),
        "neurokit2_integrated_psd_power_over_variance_ratio": psd_ratio(nk_power, input_var),
    }
    if experimental_nfft is not None:
        experimental_counts = band_bin_counts(experimental_freqs)
        experimental_power = integrated_power(experimental_freqs, experimental_psd)
        row.update(
            {
                "experimental_native_nfft": experimental_nfft,
                "experimental_native_bins_total": experimental_counts.get("total", 0),
                "experimental_native_band_bins_vlf": experimental_counts.get("vlf", 0),
                "experimental_native_band_bins_lf": experimental_counts.get("lf", 0),
                "experimental_native_band_bins_hf": experimental_counts.get("hf", 0),
                "experimental_native_integrated_psd_power": experimental_power,
                "experimental_native_integrated_psd_power_over_variance_ratio": psd_ratio(
                    experimental_power, input_var
                ),
            }
        )
    return row


def write_summary(
    path: Path,
    input_path: Path,
    duration_s: float,
    warnings_list: List[str],
    metric_rows: List[Dict[str, Any]],
    similarity: Dict[str, Any],
    disagreement_category: str,
    before_compare: Dict[str, float],
    after_compare: Dict[str, float],
) -> None:
    top_discrepancies = sorted(
        [row for row in metric_rows if finite(row["relative_error_pct"])],
        key=lambda row: row["relative_error_pct"],
        reverse=True,
    )[:5]

    lines = [
        f"# Frequency Outlier Inspection: {input_path.name}",
        "",
        "This report is generated for manual validation review only. It does not indicate which implementation is correct.",
        "",
        "## File",
        "",
        f"- Source file: `{input_path}`",
        f"- Duration: {duration_s:.2f} seconds",
        "",
        "## Warning messages",
        "",
    ]
    if warnings_list:
        lines.extend(f"- {warning}" for warning in warnings_list)
    else:
        lines.append("- No warning messages were captured.")

    lines.extend(
        [
            "",
            "## Key discrepancies",
            "",
            "| Metric | Native | NeuroKit2 | Absolute error | Relative error |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in top_discrepancies:
        lines.append(
            "| {metric} | {native:.6g} | {nk:.6g} | {abs_err:.6g} | {rel:.2f}% |".format(
                metric=row["metric"],
                native=row["native_value"],
                nk=row["neurokit2_value"],
                abs_err=row["absolute_error"],
                rel=row["relative_error_pct"],
            )
        )

    corr = similarity.get("correlation", math.nan)
    corr_text = f"{corr:.3f}" if finite(corr) else "not available"
    lines.extend(
        [
            "",
            "## Preprocessing signal comparison",
            "",
            (
                f"- Resampled-signal Pearson correlation: "
                f"{format_optional_float(before_compare['pearson_correlation'], 3)}."
            ),
            f"- Resampled-signal RMSE: {format_optional_float(before_compare['rmse_ms'], 3)} ms.",
            (
                f"- Detrended-signal Pearson correlation: "
                f"{format_optional_float(after_compare['pearson_correlation'], 3)}."
            ),
            f"- Detrended-signal RMSE: {format_optional_float(after_compare['rmse_ms'], 3)} ms.",
            (
                "- Low correlation or high RMSE before PSD estimation may suggest that the "
                "disagreement originates in interpolation, detrending, clipping, units, or other "
                "pre-PSD signal preparation differences."
            ),
            "",
            "## PSD shape assessment",
            "",
            f"- Shape assessment: {similarity.get('label', 'unclear/manual review needed')}.",
            f"- Log-PSD correlation on common 0-0.4 Hz grid: {corr_text}.",
            (
                "- This automated assessment is heuristic; the overlay plot requires manual review "
                "before drawing conclusions."
            ),
            "",
            "## Suggested disagreement category",
            "",
            f"- {disagreement_category}",
            "",
            (
                "Interpretation should remain cautious. The disagreement may suggest methodological "
                "mismatch, amplitude/scaling differences, frequency-shape differences, low-bin/duration "
                "limitations, or data-specific issues."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional_float(value: Any, digits: int) -> str:
    if not finite(value):
        return "not available"
    return f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    output_dir = resolve_output_dir(args, input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    captured_warning_texts: List[str] = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        rr_ms = load_rr_intervals_ms(input_path)
        analyzer = HRVFreqDomainAnalysis(
            rr_ms,
            sampling_rate=args.interpolation_rate,
            detrend_method=normalize_detrend_method(args.detrend_method),
            detrend_lambda=args.detrend_lambda,
            window_type=args.window_type,
            segment_length=args.segment_length,
            overlap_ratio=args.overlap_ratio,
            ar_order=args.ar_order,
            enable_diagnostics=args.enable_diagnostics,
        )
        native_results = analyzer.get_results()
        captured_warning_texts = dedupe(str(item.message) for item in captured)

    native_nperseg, native_noverlap = effective_welch_params(analyzer)
    native_time_s, native_resampled_ms = native_resampled_rr_ms(analyzer)
    nk_time_s, nk_resampled_ms = neurokit2_resampled_rr_ms(
        analyzer.rr_intervals_ms,
        args.interpolation_rate,
        args.neurokit_interpolation_method,
    )
    native_detrended_ms = detrend_signal_ms(
        native_resampled_ms,
        normalize_detrend_method(args.detrend_method),
        args.detrend_lambda,
    )
    nk_detrended_ms = detrend_signal_ms(
        nk_resampled_ms,
        normalize_detrend_method(args.detrend_method),
        args.detrend_lambda,
    )
    before_preprocessing_compare = common_grid_compare(
        native_time_s,
        native_resampled_ms,
        nk_time_s,
        nk_resampled_ms,
        args.interpolation_rate,
    )
    after_preprocessing_compare = common_grid_compare(
        native_time_s,
        native_detrended_ms,
        nk_time_s,
        nk_detrended_ms,
        args.interpolation_rate,
    )

    if native_nperseg is None or native_noverlap is None:
        nk_freqs = np.array([])
        nk_psd = np.array([])
        nk_nperseg = None
    else:
        nk_freqs, nk_psd, nk_nperseg, _nk_noverlap, _nk_duration, _nk_adjustment = neurokit2_welch_psd(
            path=input_path,
            rr_ms=analyzer.rr_intervals_ms,
            sampling_rate=args.interpolation_rate,
            interpolation_method=args.neurokit_interpolation_method,
            window_type=args.window_type,
            nperseg=native_nperseg,
            noverlap=native_noverlap,
            detrend_method=normalize_detrend_method(args.detrend_method),
            detrend_lambda=args.detrend_lambda,
        )

    experimental_freqs, experimental_psd, experimental_nfft = experimental_native_welch_psd(
        analyzer,
        native_nperseg,
        native_noverlap,
        args.experimental_native_welch_nfft_multiplier,
    )

    native_metrics = {
        key: float(native_results.get(f"welch_{key}", native_results.get(key, math.nan)))
        for key in METRICS.values()
    }
    nk_metrics = metrics_from_psd(nk_freqs, nk_psd)
    experimental_metrics = metrics_from_psd(experimental_freqs, experimental_psd)
    experimental_enabled = experimental_nfft is not None
    metric_rows = metric_table_rows(
        native_metrics,
        nk_metrics,
        experimental_metrics,
        experimental_enabled,
    )

    diagnostics = native_results.get("frequency_diagnostics", {}) if args.enable_diagnostics else {}
    warnings_list = dedupe(captured_warning_texts + collect_warnings(diagnostics))

    duration_s = float(rr_start_times_s(analyzer.rr_intervals_ms)[-1]) if len(analyzer.rr_intervals_ms) else 0.0
    native_counts = band_bin_counts(analyzer.freqs)
    nk_counts = band_bin_counts(nk_freqs)
    similarity = shape_similarity(analyzer.freqs, analyzer.psd, nk_freqs, nk_psd)
    disagreement_category = likely_disagreement_category(
        duration_s,
        native_counts,
        nk_counts,
        similarity,
        metric_rows,
    )

    plot_rr_tachogram(analyzer.rr_intervals_ms, input_path, output_dir / "rr_tachogram.png")
    plot_signal_overlay(
        native_time_s,
        native_resampled_ms,
        nk_time_s,
        nk_resampled_ms,
        output_dir / "interpolated_rr_native_vs_neurokit2.png",
        "Interpolated RR entering Welch preparation",
        "RR interval (ms)",
    )
    plot_signal_overlay(
        native_time_s,
        native_detrended_ms,
        nk_time_s,
        nk_detrended_ms,
        output_dir / "detrended_rr_native_vs_neurokit2.png",
        "Detrended RR diagnostic comparison",
        "Detrended RR interval (ms)",
    )
    plot_psd(
        analyzer.freqs,
        analyzer.psd,
        nk_freqs,
        nk_psd,
        experimental_freqs,
        experimental_psd,
        output_dir / "psd_native_vs_neurokit2.png",
    )

    write_csv(
        output_dir / "metric_table.csv",
        metric_rows,
        [
            "metric",
            "native_value",
            "neurokit2_value",
            "absolute_error",
            "relative_error_pct",
            "experimental_native_value",
            "experimental_relative_error_pct",
        ],
    )
    write_csv(
        output_dir / "preprocessing_summary.csv",
        preprocessing_summary_rows(
            analyzer.rr_intervals_ms,
            native_resampled_ms,
            nk_resampled_ms,
            native_detrended_ms,
            nk_detrended_ms,
            args,
            before_preprocessing_compare,
            after_preprocessing_compare,
        ),
        [
            "pipeline",
            "rr_interval_count",
            "rr_mean_ms",
            "rr_std_ms",
            "rr_min_ms",
            "rr_max_ms",
            "interpolation_method",
            "interpolation_rate_hz",
            "signal_length",
            "detrending_method",
            "signal_variance_before_detrending_ms2",
            "signal_variance_after_detrending_ms2",
            "pearson_correlation_resampled",
            "pearson_correlation_detrended",
            "rmse_resampled_ms",
            "rmse_detrended_ms",
            "correlation_samples_resampled",
            "correlation_samples_detrended",
        ],
    )
    write_csv(
        output_dir / "psd_diagnostics.csv",
        [
            diagnostics_row(
                analyzer,
                analyzer.freqs,
                analyzer.psd,
                nk_freqs,
                nk_psd,
                native_nperseg,
                nk_nperseg,
                experimental_freqs,
                experimental_psd,
                experimental_nfft,
            )
        ],
        [
            "native_nperseg",
            "neurokit2_nperseg",
            "native_nfft",
            "neurokit2_nfft",
            "native_bins_total",
            "neurokit2_bins_total",
            "native_band_bins_vlf",
            "native_band_bins_lf",
            "native_band_bins_hf",
            "neurokit2_band_bins_vlf",
            "neurokit2_band_bins_lf",
            "neurokit2_band_bins_hf",
            "native_integrated_psd_power",
            "neurokit2_integrated_psd_power",
            "input_signal_variance",
            "native_integrated_psd_power_over_variance_ratio",
            "neurokit2_integrated_psd_power_over_variance_ratio",
            "experimental_native_nfft",
            "experimental_native_bins_total",
            "experimental_native_band_bins_vlf",
            "experimental_native_band_bins_lf",
            "experimental_native_band_bins_hf",
            "experimental_native_integrated_psd_power",
            "experimental_native_integrated_psd_power_over_variance_ratio",
        ],
    )
    write_summary(
        output_dir / "inspection_summary.md",
        input_path,
        duration_s,
        warnings_list,
        metric_rows,
        similarity,
        disagreement_category,
        before_preprocessing_compare,
        after_preprocessing_compare,
    )

    print(f"Wrote manual inspection outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
