"""
Focused PSD-convention audit for remaining Kubios agreement outliers.

Validation-only helper. It recomputes diagnostic Welch PSD variants for a small
manual-review subset and writes research artifacts only. It does not modify
production HRV Studio code.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

try:
    from scipy.integrate import simpson
except Exception:  # pragma: no cover - depends on scipy version
    simpson = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from validate_freq_domain_neurokit2 import (
    effective_welch_params,
    load_rr_intervals_ms,
    rr_start_times_s,
    safe_welch_params_for_signal,
)


REQUESTED_CLEANED_CSV = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "parsed_results_50_none_120s_75pct_after_arm_a"
    / "cleaned_valid_only.csv"
)
FALLBACK_CLEANED_CSV = (
    PROJECT_ROOT
    / "validation"
    / "research_notes"
    / "manual_review_sensitivity_analysis"
    / "cleaned_valid_only.csv"
)
DEFAULT_COMPARISON_CSV = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "parsed_results_50_none_120s_75pct_after_arm_a"
    / "comparison_with_hrvstudio.csv"
)
DEFAULT_VALIDATION_CSV = (
    PROJECT_ROOT
    / "validation"
    / "runs"
    / "v07_kubios_subset_50_none_120_75_after_arm_a"
    / "freq_domain_neurokit2_validation.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "validation" / "research_notes" / "psd_convention_audit"

TARGET_SUBSETS = ("OUT006", "OUT003", "OUT001", "CH001", "CH005")
METRICS = ("VLF", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu")
BANDS = {
    "VLF": (0.0, 0.04),
    "LF": (0.04, 0.15),
    "HF": (0.15, 0.40),
    "total_power": (0.0, 0.40),
}


@dataclass(frozen=True)
class Variant:
    name: str
    integration: str = "trapz"
    include_dc: bool = True
    vlf_low: float = 0.0
    interval_mode: str = "closed"
    boundary_interp: bool = False
    nfft_mode: str = "native"
    scale_mode: str = "none"
    one_sided_mode: str = "scipy_default"
    average: str = "mean"
    total_mode: str = "band_0_0_40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused PSD convention audit.")
    parser.add_argument("--cleaned-csv", default=str(REQUESTED_CLEANED_CSV))
    parser.add_argument("--comparison-csv", default=str(DEFAULT_COMPARISON_CSV))
    parser.add_argument("--validation-csv", default=str(DEFAULT_VALIDATION_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true", help="Allow overwriting this audit folder.")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_cleaned_csv(requested: Path) -> Tuple[Path, str]:
    if requested.exists():
        return requested, "requested"
    if requested == REQUESTED_CLEANED_CSV and FALLBACK_CLEANED_CSV.exists():
        return FALLBACK_CLEANED_CSV, "fallback_manual_review_sensitivity_analysis"
    raise SystemExit(f"Missing cleaned comparison CSV: {requested}")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_cell(row.get(column)) for column in columns})


def csv_cell(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return "" if value is None else value


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def fmt(value: Any, digits: int = 2) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def rel_error_pct(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference) or reference == 0:
        return math.nan
    return abs(value - reference) / abs(reference) * 100.0


def next_power_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def variant_list() -> List[Variant]:
    variants = [
        Variant("baseline_trapz_closed_dc"),
        Variant("rectangular_bin_sum", integration="rectangular"),
        Variant("simpson", integration="simpson"),
        Variant("exclude_dc", include_dc=False),
        Variant("vlf_low_0_0033", vlf_low=0.0033),
        Variant("half_open_bands", interval_mode="half_open"),
        Variant("boundary_interpolated_trapz", boundary_interp=True),
        Variant("nfft_2x", nfft_mode="2x"),
        Variant("nfft_nextpow2", nfft_mode="nextpow2"),
        Variant("hann_energy_multiply", scale_mode="hann_mean_square"),
        Variant("hann_energy_divide", scale_mode="divide_hann_mean_square"),
        Variant("hann_coherent_multiply", scale_mode="hann_coherent_gain_sq"),
        Variant("hann_coherent_divide", scale_mode="divide_hann_coherent_gain_sq"),
        Variant("one_sided_half_positive", one_sided_mode="half_positive"),
        Variant("one_sided_double_positive", one_sided_mode="double_positive"),
        Variant("welch_median_average", average="median"),
        Variant("total_0_0033_0_40", total_mode="total_low_0_0033"),
        Variant("total_vlf_lf_hf_sum", total_mode="sum_vlf_lf_hf"),
        Variant(
            "combined_grid_boundary_no_dc",
            include_dc=False,
            vlf_low=0.0033,
            boundary_interp=True,
            nfft_mode="2x",
            total_mode="total_low_0_0033",
        ),
    ]
    if simpson is None:
        variants = [variant for variant in variants if variant.integration != "simpson"]
    return variants


def target_rows(cleaned_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    for row in cleaned_rows:
        subset_id = row.get("subset_id", "")
        if subset_id in TARGET_SUBSETS and subset_id not in rows:
            rows[subset_id] = row
    missing = [subset_id for subset_id in TARGET_SUBSETS if subset_id not in rows]
    if missing:
        raise SystemExit(f"Missing target subset(s) in cleaned CSV: {', '.join(missing)}")
    return rows


def kubios_values_for_targets(cleaned_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = {subset_id: {} for subset_id in TARGET_SUBSETS}
    for row in cleaned_rows:
        subset_id = row.get("subset_id", "")
        metric = row.get("metric", "")
        if subset_id in values and metric in METRICS:
            values[subset_id][metric] = finite_float(row.get("kubios_value"))
    return values


def clean_signal_ms(signal_ms: np.ndarray) -> np.ndarray:
    values = np.asarray(signal_ms, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    return values - float(np.mean(values))


def neurokit_resampled_ms(rr_ms: np.ndarray, sampling_rate: float = 4.0) -> np.ndarray:
    import neurokit2 as nk

    intervals, _intervals_time, _rate = nk.intervals_process(
        rr_ms,
        intervals_time=rr_start_times_s(rr_ms),
        interpolate=True,
        interpolation_rate=sampling_rate,
        method="cubic",
    )
    return np.asarray(intervals, dtype=float)


def nfft_for_mode(mode: str, nperseg: int) -> int:
    if mode == "2x":
        return max(nperseg, nperseg * 2)
    if mode == "nextpow2":
        return max(nperseg, next_power_two(nperseg))
    return nperseg


def adjusted_psd(freqs: np.ndarray, psd: np.ndarray, variant: Variant, fs: float) -> np.ndarray:
    out = np.asarray(psd, dtype=float).copy()
    if variant.one_sided_mode == "half_positive":
        mask = (freqs > 0) & (freqs < fs / 2)
        out[mask] *= 0.5
    elif variant.one_sided_mode == "double_positive":
        mask = (freqs > 0) & (freqs < fs / 2)
        out[mask] *= 2.0

    if variant.scale_mode != "none":
        window = signal.get_window("hann", len(out) if len(out) > 1 else 2)
        mean_square = float(np.mean(window**2))
        coherent_gain_sq = float(np.mean(window) ** 2)
        factors = {
            "hann_mean_square": mean_square,
            "divide_hann_mean_square": 1.0 / mean_square if mean_square else math.nan,
            "hann_coherent_gain_sq": coherent_gain_sq,
            "divide_hann_coherent_gain_sq": 1.0 / coherent_gain_sq if coherent_gain_sq else math.nan,
        }
        factor = factors.get(variant.scale_mode, 1.0)
        if math.isfinite(factor):
            out *= factor
    return out


def band_mask(freqs: np.ndarray, low: float, high: float, variant: Variant) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high) if variant.interval_mode == "closed" else (freqs >= low) & (freqs < high)
    if not variant.include_dc:
        mask &= freqs > 0.0
    return mask & np.isfinite(freqs)


def integrate_band(freqs: np.ndarray, psd: np.ndarray, low: float, high: float, variant: Variant) -> float:
    if len(freqs) < 2 or len(psd) < 2:
        return math.nan
    if variant.boundary_interp:
        finite = np.isfinite(freqs) & np.isfinite(psd)
        x = freqs[finite]
        y = psd[finite]
        if len(x) < 2:
            return math.nan
        lo = max(low, float(x[0]))
        hi = min(high, float(x[-1]))
        if not variant.include_dc:
            lo = max(lo, np.nextafter(0.0, 1.0))
        if hi <= lo:
            return 0.0
        inner = (x > lo) & (x < hi)
        x_band = np.concatenate([[lo], x[inner], [hi]])
        y_band = np.interp(x_band, x, y)
    else:
        mask = band_mask(freqs, low, high, variant) & np.isfinite(psd)
        x_band = freqs[mask]
        y_band = psd[mask]

    if len(x_band) == 0:
        return 0.0
    if variant.integration == "rectangular":
        spacing = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 0.0
        return float(max(0.0, np.sum(y_band) * spacing))
    if len(x_band) < 2:
        return 0.0
    if variant.integration == "simpson" and simpson is not None and len(x_band) >= 3:
        return float(max(0.0, simpson(y_band, x=x_band)))
    return float(max(0.0, np.trapezoid(y_band, x_band)))


def metrics_from_variant(freqs: np.ndarray, psd: np.ndarray, variant: Variant) -> Dict[str, float]:
    vlf_low = variant.vlf_low
    total_low = 0.0033 if variant.total_mode == "total_low_0_0033" else 0.0
    vlf = integrate_band(freqs, psd, vlf_low, 0.04, variant)
    lf = integrate_band(freqs, psd, 0.04, 0.15, variant)
    hf = integrate_band(freqs, psd, 0.15, 0.40, variant)
    if variant.total_mode == "sum_vlf_lf_hf":
        total = vlf + lf + hf
    else:
        total = integrate_band(freqs, psd, total_low, 0.40, variant)
    lf_hf = lf / hf if math.isfinite(lf) and math.isfinite(hf) and hf > 0 else math.nan
    denom = lf + hf
    lf_nu = lf / denom * 100.0 if denom > 0 else math.nan
    hf_nu = hf / denom * 100.0 if denom > 0 else math.nan
    return {
        "VLF": vlf,
        "LF": lf,
        "HF": hf,
        "total_power": total,
        "LF/HF": lf_hf,
        "LF_nu": lf_nu,
        "HF_nu": hf_nu,
    }


def compute_psd(signal_ms: np.ndarray, fs: float, nperseg: int, noverlap: int, variant: Variant) -> Tuple[np.ndarray, np.ndarray, int]:
    nperseg = max(2, min(nperseg, len(signal_ms)))
    noverlap = max(0, min(noverlap, nperseg - 1))
    nfft = nfft_for_mode(variant.nfft_mode, nperseg)
    freqs, psd = signal.welch(
        signal_ms,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
        average=variant.average,
    )
    return freqs, adjusted_psd(freqs, psd, variant, fs), nfft


def psd_integral(freqs: np.ndarray, psd: np.ndarray) -> float:
    mask = np.isfinite(freqs) & np.isfinite(psd) & (freqs >= 0.0) & (freqs <= 0.40)
    if np.count_nonzero(mask) < 2:
        return math.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def overall_error(errors: Dict[str, float]) -> float:
    finite = [value for value in errors.values() if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def analyze_file(
    subset_id: str,
    input_path: Path,
    kubios: Dict[str, float],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fs = 4.0
    rr_raw = load_rr_intervals_ms(input_path)
    analyzer = HRVFreqDomainAnalysis(
        rr_raw,
        sampling_rate=fs,
        detrend_method=None,
        window_type="hann",
        segment_length=120.0,
        overlap_ratio=0.75,
        enable_diagnostics=True,
    )
    analyzer.get_results()
    native_nperseg, native_noverlap = effective_welch_params(analyzer)
    if native_nperseg is None or native_noverlap is None:
        raise RuntimeError(f"Cannot determine native Welch parameters for {input_path}")

    native_signal = clean_signal_ms(analyzer.time_domain_s * 1000.0)
    nk_signal = clean_signal_ms(neurokit_resampled_ms(analyzer.rr_intervals_ms, fs))
    requested_nperseg = int(120.0 * fs)
    requested_noverlap = int(requested_nperseg * 0.75)
    nk_nperseg, nk_noverlap, nk_adjustment = safe_welch_params_for_signal(
        path=input_path,
        n_samples=len(nk_signal),
        requested_nperseg=requested_nperseg,
        requested_noverlap=requested_noverlap,
    )

    signals = {
        "native_resampled": (native_signal, int(native_nperseg), int(native_noverlap)),
        "neurokit2_resampled": (nk_signal, int(nk_nperseg), int(nk_noverlap)),
    }
    rows: List[Dict[str, Any]] = []
    psd_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    variants = variant_list()

    for signal_name, (signal_values, nperseg, noverlap) in signals.items():
        variance = float(np.var(signal_values)) if len(signal_values) else math.nan
        for variant in variants:
            try:
                freqs, psd, nfft = compute_psd(signal_values, fs, nperseg, noverlap, variant)
            except Exception as exc:
                freqs = np.array([])
                psd = np.array([])
                nfft = math.nan  # type: ignore[assignment]
                metric_values = {metric: math.nan for metric in METRICS}
                errors = {metric: math.nan for metric in METRICS}
                warning = str(exc)
            else:
                metric_values = metrics_from_variant(freqs, psd, variant)
                errors = {
                    metric: rel_error_pct(metric_values.get(metric, math.nan), kubios.get(metric, math.nan))
                    for metric in METRICS
                }
                warning = ""
                psd_cache[(signal_name, variant.name)] = (freqs, psd)

            row: Dict[str, Any] = {
                "subset_id": subset_id,
                "source_file": str(input_path),
                "signal_source": signal_name,
                "variant": variant.name,
                "integration": variant.integration,
                "include_dc": variant.include_dc,
                "vlf_low_hz": variant.vlf_low,
                "interval_mode": variant.interval_mode,
                "boundary_interp": variant.boundary_interp,
                "nfft_mode": variant.nfft_mode,
                "nfft": nfft,
                "scale_mode": variant.scale_mode,
                "one_sided_mode": variant.one_sided_mode,
                "average": variant.average,
                "total_mode": variant.total_mode,
                "nperseg": nperseg,
                "noverlap": noverlap,
                "signal_samples": len(signal_values),
                "signal_variance_ms2": variance,
                "psd_integral_0_0_4": psd_integral(freqs, psd),
                "psd_integral_to_variance_ratio": psd_integral(freqs, psd) / variance if variance > 0 else math.nan,
                "overall_relative_error_pct": overall_error(errors),
                "warning": warning,
            }
            for metric in METRICS:
                row[f"{metric}_value"] = metric_values.get(metric, math.nan)
                row[f"{metric}_kubios"] = kubios.get(metric, math.nan)
                row[f"{metric}_relative_error_pct"] = errors.get(metric, math.nan)
            rows.append(row)

    best = min(rows, key=lambda row: finite_float(row["overall_relative_error_pct"]))
    baseline_native = psd_cache.get(("native_resampled", "baseline_trapz_closed_dc"))
    baseline_nk = psd_cache.get(("neurokit2_resampled", "baseline_trapz_closed_dc"))
    best_psd = psd_cache.get((str(best["signal_source"]), str(best["variant"])))
    plot_file_outputs(output_dir / subset_id, subset_id, baseline_native, baseline_nk, best_psd, best, rows)

    metadata = {
        "subset_id": subset_id,
        "input_path": str(input_path),
        "rr_count_raw": int(len(rr_raw)),
        "rr_count_after_cleanup": int(len(analyzer.rr_intervals_ms)),
        "native_samples": int(len(native_signal)),
        "neurokit2_samples": int(len(nk_signal)),
        "native_nperseg": int(native_nperseg),
        "native_noverlap": int(native_noverlap),
        "neurokit2_nperseg": int(nk_nperseg),
        "neurokit2_noverlap": int(nk_noverlap),
        "neurokit2_adjusted": bool(nk_adjustment),
        "best_signal_source": best["signal_source"],
        "best_variant": best["variant"],
        "best_overall_error_pct": best["overall_relative_error_pct"],
        "baseline_native_error_pct": next(
            row["overall_relative_error_pct"]
            for row in rows
            if row["signal_source"] == "native_resampled" and row["variant"] == "baseline_trapz_closed_dc"
        ),
    }
    return rows, metadata


def plot_file_outputs(
    folder: Path,
    subset_id: str,
    baseline_native: Optional[Tuple[np.ndarray, np.ndarray]],
    baseline_nk: Optional[Tuple[np.ndarray, np.ndarray]],
    best_psd: Optional[Tuple[np.ndarray, np.ndarray]],
    best: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if baseline_native and baseline_nk:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(baseline_native[0], baseline_native[1], label="Native baseline", linewidth=1.5)
        ax.plot(baseline_nk[0], baseline_nk[1], label="NeuroKit2-style baseline", linewidth=1.5)
        ax.set_xlim(0, 0.40)
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (ms^2/Hz)")
        ax.set_title(f"{subset_id} baseline PSD overlay")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "baseline_psd_overlay.png", dpi=160)
        plt.close(fig)

    if baseline_native and best_psd:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(baseline_native[0], baseline_native[1], label="Native baseline", linewidth=1.5)
        label = f"Best: {best['signal_source']} / {best['variant']}"
        ax.plot(best_psd[0], best_psd[1], label=label, linewidth=1.5)
        ax.set_xlim(0, 0.40)
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (ms^2/Hz)")
        ax.set_title(f"{subset_id} best variant PSD overlay")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "best_variant_psd_overlay.png", dpi=160)
        plt.close(fig)

    if baseline_native:
        freqs, psd = baseline_native
        mask = (freqs >= 0) & (freqs <= 0.4) & np.isfinite(psd)
        x = freqs[mask]
        y = psd[mask]
        cumulative = np.zeros_like(x)
        if len(x) > 1:
            cumulative[1:] = np.cumsum((y[:-1] + y[1:]) * 0.5 * np.diff(x))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, cumulative, linewidth=1.7)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Cumulative power (ms^2)")
        ax.set_title(f"{subset_id} baseline cumulative power")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(folder / "cumulative_power_by_frequency.png", dpi=160)
        plt.close(fig)

    top_rows = sorted(rows, key=lambda row: finite_float(row["overall_relative_error_pct"]))[:6]
    labels = [f"{row['signal_source']}\n{row['variant']}" for row in top_rows]
    values = [finite_float(row["overall_relative_error_pct"]) for row in top_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean relative error vs Kubios (%)")
    ax.set_title(f"{subset_id} best band-integration variants")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "band_integration_comparison.png", dpi=160)
    plt.close(fig)


def best_by_file(rows: Sequence[Dict[str, Any]], metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for subset_id in TARGET_SUBSETS:
        file_rows = [row for row in rows if row["subset_id"] == subset_id]
        best = min(file_rows, key=lambda row: finite_float(row["overall_relative_error_pct"]))
        base = next(
            row
            for row in file_rows
            if row["signal_source"] == "native_resampled" and row["variant"] == "baseline_trapz_closed_dc"
        )
        out.append(
            {
                "subset_id": subset_id,
                "best_signal_source": best["signal_source"],
                "best_variant": best["variant"],
                "best_overall_relative_error_pct": best["overall_relative_error_pct"],
                "native_baseline_overall_relative_error_pct": base["overall_relative_error_pct"],
                "absolute_change_pct_points": finite_float(best["overall_relative_error_pct"]) - finite_float(base["overall_relative_error_pct"]),
                "percent_improvement": (
                    (finite_float(base["overall_relative_error_pct"]) - finite_float(best["overall_relative_error_pct"]))
                    / finite_float(base["overall_relative_error_pct"])
                    * 100.0
                    if finite_float(base["overall_relative_error_pct"]) > 0
                    else math.nan
                ),
                **metadata[subset_id],
            }
        )
    return out


def best_by_metric(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    variant_keys = sorted({(row["signal_source"], row["variant"]) for row in rows})
    for metric in METRICS:
        candidates = []
        for signal_source, variant in variant_keys:
            values = [
                finite_float(row.get(f"{metric}_relative_error_pct"))
                for row in rows
                if row["signal_source"] == signal_source and row["variant"] == variant
            ]
            finite = [value for value in values if math.isfinite(value)]
            if finite:
                candidates.append(
                    {
                        "metric": metric,
                        "signal_source": signal_source,
                        "variant": variant,
                        "mean_relative_error_pct": mean(finite),
                        "median_relative_error_pct": median(finite),
                        "files": len(finite),
                    }
                )
        candidates.sort(key=lambda row: row["mean_relative_error_pct"])
        for rank, row in enumerate(candidates[:5], start=1):
            row["rank"] = rank
            out.append(row)
    return out


def aggregate_variant_ranking(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, float, int]]:
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        value = finite_float(row["overall_relative_error_pct"])
        if math.isfinite(value):
            grouped.setdefault((row["signal_source"], row["variant"]), []).append(value)
    ranked = [
        (signal_source, variant, mean(values), len(values))
        for (signal_source, variant), values in grouped.items()
    ]
    ranked.sort(key=lambda item: item[2])
    return ranked


def write_summary(
    path: Path,
    cleaned_csv: Path,
    cleaned_source: str,
    all_rows: Sequence[Dict[str, Any]],
    by_file: Sequence[Dict[str, Any]],
    by_metric: Sequence[Dict[str, Any]],
) -> None:
    aggregate = aggregate_variant_ranking(all_rows)
    top_aggregate = aggregate[:10]
    baseline_errors = [
        row["native_baseline_overall_relative_error_pct"] for row in by_file
    ]
    best_errors = [row["best_overall_relative_error_pct"] for row in by_file]
    repeated_best = {}
    for row in by_file:
        key = f"{row['best_signal_source']} / {row['best_variant']}"
        repeated_best[key] = repeated_best.get(key, 0) + 1

    file_table = [
        [
            row["subset_id"],
            row["best_signal_source"],
            row["best_variant"],
            fmt(row["native_baseline_overall_relative_error_pct"]),
            fmt(row["best_overall_relative_error_pct"]),
            fmt(row["percent_improvement"]),
        ]
        for row in by_file
    ]
    aggregate_table = [
        [signal_source, variant, fmt(error), count]
        for signal_source, variant, error, count in top_aggregate
    ]
    metric_table = [
        [
            row["metric"],
            row["rank"],
            row["signal_source"],
            row["variant"],
            fmt(row["mean_relative_error_pct"]),
            fmt(row["median_relative_error_pct"]),
        ]
        for row in by_metric
        if int(row["rank"]) <= 3
    ]

    lines = [
        "# PSD Convention Audit Summary",
        "",
        "This validation-only audit recomputed diagnostic Welch PSD variants for the remaining worst Kubios agreement outliers. It does not establish Kubios as ground truth and does not change production behavior.",
        "",
        "## Inputs",
        "",
        f"- Cleaned comparison CSV used: `{cleaned_csv}`",
        f"- Cleaned CSV source: `{cleaned_source}`",
        "- Target files: OUT006, OUT003, OUT001, CH001, CH005",
        "- Settings: no detrending convention with one global mean removed, 4 Hz interpolation, 120 s segment length, 75% overlap, Hann window, VLF 0-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz.",
        "",
        "## Best Variant By File",
        "",
        *markdown_table(
            ["Subset", "Signal source", "Best variant", "Native baseline mean err %", "Best mean err %", "Improvement %"],
            file_table,
        ),
        "",
        "## Best Aggregate Variants",
        "",
        *markdown_table(["Signal source", "Variant", "Mean file-level err %", "Files"], aggregate_table),
        "",
        "## Best Variants By Metric",
        "",
        *markdown_table(["Metric", "Rank", "Signal source", "Variant", "Mean err %", "Median err %"], metric_table),
        "",
        "## Interpretation",
        "",
        f"- Native baseline mean file-level error across the five files was {fmt(mean(baseline_errors))}%; the per-file best diagnostic variants averaged {fmt(mean(best_errors))}%. This shows some convention sensitivity, but the best setting is selected post hoc per file.",
        f"- Best variants repeated by file: {', '.join(f'{key}: {count}' for key, count in sorted(repeated_best.items()))}.",
        "- No single PSD convention should be treated as a production fix unless it improves most files and most metrics without relying on file-specific tuning.",
        "- If the best aggregate variants are mostly nfft/grid or boundary variants, the residual mismatch is more likely a convention/documentation issue. If window or one-sided scaling variants dominate, that would suggest a scale convention, but those variants are diagnostic only.",
        "- Mismatches should be read cautiously: OUT006 remains a retained pathological edge case, while CH001/CH005/OUT001/OUT003 may still reflect Kubios selection, interpolation, or band-grid differences.",
        "- Recommended next step is documentation or a validation-only Kubios-compatible mode rather than changing production code, unless the same convention wins consistently on a larger curated set.",
        "",
        "## Per-File Plots",
        "",
    ]
    for row in by_file:
        folder = f"validation/research_notes/psd_convention_audit/{row['subset_id']}"
        lines.extend(
            [
                f"- `{row['subset_id']}`:",
                f"  - `{folder}/baseline_psd_overlay.png`",
                f"  - `{folder}/best_variant_psd_overlay.png`",
                f"  - `{folder}/cumulative_power_by_frequency.png`",
                f"  - `{folder}/band_integration_comparison.png`",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def ensure_no_overwrite(output_dir: Path, force: bool) -> None:
    targets = [
        output_dir / "psd_convention_audit_summary.md",
        output_dir / "variant_results_all.csv",
        output_dir / "best_variant_by_file.csv",
        output_dir / "best_variant_by_metric.csv",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise SystemExit(
            "Refusing to overwrite existing PSD audit outputs. Re-run with --force if intended:\n"
            + "\n".join(str(path) for path in existing)
        )


def main() -> int:
    args = parse_args()
    cleaned_requested = project_path(args.cleaned_csv)
    cleaned_csv, cleaned_source = resolve_cleaned_csv(cleaned_requested)
    output_dir = project_path(args.output_dir)
    ensure_no_overwrite(output_dir, args.force)

    cleaned_rows = read_csv(cleaned_csv)
    targets = target_rows(cleaned_rows)
    kubios_values = kubios_values_for_targets(cleaned_rows)

    all_rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset_id in TARGET_SUBSETS:
        input_path = project_path(targets[subset_id]["kubios_input_txt"])
        rows, meta = analyze_file(subset_id, input_path, kubios_values[subset_id], output_dir)
        all_rows.extend(rows)
        metadata[subset_id] = meta

    by_file = best_by_file(all_rows, metadata)
    by_metric = best_by_metric(all_rows)
    write_csv(output_dir / "variant_results_all.csv", all_rows, list(all_rows[0].keys()))
    write_csv(output_dir / "best_variant_by_file.csv", by_file, list(by_file[0].keys()))
    write_csv(output_dir / "best_variant_by_metric.csv", by_metric, list(by_metric[0].keys()))
    write_summary(
        output_dir / "psd_convention_audit_summary.md",
        cleaned_csv,
        cleaned_source,
        all_rows,
        by_file,
        by_metric,
    )

    print(f"Targets audited: {', '.join(TARGET_SUBSETS)}")
    print(f"Cleaned CSV used: {cleaned_csv} ({cleaned_source})")
    print(f"Variant rows: {len(all_rows)}")
    print(f"Wrote outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
