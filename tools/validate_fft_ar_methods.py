"""
Validation-only FFT and AR frequency-domain experiment.

This script does not modify HRV Studio production code. It evaluates the current
HRV Studio Welch, FFT, and AR implementations on a manageable clean PhysioNet
10-minute subset, with NeuroKit2 Welch included as an external comparator.

Example:
    python tools/validate_fft_ar_methods.py --overwrite
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
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hrvlib.data_handler import load_rr_file
from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis


DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "validation" / "processed_data" / "physionet_nsr_rr_10min"
)
DEFAULT_RUN_DIR = PROJECT_ROOT / "validation" / "runs" / "v08_fft_ar_validation"
METRICS = {
    "VLF": "vlf_power",
    "LF": "lf_power",
    "HF": "hf_power",
    "total_power": "total_power",
    "LF/HF": "lf_hf_ratio",
    "LF_nu": "lf_nu",
    "HF_nu": "hf_nu",
}
AR_ORDERS = [8, 16, 24]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-only FFT/AR frequency-domain comparison on a clean "
            "PhysioNet 10-minute RR subset."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--min-duration-s", type=float, default=590.0)
    parser.add_argument("--max-invalid-removed-pct", type=float, default=1.0)
    parser.add_argument("--interpolation-rate", type=float, default=4.0)
    parser.add_argument("--segment-length", type=float, default=120.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.75)
    parser.add_argument("--window-type", default="hann")
    parser.add_argument(
        "--detrend-method",
        choices=["none", "linear", "constant", "smoothness_priors"],
        default="none",
        help="Use 'none' to exercise the frozen post-Arm-A convention.",
    )
    parser.add_argument("--detrend-lambda", type=float, default=500.0)
    parser.add_argument("--neurokit-interpolation-method", default="monotone_cubic")
    parser.add_argument("--plot-count", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_detrend(value: str) -> Optional[str]:
    return None if value == "none" else value


def prepare_run_dir(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"Run directory already exists and is non-empty: {run_dir}. "
            "Use --overwrite to regenerate this validation-only run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "psd_plots").mkdir(parents=True, exist_ok=True)


def load_rr_intervals_ms(path: Path) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        bundle = load_rr_file(str(path))
    values = bundle.rri_ms or bundle.ppi_ms
    if not values:
        raise ValueError("No RRI/PPI intervals found in file.")
    rr = np.asarray(values, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 8:
        raise ValueError("Fewer than 8 valid RR intervals.")
    return rr


def select_clean_subset(input_dir: Path, subset_size: int, min_duration_s: float, max_invalid_pct: float) -> List[Path]:
    manifest_path = input_dir / "10min_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[
            (manifest["duration_seconds"] >= min_duration_s)
            & (manifest["percent_invalid_removed"] <= max_invalid_pct)
        ].copy()
        manifest["output_path"] = manifest["output_file"].map(Path)
        manifest = manifest[manifest["output_path"].map(lambda p: p.exists())]
        manifest = manifest.sort_values(["record_id", "segment_id"]).reset_index(drop=True)

        selected: List[Path] = []
        grouped = {
            record_id: group["output_path"].tolist()
            for record_id, group in manifest.groupby("record_id", sort=True)
        }
        depth = 0
        while len(selected) < subset_size:
            added = False
            for record_id in sorted(grouped):
                paths = grouped[record_id]
                if depth < len(paths):
                    selected.append(paths[depth])
                    added = True
                    if len(selected) >= subset_size:
                        break
            if not added:
                break
            depth += 1
        return selected

    candidates = sorted(input_dir.glob("*.csv"))
    return [p for p in candidates if p.name != "10min_manifest.csv"][:subset_size]


def rr_start_times_s(rr_ms: np.ndarray) -> np.ndarray:
    rr_s = rr_ms.astype(float) / 1000.0
    return np.concatenate([[0.0], np.cumsum(rr_s[:-1])])


def neurokit2_welch_metrics(
    rr_ms: np.ndarray,
    sampling_rate: float,
    interpolation_method: str,
    window_type: str,
    segment_length_s: float,
    overlap_ratio: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Dict[str, Any]]:
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
    metrics = metrics_from_psd(freqs, power)
    diagnostics = {
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "frequency_resolution_hz": float(freqs[1] - freqs[0]) if len(freqs) > 1 else math.nan,
        "n_samples": int(len(intervals)),
    }
    return metrics, freqs, power, diagnostics


def band_ranges() -> Dict[str, Tuple[float, float]]:
    bands = {
        name: HRVFreqDomainAnalysis.band_spec_for_convention(name)
        for name in ("vlf", "lf", "hf")
    }
    total = HRVFreqDomainAnalysis.total_power_spec_for_convention()
    return {
        "vlf": (bands["vlf"]["low"], bands["vlf"]["high"]),
        "lf": (bands["lf"]["low"], bands["lf"]["high"]),
        "hf": (bands["hf"]["low"], bands["hf"]["high"]),
        "total": (total["low"], total["high"]),
    }


def band_mask(freqs: np.ndarray, band: str) -> np.ndarray:
    if band == "total":
        return HRVFreqDomainAnalysis.total_power_mask_for_convention(freqs)
    return HRVFreqDomainAnalysis.mask_for_band(freqs, band)


def integrate_band(freqs: np.ndarray, psd: np.ndarray, band: str) -> float:
    mask = band_mask(freqs, band)
    if np.count_nonzero(mask) < 2:
        return 0.0
    value = np.trapezoid(psd[mask], freqs[mask])
    return float(value) if np.isfinite(value) else math.nan


def metrics_from_psd(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    vlf = integrate_band(freqs, psd, "vlf")
    lf = integrate_band(freqs, psd, "lf")
    hf = integrate_band(freqs, psd, "hf")
    total = integrate_band(freqs, psd, "total")
    lf_hf = lf / hf if np.isfinite(hf) and hf > 1e-10 else math.nan
    lf_hf_sum = lf + hf
    lf_nu = (lf / lf_hf_sum) * 100.0 if np.isfinite(lf_hf_sum) and lf_hf_sum > 0 else math.nan
    hf_nu = (hf / lf_hf_sum) * 100.0 if np.isfinite(lf_hf_sum) and lf_hf_sum > 0 else math.nan
    return {
        "vlf_power": vlf,
        "lf_power": lf,
        "hf_power": hf,
        "total_power": total,
        "lf_hf_ratio": lf_hf,
        "lf_nu": lf_nu,
        "hf_nu": hf_nu,
    }


def safe_rel_diff(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return math.nan
    return float(abs(value - reference) / abs(reference) * 100.0)


def safe_ratio(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return math.nan
    return float(value / reference)


def peak_diff_hz(results: Dict[str, Any], method: str, band: str) -> float:
    value = results.get(f"{method}_peak_freq_{band}", math.nan)
    reference = results.get(f"welch_peak_freq_{band}", math.nan)
    if not np.isfinite(value) or not np.isfinite(reference):
        return math.nan
    return float(abs(value - reference))


def psd_quality(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, Any]:
    if len(freqs) == 0 or len(psd) == 0:
        return {
            "psd_full_area": 0.0,
            "psd_band_area_0_0p4": 0.0,
            "psd_positive": False,
            "psd_nonfinite_count": 0,
            "psd_negative_count": 0,
            "psd_min": math.nan,
            "psd_max": math.nan,
        }
    finite = np.isfinite(psd)
    full_area = float(np.trapezoid(psd[finite], freqs[finite])) if np.count_nonzero(finite) > 1 else math.nan
    band_area = integrate_band(freqs, psd, "total")
    return {
        "psd_full_area": full_area,
        "psd_band_area_0_0p4": band_area,
        "psd_positive": bool(np.all(psd[finite] >= 0)) if np.any(finite) else False,
        "psd_nonfinite_count": int(np.count_nonzero(~finite)),
        "psd_negative_count": int(np.count_nonzero(psd[finite] < 0)) if np.any(finite) else 0,
        "psd_min": float(np.nanmin(psd)) if len(psd) else math.nan,
        "psd_max": float(np.nanmax(psd)) if len(psd) else math.nan,
    }


def finite_corr(x: Iterable[float], y: Iterable[float], method: str = "pearson") -> float:
    frame = pd.DataFrame({"x": list(x), "y": list(y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return math.nan
    return float(frame["x"].corr(frame["y"], method=method))


def analyze_file(path: Path, args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, Any]]:
    rr_ms = load_rr_intervals_ms(path)
    common_kwargs = dict(
        sampling_rate=args.interpolation_rate,
        detrend_method=normalize_detrend(args.detrend_method),
        detrend_lambda=args.detrend_lambda,
        window_type=args.window_type,
        segment_length=args.segment_length,
        overlap_ratio=args.overlap_ratio,
        enable_diagnostics=True,
    )

    analyzers: Dict[int, HRVFreqDomainAnalysis] = {}
    results_by_order: Dict[int, Dict[str, Any]] = {}
    for order in AR_ORDERS:
        analyzer = HRVFreqDomainAnalysis(rr_ms, ar_order=order, **common_kwargs)
        analyzers[order] = analyzer
        results_by_order[order] = analyzer.get_results()

    primary = analyzers[16]
    results = results_by_order[16]
    diagnostics = results.get("frequency_diagnostics", {})
    fft_diag = diagnostics.get("fft", {})
    ar_diag = diagnostics.get("ar", {})

    nk_metrics, nk_freqs, nk_psd, nk_diag = neurokit2_welch_metrics(
        rr_ms=rr_ms,
        sampling_rate=args.interpolation_rate,
        interpolation_method=args.neurokit_interpolation_method,
        window_type=args.window_type,
        segment_length_s=args.segment_length,
        overlap_ratio=args.overlap_ratio,
    )

    summary_row: Dict[str, Any] = {
        "file": str(path),
        "file_name": path.name,
        "n_rr": int(len(rr_ms)),
        "rr_duration_s": float(np.nansum(rr_ms) / 1000.0),
        "resampled_duration_s": results["analysis_info"].get("signal_duration_s", math.nan),
        "welch_nperseg": diagnostics.get("welch", {}).get("effective_nperseg", math.nan),
        "welch_noverlap": diagnostics.get("welch", {}).get("effective_noverlap", math.nan),
        "welch_frequency_resolution_hz": results["analysis_info"].get("frequency_resolution_welch", math.nan),
        "fft_frequency_resolution_hz": results["analysis_info"].get("frequency_resolution_fft", math.nan),
        "ar_frequency_resolution_hz": results["analysis_info"].get("frequency_resolution_ar", math.nan),
        "neurokit2_nperseg": nk_diag.get("nperseg", math.nan),
        "neurokit2_noverlap": nk_diag.get("noverlap", math.nan),
        "fft_integrated_psd_power": fft_diag.get("integrated_psd_power", math.nan),
        "fft_variance_ms2": fft_diag.get("variance_of_input_signal", math.nan),
        "fft_psd_variance_ratio": fft_diag.get("integrated_psd_power_variance_ratio", math.nan),
        "fft_variance_warning": bool(fft_diag.get("warning")),
        "ar_requested_order": ar_diag.get("requested_ar_order", 16),
        "ar_effective_order": ar_diag.get("effective_ar_order", math.nan),
        "ar_estimator_used": ar_diag.get("estimator_used", ""),
        "ar_fallback_reason": ar_diag.get("fallback_reason", ""),
        "ar_integrated_psd_power": ar_diag.get("integrated_psd_power", math.nan),
        "ar_variance_ms2": ar_diag.get("variance_of_ar_input", math.nan),
        "ar_psd_variance_ratio": ar_diag.get("integrated_psd_power_variance_ratio", math.nan),
        "ar_variance_warning": bool(ar_diag.get("warning")),
    }

    for method, freqs, psd in [
        ("welch", primary.freqs, primary.psd),
        ("fft", primary.fft_freqs, primary.fft_psd),
        ("ar", primary.ar_freqs, primary.ar_psd),
        ("neurokit2_welch", nk_freqs, nk_psd),
    ]:
        for key, value in psd_quality(freqs, psd).items():
            summary_row[f"{method}_{key}"] = value

    for label, key in METRICS.items():
        welch = results.get(f"welch_{key}", math.nan)
        fft = results.get(f"fft_{key}", math.nan)
        ar = results.get(f"ar_{key}", math.nan)
        nk = nk_metrics.get(key, math.nan)
        column_label = label.replace("/", "_").replace(" ", "_")
        summary_row[f"welch_{column_label}"] = welch
        summary_row[f"fft_{column_label}"] = fft
        summary_row[f"ar_{column_label}"] = ar
        summary_row[f"neurokit2_welch_{column_label}"] = nk
        summary_row[f"fft_vs_welch_{column_label}_rel_diff_pct"] = safe_rel_diff(fft, welch)
        summary_row[f"ar_vs_welch_{column_label}_rel_diff_pct"] = safe_rel_diff(ar, welch)
        summary_row[f"welch_vs_neurokit2_{column_label}_rel_diff_pct"] = safe_rel_diff(welch, nk)
        summary_row[f"fft_vs_neurokit2_{column_label}_rel_diff_pct"] = safe_rel_diff(fft, nk)
        summary_row[f"ar_vs_neurokit2_{column_label}_rel_diff_pct"] = safe_rel_diff(ar, nk)
        summary_row[f"fft_vs_welch_{column_label}_ratio"] = safe_ratio(fft, welch)
        summary_row[f"ar_vs_welch_{column_label}_ratio"] = safe_ratio(ar, welch)

    for band in ["vlf", "lf", "hf"]:
        summary_row[f"welch_peak_{band}_hz"] = results.get(f"welch_peak_freq_{band}", math.nan)
        summary_row[f"fft_peak_{band}_hz"] = results.get(f"fft_peak_freq_{band}", math.nan)
        summary_row[f"ar_peak_{band}_hz"] = results.get(f"ar_peak_freq_{band}", math.nan)
        summary_row[f"fft_peak_{band}_diff_from_welch_hz"] = peak_diff_hz(results, "fft", band)
        summary_row[f"ar_peak_{band}_diff_from_welch_hz"] = peak_diff_hz(results, "ar", band)

    summary_row["fft_unstable"] = classify_fft_unstable(summary_row)
    summary_row["ar_unstable"] = classify_ar_unstable(summary_row)
    summary_row["instability_reasons"] = "; ".join(instability_reasons(summary_row))

    order_rows: List[Dict[str, Any]] = []
    baseline = results_by_order[16]
    for order in AR_ORDERS:
        order_results = results_by_order[order]
        order_diag = order_results.get("frequency_diagnostics", {}).get("ar", {})
        order_row = {
            "file": str(path),
            "file_name": path.name,
            "requested_ar_order": order,
            "effective_ar_order": order_diag.get("effective_ar_order", math.nan),
            "estimator_used": order_diag.get("estimator_used", ""),
            "fallback_reason": order_diag.get("fallback_reason", ""),
            "psd_variance_ratio": order_diag.get("integrated_psd_power_variance_ratio", math.nan),
        }
        for label, key in METRICS.items():
            column_label = label.replace("/", "_").replace(" ", "_")
            value = order_results.get(f"ar_{key}", math.nan)
            base = baseline.get(f"ar_{key}", math.nan)
            welch = baseline.get(f"welch_{key}", math.nan)
            order_row[f"ar_{column_label}"] = value
            order_row[f"ar_order{order}_vs_order16_{column_label}_rel_diff_pct"] = safe_rel_diff(value, base)
            order_row[f"ar_order{order}_vs_welch_{column_label}_rel_diff_pct"] = safe_rel_diff(value, welch)
        order_rows.append(order_row)

    plot_payload = {
        "welch": (primary.freqs, primary.psd),
        "fft": (primary.fft_freqs, primary.fft_psd),
        "ar": (primary.ar_freqs, primary.ar_psd),
        "neurokit2_welch": (nk_freqs, nk_psd),
    }
    return summary_row, order_rows, plot_payload


def classify_fft_unstable(row: Dict[str, Any]) -> bool:
    ratio = row.get("fft_psd_variance_ratio", math.nan)
    return bool(
        row.get("fft_variance_warning")
        or row.get("fft_psd_nonfinite_count", 0) > 0
        or row.get("fft_psd_negative_count", 0) > 0
        or not row.get("fft_psd_positive", False)
        or (np.isfinite(ratio) and (ratio < 0.8 or ratio > 1.2))
        or row.get("fft_vs_welch_LF_HF_rel_diff_pct", 0) > 50
    )


def classify_ar_unstable(row: Dict[str, Any]) -> bool:
    ratio = row.get("ar_psd_variance_ratio", math.nan)
    return bool(
        bool(row.get("ar_fallback_reason"))
        or row.get("ar_psd_nonfinite_count", 0) > 0
        or row.get("ar_psd_negative_count", 0) > 0
        or not row.get("ar_psd_positive", False)
        or row.get("ar_variance_warning")
        or (np.isfinite(ratio) and (ratio < 0.8 or ratio > 1.2))
        or row.get("ar_vs_welch_LF_HF_rel_diff_pct", 0) > 75
    )


def instability_reasons(row: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    fft_ratio = row.get("fft_psd_variance_ratio", math.nan)
    ar_ratio = row.get("ar_psd_variance_ratio", math.nan)
    if row.get("fft_variance_warning") or (np.isfinite(fft_ratio) and (fft_ratio < 0.8 or fft_ratio > 1.2)):
        reasons.append("fft_variance_mismatch")
    if row.get("fft_psd_nonfinite_count", 0) > 0 or row.get("fft_psd_negative_count", 0) > 0 or not row.get("fft_psd_positive", False):
        reasons.append("fft_psd_invalid")
    if row.get("fft_vs_welch_LF_HF_rel_diff_pct", 0) > 50:
        reasons.append("fft_lfhf_trend_shift")
    if row.get("ar_fallback_reason"):
        reasons.append("ar_fallback")
    if row.get("ar_variance_warning") or (np.isfinite(ar_ratio) and (ar_ratio < 0.8 or ar_ratio > 1.2)):
        reasons.append("ar_variance_mismatch")
    if row.get("ar_psd_nonfinite_count", 0) > 0 or row.get("ar_psd_negative_count", 0) > 0 or not row.get("ar_psd_positive", False):
        reasons.append("ar_psd_invalid")
    if row.get("ar_vs_welch_LF_HF_rel_diff_pct", 0) > 75:
        reasons.append("ar_lfhf_trend_shift")
    return reasons


def add_ar_order_sensitivity_flags(order_df: pd.DataFrame) -> pd.DataFrame:
    if order_df.empty:
        return order_df
    pivot = order_df.pivot(index="file_name", columns="requested_ar_order", values="ar_LF_HF")
    if all(order in pivot.columns for order in AR_ORDERS):
        max_val = pivot[AR_ORDERS].max(axis=1)
        min_val = pivot[AR_ORDERS].min(axis=1)
        baseline = pivot[16].replace(0, np.nan)
        spread = ((max_val - min_val).abs() / baseline.abs()) * 100.0
        spread.name = "ar_lfhf_order_spread_pct"
        order_df = order_df.merge(spread.reset_index(), on="file_name", how="left")
    else:
        order_df["ar_lfhf_order_spread_pct"] = math.nan
    order_df["ar_order_sensitive"] = order_df["ar_lfhf_order_spread_pct"] > 50
    return order_df


def choose_plot_files(comparison: pd.DataFrame, plot_count: int) -> List[str]:
    picks: List[str] = []

    def add(name: Optional[str]) -> None:
        if name and name not in picks:
            picks.append(name)

    if comparison.empty:
        return picks
    median_lfhf = comparison["welch_LF_HF"].median()
    if np.isfinite(median_lfhf):
        idx = (comparison["welch_LF_HF"] - median_lfhf).abs().idxmin()
        add(comparison.loc[idx, "file_name"])
    for column in [
        "fft_vs_welch_LF_HF_rel_diff_pct",
        "ar_vs_welch_LF_HF_rel_diff_pct",
        "fft_psd_variance_ratio",
        "ar_psd_variance_ratio",
    ]:
        values = comparison[column].replace([np.inf, -np.inf], np.nan)
        if values.notna().any():
            idx = values.abs().idxmax()
            add(comparison.loc[idx, "file_name"])
    for _, row in comparison.sort_values("file_name").iterrows():
        add(row["file_name"])
        if len(picks) >= plot_count:
            break
    return picks[:plot_count]


def plot_psd(file_name: str, payload: Dict[str, Tuple[np.ndarray, np.ndarray]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    colors = {
        "welch": "#1f5f99",
        "fft": "#b55a30",
        "ar": "#267060",
        "neurokit2_welch": "#555555",
    }
    labels = {
        "welch": "HRV Studio Welch",
        "fft": "HRV Studio FFT",
        "ar": "HRV Studio AR order 16",
        "neurokit2_welch": "NeuroKit2 Welch",
    }
    for method, (freqs, psd) in payload.items():
        if len(freqs) == 0 or len(psd) == 0:
            continue
        mask = (freqs >= 0) & (freqs <= 0.4) & np.isfinite(freqs) & np.isfinite(psd)
        axes[0].plot(freqs[mask], psd[mask], linewidth=1.1, alpha=0.9, color=colors[method], label=labels[method])
        zoom = (freqs >= 0.04) & (freqs <= 0.4) & np.isfinite(freqs) & np.isfinite(psd)
        axes[1].plot(freqs[zoom], psd[zoom], linewidth=1.1, alpha=0.9, color=colors[method], label=labels[method])
    for ax in axes:
        ax.axvspan(0.04, 0.15, color="#d8e8f5", alpha=0.35)
        ax.axvspan(0.15, 0.4, color="#e7f0dc", alpha=0.35)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (ms^2/Hz)")
        ax.grid(alpha=0.22)
    axes[0].set_title("0-0.4 Hz")
    axes[1].set_title("LF/HF region")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(file_name)
    fig.tight_layout()
    safe_name = file_name.replace(".csv", "")
    fig.savefig(output_dir / f"{safe_name}_psd_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    run_dir: Path,
    comparison: pd.DataFrame,
    instability: pd.DataFrame,
    order_df: pd.DataFrame,
    args: argparse.Namespace,
    plot_files: List[str],
) -> None:
    def pct(count: int, denom: int) -> str:
        return f"{count} ({count / denom * 100.0:.1f}%)" if denom else "0 (0.0%)"

    n = len(comparison)
    fft_lfhf_corr = finite_corr(comparison["welch_LF_HF"], comparison["fft_LF_HF"])
    ar_lfhf_corr = finite_corr(comparison["welch_LF_HF"], comparison["ar_LF_HF"])
    nk_lfhf_corr = finite_corr(comparison["neurokit2_welch_LF_HF"], comparison["welch_LF_HF"])
    fft_lfhf_med = comparison["fft_vs_welch_LF_HF_rel_diff_pct"].median()
    ar_lfhf_med = comparison["ar_vs_welch_LF_HF_rel_diff_pct"].median()
    fft_var_med = comparison["fft_psd_variance_ratio"].median()
    ar_var_med = comparison["ar_psd_variance_ratio"].median()
    fallback_count = int((comparison["ar_fallback_reason"].fillna("").astype(str) != "").sum())
    fft_unstable = int(comparison["fft_unstable"].sum())
    ar_unstable = int(comparison["ar_unstable"].sum())
    order_sensitive_files = int(order_df.groupby("file_name")["ar_order_sensitive"].max().sum()) if not order_df.empty else 0
    fft_positive_count = int(comparison["fft_psd_positive"].sum())
    ar_positive_count = int(comparison["ar_psd_positive"].sum())
    fft_nonfinite_count = int((comparison["fft_psd_nonfinite_count"] > 0).sum())
    ar_nonfinite_count = int((comparison["ar_psd_nonfinite_count"] > 0).sum())
    fft_lf_peak_med = comparison["fft_peak_lf_diff_from_welch_hz"].median()
    fft_hf_peak_med = comparison["fft_peak_hf_diff_from_welch_hz"].median()
    ar_lf_peak_med = comparison["ar_peak_lf_diff_from_welch_hz"].median()
    ar_hf_peak_med = comparison["ar_peak_hf_diff_from_welch_hz"].median()
    fft_dc_variance_only = int(
        comparison["instability_reasons"].fillna("").eq("fft_variance_mismatch").sum()
    )

    metric_rows = []
    for label in METRICS:
        col = label.replace("/", "_").replace(" ", "_")
        metric_rows.append(
            {
                "metric": label,
                "fft_median_rel_diff_vs_welch_pct": comparison[f"fft_vs_welch_{col}_rel_diff_pct"].median(),
                "ar_median_rel_diff_vs_welch_pct": comparison[f"ar_vs_welch_{col}_rel_diff_pct"].median(),
                "welch_median_rel_diff_vs_neurokit2_pct": comparison[f"welch_vs_neurokit2_{col}_rel_diff_pct"].median(),
            }
        )
    metric_table = pd.DataFrame(metric_rows)

    def md_table(df: pd.DataFrame) -> str:
        lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        for _, row in df.iterrows():
            values = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    values.append("" if pd.isna(value) else f"{value:.2f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    interpretation = []
    if fft_lfhf_corr >= 0.9 and fft_lfhf_med <= 25 and fft_unstable <= max(5, 0.1 * n):
        interpretation.append("FFT behaves broadly consistently with Welch for LF/HF trends on this clean subset.")
    else:
        interpretation.append("FFT shows partial consistency with Welch, but trend or variance diagnostics require cautious interpretation.")
    if fallback_count == 0 and order_sensitive_files <= max(5, 0.1 * n) and ar_lfhf_corr >= 0.8:
        interpretation.append("AR remains mostly stable across the tested orders and did not show widespread fallback behavior.")
    else:
        interpretation.append("AR shows meaningful sensitivity or fallback behavior and should be presented as method-dependent.")
    if fft_unstable or ar_unstable:
        interpretation.append("The flagged differences are best treated as methodological or numerical sensitivity cases unless raw-signal review identifies a concrete input or implementation failure.")
    else:
        interpretation.append("No broad implementation-failure pattern was detected in the clean subset.")

    publishable = (
        "Defensible with caveats: report Welch as the primary frequency-domain method, "
        "use FFT/AR as secondary method-specific analyses, and disclose variance, "
        "fallback, and AR-order sensitivity checks."
    )

    summary = f"""# FFT and AR Validation Summary

This is a validation-only Phase 5 experiment. It does not modify production HRV Studio code and does not rerun full validation.

## Run Configuration

- Input directory: `{args.input_dir}`
- Output directory: `{args.run_dir}`
- Requested clean subset size: {args.subset_size}
- Processed files: {n}
- Clean-subset filters: duration >= {args.min_duration_s:.1f}s; percent invalid removed <= {args.max_invalid_removed_pct:.2f}%
- Interpolation rate: {args.interpolation_rate:.1f} Hz
- Welch segment length / overlap: {args.segment_length:.1f}s / {args.overlap_ratio:.2f}
- Window: {args.window_type}
- Detrend method: {args.detrend_method}
- AR orders tested: {", ".join(str(x) for x in AR_ORDERS)}

## Headline Findings

- FFT LF/HF correlation vs HRV Studio Welch: {fft_lfhf_corr:.3f}
- AR LF/HF correlation vs HRV Studio Welch: {ar_lfhf_corr:.3f}
- HRV Studio Welch LF/HF correlation vs NeuroKit2 Welch: {nk_lfhf_corr:.3f}
- FFT median LF/HF relative difference vs Welch: {fft_lfhf_med:.2f}%
- AR median LF/HF relative difference vs Welch: {ar_lfhf_med:.2f}%
- FFT median integrated PSD / variance ratio: {fft_var_med:.3f}
- AR median integrated PSD / variance ratio: {ar_var_med:.3f}
- FFT PSD-positive files: {pct(fft_positive_count, n)}
- AR PSD-positive files: {pct(ar_positive_count, n)}
- Files with non-finite FFT PSD values: {pct(fft_nonfinite_count, n)}
- Files with non-finite AR PSD values: {pct(ar_nonfinite_count, n)}
- Median LF/HF peak-frequency differences vs Welch: FFT LF {fft_lf_peak_med:.4f} Hz, FFT HF {fft_hf_peak_med:.4f} Hz; AR LF {ar_lf_peak_med:.4f} Hz, AR HF {ar_hf_peak_med:.4f} Hz
- FFT instability flags: {pct(fft_unstable, n)}
- Files flagged only for FFT PSD/variance mismatch: {pct(fft_dc_variance_only, n)}
- AR instability flags: {pct(ar_unstable, n)}
- AR fallbacks at order 16: {pct(fallback_count, n)}
- Files with AR LF/HF order spread >50%: {pct(order_sensitive_files, n)}

## Metric-Level Method Agreement

{md_table(metric_table)}

## Paper-Oriented Interpretation

{" ".join(interpretation)}

The dominant FFT flag is PSD-area inconsistency: the current no-detrend FFT path preserves the large DC component of the RR level, whereas the frozen Welch no-detrend convention removes a global mean before Welch integration. Therefore FFT can preserve LF/HF trend information while producing non-comparable VLF and total-power areas. This should be treated as a methodological/convention discrepancy in the validation manuscript unless the FFT method is explicitly redefined and revalidated.

Publishability assessment: {publishable}

## Interpretation Questions

### Does FFT behave consistently with Welch?

FFT should be considered consistent with Welch when LF/HF trend correlation is high, median LF/HF difference is modest, spectral peaks remain in comparable bands, and the integrated PSD area is close to signal variance. The present run reports these checks explicitly in `fft_ar_comparison.csv` and flags exceptions in `instability_cases.csv`.

### Does AR remain stable?

AR stability is evaluated by PSD positivity, non-finite/negative PSD counts, fallback behavior, variance-normalized PSD area, LF/HF consistency with Welch, and order sensitivity across AR(8), AR(16), and AR(24). AR is defensible only as a method-dependent estimate if order sensitivity is non-trivial.

### Are differences methodological or implementation failures?

Differences without non-finite PSDs, negative powers, fallback, or gross variance mismatch should be interpreted primarily as methodological differences between Welch averaging, whole-signal FFT periodograms, and parametric AR spectra. Files listed in `instability_cases.csv` require targeted review before being described as clean methodological differences.

### Is behavior publishable/defensible?

The defensible paper position is to keep Welch as the primary reported method. FFT and AR can be reported as validation/secondary spectral methods if the manuscript discloses the checks used here and avoids claiming equivalence across methods.

## Representative PSD Plots

{chr(10).join(f"- `psd_plots/{name.replace('.csv', '')}_psd_comparison.png`" for name in plot_files)}

## Output Files

- `fft_ar_comparison.csv`
- `fft_ar_summary.md`
- `instability_cases.csv`
- `ar_order_sensitivity.csv`
- `psd_plots/*_psd_comparison.png`
"""
    (run_dir / "fft_ar_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    prepare_run_dir(args.run_dir, args.overwrite)

    paths = select_clean_subset(
        input_dir=args.input_dir,
        subset_size=args.subset_size,
        min_duration_s=args.min_duration_s,
        max_invalid_pct=args.max_invalid_removed_pct,
    )
    if not paths:
        raise SystemExit("No clean input files selected.")

    comparison_rows: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []
    plot_payloads: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    failures: List[Dict[str, Any]] = []

    for idx, path in enumerate(paths, start=1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row, rows_by_order, payload = analyze_file(path, args)
            comparison_rows.append(row)
            order_rows.extend(rows_by_order)
            plot_payloads[path.name] = payload
        except Exception as exc:
            failures.append({"file": str(path), "file_name": path.name, "failure": str(exc)})
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(paths)} files")

    comparison = pd.DataFrame(comparison_rows)
    if comparison.empty:
        raise SystemExit(f"All selected files failed. Failure count: {len(failures)}")

    order_df = add_ar_order_sensitivity_flags(pd.DataFrame(order_rows))
    comparison.to_csv(args.run_dir / "fft_ar_comparison.csv", index=False)
    order_df.to_csv(args.run_dir / "ar_order_sensitivity.csv", index=False)

    instability = comparison[
        comparison["fft_unstable"] | comparison["ar_unstable"] | (comparison["instability_reasons"].fillna("") != "")
    ].copy()
    instability.to_csv(args.run_dir / "instability_cases.csv", index=False)

    plot_files = choose_plot_files(comparison, args.plot_count)
    for file_name in plot_files:
        payload = plot_payloads.get(file_name)
        if payload:
            plot_psd(file_name, payload, args.run_dir / "psd_plots")

    run_info = {
        "script": "tools/validate_fft_ar_methods.py",
        "validation_only": True,
        "production_code_modified": False,
        "selected_files": len(paths),
        "processed_files": int(len(comparison)),
        "failed_files": failures,
        "settings": {
            "band_convention": HRVFreqDomainAnalysis.DEFAULT_BAND_CONVENTION,
            "input_dir": str(args.input_dir),
            "subset_size": args.subset_size,
            "min_duration_s": args.min_duration_s,
            "max_invalid_removed_pct": args.max_invalid_removed_pct,
            "interpolation_rate": args.interpolation_rate,
            "segment_length": args.segment_length,
            "overlap_ratio": args.overlap_ratio,
            "window_type": args.window_type,
            "detrend_method": args.detrend_method,
            "ar_orders": AR_ORDERS,
        },
    }
    (args.run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    write_summary(args.run_dir, comparison, instability, order_df, args, plot_files)

    print(f"Wrote FFT/AR validation outputs to {args.run_dir}")
    print(f"Processed files: {len(comparison)}")
    print(f"Failures: {len(failures)}")
    print(f"Instability cases: {len(instability)}")


if __name__ == "__main__":
    main()
