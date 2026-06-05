"""
Validation-only investigation of v06 detrend_method=none frequency-domain outputs.

This script does not modify production code. It reproduces native HRV Studio and
NeuroKit2 PSDs for the three manually exported Kubios pilot files, writes
diagnostic plots, and summarizes likely failure modes cautiously.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from validate_freq_domain_neurokit2 import (
    effective_welch_params,
    load_rr_intervals_ms,
    metrics_from_psd,
    validation_welch_psd_with_mode,
)


RUN_DIR = PROJECT_ROOT / "validation" / "runs" / "v06_kubios_subset_none_120_75"
PARSED_DIR = PROJECT_ROOT / "validation" / "kubios_subset" / "parsed_results_120s_75pct_none"
NOTE_PATH = PROJECT_ROOT / "validation" / "research_notes" / "v06_none_mode_investigation.md"
PLOT_DIR = PROJECT_ROOT / "validation" / "research_notes" / "v06_none_mode_investigation_plots"

FILES = {
    "CH001": PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "input_ascii_rr"
    / "clean_high_agreement"
    / "CH001__nsr020_segment_000.txt",
    "OUT001": PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "input_ascii_rr"
    / "remaining_outliers"
    / "OUT001__nsr009_segment_012.txt",
    "VLF001": PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "input_ascii_rr"
    / "vlf_sensitive"
    / "VLF001__nsr016_segment_054.txt",
}

BANDS = {
    "VLF": (0.0, 0.04),
    "LF": (0.04, 0.15),
    "HF": (0.15, 0.4),
    "total_power": (0.0, 0.4),
}

METRIC_KEYS = {
    "VLF": "vlf_power",
    "LF": "lf_power",
    "HF": "hf_power",
    "total_power": "total_power",
    "LF/HF": "lf_hf_ratio",
    "LF_nu": "lf_nu",
    "HF_nu": "hf_nu",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if finite(value) else "not available"


def rel_err(reference: float, value: float) -> float:
    if not finite(reference) or not finite(value) or abs(reference) <= 1e-12:
        return math.nan
    return abs(value - reference) / abs(reference) * 100.0


def integrate(freqs: np.ndarray, psd: np.ndarray, low: float, high: float, include_dc: bool = True) -> float:
    mask = np.isfinite(freqs) & np.isfinite(psd) & (freqs >= low) & (freqs <= high)
    if not include_dc:
        mask &= freqs > 0.0
    if np.count_nonzero(mask) < 2:
        return math.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def cumulative_power(freqs: np.ndarray, psd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(freqs) & np.isfinite(psd) & (freqs >= 0.0) & (freqs <= 0.4)
    x = freqs[mask]
    y = psd[mask]
    if len(x) == 0:
        return x, y
    out = np.zeros(len(x), dtype=float)
    if len(x) > 1:
        out[1:] = np.cumsum((y[:-1] + y[1:]) * 0.5 * np.diff(x))
    return x, out


def native_scipy_welch(
    analyzer: HRVFreqDomainAnalysis,
    detrend: Any,
    subtract_mean: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    nperseg, noverlap = effective_welch_params(analyzer)
    if nperseg is None or noverlap is None:
        return np.array([]), np.array([])
    x = analyzer.time_domain_s.copy()
    if subtract_mean:
        x = x - np.nanmean(x)
    freqs, psd_s2 = signal.welch(
        x=x,
        fs=analyzer.sampling_rate,
        window=analyzer.window_type,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
        average="mean",
    )
    return freqs, psd_s2 * 1e6


def read_kubios_values() -> Dict[str, Dict[str, float]]:
    rows = read_csv(PARSED_DIR / "kubios_parsed_results.csv")
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        subset = row["subset_id"]
        out[subset] = {
            "VLF": float(row["kubios_vlf"]),
            "LF": float(row["kubios_lf"]),
            "HF": float(row["kubios_hf"]),
            "total_power": float(row["kubios_total_power"]),
            "LF/HF": float(row["kubios_lf_hf"]),
            "LF_nu": float(row["kubios_lf_nu"]),
            "HF_nu": float(row["kubios_hf_nu"]),
        }
    return out


def first_vlf_table(freqs: np.ndarray, psd: np.ndarray) -> List[Dict[str, float]]:
    rows = []
    finite_mask = np.isfinite(freqs) & np.isfinite(psd) & (freqs <= 0.05)
    f = freqs[finite_mask]
    p = psd[finite_mask]
    for idx, (freq, power) in enumerate(zip(f[:8], p[:8])):
        rows.append({"index": idx, "frequency": float(freq), "psd": float(power)})
    return rows


def plot_psd(
    subset: str,
    native_freqs: np.ndarray,
    native_psd: np.ndarray,
    nk_freqs: np.ndarray,
    nk_psd: np.ndarray,
    mean_removed_freqs: np.ndarray,
    mean_removed_psd: np.ndarray,
    kubios: Dict[str, float],
    output: Path,
    zoom: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for band, (low, high) in [("VLF", (0, 0.04)), ("LF", (0.04, 0.15)), ("HF", (0.15, 0.4))]:
        ax.axvspan(low, high, alpha=0.12, label=band if not zoom else None)
    ax.plot(native_freqs, native_psd, marker="o", markersize=3, linewidth=1.4, label="HRV Studio native")
    ax.plot(nk_freqs, nk_psd, marker="o", markersize=3, linewidth=1.4, label="NeuroKit2")
    ax.plot(mean_removed_freqs, mean_removed_psd, linestyle="--", linewidth=1.2, label="Native mean removed diagnostic")
    for metric, (low, high) in BANDS.items():
        if metric == "total_power":
            continue
        width = max(high - low, 1e-12)
        ax.hlines(
            kubios[metric] / width,
            low,
            high,
            colors="#333333",
            linestyles=":",
            linewidth=1.4,
            label="Kubios band avg PSD" if metric == "VLF" else None,
        )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 0.06 if zoom else 0.4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (ms^2/Hz)")
    ax.set_title(f"{subset} {'zoomed VLF' if zoom else 'native vs NeuroKit2 PSD'}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_cumulative(
    subset: str,
    native_freqs: np.ndarray,
    native_psd: np.ndarray,
    nk_freqs: np.ndarray,
    nk_psd: np.ndarray,
    mean_removed_freqs: np.ndarray,
    mean_removed_psd: np.ndarray,
    kubios: Dict[str, float],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for label, freqs, psd in [
        ("HRV Studio native", native_freqs, native_psd),
        ("NeuroKit2", nk_freqs, nk_psd),
        ("Native mean removed diagnostic", mean_removed_freqs, mean_removed_psd),
    ]:
        x, y = cumulative_power(freqs, psd)
        ax.plot(x, y, linewidth=1.5, label=label)
    ax.axhline(kubios["VLF"], color="#555555", linestyle=":", linewidth=1.0, label="Kubios VLF")
    ax.axhline(kubios["total_power"], color="#111111", linestyle="--", linewidth=1.0, label="Kubios total")
    ax.axvline(0.04, color="#777777", linewidth=0.9)
    ax.axvline(0.15, color="#777777", linewidth=0.9)
    ax.set_xlim(0.0, 0.4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cumulative integrated power (ms^2)")
    ax.set_title(f"{subset} cumulative power by frequency")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def metric_snapshot(
    kubios: Dict[str, float],
    native_metrics: Dict[str, float],
    nk_metrics: Dict[str, float],
    mean_removed_metrics: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows = []
    for metric, key in METRIC_KEYS.items():
        rows.append(
            {
                "metric": metric,
                "kubios": kubios[metric],
                "native": native_metrics.get(key, math.nan),
                "neurokit2": nk_metrics.get(key, math.nan),
                "native_mean_removed": mean_removed_metrics.get(key, math.nan),
                "native_rel_err_pct": rel_err(kubios[metric], native_metrics.get(key, math.nan)),
                "neurokit2_rel_err_pct": rel_err(kubios[metric], nk_metrics.get(key, math.nan)),
                "mean_removed_rel_err_pct": rel_err(kubios[metric], mean_removed_metrics.get(key, math.nan)),
            }
        )
    return rows


def analyze_subset(subset: str, path: Path, kubios: Dict[str, float]) -> Dict[str, Any]:
    rr_ms = load_rr_intervals_ms(path)
    analyzer = HRVFreqDomainAnalysis(
        rr_ms,
        sampling_rate=4.0,
        detrend_method=None,
        detrend_lambda=500.0,
        window_type="hann",
        segment_length=120.0,
        overlap_ratio=0.75,
        ar_order=16,
        enable_diagnostics=True,
    )
    nperseg, noverlap = effective_welch_params(analyzer)
    nk_freqs, nk_psd, nk_nperseg, nk_noverlap, _duration, _adjustment = validation_welch_psd_with_mode(
        path=path,
        rr_ms=analyzer.rr_intervals_ms,
        sampling_rate=4.0,
        interpolation_method="cubic",
        window_type="hann",
        nperseg=int(120.0 * 4.0),
        noverlap=int(120.0 * 4.0 * 0.75),
        detrend_method=None,
        detrend_lambda=500.0,
        welch_detrend_mode="current",
    )
    mean_freqs, mean_psd = native_scipy_welch(analyzer, detrend=False, subtract_mean=True)
    constant_freqs, constant_psd = native_scipy_welch(analyzer, detrend="constant")

    native_metrics = analyzer.spectral_metrics
    nk_metrics = metrics_from_psd(nk_freqs, nk_psd)
    mean_removed_metrics = metrics_from_psd(mean_freqs, mean_psd)
    constant_metrics = metrics_from_psd(constant_freqs, constant_psd)

    subset_dir = PLOT_DIR / subset
    subset_dir.mkdir(parents=True, exist_ok=True)
    plot_psd(
        subset,
        analyzer.freqs,
        analyzer.psd,
        nk_freqs,
        nk_psd,
        mean_freqs,
        mean_psd,
        kubios,
        subset_dir / "psd_overlay.png",
    )
    plot_psd(
        subset,
        analyzer.freqs,
        analyzer.psd,
        nk_freqs,
        nk_psd,
        mean_freqs,
        mean_psd,
        kubios,
        subset_dir / "zoomed_vlf.png",
        zoom=True,
    )
    plot_cumulative(
        subset,
        analyzer.freqs,
        analyzer.psd,
        nk_freqs,
        nk_psd,
        mean_freqs,
        mean_psd,
        kubios,
        subset_dir / "cumulative_power_by_frequency.png",
    )

    native_no_dc = {name: integrate(analyzer.freqs, analyzer.psd, *band, include_dc=False) for name, band in BANDS.items()}
    native_with_dc = {name: integrate(analyzer.freqs, analyzer.psd, *band, include_dc=True) for name, band in BANDS.items()}
    mean_removed_with_dc = {name: integrate(mean_freqs, mean_psd, *band, include_dc=True) for name, band in BANDS.items()}
    constant_with_dc = {name: integrate(constant_freqs, constant_psd, *band, include_dc=True) for name, band in BANDS.items()}

    psd_finite = analyzer.psd[np.isfinite(analyzer.psd)]
    time_finite = analyzer.time_domain_s[np.isfinite(analyzer.time_domain_s)]
    rr_s = analyzer.rr_intervals_ms.astype(float) / 1000.0
    rr_start_times = np.concatenate([[0.0], np.cumsum(rr_s[:-1])]) if len(rr_s) else np.array([])
    duplicate_start_times = (
        int(len(rr_start_times) - len(np.unique(rr_start_times))) if len(rr_start_times) else 0
    )
    return {
        "subset": subset,
        "path": path,
        "plot_dir": subset_dir,
        "rr_count": int(len(analyzer.rr_intervals_ms)),
        "raw_nan_count": int(np.count_nonzero(~np.isfinite(rr_ms))),
        "rr_nonpositive_count": int(np.count_nonzero(analyzer.rr_intervals_ms <= 0)),
        "rr_min_ms": float(np.nanmin(analyzer.rr_intervals_ms)) if len(analyzer.rr_intervals_ms) else math.nan,
        "rr_duplicate_start_time_count": duplicate_start_times,
        "native_time_samples": int(len(analyzer.time_domain_s)),
        "native_time_nan_count": int(np.count_nonzero(~np.isfinite(analyzer.time_domain_s))),
        "native_time_mean_ms": float(np.mean(time_finite) * 1000.0) if len(time_finite) else math.nan,
        "native_time_std_ms": float(np.std(time_finite) * 1000.0) if len(time_finite) else math.nan,
        "native_psd_nan_count": int(np.count_nonzero(~np.isfinite(analyzer.psd))),
        "native_psd_dc": float(analyzer.psd[0]) if len(analyzer.psd) else math.nan,
        "native_psd_first_vlf": float(analyzer.psd[1]) if len(analyzer.psd) > 1 else math.nan,
        "native_psd_max": float(np.max(psd_finite)) if len(psd_finite) else math.nan,
        "native_psd_argmax_freq": float(analyzer.freqs[np.nanargmax(analyzer.psd)]) if len(psd_finite) and np.any(np.isfinite(analyzer.psd)) else math.nan,
        "neurokit2_psd_dc": float(nk_psd[0]) if len(nk_psd) else math.nan,
        "neurokit2_psd_first_vlf": float(nk_psd[1]) if len(nk_psd) > 1 else math.nan,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nk_nperseg": nk_nperseg,
        "nk_noverlap": nk_noverlap,
        "metric_rows": metric_snapshot(kubios, native_metrics, nk_metrics, mean_removed_metrics),
        "native_with_dc": native_with_dc,
        "native_no_dc": native_no_dc,
        "native_mean_removed": mean_removed_with_dc,
        "native_constant_detrend": constant_with_dc,
        "first_vlf_native": first_vlf_table(analyzer.freqs, analyzer.psd),
        "first_vlf_neurokit2": first_vlf_table(nk_freqs, nk_psd),
        "first_vlf_mean_removed": first_vlf_table(mean_freqs, mean_psd),
        "frequency_diagnostics": analyzer.get_results().get("frequency_diagnostics", {}),
    }


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def plot_links(summary: Dict[str, Any]) -> str:
    rel = summary["plot_dir"].relative_to(PROJECT_ROOT)
    return (
        f"`{rel / 'psd_overlay.png'}`, "
        f"`{rel / 'zoomed_vlf.png'}`, "
        f"`{rel / 'cumulative_power_by_frequency.png'}`"
    )


def write_note(summaries: Sequence[Dict[str, Any]]) -> None:
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    overview_rows = []
    for item in summaries:
        vlf = next(row for row in item["metric_rows"] if row["metric"] == "VLF")
        total = next(row for row in item["metric_rows"] if row["metric"] == "total_power")
        overview_rows.append(
            [
                item["subset"],
                fmt(vlf["kubios"]),
                fmt(vlf["native"]),
                fmt(vlf["neurokit2"]),
                fmt(vlf["native_mean_removed"]),
                fmt(vlf["native_rel_err_pct"], 2),
                fmt(total["native"]),
                fmt(total["neurokit2"]),
                fmt(item["native_psd_dc"], 2),
                item["native_psd_nan_count"],
            ]
        )

    lines: List[str] = [
        "# v06 none/current Native Frequency-Domain Investigation",
        "",
        "Scope: validation-only investigation of HRV Studio native frequency-domain computation under `detrend_method=none` and `welch_detrend_mode=current`. No production HRV code was modified.",
        "",
        "## Short Answer",
        "",
        "The current evidence points to two issues rather than one uniform PSD scaling problem.",
        "",
        "- CH001 and OUT001: the native Welch computation appears to include very large DC / first-near-zero content when `detrend_method=None` maps to `scipy.signal.welch(detrend=False)`. This inflates VLF and total power while LF, HF, LF/HF, and normalized powers remain comparatively close to Kubios/NeuroKit2.",
        "- VLF001: native Welch PSD contains non-finite values, so `total_power` becomes non-finite and `_compute_spectral_metrics()` returns zero/default band powers for VLF/LF/HF. This is a separate robustness problem in the no-detrend path or its preprocessing inputs.",
        "- Cautious classification: likely a real HRV Studio behavior/bug under `detrend_method=none`, centered on mean/DC handling and non-finite PSD handling. It does not look like a simple global PSD scaling or one-sided FFT normalization error for the main CH001/OUT001 symptom.",
        "",
        "## Key Metrics",
        "",
        *table(
            [
                "File",
                "Kubios VLF",
                "Native VLF",
                "NeuroKit2 VLF",
                "Native mean-removed VLF",
                "Native VLF err %",
                "Native total",
                "NeuroKit2 total",
                "Native DC PSD",
                "Native PSD NaNs",
            ],
            overview_rows,
        ),
        "",
        "## Checks Against Requested Hypotheses",
        "",
        "- PSD scaling / FFT normalization / one-sided PSD scaling: not the leading explanation for CH001 and OUT001. LF/HF shape and non-VLF bands are close enough that a uniform scaling error is unlikely. The native FFT diagnostic does show variance inconsistency when no detrending is used, but the reported validation metric is Welch, not FFT.",
        "- Integration of DC bin: strongly implicated for CH001 and OUT001. The native VLF band starts at 0.0 Hz and the no-detrend Welch call preserves the large mean RR level, producing extreme DC/near-zero density.",
        "- VLF band mask: native VLF is `(freqs >= 0.0) & (freqs <= 0.04)`, so DC is included. ULF is also defined as `0.0-0.003`, which overlaps VLF, but the immediate v06 symptom is VLF/total inflation rather than ULF reporting.",
        "- total_power integration: native total power uses the same lower bound as VLF, `0.0-0.4 Hz`, so any DC/near-zero inflation is carried directly into total power.",
        "- `detrend_method=none` preprocessing: the native path bypasses both global and segment detrending. For Welch it passes `detrend=False`, which is expected from the code but probably not comparable to Kubios/NeuroKit2 output when the mean RR level is retained.",
        "- Extreme DC inflation: present for CH001 and OUT001. The plots and first-bin tables show the native PSD dominated by the first bins.",
        "- VLF001 zeros: the native PSD contains non-finite values. `_compute_spectral_metrics()` does not reject non-finite total power before continuing; NaN band integrations are then clamped/defaulted to zeros for VLF/LF/HF and normalized powers.",
        "- `current` mode: in v06, `current` preserves NeuroKit2's existing `signal_psd()` behavior. It does not mimic HRV Studio native `detrend=False`; therefore `current` is expected to behave differently from native none-mode near DC.",
        "",
        "## Per-File Details",
        "",
    ]

    for item in summaries:
        lines.extend(
            [
                f"### {item['subset']}",
                "",
                f"- Plots: {plot_links(item)}",
                f"- RR count: {item['rr_count']}; native resampled samples: {item['native_time_samples']}; native time-domain NaN/non-finite count: {item['native_time_nan_count']}.",
                f"- RR minimum: {fmt(item['rr_min_ms'], 3)} ms; nonpositive RR count: {item['rr_nonpositive_count']}; duplicate RR start-time count: {item['rr_duplicate_start_time_count']}.",
                f"- Native time-domain mean/std: {fmt(item['native_time_mean_ms'], 3)} / {fmt(item['native_time_std_ms'], 3)} ms.",
                f"- Welch parameters native/NeuroKit2: nperseg `{item['nperseg']}` / `{item['nk_nperseg']}`, noverlap `{item['noverlap']}` / `{item['nk_noverlap']}`.",
                f"- Native VLF with DC: {fmt(item['native_with_dc']['VLF'])}; excluding DC: {fmt(item['native_no_dc']['VLF'])}; mean-removed diagnostic: {fmt(item['native_mean_removed']['VLF'])}; constant-detrend diagnostic: {fmt(item['native_constant_detrend']['VLF'])}.",
                f"- Native total with DC: {fmt(item['native_with_dc']['total_power'])}; excluding DC: {fmt(item['native_no_dc']['total_power'])}; mean-removed diagnostic: {fmt(item['native_mean_removed']['total_power'])}.",
                "",
                "First native VLF bins:",
                "",
                *table(
                    ["Index", "Frequency", "Native PSD", "NeuroKit2 PSD", "Mean-removed native PSD"],
                    [
                        [
                            idx,
                            fmt(item["first_vlf_native"][idx]["frequency"], 6) if idx < len(item["first_vlf_native"]) else "",
                            fmt(item["first_vlf_native"][idx]["psd"], 3) if idx < len(item["first_vlf_native"]) else "",
                            fmt(item["first_vlf_neurokit2"][idx]["psd"], 3) if idx < len(item["first_vlf_neurokit2"]) else "",
                            fmt(item["first_vlf_mean_removed"][idx]["psd"], 3) if idx < len(item["first_vlf_mean_removed"]) else "",
                        ]
                        for idx in range(0, 6)
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Cautious Conclusion",
            "",
            "This looks more like `a) a real HRV Studio no-detrend behavior/bug` plus `c) a DC integration problem` than a pure settings mismatch. The settings are intentionally matched at a high level, but the implementations do not handle the mean/DC component the same way under `none/current`.",
            "",
            "The strongest evidence is that removing only the mean from the native Welch input collapses CH001/OUT001 VLF and total power toward Kubios/NeuroKit2-like magnitudes, while LF/HF is already close. VLF001 should be treated separately: the native PSD becomes non-finite, so zeros are a downstream defaulting artifact rather than valid zero physiological power.",
            "",
            "Recommended next code-review target before any fix: native `HRVFreqDomainAnalysis._compute_welch_psd()` and `_compute_spectral_metrics()` handling of `detrend_method=None`, finite-signal/finite-PSD validation, and whether total/VLF should include the DC bin for HRV band-power reporting.",
            "",
        ]
    )
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    kubios = read_kubios_values()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [analyze_subset(subset, path, kubios[subset]) for subset, path in FILES.items()]
    write_note(summaries)
    print(f"Wrote {NOTE_PATH}")
    print(f"Wrote plots under {PLOT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
