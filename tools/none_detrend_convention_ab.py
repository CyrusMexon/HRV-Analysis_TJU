"""
Validation-only A/B comparison for no-detrend Welch conventions after the DC fix.

Arm A: subtract one global mean from the full interpolated RR signal, then Welch
with detrend=False.
Arm B: Welch with detrend="constant" per segment.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from validate_freq_domain_neurokit2 import load_rr_intervals_ms


PARSED_DIR = PROJECT_ROOT / "validation" / "kubios_subset" / "parsed_results_120s_75pct_none_after_fix"
OUTPUT = PROJECT_ROOT / "validation" / "research_notes" / "none_detrend_convention_ab.md"

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

METRICS = ("VLF", "LF", "HF", "total_power", "LF/HF")
BANDS = {
    "VLF": (0.0, 0.04),
    "LF": (0.04, 0.15),
    "HF": (0.15, 0.4),
    "total_power": (0.0, 0.4),
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


def abs_err(reference: float, value: float) -> float:
    if not finite(reference) or not finite(value):
        return math.nan
    return abs(value - reference)


def rel_err(reference: float, value: float) -> float:
    if not finite(reference) or not finite(value) or abs(reference) <= 1e-12:
        return math.nan
    return abs(value - reference) / abs(reference) * 100.0


def integrate(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = np.isfinite(freqs) & np.isfinite(psd) & (freqs >= low) & (freqs <= high)
    if np.count_nonzero(mask) < 2:
        return math.nan
    return float(max(0.0, np.trapezoid(psd[mask], freqs[mask])))


def metrics_from_psd(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    vlf = integrate(freqs, psd, *BANDS["VLF"])
    lf = integrate(freqs, psd, *BANDS["LF"])
    hf = integrate(freqs, psd, *BANDS["HF"])
    total = integrate(freqs, psd, *BANDS["total_power"])
    lf_hf = lf / hf if finite(lf) and finite(hf) and hf > 1e-12 else math.nan
    return {"VLF": vlf, "LF": lf, "HF": hf, "total_power": total, "LF/HF": lf_hf}


def read_reference_values() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    kubios_rows = read_csv(PARSED_DIR / "kubios_parsed_results.csv")
    comparison_rows = read_csv(PARSED_DIR / "comparison_with_hrvstudio.csv")
    kubios: Dict[str, Dict[str, float]] = {}
    neurokit: Dict[str, Dict[str, float]] = {}

    for row in kubios_rows:
        subset = row["subset_id"]
        kubios[subset] = {
            "VLF": float(row["kubios_vlf"]),
            "LF": float(row["kubios_lf"]),
            "HF": float(row["kubios_hf"]),
            "total_power": float(row["kubios_total_power"]),
            "LF/HF": float(row["kubios_lf_hf"]),
        }

    for row in comparison_rows:
        subset = row["subset_id"]
        metric = row["metric"]
        if metric in METRICS:
            neurokit.setdefault(subset, {})[metric] = float(row["neurokit_value"])
    return kubios, neurokit


def compute_arm_metrics(path: Path) -> Dict[str, Any]:
    rr_ms = load_rr_intervals_ms(path)
    analyzer = HRVFreqDomainAnalysis(
        rr_ms,
        sampling_rate=4.0,
        detrend_method=None,
        window_type="hann",
        segment_length=120.0,
        overlap_ratio=0.75,
        ar_order=16,
        enable_diagnostics=True,
    )
    nperseg = 480
    noverlap = 360
    window = analyzer._get_window(nperseg)
    signal_s = analyzer.time_domain_s

    freqs_a, psd_a_s2 = signal.welch(
        signal_s - np.mean(signal_s),
        fs=4.0,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        scaling="density",
        average="mean",
    )
    freqs_b, psd_b_s2 = signal.welch(
        signal_s,
        fs=4.0,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend="constant",
        scaling="density",
        average="mean",
    )
    return {
        "arm_a": metrics_from_psd(freqs_a, psd_a_s2 * 1e6),
        "arm_b": metrics_from_psd(freqs_b, psd_b_s2 * 1e6),
        "diagnostics": analyzer.get_results().get("frequency_diagnostics", {}),
    }


def closer(reference: float, arm_a: float, arm_b: float) -> str:
    a_err = abs_err(reference, arm_a)
    b_err = abs_err(reference, arm_b)
    if not finite(a_err) and not finite(b_err):
        return "not available"
    if finite(a_err) and finite(b_err) and abs(a_err - b_err) <= 1e-9:
        return "tie"
    return "Arm A" if finite(a_err) and (not finite(b_err) or a_err < b_err) else "Arm B"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def mean_finite(values: Sequence[float]) -> float:
    finite_values = [float(v) for v in values if finite(v)]
    return float(np.mean(finite_values)) if finite_values else math.nan


def main() -> int:
    kubios, neurokit = read_reference_values()
    rows: List[Dict[str, Any]] = []
    diagnostics_rows = []
    for subset, path in FILES.items():
        result = compute_arm_metrics(path)
        diagnostics = result["diagnostics"]
        diagnostics_rows.append(
            [
                subset,
                diagnostics.get("invalid_rr_removed_count", 0),
                diagnostics.get("duplicate_time_points_detected", False),
                diagnostics.get("mean_removed_for_none_detrend", False),
                diagnostics.get("nonfinite_psd", False),
            ]
        )
        for metric in METRICS:
            ref = kubios[subset][metric]
            a_value = result["arm_a"][metric]
            b_value = result["arm_b"][metric]
            rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "kubios": ref,
                    "neurokit2": neurokit[subset][metric],
                    "arm_a": a_value,
                    "arm_b": b_value,
                    "arm_a_rel_err": rel_err(ref, a_value),
                    "arm_b_rel_err": rel_err(ref, b_value),
                    "closer": closer(ref, a_value, b_value),
                }
            )

    metric_summary = []
    for metric in METRICS:
        metric_rows = [row for row in rows if row["metric"] == metric]
        a_mean = mean_finite([row["arm_a_rel_err"] for row in metric_rows])
        b_mean = mean_finite([row["arm_b_rel_err"] for row in metric_rows])
        metric_summary.append(
            [
                metric,
                fmt(a_mean, 2),
                fmt(b_mean, 2),
                "Arm A" if finite(a_mean) and finite(b_mean) and a_mean < b_mean else "Arm B",
            ]
        )

    overall_a = mean_finite([row["arm_a_rel_err"] for row in rows])
    overall_b = mean_finite([row["arm_b_rel_err"] for row in rows])

    detail_rows = [
        [
            row["subset"],
            row["metric"],
            fmt(row["kubios"]),
            fmt(row["neurokit2"]),
            fmt(row["arm_a"]),
            fmt(row["arm_b"]),
            fmt(row["arm_a_rel_err"], 2),
            fmt(row["arm_b_rel_err"], 2),
            row["closer"],
        ]
        for row in rows
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# None-Detrend Convention A/B",
        "",
        "Validation-only experiment after the DC fix. Production code was not modified by this experiment.",
        "",
        "## Arms",
        "",
        "- Arm A: remove one global mean from the full interpolated RR signal, then run Welch with `detrend=False`.",
        "- Arm B: run Welch with `detrend=\"constant\"` per segment.",
        "- Shared settings: `fs=4 Hz`, `nperseg=480`, `noverlap=360`, `window=hann`, `nfft=480`, `scaling=density`, `average=mean`.",
        "- Bands: VLF `0.0-0.04`, LF `0.04-0.15`, HF `0.15-0.40`, total `0.0-0.40` Hz.",
        "",
        "## Results",
        "",
        *table(
            [
                "File",
                "Metric",
                "Kubios",
                "NeuroKit2",
                "Arm A global mean",
                "Arm B segment constant",
                "Arm A err %",
                "Arm B err %",
                "Closer to Kubios",
            ],
            detail_rows,
        ),
        "",
        "## Mean Relative Error vs Kubios",
        "",
        f"- Arm A overall mean relative error: {fmt(overall_a, 2)}%",
        f"- Arm B overall mean relative error: {fmt(overall_b, 2)}%",
        "",
        *table(
            ["Metric", "Arm A mean err %", "Arm B mean err %", "Lower mean error"],
            metric_summary,
        ),
        "",
        "## Input Diagnostics",
        "",
        *table(
            [
                "File",
                "Invalid RR removed",
                "Duplicate time points",
                "Mean removed diagnostic flag",
                "Nonfinite PSD",
            ],
            diagnostics_rows,
        ),
        "",
        "## Cautious Recommendation",
        "",
    ]

    if finite(overall_a) and finite(overall_b) and overall_a < overall_b:
        recommendation = (
            "Arm A is modestly closer to Kubios overall in this n=3 pilot. "
            "It better preserves the low-frequency magnitude seen in Kubios/NeuroKit2 "
            "for CH001 and OUT001 than per-segment constant detrending."
        )
    else:
        recommendation = (
            "Arm B is closer to Kubios overall in this n=3 pilot. It is also the "
            "current post-fix implementation convention."
        )
    lines.extend(
        [
            recommendation,
            "",
            "This should be treated as a convention-finding result, not a general accuracy claim. "
            "The pilot is only three files, and VLF001 still has invalid RR intervals that require "
            "cleanup before either arm can be compared fairly. If Kubios-compatible no-detrend "
            "behavior is the target, this evidence favors documenting and considering Arm A "
            "as the closer convention for `detrend_method=None`, then validating on a larger "
            "manual-export subset before changing production behavior again.",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
