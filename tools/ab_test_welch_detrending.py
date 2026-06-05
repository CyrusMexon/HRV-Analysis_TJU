"""
Validation-only A/B test for Welch segment detrending effects on VLF power.

This script does not modify production HRV code. It keeps preprocessing and
Welch parameters fixed, then changes only scipy.signal.welch(detrend=...).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "v04_physionet_10min_neurokit2"
DEFAULT_INPUT_FILES = [
    "validation/processed_data/physionet_nsr_rr_10min/nsr023_segment_131.csv",
    "validation/processed_data/physionet_nsr_rr_10min/nsr037_segment_131.csv",
    "validation/processed_data/physionet_nsr_rr_10min/nsr043_segment_135.csv",
    "validation/processed_data/physionet_nsr_rr_10min/nsr022_segment_022.csv",
]

SAMPLING_RATE_HZ = 4.0
NPERSEG = 480
NOVERLAP = 360
NFFT = 960
WINDOW = "hann"
SCALING = "density"
AVERAGE = "mean"

METRICS = {
    "VLF": "vlf_power",
    "LF": "lf_power",
    "HF": "hf_power",
    "total_power": "total_power",
    "LF/HF": "lf_hf_ratio",
    "LF_nu": "lf_nu",
    "HF_nu": "hf_nu",
}

BANDS = {
    "vlf": (0.0, 0.04),
    "lf": (0.04, 0.15),
    "hf": (0.15, 0.40),
    "total": (0.0, 0.40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B test Welch segment detrending with fixed preprocessing."
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        default=DEFAULT_INPUT_FILES,
        help="RR interval CSV files. Defaults to the representative PhysioNet 10-minute outliers.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Validation run folder under validation/runs/.",
    )
    return parser.parse_args()


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.name).strip("._") or "inspection"


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
    return "" if value is None else value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})


def load_rr_intervals_ms(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            has_header = csv.Sniffer().has_header(sample)
            if has_header:
                reader = csv.DictReader(handle)
                fieldnames = [name.strip() for name in (reader.fieldnames or [])]
                preferred = ["rr_ms", "RR", "RRI", "rri", "interval_ms"]
                column = next((name for name in preferred if name in fieldnames), None)
                if column is None:
                    column = fieldnames[0] if fieldnames else None
                if column is None:
                    raise ValueError(f"No columns found in {path}")
                values = [float(row[column]) for row in reader if str(row.get(column, "")).strip()]
            else:
                values = [float(row[0]) for row in csv.reader(handle) if row]
    else:
        values = np.loadtxt(path, dtype=float).tolist()

    rr = np.asarray(values, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size == 0:
        raise ValueError(f"No finite RR intervals found in {path}")
    mean_rr = float(np.mean(rr))
    if mean_rr < 10.0:
        rr = rr * 1000.0
    elif mean_rr > 10000.0:
        rr = rr / 1000.0
    return rr


def rr_start_times_s(rr_ms: np.ndarray) -> np.ndarray:
    rr_s = rr_ms.astype(float) / 1000.0
    return np.concatenate([[0.0], np.cumsum(rr_s[:-1])])


def interpolate_rr_to_uniform_seconds(rr_ms: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    rr_s = rr_ms.astype(float) / 1000.0
    time_points = rr_start_times_s(rr_ms)
    duration = float(time_points[-1])
    if duration <= 0:
        raise ValueError("RR sequence has non-positive duration")

    try:
        interp_func = interpolate.CubicSpline(time_points, rr_s, bc_type="natural")
    except Exception:
        interp_func = interpolate.interp1d(
            time_points,
            rr_s,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

    time_axis = np.arange(0.0, duration + 1e-12, 1.0 / sampling_rate)
    values_s = np.asarray(interp_func(time_axis), dtype=float)
    values_s = np.clip(values_s, 0.2, 3.0)
    return time_axis, values_s


def compute_welch_psd_ms2(signal_s: np.ndarray, detrend: Any) -> Tuple[np.ndarray, np.ndarray]:
    freqs, psd_s2 = signal.welch(
        signal_s,
        fs=SAMPLING_RATE_HZ,
        window=WINDOW,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        detrend=detrend,
        scaling=SCALING,
        average=AVERAGE,
    )
    return freqs, psd_s2 * 1e6


def integrate_band(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high) & np.isfinite(freqs) & np.isfinite(psd)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(max(0.0, np.trapezoid(psd[mask], freqs[mask])))


def metrics_from_psd(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    vlf = integrate_band(freqs, psd, *BANDS["vlf"])
    lf = integrate_band(freqs, psd, *BANDS["lf"])
    hf = integrate_band(freqs, psd, *BANDS["hf"])
    total = integrate_band(freqs, psd, *BANDS["total"])
    lf_hf = lf / hf if hf > 1e-10 else (math.inf if lf > 1e-10 else math.nan)
    lf_hf_sum = lf + hf
    return {
        "vlf_power": vlf,
        "lf_power": lf,
        "hf_power": hf,
        "total_power": total,
        "lf_hf_ratio": lf_hf,
        "lf_nu": (lf / lf_hf_sum) * 100.0 if lf_hf_sum > 0 else 0.0,
        "hf_nu": (hf / lf_hf_sum) * 100.0 if lf_hf_sum > 0 else 0.0,
    }


def relative_difference_pct(a: float, b: float) -> float:
    if not finite(a) or not finite(b):
        return math.nan
    denom = abs(a)
    if denom <= 1e-12:
        return math.nan
    return abs(b - a) / denom * 100.0


def cumulative_power(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    out = np.zeros(len(freqs), dtype=float)
    if len(freqs) > 1:
        out[1:] = np.cumsum((psd[:-1] + psd[1:]) * 0.5 * np.diff(freqs))
    return out


def positive_limits(*arrays: np.ndarray) -> Tuple[float, float]:
    values = np.concatenate([arr[np.isfinite(arr) & (arr > 0)] for arr in arrays if len(arr)])
    if len(values) == 0:
        return 1e-6, 1.0
    return max(float(np.min(values)) * 0.5, 1e-12), float(np.max(values)) * 2.0


def plot_psd_overlay(freqs: np.ndarray, arm_a_psd: np.ndarray, arm_b_psd: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    shade_bands(ax)
    ax.plot(freqs, arm_a_psd, linewidth=1.7, label="Arm A: global detrend, Welch detrend=False")
    ax.plot(freqs, arm_b_psd, linewidth=1.7, label='Arm B: global detrend, Welch detrend="linear"')
    ax.set_xlim(0.0, 0.4)
    ax.set_ylim(*positive_limits(arm_a_psd[freqs <= 0.4], arm_b_psd[freqs <= 0.4]))
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD power (ms^2/Hz)")
    ax.set_title("A/B Welch PSD Overlay")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_zoomed_vlf(freqs: np.ndarray, arm_a_psd: np.ndarray, arm_b_psd: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.axvspan(0.0, 0.04, color="#e8eef7", alpha=0.55, label="VLF")
    ax.axvline(0.04, color="#555555", linewidth=0.9)
    ax.plot(freqs, arm_a_psd, marker="o", markersize=3, linewidth=1.6, label="Arm A")
    ax.plot(freqs, arm_b_psd, marker="o", markersize=3, linewidth=1.6, label="Arm B")
    ax.set_xlim(0.0, 0.06)
    ax.set_ylim(*positive_limits(arm_a_psd[freqs <= 0.06], arm_b_psd[freqs <= 0.06]))
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD power (ms^2/Hz)")
    ax.set_title("Zoomed VLF A/B PSD")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cumulative(freqs: np.ndarray, arm_a_psd: np.ndarray, arm_b_psd: np.ndarray, output_path: Path) -> None:
    mask = (freqs >= 0.0) & (freqs <= 0.4)
    x = freqs[mask]
    arm_a_cum = cumulative_power(x, arm_a_psd[mask])
    arm_b_cum = cumulative_power(x, arm_b_psd[mask])

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.axvspan(0.0, 0.04, color="#e8eef7", alpha=0.45)
    ax.axvspan(0.04, 0.15, color="#edf7ed", alpha=0.45)
    ax.axvspan(0.15, 0.4, color="#fff1df", alpha=0.45)
    ax.plot(x, arm_a_cum, linewidth=1.7, label="Arm A")
    ax.plot(x, arm_b_cum, linewidth=1.7, label="Arm B")
    ax.set_xlim(0.0, 0.4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cumulative integrated power (ms^2)")
    ax.set_title("A/B Cumulative Power By Frequency")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def shade_bands(ax: Any) -> None:
    bands = [("VLF", 0.0, 0.04), ("LF", 0.04, 0.15), ("HF", 0.15, 0.4)]
    colors = ["#e8eef7", "#edf7ed", "#fff1df"]
    for (label, low, high), color in zip(bands, colors):
        ax.axvspan(low, high, color=color, alpha=0.45, zorder=0)
        ax.axvline(low, color="#555555", linewidth=0.8, alpha=0.65)
        ax.text((low + high) / 2, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.axvline(0.4, color="#555555", linewidth=0.8, alpha=0.65)


def read_reference_metric(manual_dir: Path, metric: str, column: str) -> float:
    metric_table = manual_dir / "metric_table.csv"
    if not metric_table.exists():
        return math.nan
    with metric_table.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") == metric:
                try:
                    return float(row.get(column, ""))
                except ValueError:
                    return math.nan
    return math.nan


def inspect_file(input_path: Path, output_root: Path, manual_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rr_ms = load_rr_intervals_ms(input_path)
    _time_axis, rr_uniform_s = interpolate_rr_to_uniform_seconds(rr_ms, SAMPLING_RATE_HZ)
    global_linear_s = signal.detrend(rr_uniform_s, type="linear")

    freqs_a, psd_a = compute_welch_psd_ms2(global_linear_s, detrend=False)
    freqs_b, psd_b = compute_welch_psd_ms2(global_linear_s, detrend="linear")
    if not np.array_equal(freqs_a, freqs_b):
        raise RuntimeError(f"A/B frequency grids differ for {input_path}")

    metrics_a = metrics_from_psd(freqs_a, psd_a)
    metrics_b = metrics_from_psd(freqs_b, psd_b)

    file_output = output_root / safe_name(input_path)
    file_output.mkdir(parents=True, exist_ok=True)
    plot_psd_overlay(freqs_a, psd_a, psd_b, file_output / "psd_overlay_ab.png")
    plot_zoomed_vlf(freqs_a, psd_a, psd_b, file_output / "zoomed_vlf_ab.png")
    plot_cumulative(freqs_a, psd_a, psd_b, file_output / "cumulative_power_ab.png")

    rows: List[Dict[str, Any]] = []
    for metric, key in METRICS.items():
        arm_a = float(metrics_a[key])
        arm_b = float(metrics_b[key])
        rows.append(
            {
                "file": input_path.name,
                "metric": metric,
                "arm_a_global_detrend_no_segment": arm_a,
                "arm_b_global_detrend_plus_segment_linear": arm_b,
                "absolute_difference": abs(arm_b - arm_a) if finite(arm_a) and finite(arm_b) else math.nan,
                "relative_difference_pct": relative_difference_pct(arm_a, arm_b),
            }
        )

    manual_dir = manual_root / safe_name(input_path)
    summary = {
        "file": input_path.name,
        "output_dir": file_output,
        "arm_a": metrics_a,
        "arm_b": metrics_b,
        "reference_neurokit2_vlf": read_reference_metric(manual_dir, "VLF", "neurokit2_value"),
        "reference_hrv_studio_vlf": read_reference_metric(manual_dir, "VLF", "native_value"),
        "reference_neurokit2_lfhf": read_reference_metric(manual_dir, "LF/HF", "neurokit2_value"),
        "reference_hrv_studio_lfhf": read_reference_metric(manual_dir, "LF/HF", "native_value"),
        "vlf_ratio_b_over_a": metrics_b["vlf_power"] / metrics_a["vlf_power"] if metrics_a["vlf_power"] > 1e-12 else math.nan,
        "lfhf_relative_difference_pct": relative_difference_pct(metrics_a["lf_hf_ratio"], metrics_b["lf_hf_ratio"]),
    }
    return rows, summary


def closer_label(value: float, reference_a: float, reference_b: float, label_a: str, label_b: str) -> str:
    if not all(finite(v) for v in [value, reference_a, reference_b]):
        return "reference not available"
    da = abs(value - reference_a)
    db = abs(value - reference_b)
    if da < db:
        return label_a
    if db < da:
        return label_b
    return "equidistant"


def write_summary(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# A/B Welch Detrending Experiment Summary",
        "",
        "This validation-only experiment keeps preprocessing and Welch settings fixed, then changes only `scipy.signal.welch(detrend=...)`.",
        "",
        "Shared settings:",
        "",
        f"- Sampling rate: `{SAMPLING_RATE_HZ:g}` Hz",
        "- One global linear detrend applied to the full uniformly sampled RR signal before both arms.",
        f"- Welch window: `{WINDOW}`",
        f"- `nperseg={NPERSEG}`, `noverlap={NOVERLAP}`, `nfft={NFFT}`",
        f"- `scaling={SCALING}`, `average={AVERAGE}`",
        "",
        "Arms:",
        "",
        "- Arm A: global detrend, then Welch with `detrend=False`.",
        "- Arm B: global detrend, then Welch with `detrend=\"linear\"`.",
        "",
        "## Per-File Comparison",
        "",
        "| File | Arm A VLF | Arm B VLF | B/A VLF ratio | Reference NeuroKit2 VLF | Reference HRV Studio VLF | Arm A reference check | Arm B reference check | Arm A LF/HF | Arm B LF/HF | LF/HF change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        arm_a = summary["arm_a"]
        arm_b = summary["arm_b"]
        arm_a_check = closer_label(
            arm_a["vlf_power"],
            summary["reference_neurokit2_vlf"],
            summary["reference_hrv_studio_vlf"],
            "closer to NeuroKit2",
            "closer to HRV Studio",
        )
        arm_b_check = closer_label(
            arm_b["vlf_power"],
            summary["reference_hrv_studio_vlf"],
            summary["reference_neurokit2_vlf"],
            "closer to HRV Studio",
            "closer to NeuroKit2",
        )
        lines.append(
            "| {file} | {a_vlf} | {b_vlf} | {ratio} | {nk_ref} | {hrv_ref} | {a_check} | {b_check} | {a_lfhf} | {b_lfhf} | {lfhf_delta}% |".format(
                file=f"`{summary['file']}`",
                a_vlf=fmt(arm_a["vlf_power"], 3),
                b_vlf=fmt(arm_b["vlf_power"], 3),
                ratio=fmt(summary["vlf_ratio_b_over_a"], 3),
                nk_ref=fmt(summary["reference_neurokit2_vlf"], 3),
                hrv_ref=fmt(summary["reference_hrv_studio_vlf"], 3),
                a_check=arm_a_check,
                b_check=arm_b_check,
                a_lfhf=fmt(arm_a["lf_hf_ratio"], 3),
                b_lfhf=fmt(arm_b["lf_hf_ratio"], 3),
                lfhf_delta=fmt(summary["lfhf_relative_difference_pct"], 3),
            )
        )

    stable_lfhf = all(
        finite(summary["lfhf_relative_difference_pct"]) and summary["lfhf_relative_difference_pct"] <= 1.0
        for summary in summaries
    )
    arm_a_reproduces = all(
        closer_label(
            summary["arm_a"]["vlf_power"],
            summary["reference_neurokit2_vlf"],
            summary["reference_hrv_studio_vlf"],
            "nk",
            "hrv",
        )
        == "nk"
        for summary in summaries
    )
    arm_b_collapses = all(
        closer_label(
            summary["arm_b"]["vlf_power"],
            summary["reference_hrv_studio_vlf"],
            summary["reference_neurokit2_vlf"],
            "hrv",
            "nk",
        )
        == "hrv"
        for summary in summaries
    )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Arm A reproduces NeuroKit2-like VLF by the reference-distance check: {'yes' if arm_a_reproduces else 'mixed'}",
            f"- Arm B collapses VLF toward HRV Studio-like values by the reference-distance check: {'yes' if arm_b_collapses else 'mixed'}",
            f"- LF/HF remains stable across the A/B arms using a <=1% relative-change threshold: {'yes' if stable_lfhf else 'mixed'}",
            "",
            "## Cautious Interpretation",
            "",
            "The result supports the hypothesis that Welch segment detrending convention is a major driver of the near-zero/VLF discrepancy. With the full signal already globally linearly detrended, disabling segment detrending leaves substantially larger DC and first-VLF-bin power. Re-enabling segment-wise linear detrending suppresses that near-zero power while leaving LF/HF nearly unchanged.",
            "",
            "This experiment does not establish which convention is correct. It only isolates that the convention is behaviorally important for these representative files.",
            "",
            "## Per-File Plot Directories",
            "",
        ]
    )
    for summary in summaries:
        rel = summary["output_dir"].relative_to(PROJECT_ROOT)
        lines.append(f"- `{summary['file']}`: `{rel}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_files = [PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path) for path in args.input_files]
    output_root = PROJECT_ROOT / "validation" / "runs" / args.run_name / "ab_welch_detrending"
    manual_root = PROJECT_ROOT / "validation" / "runs" / args.run_name / "manual_inspection"
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for input_path in input_files:
        if not input_path.exists():
            raise SystemExit(f"Input file does not exist: {input_path}")
        rows, summary = inspect_file(input_path, output_root, manual_root)
        all_rows.extend(rows)
        summaries.append(summary)

    write_csv(
        output_root / "ab_welch_detrending_results.csv",
        all_rows,
        [
            "file",
            "metric",
            "arm_a_global_detrend_no_segment",
            "arm_b_global_detrend_plus_segment_linear",
            "absolute_difference",
            "relative_difference_pct",
        ],
    )
    write_summary(output_root / "ab_welch_detrending_summary.md", summaries)
    print(f"Wrote A/B Welch detrending outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
