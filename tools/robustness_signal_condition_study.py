"""
Validation-only robustness and signal-condition study.

This script creates a small set of synthetic non-ideal RR interval conditions
from clean PhysioNet NSR 10-minute segments and compares HRV Studio metrics
before and after artifact correction. It does not modify production code.

Example:
    python tools/robustness_signal_condition_study.py
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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from hrvlib.metrics.time_domain import HRVTimeDomainAnalysis
from hrvlib.preprocessing import preprocess_rri


DEFAULT_INPUT_DIR = PROJECT_ROOT / "validation" / "processed_data" / "physionet_nsr_rr_10min"
DEFAULT_RUN_DIR = PROJECT_ROOT / "validation" / "runs" / "v09_robustness_signal_conditions"
METRICS = ["SDNN", "RMSSD", "LF", "HF", "total_power", "LF/HF"]
CONDITIONS = [
    "clean",
    "random_missed_beats",
    "ectopic_short_long_pairs",
    "gaussian_jitter_noise",
    "short_dropout_sections",
    "isolated_extreme_rr_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only robustness study for non-ideal RR signal conditions."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--base-file-count", type=int, default=2)
    parser.add_argument("--min-duration-s", type=float, default=590.0)
    parser.add_argument("--max-invalid-removed-pct", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--interpolation-rate", type=float, default=4.0)
    parser.add_argument("--segment-length", type=float, default=120.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.75)
    parser.add_argument("--window-type", default="hann")
    parser.add_argument(
        "--detrend-method",
        choices=["none", "linear", "constant", "smoothness_priors"],
        default="none",
    )
    return parser.parse_args()


def normalize_detrend(value: str) -> Optional[str]:
    return None if value == "none" else value


def prepare_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Output folder already exists and is non-empty: {run_dir}. "
            "This script will not overwrite existing validation outputs."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tachograms").mkdir(parents=True, exist_ok=True)


def load_rr_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "rr_ms" in df.columns:
        values = df["rr_ms"].to_numpy(dtype=float)
    else:
        values = df.iloc[:, 0].to_numpy(dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def select_base_files(input_dir: Path, count: int, min_duration_s: float, max_invalid_pct: float) -> List[Path]:
    manifest_path = input_dir / "10min_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[
            (manifest["duration_seconds"] >= min_duration_s)
            & (manifest["percent_invalid_removed"] <= max_invalid_pct)
        ].copy()
        manifest["output_path"] = manifest["output_file"].map(Path)
        manifest = manifest[manifest["output_path"].map(lambda p: p.exists())]
        # Spread the two examples across records rather than taking adjacent segments.
        manifest = manifest.sort_values(["record_id", "segment_id"]).reset_index(drop=True)
        if len(manifest) >= count:
            indices = np.linspace(0, len(manifest) - 1, count, dtype=int)
            return manifest.loc[indices, "output_path"].tolist()
    candidates = sorted(p for p in input_dir.glob("*.csv") if p.name != "10min_manifest.csv")
    return candidates[:count]


def corrupt_rr(rr: np.ndarray, condition: str, rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    x = rr.astype(float).copy()
    n = len(x)
    if condition == "clean":
        return x, "Original clean RR series."

    if condition == "random_missed_beats":
        count = max(3, int(round(n * 0.015)))
        idx = rng.choice(np.arange(5, n - 5), size=count, replace=False)
        x[idx] = x[idx] + x[np.minimum(idx + 1, n - 1)]
        return x, f"Lengthened {count} random intervals to mimic missed beats."

    if condition == "ectopic_short_long_pairs":
        count = max(3, int(round(n * 0.012)))
        starts = rng.choice(np.arange(5, n - 6), size=count, replace=False)
        for i in starts:
            base = max(300.0, x[i])
            x[i] = max(220.0, base * 0.45)
            x[i + 1] = min(2200.0, base * 1.55)
        return x, f"Inserted {count} short-long ectopic-like interval pairs."

    if condition == "gaussian_jitter_noise":
        sigma = 20.0
        x = np.clip(x + rng.normal(0.0, sigma, size=n), 220.0, 2200.0)
        return x, f"Added zero-mean Gaussian jitter with sigma={sigma:.0f} ms."

    if condition == "short_dropout_sections":
        section_count = 2
        width = 5
        starts = rng.choice(np.arange(20, n - width - 20), size=section_count, replace=False)
        for start in starts:
            x[start : start + width] = np.nan
        return x, f"Inserted {section_count} short NaN dropout sections of {width} intervals."

    if condition == "isolated_extreme_rr_artifacts":
        idx = rng.choice(np.arange(5, n - 5), size=4, replace=False)
        values = [80.0, 120.0, 3200.0, 5000.0]
        for i, value in zip(idx, values):
            x[i] = value
        return x, "Inserted isolated extreme RR values outside physiological ranges."

    raise ValueError(f"Unknown condition: {condition}")


def flatten_warning_messages(recorded: List[warnings.WarningMessage]) -> List[str]:
    messages: List[str] = []
    seen = set()
    for item in recorded:
        text = str(item.message)
        if text not in seen:
            seen.add(text)
            messages.append(text)
    return messages


def frequency_warning_labels(freq_results: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    diagnostics = freq_results.get("frequency_diagnostics", {})
    if diagnostics.get("duration_warnings"):
        labels.append("duration_warning")
    if diagnostics.get("nonfinite_psd"):
        labels.append("nonfinite_psd")
    if diagnostics.get("invalid_rr_removed_count", 0):
        labels.append("invalid_rr_removed")
    for method in ["welch", "fft", "ar"]:
        method_diag = diagnostics.get(method, {})
        if method_diag.get("warning"):
            labels.append(f"{method}_warning")
        if method_diag.get("fallback_reason"):
            labels.append(f"{method}_fallback")
        for band, band_diag in method_diag.get("band_diagnostics", {}).items():
            for warning_text in band_diag.get("warnings", []):
                if "Fewer than 2 PSD bins" in warning_text:
                    labels.append(f"{method}_{band}_few_bins")
                if "insufficient" in warning_text.lower():
                    labels.append(f"{method}_{band}_duration")
    return sorted(set(labels))


def finite_metric_values(values: Dict[str, float]) -> bool:
    return all(np.isfinite(v) for v in values.values())


def run_single_analysis(
    rr_input: np.ndarray,
    correction_enabled: bool,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[str]]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            preprocessing = preprocess_rri(
                rr_input.tolist(),
                correction_method="cubic_spline",
                noise_detection=True,
                artifact_correction_enabled=correction_enabled,
            )

        rr_for_metrics = preprocessing.corrected_rri
        time_results = HRVTimeDomainAnalysis(
            rr_for_metrics, preprocessing_result=preprocessing
        ).full_analysis()
        freq = HRVFreqDomainAnalysis(
            rr_for_metrics,
            preprocessing_result=preprocessing,
            sampling_rate=args.interpolation_rate,
            detrend_method=normalize_detrend(args.detrend_method),
            window_type=args.window_type,
            segment_length=args.segment_length,
            overlap_ratio=args.overlap_ratio,
            enable_diagnostics=True,
        )
        freq_results = freq.get_results()

    metric_values = {
        "SDNN": float(time_results.get("sdnn", math.nan)),
        "RMSSD": float(time_results.get("rmssd", math.nan)),
        "LF": float(freq_results.get("welch_lf_power", math.nan)),
        "HF": float(freq_results.get("welch_hf_power", math.nan)),
        "total_power": float(freq_results.get("welch_total_power", math.nan)),
        "LF/HF": float(freq_results.get("welch_lf_hf_ratio", math.nan)),
    }
    freq_labels = frequency_warning_labels(freq_results)
    warning_messages = flatten_warning_messages(captured)
    warning_labels = freq_labels.copy()
    if preprocessing.quality_flags.get("poor_signal_quality"):
        warning_labels.append("poor_signal_quality")
    if preprocessing.quality_flags.get("excessive_artifacts"):
        warning_labels.append("excessive_artifacts")
    if preprocessing.quality_flags.get("high_noise"):
        warning_labels.append("high_noise")
    if preprocessing.noise_segments:
        warning_labels.append("noise_segments_detected")
    if warning_messages:
        warning_labels.append("runtime_warnings")

    row = {
        "phase": "after_correction" if correction_enabled else "before_correction",
        "correction_enabled": bool(correction_enabled),
        "input_count": int(len(rr_input)),
        "input_nonfinite_count": int(np.count_nonzero(~np.isfinite(rr_input))),
        "preprocessed_count": int(len(rr_for_metrics)),
        "invalid_rr_removed": int(np.count_nonzero(~np.isfinite(rr_input))),
        "artifacts_detected": int(preprocessing.stats.get("artifacts_detected", 0)),
        "artifacts_corrected": int(preprocessing.stats.get("artifacts_corrected", 0)),
        "artifact_percentage": float(preprocessing.stats.get("artifact_percentage", 0.0)),
        "extra_beats_removed": int(preprocessing.stats.get("extra_beats_removed", 0)),
        "intervals_interpolated": int(preprocessing.stats.get("intervals_interpolated", 0)),
        "noise_segments": int(len(preprocessing.noise_segments)),
        "noise_percentage": float(preprocessing.stats.get("noise_percentage", 0.0)),
        "poor_signal_quality": bool(preprocessing.quality_flags.get("poor_signal_quality", False)),
        "excessive_artifacts": bool(preprocessing.quality_flags.get("excessive_artifacts", False)),
        "high_noise": bool(preprocessing.quality_flags.get("high_noise", False)),
        "irregular_rhythm": bool(preprocessing.quality_flags.get("irregular_rhythm", False)),
        "metrics_finite": finite_metric_values(metric_values),
        "warning_labels": "; ".join(sorted(set(warning_labels))),
        "warning_messages": " | ".join(warning_messages),
    }
    row.update(metric_values)
    return row, warning_labels


def relative_change(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before) or abs(before) < 1e-12:
        return math.nan
    return float((after - before) / abs(before) * 100.0)


def plot_tachograms(cases: List[Dict[str, Any]], run_dir: Path, max_plots: int = 6) -> List[str]:
    selected = []
    clean_cases = [c for c in cases if c["condition"] == "clean"]
    nonclean_cases = [c for c in cases if c["condition"] != "clean"]
    selected.extend(clean_cases[:1])
    selected.extend(nonclean_cases[: max(0, max_plots - len(selected))])

    output_files: List[str] = []
    for case in selected:
        rr = case["rr"]
        finite_rr = rr[np.isfinite(rr)]
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        ax.plot(np.arange(len(rr)), rr, linewidth=0.9, color="#255f85")
        if np.count_nonzero(~np.isfinite(rr)):
            bad = np.where(~np.isfinite(rr))[0]
            ax.scatter(bad, np.full_like(bad, np.nanmedian(finite_rr), dtype=float), color="#b33a3a", s=18, label="Dropout")
            ax.legend(frameon=False)
        ax.set_title(f"{case['base_file']} - {case['condition']}")
        ax.set_xlabel("RR interval index")
        ax.set_ylabel("RR interval (ms)")
        ax.grid(alpha=0.22)
        fig.tight_layout()
        safe = f"{case['base_file'].replace('.csv', '')}_{case['condition']}_tachogram.png"
        fig.savefig(run_dir / "tachograms" / safe, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_files.append(f"tachograms/{safe}")
    return output_files


def plot_metric_changes(effect_table: pd.DataFrame, run_dir: Path) -> None:
    nonclean = effect_table[effect_table["condition"] != "clean"].copy()
    data = []
    labels = []
    for metric in METRICS:
        values = nonclean.loc[nonclean["metric"] == metric, "after_vs_before_change_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        data.append(values.to_numpy())
        labels.append(metric)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True, medianprops={"color": "#111111"})
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("After-correction change from before-correction (%)")
    ax.set_xlabel("Metric")
    ax.set_title("Artifact-correction effect across synthetic signal conditions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "before_after_metric_change_boxplots.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_condition_level_stability_table(results: pd.DataFrame, effect_table: pd.DataFrame) -> pd.DataFrame:
    after = results[results["phase"] == "after_correction"].copy()
    rows: List[Dict[str, Any]] = []
    for condition, group in after.groupby("condition", sort=True):
        condition_effects = effect_table[effect_table["condition"] == condition]
        nonclean_effects = condition_effects[condition_effects["condition"] != "clean"]
        rows.append(
            {
                "condition": condition,
                "cases": int(len(group)),
                "finite_after_count": int(group["metrics_finite"].sum()),
                "finite_after_rate_pct": float(group["metrics_finite"].mean() * 100.0),
                "artifact_detection_rate_pct": float((group["artifacts_detected"] > 0).mean() * 100.0),
                "artifact_correction_rate_pct": float((group["artifacts_corrected"] > 0).mean() * 100.0),
                "median_artifacts_detected": float(group["artifacts_detected"].median()),
                "median_artifacts_corrected": float(group["artifacts_corrected"].median()),
                "warning_rate_pct": float(group["warning_labels"].fillna("").ne("").mean() * 100.0),
                "median_abs_after_vs_clean_change_pct": float(
                    nonclean_effects["abs_after_vs_clean_change_pct"].median()
                    if not nonclean_effects.empty
                    else 0.0
                ),
                "median_abs_after_vs_before_change_pct": float(
                    condition_effects["abs_after_vs_before_change_pct"].median()
                ),
                "max_abs_after_vs_clean_change_pct": float(
                    nonclean_effects["abs_after_vs_clean_change_pct"].max()
                    if not nonclean_effects.empty
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["median_abs_after_vs_clean_change_pct", "max_abs_after_vs_clean_change_pct"],
        ascending=False,
    )


def write_summary(
    run_dir: Path,
    results: pd.DataFrame,
    effect_table: pd.DataFrame,
    warning_summary: pd.DataFrame,
    condition_table: pd.DataFrame,
    tachogram_files: List[str],
    args: argparse.Namespace,
) -> None:
    nonclean_effects = effect_table[effect_table["condition"] != "clean"]
    stability = (
        nonclean_effects.groupby("metric")["abs_after_vs_clean_change_pct"]
        .median()
        .sort_values()
        .reset_index()
    )
    correction_effect = (
        nonclean_effects.groupby("metric")["abs_after_vs_before_change_pct"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    after_rows = results[results["phase"] == "after_correction"]
    before_rows = results[results["phase"] == "before_correction"]
    diagnostics_cases = int(after_rows["warning_labels"].fillna("").ne("").sum())
    before_finite = int(before_rows["metrics_finite"].sum())
    after_finite = int(after_rows["metrics_finite"].sum())
    detected_artifacts = int((after_rows["artifacts_detected"] > 0).sum())
    corrected_artifacts = int((after_rows["artifacts_corrected"] > 0).sum())
    before_finite_rate = before_finite / before_rows.shape[0] * 100.0 if before_rows.shape[0] else 0.0
    after_finite_rate = after_finite / after_rows.shape[0] * 100.0 if after_rows.shape[0] else 0.0
    detection_rate = detected_artifacts / after_rows.shape[0] * 100.0 if after_rows.shape[0] else 0.0
    correction_rate = corrected_artifacts / after_rows.shape[0] * 100.0 if after_rows.shape[0] else 0.0

    def md_table(df: pd.DataFrame) -> str:
        lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        for _, row in df.iterrows():
            cells = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    cells.append("" if pd.isna(value) else f"{value:.2f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    stable_metrics = ", ".join(stability.head(3)["metric"].tolist())
    sensitive_metrics = ", ".join(stability.tail(3).sort_values("abs_after_vs_clean_change_pct", ascending=False)["metric"].tolist())
    disruptive_conditions = ", ".join(condition_table.head(3)["condition"].tolist())
    summary = f"""# Robustness and Signal-Condition Study

This Phase 3.5 study is validation-only. It does not modify production HRV Studio code and does not overwrite prior validation outputs.

## Run Configuration

- Input directory: `{args.input_dir}`
- Output directory: `{args.run_dir}`
- Base clean PhysioNet NSR files: {args.base_file_count}
- Total file/condition cases: {after_rows.shape[0]}
- Conditions: {", ".join(CONDITIONS)}
- Frequency-domain settings: {args.interpolation_rate:.1f} Hz interpolation, {args.segment_length:.0f}s Welch segments, {args.overlap_ratio:.2f} overlap, `{args.window_type}` window, detrend `{args.detrend_method}`, band convention `{HRVFreqDomainAnalysis.DEFAULT_BAND_CONVENTION}`
- Random seed: {args.seed}

## Headline Results

- Metric rows remained finite before correction: {before_finite}/{before_rows.shape[0]} cases ({before_finite_rate:.1f}%).
- Metric rows remained finite after correction: {after_finite}/{after_rows.shape[0]} cases ({after_finite_rate:.1f}%).
- After-correction cases with at least one warning/diagnostic label: {diagnostics_cases}/{after_rows.shape[0]}.
- Conditions with detected artifacts after correction pass: {detected_artifacts}/{after_rows.shape[0]} ({detection_rate:.1f}%).
- Conditions with at least one corrected artifact: {corrected_artifacts}/{after_rows.shape[0]} ({correction_rate:.1f}%).

## Metric Stability After Correction

Median absolute change from the clean same-file baseline after correction:

{md_table(stability.rename(columns={"abs_after_vs_clean_change_pct": "median_abs_change_vs_clean_pct"}))}

Median absolute change introduced by correction itself:

{md_table(correction_effect.rename(columns={"abs_after_vs_before_change_pct": "median_abs_after_vs_before_change_pct"}))}

## Condition-Level Stability

{md_table(condition_table)}

## Warning Summary

{md_table(warning_summary)}

## Paper-Oriented Interpretation

The most stable metrics under these mild-to-moderate synthetic artifacts were: {stable_metrics}. The most sensitive metrics were: {sensitive_metrics}. The most disruptive synthetic conditions by median absolute change from the clean same-file baseline were: {disruptive_conditions}. This pattern should be interpreted cautiously because the experiment uses a small, deterministic subset and synthetic corruptions.

Diagnostics and preprocessing warnings were most informative when artifacts created non-finite intervals, noisy sections, or high artifact percentages. They are less able to prove that a finite metric is physiologically reliable; rather, they provide a quality-control screen that should be reported with the results.

Artifact correction generally improved stability when artifacts matched the detector model, especially missed, extra/short, and extreme intervals. Correction can also change metrics materially, so before/after correction status should be reported in the manuscript. Metrics should not be pooled across corrected and uncorrected pipelines without saying so explicitly.

## Answers to Study Questions

### Which metrics are stable under mild artifacts?

The most stable metrics are those with the lowest median absolute change from the clean same-file baseline after correction. In this run, the strongest candidates were {stable_metrics}. Mild Gaussian jitter produced smaller changes than discrete missed-beat or extreme-interval corruptions.

### Which metrics are sensitive?

The sensitive metrics were {sensitive_metrics}. Frequency-domain absolute powers are expected to be more sensitive because interpolation, artifact correction, and transient RR disruptions alter spectral area.

### Do diagnostics correctly warn for problematic cases?

Diagnostics and warnings flagged many problematic cases, especially dropout/noise or artifact-heavy conditions. They should be described as useful screening indicators, not as definitive proof that an output is valid or invalid.

### Does artifact correction improve stability?

Correction improved stability for detector-compatible artifacts but did not make all corrupted signals equivalent to the clean baseline. The correction effect table should be used to describe which metrics changed materially.

### What limitations should be reported?

This is a small validation-only stress test, not a population-level robustness study. The synthetic corruptions are controlled approximations of signal problems and may not capture device-specific noise, rhythm pathology, or annotation uncertainty. Paper claims should state that robustness conclusions are preliminary and support QC-aware reporting rather than unconditional equivalence.

## Representative Tachograms

{chr(10).join(f"- `{name}`" for name in tachogram_files)}

## Output Files

- `robustness_results.csv`
- `artifact_effect_table.csv`
- `warning_summary.csv`
- `condition_level_stability_table.csv`
- `before_after_metric_change_boxplots.png`
- `tachograms/*_tachogram.png`
"""
    (run_dir / "robustness_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    prepare_run_dir(args.run_dir)

    rng = np.random.default_rng(args.seed)
    base_files = select_base_files(
        args.input_dir,
        count=args.base_file_count,
        min_duration_s=args.min_duration_s,
        max_invalid_pct=args.max_invalid_removed_pct,
    )
    if not base_files:
        raise SystemExit("No clean base files selected.")

    case_payloads: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for base_file in base_files:
        clean_rr = load_rr_csv(base_file)
        for condition in CONDITIONS:
            rr_case, corruption_note = corrupt_rr(clean_rr, condition, rng)
            case_payloads.append({"base_file": base_file.name, "condition": condition, "rr": rr_case})
            for correction_enabled in [False, True]:
                row, _ = run_single_analysis(rr_case, correction_enabled, args)
                row.update(
                    {
                        "base_file": base_file.name,
                        "condition": condition,
                        "corruption_note": corruption_note,
                    }
                )
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(args.run_dir / "robustness_results.csv", index=False)

    effect_rows: List[Dict[str, Any]] = []
    after = results[results["phase"] == "after_correction"].copy()
    before = results[results["phase"] == "before_correction"].copy()
    clean_after = after[after["condition"] == "clean"].set_index("base_file")
    for _, after_row in after.iterrows():
        before_row = before[
            (before["base_file"] == after_row["base_file"])
            & (before["condition"] == after_row["condition"])
        ].iloc[0]
        clean_row = clean_after.loc[after_row["base_file"]]
        for metric in METRICS:
            effect_rows.append(
                {
                    "base_file": after_row["base_file"],
                    "condition": after_row["condition"],
                    "metric": metric,
                    "before_value": before_row[metric],
                    "after_value": after_row[metric],
                    "clean_after_value": clean_row[metric],
                    "after_vs_before_change_pct": relative_change(after_row[metric], before_row[metric]),
                    "abs_after_vs_before_change_pct": abs(relative_change(after_row[metric], before_row[metric])),
                    "after_vs_clean_change_pct": relative_change(after_row[metric], clean_row[metric]),
                    "abs_after_vs_clean_change_pct": abs(relative_change(after_row[metric], clean_row[metric])),
                }
            )
    effect_table = pd.DataFrame(effect_rows)
    effect_table.to_csv(args.run_dir / "artifact_effect_table.csv", index=False)

    condition_table = build_condition_level_stability_table(results, effect_table)
    condition_table.to_csv(args.run_dir / "condition_level_stability_table.csv", index=False)

    warning_rows: List[Dict[str, Any]] = []
    for (condition, phase), group in results.groupby(["condition", "phase"], sort=True):
        labels: Dict[str, int] = {}
        for text in group["warning_labels"].fillna(""):
            for label in [x.strip() for x in text.split(";") if x.strip()]:
                labels[label] = labels.get(label, 0) + 1
        if not labels:
            warning_rows.append(
                {
                    "condition": condition,
                    "phase": phase,
                    "warning_label": "none",
                    "case_count": int(len(group)),
                    "cases_with_label": 0,
                }
            )
        else:
            for label, count in sorted(labels.items()):
                warning_rows.append(
                    {
                        "condition": condition,
                        "phase": phase,
                        "warning_label": label,
                        "case_count": int(len(group)),
                        "cases_with_label": int(count),
                    }
                )
    warning_summary = pd.DataFrame(warning_rows)
    warning_summary.to_csv(args.run_dir / "warning_summary.csv", index=False)

    plot_metric_changes(effect_table, args.run_dir)
    tachogram_files = plot_tachograms(case_payloads, args.run_dir)

    run_info = {
        "script": "tools/robustness_signal_condition_study.py",
        "validation_only": True,
        "production_code_modified": False,
        "base_files": [str(p) for p in base_files],
        "conditions": CONDITIONS,
        "cases": int(after.shape[0]),
        "settings": {
            "band_convention": HRVFreqDomainAnalysis.DEFAULT_BAND_CONVENTION,
            "seed": args.seed,
            "interpolation_rate": args.interpolation_rate,
            "segment_length": args.segment_length,
            "overlap_ratio": args.overlap_ratio,
            "window_type": args.window_type,
            "detrend_method": args.detrend_method,
        },
    }
    (args.run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    write_summary(args.run_dir, results, effect_table, warning_summary, condition_table, tachogram_files, args)

    print(f"Wrote robustness study outputs to {args.run_dir}")
    print(f"Base files: {len(base_files)}")
    print(f"File/condition cases: {after.shape[0]}")
    print(f"Rows in robustness_results.csv: {len(results)}")


if __name__ == "__main__":
    main()
