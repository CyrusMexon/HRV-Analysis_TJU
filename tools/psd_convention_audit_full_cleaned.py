"""
Full cleaned-subset audit for the leading Hann coherent-gain diagnostic scale.

Validation-only helper. Reads the cleaned Kubios comparison CSV, recomputes a
single diagnostic Welch variant, and writes research artifacts only.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from psd_convention_audit import METRICS, finite_float, integrate_band, rel_error_pct
from psd_convention_audit import Variant
from validate_freq_domain_neurokit2 import effective_welch_params, load_rr_intervals_ms


DEFAULT_CLEANED_CSV = (
    PROJECT_ROOT
    / "validation"
    / "research_notes"
    / "manual_review_sensitivity_analysis"
    / "cleaned_valid_only.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "validation" / "research_notes" / "psd_convention_audit_full_cleaned"
)

POWER_METRICS = ("VLF", "LF", "HF", "total_power")
RATIO_METRICS = ("LF/HF", "LF_nu", "HF_nu")
BANDS = {
    "VLF": (0.0, 0.04),
    "LF": (0.04, 0.15),
    "HF": (0.15, 0.40),
    "total_power": (0.0, 0.40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full cleaned-subset Hann coherent-gain PSD diagnostic audit."
    )
    parser.add_argument("--cleaned-csv", default=str(DEFAULT_CLEANED_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true", help="Allow overwriting this audit folder.")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    return "" if value is None else value


def fmt(value: Any, digits: int = 2) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def ensure_no_overwrite(output_dir: Path, force: bool) -> None:
    targets = [
        output_dir / "full_cleaned_hann_scale_summary.md",
        output_dir / "full_cleaned_hann_scale_results.csv",
        output_dir / "before_after_hann_scale_by_metric.csv",
        output_dir / "remaining_outliers_after_hann_scale.csv",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise SystemExit(
            "Refusing to overwrite existing full-cleaned audit outputs. Re-run with --force if intended:\n"
            + "\n".join(str(path) for path in existing)
        )


def rows_by_file(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("subset_id", "")].append(row)
    return {key: value for key, value in grouped.items() if key}


def clean_signal_ms(signal_s: np.ndarray) -> np.ndarray:
    signal_ms = np.asarray(signal_s, dtype=float) * 1000.0
    signal_ms = signal_ms[np.isfinite(signal_ms)]
    if signal_ms.size == 0:
        return signal_ms
    return signal_ms - float(np.mean(signal_ms))


def hann_coherent_factor(nperseg: int) -> float:
    window = signal.get_window("hann", nperseg)
    return float(np.mean(window) ** 2)


def compute_hann_scaled_metrics(input_path: Path) -> Dict[str, float]:
    rr_ms = load_rr_intervals_ms(input_path)
    analyzer = HRVFreqDomainAnalysis(
        rr_ms,
        sampling_rate=4.0,
        detrend_method=None,
        window_type="hann",
        segment_length=120.0,
        overlap_ratio=0.75,
        enable_diagnostics=True,
    )
    analyzer.get_results()
    nperseg, noverlap = effective_welch_params(analyzer)
    if nperseg is None or noverlap is None:
        return {metric: math.nan for metric in METRICS}
    signal_ms = clean_signal_ms(analyzer.time_domain_s)
    nperseg = min(int(nperseg), len(signal_ms))
    noverlap = min(int(noverlap), max(0, nperseg - 1))
    freqs, psd = signal.welch(
        signal_ms,
        fs=4.0,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        scaling="density",
        average="mean",
    )
    psd = psd * hann_coherent_factor(nperseg)
    variant = Variant("hann_coherent_multiply")
    vlf = integrate_band(freqs, psd, *BANDS["VLF"], variant)
    lf = integrate_band(freqs, psd, *BANDS["LF"], variant)
    hf = integrate_band(freqs, psd, *BANDS["HF"], variant)
    total = integrate_band(freqs, psd, *BANDS["total_power"], variant)
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


def mean_finite(values: Iterable[Any]) -> float:
    finite = [finite_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def median_finite(values: Iterable[Any]) -> float:
    finite = [finite_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return median(finite) if finite else math.nan


def stdev_finite(values: Iterable[Any]) -> float:
    finite = [finite_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return stdev(finite) if len(finite) > 1 else 0.0 if finite else math.nan


def percent_improvement(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or before == 0:
        return math.nan
    return (before - after) / abs(before) * 100.0


def build_results(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped = rows_by_file(rows)
    result_rows: List[Dict[str, Any]] = []
    for subset_id, file_rows in sorted(grouped.items()):
        first = file_rows[0]
        input_path = project_path(first["kubios_input_txt"])
        diagnostic = compute_hann_scaled_metrics(input_path)
        for row in file_rows:
            metric = row["metric"]
            kubios = finite_float(row["kubios_value"])
            baseline = finite_float(row["hrvstudio_native_value"])
            scaled = finite_float(diagnostic.get(metric))
            before_error = rel_error_pct(baseline, kubios)
            after_error = rel_error_pct(scaled, kubios)
            result_rows.append(
                {
                    "subset_id": subset_id,
                    "category": first.get("category", ""),
                    "source_file": first.get("source_file", ""),
                    "kubios_input_txt": first.get("kubios_input_txt", ""),
                    "metric": metric,
                    "kubios_value": kubios,
                    "baseline_hrvstudio_value": baseline,
                    "hann_coherent_multiply_value": scaled,
                    "before_relative_error_pct": before_error,
                    "after_relative_error_pct": after_error,
                    "absolute_change_pct_points": after_error - before_error,
                    "percent_improvement": percent_improvement(before_error, after_error),
                    "overcorrected_direction": (
                        "baseline_below_scaled_above"
                        if baseline < kubios < scaled
                        else "baseline_above_scaled_below"
                        if baseline > kubios > scaled
                        else ""
                    ),
                }
            )
    return result_rows


def summarize_by_metric(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in [*METRICS, "OVERALL"]:
        metric_rows = results if metric == "OVERALL" else [row for row in results if row["metric"] == metric]
        before_values = [row["before_relative_error_pct"] for row in metric_rows]
        after_values = [row["after_relative_error_pct"] for row in metric_rows]
        before_mean = mean_finite(before_values)
        after_mean = mean_finite(after_values)
        before_median = median_finite(before_values)
        after_median = median_finite(after_values)
        rows.append(
            {
                "metric": metric,
                "n_rows": len(metric_rows),
                "before_mean_relative_error_pct": before_mean,
                "after_mean_relative_error_pct": after_mean,
                "mean_absolute_change_pct_points": after_mean - before_mean,
                "mean_percent_improvement": percent_improvement(before_mean, after_mean),
                "before_median_relative_error_pct": before_median,
                "after_median_relative_error_pct": after_median,
                "median_absolute_change_pct_points": after_median - before_median,
                "median_percent_improvement": percent_improvement(before_median, after_median),
                "before_std_relative_error_pct": stdev_finite(before_values),
                "after_std_relative_error_pct": stdev_finite(after_values),
                "overcorrection_rows": sum(1 for row in metric_rows if row["overcorrected_direction"]),
            }
        )
    return rows


def summarize_categories(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    categories = sorted({row["category"] for row in results})
    for category in categories:
        category_rows = [row for row in results if row["category"] == category]
        files = sorted({row["subset_id"] for row in category_rows})
        before_mean = mean_finite(row["before_relative_error_pct"] for row in category_rows)
        after_mean = mean_finite(row["after_relative_error_pct"] for row in category_rows)
        rows.append(
            {
                "category": category,
                "files": len(files),
                "before_mean": before_mean,
                "after_mean": after_mean,
                "mean_improvement_pct": percent_improvement(before_mean, after_mean),
                "before_median": median_finite(row["before_relative_error_pct"] for row in category_rows),
                "after_median": median_finite(row["after_relative_error_pct"] for row in category_rows),
            }
        )
    return rows


def remaining_outliers(results: Sequence[Dict[str, Any]], limit: int = 15) -> List[Dict[str, Any]]:
    rows = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["subset_id"]].append(row)
    for subset_id, file_rows in grouped.items():
        worst = sorted(file_rows, key=lambda row: finite_float(row["after_relative_error_pct"]), reverse=True)[:3]
        rows.append(
            {
                "rank": 0,
                "subset_id": subset_id,
                "category": file_rows[0]["category"],
                "after_mean_relative_error_pct": mean_finite(row["after_relative_error_pct"] for row in file_rows),
                "after_median_relative_error_pct": median_finite(row["after_relative_error_pct"] for row in file_rows),
                "before_mean_relative_error_pct": mean_finite(row["before_relative_error_pct"] for row in file_rows),
                "worst_after_metrics": "; ".join(
                    f"{row['metric']}={fmt(row['after_relative_error_pct'])}%"
                    for row in worst
                ),
                "source_file": file_rows[0]["source_file"],
            }
        )
    rows.sort(key=lambda row: finite_float(row["after_mean_relative_error_pct"]), reverse=True)
    for index, row in enumerate(rows[:limit], start=1):
        row["rank"] = index
    return rows[:limit]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def write_summary(
    path: Path,
    results: Sequence[Dict[str, Any]],
    metric_summary: Sequence[Dict[str, Any]],
    category_summary: Sequence[Dict[str, Any]],
    outliers: Sequence[Dict[str, Any]],
) -> None:
    files = sorted({row["subset_id"] for row in results})
    overall = next(row for row in metric_summary if row["metric"] == "OVERALL")
    power_rows = [row for row in metric_summary if row["metric"] in POWER_METRICS]
    ratio_rows = [row for row in metric_summary if row["metric"] in RATIO_METRICS]
    over_rows = [row for row in results if row["overcorrected_direction"]]
    metric_table = [
        [
            row["metric"],
            fmt(row["before_mean_relative_error_pct"]),
            fmt(row["after_mean_relative_error_pct"]),
            fmt(row["mean_percent_improvement"]),
            fmt(row["before_median_relative_error_pct"]),
            fmt(row["after_median_relative_error_pct"]),
            fmt(row["median_percent_improvement"]),
            row["overcorrection_rows"],
        ]
        for row in metric_summary
    ]
    category_table = [
        [
            row["category"],
            row["files"],
            fmt(row["before_mean"]),
            fmt(row["after_mean"]),
            fmt(row["mean_improvement_pct"]),
            fmt(row["before_median"]),
            fmt(row["after_median"]),
        ]
        for row in category_summary
    ]
    outlier_table = [
        [
            row["rank"],
            row["subset_id"],
            row["category"],
            fmt(row["after_mean_relative_error_pct"]),
            fmt(row["before_mean_relative_error_pct"]),
            row["worst_after_metrics"],
        ]
        for row in outliers[:10]
    ]

    lines = [
        "# Full Cleaned Hann Coherent-Gain PSD Diagnostic Audit",
        "",
        "This validation-only analysis applies the leading diagnostic variant from the focused PSD-convention audit, `hann_coherent_multiply`, across the full cleaned Kubios subset. It does not change production behavior and does not assert that Kubios is ground truth.",
        "",
        "## Counts",
        "",
        f"- Files analyzed: {len(files)}",
        f"- Metric rows analyzed: {len(results)}",
        f"- Overcorrection rows: {len(over_rows)}",
        "",
        "## Overall Result",
        "",
        f"- Overall mean relative error: {fmt(overall['before_mean_relative_error_pct'])}% before, {fmt(overall['after_mean_relative_error_pct'])}% after ({fmt(overall['mean_percent_improvement'])}% improvement).",
        f"- Overall median relative error: {fmt(overall['before_median_relative_error_pct'])}% before, {fmt(overall['after_median_relative_error_pct'])}% after ({fmt(overall['median_percent_improvement'])}% improvement).",
        "",
        "## By Metric",
        "",
        *markdown_table(
            [
                "Metric",
                "Before mean %",
                "After mean %",
                "Mean improvement %",
                "Before median %",
                "After median %",
                "Median improvement %",
                "Overcorrection rows",
            ],
            metric_table,
        ),
        "",
        "## By Category",
        "",
        *markdown_table(
            [
                "Category",
                "Files",
                "Before mean %",
                "After mean %",
                "Mean improvement %",
                "Before median %",
                "After median %",
            ],
            category_table,
        ),
        "",
        "## Remaining Outliers After Hann Scale",
        "",
        *markdown_table(
            [
                "Rank",
                "Subset",
                "Category",
                "After mean %",
                "Before mean %",
                "Worst after metrics",
            ],
            outlier_table,
        ),
        "",
        "## Interpretation",
        "",
        f"- Absolute power metrics improved broadly on mean error: {', '.join(f'{row['metric']} {fmt(row['mean_percent_improvement'])}%' for row in power_rows)}.",
        f"- Ratio/normalized metrics are mostly stable by construction or change less directly with a uniform PSD scale: {', '.join(f'{row['metric']} {fmt(row['mean_percent_improvement'])}%' for row in ratio_rows)}.",
        "- Category-level results should be interpreted cautiously because the category sizes are small and remaining edge cases can dominate means.",
        f"- The diagnostic scale still overcorrects some rows ({len(over_rows)} metric rows), so it is not cleanly safe as a production-wide change.",
        "- The current evidence supports treating `hann_coherent_multiply` as either an optional Kubios-compatible validation mode or a documentation finding, not as an immediate production default change.",
        "- A production change would require confirming the same scaling convention against a larger curated set and checking whether it worsens agreement with non-Kubios references or established PSD variance behavior.",
        "",
        "## Recommendation",
        "",
        "- Recommended classification: **b) optional Kubios-compatible mode** for validation/reporting experiments, with accompanying documentation. A documentation-only note is also defensible if the project wants to avoid any alternate computation mode.",
        "- Not recommended at this point: making this a production default.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cleaned_csv = project_path(args.cleaned_csv)
    output_dir = project_path(args.output_dir)
    ensure_no_overwrite(output_dir, args.force)

    cleaned_rows = read_csv(cleaned_csv)
    results = build_results(cleaned_rows)
    metric_summary = summarize_by_metric(results)
    category_summary = summarize_categories(results)
    outliers = remaining_outliers(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "full_cleaned_hann_scale_results.csv", results, list(results[0].keys()))
    write_csv(
        output_dir / "before_after_hann_scale_by_metric.csv",
        metric_summary,
        list(metric_summary[0].keys()),
    )
    write_csv(
        output_dir / "remaining_outliers_after_hann_scale.csv",
        outliers,
        list(outliers[0].keys()),
    )
    write_summary(
        output_dir / "full_cleaned_hann_scale_summary.md",
        results,
        metric_summary,
        category_summary,
        outliers,
    )

    overall = next(row for row in metric_summary if row["metric"] == "OVERALL")
    print(f"Files analyzed: {len({row['subset_id'] for row in results})}")
    print(
        "Overall mean relative error before/after: "
        f"{fmt(overall['before_mean_relative_error_pct'])}% / "
        f"{fmt(overall['after_mean_relative_error_pct'])}%"
    )
    print(
        "Overall median relative error before/after: "
        f"{fmt(overall['before_median_relative_error_pct'])}% / "
        f"{fmt(overall['after_median_relative_error_pct'])}%"
    )
    print(f"Wrote outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
