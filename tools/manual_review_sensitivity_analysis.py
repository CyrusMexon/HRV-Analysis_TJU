"""
Validation-only sensitivity analysis after manual-review exclusions.

This script reads the Kubios comparison CSV, removes manually flagged files from
the primary agreement set, and writes before/after summary artifacts. It does
not import or modify production HRV Studio analysis code.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_CSV = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "parsed_results_50_none_120s_75pct_after_arm_a"
    / "comparison_valid_only.csv"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "validation" / "kubios_subset" / "subset_manifest.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "validation" / "research_notes" / "manual_review_sensitivity_analysis"
)

METRICS = ("VLF", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu")
COMPARATORS = {
    "HRV Studio": "hrvstudio_relative_error_pct",
    "NeuroKit2": "neurokit_relative_error_pct",
}
MANUAL_EXCLUDE_IDS = ("OUT005", "VLF005", "VLF002", "RC003")
MANUAL_KEEP_IDS = ("OUT006",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Kubios agreement before/after manual-review exclusions."
    )
    parser.add_argument("--comparison-csv", default=str(DEFAULT_COMPARISON_CSV))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(MANUAL_EXCLUDE_IDS),
        help="Subset IDs or filename prefixes to exclude from cleaned analysis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting outputs in the target output directory.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_cell(row.get(column)) for column in columns})


def csv_cell(value: object) -> object:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    return "" if value is None else value


def finite_float(value: object) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def fmt(value: object, digits: int = 2) -> str:
    numeric = finite_float(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def identifiers_for_row(row: Dict[str, str]) -> Set[str]:
    values = {
        row.get("subset_id", ""),
        row.get("file_id", ""),
        row.get("source_file", ""),
        row.get("kubios_input_txt", ""),
        row.get("validation_input_file", ""),
    }
    identifiers: Set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        path = Path(text)
        candidates = {text, path.name, path.stem}
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            identifiers.add(candidate.upper())
            identifiers.add(candidate.split("__", 1)[0].upper())
    return identifiers


def row_matches_exclusion(row: Dict[str, str], excluded_ids: Sequence[str]) -> Optional[str]:
    identifiers = identifiers_for_row(row)
    for excluded_id in excluded_ids:
        target = excluded_id.upper()
        if any(identifier == target or identifier.startswith(f"{target}__") for identifier in identifiers):
            return excluded_id
    return None


def subset_ids(rows: Sequence[Dict[str, str]]) -> Set[str]:
    return {row.get("subset_id", "").strip() for row in rows if row.get("subset_id", "").strip()}


def values_for(rows: Sequence[Dict[str, str]], metric: Optional[str], column: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if metric is not None and row.get("metric") != metric:
            continue
        numeric = finite_float(row.get(column))
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "n": 0}
    return {
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def percent_improvement(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or before == 0:
        return math.nan
    return (before - after) / abs(before) * 100.0


def metric_summary_rows(
    before_rows: Sequence[Dict[str, str]],
    after_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for comparator, column in COMPARATORS.items():
        for metric in [*METRICS, "OVERALL"]:
            metric_filter = None if metric == "OVERALL" else metric
            before_stats = stats(values_for(before_rows, metric_filter, column))
            after_stats = stats(values_for(after_rows, metric_filter, column))
            row: Dict[str, object] = {
                "comparator": comparator,
                "metric": metric,
                "before_n_rows": before_stats["n"],
                "after_n_rows": after_stats["n"],
            }
            for key in ("mean", "median", "std"):
                before_value = before_stats[key]
                after_value = after_stats[key]
                row[f"before_{key}_relative_error_pct"] = before_value
                row[f"after_{key}_relative_error_pct"] = after_value
                row[f"{key}_absolute_change_pct_points"] = after_value - before_value
                row[f"{key}_percent_improvement"] = percent_improvement(before_value, after_value)
            summary_rows.append(row)
    return summary_rows


def manifest_by_subset(manifest_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {row.get("subset_id", "").strip(): row for row in manifest_rows if row.get("subset_id")}


def build_excluded_log(
    before_rows: Sequence[Dict[str, str]],
    after_rows: Sequence[Dict[str, str]],
    requested_exclusions: Sequence[str],
    manifest: Dict[str, Dict[str, str]],
) -> List[Dict[str, object]]:
    before_ids = subset_ids(before_rows)
    after_ids = subset_ids(after_rows)
    rows: List[Dict[str, object]] = []
    for subset_id in requested_exclusions:
        manifest_row = manifest.get(subset_id, {})
        rows.append(
            {
                "subset_id": subset_id,
                "category": manifest_row.get("category", ""),
                "source_file": manifest_row.get("source_csv", ""),
                "kubios_input_txt": manifest_row.get("kubios_input_txt", ""),
                "requested_manual_action": "exclude",
                "status_in_input_csv": "present" if subset_id in before_ids else "already_absent",
                "status_after_cleaning": (
                    "removed"
                    if subset_id in before_ids and subset_id not in after_ids
                    else "not_present"
                    if subset_id not in before_ids and subset_id not in after_ids
                    else "still_present"
                ),
                "notes": (
                    "Removed by manual-review sensitivity analysis."
                    if subset_id in before_ids
                    else "Not present in comparison_valid_only.csv before this step."
                ),
            }
        )
    return rows


def remaining_outlier_rows(
    cleaned_rows: Sequence[Dict[str, str]],
    manifest: Dict[str, Dict[str, str]],
    limit: int = 10,
) -> List[Dict[str, object]]:
    by_file: Dict[str, List[Dict[str, str]]] = {}
    for row in cleaned_rows:
        subset_id = row.get("subset_id", "").strip()
        if subset_id:
            by_file.setdefault(subset_id, []).append(row)

    ranked: List[Dict[str, object]] = []
    for subset_id, rows in by_file.items():
        hrv_values = values_for(rows, None, "hrvstudio_relative_error_pct")
        neurokit_values = values_for(rows, None, "neurokit_relative_error_pct")
        worst = sorted(
            rows,
            key=lambda row: finite_float(row.get("hrvstudio_relative_error_pct")),
            reverse=True,
        )[:3]
        manifest_row = manifest.get(subset_id, {})
        ranked.append(
            {
                "rank": 0,
                "subset_id": subset_id,
                "category": rows[0].get("category") or manifest_row.get("category", ""),
                "source_file": rows[0].get("source_file") or manifest_row.get("source_csv", ""),
                "hrvstudio_mean_relative_error_pct": mean(hrv_values) if hrv_values else math.nan,
                "hrvstudio_median_relative_error_pct": median(hrv_values) if hrv_values else math.nan,
                "neurokit_mean_relative_error_pct": mean(neurokit_values) if neurokit_values else math.nan,
                "worst_hrvstudio_metrics": "; ".join(
                    f"{row.get('metric')}={fmt(row.get('hrvstudio_relative_error_pct'))}%"
                    for row in worst
                ),
            }
        )

    ranked.sort(key=lambda row: finite_float(row["hrvstudio_mean_relative_error_pct"]), reverse=True)
    for index, row in enumerate(ranked[:limit], start=1):
        row["rank"] = index
    return ranked[:limit]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def primary_rows(rows: Sequence[Dict[str, object]], comparator: str) -> List[Dict[str, object]]:
    return [row for row in rows if row.get("comparator") == comparator]


def write_summary(
    path: Path,
    before_rows: Sequence[Dict[str, str]],
    after_rows: Sequence[Dict[str, str]],
    summary_rows: Sequence[Dict[str, object]],
    excluded_log: Sequence[Dict[str, object]],
    outliers: Sequence[Dict[str, object]],
    plot_path: Optional[Path],
) -> None:
    before_files = subset_ids(before_rows)
    after_files = subset_ids(after_rows)
    overall_hrv = next(
        row for row in summary_rows if row["comparator"] == "HRV Studio" and row["metric"] == "OVERALL"
    )
    overall_nk = next(
        row for row in summary_rows if row["comparator"] == "NeuroKit2" and row["metric"] == "OVERALL"
    )

    hrv_metric_table = [
        [
            row["metric"],
            fmt(row["before_mean_relative_error_pct"]),
            fmt(row["after_mean_relative_error_pct"]),
            fmt(row["mean_absolute_change_pct_points"]),
            fmt(row["mean_percent_improvement"]),
            fmt(row["before_median_relative_error_pct"]),
            fmt(row["after_median_relative_error_pct"]),
            fmt(row["median_percent_improvement"]),
        ]
        for row in primary_rows(summary_rows, "HRV Studio")
    ]
    nk_metric_table = [
        [
            row["metric"],
            fmt(row["before_mean_relative_error_pct"]),
            fmt(row["after_mean_relative_error_pct"]),
            fmt(row["mean_percent_improvement"]),
            fmt(row["before_median_relative_error_pct"]),
            fmt(row["after_median_relative_error_pct"]),
        ]
        for row in primary_rows(summary_rows, "NeuroKit2")
    ]
    excluded_table = [
        [
            row["subset_id"],
            row["category"],
            row["status_in_input_csv"],
            row["status_after_cleaning"],
            row["notes"],
        ]
        for row in excluded_log
    ]
    outlier_table = [
        [
            row["rank"],
            row["subset_id"],
            row["category"],
            fmt(row["hrvstudio_mean_relative_error_pct"]),
            fmt(row["hrvstudio_median_relative_error_pct"]),
            row["worst_hrvstudio_metrics"],
        ]
        for row in outliers
    ]

    lines: List[str] = [
        "# Manual-Review Sensitivity Analysis",
        "",
        "This validation-only analysis recomputes Kubios agreement after the manual-review exclusions `OUT005`, `VLF005`, `VLF002`, and `RC003`. `OUT006` is intentionally retained as a pathological edge case.",
        "",
        "## File Counts",
        "",
        f"- Files before cleaning: {len(before_files)}",
        f"- Requested exclusions: {', '.join(MANUAL_EXCLUDE_IDS)}",
        f"- Requested exclusions present in input CSV: {sum(1 for row in excluded_log if row['status_in_input_csv'] == 'present')}",
        f"- Files after cleaning: {len(after_files)}",
        f"- `OUT006` retained after cleaning: {'yes' if 'OUT006' in after_files else 'no'}",
        "",
        *markdown_table(
            ["Subset", "Category", "Input status", "After status", "Notes"],
            excluded_table,
        ),
        "",
        "## Overall Agreement",
        "",
        f"- HRV Studio overall mean relative error: {fmt(overall_hrv['before_mean_relative_error_pct'])}% before, {fmt(overall_hrv['after_mean_relative_error_pct'])}% after ({fmt(overall_hrv['mean_percent_improvement'])}% improvement).",
        f"- HRV Studio overall median relative error: {fmt(overall_hrv['before_median_relative_error_pct'])}% before, {fmt(overall_hrv['after_median_relative_error_pct'])}% after ({fmt(overall_hrv['median_percent_improvement'])}% improvement).",
        f"- NeuroKit2 overall mean relative error: {fmt(overall_nk['before_mean_relative_error_pct'])}% before, {fmt(overall_nk['after_mean_relative_error_pct'])}% after ({fmt(overall_nk['mean_percent_improvement'])}% improvement).",
        f"- NeuroKit2 overall median relative error: {fmt(overall_nk['before_median_relative_error_pct'])}% before, {fmt(overall_nk['after_median_relative_error_pct'])}% after ({fmt(overall_nk['median_percent_improvement'])}% improvement).",
        "",
        "## HRV Studio Before vs After",
        "",
        *markdown_table(
            [
                "Metric",
                "Before mean %",
                "After mean %",
                "Mean change",
                "Mean improvement %",
                "Before median %",
                "After median %",
                "Median improvement %",
            ],
            hrv_metric_table,
        ),
        "",
        "## NeuroKit2 Comparator Before vs After",
        "",
        *markdown_table(
            [
                "Metric",
                "Before mean %",
                "After mean %",
                "Mean improvement %",
                "Before median %",
                "After median %",
            ],
            nk_metric_table,
        ),
        "",
        "## Worst Remaining Outliers",
        "",
        *markdown_table(
            [
                "Rank",
                "Subset",
                "Category",
                "HRV mean %",
                "HRV median %",
                "Worst HRV metrics",
            ],
            outlier_table,
        ),
        "",
        "## Interpretation For Paper Drafting",
        "",
        "- The cleaned analysis improves the aggregate HRV Studio agreement metrics, which supports the cautious interpretation that part of the original disagreement was driven by pathological recordings or problematic input selections.",
        "- Agreement does not become uniformly close after cleaning. Remaining mean errors are still influenced by retained edge cases, including `OUT006`, and by broader method differences in VLF, LF, HF, and total power.",
        "- The median relative errors are more stable than the means and are the more defensible high-level summary for this manually reviewed subset.",
        "- The remaining disagreement appears mixed: some residual differences are isolated/pathological, while the absolute-power bands still show systematic sensitivity to preprocessing, interpolation, and Welch/bin conventions.",
        "- Additional manual review is still warranted before using the cleaned 50-file subset as a definitive agreement claim. The remaining-outlier shortlist identifies the next files to inspect.",
        "",
    ]
    if plot_path:
        lines.extend(["## Distribution Plot", "", f"- `{plot_path.relative_to(PROJECT_ROOT)}`", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def make_boxplot(
    output_path: Path,
    before_rows: Sequence[Dict[str, str]],
    after_rows: Sequence[Dict[str, str]],
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    before = values_for(before_rows, None, "hrvstudio_relative_error_pct")
    after = values_for(after_rows, None, "hrvstudio_relative_error_pct")
    if not before or not after:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([before, after], tick_labels=["Before", "After"], showfliers=False)
    ax.set_ylabel("HRV Studio relative error vs Kubios (%)")
    ax.set_title("Manual-review sensitivity analysis")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def ensure_outputs_do_not_exist(output_dir: Path, force: bool) -> None:
    targets = [
        output_dir / "manual_review_summary.md",
        output_dir / "before_vs_after_metrics.csv",
        output_dir / "cleaned_valid_only.csv",
        output_dir / "excluded_files_log.csv",
        output_dir / "remaining_outlier_shortlist.csv",
        output_dir / "relative_error_before_after_boxplot.png",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        joined = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing outputs. Re-run with --force if intended:\n{joined}")


def main() -> int:
    args = parse_args()
    comparison_csv = project_path(args.comparison_csv)
    manifest_path = project_path(args.manifest)
    output_dir = project_path(args.output_dir)
    ensure_outputs_do_not_exist(output_dir, args.force)

    rows = read_csv(comparison_csv)
    manifest = manifest_by_subset(read_csv(manifest_path))

    cleaned_rows = [
        row for row in rows if row_matches_exclusion(row, args.exclude) is None
    ]
    summary_rows = metric_summary_rows(rows, cleaned_rows)
    excluded_log = build_excluded_log(rows, cleaned_rows, args.exclude, manifest)
    outliers = remaining_outlier_rows(cleaned_rows, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cleaned_valid_only.csv", cleaned_rows, rows[0].keys())
    write_csv(
        output_dir / "before_vs_after_metrics.csv",
        summary_rows,
        [
            "comparator",
            "metric",
            "before_n_rows",
            "after_n_rows",
            "before_mean_relative_error_pct",
            "after_mean_relative_error_pct",
            "mean_absolute_change_pct_points",
            "mean_percent_improvement",
            "before_median_relative_error_pct",
            "after_median_relative_error_pct",
            "median_absolute_change_pct_points",
            "median_percent_improvement",
            "before_std_relative_error_pct",
            "after_std_relative_error_pct",
            "std_absolute_change_pct_points",
            "std_percent_improvement",
        ],
    )
    write_csv(
        output_dir / "excluded_files_log.csv",
        excluded_log,
        [
            "subset_id",
            "category",
            "source_file",
            "kubios_input_txt",
            "requested_manual_action",
            "status_in_input_csv",
            "status_after_cleaning",
            "notes",
        ],
    )
    write_csv(
        output_dir / "remaining_outlier_shortlist.csv",
        outliers,
        [
            "rank",
            "subset_id",
            "category",
            "source_file",
            "hrvstudio_mean_relative_error_pct",
            "hrvstudio_median_relative_error_pct",
            "neurokit_mean_relative_error_pct",
            "worst_hrvstudio_metrics",
        ],
    )
    plot_path = make_boxplot(output_dir / "relative_error_before_after_boxplot.png", rows, cleaned_rows)
    write_summary(
        output_dir / "manual_review_summary.md",
        rows,
        cleaned_rows,
        summary_rows,
        excluded_log,
        outliers,
        plot_path,
    )

    before_files = subset_ids(rows)
    after_files = subset_ids(cleaned_rows)
    excluded_present = [row["subset_id"] for row in excluded_log if row["status_in_input_csv"] == "present"]
    overall_hrv = next(
        row for row in summary_rows if row["comparator"] == "HRV Studio" and row["metric"] == "OVERALL"
    )

    print(f"Files before cleaning: {len(before_files)}")
    print(f"Requested exclusions present and removed: {', '.join(excluded_present) if excluded_present else 'none'}")
    print(f"Files after cleaning: {len(after_files)}")
    print(f"OUT006 retained: {'yes' if 'OUT006' in after_files else 'no'}")
    print(
        "HRV Studio overall mean/median relative error after cleaning: "
        f"{fmt(overall_hrv['after_mean_relative_error_pct'])}% / "
        f"{fmt(overall_hrv['after_median_relative_error_pct'])}%"
    )
    print(f"Wrote outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
