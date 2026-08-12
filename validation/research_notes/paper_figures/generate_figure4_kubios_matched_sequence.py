"""Generate matched-sequence Figure 4 for the Kubios benchmark.

This script uses stored validation artifacts only. It does not rerun validation
experiments and does not overwrite the original Figure 4 files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "validation" / "research_notes" / "paper_figures"
REPORT_PATH = PROJECT_ROOT / "validation" / "research_notes" / "figure4_sequence_harmonization_update.md"

ORIGINAL_FIGURE = OUT_DIR / "figure4_kubios_benchmark.png"
OUTPUT_PNG = OUT_DIR / "figure4_kubios_matched_sequence.png"
OUTPUT_PDF = OUT_DIR / "figure4_kubios_matched_sequence.pdf"
OUTPUT_SVG = OUT_DIR / "figure4_kubios_matched_sequence.svg"

ORIGINAL_SOURCE_NOTES = OUT_DIR / "figure4_kubios_benchmark_source_notes.md"
ORIGINAL_SUMMARY = (
    PROJECT_ROOT
    / "validation"
    / "research_notes"
    / "final_validation_results_package"
    / "final_kubios_metric_table.csv"
)
ORIGINAL_ROWS = (
    PROJECT_ROOT
    / "validation"
    / "research_notes"
    / "manual_review_sensitivity_analysis"
    / "cleaned_valid_only.csv"
)
MATCHED_SUMMARY = (
    PROJECT_ROOT / "validation" / "kubios_subset" / "frequency_domain_kubios_matched_sequence_summary.csv"
)
MATCHED_ROWS = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "frequency_domain_kubios_matched_sequence_per_recording.csv"
)
ORIGINAL_VS_MATCHED = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "frequency_domain_original_vs_matched_sequence_summary.csv"
)
PARSED_EXPORTS = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "parsed_results_50_none_120s_75pct_after_arm_a"
    / "kubios_parsed_results.csv"
)
COMPARISON_ALL = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "parsed_results_50_none_120s_75pct_after_arm_a"
    / "comparison_all.csv"
)
COMPARISON_VALID = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "parsed_results_50_none_120s_75pct_after_arm_a"
    / "comparison_valid_only.csv"
)
COMPARISON_EXCLUDED = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "parsed_results_50_none_120s_75pct_after_arm_a"
    / "comparison_excluded_files.csv"
)

METRIC_ORDER = ["LF_nu", "HF_nu", "LF/HF", "total_power", "LF", "HF", "VLF"]
METRIC_LABELS = {
    "LF_nu": "LFnu",
    "HF_nu": "HFnu",
    "LF/HF": "LF/HF",
    "total_power": "Total power",
    "LF": "LF",
    "HF": "HF",
    "VLF": "VLF",
}


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#bdbdbd", linestyle=(0, (1, 1.65)), linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched_summary = pd.read_csv(MATCHED_SUMMARY)
    matched_rows = pd.read_csv(MATCHED_ROWS)
    original_vs_matched = pd.read_csv(ORIGINAL_VS_MATCHED)
    return matched_summary, matched_rows, original_vs_matched


def plot_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A. Kubios subset filtering")

    steps = [
        ("Kubios exports\nparsed", "n = 50", "#f0f0f0"),
        ("Matched to\nvalidation rows", "n = 48", "#e8eef6"),
        ("Retained after\nautomatic QC", "n = 46", "#dce8f4"),
        ("Retained after\nmanual review", "n = 44", "#c8d9ee"),
    ]
    x = 0.50
    y_positions = [0.82, 0.61, 0.40, 0.19]
    box_w = 0.45
    box_h = 0.18
    for index, ((label, count, color), y) in enumerate(zip(steps, y_positions)):
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.9,
        )
        ax.add_patch(box)
        line_1, line_2 = label.split("\n")
        ax.text(x, y + 0.055, line_1, ha="center", va="center", fontsize=7.4)
        ax.text(x, y + 0.020, line_2, ha="center", va="center", fontsize=7.4)
        ax.text(x, y - 0.055, count, ha="center", va="center", fontsize=9.2, fontweight="bold")
        if index < len(y_positions) - 1:
            ax.annotate(
                "",
                xy=(x, y_positions[index + 1] + box_h / 2 + 0.008),
                xytext=(x, y - box_h / 2 - 0.008),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9),
            )
    ax.text(
        x,
        0.035,
        "Frequency-domain analysis set: 44 files / 308 metric rows",
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
    )


def plot_panel_b(ax: plt.Axes, matched_summary: pd.DataFrame) -> None:
    panel_title(ax, "B. Median relative error by metric")
    values = (
        matched_summary.set_index("Metric").loc[METRIC_ORDER, "median_relative_error_pct"].to_numpy()
    )
    labels = [METRIC_LABELS[metric] for metric in METRIC_ORDER]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color="#4c78a8", edgecolor="#222222", linewidth=0.5)
    ax.set_ylabel("Median relative error (%)", fontsize=8.5)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 16)
    style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_panel_c(ax: plt.Axes, matched_rows: pd.DataFrame, matched_summary: pd.DataFrame) -> None:
    panel_title(ax, "C. LF/HF agreement")
    rows = matched_rows[matched_rows["Metric"].eq("LF/HF")].copy()
    rows = rows[np.isfinite(rows["kubios_value"]) & np.isfinite(rows["selected_hrvstudio_value"])]
    rows = rows[(rows["kubios_value"] > 0) & (rows["selected_hrvstudio_value"] > 0)]
    x = rows["kubios_value"].to_numpy()
    y = rows["selected_hrvstudio_value"].to_numpy()
    ax.scatter(
        x,
        y,
        s=20,
        color="#4c78a8",
        edgecolor="#222222",
        linewidth=0.35,
        alpha=0.88,
    )
    lower = min(x.min(), y.min()) * 0.72
    upper = max(x.max(), y.max()) * 1.28
    ax.plot([lower, upper], [lower, upper], color="#222222", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Kubios LF/HF", fontsize=8.5)
    ax.set_ylabel("HRV Studio LF/HF", fontsize=8.5)
    ax.grid(which="major", color="#bdbdbd", linestyle=(0, (1, 1.65)), linewidth=0.5)
    ax.grid(which="minor", visible=False)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    summary = matched_summary.set_index("Metric").loc["LF/HF"]
    ax.text(
        0.05,
        0.94,
        f"Pearson r = {summary['pearson_correlation']:.3f}\nMedian error = {summary['median_relative_error_pct']:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#999999", linewidth=0.5),
    )


def plot_panel_d(ax: plt.Axes, original_vs_matched: pd.DataFrame) -> None:
    panel_title(ax, "D. Original vs matched median error")
    table = original_vs_matched.set_index("Metric").loc[METRIC_ORDER]
    original = table["Original median RE %"].to_numpy()
    matched = table["Matched-sequence median RE %"].to_numpy()
    labels = [METRIC_LABELS[metric] for metric in METRIC_ORDER]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(
        x - width / 2,
        original,
        width=width,
        color="#d0d0d0",
        edgecolor="#222222",
        linewidth=0.5,
        label="Before sequence harmonization",
    )
    ax.bar(
        x + width / 2,
        matched,
        width=width,
        color="#4c78a8",
        edgecolor="#222222",
        linewidth=0.5,
        label="After sequence harmonization",
    )
    ax.set_ylabel("Median relative error (%)", fontsize=8.5)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 36)
    style_axis(ax)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False, handlelength=1.4)


def build_figure() -> plt.Figure:
    matched_summary, matched_rows, original_vs_matched = load_inputs()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.31666, 6.31666), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.105, wspace=0.34, hspace=0.43)

    plot_panel_a(axes[0, 0])
    plot_panel_b(axes[0, 1], matched_summary)
    plot_panel_c(axes[1, 0], matched_rows, matched_summary)
    plot_panel_d(axes[1, 1], original_vs_matched)
    return fig


def write_report() -> None:
    matched_summary, matched_rows, original_vs_matched = load_inputs()
    lfhf = matched_summary.set_index("Metric").loc["LF/HF"]
    report = f"""# Figure 4 Sequence Harmonization Update

## Original Figure 4 Source

- Existing rendered figure: `{rel(ORIGINAL_FIGURE)}`
- Existing source notes: `{rel(ORIGINAL_SOURCE_NOTES)}`
- Original generator script: not found in the repository; the source notes indicate Figure 4 was regenerated from stored validation CSV artifacts with Matplotlib.
- Plotting library used for this update: Matplotlib.

## Original Input Files

- `{rel(ORIGINAL_SUMMARY)}`
- `{rel(ORIGINAL_ROWS)}`
- `{rel(PARSED_EXPORTS)}`
- `{rel(COMPARISON_ALL)}`
- `{rel(COMPARISON_VALID)}`
- `{rel(COMPARISON_EXCLUDED)}`

## New Input Files

- `{rel(MATCHED_SUMMARY)}`
- `{rel(MATCHED_ROWS)}`
- `{rel(ORIGINAL_VS_MATCHED)}`

## Output Files

- `{rel(OUTPUT_PNG)}`
- `{rel(OUTPUT_PDF)}`
- `{rel(OUTPUT_SVG)}`

## Exact Changes

- Panel A was kept unchanged: n=50 parsed exports, n=48 matched rows, n=46 automatic QC retained, n=44 manual-review retained.
- Panel B now uses matched-sequence median relative errors from `{rel(MATCHED_SUMMARY)}`.
- Panel C now uses `kubios_value` versus `selected_hrvstudio_value` for LF/HF from `{rel(MATCHED_ROWS)}`. The annotation is Pearson r = {lfhf['pearson_correlation']:.3f} and median error = {lfhf['median_relative_error_pct']:.2f}%.
- Panel D was changed from mean-versus-median error to paired original-versus-sequence-matched median relative error bars using `{rel(ORIGINAL_VS_MATCHED)}`.

## Visual Style Preservation

The update preserves the original 2x2 A-D layout, DejaVu Sans typography, panel-label format, manuscript-scale dimensions, bar/scatter styling, grid treatment, and compact source-data-only design. The substantive changes are limited to the data sources and the requested Panel D comparison.

## Row Counts

- Matched summary rows: {len(matched_summary)}
- Matched per-recording rows: {len(matched_rows)}
- LF/HF rows plotted in Panel C before finite/positive filtering: {len(matched_rows[matched_rows['Metric'].eq('LF/HF')])}
- Original-versus-matched summary rows: {len(original_vs_matched)}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    fig = build_figure()
    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_SVG)
    plt.close(fig)
    write_report()
    print(f"Wrote {rel(OUTPUT_PNG)}")
    print(f"Wrote {rel(OUTPUT_PDF)}")
    print(f"Wrote {rel(OUTPUT_SVG)}")
    print(f"Wrote {rel(REPORT_PATH)}")


if __name__ == "__main__":
    main()
