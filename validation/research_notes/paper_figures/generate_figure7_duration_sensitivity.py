"""Generate Figure 7: duration sensitivity.

This script uses stored v10 duration-sensitivity validation outputs only. It
does not rerun validation experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = PROJECT_ROOT / "validation" / "runs" / "v10_duration_sensitivity_validation"
OUT_DIR = PROJECT_ROOT / "validation" / "research_notes" / "paper_figures"

DURATIONS = ["30s", "60s", "2min", "5min", "10min"]
DURATION_LABELS = ["30 s", "60 s", "2 min", "5 min", "10 min"]
X = list(range(len(DURATIONS)))

METRICS = ["LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu", "VLF"]
METRIC_COLORS = {
    "LF": "#4c78a8",
    "HF": "#59a14f",
    "total_power": "#9c755f",
    "LF/HF": "#f28e2b",
    "LF_nu": "#76b7b2",
    "HF_nu": "#b07aa1",
    "VLF": "#e15759",
}
METRIC_MARKERS = {
    "LF": "o",
    "HF": "s",
    "total_power": "D",
    "LF/HF": "^",
    "LF_nu": "v",
    "HF_nu": "P",
    "VLF": "^",
}


def duration_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["duration_label"] = pd.Categorical(
        ordered["duration_label"], categories=DURATIONS, ordered=True
    )
    return ordered.sort_values("duration_label")


def load_panel_a() -> pd.DataFrame:
    frame = pd.read_csv(RUN_DIR / "duration_metric_table.csv")
    frame = frame[frame["metric"].isin(METRICS)]
    return duration_order(frame)


def load_panel_b() -> pd.DataFrame:
    frame = pd.read_csv(RUN_DIR / "duration_results.csv")
    frame = frame[frame["metric"].eq("VLF")]
    summary = (
        frame.groupby("duration_label", observed=True)["relative_error_vs_neurokit2_pct"]
        .agg(median="median", p90=lambda s: s.quantile(0.90), n="count")
        .reset_index()
    )
    return duration_order(summary)


def load_panel_c() -> pd.DataFrame:
    frame = pd.read_csv(RUN_DIR / "duration_results.csv")
    row_summary = (
        frame.groupby("duration_label", observed=True)
        .agg(
            files=("file_name", "nunique"),
            metric_rows=("metric", "count"),
            finite_output_rate_pct=("finite_hrvstudio", lambda s: s.mean() * 100),
            warning_visible_rate_pct=("warning_count", lambda s: (s > 0).mean() * 100),
            duration_warning_rate_pct=("duration_warning", lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    return duration_order(row_summary)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#c7c7c7", linestyle=(0, (1, 1.65)), linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_axisbelow(True)


def add_short_duration_region(ax: plt.Axes, text: str) -> None:
    ax.axvspan(-0.45, 2.5, color="#eeeeee", zorder=0)
    ax.axvline(3, color="#555555", linestyle=(0, (3.7, 1.6)), linewidth=1.0)
    ax.text(3.02, 0.86, "5 min", transform=ax.get_xaxis_transform(), fontsize=8, color="#444444")
    ax.text(
        0.0,
        -0.25,
        text,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        color="#222222",
        clip_on=False,
    )


def build_figure() -> plt.Figure:
    panel_a = load_panel_a()
    panel_b = load_panel_b()
    panel_c = load_panel_c()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.72, 3.15), dpi=300)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.30, wspace=0.36)

    ax = axes[0]
    add_short_duration_region(ax, "<5 min: caution for spectral metrics")
    for metric in METRICS:
        rows = panel_a[panel_a["metric"].eq(metric)].set_index("duration_label").loc[DURATIONS]
        y = rows["median_relative_error_vs_neurokit2_pct"].to_numpy()
        ax.plot(
            X,
            y,
            marker=METRIC_MARKERS[metric],
            color=METRIC_COLORS[metric],
            linewidth=1.25,
            markersize=4.2,
            label=metric,
        )
    ax.set_title("A. Metric agreement across durations", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("Median relative error (%)", fontsize=9)
    ax.set_xticks(X, DURATION_LABELS)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.65, 4.25)
    style_axis(ax)
    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=8,
        frameon=False,
        handlelength=1.4,
        borderaxespad=0.4,
    )

    ax = axes[1]
    add_short_duration_region(ax, "VLF remains convention-sensitive")
    ax.plot(
        X,
        panel_b.set_index("duration_label").loc[DURATIONS, "median"],
        marker="o",
        color="#e15759",
        linewidth=1.5,
        markersize=4.5,
        label="Median",
    )
    ax.plot(
        X,
        panel_b.set_index("duration_label").loc[DURATIONS, "p90"],
        marker="s",
        color="#7f7f7f",
        linewidth=1.3,
        linestyle=(0, (3.7, 1.6)),
        markersize=4.0,
        label="P90",
    )
    for xi, value in zip(X[1:], panel_b.set_index("duration_label").loc[DURATIONS[1:], "median"]):
        ax.text(xi, value + 3.1, f"{value:.1f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_title("B. VLF sensitivity to recording duration", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("VLF relative error (%)", fontsize=9)
    ax.set_xticks(X, DURATION_LABELS)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.65, 4.25)
    style_axis(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=1.8, borderaxespad=0.4)

    ax = axes[2]
    width = 0.34
    finite = panel_c.set_index("duration_label").loc[DURATIONS, "finite_output_rate_pct"]
    warning = panel_c.set_index("duration_label").loc[DURATIONS, "warning_visible_rate_pct"]
    duration_warning = panel_c.set_index("duration_label").loc[DURATIONS, "duration_warning_rate_pct"]
    bars_finite = ax.bar(
        [x - width / 2 for x in X],
        finite,
        width=width,
        color="#d0d0d0",
        edgecolor="#222222",
        linewidth=0.6,
        hatch="////",
        label="Finite output",
    )
    ax.bar(
        [x + width / 2 for x in X],
        warning,
        width=width,
        color="#4c78a8",
        edgecolor="#222222",
        linewidth=0.6,
        label="Warning visible",
    )
    ax.plot(
        X,
        duration_warning,
        color="#222222",
        marker="D",
        markersize=4.8,
        linewidth=1.0,
        linestyle=(0, (3.7, 1.6)),
        label="Duration warning",
    )
    for bar, files in zip(bars_finite, panel_c["files"]):
        ax.text(
            bar.get_x() + width,
            104,
            f"{int(files)} files",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#555555",
        )
    ax.text(
        0.04,
        0.62,
        "Finite output does not\nimply interpretability",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d0d0d0", alpha=0.92),
    )
    ax.set_title("C. Numerical stability and warning behavior", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("Rows / recordings (%)", fontsize=9)
    ax.set_xticks(X, DURATION_LABELS)
    ax.set_ylim(0, 118)
    ax.set_xlim(-0.65, 4.25)
    style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=8,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    base = OUT_DIR / "figure7_duration_sensitivity"
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
