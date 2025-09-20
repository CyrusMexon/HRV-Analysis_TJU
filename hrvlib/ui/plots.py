"""
plots.py - Centralized plotting logic for HRV GUI and reports
Keeps visualization logic separate from GUI widgets (clean architecture)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from hrvlib.data_handler import DataBundle
from hrvlib.pipeline import HRVAnalysisResults


def plot_rr_intervals_enhanced(
    ax, bundle: DataBundle, time_domain: dict, preprocessing_stats: dict
):
    """Enhanced RR interval plot with preprocessing annotations."""
    if not bundle or not bundle.rri_ms:
        ax.set_title("No RR intervals available")
        return

    rri = np.array(bundle.rri_ms)
    t = np.cumsum(rri) / 1000.0
    ax.plot(t, rri, "b-", lw=1.2, label="RR intervals")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RR Interval (ms)")
    ax.set_title("RR Intervals Over Time")
    ax.grid(True, alpha=0.3)

    # Add preprocessing info if available
    if preprocessing_stats and "corrected_beats" in preprocessing_stats:
        ax.text(
            0.02,
            0.95,
            f"Corrected beats: {preprocessing_stats['corrected_beats']} ({preprocessing_stats.get('corrected_beats_percentage',0):.1f}%)",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

    ax.legend(loc="upper right")


def plot_poincare_analysis(ax, bundle: DataBundle, nonlinear: dict):
    """Poincaré plot with SD1/SD2 ellipse."""
    if not bundle or not bundle.rri_ms:
        ax.set_title("No data for Poincaré")
        return

    rri = np.array(bundle.rri_ms)
    x = rri[:-1]
    y = rri[1:]
    ax.scatter(x, y, alpha=0.6, s=10, c="blue", label="RR scatter")

    if nonlinear and "SD1" in nonlinear and "SD2" in nonlinear:
        sd1 = nonlinear["SD1"]
        sd2 = nonlinear["SD2"]
        mean_rr = np.mean(rri)

        # Draw ellipse approximation (45° rotated)
        ax.plot(
            [mean_rr - sd2, mean_rr + sd2],
            [mean_rr - sd1, mean_rr + sd1],
            "r--",
            lw=2,
            label=f"SD1={sd1:.1f}, SD2={sd2:.1f}",
        )

    ax.set_xlabel("RR_n (ms)")
    ax.set_ylabel("RR_n+1 (ms)")
    ax.set_title("Poincaré Plot")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_dfa_analysis(ax, nonlinear: dict):
    """Plot DFA scaling exponents."""
    if not nonlinear or "DFA_alpha1" not in nonlinear:
        ax.set_title("No DFA results")
        return

    alpha1 = nonlinear.get("DFA_alpha1")
    alpha2 = nonlinear.get("DFA_alpha2")

    scales = np.arange(1, 10)
    f1 = alpha1 * np.log(scales)
    f2 = alpha2 * np.log(scales)

    ax.plot(scales, f1, "g-", lw=1.5, label=f"α1={alpha1:.2f}")
    ax.plot(scales, f2, "r-", lw=1.5, label=f"α2={alpha2:.2f}")

    ax.set_xscale("log")
    ax.set_xlabel("Window size (log)")
    ax.set_ylabel("Fluctuation function (log)")
    ax.set_title("DFA Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_quality_assessment(ax, quality: dict, preprocessing_stats: dict):
    """Visualize quality metrics as bar chart / indicators."""
    if not quality:
        ax.set_title("No quality assessment available")
        return

    metrics = ["corrected_beats_percentage", "artifact_density"]
    values = [quality.get(m, 0) for m in metrics]

    ax.bar(metrics, values, color=["orange", "red"], alpha=0.7)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("Quality Assessment")
    ax.grid(True, alpha=0.3)


def create_summary_figure(
    bundle: DataBundle, results: HRVAnalysisResults
) -> plt.Figure:
    """Create a comprehensive summary figure for reports."""

    fig = plt.figure(figsize=(11, 8.5))  # Letter size
    gs = GridSpec(
        4, 3, figure=fig, height_ratios=[1.5, 1, 1, 0.8], width_ratios=[2, 1, 1]
    )

    # Top row: RR intervals
    ax1 = fig.add_subplot(gs[0, :2])
    plot_rr_intervals_enhanced(
        ax1, bundle, results.time_domain, results.preprocessing_stats
    )

    # Top right: HR histogram
    ax2 = fig.add_subplot(gs[0, 2])
    if bundle.rri_ms:
        hr = 60000.0 / np.array(bundle.rri_ms)
        ax2.hist(hr, bins=30, alpha=0.7, color="red", edgecolor="black")
        ax2.set_xlabel("Heart Rate (bpm)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("HR Distribution")

    # Frequency domain PSD
    if results.frequency_domain and "psd_frequencies" in results.frequency_domain:
        ax3 = fig.add_subplot(gs[1, :2])
        freqs = results.frequency_domain["psd_frequencies"]
        power = results.frequency_domain["psd_power"]
        ax3.semilogy(freqs, power, "b-", linewidth=1.5)
        ax3.axvspan(0.04, 0.15, alpha=0.2, color="green", label="LF")
        ax3.axvspan(0.15, 0.4, alpha=0.2, color="red", label="HF")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power (ms²/Hz)")
        ax3.set_title("Power Spectral Density")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Band powers
    ax4 = fig.add_subplot(gs[1, 2])
    if results.frequency_domain:
        labels, powers, colors = [], [], []
        for band, color in [
            ("LF_power", "green"),
            ("HF_power", "red"),
            ("VLF_power", "purple"),
        ]:
            if band in results.frequency_domain:
                labels.append(band.replace("_power", ""))
                powers.append(results.frequency_domain[band])
                colors.append(color)
        if powers:
            ax4.bar(labels, powers, color=colors, alpha=0.7)
            ax4.set_ylabel("Power (ms²)")
            ax4.set_title("Band Powers")

    # Nonlinear analysis
    if results.nonlinear:
        ax5 = fig.add_subplot(gs[2, 0])
        plot_poincare_analysis(ax5, bundle, results.nonlinear)

        ax6 = fig.add_subplot(gs[2, 1])
        plot_dfa_analysis(ax6, results.nonlinear)

    # Quality assessment
    ax7 = fig.add_subplot(gs[2, 2])
    if results.quality_assessment:
        plot_quality_assessment(
            ax7, results.quality_assessment, results.preprocessing_stats
        )

    # Bottom row: Metrics summary table
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis("off")

    table_data = []

    if results.time_domain:
        table_data.append(["Time Domain", "", ""])
        for metric, unit in [("SDNN", "ms"), ("RMSSD", "ms"), ("pNN50", "%")]:
            if metric in results.time_domain:
                value = results.time_domain[metric]
                table_data.append([f"{metric}: {value:.1f} {unit}", "", ""])

    if results.frequency_domain:
        table_data.append(["Frequency Domain", "", ""])
        for metric, unit in [
            ("LF_power", "ms²"),
            ("HF_power", "ms²"),
            ("LF_HF_ratio", ""),
        ]:
            if metric in results.frequency_domain:
                value = results.frequency_domain[metric]
                unit_str = f" {unit}" if unit else ""
                table_data.append([f"{metric}: {value:.2f}{unit_str}", "", ""])

    if table_data:
        table = ax8.table(
            cellText=table_data,
            cellLoc="left",
            loc="center",
            colWidths=[0.33, 0.33, 0.34],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

    plt.suptitle("HRV Analysis Summary", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def plot_pipeline_results(
    fig: plt.Figure, bundle: DataBundle, results: HRVAnalysisResults
):
    """Plot pipeline results into the provided Matplotlib figure for the GUI signal viewer."""

    fig.clear()
    gs = GridSpec(2, 2, figure=fig)

    # 1. RR intervals
    ax1 = fig.add_subplot(gs[0, 0])
    plot_rr_intervals_enhanced(
        ax1, bundle, results.time_domain, results.preprocessing_stats
    )

    # 2. Frequency domain (PSD)
    ax2 = fig.add_subplot(gs[0, 1])
    if results.frequency_domain and "psd_frequencies" in results.frequency_domain:
        freqs = results.frequency_domain["psd_frequencies"]
        power = results.frequency_domain["psd_power"]
        ax2.semilogy(freqs, power, "b-", linewidth=1.5)
        ax2.axvspan(0.04, 0.15, alpha=0.2, color="green", label="LF")
        ax2.axvspan(0.15, 0.4, alpha=0.2, color="red", label="HF")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Power (ms²/Hz)")
        ax2.set_title("PSD")
        ax2.legend()

    # 3. Poincaré
    ax3 = fig.add_subplot(gs[1, 0])
    plot_poincare_analysis(ax3, bundle, results.nonlinear)

    # 4. Quality assessment
    ax4 = fig.add_subplot(gs[1, 1])
    plot_quality_assessment(
        ax4, results.quality_assessment, results.preprocessing_stats
    )

    fig.tight_layout()
    return fig
