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
        ax.text(
            0.5,
            0.5,
            "No data to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
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
    """FIXED: Plot DFA scaling exponents using actual DFA curve data from pipeline."""
    ax.clear()

    dfa_data = None
    dfa_alpha1 = None
    dfa_alpha2 = None
    box_sizes = None
    fluctuations = None

    if nonlinear:
        # The pipeline stores DFA data in results["nonlinear"]["dfa"]
        if "dfa" in nonlinear and isinstance(nonlinear["dfa"], dict):
            dfa_data = nonlinear["dfa"]
            dfa_alpha1 = dfa_data.get("alpha1")
            dfa_alpha2 = dfa_data.get("alpha2")
            box_sizes = dfa_data.get("box_sizes")
            fluctuations = dfa_data.get("fluctuations")

    has_alpha1 = dfa_alpha1 is not None and np.isfinite(dfa_alpha1)
    has_alpha2 = dfa_alpha2 is not None and np.isfinite(dfa_alpha2)

    valid_boxes = None
    valid_flucts = None

    if (
        box_sizes is not None
        and fluctuations is not None
        and len(box_sizes) == len(fluctuations)
    ):
        valid_mask = (
            (box_sizes > 0)
            & (fluctuations > 0)
            & np.isfinite(box_sizes)
            & np.isfinite(fluctuations)
        )
        if np.any(valid_mask):
            valid_boxes = box_sizes[valid_mask]
            valid_flucts = fluctuations[valid_mask]

    has_curve = valid_boxes is not None

    # If no DFA data available or nothing valid to plot
    if dfa_data is None or (not has_alpha1 and not has_alpha2 and not has_curve):
        ax.set_title("DFA Analysis - No Data")
        ax.text(
            0.5,
            0.5,
            "DFA analysis not available or insufficient data\n(requires >= 40 RR intervals)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            style="italic",
        )
        ax.set_xlabel("Window size (beats)")
        ax.set_ylabel("Fluctuation function")
        return

    # Plot actual DFA curve if we have the raw data
    if has_curve:
        ax.loglog(
            valid_boxes,
            valid_flucts,
            "bo-",
            markersize=4,
            linewidth=1,
            label="DFA curve",
            alpha=0.7,
        )

        # Add trend lines for alpha1 and alpha2 if available
        if has_alpha1:
            # Short-term scaling (typically 4-11 beats)
            short_mask = (valid_boxes >= 4) & (valid_boxes <= 11)
            if np.any(short_mask):
                short_boxes = valid_boxes[short_mask]
                if len(short_boxes) > 0:
                    # Create trend line
                    trend_boxes = np.linspace(short_boxes.min(), short_boxes.max(), 20)
                    trend_flucts = trend_boxes**dfa_alpha1 * np.exp(
                        np.log(valid_flucts[short_mask][0])
                        - dfa_alpha1 * np.log(short_boxes[0])
                    )
                    ax.loglog(
                        trend_boxes,
                        trend_flucts,
                        "g-",
                        linewidth=2,
                        label=f"alpha1={dfa_alpha1:.3f}",
                    )

        if has_alpha2:
            # Long-term scaling (>11 beats)
            long_mask = valid_boxes > 11
            if np.any(long_mask):
                long_boxes = valid_boxes[long_mask]
                if len(long_boxes) > 0:
                    # Create trend line
                    trend_boxes = np.linspace(long_boxes.min(), long_boxes.max(), 20)
                    trend_flucts = trend_boxes**dfa_alpha2 * np.exp(
                        np.log(valid_flucts[long_mask][0])
                        - dfa_alpha2 * np.log(long_boxes[0])
                    )
                    ax.loglog(
                        trend_boxes,
                        trend_flucts,
                        "r-",
                        linewidth=2,
                        label=f"alpha2={dfa_alpha2:.3f}",
                    )
    else:
        # Fallback to synthetic curves if no raw data available
        scales = np.logspace(0.6, 1.8, 20)  # 4 to 64 beats
        if has_alpha1:
            f1 = scales**dfa_alpha1
            ax.loglog(scales, f1, "g-", lw=2, label=f"alpha1={dfa_alpha1:.3f}")
        if has_alpha2:
            f2 = scales**dfa_alpha2 * 2
            ax.loglog(scales, f2, "r-", lw=2, label=f"alpha2={dfa_alpha2:.3f}")

    ax.set_xlabel("Window size (beats)")
    ax.set_ylabel("Fluctuation function")
    ax.set_title("Detrended Fluctuation Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    interpretation = ""
    if dfa_alpha1 is not None and not np.isnan(dfa_alpha1):
        if dfa_alpha1 < 0.5:
            interpretation += "α₁<0.5: Anti-correlated\n"
        elif dfa_alpha1 > 1.5:
            interpretation += "α₁>1.5: Non-stationary\n"
        else:
            interpretation += "α₁≈1.0: Pink noise (healthy)\n"

    if interpretation:
        ax.text(
            0.98,
            0.02,
            interpretation.strip(),
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )


def plot_quality_assessment(ax, quality: dict, preprocessing_stats: dict):
    """FIXED: Visualize quality metrics with better data handling."""
    ax.clear()

    # Collect quality metrics from both sources
    metrics_data = {}

    # Get data from preprocessing_stats
    if preprocessing_stats:
        corrected_pct = (
            preprocessing_stats.get("corrected_beats_percentage", 0)
            or preprocessing_stats.get("artifact_percentage", 0)
            or preprocessing_stats.get("artifacts_corrected", 0)
        )
        if corrected_pct > 0:
            metrics_data["Corrected\nBeats (%)"] = corrected_pct

    # Get data from quality assessment
    if quality:
        artifact_pct = quality.get("artifact_percentage", 0)
        if artifact_pct > 0:
            metrics_data["Artifacts (%)"] = artifact_pct

        duration = quality.get("duration_s")
        if duration:
            metrics_data["Duration (min)"] = duration / 60.0

    # If no meaningful data, show placeholder
    if not metrics_data or all(v == 0 for v in metrics_data.values()):
        ax.set_title("Quality Assessment")
        ax.text(
            0.5,
            0.5,
            "Quality metrics\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            style="italic",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    # Create bar chart
    labels = list(metrics_data.keys())
    values = list(metrics_data.values())

    # Color code bars based on values
    colors = []
    for label, value in zip(labels, values):
        if "Corrected" in label or "Artifact" in label:
            if value > 5:
                colors.append("red")
            elif value > 2:
                colors.append("orange")
            else:
                colors.append("green")
        else:
            colors.append("blue")

    bars = ax.bar(labels, values, color=colors, alpha=0.7)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(values) * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Quality Assessment")
    ax.grid(True, alpha=0.3, axis="y")

    # Set appropriate y-axis limits
    if any("%" in label for label in labels):
        ax.set_ylim(0, max(100, max(values) * 1.2))
    else:
        ax.set_ylim(0, max(values) * 1.2)

    # Rotate x-labels if needed
    if len(labels) > 2:
        ax.tick_params(axis="x", rotation=45)


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

        # Try multiple key variations to match what the app uses
        band_mapping = [
            (["VLF_power", "vlf_power"], "VLF", "purple"),
            (["LF_power", "lf_power"], "LF", "green"),
            (["HF_power", "hf_power"], "HF", "red"),
        ]

        for key_variants, label, color in band_mapping:
            for key in key_variants:
                if key in results.frequency_domain:
                    value = results.frequency_domain[key]
                    if isinstance(value, (int, float)) and value > 0:
                        labels.append(label)
                        powers.append(value)
                        colors.append(color)
                        break  # Found the key, don't check other variants

        if powers:
            bars = ax4.bar(labels, powers, color=colors, alpha=0.7)
            ax4.set_ylabel("Power (msÂ²)")
            ax4.set_title("Band Powers")

            # Add value labels on bars (same as app)
            for bar, power in zip(bars, powers):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(powers) * 0.01,
                    f"{power:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            # Show message if no data
            ax4.text(
                0.5,
                0.5,
                "No frequency\nband data\navailable",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Band Powers")

    # Nonlinear analysis
    if results.nonlinear:
        ax5 = fig.add_subplot(gs[2, 0])
        plot_poincare_analysis(ax5, bundle, results.nonlinear)

        ax6 = fig.add_subplot(gs[2, 1])
        plot_dfa_analysis(ax6, results.nonlinear)

    # Quality assessment
    ax7 = fig.add_subplot(gs[2, 2])
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
    """FIXED: Plot pipeline results with better error handling and data validation."""

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
        try:
            freqs = results.frequency_domain["psd_frequencies"]
            power = results.frequency_domain["psd_power"]
            if freqs is not None and power is not None and len(freqs) == len(power):
                ax2.semilogy(freqs, power, "b-", linewidth=1.5)
                ax2.axvspan(0.04, 0.15, alpha=0.2, color="green", label="LF")
                ax2.axvspan(0.15, 0.4, alpha=0.2, color="red", label="HF")
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Power (ms²/Hz)")
                ax2.set_title("PSD")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.set_title("PSD - Data Error")
                ax2.text(
                    0.5,
                    0.5,
                    "Invalid PSD data",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
        except Exception as e:
            ax2.set_title("PSD - Error")
            ax2.text(
                0.5,
                0.5,
                f"Error plotting PSD:\n{str(e)[:50]}",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
    else:
        ax2.set_title("PSD - No Data")
        ax2.text(
            0.5,
            0.5,
            "Frequency domain\nanalysis not available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    # 3. Poincaré
    ax3 = fig.add_subplot(gs[1, 0])
    plot_poincare_analysis(ax3, bundle, results.nonlinear)

    # 4. Quality assessment (FIXED)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_quality_assessment(
        ax4, results.quality_assessment, results.preprocessing_stats
    )

    fig.tight_layout()
    return fig
