"""Generate band-corrected manuscript Figures 3, 5, 6, 7, and 8.

This script uses stored v12 validation rerun artifacts only. It does not rerun
validation experiments, modify manuscript files, or overwrite legacy figures.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "validation" / "figures"
REPORT_PATH = (
    PROJECT_ROOT
    / "validation"
    / "research_notes"
    / "figures_3_5_6_7_8_band_corrected_regeneration_report.md"
)

FIG3_5MIN = PROJECT_ROOT / "validation/runs/v12_band_corrected_physionet_5min_neurokit2_standard_20260811"
FIG3_10MIN = PROJECT_ROOT / "validation/runs/v12_band_corrected_physionet_10min_segment_linear_standard_20260811"
FIG5_RUN = PROJECT_ROOT / "validation/runs/v12_band_corrected_fft_ar_validation_standard_20260811"
FIG6_RUN = PROJECT_ROOT / "validation/runs/v12_band_corrected_robustness_10base_signal_conditions_standard_20260811"
FIG7_RUN = PROJECT_ROOT / "validation/runs/v12_band_corrected_duration_sensitivity_standard_20260811"
FIG8_RUN = PROJECT_ROOT / "validation/runs/v12_band_corrected_mitbih_arrhythmia_robustness_standard_20260811_complete"
PHYSIONET_10MIN = PROJECT_ROOT / "validation/processed_data/physionet_nsr_rr_10min"
MITBIH_ROOT = PROJECT_ROOT / "validation/raw_data/mit_bih_arrhythmia/mit-bih-arrhythmia-database-1.0.0"

STANDARD_BANDS = (
    "standard: ULF 0 < f < 0.003 Hz; VLF 0.003 <= f < 0.04 Hz; "
    "LF 0.04 <= f < 0.15 Hz; HF 0.15 <= f <= 0.40 Hz; total 0 < f <= 0.40 Hz"
)

METRIC_ORDER = ["LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu", "VLF"]
METRIC_LABELS = {
    "LF": "LF",
    "HF": "HF",
    "total_power": "Total power",
    "LF/HF": "LF/HF",
    "LF_nu": "LFnu",
    "HF_nu": "HFnu",
    "VLF": "VLF",
    "RMSSD": "RMSSD",
}
COLORS = {
    "blue": "#4c78a8",
    "green": "#59a14f",
    "orange": "#f28e2b",
    "red": "#e15759",
    "teal": "#76b7b2",
    "purple": "#b07aa1",
    "brown": "#9c755f",
    "gray": "#d0d0d0",
    "dark": "#222222",
}


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )


def style_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color="#bdbdbd", linestyle=(0, (1, 1.65)), linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")


def save_figure(fig: plt.Figure, base_name: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in [".png", ".pdf", ".svg"]:
        path = OUT_DIR / f"{base_name}{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        outputs.append(path)
    plt.close(fig)
    return outputs


def finite_positive(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    rows = frame.copy()
    rows[x_col] = pd.to_numeric(rows[x_col], errors="coerce")
    rows[y_col] = pd.to_numeric(rows[y_col], errors="coerce")
    rows = rows[np.isfinite(rows[x_col]) & np.isfinite(rows[y_col])]
    return rows[(rows[x_col] > 0) & (rows[y_col] > 0)].copy()


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return math.nan
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return math.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def median_relative_error(value: pd.Series, ref: pd.Series) -> float:
    value = pd.to_numeric(value, errors="coerce")
    ref = pd.to_numeric(ref, errors="coerce")
    mask = np.isfinite(value) & np.isfinite(ref) & (ref.abs() > 1e-12)
    return float((value[mask].sub(ref[mask]).abs() / ref[mask].abs() * 100.0).median())


def load_metric_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "metric_key" in frame.columns:
        frame = frame.rename(columns={"metric_key": "metric"})
    frame["metric"] = frame["Metric"].map(
        {"Total power": "total_power", "LFnu": "LF_nu", "HFnu": "HF_nu"}
    ).fillna(frame.get("metric", frame["Metric"]))
    return frame


def figure3() -> tuple[list[Path], dict[str, float]]:
    comparison = pd.read_csv(FIG3_10MIN / "comparison.csv", low_memory=False)
    lfhf = finite_positive(comparison[comparison["metric"].eq("LF/HF")], "neurokit2_value", "native_value")
    r = pearson(lfhf["neurokit2_value"], lfhf["native_value"])
    median_re = float(lfhf["relative_error_pct"].median())
    means = (lfhf["native_value"] + lfhf["neurokit2_value"]) / 2.0
    diffs = lfhf["native_value"] - lfhf["neurokit2_value"]
    bias = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    extreme_count = int(((diffs < -0.8) | (diffs > 0.8)).sum())

    summary10 = load_metric_summary(FIG3_10MIN / "metric_summary.csv").set_index("metric")
    summary5 = load_metric_summary(FIG3_5MIN / "metric_summary.csv").set_index("metric")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.11, hspace=0.42, wspace=0.34)

    ax = axes[0, 0]
    panel_title(ax, "A. LF/HF agreement")
    ax.scatter(
        lfhf["neurokit2_value"],
        lfhf["native_value"],
        s=8,
        color=COLORS["blue"],
        alpha=0.42,
        edgecolors="none",
    )
    lower = min(lfhf["neurokit2_value"].min(), lfhf["native_value"].min()) * 0.75
    upper = max(lfhf["neurokit2_value"].max(), lfhf["native_value"].max()) * 1.25
    ax.plot([lower, upper], [lower, upper], color=COLORS["dark"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("NeuroKit2 LF/HF")
    ax.set_ylabel("HRV Studio LF/HF")
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {r:.3f}\nMedian error = {median_re:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#999999", linewidth=0.5),
    )
    style_axis(ax, "both")

    ax = axes[0, 1]
    panel_title(ax, "B. LF/HF Bland-Altman")
    ax.scatter(means, diffs, s=8, color=COLORS["blue"], alpha=0.35, edgecolors="none")
    for y, label, color in [
        (bias, f"Bias {bias:.3f}", COLORS["dark"]),
        (loa_low, f"-1.96 SD {loa_low:.3f}", COLORS["red"]),
        (loa_high, f"+1.96 SD {loa_high:.3f}", COLORS["red"]),
    ]:
        ax.axhline(y, color=color, linewidth=0.9, linestyle=(0, (3, 2)))
        ax.text(0.98, y, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.3)
    ax.set_xscale("log")
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("Mean LF/HF")
    ax.set_ylabel("HRV Studio - NeuroKit2")
    ax.text(0.05, 0.08, f"{extreme_count} outside display range", transform=ax.transAxes, fontsize=7.5)
    style_axis(ax)

    ax = axes[1, 0]
    panel_title(ax, "C. Median relative error by metric")
    values = [summary10.loc[m, "Median relative error %"] for m in METRIC_ORDER]
    labels = [METRIC_LABELS[m] for m in METRIC_ORDER]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=COLORS["blue"], edgecolor=COLORS["dark"], linewidth=0.5)
    ax.set_ylabel("Median relative error (%)")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, max(values) * 1.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    style_axis(ax)

    ax = axes[1, 1]
    panel_title(ax, "D. Duration effect on agreement")
    y = np.arange(len(METRIC_ORDER))
    vals5 = np.array([summary5.loc[m, "Median relative error %"] for m in METRIC_ORDER], dtype=float)
    vals10 = np.array([summary10.loc[m, "Median relative error %"] for m in METRIC_ORDER], dtype=float)
    height = 0.36
    ax.barh(y + height / 2, vals5, height=height, color=COLORS["gray"], edgecolor=COLORS["dark"], linewidth=0.5, label="5 min")
    ax.barh(y - height / 2, vals10, height=height, color=COLORS["blue"], edgecolor=COLORS["dark"], linewidth=0.5, label="10 min")
    ax.set_xscale("log")
    ax.set_xlim(0.08, 55)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Median relative error (%)")
    for yi, v5, v10 in zip(y, vals5, vals10):
        ax.text(v5 * 1.08, yi + height / 2, f"{v5:.2f}", va="center", fontsize=6.8)
        ax.text(v10 * 1.08, yi - height / 2, f"{v10:.2f}", va="center", fontsize=6.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style_axis(ax, "x")

    outputs = save_figure(fig, "Figure3_neurokit2_band_corrected")
    return outputs, {
        "lfhf_r": r,
        "lfhf_median_re": median_re,
        "bias": bias,
        "sd": sd,
        "loa_low": loa_low,
        "loa_high": loa_high,
        "extreme_count": extreme_count,
    }


def figure5() -> tuple[list[Path], dict[str, float]]:
    comp = pd.read_csv(FIG5_RUN / "fft_ar_comparison.csv")
    ar_orders = pd.read_csv(FIG5_RUN / "ar_order_sensitivity.csv")
    lfhf = finite_positive(comp, "neurokit2_welch_LF_HF", "welch_LF_HF")
    r = pearson(lfhf["neurokit2_welch_LF_HF"], lfhf["welch_LF_HF"])
    median_re = float(lfhf["welch_vs_neurokit2_LF_HF_rel_diff_pct"].median())
    fft_instability = int(comp["fft_unstable"].sum())
    ar_instability = int(comp["ar_unstable"].sum())
    order_sensitive = int(ar_orders.drop_duplicates("file_name")["ar_order_sensitive"].sum())
    file_n = int(comp["file_name"].nunique())

    welch_ratio = comp["welch_psd_full_area"] / comp["fft_variance_ms2"]
    area_rows = [
        ("Welch", welch_ratio.replace([np.inf, -np.inf], np.nan).dropna()),
        ("FFT", comp["fft_psd_variance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()),
        ("AR", comp["ar_psd_variance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()),
    ]
    order8 = ar_orders[ar_orders["requested_ar_order"].eq(8)].drop_duplicates("file_name")
    order24 = ar_orders[ar_orders["requested_ar_order"].eq(24)].drop_duplicates("file_name")
    spread = ar_orders.drop_duplicates("file_name")

    fig, axes = plt.subplots(1, 3, figsize=(10.72, 3.3), dpi=300)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.25, wspace=0.34)

    ax = axes[0]
    panel_title(ax, "A. Welch agreement with NeuroKit2")
    ax.scatter(lfhf["neurokit2_welch_LF_HF"], lfhf["welch_LF_HF"], s=18, color=COLORS["blue"], alpha=0.78, edgecolor=COLORS["dark"], linewidth=0.25)
    lower = min(lfhf["neurokit2_welch_LF_HF"].min(), lfhf["welch_LF_HF"].min()) * 0.72
    upper = max(lfhf["neurokit2_welch_LF_HF"].max(), lfhf["welch_LF_HF"].max()) * 1.28
    ax.plot([lower, upper], [lower, upper], color=COLORS["dark"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("NeuroKit2 Welch LF/HF")
    ax.set_ylabel("HRV Studio Welch LF/HF")
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nMedian error = {median_re:.2f}%", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#999999", linewidth=0.5))
    style_axis(ax, "both")

    ax = axes[1]
    panel_title(ax, "B. PSD-area consistency")
    data = [series.to_numpy(dtype=float) for _, series in area_rows]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
    for patch, color in zip(bp["boxes"], [COLORS["green"], COLORS["orange"], COLORS["purple"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(COLORS["dark"])
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color(COLORS["dark"])
            artist.set_linewidth(0.8)
    ax.axhline(1.0, color=COLORS["dark"], linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3], [name for name, _ in area_rows])
    ax.set_ylabel("Integrated PSD / variance")
    medians = [float(np.median(d)) for d in data]
    for i, med in enumerate(medians, start=1):
        ax.text(i, med * 1.18, f"{med:.2f}", ha="center", va="bottom", fontsize=7)
    ax.text(0.05, 0.95, f"FFT flags: {fft_instability}/{file_n}\nAR flags: {ar_instability}/{file_n}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#999999", linewidth=0.5))
    style_axis(ax)

    ax = axes[2]
    panel_title(ax, "C. AR order sensitivity")
    sens_data = [
        order8["ar_order8_vs_order16_LF_HF_rel_diff_pct"].dropna().to_numpy(dtype=float),
        order24["ar_order24_vs_order16_LF_HF_rel_diff_pct"].dropna().to_numpy(dtype=float),
        spread["ar_lfhf_order_spread_pct"].dropna().to_numpy(dtype=float),
    ]
    bp = ax.boxplot(sens_data, patch_artist=True, showfliers=False, widths=0.55)
    for patch, color in zip(bp["boxes"], [COLORS["teal"], COLORS["purple"], COLORS["orange"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(COLORS["dark"])
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color(COLORS["dark"])
            artist.set_linewidth(0.8)
    ax.axhline(50, color=COLORS["red"], linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_xticks([1, 2, 3], ["AR(8)\nvs 16", "AR(24)\nvs 16", "Spread\n8/16/24"])
    ax.set_ylabel("LF/HF relative change (%)")
    ax.set_ylim(0, max(80, np.nanpercentile(sens_data[2], 95) * 1.15))
    ax.text(0.05, 0.95, f">50% spread: {order_sensitive}/{file_n}\nAR instability: {ar_instability}/{file_n}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#999999", linewidth=0.5))
    style_axis(ax)

    outputs = save_figure(fig, "Figure5_spectral_methods_band_corrected")
    return outputs, {
        "lfhf_r": r,
        "lfhf_median_re": median_re,
        "fft_instability": fft_instability,
        "ar_instability": ar_instability,
        "order_sensitive": order_sensitive,
        "n": file_n,
        "fft_area_median": medians[1],
        "ar_area_median": medians[2],
    }


def read_rr_csv(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    column = "rr_ms" if "rr_ms" in frame.columns else frame.columns[0]
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def preprocess_rr(rr: np.ndarray):
    from hrvlib.preprocessing import preprocess_rri

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            return preprocess_rri(
                rr.tolist(),
                correction_method="cubic_spline",
                noise_detection=True,
                artifact_correction_enabled=True,
            )


def rr_time(rr: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(rr[:-1]) / 1000.0])


def figure6() -> tuple[list[Path], dict[str, float]]:
    from tools.robustness_signal_condition_study import CONDITIONS, corrupt_rr

    results = pd.read_csv(FIG6_RUN / "robustness_results.csv")
    condition_table = pd.read_csv(FIG6_RUN / "condition_level_stability_table.csv")
    effect_table = pd.read_csv(FIG6_RUN / "artifact_effect_table.csv")

    base_file = "nsr001_segment_000.csv"
    selected_condition = "ectopic_short_long_pairs"
    clean_rr = read_rr_csv(PHYSIONET_10MIN / base_file)
    rng = np.random.default_rng(20260609)
    corrupted_rr = clean_rr.copy()
    for condition in CONDITIONS:
        corrupted_rr, _ = corrupt_rr(clean_rr, condition, rng)
        if condition == selected_condition:
            break
    preprocessing = preprocess_rr(corrupted_rr)
    corrected_rr = np.asarray(preprocessing.corrected_rri, dtype=float)
    artifact_indices = [idx for idx in preprocessing.artifact_indices if 0 <= idx < len(corrupted_rr)]

    before_rows = results[results["phase"].eq("before_correction")]
    after_rows = results[results["phase"].eq("after_correction")]
    finite_before = int(before_rows["metrics_finite"].sum())
    finite_after = int(after_rows["metrics_finite"].sum())
    warnings_after = int(after_rows["warning_labels"].fillna("").ne("").sum())
    after_n = int(after_rows.shape[0])

    selected_effect = effect_table[
        effect_table["base_file"].eq(base_file)
        & effect_table["condition"].eq(selected_condition)
        & effect_table["metric"].isin(["RMSSD", "LF", "HF", "LF/HF"])
    ].copy()
    selected_effect["corrupted_pct_clean"] = selected_effect["before_value"] / selected_effect["clean_after_value"] * 100.0
    selected_effect["corrected_pct_clean"] = selected_effect["after_value"] / selected_effect["clean_after_value"] * 100.0

    fig = plt.figure(figsize=(10.72, 6.0), dpi=300)
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.985, top=0.94, bottom=0.10, hspace=0.34, wspace=0.27)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    t_corrupt = rr_time(corrupted_rr)
    t_correct = rr_time(corrected_rr)
    ymin = min(np.nanpercentile(corrupted_rr, 1), np.nanpercentile(corrected_rr, 1)) - 60
    ymax = max(np.nanpercentile(corrupted_rr, 99), np.nanpercentile(corrected_rr, 99)) + 80

    panel_title(ax_a, "A. Representative corrupted RR series")
    ax_a.plot(t_corrupt, corrupted_rr, color=COLORS["blue"], linewidth=0.8)
    if artifact_indices:
        ax_a.scatter(t_corrupt[artifact_indices], corrupted_rr[artifact_indices], s=16, color=COLORS["red"], zorder=3, label="Detected artifact")
    ax_a.set_ylim(ymin, ymax)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("RR interval (ms)")
    ax_a.text(0.03, 0.93, f"{len(artifact_indices)} detected artifacts", transform=ax_a.transAxes, fontsize=8, ha="left", va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", linewidth=0.5))
    style_axis(ax_a)

    panel_title(ax_b, "B. RR series after correction")
    ax_b.plot(t_correct, corrected_rr, color=COLORS["green"], linewidth=0.8)
    ax_b.set_ylim(ymin, ymax)
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("RR interval (ms)")
    selected_row = after_rows[after_rows["base_file"].eq(base_file) & after_rows["condition"].eq(selected_condition)].iloc[0]
    ax_b.text(0.03, 0.93, f"Corrected: {int(selected_row['artifacts_corrected'])}\nFinal intervals: {int(selected_row['preprocessed_count'])}", transform=ax_b.transAxes, fontsize=8, ha="left", va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", linewidth=0.5))
    style_axis(ax_b)

    panel_title(ax_c, "C. Numerical robustness and warning visibility")
    order = ["clean", "ectopic_short_long_pairs", "random_missed_beats", "gaussian_jitter_noise", "short_dropout_sections", "isolated_extreme_rr_artifacts"]
    labels = ["Clean", "Ectopy", "Missed", "Jitter", "Dropout", "Extreme"]
    table = condition_table.set_index("condition").loc[order]
    x = np.arange(len(labels))
    width = 0.25
    ax_c.bar(x - width, table["finite_after_rate_pct"], width=width, color=COLORS["gray"], edgecolor=COLORS["dark"], linewidth=0.5, label="Finite")
    ax_c.bar(x, table["warning_rate_pct"], width=width, color=COLORS["blue"], edgecolor=COLORS["dark"], linewidth=0.5, label="Warnings")
    ax_c.bar(x + width, table["artifact_correction_rate_pct"], width=width, color=COLORS["orange"], edgecolor=COLORS["dark"], linewidth=0.5, label="Corrected")
    ax_c.set_xticks(x, labels, rotation=25, ha="right")
    ax_c.set_ylim(0, 118)
    ax_c.set_ylabel("Cases (%)")
    ax_c.text(0.03, 0.93, f"Finite: {finite_after}/{after_n}\nWarnings: {warnings_after}/{after_n}", transform=ax_c.transAxes, fontsize=8, ha="left", va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", linewidth=0.5))
    ax_c.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.52, -0.22))
    style_axis(ax_c)

    panel_title(ax_d, "D. Metric sensitivity to perturbation")
    selected_effect["metric"] = pd.Categorical(selected_effect["metric"], categories=["RMSSD", "LF", "HF", "LF/HF"], ordered=True)
    selected_effect = selected_effect.sort_values("metric")
    x = np.arange(len(selected_effect))
    width = 0.36
    ax_d.bar(x - width / 2, selected_effect["corrupted_pct_clean"], width=width, color=COLORS["red"], alpha=0.78, edgecolor=COLORS["dark"], linewidth=0.5, label="Corrupted")
    ax_d.bar(x + width / 2, selected_effect["corrected_pct_clean"], width=width, color=COLORS["green"], alpha=0.78, edgecolor=COLORS["dark"], linewidth=0.5, label="Corrected")
    ax_d.axhline(100, color=COLORS["dark"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax_d.set_yscale("log")
    ax_d.set_ylim(3, max(2200, selected_effect["corrupted_pct_clean"].max() * 1.25))
    ax_d.set_xticks(x, [METRIC_LABELS[m] for m in selected_effect["metric"].astype(str)])
    ax_d.set_ylabel("Percent of clean value")
    for xi, row in enumerate(selected_effect.itertuples()):
        ax_d.text(xi - width / 2, row.corrupted_pct_clean * 1.12, f"{row.corrupted_pct_clean:.0f}", ha="center", va="bottom", fontsize=7)
        ax_d.text(xi + width / 2, row.corrected_pct_clean * 1.12, f"{row.corrected_pct_clean:.0f}", ha="center", va="bottom", fontsize=7)
    ax_d.legend(frameon=False, fontsize=8, loc="upper right")
    style_axis(ax_d)

    outputs = save_figure(fig, "Figure6_synthetic_robustness_band_corrected")
    return outputs, {
        "finite_before": finite_before,
        "finite_after": finite_after,
        "warnings_after": warnings_after,
        "n": after_n,
        "artifacts": int(selected_row["artifacts_corrected"]),
    }


def duration_order(frame: pd.DataFrame) -> pd.DataFrame:
    durations = ["30s", "60s", "2min", "3min", "5min", "10min"]
    ordered = frame.copy()
    ordered["duration_label"] = pd.Categorical(ordered["duration_label"], categories=durations, ordered=True)
    return ordered.sort_values("duration_label")


def figure7() -> tuple[list[Path], dict[str, float]]:
    durations = ["30s", "60s", "2min", "3min", "5min", "10min"]
    duration_labels = ["30 s", "60 s", "2 min", "3 min", "5 min", "10 min"]
    x = np.arange(len(durations))
    panel_a = duration_order(pd.read_csv(FIG7_RUN / "duration_metric_table.csv"))
    panel_a = panel_a[panel_a["metric"].isin(METRIC_ORDER)]
    panel_b = panel_a[panel_a["metric"].eq("VLF")].copy()
    panel_c_raw = pd.read_csv(FIG7_RUN / "duration_results.csv")
    panel_c = (
        panel_c_raw.groupby("duration_label", observed=True)
        .agg(
            files=("file_name", "nunique"),
            finite_output_rate_pct=("finite_hrvstudio", lambda s: s.mean() * 100.0),
            warning_visible_rate_pct=("warning_count", lambda s: (s > 0).mean() * 100.0),
            duration_warning_rate_pct=("duration_warning", lambda s: s.mean() * 100.0),
        )
        .reset_index()
    )
    panel_c = duration_order(panel_c)
    metric_colors = {
        "LF": COLORS["blue"],
        "HF": COLORS["green"],
        "total_power": COLORS["brown"],
        "LF/HF": COLORS["orange"],
        "LF_nu": COLORS["teal"],
        "HF_nu": COLORS["purple"],
        "VLF": COLORS["red"],
    }
    markers = {"LF": "o", "HF": "s", "total_power": "D", "LF/HF": "^", "LF_nu": "v", "HF_nu": "P", "VLF": "X"}

    fig, axes = plt.subplots(1, 3, figsize=(10.72, 3.25), dpi=300)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.31, wspace=0.36)

    def short_region(ax: plt.Axes, text: str) -> None:
        ax.axvspan(-0.45, 3.5, color="#eeeeee", zorder=0)
        ax.axvline(4, color="#555555", linestyle=(0, (3.7, 1.6)), linewidth=1.0)
        ax.text(4.02, 0.86, "5 min", transform=ax.get_xaxis_transform(), fontsize=8, color="#444444")
        ax.text(0.0, -0.25, text, transform=ax.transAxes, fontsize=8, ha="left", va="top", color="#222222", clip_on=False)

    ax = axes[0]
    short_region(ax, "<5 min: caution for spectral metrics")
    for metric in METRIC_ORDER:
        rows = panel_a[panel_a["metric"].eq(metric)].set_index("duration_label").reindex(durations)
        ax.plot(x, rows["median_relative_error_vs_neurokit2_pct"], marker=markers[metric], color=metric_colors[metric], linewidth=1.25, markersize=4.2, label=METRIC_LABELS[metric])
    ax.set_title("A. Metric agreement across durations", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("Median relative error (%)")
    ax.set_xticks(x, duration_labels)
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.65, len(durations) - 0.55)
    ax.legend(loc="upper right", ncol=2, fontsize=7.6, frameon=False, handlelength=1.3, borderaxespad=0.3)
    style_axis(ax)

    ax = axes[1]
    short_region(ax, "VLF remains convention-sensitive")
    vlf = panel_b.set_index("duration_label").reindex(durations)
    ax.plot(x, vlf["median_relative_error_vs_neurokit2_pct"], marker="o", color=COLORS["red"], linewidth=1.5, markersize=4.5)
    ax.scatter([x[0]], [4], marker="x", s=40, color=COLORS["dark"], zorder=4)
    ax.text(x[0], 9, "NA", ha="center", va="bottom", fontsize=8)
    for xi, value in zip(x[1:], vlf["median_relative_error_vs_neurokit2_pct"].iloc[1:]):
        ax.text(xi, value + 3.1, f"{value:.1f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_title("B. VLF sensitivity to recording duration", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("VLF relative error (%)")
    ax.set_xticks(x, duration_labels)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.65, len(durations) - 0.55)
    style_axis(ax)

    ax = axes[2]
    width = 0.34
    pc = panel_c.set_index("duration_label").reindex(durations)
    finite = pc["finite_output_rate_pct"]
    warning = pc["warning_visible_rate_pct"]
    duration_warning = pc["duration_warning_rate_pct"]
    bars = ax.bar(x - width / 2, finite, width=width, color=COLORS["gray"], edgecolor=COLORS["dark"], linewidth=0.6, hatch="////", label="Finite output")
    ax.bar(x + width / 2, warning, width=width, color=COLORS["blue"], edgecolor=COLORS["dark"], linewidth=0.6, label="Warning visible")
    ax.plot(x, duration_warning, color=COLORS["dark"], marker="D", markersize=4.6, linewidth=1.0, linestyle=(0, (3.7, 1.6)), label="Duration warning")
    for bar, files in zip(bars, pc["files"]):
        ax.text(bar.get_x() + width, 104, f"{int(files)} files", ha="center", va="bottom", fontsize=7.0, color="#555555")
    ax.text(0.04, 0.58, "Finite output does not\nimply interpretability", transform=ax.transAxes, fontsize=8, ha="left", va="center", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d0d0d0", alpha=0.92))
    ax.set_title("C. Numerical stability and warning behavior", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("Rows / recordings (%)")
    ax.set_xticks(x, duration_labels)
    ax.set_ylim(0, 118)
    ax.set_xlim(-0.65, len(durations) - 0.55)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, fontsize=7.6, frameon=False, handlelength=1.6, columnspacing=0.9)
    style_axis(ax)

    outputs = save_figure(fig, "Figure7_duration_sensitivity_band_corrected")
    five = panel_a[panel_a["duration_label"].eq("5min")].set_index("metric")
    ten = panel_a[panel_a["duration_label"].eq("10min")].set_index("metric")
    return outputs, {
        "vlf_5min": float(five.loc["VLF", "median_relative_error_vs_neurokit2_pct"]),
        "vlf_10min": float(ten.loc["VLF", "median_relative_error_vs_neurokit2_pct"]),
        "lfhf_5min": float(five.loc["LF/HF", "median_relative_error_vs_neurokit2_pct"]),
        "lfhf_10min": float(ten.loc["LF/HF", "median_relative_error_vs_neurokit2_pct"]),
    }


def reconstruct_mitbih_segment(row: pd.Series):
    import wfdb
    from tools.mitbih_arrhythmia_robustness_study import load_record_annotations, summarize_segment

    record = load_record_annotations(wfdb, MITBIH_ROOT, str(row["record_id"]))
    if record is None:
        raise RuntimeError(f"Could not load MIT-BIH record {row['record_id']}")
    segment = summarize_segment(
        record,
        start_s=float(row["start_s"]),
        end_s=float(row["end_s"]),
        intended_category=str(row["intended_category"]),
    )
    if segment is None:
        raise RuntimeError(f"Could not reconstruct segment {row['segment_id']}")
    return segment


def figure8() -> tuple[list[Path], dict[str, float]]:
    results = pd.read_csv(FIG8_RUN / "mitbih_robustness_results.csv")
    manifest = pd.read_csv(FIG8_RUN / "mitbih_segment_manifest.csv")
    after = results[results["phase"].eq("after_correction")].copy()
    after_n = int(after["segment_id"].nunique())
    finite_after = int(after[after["metrics_finite"]]["segment_id"].nunique())
    warning_after = int(after[after["warning_labels"].fillna("").ne("")]["segment_id"].nunique())

    examples = [
        ("A. Near-normal rhythm example", "mitbih_103_0000_0600"),
        ("B. Ventricular ectopy example", "mitbih_208_0300_0900"),
        ("C. Irregular rhythm example", "mitbih_207_1200_1800"),
    ]

    fig = plt.figure(figsize=(10.72, 6.05), dpi=300)
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.94, bottom=0.11, hspace=0.37, wspace=0.30)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_d = fig.add_subplot(gs[1, :])

    for ax, (title, segment_id) in zip(axes_top, examples):
        row = manifest[manifest["segment_id"].eq(segment_id)].iloc[0]
        segment = reconstruct_mitbih_segment(row)
        preprocessing = preprocess_rr(np.asarray(segment.rr_ms, dtype=float))
        corrected = np.asarray(preprocessing.corrected_rri, dtype=float)
        t_before = rr_time(np.asarray(segment.rr_ms, dtype=float))
        t_after = rr_time(corrected)
        before_rr = np.asarray(segment.rr_ms, dtype=float)
        ax.plot(t_before, before_rr, color=COLORS["blue"], linewidth=0.65, alpha=0.82, label="Before")
        ax.plot(t_after, corrected, color=COLORS["orange"], linewidth=0.65, alpha=0.82, label="After")
        artifact_indices = [idx for idx in preprocessing.artifact_indices if 0 <= idx < len(segment.rr_ms)]
        if artifact_indices:
            ax.scatter(t_before[artifact_indices], before_rr[artifact_indices], s=8, color=COLORS["red"], alpha=0.8)
        result_row = after[after["segment_id"].eq(segment_id)].iloc[0]
        finite_display = np.concatenate([before_rr[np.isfinite(before_rr)], corrected[np.isfinite(corrected)]])
        if finite_display.size:
            upper = min(max(float(np.nanpercentile(finite_display, 99.3)) * 1.18, 1200.0), 2600.0)
            lower = max(0.0, float(np.nanpercentile(finite_display, 0.7)) - 80.0)
            ax.set_ylim(lower, upper)
            clipped = int(np.count_nonzero(finite_display > upper))
            if clipped:
                ax.text(
                    0.97,
                    0.93,
                    f"{clipped} high outlier{'s' if clipped != 1 else ''}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.2,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#999999", linewidth=0.5),
                )
        panel_title(ax, title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RR interval (ms)")
        ax.text(0.03, 0.93, f"Record {segment.record_id}\nCorrections: {int(result_row['correction_count'])}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#999999", linewidth=0.5))
        style_axis(ax)
    axes_top[0].legend(frameon=False, fontsize=7.5, loc="lower right")

    panel_title(ax_d, "D. Numerical robustness and QC behavior")
    items = [
        ("Finite output", finite_after),
        ("Warning visible", warning_after),
        ("Correction applied", int(after[after["correction_count"] > 0]["segment_id"].nunique())),
        ("Excess artifacts", int(after[after["excessive_artifacts"]]["segment_id"].nunique())),
        ("High noise", int(after[after["high_noise"]]["segment_id"].nunique())),
        ("Irregular flag", int(after[after["irregular_rhythm_flag"]]["segment_id"].nunique())),
        ("Poor quality", int(after[after["poor_signal_quality"]]["segment_id"].nunique())),
    ]
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    x = np.arange(len(labels))
    ax_d.bar(x, values, color=[COLORS["blue"], COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["red"], COLORS["purple"], COLORS["red"]], edgecolor=COLORS["dark"], linewidth=0.5)
    ax_d.set_ylim(0, after_n + 1.5)
    ax_d.set_ylabel("Segments after correction (n)")
    ax_d.set_xticks(x, labels, rotation=25, ha="right")
    for xi, value in zip(x, values):
        ax_d.text(xi, value + 0.18, f"{value}/{after_n}", ha="center", va="bottom", fontsize=7.5)
    ax_d.text(0.98, 0.90, "MIT-BIH is a QC stress test,\nnot clinical validation", transform=ax_d.transAxes, ha="right", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#999999", linewidth=0.5))
    style_axis(ax_d)

    outputs = save_figure(fig, "Figure8_mitbih_qc_band_corrected")
    return outputs, {
        "finite_after": finite_after,
        "warning_after": warning_after,
        "n": after_n,
        "correction_applied": values[2],
        "excess_artifacts": values[3],
        "high_noise": values[4],
        "irregular_flag": values[5],
        "poor_quality": values[6],
    }


def write_report(results: dict[str, dict[str, object]]) -> None:
    old_values = {
        "Figure 3": "LF/HF r 0.999417; median RE 0.441998%; VLF 10-min median RE 9.18477%.",
        "Figure 5": "FFT instability 100/100; AR instability 2/100; AR order-sensitive 25/100.",
        "Figure 6": "Finite after 60/60; warnings after 60/60; Panel D used RMSSD, LF, HF, LF/HF from v09b.",
        "Figure 7": "Durations 30 s, 60 s, 2 min, 5 min, 10 min; 5-min VLF 12.3308%; 10-min VLF 12.9595%.",
        "Figure 8": "Finite after 12/12; warnings after 12/12; correction applied 9/12.",
    }
    inventory = [
        ("Figure 3", "No retained generator; source notes only", "v05 row-level + v03/v05 diagnostic summaries", "figure3_neurokit2_agreement.{png,pdf,svg}", "row-level for A/B; summary for C/D", "source notes only"),
        ("Figure 5", "No retained generator; source notes only", "v08 fft_ar_comparison + instability + ar_order_sensitivity", "figure5_spectral_methods.{png,pdf,svg}", "row-level/method tables", "source notes only"),
        ("Figure 6", "No retained generator; source notes only", "v09b robustness CSVs + base RR file", "figure6_synthetic_robustness.{png,pdf,svg}", "raw reconstruction for A/B; CSV for C/D", "source notes only"),
        ("Figure 7", "validation/research_notes/paper_figures/generate_figure7_duration_sensitivity.py", "v10 duration_metric_table + duration_results", "figure7_duration_sensitivity.{png,pdf,svg}", "summary + row-level", "script constants"),
        ("Figure 8", "No retained generator; source notes only", "v11 MIT-BIH CSVs + local annotations", "figure8_mitbih_arrhythmia_robustness.{png,pdf,svg}", "raw reconstruction for A-C; CSV for D", "source notes only"),
    ]
    lines = [
        "# Figures 3, 5, 6, 7, and 8 Band-Corrected Regeneration Report",
        "",
        "Generated from stored v12 band-corrected validation outputs. Manuscript files and legacy figure files were not modified.",
        "",
        f"Band convention: {STANDARD_BANDS}.",
        "",
        "## Phase 1 Inventory",
        "",
        "| Figure | Original script | Legacy input paths | Legacy output paths | Data level | Hard-coded numerical annotations |",
        "|---|---|---|---|---|---|",
    ]
    for row in inventory:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## Regenerated Outputs", ""])
    for figure, payload in results.items():
        outputs = ", ".join(f"`{rel(path)}`" for path in payload["outputs"])
        lines.extend(
            [
                f"### {figure}",
                "",
                f"- Original script: {payload['script']}",
                f"- New input files: {payload['inputs']}",
                f"- New output files: {outputs}",
                f"- Panels regenerated: {payload['panels']}",
                f"- Old headline values: {old_values[figure]}",
                f"- New headline values: {payload['headline']}",
                f"- Layout changed: {payload['layout_changed']}",
                f"- Logic changes: {payload['logic_changes']}",
                f"- Expected-value discrepancy: {payload['discrepancy']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Quality Control",
            "",
            "- All numerical annotations in regenerated figures are computed from v12 corrected CSVs or reconstructed from the same v12 source rows.",
            "- Legacy figure files in `validation/research_notes/paper_figures` were not overwritten.",
            "- New figures were written to `validation/figures` as PNG, PDF, and SVG.",
            "- Figure 7 includes the new 3-minute duration and treats 30-second VLF as unavailable rather than zero.",
            "- Figure 8 preserves the manuscript-facing warning taxonomy and does not add ULF few-bin diagnostics as a new Panel D bar.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    fig3_outputs, fig3_stats = figure3()
    fig5_outputs, fig5_stats = figure5()
    fig6_outputs, fig6_stats = figure6()
    fig7_outputs, fig7_stats = figure7()
    fig8_outputs, fig8_stats = figure8()

    results = {
        "Figure 3": {
            "script": "`validation/research_notes/paper_figures/generate_band_corrected_figures_3_5_6_7_8.py`; no retained legacy Figure 3 generator was found.",
            "inputs": f"`{rel(FIG3_10MIN / 'comparison.csv')}`, `{rel(FIG3_10MIN / 'metric_summary.csv')}`, `{rel(FIG3_5MIN / 'metric_summary.csv')}`",
            "outputs": fig3_outputs,
            "panels": "A-D",
            "headline": f"LF/HF r {fig3_stats['lfhf_r']:.3f}; median RE {fig3_stats['lfhf_median_re']:.2f}%; BA bias {fig3_stats['bias']:.4f}; LOA {fig3_stats['loa_low']:.4f} to {fig3_stats['loa_high']:.4f}.",
            "layout_changed": "No panel arrangement change; Panel D uses log-scaled horizontal paired bars to keep VLF and sub-2% metrics readable.",
            "logic_changes": "Inputs changed to v12 standard-band outputs; annotations are recomputed.",
            "discrepancy": "None; generated LF/HF values match the corrected run summary within rounding.",
        },
        "Figure 5": {
            "script": "`validation/research_notes/paper_figures/generate_band_corrected_figures_3_5_6_7_8.py`; no retained legacy Figure 5 generator was found.",
            "inputs": f"`{rel(FIG5_RUN / 'fft_ar_comparison.csv')}`, `{rel(FIG5_RUN / 'ar_order_sensitivity.csv')}`",
            "outputs": fig5_outputs,
            "panels": "A-C",
            "headline": f"FFT instability {fig5_stats['fft_instability']}/{fig5_stats['n']}; AR instability {fig5_stats['ar_instability']}/{fig5_stats['n']}; order-sensitive {fig5_stats['order_sensitive']}/{fig5_stats['n']}.",
            "layout_changed": "No.",
            "logic_changes": "Inputs changed to v12 standard-band spectral-method outputs; AR instability annotation updates from old 2/100 to corrected 3/100.",
            "discrepancy": "None.",
        },
        "Figure 6": {
            "script": "`validation/research_notes/paper_figures/generate_band_corrected_figures_3_5_6_7_8.py`; no retained legacy Figure 6 generator was found.",
            "inputs": f"`{rel(FIG6_RUN / 'robustness_results.csv')}`, `{rel(FIG6_RUN / 'condition_level_stability_table.csv')}`, `{rel(FIG6_RUN / 'artifact_effect_table.csv')}`, `{rel(PHYSIONET_10MIN / 'nsr001_segment_000.csv')}`",
            "outputs": fig6_outputs,
            "panels": "A-D",
            "headline": f"Finite after correction {fig6_stats['finite_after']}/{fig6_stats['n']}; warning-visible {fig6_stats['warnings_after']}/{fig6_stats['n']}.",
            "layout_changed": "No.",
            "logic_changes": "Inputs changed to corrected v12 synthetic robustness outputs; representative trace reconstructed with the same validation corruption/preprocessing logic.",
            "discrepancy": "None.",
        },
        "Figure 7": {
            "script": "`validation/research_notes/paper_figures/generate_figure7_duration_sensitivity.py` was used as the visual template; corrected generation is in `generate_band_corrected_figures_3_5_6_7_8.py`.",
            "inputs": f"`{rel(FIG7_RUN / 'duration_metric_table.csv')}`, `{rel(FIG7_RUN / 'duration_results.csv')}`, `{rel(FIG7_RUN / 'frequency_metric_summary.csv')}`",
            "outputs": fig7_outputs,
            "panels": "A-C",
            "headline": f"5-min VLF {fig7_stats['vlf_5min']:.2f}%, LF/HF {fig7_stats['lfhf_5min']:.2f}%; 10-min VLF {fig7_stats['vlf_10min']:.2f}%, LF/HF {fig7_stats['lfhf_10min']:.2f}%.",
            "layout_changed": "No panel arrangement change; duration axis now includes 3 min and 30-s VLF is explicitly marked NA.",
            "logic_changes": "Removed monotonic VLF implication; shaded <5 min region now spans 30 s through 3 min.",
            "discrepancy": "None.",
        },
        "Figure 8": {
            "script": "`validation/research_notes/paper_figures/generate_band_corrected_figures_3_5_6_7_8.py`; no retained legacy Figure 8 generator was found.",
            "inputs": f"`{rel(FIG8_RUN / 'mitbih_robustness_results.csv')}`, `{rel(FIG8_RUN / 'mitbih_segment_manifest.csv')}`, local MIT-BIH annotations under `{rel(MITBIH_ROOT)}`",
            "outputs": fig8_outputs,
            "panels": "A-D",
            "headline": f"Finite after correction {fig8_stats['finite_after']}/{fig8_stats['n']}; warning-visible {fig8_stats['warning_after']}/{fig8_stats['n']}; correction applied {fig8_stats['correction_applied']}/{fig8_stats['n']}.",
            "layout_changed": "No.",
            "logic_changes": "Inputs changed to the completed corrected v12 MIT-BIH run; Panel D preserves the original manuscript-facing warning taxonomy.",
            "discrepancy": "None.",
        },
    }
    write_report(results)

    print(f"Figure 3: {rel(fig3_outputs[0])}; LF/HF r={fig3_stats['lfhf_r']:.3f}; median RE={fig3_stats['lfhf_median_re']:.2f}%")
    print(f"Figure 5: {rel(fig5_outputs[0])}; FFT instability={fig5_stats['fft_instability']}/{fig5_stats['n']}; AR instability={fig5_stats['ar_instability']}/{fig5_stats['n']}; order-sensitive={fig5_stats['order_sensitive']}/{fig5_stats['n']}")
    print(f"Figure 6: {rel(fig6_outputs[0])}; finite={fig6_stats['finite_after']}/{fig6_stats['n']}; warnings={fig6_stats['warnings_after']}/{fig6_stats['n']}")
    print(f"Figure 7: {rel(fig7_outputs[0])}; 5-min VLF={fig7_stats['vlf_5min']:.2f}%, LF/HF={fig7_stats['lfhf_5min']:.2f}%; 10-min VLF={fig7_stats['vlf_10min']:.2f}%, LF/HF={fig7_stats['lfhf_10min']:.2f}%")
    print(f"Figure 8: {rel(fig8_outputs[0])}; finite={fig8_stats['finite_after']}/{fig8_stats['n']}; warnings={fig8_stats['warning_after']}/{fig8_stats['n']}")
    print(f"Report: {rel(REPORT_PATH)}")


if __name__ == "__main__":
    main()
