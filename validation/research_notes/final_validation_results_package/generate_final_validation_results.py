from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "validation" / "research_notes" / "final_validation_results_package"

CLEANED = ROOT / "validation" / "research_notes" / "manual_review_sensitivity_analysis" / "cleaned_valid_only.csv"
VALID_ONLY = ROOT / "validation" / "kubios_subset" / "parsed_results_50_none_120s_75pct_after_arm_a" / "comparison_valid_only.csv"
AUTO_EXCLUDED = ROOT / "validation" / "kubios_subset" / "parsed_results_50_none_120s_75pct_after_arm_a" / "comparison_excluded_files.csv"
MANUAL_EXCLUDED = ROOT / "validation" / "research_notes" / "manual_review_sensitivity_analysis" / "excluded_files_log.csv"

METRICS = ["VLF", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu"]
CATEGORIES = [
    "clean_high_agreement",
    "vlf_sensitive",
    "remaining_outliers",
    "short_or_adjusted",
    "random_controls",
]


def rel_iqr(series: pd.Series) -> float:
    q1, q3 = series.quantile([0.25, 0.75])
    return q3 - q1


def finite_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)


def summarize_group(group: pd.DataFrame) -> dict:
    diff = group["hrvstudio_native_value"] - group["kubios_value"]
    rel = group["hrvstudio_relative_error_pct"]
    return {
        "n_files": group["subset_id"].nunique(),
        "n_rows": len(group),
        "mean_relative_error_pct": rel.mean(),
        "median_relative_error_pct": rel.median(),
        "iqr_relative_error_pct": rel_iqr(rel),
        "std_relative_error_pct": rel.std(ddof=1),
        "pearson_correlation": finite_corr(group["kubios_value"], group["hrvstudio_native_value"], "pearson"),
        "spearman_correlation": finite_corr(group["kubios_value"], group["hrvstudio_native_value"], "spearman"),
        "bias_hrvstudio_minus_kubios": diff.mean(),
        "mae": diff.abs().mean(),
        "rmse": math.sqrt(np.mean(np.square(diff))),
    }


def fmt(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str], digits=2) -> str:
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(fmt(value, digits))
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def save_metric_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        row = {"metric": metric}
        row.update(summarize_group(cleaned[cleaned["metric"] == metric]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "final_kubios_metric_table.csv", index=False, float_format="%.6f")
    return table


def save_category_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        row = {"category": category}
        row.update(summarize_group(cleaned[cleaned["category"] == category]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "final_kubios_category_table.csv", index=False, float_format="%.6f")
    return table


def save_exclusion_table() -> pd.DataFrame:
    auto = pd.read_csv(AUTO_EXCLUDED)
    manual = pd.read_csv(MANUAL_EXCLUDED)
    manual_by_subset = manual.set_index("subset_id").to_dict("index")

    rows = []
    for _, r in auto.iterrows():
        subset = r["subset_id"]
        reasons = str(r["exclusion_reasons"])
        if subset in {"CH004", "CH012"}:
            exclusion_type = "automatic QC exclusion; validation input failure"
            reason_class = "validation input failure"
            primary_reason = "No RRI/PPI intervals found; validation rows missing"
        elif subset == "OUT005":
            exclusion_type = "automatic QC exclusion; preprocessing instability"
            reason_class = "preprocessing instability"
            primary_reason = "Native power >100x Kubios; adjusted Welch overlap event"
        elif subset == "VLF005":
            exclusion_type = "automatic QC exclusion; invalid/short signal; preprocessing instability"
            reason_class = "invalid/short signal; preprocessing instability"
            primary_reason = "Nonfinite metrics, all native powers zero, native total power nonpositive; adjusted Welch overlap event"
        else:
            exclusion_type = "automatic QC exclusion"
            reason_class = "automatic QC exclusion"
            primary_reason = reasons
        rows.append(
            {
                "subset_id": subset,
                "category": r["category"],
                "exclusion_stage": "automatic_qc",
                "exclusion_type": exclusion_type,
                "reason_class": reason_class,
                "primary_reason": primary_reason,
                "raw_exclusion_reasons": reasons,
                "source_file": r["source_file"],
                "kubios_input_txt": r["kubios_input_txt"],
                "status_in_input_csv": "excluded_before_valid_only",
                "status_after_cleaning": "not_present",
                "manual_review_status": manual_by_subset.get(subset, {}).get("status_in_input_csv", ""),
                "max_hrvstudio_relative_error_pct": r["max_hrvstudio_relative_error_pct"],
                "max_neurokit_relative_error_pct": r["max_neurokit_relative_error_pct"],
            }
        )

    auto_subsets = set(auto["subset_id"])
    for _, r in manual.iterrows():
        subset = r["subset_id"]
        if subset in auto_subsets:
            continue
        if subset == "VLF002":
            reason_class = "invalid/short signal; preprocessing instability"
            primary_reason = "Manual-review exclusion after zero/near-zero HRV Studio powers and adjusted Welch overlap event"
        elif subset == "RC003":
            reason_class = "invalid/short signal; preprocessing instability"
            primary_reason = "Manual-review exclusion after zero HRV Studio powers/ratios and adjusted Welch overlap event"
        else:
            reason_class = "manual-review exclusion"
            primary_reason = str(r["notes"])
        rows.append(
            {
                "subset_id": subset,
                "category": r["category"],
                "exclusion_stage": "manual_review",
                "exclusion_type": "manual-review exclusion",
                "reason_class": reason_class,
                "primary_reason": primary_reason,
                "raw_exclusion_reasons": str(r["notes"]),
                "source_file": r["source_file"],
                "kubios_input_txt": r["kubios_input_txt"],
                "status_in_input_csv": r["status_in_input_csv"],
                "status_after_cleaning": r["status_after_cleaning"],
                "manual_review_status": "present_then_removed",
                "max_hrvstudio_relative_error_pct": "",
                "max_neurokit_relative_error_pct": "",
            }
        )

    table = pd.DataFrame(rows)
    table = table[table["subset_id"].isin(["CH004", "CH012", "OUT005", "VLF005", "VLF002", "RC003"])]
    order = {"CH004": 0, "CH012": 1, "OUT005": 2, "VLF005": 3, "VLF002": 4, "RC003": 5}
    table = table.sort_values("subset_id", key=lambda s: s.map(order)).reset_index(drop=True)
    table.to_csv(OUT / "final_exclusion_table.csv", index=False)
    return table


def setup_plot():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def save_relative_error_boxplot(cleaned: pd.DataFrame):
    data = [cleaned.loc[cleaned["metric"] == m, "hrvstudio_relative_error_pct"].values for m in METRICS]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.boxplot(data, tick_labels=METRICS, showfliers=False, patch_artist=True, medianprops={"color": "#111111"})
    for patch in ax.artists:
        patch.set_facecolor("#d9e8f5")
    ax.set_ylabel("Relative error vs Kubios (%)")
    ax.set_xlabel("Frequency-domain metric")
    ax.set_title("HRV Studio relative error by metric after automatic QC and manual review")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "relative_error_boxplot_by_metric.png", bbox_inches="tight")
    plt.close(fig)


def save_bland_altman(cleaned: pd.DataFrame, metric: str, filename: str):
    d = cleaned[cleaned["metric"] == metric].copy()
    avg = (d["hrvstudio_native_value"] + d["kubios_value"]) / 2
    diff = d["hrvstudio_native_value"] - d["kubios_value"]
    mean_diff = diff.mean()
    sd = diff.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.scatter(avg, diff, s=30, alpha=0.78, color="#2c6f9f", edgecolor="white", linewidth=0.4)
    ax.axhline(mean_diff, color="#111111", linewidth=1.2, label=f"Bias = {mean_diff:.2f}")
    ax.axhline(mean_diff + 1.96 * sd, color="#b33a3a", linestyle="--", linewidth=1, label="95% limits")
    ax.axhline(mean_diff - 1.96 * sd, color="#b33a3a", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Mean of HRV Studio and Kubios {metric}")
    ax.set_ylabel(f"HRV Studio - Kubios {metric}")
    ax.set_title(f"Bland-Altman comparison: {metric}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def save_scatter(cleaned: pd.DataFrame, metric: str, filename: str):
    d = cleaned[cleaned["metric"] == metric].copy()
    x = d["kubios_value"]
    y = d["hrvstudio_native_value"]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(x, y, s=32, alpha=0.78, color="#267060", edgecolor="white", linewidth=0.4)
    positive = pd.concat([x, y], axis=0)
    positive = positive[positive > 0]
    if len(positive) and metric in {"LF", "HF", "LF/HF"}:
        ax.set_xscale("log")
        ax.set_yscale("log")
        lower = positive.min() * 0.8
        upper = positive.max() * 1.2
    else:
        lower = min(x.min(), y.min())
        upper = max(x.max(), y.max())
    ax.plot([lower, upper], [lower, upper], color="#111111", linewidth=1, label="Identity")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"Kubios {metric}")
    ax.set_ylabel(f"HRV Studio {metric}")
    ax.set_title(f"Scatter comparison: {metric}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def save_before_after_qc(valid_only: pd.DataFrame, cleaned: pd.DataFrame):
    before = valid_only["hrvstudio_relative_error_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    after = cleaned["hrvstudio_relative_error_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].boxplot(
        [before, after],
        tick_labels=["Before\nmanual review", "After\nmanual review"],
        showfliers=False,
        patch_artist=True,
    )
    axes[0].set_ylabel("Relative error vs Kubios (%)")
    axes[0].set_title("Distribution shift")
    axes[0].grid(axis="y", alpha=0.25)
    bins = np.linspace(0, min(250, max(before.quantile(0.95), after.quantile(0.95))), 28)
    axes[1].hist(before.clip(upper=bins[-1]), bins=bins, alpha=0.55, label="Before manual review", color="#9c6b4e")
    axes[1].hist(after.clip(upper=bins[-1]), bins=bins, alpha=0.65, label="After manual review", color="#2c6f9f")
    axes[1].set_xlabel("Relative error vs Kubios (%)")
    axes[1].set_ylabel("Metric rows")
    axes[1].set_title("QC error distribution")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "before_after_qc_error_distribution.png", bbox_inches="tight")
    plt.close(fig)


def write_reports(cleaned: pd.DataFrame, metric_table: pd.DataFrame, category_table: pd.DataFrame, exclusion_table: pd.DataFrame):
    overall = summarize_group(cleaned)
    metric_short = metric_table[["metric", "mean_relative_error_pct", "median_relative_error_pct", "pearson_correlation", "spearman_correlation"]]
    category_short = category_table[["category", "n_files", "mean_relative_error_pct", "median_relative_error_pct"]]

    final_summary = f"""# Final Validation Results Package

Generated from frozen validation artifacts after the Arm A no-detrend fix. Production HRV code was not modified and full validation was not rerun.

## Source Artifacts

- Cleaned Kubios comparison: `validation/research_notes/manual_review_sensitivity_analysis/cleaned_valid_only.csv`
- Pre-cleaning Kubios valid-only comparison: `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_valid_only.csv`
- Exclusion records: `validation/research_notes/manual_review_sensitivity_analysis/excluded_files_log.csv` and `validation/kubios_subset/parsed_results_50_none_120s_75pct_after_arm_a/comparison_excluded_files.csv`
- Large-scale summaries: `validation/runs/v03_physionet_5min_neurokit2/diagnostic_summary.md` and `validation/runs/v05_physionet_10min_segment_linear/diagnostic_summary.md`

## Large-Scale NeuroKit2 Validation Summary

### 5-min PhysioNet Validation

- Analyzed 106,253 metric rows across 15,179 files.
- Rows exceeding 20% relative error: 15,089; rows exceeding 80% relative error: 177.
- Metric-level median relative errors were low for LF (0.56%), HF (0.18%), LF/HF (0.66%), LF_nu (0.20%), and HF_nu (0.39%).
- VLF and total_power were the main higher-discrepancy metrics: VLF mean/median relative error 31.45%/28.23%; total_power mean/median relative error 15.15%/9.26%.
- The diagnostic report identified 9,899 files with an absolute-power mismatch pattern while preserving relative spectral distribution.

### 10-min PhysioNet Validation

- Analyzed 53,186 metric rows across 7,598 files using segment-wise linear Welch detrending.
- Rows exceeding 20% relative error: 271; rows exceeding 80% relative error: 2.
- Metric-level median relative errors were low for LF (0.39%), HF (0.11%), total_power (0.87%), LF/HF (0.44%), LF_nu (0.14%), and HF_nu (0.26%).
- VLF remained the largest p90 relative-error metric, with mean/median relative error 9.95%/9.18% and p90 14.81%.
- The 10-min analysis reduced the broad absolute-power discrepancy pattern to 197 files.

### Detrending Convention Finding

The large-scale runs indicate that Welch detrending convention materially affects VLF and total_power. Segment-wise linear detrending in the 10-min run markedly reduced discrepancies relative to the 5-min diagnostic pattern, while the Kubios subset was intentionally evaluated with no detrending to match the Kubios export convention after the Arm A fix.

### VLF/Total-Power Issue Explanation

VLF and total_power are especially sensitive because low-frequency bins, DC removal, interpolation, windowing, segment length, and bin-boundary integration can change the absolute spectral area without necessarily changing LF/HF or normalized-unit structure. This supports interpreting absolute-power disagreement as a methodological-convention issue unless raw-signal review indicates an implementation or input failure.

## Kubios Subset Summary

- Kubios exports parsed: 50/50.
- Matched validation rows: 48 files.
- Valid after automatic QC: 46 files.
- Retained after manual-review exclusions: 44 files.
- Final cleaned HRV Studio overall mean relative error: {overall['mean_relative_error_pct']:.2f}%.
- Final cleaned HRV Studio overall median relative error: {overall['median_relative_error_pct']:.2f}%.

### Metric-Level HRV Studio Agreement

{markdown_table(metric_short, ['metric', 'mean_relative_error_pct', 'median_relative_error_pct', 'pearson_correlation', 'spearman_correlation'], ['Metric', 'Mean rel err %', 'Median rel err %', 'Pearson r', 'Spearman rho'])}

### Category-Level HRV Studio Agreement

{markdown_table(category_short, ['category', 'n_files', 'mean_relative_error_pct', 'median_relative_error_pct'], ['Category', 'Files', 'Mean rel err %', 'Median rel err %'])}

## Exclusion Summary

{markdown_table(exclusion_table[['subset_id', 'category', 'exclusion_stage', 'reason_class', 'primary_reason']], ['subset_id', 'category', 'exclusion_stage', 'reason_class', 'primary_reason'], ['Subset', 'Category', 'Stage', 'Reason class', 'Primary reason'])}

## Main Interpretation

- Agreement is strongest for normalized and ratio metrics, especially LF_nu, LF/HF, and HF_nu by median relative error.
- Absolute spectral powers remain more sensitive to methodological conventions and retained pathological records.
- VLF and total_power are particularly convention-sensitive because they depend heavily on low-frequency treatment, detrending, segment duration, and frequency-bin integration.
- Pathological or unstable files require QC before being used for agreement claims; all-matched means are not appropriate as headline results.

## Paper-Ready Cautious Conclusions

The validation evidence supports cautious use of HRV Studio frequency-domain outputs when preprocessing settings and quality-control criteria are explicitly reported. The strongest support is for LF/HF and normalized-unit metrics, where the relative spectral distribution is often preserved. The evidence does not support claiming full equivalence with Kubios, particularly for absolute spectral powers and VLF-derived quantities, which remain sensitive to methodological conventions and input quality.

## Remaining Work Before Manuscript

- Review the retained high-error files, especially `OUT006`, `OUT003`, `OUT001`, `CH001`, `CH005`, `RC001`, `RC004`, and `OUT004`.
- Confirm Kubios report sample-limit and segment-selection fields for the subset before presenting absolute-power comparisons as definitive.
- Decide whether the manuscript should report medians as the primary Kubios subset statistic, with means treated as outlier-sensitive secondary statistics.
- Add a methods paragraph documenting detrending, Welch segment length, overlap, interpolation rate, window type, and QC exclusions.
"""

    (OUT / "final_validation_summary.md").write_text(final_summary, encoding="utf-8")

    paper_draft = f"""# Paper Results Draft

## Results: Large-Scale NeuroKit2 Validation

Frequency-domain outputs were evaluated against NeuroKit2 in two large PhysioNet validation runs. In the 5-min run, 106,253 metric rows from 15,179 files were analyzed. Median relative errors were low for LF (0.56%), HF (0.18%), LF/HF (0.66%), LF_nu (0.20%), and HF_nu (0.39%). VLF and total_power showed larger discrepancies, with VLF mean/median relative error of 31.45%/28.23% and total_power mean/median relative error of 15.15%/9.26%.

In the 10-min segment-wise linear detrending run, 53,186 metric rows from 7,598 files were analyzed. Discrepancies were substantially reduced for most metrics: median relative errors were 0.39% for LF, 0.11% for HF, 0.87% for total_power, 0.44% for LF/HF, 0.14% for LF_nu, and 0.26% for HF_nu. VLF remained the largest p90 relative-error metric, with mean/median relative error of 9.95%/9.18%. These findings indicate strong numerical agreement for most LF/HF and normalized metrics under matched large-scale validation conditions, while VLF remains more sensitive to the spectral estimation convention.

## Results: Kubios Subset Validation

The Kubios subset workflow parsed all 50 available Kubios exports. Forty-eight files matched validation rows, 46 files remained after automatic QC, and 44 files remained after manual-review exclusions. In the final cleaned 44-file subset, HRV Studio had an overall mean relative error of {overall['mean_relative_error_pct']:.2f}% and an overall median relative error of {overall['median_relative_error_pct']:.2f}% versus Kubios.

Metric-level results were heterogeneous. Median relative errors were {metric_table.loc[metric_table.metric == 'LF_nu', 'median_relative_error_pct'].iloc[0]:.2f}% for LF_nu, {metric_table.loc[metric_table.metric == 'HF_nu', 'median_relative_error_pct'].iloc[0]:.2f}% for HF_nu, and {metric_table.loc[metric_table.metric == 'LF/HF', 'median_relative_error_pct'].iloc[0]:.2f}% for LF/HF. Absolute spectral powers showed larger median relative errors: {metric_table.loc[metric_table.metric == 'VLF', 'median_relative_error_pct'].iloc[0]:.2f}% for VLF, {metric_table.loc[metric_table.metric == 'LF', 'median_relative_error_pct'].iloc[0]:.2f}% for LF, {metric_table.loc[metric_table.metric == 'HF', 'median_relative_error_pct'].iloc[0]:.2f}% for HF, and {metric_table.loc[metric_table.metric == 'total_power', 'median_relative_error_pct'].iloc[0]:.2f}% for total_power. These results support reporting medians as the primary subset statistic because means remain strongly influenced by retained edge cases.

## Discussion: Interpretation of Frequency-Domain Discrepancies

The validation results do not justify a claim that HRV Studio is fully equivalent to Kubios. Instead, they indicate that agreement is strongest for LF/HF and normalized metrics, where relative spectral distribution is less affected by absolute power scaling and low-frequency conventions. Absolute power metrics, especially VLF and total_power, remain sensitive to methodological choices including detrending, interpolation, Welch segment length and overlap, windowing, and frequency-bin integration.

The manual-review and QC results also show that pathological files can dominate mean relative error. Files with validation input failures, nonfinite outputs, zero or near-zero native powers, extreme power ratios, short/adjusted Welch windows, or preprocessing instability should not be pooled with clean files for headline agreement claims. A cautious manuscript interpretation is that HRV Studio frequency-domain metrics show good agreement for ratio and normalized measures under matched settings, while absolute spectral powers require explicit reporting of preprocessing conventions and continued QC-based review.
"""

    (OUT / "paper_results_draft.md").write_text(paper_draft, encoding="utf-8")


def main():
    setup_plot()
    cleaned = pd.read_csv(CLEANED)
    valid_only = pd.read_csv(VALID_ONLY)
    cleaned = cleaned[cleaned["metric"].isin(METRICS)].copy()
    valid_only = valid_only[valid_only["metric"].isin(METRICS)].copy()

    metric_table = save_metric_table(cleaned)
    category_table = save_category_table(cleaned)
    exclusion_table = save_exclusion_table()

    save_relative_error_boxplot(cleaned)
    save_bland_altman(cleaned, "LF", "bland_altman_LF.png")
    save_bland_altman(cleaned, "HF", "bland_altman_HF.png")
    save_bland_altman(cleaned, "total_power", "bland_altman_total_power.png")
    save_scatter(cleaned, "LF", "scatter_LF.png")
    save_scatter(cleaned, "HF", "scatter_HF.png")
    save_scatter(cleaned, "LF/HF", "scatter_LFHF.png")
    save_before_after_qc(valid_only, cleaned)

    write_reports(cleaned, metric_table, category_table, exclusion_table)

    overall = summarize_group(cleaned)
    print(f"Created final validation package at {OUT}")
    print(f"Cleaned files: {cleaned['subset_id'].nunique()}")
    print(f"Metric rows: {len(cleaned)}")
    print(f"Overall HRV Studio mean relative error: {overall['mean_relative_error_pct']:.2f}%")
    print(f"Overall HRV Studio median relative error: {overall['median_relative_error_pct']:.2f}%")


if __name__ == "__main__":
    main()
