"""Diagnostic audit for HRV frequency-band integration definitions.

This script does not modify production code or rerun full validation. It
reuses current PSD outputs from HRVFreqDomainAnalysis and compares current
metric integration with two isolated candidate integration schemes.
"""

from __future__ import annotations

import csv
import math
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis


REPORT = ROOT / "validation" / "research_notes" / "frequency_band_definition_audit.md"
IMPACT_CSV = ROOT / "validation" / "research_notes" / "frequency_band_definition_impact.csv"
KUBIOS_CONVENTION_CSV = (
    ROOT / "validation" / "research_notes" / "frequency_band_definition_kubios_psd_check.csv"
)

KUBIOS_MATCHED_PER = (
    ROOT / "validation" / "kubios_subset" / "frequency_domain_kubios_matched_sequence_per_recording.csv"
)
KUBIOS_AUDIT = (
    ROOT
    / "validation"
    / "kubios_subset"
    / "frequency_domain_sequence_debug"
    / "frequency_domain_sequence_audit.csv"
)
SMOOTHNESS_MANIFEST = (
    ROOT / "validation" / "kubios_subset" / "smoothness_priors_pilot" / "pilot_manifest.csv"
)
CLEANED_KUBIOS = (
    ROOT / "validation" / "research_notes" / "manual_review_sensitivity_analysis" / "cleaned_valid_only.csv"
)

METRICS = [
    "ulf_power",
    "vlf_power",
    "lf_power",
    "hf_power",
    "total_power",
    "lf_hf_ratio",
    "lf_nu",
    "hf_nu",
    "ulf_power_nu",
    "vlf_power_nu",
    "lf_power_nu",
    "hf_power_nu",
    "relative_lf_power",
    "relative_hf_power",
]
POWER_METRICS = {"ulf_power", "vlf_power", "lf_power", "hf_power", "total_power"}
KUBIOS_MAT_POWER_FIELDS = {
    "VLF": "VLF_power",
    "LF": "LF_power",
    "HF": "HF_power",
    "total_power": "tot_power",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load_rr_ms(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if text and "," in text[0] and any(name in text[0].lower() for name in ["rr", "rri"]):
        frame = pd.read_csv(path)
        for column in ["rr_ms", "rri_ms", "RR", "rr"]:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
                if values.size and np.nanmedian(values) < 10:
                    values *= 1000.0
                return values
    values = []
    for line in text:
        line = line.strip()
        if not line or any(ch.isalpha() for ch in line):
            continue
        token = line.replace(",", " ").replace(";", " ").split()[0]
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value) and value > 0:
            values.append(value)
    rr = np.asarray(values, dtype=float)
    if rr.size and np.nanmedian(rr) < 10:
        rr *= 1000.0
    return rr


def truncate_rr(rr_ms: np.ndarray, duration_s: float) -> np.ndarray:
    cumulative = np.cumsum(rr_ms) / 1000.0
    keep = cumulative <= duration_s
    if np.count_nonzero(keep) >= 3:
        return rr_ms[keep]
    return rr_ms[: min(len(rr_ms), max(3, np.count_nonzero(cumulative <= duration_s) + 1))]


def selected_rr_from_mat(path: Path) -> np.ndarray:
    res = loadmat(path, squeeze_me=True, struct_as_record=False)["Res"]
    return np.asarray(res.HRV.Data.RRs).squeeze().astype(float) * 1000.0


def integrate(freqs: np.ndarray, psd: np.ndarray, mask: np.ndarray) -> float:
    mask = mask & np.isfinite(freqs) & np.isfinite(psd)
    if np.count_nonzero(mask) < 2:
        return 0.0
    value = float(np.trapezoid(psd[mask], freqs[mask]))
    if not math.isfinite(value):
        return math.nan
    return max(0.0, value)


def candidate_metrics(freqs: np.ndarray, psd: np.ndarray, candidate: str) -> dict[str, float]:
    if candidate == "candidate_a":
        masks = {
            "ulf_power": (freqs >= 0.0) & (freqs < 0.003),
            "vlf_power": (freqs >= 0.003) & (freqs < 0.04),
            "lf_power": (freqs >= 0.04) & (freqs < 0.15),
            "hf_power": (freqs >= 0.15) & (freqs <= 0.40),
            "total_power": (freqs >= 0.0) & (freqs <= 0.40),
        }
    elif candidate == "candidate_b":
        masks = {
            "ulf_power": (freqs > 0.0) & (freqs < 0.003),
            "vlf_power": (freqs >= 0.003) & (freqs < 0.04),
            "lf_power": (freqs >= 0.04) & (freqs < 0.15),
            "hf_power": (freqs >= 0.15) & (freqs <= 0.40),
            "total_power": (freqs > 0.0) & (freqs <= 0.40),
        }
    else:
        raise ValueError(candidate)

    out = {metric: integrate(freqs, psd, mask) for metric, mask in masks.items()}
    total = out["total_power"]
    for band in ["ulf", "vlf", "lf", "hf"]:
        power = out[f"{band}_power"]
        out[f"{band}_power_nu"] = (power / total * 100.0) if total > 0 else math.nan
    lf = out["lf_power"]
    hf = out["hf_power"]
    out["lf_hf_ratio"] = lf / hf if hf > 1e-10 else (math.inf if lf > 1e-10 else math.nan)
    lf_hf_sum = lf + hf
    if lf_hf_sum > 0 and math.isfinite(lf_hf_sum):
        out["lf_nu"] = lf / lf_hf_sum * 100.0
        out["hf_nu"] = hf / lf_hf_sum * 100.0
        out["relative_lf_power"] = out["lf_nu"]
        out["relative_hf_power"] = out["hf_nu"]
    else:
        out["lf_nu"] = math.nan
        out["hf_nu"] = math.nan
        out["relative_lf_power"] = math.nan
        out["relative_hf_power"] = math.nan
    return out


def current_metrics(results: dict, method: str) -> dict[str, float]:
    prefix = "" if method == "welch" else f"{method}_"
    return {metric: float(results.get(f"{prefix}{metric}", math.nan)) for metric in METRICS}


def analyze_sample(sample: dict) -> list[dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(StringIO()):
            analyzer = HRVFreqDomainAnalysis(
                sample["rr_ms"],
                sampling_rate=4.0,
                detrend_method=sample["detrend_method"],
                detrend_lambda=sample.get("detrend_lambda", 500.0),
                window_type="hann",
                segment_length=120.0,
                overlap_ratio=0.75,
                ar_order=16,
                enable_diagnostics=True,
            )
            results = analyzer.get_results()

    method_arrays = {
        "welch": (analyzer.freqs, analyzer.psd),
        "fft": (analyzer.fft_freqs, analyzer.fft_psd),
        "ar": (analyzer.ar_freqs, analyzer.ar_psd),
    }
    rows = []
    for method, (freqs, psd) in method_arrays.items():
        current = current_metrics(results, method)
        candidates = {
            "candidate_a": candidate_metrics(freqs, psd, "candidate_a"),
            "candidate_b": candidate_metrics(freqs, psd, "candidate_b"),
        }
        for candidate, values in candidates.items():
            for metric in METRICS:
                cur = current[metric]
                new = values[metric]
                abs_change = abs(new - cur) if math.isfinite(new) and math.isfinite(cur) else math.nan
                rel_change = (
                    abs_change / abs(cur) * 100.0
                    if math.isfinite(abs_change) and math.isfinite(cur) and abs(cur) > 1e-12
                    else math.nan
                )
                material = (
                    (math.isfinite(rel_change) and rel_change >= 1.0 and abs_change > 1e-6)
                    or (not math.isfinite(rel_change) and math.isfinite(abs_change) and abs_change > 1e-6)
                )
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "sample_group": sample["sample_group"],
                        "source": sample["source"],
                        "detrend_method": str(sample["detrend_method"]),
                        "duration_s": float(np.sum(sample["rr_ms"]) / 1000.0),
                        "rr_count": int(len(sample["rr_ms"])),
                        "method": method,
                        "candidate": candidate,
                        "metric": metric,
                        "current_value": cur,
                        "candidate_value": new,
                        "absolute_change": abs_change,
                        "relative_change_pct": rel_change,
                        "materially_affected": material,
                    }
                )
    return rows


def build_samples() -> list[dict]:
    samples = []

    cleaned = pd.read_csv(CLEANED_KUBIOS)
    normal_sources = (
        cleaned[cleaned["category"].eq("clean_high_agreement")]["source_file"]
        .drop_duplicates()
        .head(8)
        .tolist()
    )
    for source in normal_sources:
        rr = load_rr_ms(ROOT / source)
        samples.append(
            {
                "sample_id": Path(source).stem,
                "sample_group": "normal_physionet_10min_linear",
                "source": source,
                "rr_ms": rr,
                "detrend_method": "linear",
            }
        )

    if KUBIOS_MATCHED_PER.exists():
        per = pd.read_csv(KUBIOS_MATCHED_PER)
        unique = per[["pilot_id", "kubios_mat_path"]].drop_duplicates()
        for _, row in unique.iterrows():
            mat_path = ROOT / row["kubios_mat_path"]
            samples.append(
                {
                    "sample_id": row["pilot_id"],
                    "sample_group": "kubios_44_selected_none",
                    "source": row["kubios_mat_path"],
                    "rr_ms": selected_rr_from_mat(mat_path),
                    "detrend_method": None,
                }
            )

    if SMOOTHNESS_MANIFEST.exists():
        manifest = pd.read_csv(SMOOTHNESS_MANIFEST)
        for _, row in manifest.iterrows():
            rr = load_rr_ms(ROOT / row["source_rr_file"])
            samples.append(
                {
                    "sample_id": row["pilot_id"],
                    "sample_group": "smoothness_priors_10",
                    "source": row["source_rr_file"],
                    "rr_ms": rr,
                    "detrend_method": "smoothness_priors",
                    "detrend_lambda": 500.0,
                }
            )

    dc_paths = [
        "validation/kubios_subset/input_ascii_rr/clean_high_agreement/CH001__nsr020_segment_000.txt",
        "validation/kubios_subset/input_ascii_rr/remaining_outliers/OUT001__nsr009_segment_012.txt",
        "validation/kubios_subset/input_ascii_rr/remaining_outliers/OUT006__nsr015_segment_089.txt",
        "validation/kubios_subset/input_ascii_rr/vlf_sensitive/VLF001__nsr016_segment_054.txt",
        "validation/kubios_subset/input_ascii_rr/vlf_sensitive/VLF002__nsr007_segment_062.txt",
        "validation/kubios_subset/input_ascii_rr/vlf_sensitive/VLF005__nsr008_segment_046.txt",
        "validation/kubios_subset/input_ascii_rr/random_controls/RC003__nsr002_segment_069.txt",
    ]
    for source in dc_paths:
        path = ROOT / source
        if path.exists():
            samples.append(
                {
                    "sample_id": Path(source).stem,
                    "sample_group": "dc_vlf_sensitive_none",
                    "source": source,
                    "rr_ms": load_rr_ms(path),
                    "detrend_method": None,
                }
            )

    if normal_sources:
        rr = load_rr_ms(ROOT / normal_sources[0])
        for duration in [30, 60, 120, 300, 600]:
            samples.append(
                {
                    "sample_id": f"{Path(normal_sources[0]).stem}_{duration}s",
                    "sample_group": "duration_truncated_linear",
                    "source": normal_sources[0],
                    "rr_ms": truncate_rr(rr, duration),
                    "detrend_method": "linear",
                }
            )
    return samples


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], digits: int = 3) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())

    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"
        return str(value)

    body = [[fmt(row.get(header, "")) for header in headers] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in body))
        for index, header in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in body
    )
    return "\n".join(lines)


def summarize_impact(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for candidate in ["candidate_a", "candidate_b"]:
        for metric in METRICS:
            subset = frame[(frame["candidate"] == candidate) & (frame["metric"] == metric)]
            rel = pd.to_numeric(subset["relative_change_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            abs_change = pd.to_numeric(subset["absolute_change"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            rows.append(
                {
                    "candidate": candidate,
                    "metric": metric,
                    "n": int(len(subset)),
                    "median_abs_change": float(abs_change.median(skipna=True)),
                    "median_relative_change_pct": float(rel.median(skipna=True)),
                    "max_abs_change": float(abs_change.max(skipna=True)),
                    "max_relative_change_pct": float(rel.max(skipna=True)),
                    "materially_affected_n": int(subset["materially_affected"].sum()),
                    "materially_affected_pct": float(subset["materially_affected"].mean() * 100.0),
                }
            )
    return rows


def kubios_psd_convention_check() -> tuple[list[dict], dict[str, float]]:
    if not KUBIOS_AUDIT.exists():
        return [], {}
    audit = pd.read_csv(KUBIOS_AUDIT)
    rows = []
    for _, row in audit.iterrows():
        mat_path_text = None
        if KUBIOS_MATCHED_PER.exists():
            per = pd.read_csv(KUBIOS_MATCHED_PER)
            match = per[per["pilot_id"].eq(row["pilot_id"])]
            if not match.empty:
                mat_path_text = match.iloc[0]["kubios_mat_path"]
        if not mat_path_text:
            continue
        mat_path = ROOT / mat_path_text
        res = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["Res"]
        welch = res.HRV.Frequency.Welch
        freqs = np.asarray(welch.F).squeeze().astype(float)
        psd = np.asarray(welch.PSD).squeeze().astype(float) * 1_000_000.0
        for metric, field in KUBIOS_MAT_POWER_FIELDS.items():
            stored = float(np.asarray(getattr(welch, field)).squeeze()) * 1_000_000.0
            if metric == "VLF":
                inclusive = integrate(freqs, psd, (freqs >= 0.0) & (freqs <= 0.04))
                no_dc = integrate(freqs, psd, (freqs > 0.0) & (freqs <= 0.04))
                non_overlap = integrate(freqs, psd, (freqs >= 0.003) & (freqs < 0.04))
            elif metric == "LF":
                inclusive = integrate(freqs, psd, (freqs >= 0.04) & (freqs <= 0.15))
                no_dc = inclusive
                non_overlap = integrate(freqs, psd, (freqs >= 0.04) & (freqs < 0.15))
            elif metric == "HF":
                inclusive = integrate(freqs, psd, (freqs >= 0.15) & (freqs <= 0.40))
                no_dc = inclusive
                non_overlap = inclusive
            else:
                inclusive = integrate(freqs, psd, (freqs >= 0.0) & (freqs <= 0.40))
                no_dc = integrate(freqs, psd, (freqs > 0.0) & (freqs <= 0.40))
                non_overlap = inclusive
            rows.append(
                {
                    "pilot_id": row["pilot_id"],
                    "metric": metric,
                    "kubios_stored": stored,
                    "integrated_inclusive": inclusive,
                    "abs_diff_inclusive": abs(inclusive - stored),
                    "integrated_no_dc": no_dc,
                    "abs_diff_no_dc": abs(no_dc - stored),
                    "integrated_non_overlap": non_overlap,
                    "abs_diff_non_overlap": abs(non_overlap - stored),
                }
            )
    if not rows:
        return rows, {}
    frame = pd.DataFrame(rows)
    finite_stored = frame["kubios_stored"].abs() > 1e-12
    frame["rel_diff_inclusive_pct"] = np.where(
        finite_stored, frame["abs_diff_inclusive"] / frame["kubios_stored"].abs() * 100.0, np.nan
    )
    frame["rel_diff_no_dc_pct"] = np.where(
        finite_stored, frame["abs_diff_no_dc"] / frame["kubios_stored"].abs() * 100.0, np.nan
    )
    frame["rel_diff_non_overlap_pct"] = np.where(
        finite_stored, frame["abs_diff_non_overlap"] / frame["kubios_stored"].abs() * 100.0, np.nan
    )
    summary = {
        "kubios_rows": float(len(frame)),
        "median_abs_diff_inclusive": float(frame["abs_diff_inclusive"].median()),
        "max_abs_diff_inclusive": float(frame["abs_diff_inclusive"].max()),
        "median_rel_diff_inclusive_vlf": float(
            frame[frame["metric"].eq("VLF")]["rel_diff_inclusive_pct"].median()
        ),
        "median_rel_diff_inclusive_total": float(
            frame[frame["metric"].eq("total_power")]["rel_diff_inclusive_pct"].median()
        ),
        "median_rel_diff_non_overlap_vlf": float(
            frame[frame["metric"].eq("VLF")]["rel_diff_non_overlap_pct"].median()
        ),
        "median_rel_diff_no_dc_total": float(
            frame[frame["metric"].eq("total_power")]["rel_diff_no_dc_pct"].median()
        ),
        "median_abs_diff_no_dc_vlf_total": float(
            frame[frame["metric"].isin(["VLF", "total_power"])]["abs_diff_no_dc"].median()
        ),
        "median_abs_diff_non_overlap_vlf": float(
            frame[frame["metric"].eq("VLF")]["abs_diff_non_overlap"].median()
        ),
    }
    return rows, summary


def main() -> int:
    samples = build_samples()
    impact_rows = []
    for sample in samples:
        try:
            impact_rows.extend(analyze_sample(sample))
        except Exception as exc:
            impact_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "sample_group": sample["sample_group"],
                    "source": sample["source"],
                    "detrend_method": str(sample["detrend_method"]),
                    "error": str(exc),
                }
            )
    write_csv(IMPACT_CSV, impact_rows)
    impact = pd.DataFrame([row for row in impact_rows if "metric" in row])
    impact_summary = summarize_impact(impact)
    kubios_rows, kubios_summary = kubios_psd_convention_check()
    write_csv(KUBIOS_CONVENTION_CSV, kubios_rows)

    sample_counts = (
        impact[["sample_id", "sample_group"]]
        .drop_duplicates()
        .groupby("sample_group")["sample_id"]
        .count()
        .reset_index(name="n_samples")
        .to_dict("records")
    )
    key_metrics = [
        row
        for row in impact_summary
        if row["metric"]
        in [
            "ulf_power",
            "vlf_power",
            "lf_power",
            "hf_power",
            "total_power",
            "lf_hf_ratio",
            "lf_nu",
            "hf_nu",
            "vlf_power_nu",
            "lf_power_nu",
            "hf_power_nu",
        ]
    ]
    material_by_group = (
        impact.groupby(["candidate", "sample_group", "metric"], dropna=False)["materially_affected"]
        .mean()
        .reset_index()
    )
    no_detrend_vlf = material_by_group[
        material_by_group["sample_group"].str.contains("none", na=False)
        & material_by_group["metric"].isin(["vlf_power", "total_power"])
    ]
    no_detrend_note = (
        no_detrend_vlf.to_dict("records")[:12] if not no_detrend_vlf.empty else []
    )

    report = f"""# Frequency Band Definition Audit

## Executive Summary

This is a diagnostic-only audit. Production code, manuscript files, and existing validation outputs were not modified.

Current HRV Studio code defines overlapping low-frequency bands: ULF is `0.0-0.003 Hz`, VLF is `0.0-0.04 Hz`, LF is `0.04-0.15 Hz`, HF is `0.15-0.40 Hz`, and total power is `0.0-0.40 Hz`. Band masks are inclusive at both ends in the production spectral metric function, and integration uses `np.trapezoid`.

The main empirical effect of non-overlapping candidate definitions is concentrated in ULF, VLF, VLF percent-of-total, and total power when the DC bin is excluded. LF/HF, LFnu, and HFnu can also change when exact boundary bins such as `0.15 Hz` are present, because production LF currently includes the `0.15 Hz` bin and HF also includes it.

## Task 1: Confirmed Implementation

Authoritative implementation: `hrvlib/metrics/freq_domain.py`.

- Band definitions: `HRVFreqDomainAnalysis.DEFAULT_FREQ_BANDS`.
- PSD generation: `_compute_welch_psd()`, `_compute_fft_psd()`, `_compute_ar_psd()`.
- Spectral metric integration: `_compute_spectral_metrics(use_ar=False, use_fft=False)`.
- Diagnostics use the same band definitions in `_compute_band_diagnostics()`.

Current masks in `_compute_spectral_metrics()`:

- ULF: `(freqs >= 0.0) & (freqs <= 0.003)`.
- VLF: `(freqs >= 0.0) & (freqs <= 0.04)`.
- LF: `(freqs >= 0.04) & (freqs <= 0.15)`.
- HF: `(freqs >= 0.15) & (freqs <= 0.4)`.
- Total: `(freqs >= 0.0) & (freqs <= 0.4)`, implemented as the lower bound of VLF through the upper bound of HF.

Integration method: `np.trapezoid(psd[mask], freqs[mask])`, clamped to non-negative for individual bands. The 0-Hz/DC bin is included in ULF, VLF, and total power when present. Exact boundary bins are included by both adjacent band masks. The exact boundary point itself has no finite width, but it participates as an endpoint in trapezoids on both sides and is double-counted in bin counts and any simple sum of band powers.

Welch, FFT, and AR all use `_compute_spectral_metrics()`:

- Welch: `self.spectral_metrics = self._compute_spectral_metrics()`.
- FFT: `self.fft_spectral_metrics = self._compute_spectral_metrics(use_fft=True)`.
- AR: `self.ar_spectral_metrics = self._compute_spectral_metrics(use_ar=True)`.

## Task 2: Derived Metrics

Production formulas from `_compute_spectral_metrics()`:

- `ulf_power`, `vlf_power`, `lf_power`, `hf_power`: trapezoidal band integration over the masks above.
- `total_power`: trapezoidal integration over `0.0 <= f <= 0.4`.
- `ulf_power_nu`, `vlf_power_nu`, `lf_power_nu`, `hf_power_nu`: each band power divided by `total_power` and multiplied by 100. These are percent-of-total outputs, despite the `_nu` suffix.
- `lf_hf_ratio`: `lf_power / hf_power`.
- `lf_nu`, `hf_nu`: `LF/(LF+HF)*100` and `HF/(LF+HF)*100`.
- `relative_lf_power`, `relative_hf_power`: same formulas as `lf_nu` and `hf_nu`.
- `peak_freq_vlf`, `peak_freq_lf`, `peak_freq_hf`: peak PSD location inside the same inclusive band masks.

Metrics expected to change under non-overlapping Candidate A:

- Always definition-affected: `ulf_power`, `vlf_power`, `ulf_power_nu`, `vlf_power_nu`, `peak_freq_vlf`.
- Potentially affected by exact boundary bins: `lf_power`, `lf_power_nu`, `lf_hf_ratio`, `lf_nu`, `hf_nu`, `relative_lf_power`, `relative_hf_power`, `peak_freq_lf`.
- Usually unaffected unless boundary bins exist: `hf_power`, `hf_power_nu`.
- `total_power` is unchanged in Candidate A as defined here because it remains direct integration over `0.0 <= f <= 0.4`.

Metrics expected to change under Candidate B:

- All Candidate A changes.
- `total_power` and all percent-of-total metrics can change because Candidate B integrates total power over `0.0 < f <= 0.4`, excluding the DC/first segment contribution.

## Task 3: Project Documentation and Intended Convention

Repository evidence does not show an intentional cumulative VLF definition beyond documenting current implementation:

- `validation/research_notes/final_master_validation_report/hrv_studio_system_description.md` documents ULF `0.0-0.003` and VLF `0.0-0.04`, and states total power is `0.0-0.4`.
- Manual inspection reports explicitly warn: `ULF and VLF currently overlap; definitions are reported unchanged.`
- `validation/README_validation.md` and multiple research notes identify DC / first VLF bins as convention-sensitive for VLF and total power.
- `tests/test_freq_domain.py` contains tests expecting the current result structure and default bands, including ULF and VLF outputs. It does not establish a physiological rationale for cumulative VLF.
- Local validation scripts for NeuroKit2 and duration sensitivity import `HRVFreqDomainAnalysis.DEFAULT_FREQ_BANDS`, so they mirror the production overlap rather than serving as independent convention references.

The project convention currently documented is therefore descriptive of implementation, not clear evidence that overlapping ULF/VLF was intentionally chosen as the target physiological convention.

## Task 4: Isolated Candidate Definitions

Candidate A:

- ULF: `0 <= f < 0.003`.
- VLF: `0.003 <= f < 0.04`.
- LF: `0.04 <= f < 0.15`.
- HF: `0.15 <= f <= 0.40`.
- Total: direct trapezoidal integration over `0 <= f <= 0.40`; this preserves current total-power convention while removing low-band overlap.

Candidate B:

- ULF: `0 < f < 0.003`.
- VLF/LF/HF: same as Candidate A.
- Total: direct trapezoidal integration over `0 < f <= 0.40`; this excludes the exact DC bin from physiological power calculations.

Candidate calculations are isolated in `tools/frequency_band_definition_audit.py` and do not modify production behavior.

## Task 5: Targeted Empirical Impact

Sample composition:

{md_table(sample_counts, digits=0)}

Impact summary across Welch, FFT, and AR PSDs:

{md_table(key_metrics, digits=3)}

No-detrend material-effect fractions for VLF/total-power groups:

{md_table(no_detrend_note, digits=3)}

Detailed row-level impact is saved in `{rel(IMPACT_CSV)}`.

Interpretation of observed changes:

- Candidate A materially changes VLF-related outputs because VLF no longer includes the DC/ULF region.
- Candidate A leaves direct total power unchanged by definition.
- Candidate B materially changes total power in records where the DC-to-first-positive-frequency trapezoid contributes appreciable area.
- LF/HF, LFnu, and HFnu are not affected by VLF removal, but they can change when exact LF/HF boundary bins are present, especially the `0.15 Hz` Welch bin.
- No-detrend and short-duration/VLF-sensitive cases are the most likely to show large VLF or total-power changes.

## Task 6: Manuscript Experiment Impact if Corrected

If production adopts non-overlapping bands and/or DC exclusion, affected experiments:

| Experiment | Classification | Reason |
| ---------- | -------------- | ------ |
| Section 3.1 large-scale NeuroKit2 agreement | must rerun | Frequency metrics and the local NeuroKit2 comparator integration use the current overlapping bands; VLF/total and boundary-sensitive LF metrics would change. |
| Section 3.2 sequence-harmonized Kubios benchmark | must rerun | The benchmark reports VLF, LF, HF, total power, LF/HF, LFnu, HFnu; corrected HRV Studio values and comparator-definition interpretation must be regenerated. |
| Section 3.3 Smoothness Priors sensitivity | must rerun if manuscript reports frequency values after code correction | FFT and AR summaries use the shared integration function; AR/FFT VLF, total, and boundary-sensitive LF-derived metrics could change. |
| Alternative spectral-method comparison | must rerun | It directly compares Welch/FFT/AR frequency metrics from the shared integration function. |
| Synthetic robustness testing | should rerun | Frequency-domain robustness metrics and figures include LF, HF, total power, LF/HF, and related values. |
| Duration sensitivity analysis | must rerun | The script imports current band definitions and VLF/duration conclusions depend on low-frequency integration. |
| MIT-BIH engineering/QC stress test | should rerun | Any frequency-domain QC/stress outputs would change; time-domain-only results are unaffected. |
| Figures/tables containing frequency-domain results | must regenerate | Figures 3-8/source tables with frequency-domain metrics, final validation tables, Kubios Figure 4, duration Figure 7, spectral-method Figure 5, and any VLF/total-power tables need review/regeneration. |

Time-domain and nonlinear-only validations are unaffected.

## Task 7: Comparator Conventions from Local Evidence

NeuroKit2 local validation:

- The project does not primarily use NeuroKit2's own `hrv_frequency()` band outputs in the large-scale comparator scripts.
- `tools/validate_freq_domain_neurokit2.py`, `tools/duration_sensitivity_validation.py`, and `tools/validate_fft_ar_methods.py` compute NeuroKit2 PSDs and then integrate them with `HRVFreqDomainAnalysis.DEFAULT_FREQ_BANDS`.
- Therefore local NeuroKit2 agreement numbers use the same overlapping VLF definition and inclusive masks as HRV Studio. They cannot independently validate the correctness of the band definitions.

Kubios local exports:

- Kubios text exports visibly report bands as VLF `0-0.04 Hz`, LF `0.04-0.15 Hz`, HF `0.15-0.4 Hz`; no ULF output is exposed in these exports.
- The MAT PSD cross-check supports inclusive VLF/total more strongly than DC-excluded or non-overlapping alternatives for Kubios exports. Median relative difference for inclusive VLF was `{kubios_summary.get('median_rel_diff_inclusive_vlf', math.nan):.3f}%`; median relative difference for inclusive total power was `{kubios_summary.get('median_rel_diff_inclusive_total', math.nan):.3f}%`.
- Excluding DC or using non-overlapping VLF did not reproduce Kubios VLF/total as closely in the checked exports: median relative difference for non-overlapping VLF was `{kubios_summary.get('median_rel_diff_non_overlap_vlf', math.nan):.3f}%`, and median relative difference for DC-excluded total power was `{kubios_summary.get('median_rel_diff_no_dc_total', math.nan):.3f}%`.
- The exported PSD arrays did not exactly reproduce every stored Kubios band power with simple trapezoidal masks, especially LF in some records, so this is evidence about the low-boundary/DC convention rather than proof of Kubios' complete internal integration implementation.

Kubios evidence therefore supports that the exported Kubios VLF power corresponds to `0-0.04 Hz` integration of its exported PSD, including the lowest/DC endpoint as represented in that PSD. It does not establish that HRV Studio should expose overlapping ULF and VLF simultaneously.

Kubios PSD convention details are saved in `{rel(KUBIOS_CONVENTION_CSV)}`.

## Task 8: Engineering Recommendation

Current VLF definition is demonstrably overlapping with ULF and is likely incorrect if HRV Studio intends standard mutually exclusive physiological bands. It is, however, partially aligned with Kubios' exported VLF label of `0-0.04 Hz`, so comparator-specific reporting must be explicit.

Recommended production direction, pending team approval:

- Keep ULF as a separate output only for sufficiently long recordings, with duration warnings; do not include ULF inside VLF when both are reported as separate bands.
- Use non-overlapping masks:
  - ULF: `0 < f < 0.003 Hz` or `0 <= f < 0.003 Hz` only if DC is intentionally retained for diagnostic total power.
  - VLF: `0.003 <= f < 0.04 Hz`.
  - LF: `0.04 <= f < 0.15 Hz`.
  - HF: `0.15 <= f <= 0.40 Hz`.
- Exclude the exact DC bin from physiological band-power reporting unless a comparator-specific mode explicitly requires including it.
- Define total physiological power as direct integration over `0 < f <= 0.40 Hz` for short-term HRV frequency metrics, or as the sum/integral over the selected non-overlapping physiological bands. Use one convention consistently and document it.
- Preserve a comparator-specific diagnostic mode if Kubios `0-0.04` VLF reproduction remains required.

Derived metrics affected by this recommendation:

- Directly: ULF, VLF, total power, band percent-of-total metrics, VLF peak frequency.
- Potentially by boundary handling: LF, LF/HF, LFnu, HFnu, relative LF/HF power.
- Usually not affected by VLF-only correction: HF absolute power, unless boundary conventions change.

Validation rerun scope would be broad: every manuscript frequency-domain validation table/figure and comparator summary should be regenerated after any production change. Time-domain and nonlinear validations do not need rerun for this issue.

## Generated Files

- `{rel(REPORT)}`
- `{rel(IMPACT_CSV)}`
- `{rel(KUBIOS_CONVENTION_CSV)}`

## Commands Used

```powershell
python tools/frequency_band_definition_audit.py
```
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Samples analyzed: {impact[['sample_id','sample_group']].drop_duplicates().shape[0]}")
    print(f"Impact rows: {len(impact)}")
    print(f"Report: {rel(REPORT)}")
    print(f"Impact CSV: {rel(IMPACT_CSV)}")
    print(f"Kubios convention CSV: {rel(KUBIOS_CONVENTION_CSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
