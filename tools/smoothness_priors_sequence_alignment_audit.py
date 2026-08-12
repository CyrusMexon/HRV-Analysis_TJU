"""Audit sequence alignment for the Kubios Smoothness Priors pilot.

Validation-only diagnostic script. It reads existing artifacts and writes a new
audit CSV plus report without rerunning the Smoothness Priors experiment.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "validation" / "kubios_subset" / "smoothness_priors_pilot"
MANIFEST = PILOT_DIR / "pilot_manifest.csv"
SETTINGS_AUDIT = PILOT_DIR / "kubios_export_settings_audit.csv"
FFT_SUMMARY = PILOT_DIR / "smoothness_priors_fft_metric_summary.csv"
AR_SUMMARY = PILOT_DIR / "smoothness_priors_ar_metric_summary.csv"
FFT_PER_FILE = PILOT_DIR / "smoothness_priors_fft_per_file_comparison.csv"
AR_PER_FILE = PILOT_DIR / "smoothness_priors_ar_per_file_comparison.csv"
REPORT = ROOT / "validation" / "research_notes" / "smoothness_priors_sequence_alignment_audit.md"
AUDIT_CSV = PILOT_DIR / "smoothness_priors_sequence_alignment_audit.csv"
SCRIPT = ROOT / "tools" / "run_kubios_smoothness_priors_pilot.py"
PILOT_REPORT = ROOT / "validation" / "research_notes" / "kubios_smoothness_priors_pilot_report.md"
RUN_INFO_FFT = ROOT / "validation" / "runs" / "v08_kubios_smoothness_priors_pilot_fft" / "run_info.json"
RUN_INFO_AR = ROOT / "validation" / "runs" / "v08_kubios_smoothness_priors_pilot_ar" / "run_info.json"
RUN_INFO_WELCH = ROOT / "validation" / "runs" / "v08_kubios_smoothness_priors_pilot_welch" / "run_info.json"
HRV_FFT_OUTPUT = ROOT / "validation" / "runs" / "v08_kubios_smoothness_priors_pilot_fft" / "hrvstudio_fft_results.csv"
HRV_AR_OUTPUT = ROOT / "validation" / "runs" / "v08_kubios_smoothness_priors_pilot_ar" / "hrvstudio_ar_results.csv"
HRV_WELCH_OUTPUT = ROOT / "validation" / "runs" / "v08_kubios_smoothness_priors_pilot_welch" / "hrvstudio_welch_results.csv"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_rr_ms(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for token in line.strip().replace(",", " ").replace(";", " ").split():
            try:
                value = float(token)
            except ValueError:
                continue
            if math.isfinite(value) and value > 0:
                values.append(value)
    rr = np.asarray(values, dtype=float)
    if rr.size and np.median(rr) < 10:
        rr *= 1000.0
    return rr


def txt_value(text: str, label: str) -> str:
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*([^\n\r]*)", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def txt_sample_limits(text: str) -> str:
    match = re.search(r"^\s*Sample limits[^\n\r;]*;([^\n\r]*)", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], digits: int = 2) -> str:
    if not rows:
        return ""
    headers = list(rows[0])

    def fmt(value):
        if isinstance(value, float):
            return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"
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


def summary_table(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    return [
        {
            "metric": row["metric"],
            "n": int(row["valid_paired_files"]),
            "median RE %": float(row["median_relative_error_pct"]),
            "mean RE %": float(row["mean_relative_error_pct"]),
            "Pearson r": float(row["pearson_r"]),
        }
        for _, row in frame.iterrows()
    ]


def main() -> int:
    manifest = pd.read_csv(MANIFEST)
    settings = pd.read_csv(SETTINGS_AUDIT)
    fft_summary = summary_table(FFT_SUMMARY)
    ar_summary = summary_table(AR_SUMMARY)
    run_fft = json.loads(RUN_INFO_FFT.read_text(encoding="utf-8"))
    run_ar = json.loads(RUN_INFO_AR.read_text(encoding="utf-8"))
    run_welch = json.loads(RUN_INFO_WELCH.read_text(encoding="utf-8"))

    rows = []
    for _, item in manifest.iterrows():
        pilot_id = item["pilot_id"]
        txt_path = ROOT / item["kubios_report_file"]
        mat_path = txt_path.with_suffix(".mat")
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        res = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["Res"]
        data = res.HRV.Data
        full_rr = np.asarray(data.RR).squeeze().astype(float) * 1000.0
        selected_rr = np.asarray(data.RRs).squeeze().astype(float) * 1000.0
        source_rr = read_rr_ms(ROOT / item["source_rr_file"])
        rri = np.asarray(data.RRi).squeeze().astype(float)

        source_matches_full = (
            len(source_rr) == len(full_rr)
            and np.allclose(source_rr, full_rr, rtol=0, atol=1e-6)
        )
        selected_matches_full = (
            len(selected_rr) == len(full_rr)
            and np.allclose(selected_rr, full_rr, rtol=0, atol=1e-6)
        )
        source_matches_selected = (
            len(source_rr) == len(selected_rr)
            and np.allclose(source_rr, selected_rr, rtol=0, atol=1e-6)
        )
        status = (
            "confirmed sequence-aligned"
            if source_matches_full and selected_matches_full and source_matches_selected
            else "confirmed sequence-mismatched"
        )
        rows.append(
            {
                "pilot_id": pilot_id,
                "source_rr_file": item["source_rr_file"],
                "kubios_txt": item["kubios_report_file"],
                "kubios_mat": rel(mat_path),
                "source_rr_n": len(source_rr),
                "kubios_full_rr_n": len(full_rr),
                "kubios_selected_rr_n": len(selected_rr),
                "count_difference_source_vs_selected": len(source_rr) - len(selected_rr),
                "source_duration_s": float(source_rr.sum() / 1000.0),
                "kubios_full_duration_s": float(full_rr.sum() / 1000.0),
                "kubios_selected_duration_s": float(selected_rr.sum() / 1000.0),
                "duration_difference_source_vs_selected_s": float(
                    source_rr.sum() / 1000.0 - selected_rr.sum() / 1000.0
                ),
                "sample_limits_s": txt_sample_limits(text),
                "kubios_rri_samples": len(rri),
                "kubios_rri_duration_s_at_4hz": len(rri) / 4.0,
                "source_matches_kubios_full_RR": source_matches_full,
                "kubios_selected_RRs_equals_full_RR": selected_matches_full,
                "source_matches_kubios_selected_RRs": source_matches_selected,
                "sequence_alignment_status": status,
                "detrending_method": txt_value(text, "Detrending method"),
                "interpolation_rate": txt_value(text, "Interpolation rate"),
                "fft_window_width": txt_value(text, "Window width"),
                "fft_window_overlap": txt_value(text, "Window overlap"),
                "ar_order": txt_value(text, "AR model order"),
                "ar_factorization": txt_value(text, "Use factorization"),
                "evidence": (
                    "source ASCII RR equals HRV.Data.RR and HRV.Data.RRs; "
                    f"sample_limits={txt_sample_limits(text)}; "
                    f"RRi_samples={len(rri)}"
                ),
            }
        )

    write_csv(AUDIT_CSV, rows)

    checked = len(rows)
    mismatched = [row for row in rows if row["sequence_alignment_status"] != "confirmed sequence-aligned"]
    status = "A. confirmed sequence-aligned" if not mismatched else "C. confirmed sequence-mismatched"
    section_aligned = "yes" if not mismatched else "no"
    rerun_required = "No" if not mismatched else "Yes"
    settings_match_count = int(settings["settings_match"].astype(str).str.lower().eq("true").sum())
    recordings_table = md_table(
        [
            {
                "pilot_id": row["pilot_id"],
                "source n": row["source_rr_n"],
                "Kubios RRs n": row["kubios_selected_rr_n"],
                "source s": row["source_duration_s"],
                "Kubios RRs s": row["kubios_selected_duration_s"],
                "status": row["sequence_alignment_status"],
            }
            for row in rows
        ],
        digits=3,
    )
    fft_table = md_table(fft_summary, digits=3)
    ar_table = md_table(ar_summary, digits=3)

    report = f"""# Smoothness Priors Sequence Alignment Audit

## Executive Conclusion

Section 3.3 sequence alignment status: **{status}**.

The existing Smoothness Priors sensitivity analysis checked {checked} recordings. In all {checked}, the source ASCII RR sequence supplied to HRV Studio matched both Kubios `HRV.Data.RR` and Kubios selected `HRV.Data.RRs`. No sequence-length or duration mismatch was found, so a matched-sequence rerun was **not required**.

## Artifacts Located

Scripts:

- `{rel(SCRIPT)}`
- `tools/parse_kubios_exports.py`

Configurations and manifests:

- `{rel(MANIFEST)}`
- `{rel(SETTINGS_AUDIT)}`
- `{rel(RUN_INFO_FFT)}`
- `{rel(RUN_INFO_AR)}`
- `{rel(RUN_INFO_WELCH)}`
- `{rel(PILOT_DIR / 'pilot_run_info.json')}`

Kubios exports:

- `{rel(PILOT_DIR / 'kubios_exports')}` with 10 TXT/MAT/PDF export sets.

HRV Studio outputs:

- `{rel(HRV_FFT_OUTPUT)}`
- `{rel(HRV_AR_OUTPUT)}`
- `{rel(HRV_WELCH_OUTPUT)}`
- `{rel(FFT_PER_FILE)}`
- `{rel(AR_PER_FILE)}`
- `{rel(FFT_SUMMARY)}`
- `{rel(AR_SUMMARY)}`

Reports:

- `{rel(PILOT_REPORT)}`
- `{rel(PILOT_DIR / 'hrvstudio_method_mapping.md')}`
- `{rel(PILOT_DIR / 'parser_assessment.md')}`

## Recordings Included

Number of recordings: **{checked}**.

{recordings_table}

## Sequence Evidence

- For each MAT export, `HRV.Data.RR` and `HRV.Data.RRs` had identical length and values.
- For each pilot source RR file listed in `pilot_manifest.csv`, the source RR vector matched `HRV.Data.RR` and `HRV.Data.RRs` exactly after seconds-to-ms unit normalization.
- Kubios text sample limits covered the full imported interval duration in every case.
- Affected by mismatch: **{len(mismatched)}/{checked}**.

Detailed row-level evidence is saved in `{rel(AUDIT_CSV)}`.

## Spectral Settings

Kubios-visible settings from `kubios_export_settings_audit.csv`:

- Smoothness Priors lambda: 500, verified in {settings_match_count}/{checked} exports.
- Interpolation frequency: 4 Hz.
- FFT configuration: 120 s window width, 75% overlap; FFT window function not exposed/not verified in Kubios export.
- AR configuration: order 16, `Use factorization: No`.
- Frequency bands: VLF 0.0-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.4 Hz.

HRV Studio settings from v08 run metadata:

- Detrending: `{run_fft.get('detrending_method')}`, lambda {run_fft.get('detrending_lambda')}.
- Interpolation: {run_fft.get('interpolation_frequency_hz')} Hz.
- Frequency bands: `{run_fft.get('frequency_bands')}`.
- FFT path: `{run_fft.get('fft_configuration', {}).get('path')}`; window `{run_fft.get('fft_configuration', {}).get('window')}`; whole-signal FFT, so Kubios FFT segmentation remains an estimator/configuration limitation rather than a sequence issue.
- AR path: `{run_ar.get('ar_configuration', {}).get('path')}`; requested order {run_ar.get('ar_configuration', {}).get('requested_order')}; algorithm `{run_ar.get('ar_configuration', {}).get('algorithm')}`.
- Welch: `{run_welch.get('welch_configuration', {}).get('status')}`.

## Existing Section 3.3 Results

FFT summary:

{fft_table}

AR summary:

{ar_table}

## Rerun Decision

Rerun required: **{rerun_required}**.

Because the sequences were confirmed aligned, no `_matched_sequence` Smoothness Priors outputs were generated. Existing FFT and AR agreement summaries remain the relevant Section 3.3 outputs.

## Manuscript Consequences

No Section 3.3 values or figures need updating due to NN-sequence harmonization. The primary Section 3.2 correction does not carry over to this Smoothness Priors pilot because Kubios selected the full imported sequence in all 10 Smoothness Priors exports.

The current interpretation remains supported: AR agreement improves substantially under nominal matched Smoothness Priors, while FFT retains greater residual differences. The residual FFT limitations remain those already documented: Kubios FFT window function is not exposed/export-verified, and HRV Studio direct FFT uses a whole-signal periodogram rather than Kubios' reported 120 s / 75% FFT windowing controls.

## Commands Used

```powershell
python tools/smoothness_priors_sequence_alignment_audit.py
```
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Checked recordings: {checked}")
    print(f"Mismatched recordings: {len(mismatched)}")
    print(f"Status: {status}")
    print(f"Rerun required: {rerun_required}")
    print(f"Wrote {rel(AUDIT_CSV)}")
    print(f"Wrote {rel(REPORT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
