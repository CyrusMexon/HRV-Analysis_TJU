"""
Prepare a representative PhysioNet 10-minute subset for manual Kubios validation.

This is validation-only tooling. It reads processed segment CSVs and validation
outputs, then writes Kubios-readable ASCII RR text files and a manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_CSV = (
    PROJECT_ROOT
    / "validation"
    / "runs"
    / "v05_physionet_10min_segment_linear"
    / "freq_domain_neurokit2_validation.csv"
)
DEFAULT_RUN_INFO = (
    PROJECT_ROOT
    / "validation"
    / "runs"
    / "v05_physionet_10min_segment_linear"
    / "run_info.json"
)
DEFAULT_SEGMENT_DIR = PROJECT_ROOT / "validation" / "processed_data" / "physionet_nsr_rr_10min"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "validation" / "kubios_subset"

CATEGORIES = [
    "clean_high_agreement",
    "vlf_sensitive",
    "remaining_outliers",
    "short_or_adjusted",
    "random_controls",
]
METRIC_COLUMNS = {
    "VLF": "VLF relative error",
    "LF": "LF relative error",
    "HF": "HF relative error",
    "total_power": "total_power relative error",
    "LF/HF": "LF/HF relative error",
}
FORCED_OUTLIER_NAMES = [
    "nsr009_segment_012.csv",
    "nsr010_segment_004.csv",
    "nsr040_segment_014.csv",
    "nsr025_segment_136.csv",
    "nsr021_segment_071.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Kubios-compatible ASCII RR files for a manual validation subset."
    )
    parser.add_argument(
        "--validation-csv",
        default=str(DEFAULT_VALIDATION_CSV),
        help="v05 validation CSV to select from.",
    )
    parser.add_argument(
        "--segment-dir",
        default=str(DEFAULT_SEGMENT_DIR),
        help="Directory containing processed PhysioNet 10-minute RR segment CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for Kubios subset files.",
    )
    parser.add_argument(
        "--run-info",
        default=str(DEFAULT_RUN_INFO),
        help="Validation run_info.json used to identify adjusted files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-count", type=int, default=20)
    parser.add_argument("--vlf-count", type=int, default=10)
    parser.add_argument("--outlier-count", type=int, default=10)
    parser.add_argument("--short-adjusted-count", type=int, default=5)
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument(
        "--include-ms-format",
        action="store_true",
        help="Also export one-RR-per-line millisecond TXT files under input_ascii_rr_ms_format/.",
    )
    return parser.parse_args()


def finite_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def csv_cell(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    return "" if value is None else value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})


def read_validation_rows(path: Path) -> Dict[Path, Dict[str, Any]]:
    by_file: Dict[Path, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = Path(row["input_file"]).resolve()
            info = by_file.setdefault(
                source,
                {
                    "source_csv": source,
                    "duration_seconds": finite_float(row.get("native_recording_duration_s")),
                    "rr_count": int(finite_float(row.get("rr_count")) or 0),
                    "metrics": {},
                    "adjustment_reason": row.get("neurokit2_welch_adjustment_reason", ""),
                    "welch_detrend_mode": row.get("welch_detrend_mode", ""),
                },
            )
            metric = row.get("metric", "")
            if metric:
                info["metrics"][metric] = finite_float(row.get("relative_error_pct"))
            if row.get("neurokit2_welch_adjustment_reason"):
                info["adjustment_reason"] = row["neurokit2_welch_adjustment_reason"]
    for info in by_file.values():
        metric_values = [
            info["metrics"].get(metric, math.nan)
            for metric in ["VLF", "LF", "HF", "total_power", "LF/HF", "LF_nu", "HF_nu"]
        ]
        core_values = [
            info["metrics"].get(metric, math.nan)
            for metric in METRIC_COLUMNS
        ]
        info["max_relative_error_pct"] = max_finite(metric_values)
        info["core_max_relative_error_pct"] = max_finite(core_values)
    return by_file


def load_adjusted_files(run_info_path: Path) -> set[Path]:
    if not run_info_path.exists():
        return set()
    payload = json.loads(run_info_path.read_text(encoding="utf-8"))
    adjusted = set()
    for event in payload.get("file_events", []):
        if event.get("reason") == "adjusted_noverlap_for_short_signal":
            adjusted.add(Path(event["input_file"]).resolve())
    for event in payload.get("skipped_files", []):
        if event.get("reason") == "adjusted_noverlap_for_short_signal":
            adjusted.add(Path(event["input_file"]).resolve())
    return adjusted


def max_finite(values: Iterable[float]) -> float:
    cleaned = [value for value in values if math.isfinite(value)]
    return max(cleaned) if cleaned else math.nan


def load_rr_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rr_ms: List[float] = []
    cumulative_time_s: List[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "rr_ms" not in (reader.fieldnames or []):
            raise ValueError(f"Missing rr_ms column: {path}")
        for row in reader:
            rr_ms.append(float(row["rr_ms"]))
            if "cumulative_time_seconds" in row and row["cumulative_time_seconds"] != "":
                cumulative_time_s.append(float(row["cumulative_time_seconds"]))
    rr = np.asarray(rr_ms, dtype=float)
    times = np.asarray(cumulative_time_s, dtype=float)
    if times.size != rr.size:
        rr_s = rr / 1000.0
        times = np.concatenate([[0.0], np.cumsum(rr_s[:-1])])
    return rr, times


def rr_stats(source_csv: Path) -> Dict[str, float]:
    rr_ms, times = load_rr_csv(source_csv)
    duration = float(times[-1]) if len(times) else math.nan
    return {
        "duration_seconds": duration,
        "n_rr": int(len(rr_ms)),
        "mean_rr_ms": float(np.mean(rr_ms)) if len(rr_ms) else math.nan,
        "min_rr_ms": float(np.min(rr_ms)) if len(rr_ms) else math.nan,
        "max_rr_ms": float(np.max(rr_ms)) if len(rr_ms) else math.nan,
    }


def record_segment_ids(path: Path) -> Tuple[str, str]:
    match = re.match(r"(?P<record>nsr\d+)_segment_(?P<segment>\d+)\.csv$", path.name)
    if not match:
        return path.stem, ""
    return match.group("record"), match.group("segment")


def choose_clean(candidates: List[Dict[str, Any]], selected: set[Path], count: int) -> List[Dict[str, Any]]:
    pool = [item for item in candidates if item["source_csv"] not in selected]
    pool.sort(
        key=lambda item: (
            bool(item.get("adjusted")),
            abs(item.get("duration_seconds", math.nan) - 600.0),
            item.get("core_max_relative_error_pct", math.inf),
            item["source_csv"].name,
        )
    )
    return pool[:count]


def choose_vlf_sensitive(candidates: List[Dict[str, Any]], selected: set[Path], count: int) -> List[Dict[str, Any]]:
    pool = [
        item
        for item in candidates
        if item["source_csv"] not in selected
        and finite_float(item["metrics"].get("VLF")) >= 0.0
        and finite_float(item["metrics"].get("LF/HF")) <= 5.0
        and not item.get("adjusted")
    ]
    pool.sort(
        key=lambda item: (
            -item["metrics"].get("VLF", -math.inf),
            item["metrics"].get("LF/HF", math.inf),
            abs(item.get("duration_seconds", math.nan) - 600.0),
            item["source_csv"].name,
        )
    )
    return pool[:count]


def choose_outliers(candidates: List[Dict[str, Any]], selected: set[Path], count: int) -> List[Dict[str, Any]]:
    by_name = {item["source_csv"].name: item for item in candidates}
    chosen: List[Dict[str, Any]] = []
    chosen_paths: set[Path] = set()
    for name in FORCED_OUTLIER_NAMES:
        item = by_name.get(name)
        if item is not None and item["source_csv"] not in selected:
            chosen.append(item)
            chosen_paths.add(item["source_csv"])
        if len(chosen) >= count:
            return chosen

    pool = [
        item
        for item in candidates
        if item["source_csv"] not in selected
        and item["source_csv"] not in chosen_paths
    ]
    pool.sort(
        key=lambda item: (
            -item.get("max_relative_error_pct", -math.inf),
            item["source_csv"].name,
        )
    )
    for item in pool:
        if len(chosen) >= count:
            break
        chosen.append(item)
    return chosen


def choose_short_adjusted(candidates: List[Dict[str, Any]], selected: set[Path], count: int) -> List[Dict[str, Any]]:
    pool = [
        item
        for item in candidates
        if item["source_csv"] not in selected
        and (item.get("duration_seconds", math.inf) < 300.0 or item.get("adjusted"))
    ]
    pool.sort(
        key=lambda item: (
            not item.get("adjusted"),
            item.get("duration_seconds", math.inf),
            -item.get("max_relative_error_pct", -math.inf),
            item["source_csv"].name,
        )
    )
    return pool[:count]


def choose_random_controls(
    candidates: List[Dict[str, Any]],
    selected: set[Path],
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    pool = [
        item
        for item in candidates
        if item["source_csv"] not in selected
        and not item.get("adjusted")
        and item.get("duration_seconds", 0.0) >= 300.0
    ]
    pool.sort(key=lambda item: item["source_csv"].name)
    rng = random.Random(seed)
    if len(pool) <= count:
        return pool
    return rng.sample(pool, count)


def select_subset(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any], str]]:
    selected: set[Path] = set()
    output: List[Tuple[str, Dict[str, Any], str]] = []

    category_specs = [
        (
            "clean_high_agreement",
            choose_clean(candidates, selected, args.clean_count),
            "duration near 600s and low relative error across VLF/LF/HF/total_power/LF-HF",
        ),
        (
            "vlf_sensitive",
            None,
            "high VLF relative error with low LF/HF relative error",
        ),
        (
            "remaining_outliers",
            None,
            "highest remaining maximum relative error; forced known outliers included when present",
        ),
        (
            "short_or_adjusted",
            None,
            "duration <300s or validation run required adjusted Welch overlap",
        ),
        (
            "random_controls",
            None,
            f"fixed random seed {args.seed} from remaining normal files",
        ),
    ]

    for category, initial_items, reason in category_specs:
        if category == "vlf_sensitive":
            items = choose_vlf_sensitive(candidates, selected, args.vlf_count)
        elif category == "remaining_outliers":
            items = choose_outliers(candidates, selected, args.outlier_count)
        elif category == "short_or_adjusted":
            items = choose_short_adjusted(candidates, selected, args.short_adjusted_count)
        elif category == "random_controls":
            items = choose_random_controls(candidates, selected, args.random_count, args.seed)
        else:
            items = initial_items or []
        for item in items:
            if item["source_csv"] in selected:
                continue
            selected.add(item["source_csv"])
            output.append((category, item, reason))
    return output


def ensure_structure(output_dir: Path, include_ms_format: bool) -> None:
    for root in ["input_ascii_rr", "kubios_exports"]:
        for category in CATEGORIES:
            (output_dir / root / category).mkdir(parents=True, exist_ok=True)
    if include_ms_format:
        for category in CATEGORIES:
            (output_dir / "input_ascii_rr_ms_format" / category).mkdir(parents=True, exist_ok=True)
    (output_dir / "parsed_results").mkdir(parents=True, exist_ok=True)


def write_rr_seconds(path: Path, rr_ms: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for value in rr_ms:
            handle.write(f"{value / 1000.0:.3f}\n")


def write_rr_milliseconds(path: Path, rr_ms: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for value in rr_ms:
            handle.write(f"{value:.3f}\n")


def subset_id(category: str, index: int) -> str:
    prefix = {
        "clean_high_agreement": "CH",
        "vlf_sensitive": "VLF",
        "remaining_outliers": "OUT",
        "short_or_adjusted": "SA",
        "random_controls": "RC",
    }[category]
    return f"{prefix}{index:03d}"


def build_manifest(
    selected: List[Tuple[str, Dict[str, Any], str]],
    output_dir: Path,
    include_ms_format: bool,
) -> List[Dict[str, Any]]:
    category_counts: Dict[str, int] = defaultdict(int)
    rows: List[Dict[str, Any]] = []
    for category, item, reason in selected:
        category_counts[category] += 1
        sid = subset_id(category, category_counts[category])
        source_csv = Path(item["source_csv"]).resolve()
        source_name = source_csv.name
        source_stem = source_csv.stem
        rr_ms, _times = load_rr_csv(source_csv)
        stats = rr_stats(source_csv)
        record_id, segment_id = record_segment_ids(source_csv)

        kubios_input = output_dir / "input_ascii_rr" / category / f"{sid}__{source_stem}.txt"
        write_rr_seconds(kubios_input, rr_ms)
        if include_ms_format:
            ms_path = output_dir / "input_ascii_rr_ms_format" / category / f"{sid}__{source_stem}.txt"
            write_rr_milliseconds(ms_path, rr_ms)

        export_dir = output_dir / "kubios_exports" / category / f"{sid}__{source_stem}"
        export_dir.mkdir(parents=True, exist_ok=True)

        row = {
            "subset_id": sid,
            "category": category,
            "source_csv": safe_rel(source_csv),
            "kubios_input_txt": safe_rel(kubios_input),
            "record_id": record_id,
            "segment_id": segment_id,
            "duration_seconds": stats["duration_seconds"],
            "n_rr": stats["n_rr"],
            "mean_rr_ms": stats["mean_rr_ms"],
            "min_rr_ms": stats["min_rr_ms"],
            "max_rr_ms": stats["max_rr_ms"],
            "v05 max_relative_error_pct": item.get("max_relative_error_pct", math.nan),
            "VLF relative error": item["metrics"].get("VLF", math.nan),
            "LF relative error": item["metrics"].get("LF", math.nan),
            "HF relative error": item["metrics"].get("HF", math.nan),
            "total_power relative error": item["metrics"].get("total_power", math.nan),
            "LF/HF relative error": item["metrics"].get("LF/HF", math.nan),
            "reason_selected": reason,
            "manual_kubios_status": "",
            "kubios_export_txt_path": safe_rel(export_dir / f"{sid}__{source_stem}_hrv.txt"),
            "kubios_export_pdf_path": safe_rel(export_dir / f"{sid}__{source_stem}_hrv.pdf"),
            "kubios_export_mat_path": safe_rel(export_dir / f"{sid}__{source_stem}_hrv.mat"),
            "notes": "",
        }
        rows.append(row)
    return rows


def write_readme(output_dir: Path, manifest_rows: Sequence[Dict[str, Any]], include_ms_format: bool) -> None:
    counts = defaultdict(int)
    for row in manifest_rows:
        counts[row["category"]] += 1
    example = manifest_rows[0] if manifest_rows else None
    lines = [
        "# Kubios Manual Validation Subset",
        "",
        "This folder contains a representative subset of PhysioNet 10-minute RR segments exported for manual Kubios validation.",
        "",
        "## Why ASCII RR TXT files are provided",
        "",
        "Kubios HRV 2.2 cannot directly open the processed validation CSV segment files used by this project. It can open simple Syl_Vain-style ASCII RR interval text files. The files under `input_ascii_rr/` therefore contain one RR interval per line with no header, in seconds.",
        "",
        "Example:",
        "",
        "```text",
        "0.827",
        "0.798",
        "0.801",
        "```",
        "",
        "The processed PhysioNet CSV files and raw PhysioNet data are not modified.",
        "",
        "## Subset counts",
        "",
        "| Category | Files |",
        "| --- | ---: |",
    ]
    for category in CATEGORIES:
        lines.append(f"| `{category}` | {counts[category]} |")

    lines.extend(
        [
            "",
            "## How to open files manually in Kubios",
            "",
            "1. Open Kubios HRV 2.2.",
            "2. Open an ASCII RR file from `input_ascii_rr/<category>/`.",
            "3. Confirm that RR interval values are interpreted as seconds.",
            "4. Use settings as close as possible to the current validation comparison.",
            "5. Save Kubios exports into the pre-created folder shown in `subset_manifest.csv`.",
            "",
            "## Recommended Kubios settings",
            "",
            "Use these as a starting point for manual comparison:",
            "",
            "- Detrending: none initially if matching Kubios default export behavior.",
            "- Frequency bands: VLF `0-0.04 Hz`, LF `0.04-0.15 Hz`, HF `0.15-0.4 Hz`.",
            "- Interpolation rate: `4 Hz`.",
            "- Spectrum: FFT/Welch spectrum.",
            "",
            "These settings do not establish Kubios as ground truth. They are intended to make the manual subset interpretable alongside HRV Studio and NeuroKit2 validation outputs.",
            "",
            "## Export saving rule",
            "",
            "For each analyzed file, save Kubios outputs into:",
            "",
            "`validation/kubios_subset/kubios_exports/<category>/<subset_id>__<source_name>/`",
            "",
            "Use these exact filenames inside that folder:",
            "",
            "- `<subset_id>__<source_name>_hrv.txt`",
            "- `<subset_id>__<source_name>_hrv.pdf`",
            "- `<subset_id>__<source_name>_hrv.mat`",
            "",
            "The empty per-file folders have already been created.",
            "",
            "## Later parsing",
            "",
            "Parsing is intentionally not implemented here. Placeholder files are prepared under `parsed_results/`:",
            "",
            "- `kubios_parsed_results.csv`",
            "- `comparison_with_hrvstudio.csv`",
            "",
        ]
    )
    if include_ms_format:
        lines.extend(
            [
                "## Alternative millisecond format",
                "",
                "This run also wrote `input_ascii_rr_ms_format/`, with one RR interval per line in milliseconds. Use it only if Kubios does not interpret the primary seconds-format files correctly.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Alternative millisecond format",
                "",
                "No millisecond-format copy was generated in this run. Re-run `tools/prepare_kubios_subset.py --include-ms-format` only if Kubios does not interpret the primary seconds-format files correctly.",
                "",
            ]
        )
    if example:
        lines.extend(
            [
                "## First file to try",
                "",
                f"- Kubios input: `{example['kubios_input_txt']}`",
                f"- Save exports to the folder containing `{example['kubios_export_txt_path']}`.",
                "",
            ]
        )
    (output_dir / "README_kubios_subset.md").write_text("\n".join(lines), encoding="utf-8")


def write_placeholder_parsed_results(output_dir: Path) -> None:
    parsed_dir = output_dir / "parsed_results"
    write_csv(
        parsed_dir / "kubios_parsed_results.csv",
        [],
        [
            "subset_id",
            "source_name",
            "kubios_metric",
            "kubios_value",
            "units",
            "parser_notes",
        ],
    )
    write_csv(
        parsed_dir / "comparison_with_hrvstudio.csv",
        [],
        [
            "subset_id",
            "metric",
            "hrvstudio_value",
            "kubios_value",
            "absolute_difference",
            "relative_difference_pct",
            "notes",
        ],
    )


def main() -> int:
    args = parse_args()
    validation_csv = Path(args.validation_csv)
    segment_dir = Path(args.segment_dir)
    output_dir = Path(args.output_dir)
    run_info = Path(args.run_info)

    if not validation_csv.exists():
        raise SystemExit(f"Validation CSV does not exist: {validation_csv}")
    if not segment_dir.exists():
        raise SystemExit(f"Segment directory does not exist: {segment_dir}")

    ensure_structure(output_dir, args.include_ms_format)
    adjusted_files = load_adjusted_files(run_info)
    candidates = list(read_validation_rows(validation_csv).values())
    candidates = [
        item
        for item in candidates
        if (segment_dir / item["source_csv"].name).exists()
    ]
    for item in candidates:
        item["source_csv"] = (segment_dir / item["source_csv"].name).resolve()
        item["adjusted"] = item["source_csv"] in adjusted_files or bool(item.get("adjustment_reason"))

    selected = select_subset(candidates, args)
    manifest_rows = build_manifest(selected, output_dir, args.include_ms_format)
    manifest_path = output_dir / "subset_manifest.csv"
    fieldnames = [
        "subset_id",
        "category",
        "source_csv",
        "kubios_input_txt",
        "record_id",
        "segment_id",
        "duration_seconds",
        "n_rr",
        "mean_rr_ms",
        "min_rr_ms",
        "max_rr_ms",
        "v05 max_relative_error_pct",
        "VLF relative error",
        "LF relative error",
        "HF relative error",
        "total_power relative error",
        "LF/HF relative error",
        "reason_selected",
        "manual_kubios_status",
        "kubios_export_txt_path",
        "kubios_export_pdf_path",
        "kubios_export_mat_path",
        "notes",
    ]
    write_csv(manifest_path, manifest_rows, fieldnames)
    write_readme(output_dir, manifest_rows, args.include_ms_format)
    write_placeholder_parsed_results(output_dir)

    counts = defaultdict(int)
    for row in manifest_rows:
        counts[row["category"]] += 1
    print(f"Total selected files: {len(manifest_rows)}")
    for category in CATEGORIES:
        print(f"{category}: {counts[category]}")
    print(f"Manifest: {manifest_path}")
    if manifest_rows:
        example = manifest_rows[0]
        print(f"Example Kubios file to open: {example['kubios_input_txt']}")
        print(
            "Save its Kubios exports in: "
            f"{Path(example['kubios_export_txt_path']).parent}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
