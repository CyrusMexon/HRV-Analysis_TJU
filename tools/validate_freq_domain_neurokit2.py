"""
Compare HRV Studio native frequency-domain metrics against NeuroKit2.

This is a validation utility only. It does not change HRV Studio analysis code.

Example:
    python tools/validate_freq_domain_neurokit2.py data/sample_rr.csv --run-name v02_with_diagnostics --enable-diagnostics

To match the current pipeline default detrending:
    python tools/validate_freq_domain_neurokit2.py data/sample_rr.csv --run-name v02_linear_detrend --detrend-method linear
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "validation" / "runs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hrvlib.data_handler import load_rr_file
from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis
from hrvlib.signal_processing.smoothness_priors import (
    detrend_uniform_with_smoothness_priors,
)


METRICS = {
    "VLF": "vlf_power",
    "LF": "lf_power",
    "HF": "hf_power",
    "total_power": "total_power",
    "LF/HF": "lf_hf_ratio",
    "LF_nu": "lf_nu",
    "HF_nu": "hf_nu",
}

BAND_FOR_METRIC = {
    "VLF": "vlf",
    "LF": "lf",
    "HF": "hf",
    "total_power": "total",
    "LF/HF": "lf_hf",
    "LF_nu": "lf_hf",
    "HF_nu": "lf_hf",
}

SUPPORTED_SUFFIXES = {".csv", ".txt", ".edf", ".hrm", ".fit", ".sml", ".json", ".acq"}
MIN_EFFECTIVE_WELCH_SAMPLES = 8


class ValidationSkip(Exception):
    """Expected validation-only skip that should not be treated as a failure."""

    def __init__(
        self,
        reason: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details = details or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate HRV Studio frequency-domain metrics against NeuroKit2 with "
            "matched RR input, duration, 4 Hz interpolation, bands, and Welch settings."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input RR/data files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--output",
        default="freq_domain_neurokit2_validation.csv",
        help="Output CSV filename. It is always written inside validation/runs/<run_name>/.",
    )
    parser.add_argument("--run-name", required=True, help="Run folder name under validation/runs/.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing run folder.")
    parser.add_argument(
        "--purpose",
        default="Compare HRV Studio native frequency-domain metrics against NeuroKit2.",
        help="Short description written to run_info.json and notes.md.",
    )
    parser.add_argument("--enable-diagnostics", action="store_true")
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories.")
    parser.add_argument("--interpolation-rate", type=float, default=4.0)
    parser.add_argument("--segment-length", type=float, default=120.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.75)
    parser.add_argument("--window-type", default="hann")
    parser.add_argument("--ar-order", type=int, default=16)
    parser.add_argument("--detrend-lambda", type=float, default=500.0)
    parser.add_argument(
        "--detrend-method",
        choices=["none", "linear", "constant", "smoothness_priors"],
        default="none",
        help="Native HRV Studio detrending method to use for the comparison.",
    )
    parser.add_argument(
        "--neurokit-interpolation-method",
        default="cubic",
        help=(
            "Interpolation method passed to NeuroKit2 intervals_process. "
            "Use cubic for closest conceptual match to HRV Studio cubic interpolation."
        ),
    )
    parser.add_argument(
        "--experimental-native-welch-nfft-multiplier",
        type=float,
        default=1.0,
        help=(
            "Validation-only experiment: recompute native Welch metrics with "
            "nfft = multiplier * nperseg and add comparison columns. "
            "Default HRV Studio behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--welch-detrend-mode",
        choices=["current", "segment_linear", "global_then_none"],
        default="current",
        help=(
            "Validation-only NeuroKit2 comparator Welch detrending mode. "
            "'current' preserves existing behavior exactly. 'segment_linear' "
            "uses segment-wise linear detrending inside Welch. 'global_then_none' "
            "globally detrends the interpolated RR signal first, then runs Welch "
            "with detrend=False to mimic NeuroKit2-style near-zero behavior."
        ),
    )
    return parser.parse_args()


def prepare_run_directory(run_name: str, overwrite: bool) -> Path:
    run_name = run_name.strip()
    run_path = Path(run_name)
    if not run_name or run_path.is_absolute() or len(run_path.parts) != 1 or ".." in run_path.parts:
        raise SystemExit("--run-name must be a simple folder name under validation/runs/.")

    run_dir = RUNS_ROOT / run_name
    if run_dir.exists() and not overwrite:
        raise SystemExit(
            f"Run directory already exists: {run_dir}\n"
            "Use --overwrite to write into this run folder explicitly."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def git_output(*args: str) -> Optional[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_versions() -> Dict[str, Optional[str]]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "pandas": package_version("pandas"),
        "neurokit2": package_version("neurokit2"),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            row.update(flatten_dict(value, flat_key))
        elif isinstance(value, (list, tuple)):
            row[flat_key] = json.dumps(to_jsonable(value), ensure_ascii=False)
        else:
            row[flat_key] = to_jsonable(value)
    return row


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)
        f.write("\n")


def finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def format_float(value: Optional[float], digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "not available"
    return f"{value:.{digits}f}"


def format_count_pct(count: Optional[int], total: int) -> str:
    if count is None:
        return "not available"
    pct = (count / total * 100.0) if total else 0.0
    return f"{count}/{total} ({pct:.1f}%)"


def is_metadata_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False

    name = path.name.lower()
    stem = path.stem.lower()
    return (
        "manifest" in name
        or stem.endswith("_manifest")
        or "dataset_summary" in stem
        or "quality_report" in stem
    )


def file_event(
    path: Path,
    reason: str,
    message: str,
    processed: bool = False,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "input_file": str(path),
        "reason": reason,
        "message": message,
        "processed": processed,
        "details": details or {},
    }


def hrv_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "band_convention": HRVFreqDomainAnalysis.DEFAULT_BAND_CONVENTION,
        "sampling_rate": args.interpolation_rate,
        "window_type": args.window_type,
        "segment_length": args.segment_length,
        "overlap_ratio": args.overlap_ratio,
        "detrend_method": normalize_detrend_method(args.detrend_method),
        "detrend_lambda": args.detrend_lambda,
        "ar_order": args.ar_order,
        "enable_diagnostics": args.enable_diagnostics,
        "experimental_native_welch_nfft_multiplier": args.experimental_native_welch_nfft_multiplier,
        "welch_detrend_mode": args.welch_detrend_mode,
    }


def neurokit2_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "band_convention": HRVFreqDomainAnalysis.DEFAULT_BAND_CONVENTION,
        "interpolation_rate": args.interpolation_rate,
        "interpolation_method": args.neurokit_interpolation_method,
        "normalize": False,
        "welch_detrend_mode": args.welch_detrend_mode,
    }


def build_run_info(
    args: argparse.Namespace,
    input_paths: List[Path],
    run_dir: Path,
    failures: List[Tuple[Path, Exception]],
    file_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    git_status = git_output("status", "--short")
    skipped_files = [event for event in file_events if not event.get("processed")]
    return {
        "run_name": args.run_name,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "validation_script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "purpose": args.purpose,
        "validation_type": "frequency_domain_neurokit2_comparison",
        "welch_detrend_mode": args.welch_detrend_mode,
        "git_commit_hash": git_output("rev-parse", "HEAD"),
        "git_dirty_status": "clean" if git_status == "" else git_status,
        "input_files": [str(path) for path in input_paths],
        "output_directory": str(run_dir.resolve()),
        "hrv_freq_domain_settings": hrv_settings(args),
        "neurokit2_settings": neurokit2_settings(args),
        "software_versions": software_versions(),
        "failures": [{"input_file": str(path), "error": str(exc)} for path, exc in failures],
        "skipped_files": skipped_files,
        "file_events": file_events,
        "skipped_file_count": len(skipped_files),
    }


def diagnostic_metric_keys(results: Dict[str, Any]) -> List[str]:
    keys = []
    prefixes = ["", "welch_", "fft_", "ar_"]
    for prefix in prefixes:
        for metric_key in METRICS.values():
            key = f"{prefix}{metric_key}"
            if key in results and key not in keys:
                keys.append(key)
    return keys


def metric_values_equal(left: Any, right: Any) -> bool:
    try:
        left_numeric = float(left)
        right_numeric = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(left_numeric) and math.isnan(right_numeric):
        return True
    if not math.isfinite(left_numeric) or not math.isfinite(right_numeric):
        return left_numeric == right_numeric
    left_float = finite_float(left)
    right_float = finite_float(right)
    if left_float is None or right_float is None:
        return left == right
    return bool(np.isclose(left_float, right_float, rtol=1e-12, atol=1e-12, equal_nan=True))


def diagnostics_changed_metric_outputs(
    rr_ms: np.ndarray,
    args: argparse.Namespace,
    diagnostic_results: Dict[str, Any],
) -> Optional[bool]:
    if not args.enable_diagnostics:
        return None

    baseline = HRVFreqDomainAnalysis(
        rr_ms,
        sampling_rate=args.interpolation_rate,
        detrend_method=normalize_detrend_method(args.detrend_method),
        detrend_lambda=args.detrend_lambda,
        window_type=args.window_type,
        segment_length=args.segment_length,
        overlap_ratio=args.overlap_ratio,
        ar_order=args.ar_order,
        enable_diagnostics=False,
    ).get_results()

    for key in diagnostic_metric_keys(baseline):
        if key not in diagnostic_results:
            return True
        if not metric_values_equal(baseline.get(key), diagnostic_results.get(key)):
            return True
    return False


def summarize_observations(
    rows: List[Dict[str, object]],
    diagnostics_records: List[Dict[str, Any]],
    failures: List[Tuple[Path, Exception]],
) -> List[str]:
    rows_by_file: Dict[str, Dict[str, object]] = {}
    for row in rows:
        rows_by_file.setdefault(str(row.get("input_file", "")), row)

    processed_files = sorted(path for path in rows_by_file if path)
    processed_count = len(processed_files)
    failed_count = len(failures)

    lines = [
        "## Auto-generated observations",
        "",
        "This draft is generated from validation outputs for manual review. It reports descriptive summaries only and does not make scientific or causal conclusions.",
        "",
        f"- Files processed successfully: {processed_count}",
    ]
    if failed_count:
        lines.append(f"- Files not processed due to errors: {failed_count}")

    durations = [
        value
        for value in (
            finite_float(rows_by_file[path].get("native_recording_duration_s"))
            for path in processed_files
        )
        if value is not None
    ]
    if durations:
        lines.append(
            "- Recording duration, seconds: "
            f"mean {format_float(float(np.mean(durations)))}, "
            f"min {format_float(float(np.min(durations)))}, "
            f"max {format_float(float(np.max(durations)))}"
        )
    else:
        lines.append("- Recording duration, seconds: not available")

    diagnostics_by_file = {
        str(record.get("input_file", "")): record.get("frequency_diagnostics", {})
        for record in diagnostics_records
    }
    if diagnostics_by_file:
        duration_warning_count = sum(
            1
            for path in processed_files
            if diagnostics_by_file.get(path, {}).get("duration_warnings")
        )
    else:
        duration_warning_count = sum(
            1
            for path in processed_files
            if (finite_float(rows_by_file[path].get("native_recording_duration_s")) or 0.0) < 300.0
        )
    lines.append(
        "- Files triggering duration warnings: "
        f"{format_count_pct(duration_warning_count, processed_count)}"
    )

    for band in ["vlf", "lf", "hf"]:
        count = sum(
            1
            for path in processed_files
            if (finite_float(rows_by_file[path].get(f"native_bins_{band}")) or 0.0) < 2.0
        )
        lines.append(
            f"- Files with <2 {band.upper()} bins in native Welch PSD: "
            f"{format_count_pct(count, processed_count)}"
        )

    lines.extend(["", "Mean relative errors:"])
    for metric in METRICS:
        metric_errors = [
            value
            for value in (
                finite_float(row.get("relative_error"))
                for row in rows
                if row.get("metric") == metric
            )
            if value is not None
        ]
        if metric_errors:
            mean_error = float(np.mean(metric_errors))
            lines.append(f"- {metric}: {mean_error:.4f} ({mean_error * 100.0:.2f}%)")
        else:
            lines.append(f"- {metric}: not available")

    diagnostic_change_values = {
        path: rows_by_file[path].get("diagnostics_changed_metric_outputs")
        for path in processed_files
        if rows_by_file[path].get("diagnostics_changed_metric_outputs") is not None
    }
    if diagnostic_change_values:
        changed_count = sum(1 for changed in diagnostic_change_values.values() if changed)
        lines.append(
            "- Diagnostics changed metric outputs: "
            f"{format_count_pct(changed_count, len(diagnostic_change_values))}"
        )
    else:
        lines.append(
            "- Diagnostics changed metric outputs: not assessed in this run "
            "(diagnostics were disabled or no diagnostic comparison was recorded)"
        )

    if diagnostics_by_file:
        ar_fallback_count = 0
        variance_warning_count = 0
        for path in processed_files:
            diagnostics = diagnostics_by_file.get(path, {})
            ar_diagnostics = diagnostics.get("ar", {})
            if ar_diagnostics.get("fallback_reason") or ar_diagnostics.get("estimator_used") == "welch_fallback":
                ar_fallback_count += 1
            variance_warning = any(
                diagnostics.get(method, {}).get("warning")
                for method in ["fft", "ar"]
            )
            if variance_warning:
                variance_warning_count += 1
        lines.append(
            f"- AR fallbacks reported in diagnostics: {format_count_pct(ar_fallback_count, processed_count)}"
        )
        lines.append(
            "- Files with variance consistency warnings: "
            f"{format_count_pct(variance_warning_count, processed_count)}"
        )
    else:
        lines.append("- AR fallbacks reported in diagnostics: not available")
        lines.append("- Files with variance consistency warnings: not available")

    lines.extend(
        [
            "",
            "Observed discrepancies are consistent with a descriptive validation comparison where recording duration and PSD bin coverage vary across files. This statement should be reviewed manually and should not be read as a causal explanation.",
            "",
            "## Researcher interpretation",
            "(To be completed manually)",
            "",
        ]
    )
    return lines


def write_notes(
    path: Path,
    args: argparse.Namespace,
    input_paths: List[Path],
    output_files: List[Path],
    rows: List[Dict[str, object]],
    diagnostics_records: List[Dict[str, Any]],
    failures: List[Tuple[Path, Exception]],
    file_events: List[Dict[str, Any]],
) -> None:
    settings = hrv_settings(args)
    skipped_events = [event for event in file_events if not event.get("processed")]
    adjusted_events = [
        event
        for event in file_events
        if event.get("reason") == "adjusted_noverlap_for_short_signal"
    ]
    lines = [
        f"# Validation Run: {args.run_name}",
        "",
        "## Purpose",
        args.purpose,
        "",
        "## Validation Type",
        "Frequency-domain HRV Studio vs NeuroKit2 comparison",
        "",
        "## Code State",
        f"- Diagnostics enabled/disabled: {args.enable_diagnostics}",
        "- Core PSD algorithms changed: No",
        "- Welch changed: No",
        "- FFT changed: No",
        "- AR changed: No",
        "",
        "## Input Files",
        *[f"- {path}" for path in input_paths],
        "",
        "## Skipped or Adjusted Files",
        f"- Skipped files: {len(skipped_events)}",
        f"- Files with adjusted Welch overlap: {len(adjusted_events)}",
        *[
            f"- {event['reason']}: {event['input_file']} ({event['message']})"
            for event in file_events
        ],
        "",
        "## Settings",
        f"- band_convention = {settings['band_convention']}",
        f"- sampling_rate = {settings['sampling_rate']}",
        f"- window_type = {settings['window_type']}",
        f"- segment_length = {settings['segment_length']}",
        f"- overlap_ratio = {settings['overlap_ratio']}",
        f"- detrend_method = {settings['detrend_method']}",
        f"- welch_detrend_mode = {settings['welch_detrend_mode']}",
        f"- enable_diagnostics = {settings['enable_diagnostics']}",
        "",
        "## Outputs",
        *[f"- {path.name}" for path in output_files],
        "",
        *summarize_observations(rows, diagnostics_records, failures),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_inputs(inputs: Iterable[str], recursive: bool) -> Tuple[List[Path], List[Dict[str, Any]]]:
    paths: List[Path] = []
    file_events: List[Dict[str, Any]] = []
    seen_events = set()
    for item in inputs:
        matches = [Path(p) for p in glob.glob(item)]
        candidates = matches if matches else [Path(item)]

        for candidate in candidates:
            if candidate.is_dir():
                pattern = "**/*" if recursive else "*"
                for path in candidate.glob(pattern):
                    if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                        continue
                    if is_metadata_csv(path):
                        resolved = path.resolve()
                        if resolved not in seen_events:
                            file_events.append(
                                file_event(
                                    resolved,
                                    "metadata_file_skipped",
                                    "Metadata CSV was skipped during input discovery.",
                                )
                            )
                            seen_events.add(resolved)
                        continue
                    paths.append(path)
            elif candidate.is_file():
                if is_metadata_csv(candidate):
                    resolved = candidate.resolve()
                    if resolved not in seen_events:
                        file_events.append(
                            file_event(
                                resolved,
                                "metadata_file_skipped",
                                "Metadata CSV was skipped during input discovery.",
                            )
                        )
                        seen_events.add(resolved)
                    continue
                paths.append(candidate)

    unique = sorted({p.resolve() for p in paths})
    if not unique:
        if file_events:
            raise SystemExit("No analyzable RR input files found after skipping metadata files.")
        raise SystemExit("No input files found.")
    return unique, file_events


def load_rr_intervals_ms(path: Path) -> np.ndarray:
    bundle = load_rr_file(str(path))
    values = bundle.rri_ms or bundle.ppi_ms
    if not values:
        raise ValueError("No RRI/PPI intervals found in file.")
    return np.asarray(values, dtype=float)


def normalize_detrend_method(value: str) -> Optional[str]:
    return None if value == "none" else value


def effective_welch_params(analyzer: HRVFreqDomainAnalysis) -> Tuple[Optional[int], Optional[int]]:
    n_samples = len(analyzer.time_domain_s)
    if n_samples == 0:
        return None, None

    requested = int(analyzer.segment_length * analyzer.sampling_rate)
    nperseg = min(requested, n_samples)
    if analyzer.welch_shorten_if_short and nperseg == n_samples and nperseg >= 16:
        nperseg = max(8, n_samples // 2)
    if nperseg < 8:
        return None, None
    return nperseg, int(nperseg * analyzer.overlap_ratio)


def safe_welch_params_for_signal(
    path: Path,
    n_samples: int,
    requested_nperseg: int,
    requested_noverlap: int,
) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    effective_nperseg = int(requested_nperseg)
    neurokit_short_window_rule_applied = False
    if effective_nperseg > int(n_samples) / 2:
        effective_nperseg = int(int(n_samples) / 2)
        neurokit_short_window_rule_applied = True
    if effective_nperseg < MIN_EFFECTIVE_WELCH_SAMPLES:
        raise ValidationSkip(
            "insufficient_effective_length_for_welch",
            (
                "Effective Welch signal length is too short for reliable "
                "frequency-domain HRV validation."
            ),
            {
                "requested_nperseg": int(requested_nperseg),
                "requested_noverlap": int(requested_noverlap),
                "effective_nperseg": int(effective_nperseg),
                "n_samples": int(n_samples),
                "minimum_effective_nperseg": int(MIN_EFFECTIVE_WELCH_SAMPLES),
                "neurokit_short_window_rule_applied": bool(
                    neurokit_short_window_rule_applied
                ),
            },
        )

    effective_noverlap = min(int(requested_noverlap), max(0, effective_nperseg - 1))
    adjustment = None
    if (
        effective_noverlap != int(requested_noverlap)
        or effective_nperseg != int(requested_nperseg)
    ):
        adjustment = file_event(
            path,
            "adjusted_noverlap_for_short_signal",
            (
                "Effective Welch parameters were adjusted for a short signal so "
                "noverlap < nperseg."
            ),
            processed=True,
            details={
                "requested_nperseg": int(requested_nperseg),
                "requested_noverlap": int(requested_noverlap),
                "effective_nperseg": int(effective_nperseg),
                "effective_noverlap": int(effective_noverlap),
                "n_samples": int(n_samples),
                "neurokit_short_window_rule_applied": bool(
                    neurokit_short_window_rule_applied
                ),
            },
        )
    return effective_nperseg, effective_noverlap, adjustment


def detrended_ar_input(analyzer: HRVFreqDomainAnalysis) -> Optional[np.ndarray]:
    if len(analyzer.time_domain_s) == 0:
        return None

    if analyzer.detrend_method == "smoothness_priors":
        if getattr(analyzer, "_smoothness_priors_applied", False):
            detrended = analyzer.time_domain_s
        else:
            detrended = signal.detrend(analyzer.time_domain_s, type="linear")
    elif analyzer.detrend_method == "linear":
        detrended = signal.detrend(analyzer.time_domain_s, type="linear")
    elif analyzer.detrend_method == "constant":
        detrended = signal.detrend(analyzer.time_domain_s, type="constant")
    else:
        detrended = analyzer.time_domain_s.copy()

    return detrended - np.mean(detrended)


def effective_ar_order(analyzer: HRVFreqDomainAnalysis) -> Optional[int]:
    ar_input = detrended_ar_input(analyzer)
    if ar_input is None or len(ar_input) < 8:
        return None

    n = len(ar_input)
    adaptive_order = min(analyzer.ar_order, max(4, n // 6))
    adaptive_order = max(4, adaptive_order)

    for order in range(adaptive_order, 3, -1):
        burg = analyzer._burg_try(ar_input, order)
        if burg is not None:
            return order
        yule_walker = analyzer._yule_walker_estimate(ar_input, order)
        if yule_walker is not None:
            return order
    return None


def band_ranges() -> Dict[str, Tuple[float, float]]:
    bands = {
        name: HRVFreqDomainAnalysis.band_spec_for_convention(name)
        for name in ("ulf", "vlf", "lf", "hf")
    }
    total = HRVFreqDomainAnalysis.total_power_spec_for_convention()
    return {
        "ulf": (bands["ulf"]["low"], bands["ulf"]["high"]),
        "vlf": (bands["vlf"]["low"], bands["vlf"]["high"]),
        "lf": (bands["lf"]["low"], bands["lf"]["high"]),
        "hf": (bands["hf"]["low"], bands["hf"]["high"]),
        "total": (total["low"], total["high"]),
    }


def band_mask(freqs: np.ndarray, band: str) -> np.ndarray:
    if band == "total":
        return HRVFreqDomainAnalysis.total_power_mask_for_convention(freqs)
    return HRVFreqDomainAnalysis.mask_for_band(freqs, band)


def count_bins(freqs: np.ndarray, band: str) -> int:
    if len(freqs) == 0:
        return 0
    return int(np.count_nonzero(band_mask(freqs, band)))


def band_bin_counts(freqs: np.ndarray) -> Dict[str, int]:
    counts = {name: count_bins(freqs, name) for name in ("ulf", "vlf", "lf", "hf", "total")}
    counts["lf_hf"] = counts["lf"] + counts["hf"]
    return counts


def integrate_band(freqs: np.ndarray, psd: np.ndarray, band: str) -> float:
    mask = band_mask(freqs, band)
    if not np.any(mask):
        return 0.0
    return float(max(0.0, np.trapezoid(psd[mask], freqs[mask])))


def metrics_from_psd(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    vlf = integrate_band(freqs, psd, "vlf")
    lf = integrate_band(freqs, psd, "lf")
    hf = integrate_band(freqs, psd, "hf")
    total = integrate_band(freqs, psd, "total")

    lf_hf = math.nan
    if hf > 1e-10:
        lf_hf = lf / hf
    elif lf > 1e-10:
        lf_hf = math.inf

    lf_hf_sum = lf + hf
    lf_nu = (lf / lf_hf_sum) * 100.0 if lf_hf_sum > 0 else 0.0
    hf_nu = (hf / lf_hf_sum) * 100.0 if lf_hf_sum > 0 else 0.0

    return {
        "vlf_power": vlf,
        "lf_power": lf,
        "hf_power": hf,
        "total_power": total,
        "lf_hf_ratio": lf_hf,
        "lf_nu": lf_nu,
        "hf_nu": hf_nu,
    }


def rr_start_times_s(rr_ms: np.ndarray) -> np.ndarray:
    rr_s = rr_ms.astype(float) / 1000.0
    return np.concatenate([[0.0], np.cumsum(rr_s[:-1])])


def apply_matching_detrend_ms(
    signal_ms: np.ndarray,
    detrend_method: Optional[str],
    detrend_lambda: float,
) -> np.ndarray:
    if detrend_method == "linear":
        return signal.detrend(signal_ms, type="linear")
    if detrend_method == "constant":
        return signal.detrend(signal_ms, type="constant")
    if detrend_method == "smoothness_priors":
        detrended_s = detrend_uniform_with_smoothness_priors(
            signal_ms / 1000.0,
            lambda_param=detrend_lambda,
            return_trend=False,
        )
        return detrended_s * 1000.0
    return signal_ms.copy()


def neurokit2_welch_psd(
    path: Path,
    rr_ms: np.ndarray,
    sampling_rate: float,
    interpolation_method: str,
    window_type: str,
    nperseg: int,
    noverlap: int,
    detrend_method: Optional[str],
    detrend_lambda: float,
) -> Tuple[np.ndarray, np.ndarray, int, int, float, Optional[Dict[str, Any]]]:
    import neurokit2 as nk

    rri_time = rr_start_times_s(rr_ms)
    intervals, intervals_time, interpolation_rate = nk.intervals_process(
        rr_ms,
        intervals_time=rri_time,
        interpolate=True,
        interpolation_rate=sampling_rate,
        method=interpolation_method,
    )
    intervals = np.asarray(intervals, dtype=float)

    effective_nperseg, effective_noverlap, adjustment = safe_welch_params_for_signal(
        path=path,
        n_samples=len(intervals),
        requested_nperseg=int(nperseg),
        requested_noverlap=int(noverlap),
    )

    psd = nk.signal_psd(
        intervals,
        sampling_rate=interpolation_rate,
        method="welch",
        normalize=False,
        min_frequency=-np.inf,
        max_frequency=band_ranges()["total"][1],
        window=effective_nperseg / interpolation_rate,
        window_type=window_type,
        noverlap=effective_noverlap,
        silent=True,
    )

    duration_s = float(intervals_time[-1] - intervals_time[0]) if len(intervals_time) else 0.0
    return (
        psd["Frequency"].to_numpy(dtype=float),
        psd["Power"].to_numpy(dtype=float),
        effective_nperseg,
        effective_noverlap,
        duration_s,
        adjustment,
    )


def validation_welch_psd_with_mode(
    path: Path,
    rr_ms: np.ndarray,
    sampling_rate: float,
    interpolation_method: str,
    window_type: str,
    nperseg: int,
    noverlap: int,
    detrend_method: Optional[str],
    detrend_lambda: float,
    welch_detrend_mode: str,
) -> Tuple[np.ndarray, np.ndarray, int, int, float, Optional[Dict[str, Any]]]:
    if welch_detrend_mode == "current":
        return neurokit2_welch_psd(
            path=path,
            rr_ms=rr_ms,
            sampling_rate=sampling_rate,
            interpolation_method=interpolation_method,
            window_type=window_type,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend_method=detrend_method,
            detrend_lambda=detrend_lambda,
        )

    import neurokit2 as nk

    rri_time = rr_start_times_s(rr_ms)
    intervals, intervals_time, interpolation_rate = nk.intervals_process(
        rr_ms,
        intervals_time=rri_time,
        interpolate=True,
        interpolation_rate=sampling_rate,
        method=interpolation_method,
    )
    intervals = apply_matching_detrend_ms(
        np.asarray(intervals, dtype=float),
        detrend_method,
        detrend_lambda,
    )

    effective_nperseg, effective_noverlap, adjustment = safe_welch_params_for_signal(
        path=path,
        n_samples=len(intervals),
        requested_nperseg=int(nperseg),
        requested_noverlap=int(noverlap),
    )
    nfft = int(effective_nperseg * 2)
    if welch_detrend_mode == "segment_linear":
        signal_input = intervals
        scipy_detrend: Any = "linear"
    elif welch_detrend_mode == "global_then_none":
        signal_input = apply_matching_detrend_ms(
            intervals,
            detrend_method,
            detrend_lambda,
        )
        scipy_detrend = False
    else:
        raise ValueError(f"Unsupported welch_detrend_mode: {welch_detrend_mode}")

    freqs, psd = signal.welch(
        signal_input,
        fs=interpolation_rate,
        window=window_type,
        nperseg=effective_nperseg,
        noverlap=effective_noverlap,
        nfft=nfft,
        detrend=scipy_detrend,
        scaling="density",
        average="mean",
    )
    mask = (freqs >= -np.inf) & (freqs <= band_ranges()["total"][1])
    duration_s = float(intervals_time[-1] - intervals_time[0]) if len(intervals_time) else 0.0
    return (
        freqs[mask],
        psd[mask],
        effective_nperseg,
        effective_noverlap,
        duration_s,
        adjustment,
    )


def experimental_native_welch_psd(
    analyzer: HRVFreqDomainAnalysis,
    nperseg: Optional[int],
    noverlap: Optional[int],
    nfft_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    if nperseg is None or noverlap is None or nfft_multiplier <= 1.0:
        return np.array([]), np.array([]), None

    nfft = int(round(float(nperseg) * nfft_multiplier))
    if nfft < nperseg:
        nfft = nperseg

    window = analyzer._get_window(nperseg)
    if analyzer.detrend_method == "smoothness_priors":
        if getattr(analyzer, "_smoothness_priors_applied", False):
            signal_input = analyzer.time_domain_s
            detrend_param: Any = False
        else:
            signal_input = analyzer.time_domain_s
            detrend_param = "linear"
    else:
        signal_input = analyzer.time_domain_s
        detrend_param = analyzer.detrend_method if analyzer.detrend_method else False

    freqs, psd_seconds = signal.welch(
        x=signal_input,
        fs=analyzer.sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend_param,
        scaling="density",
        average="mean",
    )
    return freqs, psd_seconds * 1e6, nfft


def relative_error(native_value: float, reference_value: float) -> float:
    if not np.isfinite(native_value) or not np.isfinite(reference_value):
        return math.nan
    denominator = abs(reference_value)
    if denominator <= 1e-12:
        return math.nan
    return abs(native_value - reference_value) / denominator


def safe_abs_error(native_value: float, reference_value: float) -> float:
    if not np.isfinite(native_value) or not np.isfinite(reference_value):
        return math.nan
    return abs(native_value - reference_value)


def analyze_file(
    path: Path, args: argparse.Namespace
) -> Tuple[List[Dict[str, object]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    detrend_method = normalize_detrend_method(args.detrend_method)
    raw_rr_ms = load_rr_intervals_ms(path)

    analyzer = HRVFreqDomainAnalysis(
        raw_rr_ms,
        sampling_rate=args.interpolation_rate,
        detrend_method=detrend_method,
        detrend_lambda=args.detrend_lambda,
        window_type=args.window_type,
        segment_length=args.segment_length,
        overlap_ratio=args.overlap_ratio,
        ar_order=args.ar_order,
        enable_diagnostics=args.enable_diagnostics,
    )
    native_results = analyzer.get_results()
    diagnostics = None
    if args.enable_diagnostics:
        diagnostics = {
            "input_file": str(path),
            "frequency_diagnostics": native_results.get("frequency_diagnostics", {}),
        }
    diagnostics_changed_outputs = diagnostics_changed_metric_outputs(
        raw_rr_ms,
        args,
        native_results,
    )

    native_nperseg, native_noverlap = effective_welch_params(analyzer)
    if native_nperseg is None or native_noverlap is None:
        raise ValidationSkip(
            "insufficient_effective_length_for_welch",
            (
                "Native effective Welch parameters could not be computed because "
                "the resampled signal is too short for frequency-domain validation."
            ),
            {
                "rr_count": int(len(raw_rr_ms)),
                "resampled_sample_count": int(len(analyzer.time_domain_s)),
                "minimum_effective_nperseg": int(MIN_EFFECTIVE_WELCH_SAMPLES),
            },
        )
    ar_order = effective_ar_order(analyzer)
    file_events: List[Dict[str, Any]] = []

    rr_ms = analyzer.rr_intervals_ms
    native_duration_s = float(rr_start_times_s(rr_ms)[-1]) if len(rr_ms) > 1 else 0.0
    native_counts = band_bin_counts(analyzer.freqs)

    requested_nperseg = int(args.segment_length * args.interpolation_rate)
    requested_noverlap = int(requested_nperseg * args.overlap_ratio)
    (
        nk_freqs,
        nk_psd,
        nk_nperseg,
        nk_noverlap,
        nk_duration_s,
        nk_adjustment,
    ) = validation_welch_psd_with_mode(
        path=path,
        rr_ms=rr_ms,
        sampling_rate=args.interpolation_rate,
        interpolation_method=args.neurokit_interpolation_method,
        window_type=args.window_type,
        nperseg=requested_nperseg,
        noverlap=requested_noverlap,
        detrend_method=detrend_method,
        detrend_lambda=args.detrend_lambda,
        welch_detrend_mode=args.welch_detrend_mode,
    )
    if nk_adjustment is not None:
        file_events.append(nk_adjustment)

    nk_metrics = metrics_from_psd(nk_freqs, nk_psd)
    nk_counts = band_bin_counts(nk_freqs)
    experimental_freqs, experimental_psd, experimental_nfft = experimental_native_welch_psd(
        analyzer,
        native_nperseg,
        native_noverlap,
        args.experimental_native_welch_nfft_multiplier,
    )
    experimental_metrics = metrics_from_psd(experimental_freqs, experimental_psd)
    experimental_counts = band_bin_counts(experimental_freqs)

    rows = []
    for label, key in METRICS.items():
        native_value = float(native_results.get(f"welch_{key}", native_results.get(key, math.nan)))
        nk_value = float(nk_metrics.get(key, math.nan))
        metric_band = BAND_FOR_METRIC[label]
        experimental_value = float(experimental_metrics.get(key, math.nan))

        row = {
            "input_file": str(path),
            "metric": label,
            "native_value": native_value,
            "neurokit2_value": nk_value,
            "absolute_error": safe_abs_error(native_value, nk_value),
            "relative_error": relative_error(native_value, nk_value),
            "relative_error_pct": relative_error(native_value, nk_value) * 100.0,
            "metric_band": metric_band,
            "native_metric_band_bins": native_counts.get(metric_band, 0),
            "neurokit2_metric_band_bins": nk_counts.get(metric_band, 0),
            "native_bins_ulf": native_counts["ulf"],
            "native_bins_vlf": native_counts["vlf"],
            "native_bins_lf": native_counts["lf"],
            "native_bins_hf": native_counts["hf"],
            "native_bins_total": native_counts["total"],
            "neurokit2_bins_ulf": nk_counts["ulf"],
            "neurokit2_bins_vlf": nk_counts["vlf"],
            "neurokit2_bins_lf": nk_counts["lf"],
            "neurokit2_bins_hf": nk_counts["hf"],
            "neurokit2_bins_total": nk_counts["total"],
            "native_welch_nperseg": native_nperseg,
            "native_welch_noverlap": native_noverlap,
            "neurokit2_requested_welch_nperseg": requested_nperseg,
            "neurokit2_requested_welch_noverlap": requested_noverlap,
            "neurokit2_welch_nperseg": nk_nperseg,
            "neurokit2_welch_noverlap": nk_noverlap,
            "neurokit2_welch_adjustment_reason": (
                nk_adjustment["reason"] if nk_adjustment is not None else None
            ),
            "effective_ar_order": ar_order,
            "rr_count": len(rr_ms),
            "native_recording_duration_s": native_duration_s,
            "neurokit2_recording_duration_s": nk_duration_s,
            "diagnostics_changed_metric_outputs": diagnostics_changed_outputs,
            "interpolation_rate_hz": args.interpolation_rate,
            "window_type": args.window_type,
            "segment_length_s": args.segment_length,
            "overlap_ratio": args.overlap_ratio,
            "detrend_method": args.detrend_method,
            "welch_detrend_mode": args.welch_detrend_mode,
            "neurokit2_interpolation_method": args.neurokit_interpolation_method,
        }
        if experimental_nfft is not None:
            row.update(
                {
                    "experimental_native_welch_nfft_multiplier": args.experimental_native_welch_nfft_multiplier,
                    "experimental_native_welch_nfft": experimental_nfft,
                    "experimental_native_welch_frequency_resolution_hz": (
                        args.interpolation_rate / experimental_nfft
                    ),
                    "experimental_native_value": experimental_value,
                    "experimental_absolute_error": safe_abs_error(experimental_value, nk_value),
                    "experimental_relative_error": relative_error(experimental_value, nk_value),
                    "experimental_relative_error_pct": (
                        relative_error(experimental_value, nk_value) * 100.0
                    ),
                    "experimental_native_metric_band_bins": experimental_counts.get(metric_band, 0),
                    "experimental_native_bins_ulf": experimental_counts["ulf"],
                    "experimental_native_bins_vlf": experimental_counts["vlf"],
                    "experimental_native_bins_lf": experimental_counts["lf"],
                    "experimental_native_bins_hf": experimental_counts["hf"],
                    "experimental_native_bins_total": experimental_counts["total"],
                }
            )
        rows.append(row)
    return rows, diagnostics, file_events


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_paths, file_events = resolve_inputs(args.inputs, args.recursive)
    run_dir = prepare_run_directory(args.run_name, args.overwrite)

    rows: List[Dict[str, object]] = []
    diagnostics_records: List[Dict[str, Any]] = []
    failures = []
    for path in input_paths:
        try:
            file_rows, diagnostics, analysis_events = analyze_file(path, args)
            rows.extend(file_rows)
            if diagnostics is not None:
                diagnostics_records.append(diagnostics)
            file_events.extend(analysis_events)
        except ValidationSkip as exc:
            file_events.append(
                file_event(path, exc.reason, exc.message, details=exc.details)
            )
        except Exception as exc:
            failures.append((path, exc))

    output_path = run_dir / Path(args.output).name
    write_csv(output_path, rows)

    output_files = [output_path] if rows else []
    if diagnostics_records:
        diagnostics_json = run_dir / "diagnostics.json"
        diagnostics_csv = run_dir / "diagnostics.csv"
        write_json(diagnostics_json, diagnostics_records)
        write_csv(diagnostics_csv, [flatten_dict(record) for record in diagnostics_records])
        output_files.extend([diagnostics_csv, diagnostics_json])

    run_info_path = run_dir / "run_info.json"
    notes_path = run_dir / "notes.md"
    write_json(run_info_path, build_run_info(args, input_paths, run_dir, failures, file_events))
    write_notes(
        notes_path,
        args,
        input_paths,
        output_files + [run_info_path],
        rows,
        diagnostics_records,
        failures,
        file_events,
    )
    output_files.extend([run_info_path, notes_path])

    skipped_count = sum(1 for event in file_events if not event.get("processed"))
    adjusted_count = sum(
        1
        for event in file_events
        if event.get("reason") == "adjusted_noverlap_for_short_signal"
    )
    processed_count = len({str(row.get("input_file")) for row in rows if row.get("input_file")})
    print(f"Wrote {len(rows)} comparison rows for {processed_count} files: {output_path}")
    if skipped_count or adjusted_count:
        print(
            f"Recorded {skipped_count} skipped files and "
            f"{adjusted_count} Welch overlap adjustments in {run_info_path}"
        )

    if failures:
        print("Failures:")
        for path, exc in failures:
            print(f"  {path}: {exc}")

    if not rows:
        raise SystemExit("No validation rows were produced.")


if __name__ == "__main__":
    main()
