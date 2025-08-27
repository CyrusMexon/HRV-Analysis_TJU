import os
import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import sys

import numpy as np

# Import preprocessing module
from hrvlib.preprocessing import PreprocessingResult, preprocess_rri

# --- Optional dependencies (loaded lazily and handled gracefully) ---
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import pyedflib  # EDF (incl. Bittium Faros often exports EDF)
except Exception:
    pyedflib = None

try:
    from fitparse import FitFile  # Garmin .fit
except Exception:
    FitFile = None

try:
    import xml.etree.ElementTree as ET  # Suunto .sml (XML)
except Exception:
    ET = None

try:
    import bioread  # Biopac/AcqKnowledge .acq
except Exception:
    bioread = None


# =========================
# Data structures (FR-6)
# =========================


@dataclass
class SourceInfo:
    path: str
    filetype: str
    device: Optional[str] = None
    notes: Optional[str] = None
    acquisition_date: Optional[str] = None  # ISO string or datetime


@dataclass
class TimeSeries:
    name: str  # 'ECG', 'PPG', 'RESP', etc.
    data: np.ndarray  # 1-D waveform
    fs: float  # sampling rate in Hz
    units: Optional[str] = None  # e.g., 'mV', 'a.u.'
    start_time: Optional[float] = None  # seconds


@dataclass
class DataBundle:
    # Intervals (ms)
    rri_ms: List[float] = field(default_factory=list)  # NN/RR intervals (ms)
    ppi_ms: List[float] = field(default_factory=list)  # pulse-to-pulse intervals (ms)

    # Waveforms
    ecg: List[TimeSeries] = field(default_factory=list)
    ppg: List[TimeSeries] = field(default_factory=list)
    resp: List[TimeSeries] = field(default_factory=list)

    meta: Dict = field(default_factory=dict)
    source: Optional[SourceInfo] = None
    preprocessing: Optional[PreprocessingResult] = None

    def summary(self) -> Dict:
        """
        FR-6: concise summary of data type(s) and source.
        """
        types = []
        if self.rri_ms:
            types.append("RRI")
        if self.ppi_ms:
            types.append("PPI")
        if self.ecg:
            types.append("ECG")
        if self.ppg:
            types.append("PPG")
        if self.resp:
            types.append("RESP")

        wf_counts = {
            "ECG_channels": len(self.ecg),
            "PPG_channels": len(self.ppg),
            "RESP_channels": len(self.resp),
        }

        rr_stats = None
        if self.rri_ms:
            arr = np.asarray(self.rri_ms, dtype=float)
            rr_stats = {
                "n_RRI": int(arr.size),
                "mean_RRI_ms": float(np.nanmean(arr)),
                "duration_s": float(np.nansum(arr) / 1000.0),
            }

        ppi_stats = None
        if self.ppi_ms:
            arr = np.asarray(self.ppi_ms, dtype=float)
            ppi_stats = {
                "n_PPI": int(arr.size),
                "mean_PPI_ms": float(np.nanmean(arr)),
                "duration_s": float(np.nansum(arr) / 1000.0),
            }

        preprocessing_stats = None
        if self.preprocessing:
            preprocessing_stats = {
                "artifacts_detected": len(self.preprocessing.artifact_indices),
                "artifact_types": self.preprocessing.artifact_types,
                "correction_method": self.preprocessing.correction_method,
                "interpolated_points": len(self.preprocessing.interpolation_indices),
            }

        return {
            "types_loaded": types or ["Unknown"],
            "waveform_channels": wf_counts,
            "rri_overview": rr_stats,
            "ppi_overview": ppi_stats,
            "preprocessing": preprocessing_stats,
            "fs": self.meta.get("fs", None),
            "source": self.source.__dict__ if self.source else None,
            "meta_keys": list(self.meta.keys()),
        }


# =========================
# Configuration
# =========================

STRICT_MODE = True
INTERACTIVE_MODE = False  # Set to True for GUI interactions


def set_interactive_mode(interactive: bool):
    """Set whether to use interactive mode for missing parameters"""
    global INTERACTIVE_MODE
    INTERACTIVE_MODE = interactive


# =========================
# Public API
# =========================


def load_rr_file(
    path: str,
    auto_preprocess: bool = False,
    preprocessing_params: Optional[Dict] = None,
) -> DataBundle:
    """
    FR-1..FR-6 dispatcher:
    Loads RRI/PPI, ECG/PPG/RESP waveforms from multiple formats.
    Returns a DataBundle with summary().
    Ensures fs metadata is available for waveform-based signals.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    bundle = None

    if ext == ".csv":
        # CSV files require pandas
        if pd is None:
            raise ImportError(
                "pandas is required for CSV files. Install with: pip install pandas"
            )
        df = pd.read_csv(path, sep=None, engine="python")
        fs = None
        if "fs" in df.columns:
            try:
                fs = int(df["fs"].iloc[0])
            except Exception:
                warnings.warn(f"Invalid fs column in {path}; ignoring")
        bundle = _load_csv(path, df_override=df)
        if fs is not None:
            bundle.meta["fs"] = fs

    elif ext == ".txt":
        # TXT files can use pandas OR manual parsing
        if pd is not None:
            try:
                # Try pandas first for more robust parsing
                df = pd.read_csv(path, sep=None, engine="python")
                fs = None
                if "fs" in df.columns:
                    try:
                        fs = int(df["fs"].iloc[0])
                    except Exception:
                        warnings.warn(f"Invalid fs column in {path}; ignoring")
                bundle = _load_txt(path, df_override=df)
                if fs is not None:
                    bundle.meta["fs"] = fs
            except Exception:
                # Fall back to manual parsing if pandas fails
                bundle = _load_txt_manual(path)
        else:
            # Use manual parsing when pandas is not available
            bundle = _load_txt_manual(path)

    elif ext == ".edf":
        _require(pyedflib, "pyedflib", "pip install pyedflib")
        bundle = _load_edf(path)
    elif ext == ".hrm":
        bundle = _load_polar_hrm(path)
    elif ext == ".fit":
        _require(FitFile, "fitparse", "pip install fitparse")
        bundle = _load_garmin_fit(path)
    elif ext == ".sml":
        _require(ET, "xml.etree.ElementTree", "Python stdlib XML should exist")
        bundle = _load_suunto_sml(path)
    elif ext == ".json":
        bundle = _load_movesense_json(path)
    elif ext == ".acq":
        _require(bioread, "bioread", "pip install bioread")
        bundle = _load_biopac_acq(path)
    else:
        # MindMedia and Bittium Faros often export txt/csv/edf already covered.
        raise ValueError(f"Unsupported file type: {ext}")

    # --- Production safety check ---
    if any([bundle.ecg, bundle.ppg, bundle.resp]):
        if "fs" not in bundle.meta:
            if STRICT_MODE:
                if INTERACTIVE_MODE:
                    # GUI case → ask user
                    try:
                        from PyQt6 import QtWidgets

                        app = QtWidgets.QApplication.instance()
                        if app is None:
                            app = QtWidgets.QApplication(sys.argv)

                        fs, ok = QtWidgets.QInputDialog.getInt(
                            None,
                            "Missing Sampling Rate",
                            f"No sampling rate found in {os.path.basename(path)}.\nPlease enter fs (Hz):",
                            1000,
                            1,
                            5000,
                            1,
                        )
                        if ok:
                            bundle.meta["fs"] = fs
                        else:
                            raise ValueError(
                                f"Sampling rate required but missing for {path}"
                            )
                    except ImportError:
                        # PyQt6 not available, fall back to console input
                        print(f"No sampling rate found in {os.path.basename(path)}")
                        while True:
                            try:
                                fs_input = input(
                                    "Please enter sampling rate (Hz, 1-5000): "
                                )
                                fs = int(fs_input)
                                if 1 <= fs <= 5000:
                                    bundle.meta["fs"] = fs
                                    break
                                else:
                                    print("Sampling rate must be between 1 and 5000 Hz")
                            except (ValueError, KeyboardInterrupt):
                                raise ValueError(
                                    f"Sampling rate required but missing for {path}"
                                )
                else:
                    # CLI case → fail fast
                    raise ValueError(
                        f"No sampling rate found for {path}. Please specify fs."
                    )
            else:
                warnings.warn("No sampling rate found; assuming 1000 Hz (DEV MODE).")
                bundle.meta["fs"] = 1000

    # FR-6 add source info if not already present
    if bundle.source is None:
        bundle.source = SourceInfo(
            path=path, filetype=ext, device=bundle.meta.get("device")
        )

    # Derive RRI/PPI from ECG/PPG if missing (Pan–Tompkins etc.)
    bundle = _extract_intervals_from_waveforms(bundle)

    # Auto-preprocessing if requested
    if auto_preprocess and bundle.rri_ms:
        try:
            params = preprocessing_params or {}
            bundle.preprocessing = preprocess_rri(bundle.rri_ms, **params)
            # Update RRI with corrected values
            bundle.rri_ms = bundle.preprocessing.corrected_rri.tolist()
            bundle.meta["preprocessing_applied"] = True
        except Exception as e:
            warnings.warn(f"Auto-preprocessing failed: {e}")

    return bundle


# =========================
# Loaders per format
# =========================


# ---- CSV / TXT (RRI/PPI or generic columns) ----
def _load_csv(path: str, df_override=None) -> DataBundle:
    _require(pd, "pandas", "pip install pandas")
    df = df_override if df_override is not None else pd.read_csv(path)

    bundle = DataBundle(meta={"format": "csv"})
    cols = {c.lower(): c for c in df.columns}

    # Explicit RRI/PPI columns
    if "rri" in cols or "rr" in cols or "nn" in cols:
        c = cols.get("rri") or cols.get("rr") or cols.get("nn")
        bundle.rri_ms = _coerce_float_list(df[c].tolist())
    if "ppi" in cols:
        bundle.ppi_ms = _coerce_float_list(df[cols["ppi"]].tolist())

    # Global fs column (applies if waveforms exist)
    fs_global = None
    if "fs" in cols:
        try:
            # Get non-empty fs values and check if they're consistent
            fs_values = df[cols["fs"]].dropna()
            fs_values = fs_values[fs_values != ""]  # Remove empty strings

            if len(fs_values) > 0:
                unique_values = fs_values.unique()
                # Remove empty strings and convert to float
                numeric_values = []
                for val in unique_values:
                    if val != "" and val is not None:
                        try:
                            numeric_values.append(float(val))
                        except (ValueError, TypeError):
                            pass

                if len(numeric_values) == 1:  # Single consistent fs value
                    fs_global = numeric_values[0]
                    bundle.meta["fs"] = fs_global
                elif len(numeric_values) > 1:
                    # Multiple different fs values - this is problematic
                    warnings.warn("Multiple or invalid fs values in CSV; ignoring.")
                # else: no valid numeric fs values found, fs_global remains None
        except Exception:
            pass

    # Waveforms
    for kind in ["ecg", "ppg", "resp"]:
        if kind in cols:
            sig = _coerce_float_list(df[cols[kind]].tolist())
            if len(sig) > 1:
                fs = fs_global  # Use the global fs first

                # Check for signal-specific fs column
                if f"fs_{kind}" in cols:
                    try:
                        fs_specific_values = df[cols[f"fs_{kind}"]].dropna()
                        fs_specific_values = fs_specific_values[
                            fs_specific_values != ""
                        ]
                        if len(fs_specific_values) > 0:
                            fs = float(fs_specific_values.iloc[0])
                    except Exception:
                        pass

                # Try to infer from time vector if no fs found
                if not fs and "t" in cols:
                    t = _coerce_float_list(df[cols["t"]].tolist())
                    if len(t) > 2:
                        dt = np.median(np.diff(t))
                        if dt and dt > 0:
                            fs = 1.0 / dt

                # Only warn if we still don't have a sampling rate
                if not fs:
                    warnings.warn(
                        f"No sampling rate found for {kind.upper()} in CSV; assuming 1000 Hz"
                    )
                    fs = 1000.0

                ts = TimeSeries(
                    name=kind.upper(),
                    data=np.asarray(sig, dtype=float),
                    fs=fs,
                    units=None,
                )
                getattr(bundle, kind).append(ts)
                # propagate fs to bundle.meta for summary/strict checks
                if "fs" not in bundle.meta:
                    bundle.meta["fs"] = fs

    # Fallback: single column → treat as RRI in ms
    if not (bundle.rri_ms or bundle.ppi_ms or bundle.ecg or bundle.ppg or bundle.resp):
        first_col = df.iloc[:, 0]
        bundle.rri_ms = _coerce_float_list(first_col.tolist())

    return bundle


def _load_txt(path: str, df_override=None) -> DataBundle:
    """
    Support simple TXT as either:
      - One number per line → RRI (ms)
      - Space/CSV separated columns with optional headers like ECG, PPG, RESP, RRI, PPI
    """
    if pd is not None:
        try:
            df = (
                df_override
                if df_override is not None
                else pd.read_csv(path, sep=None, engine="python")
            )
            # Properly forward df_override instead of re-reading file blindly
            return _load_csv(path, df_override=df)
        except Exception:
            pass

    # Fallback: plain list → assume RRI
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip().replace(",", " ")
    nums = []
    for tok in content.split():
        try:
            nums.append(float(tok))
        except Exception:
            continue
    return DataBundle(rri_ms=nums, meta={"format": "txt"})


# ---- EDF (ECG/PPG/RESP waveforms; some devices export RR too via events) ----
def _load_edf(path: str) -> DataBundle:
    f = pyedflib.EdfReader(path)
    n = f.signals_in_file
    labels = f.getSignalLabels()
    fs_list = [f.getSampleFrequency(i) for i in range(n)]

    bundle = DataBundle(meta={"format": "edf", "edf_header": f.getHeader()})
    # Channel loop
    for i in range(n):
        sig = f.readSignal(i)
        label = labels[i] if i < len(labels) else f"ch{i}"
        kind = _label_to_kind(label)
        if kind in ("ECG", "PPG", "RESP"):
            ts = TimeSeries(
                name=kind,
                data=np.asarray(sig, dtype=float),
                fs=float(fs_list[i]),
                units=None,
            )
            getattr(bundle, kind.lower()).append(ts)

    # Some EDF may have RR annotations; pyedflib does not always expose them.
    # If you standardize RR export via separate channel label, add it here.

    f.close()
    return bundle


# ---- Polar .hrm (RRI) ----
def _load_polar_hrm(path: str) -> DataBundle:
    """
    Polar HRM is plain text with sections like [Params], [HRData].
    RR data often in milliseconds as the 2nd column of HRData (tab-separated).
    """
    rri = []
    device = "Polar"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    in_hr = False
    for ln in lines:
        if ln.startswith("["):
            in_hr = ln.lower() == "[hrdata]".lower()
            continue
        if in_hr and ln:
            parts = ln.split("\t")
            # Common structure: col0 = HR (bpm), col1 = RR (ms) or similar
            if len(parts) >= 2:
                try:
                    rri.append(float(parts[1]))
                except Exception:
                    pass

    return DataBundle(
        rri_ms=rri,
        meta={"format": "hrm", "device": device},
        source=SourceInfo(path, ".hrm", device=device),
    )


# ---- Garmin .fit (RRI) ----
def _load_garmin_fit(path: str) -> DataBundle:
    fit = FitFile(path)
    rri = []
    device = "Garmin"
    # According to FIT profile, rr_interval is in seconds; can be a list per record
    for record in fit.get_messages("record"):
        for data in record:
            if data.name == "rr_interval" and data.value is not None:
                vals = data.value if isinstance(data.value, list) else [data.value]
                rri.extend([float(v) * 1000.0 for v in vals])
    return DataBundle(
        rri_ms=rri,
        meta={"format": "fit", "device": device},
        source=SourceInfo(path, ".fit", device=device),
    )


# ---- Suunto .sml (XML; RRI/PPI) ----
def _load_suunto_sml(path: str) -> DataBundle:
    tree = ET.parse(path)
    root = tree.getroot()
    device = "Suunto"
    rri = []
    ppi = []

    # Heuristics: look for elements/attributes containing RR/PPI samples
    # Common patterns:
    #   <Sample RR="802"/>  or  <RR>802</RR>  or  <RRSamples>...</RRSamples>
    for elem in root.iter():
        tag = elem.tag.lower().split("}")[-1]  # handle namespaces
        if tag in ("sample", "rr", "rri", "ppi"):
            # attribute?
            if "RR" in elem.attrib:
                try:
                    rri.append(float(elem.attrib["RR"]))
                except Exception:
                    pass
            if "PPI" in elem.attrib:
                try:
                    ppi.append(float(elem.attrib["PPI"]))
                except Exception:
                    pass
            # text?
            try:
                val = float(elem.text) if elem.text not in (None, "") else None
                if val is not None:
                    if tag in ("rr", "rri"):
                        rri.append(val)
                    elif tag == "ppi":
                        ppi.append(val)
            except Exception:
                pass

    return DataBundle(
        rri_ms=rri,
        ppi_ms=ppi,
        meta={"format": "sml", "device": device},
        source=SourceInfo(path, ".sml", device=device),
    )


# ---- Movesense (JSON; RRI/PPI) ----
def _load_movesense_json(path: str) -> DataBundle:
    device = "Movesense"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    # Try common keys
    rri = _coerce_float_list(data.get("RR", [])) or _coerce_float_list(
        data.get("RRI", [])
    )
    ppi = _coerce_float_list(data.get("PPI", []))
    # Alternative nested structures:
    if not rri and "samples" in data and isinstance(data["samples"], list):
        for s in data["samples"]:
            if isinstance(s, dict):
                if "rr" in s:
                    rri.extend(_coerce_float_list([s["rr"]]))
                if "ppi" in s:
                    ppi.extend(_coerce_float_list([s["ppi"]]))

    return DataBundle(
        rri_ms=rri,
        ppi_ms=ppi,
        meta={"format": "json", "device": device},
        source=SourceInfo(path, ".json", device=device),
    )


# ---- Biopac / AcqKnowledge (.acq) (ECG/PPG/RESP waveforms) ----
def _load_biopac_acq(path: str) -> DataBundle:
    datafile = bioread.read_file(path)
    bundle = DataBundle(
        meta={"format": "acq", "device": "Biopac"},
        source=SourceInfo(path, ".acq", device="Biopac"),
    )

    for ch in datafile.channels:
        label = ch.name or ch.channel_name or ""
        kind = _label_to_kind(label)
        if kind:
            # bioread provides sampling rate in samples_per_second
            fs = float(ch.samples_per_second) if ch.samples_per_second else None
            if not fs or fs <= 0:
                # try to infer
                fs = 1000.0
            ts = TimeSeries(
                name=kind, data=np.asarray(ch.data, dtype=float), fs=fs, units=ch.units
            )
            getattr(bundle, kind.lower()).append(ts)

    return bundle


# =========================
# Convenience Functions
# =========================


def load_and_preprocess(
    path: str, preprocessing_params: Optional[Dict] = None
) -> DataBundle:
    """
    Convenience function to load and automatically preprocess RR intervals.

    Args:
        path: Path to data file
        preprocessing_params: Parameters for preprocessing

    Returns:
        DataBundle with preprocessing applied
    """
    return load_rr_file(
        path, auto_preprocess=True, preprocessing_params=preprocessing_params
    )


def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats.

    Returns:
        List of supported file extensions
    """
    return [".csv", ".txt", ".edf", ".hrm", ".fit", ".sml", ".json", ".acq"]


def check_dependencies() -> Dict[str, bool]:
    """
    Check which optional dependencies are available.

    Returns:
        Dictionary mapping dependency names to availability
    """
    return {
        "pandas": pd is not None,
        "pyedflib": pyedflib is not None,
        "fitparse": FitFile is not None,
        "xml.etree.ElementTree": ET is not None,
        "bioread": bioread is not None,
        "scipy": True,  # scipy is required, not optional
        "numpy": True,  # numpy is required, not optional
    }


def _load_txt_manual(path: str) -> DataBundle:
    """
    Load TXT file using manual parsing (no pandas required).
    Handles simple text files with one number per line or space-separated values.
    """
    rri_values = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Skip potential header line if it contains non-numeric characters
            if line_num == 1:
                try:
                    # Try to parse the first line as numbers
                    test_values = (
                        line.replace(",", " ")
                        .replace(";", " ")
                        .replace("\t", " ")
                        .split()
                    )
                    # If any value can't be converted to float, assume it's a header
                    for val in test_values:
                        float(val)
                except ValueError:
                    # First line is likely a header, skip it
                    continue

            # Parse the line for numeric values
            try:
                # Handle various separators: comma, semicolon, tab, space
                values = (
                    line.replace(",", " ").replace(";", " ").replace("\t", " ").split()
                )
                for value_str in values:
                    value_str = value_str.strip()
                    if value_str:  # Skip empty strings
                        try:
                            value = float(value_str)
                            if value > 0:  # Only accept positive values for RRI
                                rri_values.append(value)
                        except ValueError:
                            # Skip invalid numeric values
                            continue
            except Exception:
                # Skip problematic lines
                continue

    if not rri_values:
        raise ValueError(f"No valid RR intervals found in {path}")

    # Create DataBundle
    source = SourceInfo(path=path, filetype=".txt", device="Unknown")

    bundle = DataBundle(rri_ms=rri_values, source=source)
    bundle.meta = {}

    return bundle


# =========================
# Helpers
# =========================


def _require(obj, name: str, how_to_install: str):
    if obj is None:
        raise ImportError(
            f"Missing dependency '{name}'. Install with: {how_to_install}"
        )


def _coerce_float_list(x) -> List[float]:
    out = []
    for v in x:
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out.append(float(v))
        except Exception:
            continue
    return out


def _label_to_kind(label: str) -> Optional[str]:
    """
    Heuristic EDF/ACQ channel labeling → 'ECG' | 'PPG' | 'RESP' | None
    """
    if not label:
        return None
    s = label.strip().lower()
    # ECG markers
    if any(
        k in s for k in ["ecg", "ii", "v1", "lead", "leads", "mlII", "ml-ii", "ml ii"]
    ):
        return "ECG"
    # PPG markers
    if any(k in s for k in ["ppg", "pulse", "pleth", "oxim"]):
        return "PPG"
    # Resp markers
    if any(
        k in s for k in ["resp", "respiration", "thor", "abdo", "breath", "airflow"]
    ):
        return "RESP"
    return None


# ---- NeuroKit2 QRS / peak detection ----
def _extract_intervals_from_waveforms(bundle: DataBundle) -> DataBundle:
    """
    If ECG or PPG waveforms exist but no RRI/PPI are provided,
    run peak detection to derive intervals (ms).
    """
    try:
        import neurokit2 as nk
    except ImportError:
        warnings.warn(
            "neurokit2 not installed; cannot derive RRI/PPI from waveforms. Run: pip install neurokit2"
        )
        return bundle

    # ECG → RRI
    if bundle.ecg and not bundle.rri_ms:
        ecg0 = bundle.ecg[0]  # take first ECG channel
        try:
            # Check if signal is long enough for processing
            if len(ecg0.data) < 100:  # Minimum reasonable signal length
                warnings.warn(
                    "ECG signal too short for peak detection; skipping RRI extraction"
                )
                return bundle

            # Check sampling rate is reasonable
            if ecg0.fs <= 0 or ecg0.fs > 10000:
                warnings.warn(
                    f"Invalid sampling rate {ecg0.fs} Hz for ECG; skipping RRI extraction"
                )
                return bundle

            signals, info = nk.ecg_process(ecg0.data, sampling_rate=ecg0.fs)
            rpeaks = info.get("ECG_R_Peaks", [])
            if len(rpeaks) > 1:
                rr = np.diff(rpeaks) / ecg0.fs * 1000.0  # convert samples → ms
                bundle.rri_ms = rr.tolist()
                bundle.meta["ecg_peak_method"] = "neurokit2.ecg_process"
        except Exception as e:
            warnings.warn(f"ECG peak detection failed: {e}")

    # PPG → PPI
    if bundle.ppg and not bundle.ppi_ms:
        ppg0 = bundle.ppg[0]
        try:
            # Check if signal is long enough for processing
            if len(ppg0.data) < 100:  # Minimum reasonable signal length
                warnings.warn(
                    "PPG signal too short for peak detection; skipping PPI extraction"
                )
                return bundle

            # Check sampling rate is reasonable
            if ppg0.fs <= 0 or ppg0.fs > 10000:
                warnings.warn(
                    f"Invalid sampling rate {ppg0.fs} Hz for PPG; skipping PPI extraction"
                )
                return bundle

            signals, info = nk.ppg_process(ppg0.data, sampling_rate=ppg0.fs)
            peaks = info.get("PPG_Peaks", [])
            if peaks is not None and len(peaks) > 1:
                ppi = np.diff(peaks) / ppg0.fs * 1000.0
                bundle.ppi_ms = ppi.tolist()
                bundle.meta["ppg_peak_method"] = "neurokit2.ppg_process"

            else:
                warnings.warn("No PPG peaks detected; cannot derive PPI.")
        except Exception as e:
            warnings.warn(f"PPG peak detection failed: {e}")
    return bundle
