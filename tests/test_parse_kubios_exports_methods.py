from pathlib import Path

import pytest

from tools.parse_kubios_exports import extract_frequency_section, parse_kubios_metric_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOOTHNESS_EXPORT = (
    PROJECT_ROOT
    / "validation"
    / "kubios_subset"
    / "smoothness_priors_pilot"
    / "kubios_exports"
    / "clean_high_agreement"
    / "CH001__nsr020_segment_000"
    / "CH001__nsr020_segment_000_hrv.txt"
)


@pytest.mark.skipif(not SMOOTHNESS_EXPORT.exists(), reason="Kubios pilot export is not available")
def test_kubios_frequency_table_parses_fft_and_ar_columns() -> None:
    text = SMOOTHNESS_EXPORT.read_text(encoding="utf-8-sig", errors="replace")
    section = extract_frequency_section(text, SMOOTHNESS_EXPORT)

    vlf = parse_kubios_metric_columns(section, "VLF", SMOOTHNESS_EXPORT)
    lf_hf = parse_kubios_metric_columns(section, "LF/HF", SMOOTHNESS_EXPORT)

    assert vlf["fft"] == pytest.approx(57.5923)
    assert vlf["ar"] == pytest.approx(92.2051)
    assert lf_hf["fft"] == pytest.approx(4.6462)
    assert lf_hf["ar"] == pytest.approx(4.8406)
