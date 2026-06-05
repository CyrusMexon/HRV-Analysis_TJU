import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import validate_freq_domain_neurokit2 as validator


class TestValidationWelchDetrendMode(unittest.TestCase):
    def test_default_welch_detrend_mode_is_current(self):
        with patch.object(
            sys,
            "argv",
            [
                "validate_freq_domain_neurokit2.py",
                "data/sample_rr.csv",
                "--run-name",
                "test_default",
            ],
        ):
            args = validator.parse_args()

        self.assertEqual(args.welch_detrend_mode, "current")
        self.assertEqual(validator.hrv_settings(args)["welch_detrend_mode"], "current")
        self.assertEqual(
            validator.neurokit2_settings(args)["welch_detrend_mode"],
            "current",
        )

    def test_run_info_records_selected_welch_detrend_mode(self):
        args = self._args("global_then_none")
        with tempfile.TemporaryDirectory() as tmp:
            run_info = validator.build_run_info(
                args=args,
                input_paths=[Path("sample.csv")],
                run_dir=Path(tmp),
                failures=[],
                file_events=[],
            )

        self.assertEqual(run_info["welch_detrend_mode"], "global_then_none")
        self.assertEqual(
            run_info["hrv_freq_domain_settings"]["welch_detrend_mode"],
            "global_then_none",
        )
        self.assertEqual(
            run_info["neurokit2_settings"]["welch_detrend_mode"],
            "global_then_none",
        )

    def test_global_then_none_mode_runs_on_small_rr_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rr_path = tmp_path / "rr.csv"
            self._write_rr_csv(rr_path)
            run_root = tmp_path / "runs"

            with patch.object(validator, "RUNS_ROOT", run_root), patch.object(
                sys,
                "argv",
                [
                    "validate_freq_domain_neurokit2.py",
                    str(rr_path),
                    "--run-name",
                    "global_then_none_smoke",
                    "--detrend-method",
                    "linear",
                    "--welch-detrend-mode",
                    "global_then_none",
                ],
            ):
                validator.main()

            run_dir = run_root / "global_then_none_smoke"
            run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(run_info["welch_detrend_mode"], "global_then_none")

            with (run_dir / "freq_domain_neurokit2_validation.csv").open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertTrue(rows)
            self.assertEqual(rows[0]["welch_detrend_mode"], "global_then_none")

    @staticmethod
    def _args(welch_detrend_mode):
        return type(
            "Args",
            (),
            {
                "run_name": "test_run",
                "purpose": "test",
                "interpolation_rate": 4.0,
                "window_type": "hann",
                "segment_length": 120.0,
                "overlap_ratio": 0.75,
                "detrend_method": "linear",
                "detrend_lambda": 500.0,
                "ar_order": 16,
                "enable_diagnostics": False,
                "experimental_native_welch_nfft_multiplier": 1.0,
                "neurokit_interpolation_method": "cubic",
                "welch_detrend_mode": welch_detrend_mode,
            },
        )()

    @staticmethod
    def _write_rr_csv(path):
        rng = np.random.default_rng(42)
        beats = np.arange(420)
        rr = 800.0 + 30.0 * np.sin(2 * np.pi * beats / 80.0) + rng.normal(0, 3, len(beats))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rr_ms"])
            for value in rr:
                writer.writerow([f"{value:.6f}"])


if __name__ == "__main__":
    unittest.main()
