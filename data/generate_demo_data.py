"""
HRV Test Data Generator
Creates synthetic RR interval data with known characteristics for testing HRV analysis software
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta


class HRVTestDataGenerator:
    """Generate synthetic HRV test data with controlled characteristics"""

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_normal_sinus_rhythm(self, duration_minutes=10, base_hr=70, hrv_std=50):
        """Generate normal sinus rhythm with realistic HRV"""
        duration_sec = duration_minutes * 60

        # Base RR interval (ms)
        base_rr = 60000 / base_hr

        # Generate time series with respiratory sinus arrhythmia
        n_points = int(duration_sec * 4)  # ~4 Hz equivalent sampling
        time = np.linspace(0, duration_sec, n_points)

        # Respiratory component (0.25 Hz)
        resp_freq = 0.25  # breathing rate
        resp_component = 30 * np.sin(2 * np.pi * resp_freq * time)

        # LF component (0.1 Hz)
        lf_component = 20 * np.sin(2 * np.pi * 0.1 * time + np.random.random())

        # VLF component (0.04 Hz)
        vlf_component = 15 * np.sin(2 * np.pi * 0.04 * time + np.random.random())

        # Random noise
        noise = np.random.normal(0, hrv_std / 4, n_points)

        # Combine components
        rr_intervals = base_rr + resp_component + lf_component + vlf_component + noise

        # Ensure positive values
        rr_intervals = np.maximum(rr_intervals, 300)  # Min 300ms

        return rr_intervals.tolist()

    def add_artifacts(self, rr_intervals, artifact_percentage=5):
        """Add artifacts to clean RR interval data"""
        rr_array = np.array(rr_intervals)
        n_artifacts = int(len(rr_array) * artifact_percentage / 100)

        artifact_indices = np.random.choice(len(rr_array), n_artifacts, replace=False)

        for idx in artifact_indices:
            artifact_type = np.random.choice(["missed_beat", "extra_beat", "noise"])

            if artifact_type == "missed_beat":
                # Double the RR interval (missed beat)
                rr_array[idx] *= 2.0
            elif artifact_type == "extra_beat":
                # Halve the RR interval (extra detection)
                rr_array[idx] *= 0.5
            else:  # noise
                # Add significant noise
                rr_array[idx] += np.random.normal(0, 200)

        # Ensure positive values
        rr_array = np.maximum(rr_array, 200)

        return rr_array.tolist(), artifact_indices.tolist()

    def generate_atrial_fibrillation(self, duration_minutes=5, base_hr=80):
        """Generate irregular rhythm simulating atrial fibrillation"""
        duration_sec = duration_minutes * 60
        n_beats = int(duration_sec * base_hr / 60)

        # Highly irregular RR intervals
        base_rr = 60000 / base_hr
        rr_intervals = []

        for i in range(n_beats):
            # Very irregular intervals (high randomness)
            rr = base_rr + np.random.normal(0, 150)  # High variability
            rr = max(rr, 200)  # Minimum 200ms
            rr_intervals.append(rr)

        return rr_intervals

    def generate_test_suite(self, output_dir="test_data"):
        """Generate complete test suite with various scenarios"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        test_cases = []

        # 1. Normal healthy adult (clean data)
        print("Generating normal healthy adult data...")
        normal_rri = self.generate_normal_sinus_rhythm(duration_minutes=10, base_hr=70)
        self.save_csv(
            normal_rri,
            output_path / "normal_healthy_10min.csv",
            metadata={"subject": "healthy_adult", "age": 30, "artifacts": "none"},
        )
        test_cases.append(
            {
                "file": "normal_healthy_10min.csv",
                "expected": "normal sinus rhythm, good quality",
            }
        )

        # 2. Normal with 2% artifacts (good quality threshold)
        print("Generating data with 2% artifacts...")
        normal_2pct, artifacts_2pct = self.add_artifacts(
            normal_rri, artifact_percentage=2
        )
        self.save_csv(
            normal_2pct,
            output_path / "normal_2pct_artifacts.csv",
            metadata={
                "subject": "healthy_adult",
                "artifacts": "2%",
                "artifact_indices": artifacts_2pct,
            },
        )
        test_cases.append(
            {
                "file": "normal_2pct_artifacts.csv",
                "expected": "good quality, minimal artifacts",
            }
        )

        # 3. Normal with 6% artifacts (poor quality threshold)
        print("Generating data with 6% artifacts...")
        normal_6pct, artifacts_6pct = self.add_artifacts(
            normal_rri, artifact_percentage=6
        )
        self.save_csv(
            normal_6pct,
            output_path / "normal_6pct_artifacts.csv",
            metadata={
                "subject": "healthy_adult",
                "artifacts": "6%",
                "artifact_indices": artifacts_6pct,
            },
        )
        test_cases.append(
            {
                "file": "normal_6pct_artifacts.csv",
                "expected": "poor quality warning, high artifacts",
            }
        )

        # 4. Short recording (2 minutes)
        print("Generating short recording...")
        short_rri = self.generate_normal_sinus_rhythm(duration_minutes=2, base_hr=75)
        self.save_csv(
            short_rri,
            output_path / "short_recording_2min.csv",
            metadata={"subject": "healthy_adult", "duration": "2min"},
        )
        test_cases.append(
            {
                "file": "short_recording_2min.csv",
                "expected": "duration warning, limited frequency analysis",
            }
        )

        # 5. High HRV athlete
        print("Generating athlete data...")
        athlete_rri = self.generate_normal_sinus_rhythm(
            duration_minutes=10, base_hr=50, hrv_std=80
        )
        self.save_csv(
            athlete_rri,
            output_path / "athlete_high_hrv.csv",
            metadata={"subject": "athlete", "fitness_level": "high"},
        )
        test_cases.append(
            {"file": "athlete_high_hrv.csv", "expected": "high HRV, low heart rate"}
        )

        # 6. Low HRV (stress/aging)
        print("Generating low HRV data...")
        low_hrv_rri = self.generate_normal_sinus_rhythm(
            duration_minutes=10, base_hr=85, hrv_std=20
        )
        self.save_csv(
            low_hrv_rri,
            output_path / "low_hrv_stress.csv",
            metadata={"subject": "stressed_individual", "hrv_level": "low"},
        )
        test_cases.append(
            {
                "file": "low_hrv_stress.csv",
                "expected": "low HRV metrics, elevated heart rate",
            }
        )

        # 7. Atrial fibrillation
        print("Generating atrial fibrillation data...")
        afib_rri = self.generate_atrial_fibrillation(duration_minutes=5, base_hr=90)
        self.save_csv(
            afib_rri,
            output_path / "atrial_fibrillation.csv",
            metadata={"subject": "patient", "rhythm": "atrial_fibrillation"},
        )
        test_cases.append(
            {
                "file": "atrial_fibrillation.csv",
                "expected": "irregular rhythm detection, quality warnings",
            }
        )

        # 8. Very long recording for long-term analysis
        print("Generating long recording...")
        long_rri = self.generate_normal_sinus_rhythm(duration_minutes=60, base_hr=72)
        self.save_csv(
            long_rri,
            output_path / "long_recording_60min.csv",
            metadata={"subject": "healthy_adult", "duration": "60min"},
        )
        test_cases.append(
            {
                "file": "long_recording_60min.csv",
                "expected": "all metrics reliable, complete analysis",
            }
        )

        # Save test case documentation
        test_info = {
            "generated_on": datetime.now().isoformat(),
            "generator_version": "1.0",
            "test_cases": test_cases,
            "notes": "Synthetic HRV data for testing analysis software functionality",
        }

        with open(output_path / "test_cases_info.json", "w") as f:
            json.dump(test_info, f, indent=2)

        print(f"\nGenerated {len(test_cases)} test files in {output_path}")
        print("Test suite includes:")
        for case in test_cases:
            print(f"  - {case['file']}: {case['expected']}")

        return output_path

    def save_csv(self, rr_intervals, filepath, metadata=None):
        """Save RR intervals to CSV with optional metadata"""
        df = pd.DataFrame({"RR_ms": rr_intervals})

        if metadata:
            # Add metadata as header comments
            with open(filepath, "w") as f:
                f.write("# HRV Test Data\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("\n")

            # Append CSV data
            df.to_csv(filepath, mode="a", index=False)
        else:
            df.to_csv(filepath, index=False)

    def save_txt(self, rr_intervals, filepath):
        """Save RR intervals to TXT format (one value per line)"""
        with open(filepath, "w") as f:
            for rr in rr_intervals:
                f.write(f"{rr:.3f}\n")


if __name__ == "__main__":
    # Generate test data
    generator = HRVTestDataGenerator()
    output_dir = generator.generate_test_suite()

    print(f"\n✅ Test data generated successfully in: {output_dir.absolute()}")
    print("\nTo test your HRV software:")
    print("1. Load each test file")
    print("2. Run analysis with default parameters")
    print("3. Check that warnings/quality assessments match expectations")
    print("4. Verify artifact correction on files with known artifact percentages")
    print("5. Test export functionality with different output formats")
