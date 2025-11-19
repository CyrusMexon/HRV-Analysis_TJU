from PyQt6 import QtCore
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import traceback


class PipelineWorker(QObject):
    """
    QObject-based worker to run the HRV analysis pipeline without blocking the UI.
    Uses QObject instead of QRunnable to properly support moveToThread().
    """

    # Signals
    started = pyqtSignal()
    progress = pyqtSignal(int)  # percent
    finished = pyqtSignal(object)  # result (HRVAnalysisResults)
    error = pyqtSignal(str)

    def __init__(self, bundle, analysis_parameters):
        super().__init__()
        self.bundle = bundle
        self.analysis_parameters = analysis_parameters

    def run(self):
        """Full analysis execution method"""
        self.started.emit()
        try:
            from hrvlib.pipeline import create_unified_pipeline
            import numpy as np

            # Report progress
            self.progress.emit(10)

            print("=== Full Pipeline Execution ===")
            print(f"Bundle type: {type(self.bundle)}")
            print(f"RRI length: {len(self.bundle.rri_ms)}")
            print(f"GUI Parameters: {self.analysis_parameters}")

            # Convert parameters with explicit type checking
            try:
                analysis_config = self._convert_gui_params_to_analysis_config()
                preprocessing_config = (
                    self._convert_gui_params_to_preprocessing_config()
                )

                print(f"Analysis Config: {analysis_config}")
                print(f"Preprocessing Config: {preprocessing_config}")
            except Exception as e:
                print(f"Parameter conversion failed: {e}")
                self.error.emit(f"Parameter conversion failed: {e}")
                return

            self.progress.emit(30)

            # Create pipeline
            try:
                pipeline = create_unified_pipeline(
                    bundle=self.bundle,
                    preprocessing_config=preprocessing_config,
                    analysis_config=analysis_config,
                )
                print("Pipeline creation successful")
            except Exception as e:
                print(f"Pipeline creation failed: {e}")
                self.error.emit(f"Pipeline creation failed: {e}")
                return

            self.progress.emit(50)

            # Run complete analysis
            try:
                print("Running full analysis pipeline...")
                results = pipeline.run_all()
                print("Full analysis completed successfully")

                # Debug what was computed
                print(f"Time domain computed: {results.time_domain is not None}")
                print(
                    f"Frequency domain computed: {results.frequency_domain is not None}"
                )
                print(f"Nonlinear computed: {results.nonlinear is not None}")
                print(f"Respiratory computed: {results.respiratory is not None}")

                self.progress.emit(100)
                self.finished.emit(results)

            except Exception as e:
                print(f"Full analysis failed: {e}")
                import traceback

                tb = traceback.format_exc()
                self.error.emit(
                    f"Analysis pipeline failed: {e}\n\nFull traceback:\n{tb}"
                )

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            error_msg = f"Pipeline execution failed: {str(e)}\n\nFull traceback:\n{tb}"
            print(f"Pipeline Error: {error_msg}")
            self.error.emit(error_msg)

    def _convert_gui_params_to_preprocessing_config(self):
        """Convert GUI preprocessing parameters to preprocessing config format"""
        preprocessing_params = self.analysis_parameters.get("preprocessing", {})

        # CRITICAL: Check if artifact correction is enabled
        artifact_correction_enabled = preprocessing_params.get(
            "artifact_correction", True
        )

        # Get the correction threshold from GUI (should be 0.05 = 5%)
        gui_threshold_fraction = float(
            preprocessing_params.get("correction_threshold", 0.05)
        )

        # Convert to Kubios-style thresholds:
        # - GUI shows percentage (5% = 0.05)
        # - Preprocessing function expects relative threshold (0.2 = 20% change)
        # - Kubios typically uses 20% for ectopic detection
        ectopic_threshold = 0.20  # Fixed at 20% like Kubios default

        # Map GUI parameters to the actual preprocess_rri function parameters
        return {
            "artifact_correction_enabled": artifact_correction_enabled,  # NEW: Pass the checkbox state
            "threshold_low": 300.0,  # Min RR interval (ms) - Kubios default
            "threshold_high": 2000.0,  # Max RR interval (ms) - Kubios default
            "ectopic_threshold": ectopic_threshold,  # 20% relative change for ectopic detection
            "correction_method": str(
                preprocessing_params.get("interpolation_method", "cubic_spline")
            ),
            "noise_detection": True,  # Always enable noise detection
        }

    def _convert_gui_params_to_analysis_config(self):
        """Convert GUI parameters to analysis config format"""
        # Extract analysis window if present
        analysis_window = None
        if "analysis_window" in self.analysis_parameters:
            aw = self.analysis_parameters["analysis_window"]
            start = int(aw.get("start_sec", 0))  # Ensure int
            duration = int(aw.get("duration_sec", 300))  # Ensure int
            if duration > 0:  # Only set window if duration is valid
                analysis_window = (
                    float(start),
                    float(start + duration),
                )  # Convert to float tuples

        # Convert frequency domain parameters
        freq_params = self.analysis_parameters.get("frequency_domain", {})

        # Convert detrending parameters
        detrend_params = self.analysis_parameters.get("detrending", {})

        # CRITICAL FIX: Map smoothness_priors to linear until SP is implemented
        detrend_method = str(detrend_params.get("method", "linear"))
        if detrend_method == "smoothness_priors":
            print(
                "WARNING: Smoothness Priors not implemented yet, using 'linear' detrending"
            )
            detrend_method = "linear"

        # Validate detrend_method
        if detrend_method not in ["linear", "constant"]:
            print(f"WARNING: Invalid detrend method '{detrend_method}', using 'linear'")
            detrend_method = "linear"

        return {
            "time_domain": {
                "enabled": True,
                "analysis_window": analysis_window,
            },
            "frequency_domain": {
                "enabled": True,
                "sampling_rate": 4.0,  # Standard for HRV
                "detrend_method": str(
                    detrend_params.get("method", "linear")
                ),  # Ensure string
                "window_type": str(
                    freq_params.get("window_function", "hann")
                ),  # Ensure string
                "segment_length": 120.0,
                "overlap_ratio": 0.75,
                "analysis_window": analysis_window,
            },
            "nonlinear": {
                "enabled": True,
                "include_mse": True,
                "include_dfa": True,
                "include_rqa": True,
                "mse_scales": 10,
                "analysis_window": analysis_window,
            },
            "respiratory": {
                "enabled": True,
            },
        }
