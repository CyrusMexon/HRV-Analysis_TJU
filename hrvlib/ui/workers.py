from PyQt6 import QtCore
from PyQt6.QtCore import QObject, pyqtSignal
import traceback


class WorkerSignals(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(int)  # percent
    finished = pyqtSignal(object)  # result (HRVAnalysisResults)
    error = pyqtSignal(str)


class PipelineWorker(QtCore.QObject):
    """
    QObject wrapper to run the HRV analysis pipeline without blocking the UI.
    """

    def __init__(self, bundle, analysis_parameters):
        super().__init__()
        self.signals = WorkerSignals()
        self.bundle = bundle
        self.analysis_parameters = analysis_parameters

    def run(self):
        self.signals.started.emit()
        try:
            from hrvlib.pipeline import create_unified_pipeline

            self.signals.progress.emit(10)

            # Build pipeline
            pipeline = create_unified_pipeline(
                bundle=self.bundle, analysis_config=self.analysis_parameters
            )

            # Run pipeline, pass analysis_window separately
            results = pipeline.run_all(
                analysis_window=self.analysis_parameters.get("analysis_window")
            )

            self.signals.progress.emit(100)
            self.signals.finished.emit(results)

        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Pipeline execution failed: {str(e)}\n\nFull traceback:\n{tb}"
            self.signals.error.emit(error_msg)
