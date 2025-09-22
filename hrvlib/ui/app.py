"""
app.py - Main HRV Analysis Application
Implements GUI using PyQt6 with modular widgets (widgets.py)
Meets SRS requirements v2.1.1
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6 import QtWidgets
from PyQt6.QtCore import QThread

from hrvlib.data_handler import load_rr_file, DataBundle
from hrvlib.pipeline import create_unified_pipeline, HRVAnalysisResults
from hrvlib.ui.widgets import (
    MetaPanel,
    ResultsPanel,
    AnalysisParametersWidget,
    QualityAssessmentWidget,
    SignalViewerWidget,
    ExportDialog,
    HelpDialog,
    SessionManager,
)
from hrvlib.ui.workers import PipelineWorker


class HRVMainWindow(QtWidgets.QMainWindow):
    """Main HRV Analysis Window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HRV Analysis Software v2.1.1 - SRS Compliant")
        self.resize(1400, 900)

        # Data storage
        self.current_bundle: Optional[DataBundle] = None
        self.current_results: Optional[HRVAnalysisResults] = None
        self.analysis_parameters: Dict[str, Any] = {}
        self.edit_history: list = []

        # UI setup
        self.setup_ui()
        self.setup_menu_bar()
        self.statusBar().showMessage("Ready")

    def setup_ui(self):
        """Setup central layout"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left control panel
        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setMinimumWidth(350)
        self.control_panel.setMaximumWidth(400)
        control_layout = QtWidgets.QVBoxLayout(self.control_panel)

        # Metadata
        meta_group = QtWidgets.QGroupBox("File Information")
        meta_layout = QtWidgets.QVBoxLayout(meta_group)
        self.meta_panel = MetaPanel()
        meta_layout.addWidget(self.meta_panel)

        # Parameters
        params_group = QtWidgets.QGroupBox("Analysis Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_scroll = QtWidgets.QScrollArea()
        self.params_widget = AnalysisParametersWidget()
        params_scroll.setWidget(self.params_widget)
        params_scroll.setWidgetResizable(True)
        params_scroll.setMaximumHeight(350)
        params_layout.addWidget(params_scroll)

        # Quality assessment
        self.quality_widget = QualityAssessmentWidget()

        # Control buttons
        button_group = QtWidgets.QGroupBox("Analysis Control")
        button_layout = QtWidgets.QVBoxLayout(button_group)
        self.analyze_btn = QtWidgets.QPushButton("🔍 Run Analysis")
        self.analyze_btn.setEnabled(False)
        self.export_btn = QtWidgets.QPushButton("📄 Export Results")
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.export_btn)

        control_layout.addWidget(meta_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(self.quality_widget)
        control_layout.addWidget(button_group)
        control_layout.addStretch()

        # Right tabbed interface
        self.tab_widget = QtWidgets.QTabWidget()
        self.signal_viewer = SignalViewerWidget()
        self.results_panel = ResultsPanel()
        self.tab_widget.addTab(self.signal_viewer, "📊 Signal Viewer")
        self.tab_widget.addTab(self.results_panel, "📋 Metrics Summary")

        # Assemble
        main_layout.addWidget(self.control_panel, 0)
        main_layout.addWidget(self.tab_widget, 1)

        # Connect signals
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.export_btn.clicked.connect(self.export_results)

    def setup_menu_bar(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")
        open_action = file_menu.addAction("Open File...")
        open_action.triggered.connect(self.open_file)
        save_session_action = file_menu.addAction("Save Session...")
        save_session_action.triggered.connect(self.save_session)
        load_session_action = file_menu.addAction("Load Session...")
        load_session_action.triggered.connect(self.load_session)
        file_menu.addSeparator()
        export_action = file_menu.addAction("Export Results...")
        export_action.triggered.connect(self.export_results)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Edit
        edit_menu = menubar.addMenu("&Edit")
        undo_action = edit_menu.addAction("Undo Edit")
        undo_action.triggered.connect(self.undo_edit)
        reset_action = edit_menu.addAction("Reset All Edits")
        reset_action.triggered.connect(self.reset_edits)

        # Analysis
        analysis_menu = menubar.addMenu("&Analysis")
        run_action = analysis_menu.addAction("Run Analysis")
        run_action.triggered.connect(self.run_analysis)

        # View
        view_menu = menubar.addMenu("&View")
        toggle_control_action = view_menu.addAction("Toggle Control Panel")
        toggle_control_action.triggered.connect(self.toggle_control_panel)

        # Help
        help_menu = menubar.addMenu("&Help")
        help_action = help_menu.addAction("Help Manual")
        help_action.triggered.connect(self.show_help)

    def open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open HRV File",
            str(Path.cwd()),
            "HRV Data (*.csv *.txt *.edf *.hrm *.fit *.sml *.json);;All Files (*)",
        )
        if not file_path:
            return
        try:
            bundle = load_rr_file(file_path)
            self.current_bundle = bundle

            # Use the enhanced metadata update method
            self.meta_panel.update_meta_from_bundle(bundle)

            self.analyze_btn.setEnabled(True)
            self.statusBar().showMessage(f"Loaded file: {Path(file_path).name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def run_analysis(self):
        if not self.current_bundle:
            return
        self.analysis_parameters = self.params_widget.get_parameters()
        self.thread = QThread()

        # Create worker with bundle and parameters
        self.worker = PipelineWorker(self.current_bundle, self.analysis_parameters)

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Connect error signal for better error handling
        self.worker.error.connect(self.on_analysis_error)

        self.thread.start()
        self.statusBar().showMessage("Running analysis...")

    def on_analysis_error(self, error_message):
        """Handle analysis errors"""
        QtWidgets.QMessageBox.critical(
            self, "Analysis Error", f"Analysis failed:\n{error_message}"
        )
        self.statusBar().showMessage("Analysis failed")

    def on_analysis_finished(self, results: HRVAnalysisResults):
        """Enhanced version with debugging information"""
        self.current_results = results

        # Debug: Print what we received
        print("=== Analysis Results Debug ===")
        print(f"Time domain: {results.time_domain is not None}")
        print(f"Frequency domain: {results.frequency_domain is not None}")
        print(f"Nonlinear: {results.nonlinear is not None}")
        print(f"Preprocessing stats: {results.preprocessing_stats}")
        print(f"Quality assessment: {results.quality_assessment}")
        print(f"Warnings: {results.warnings}")

        # Update displays
        self.signal_viewer.update_display(self.current_bundle, results)

        # Debug the results dict
        results_dict = results.to_dict()
        print(f"Results dict keys: {list(results_dict.keys())}")

        try:
            self.results_panel.show_results(results_dict)
        except Exception as e:
            print(f"Results panel error: {e}")

        try:
            self.quality_widget.update_quality_assessment(results)
        except Exception as e:
            print(f"Quality widget error: {e}")

        self.export_btn.setEnabled(True)
        self.statusBar().showMessage("Analysis complete")
        print("=== End Debug ===")

    def export_results(self):
        if not self.current_results:
            return
        dialog = ExportDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            settings = dialog.get_export_settings()
            QtWidgets.QMessageBox.information(
                self, "Export", f"Export settings:\n{settings}"
            )

    def save_session(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Session", str(Path.cwd()), "Session Files (*.json)"
        )
        if file_path:
            SessionManager.save_session(
                file_path,
                self.analysis_parameters,
                self.edit_history,
                str(self.current_bundle.source.path),
                self.current_results,
            )
            self.statusBar().showMessage(f"Session saved: {file_path}")

    def load_session(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Session", str(Path.cwd()), "Session Files (*.json)"
        )
        if file_path:
            session = SessionManager.load_session(file_path)
            if session:
                self.params_widget.set_parameters(
                    session.get("analysis_parameters", {})
                )
                self.statusBar().showMessage(f"Session loaded: {file_path}")

    def undo_edit(self):
        QtWidgets.QMessageBox.information(self, "Undo", "Undo not yet implemented")

    def reset_edits(self):
        QtWidgets.QMessageBox.information(
            self, "Reset", "Reset edits not yet implemented"
        )

    def toggle_control_panel(self):
        self.control_panel.setVisible(not self.control_panel.isVisible())

    def show_help(self):
        dialog = HelpDialog(self)
        dialog.exec()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = HRVMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
