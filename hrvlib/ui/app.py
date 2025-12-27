"""
app.py - Main HRV Analysis Application
Implements GUI using PyQt6 with modular widgets (widgets.py)
Meets SRS requirements v2.1.1 with manual beat editing (FR-10 to FR-14)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QThread

from hrvlib.data_handler import load_rr_file, DataBundle
from hrvlib.pipeline import create_unified_pipeline, HRVAnalysisResults
from hrvlib.beat_editor import BeatEditor, EditAction, InterpolationMethod
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
    """Main HRV Analysis Window with Manual Beat Editing Support"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HRV Studio v1.0")
        self.resize(1400, 900)

        # Data storage
        self.current_bundle: Optional[DataBundle] = None
        self.current_results: Optional[HRVAnalysisResults] = None
        self.analysis_parameters: Dict[str, Any] = {}
        self.edit_history: list = []
        self.has_manual_edits: bool = False  # Track manual edits

        # UI setup
        self.setup_ui()
        self.setup_menu_bar()
        self.statusBar().showMessage("Ready - Manual editing enabled")

    def setup_ui(self):
        """Setup central layout"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left control panel with scroll area
        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setMinimumWidth(430)
        self.control_panel.setMaximumWidth(600)

        # Create a scroll area for the entire control panel
        control_scroll = QtWidgets.QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Widget that will go inside the scroll area
        control_content = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_content)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # Metadata
        meta_group = QtWidgets.QGroupBox("File Information")
        meta_layout = QtWidgets.QVBoxLayout(meta_group)
        meta_layout.setSpacing(3)
        meta_layout.setContentsMargins(8, 10, 5, 5)
        self.meta_panel = MetaPanel()
        meta_layout.addWidget(self.meta_panel)
        meta_group.setMinimumHeight(180)  # Ensure adequate space for labels
        meta_group.setMaximumHeight(220)  # Increased from 150

        # Parameters
        params_group = QtWidgets.QGroupBox("Analysis Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_scroll = QtWidgets.QScrollArea()
        self.params_widget = AnalysisParametersWidget()
        params_scroll.setWidget(self.params_widget)
        params_scroll.setWidgetResizable(True)
        params_scroll.setMinimumHeight(300)  # Reduced from 450
        params_scroll.setMaximumHeight(400)  # Add maximum height
        params_scroll.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        params_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )  # Disable horizontal scrollbar
        params_layout.addWidget(params_scroll)
        params_layout.setContentsMargins(5, 5, 5, 5)

        # Quality assessment with compact layout
        quality_group = QtWidgets.QGroupBox("Quality Metrics")
        quality_layout = QtWidgets.QVBoxLayout(quality_group)
        self.quality_widget = QualityAssessmentWidget()
        quality_layout.addWidget(self.quality_widget)
        quality_group.setMaximumHeight(280)  # Limit quality widget height

        # Control buttons
        button_group = QtWidgets.QGroupBox("Analysis Control")
        button_layout = QtWidgets.QVBoxLayout(button_group)
        button_layout.setSpacing(8)
        button_layout.setContentsMargins(5, 10, 5, 5)
        self.analyze_btn = QtWidgets.QPushButton("🔬 Run Analysis")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(32)  # Reduced from 40
        self.analyze_btn.setMaximumHeight(35)
        self.export_btn = QtWidgets.QPushButton("📄 Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.setMinimumHeight(32)  # Reduced from 40
        self.export_btn.setMaximumHeight(35)
        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.export_btn)
        button_group.setMaximumHeight(110)  # Reduced from 140

        control_layout.addWidget(meta_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(quality_group)
        control_layout.addWidget(button_group)
        control_layout.addStretch()

        # Set the control content as the scroll area's widget
        control_scroll.setWidget(control_content)

        # Add scroll area to control panel
        control_panel_layout = QtWidgets.QVBoxLayout(self.control_panel)
        control_panel_layout.setContentsMargins(0, 0, 0, 0)
        control_panel_layout.addWidget(control_scroll)

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

        # Connect beat editing signals
        self.signal_viewer.beat_edited.connect(self.on_beat_edited)
        self.signal_viewer.quality_updated.connect(self.on_quality_updated)
        self.signal_viewer.reanalysis_requested.connect(self.on_reanalysis_requested)

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

        # Edit (UPDATED with editing features)
        edit_menu = menubar.addMenu("&Edit")
        undo_action = edit_menu.addAction("Undo Beat Edit")
        undo_action.triggered.connect(self.undo_beat_edit)
        reset_action = edit_menu.addAction("Reset All Beat Edits")
        reset_action.triggered.connect(self.reset_beat_edits)
        edit_menu.addSeparator()
        reanalyze_action = edit_menu.addAction("Reanalyze with Edits")
        reanalyze_action.triggered.connect(self.on_reanalysis_requested)

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

            # RESET: Clear any previous edits when loading new file
            self.has_manual_edits = False
            self.edit_history.clear()

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

        # Update status based on whether we have manual edits
        if self.has_manual_edits:
            self.statusBar().showMessage(
                "Running analysis with manually edited data..."
            )
        else:
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

        # Update status based on editing state
        if self.has_manual_edits:
            self.statusBar().showMessage("Analysis complete with edited data")
        else:
            self.statusBar().showMessage("Analysis complete")
        print("=== End Debug ===")

    # NEW METHODS for handling manual editing (SRS FR-10 to FR-14)

    def on_beat_edited(self, action: str, edit_info: dict):
        """Handle beat editing signals from SignalViewerWidget"""
        self.has_manual_edits = True

        # Log the edit
        edit_record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "edit_info": edit_info,
        }
        self.edit_history.append(edit_record)

        # Update status based on action
        if action == "reset":
            self.has_manual_edits = False
            self.edit_history.clear()
            self.statusBar().showMessage(
                "All manual edits reset - Original data restored"
            )
        elif action == "undo":
            # Count remaining edits (excluding undos)
            remaining_edits = len(
                [e for e in self.edit_history if e["action"] not in ["undo", "reset"]]
            )
            if remaining_edits == 0:
                self.has_manual_edits = False
            self.statusBar().showMessage(
                f"Edit undone - {remaining_edits} edits remaining"
            )
        else:
            edit_count = len(
                [e for e in self.edit_history if e["action"] not in ["undo", "reset"]]
            )
            self.statusBar().showMessage(
                f"Beat {action} applied - Total edits: {edit_count}"
            )

        # Enable export if we have results
        if self.current_results:
            self.export_btn.setEnabled(True)

    def on_quality_updated(self, corrected_percentage: float):
        """Handle quality updates from editing (SRS FR-14)"""
        if corrected_percentage > 5.0:
            # Show warning in status bar for extended time
            self.statusBar().showMessage(
                f"⚠️ WARNING: {corrected_percentage:.1f}% beats corrected - Exceeds 5% threshold (SRS FR-14)",
                15000,  # Show for 15 seconds
            )

            # Update window title to indicate quality issue
            self.setWindowTitle("HRV Studio v1.0 - ⚠️ QUALITY WARNING: >5% corrected")

        elif corrected_percentage > 2.0:
            self.statusBar().showMessage(
                f"Quality: {corrected_percentage:.1f}% beats corrected - Monitor quality"
            )
            self.setWindowTitle("HRV Studio v1.0 - Manual editing active")
        elif corrected_percentage > 0:
            self.statusBar().showMessage(
                f"Quality: {corrected_percentage:.1f}% beats corrected - Good quality"
            )
            self.setWindowTitle("HRV Studio v1.0 - Manual editing active")
        else:
            self.setWindowTitle("HRV Studio v1.0")

    def on_reanalysis_requested(self):
        """Handle reanalysis requests after editing"""
        if not self.current_bundle or not self.has_manual_edits:
            QtWidgets.QMessageBox.information(
                self,
                "Reanalysis",
                "No manual edits have been made. Use 'Run Analysis' for standard analysis.",
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Reanalyze with Edited Data",
            "Do you want to rerun the HRV analysis using the manually edited RR intervals?\n\n"
            f"Current edits: {len([e for e in self.edit_history if e['action'] not in ['undo', 'reset']])} modifications\n\n"
            "This will compute new HRV metrics based on your corrections.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Get edited RR intervals from signal viewer
            edited_rr = self.signal_viewer.get_edited_rr_intervals()
            if edited_rr is not None:
                # Update bundle with edited data
                self.current_bundle.rri_ms = edited_rr.tolist()

                # Run analysis with edited data
                self.run_analysis()

    def undo_beat_edit(self):
        """Undo last beat edit via menu"""
        if hasattr(self.signal_viewer, "_undo_last_edit"):
            self.signal_viewer._undo_last_edit()
        else:
            QtWidgets.QMessageBox.information(
                self, "Undo", "No edits to undo or editing not available"
            )

    def reset_beat_edits(self):
        """Reset all beat edits via menu"""
        if hasattr(self.signal_viewer, "_reset_all_edits"):
            self.signal_viewer._reset_all_edits()
        else:
            QtWidgets.QMessageBox.information(
                self, "Reset", "No edits to reset or editing not available"
            )

    def export_results(self):
        """Enhanced export results method with audit trail (SRS FR-13, FR-30)"""
        if not self.current_results:
            QtWidgets.QMessageBox.warning(
                self, "Export Warning", "No analysis results available for export."
            )
            return

        if not self.current_bundle:
            QtWidgets.QMessageBox.warning(
                self, "Export Warning", "No data bundle available for export."
            )
            return

        # Import the export system
        try:
            from hrvlib.export_system import HRVExporter
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                "Export system not found. Please ensure export_system.py is in hrvlib directory.",
            )
            return

        # Create export dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Export Results")
        dialog.setModal(True)
        dialog.resize(450, 500)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Title with editing info
        if self.has_manual_edits:
            edit_count = len(
                [e for e in self.edit_history if e["action"] not in ["undo", "reset"]]
            )
            title_text = (
                f"Export HRV Analysis Results\n({edit_count} manual edits applied)"
            )
            color = "#ff6b35"  # Orange for edited data
        else:
            title_text = "Export HRV Analysis Results"
            color = "#2c3e50"  # Default blue

        title_label = QtWidgets.QLabel(title_text)
        title_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; padding: 10px; color: {color};"
        )
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Format selection group
        format_group = QtWidgets.QGroupBox("Export Formats (Select at least one)")
        format_layout = QtWidgets.QVBoxLayout(format_group)

        pdf_cb = QtWidgets.QCheckBox("PDF Report (Comprehensive)")
        pdf_cb.setToolTip(
            "Multi-page PDF with plots, metrics tables, quality assessment"
        )
        pdf_cb.setChecked(True)

        csv_cb = QtWidgets.QCheckBox("CSV Metrics (Standard)")
        csv_cb.setToolTip("Comma-separated values file with all computed metrics")

        spss_cb = QtWidgets.QCheckBox("SPSS-compatible CSV")
        spss_cb.setToolTip("CSV formatted for SPSS with proper variable naming")

        audit_cb = QtWidgets.QCheckBox("Audit Trail (JSON)")
        audit_cb.setToolTip("Complete log of analysis parameters and processing steps")
        if self.has_manual_edits:
            audit_cb.setChecked(True)  # Auto-select if edits were made

        format_layout.addWidget(pdf_cb)
        format_layout.addWidget(csv_cb)
        format_layout.addWidget(spss_cb)
        format_layout.addWidget(audit_cb)

        layout.addWidget(format_group)

        # Content options group
        content_group = QtWidgets.QGroupBox("Content Options")
        content_layout = QtWidgets.QVBoxLayout(content_group)

        plots_cb = QtWidgets.QCheckBox("Include plots in PDF")
        plots_cb.setChecked(True)

        preprocessing_cb = QtWidgets.QCheckBox("Include preprocessing details")
        preprocessing_cb.setChecked(True)

        quality_cb = QtWidgets.QCheckBox("Include quality assessment")
        quality_cb.setChecked(True)

        params_cb = QtWidgets.QCheckBox("Include analysis parameters")
        params_cb.setChecked(True)

        warnings_cb = QtWidgets.QCheckBox("Include warnings and recommendations")
        warnings_cb.setChecked(True)

        # Manual editing options
        editing_cb = QtWidgets.QCheckBox("Include manual editing details")
        editing_cb.setChecked(self.has_manual_edits)
        editing_cb.setEnabled(self.has_manual_edits)
        if self.has_manual_edits:
            editing_cb.setToolTip(
                "Include complete audit trail of manual beat corrections (SRS FR-13)"
            )
        else:
            editing_cb.setToolTip("No manual edits to include")

        content_layout.addWidget(plots_cb)
        content_layout.addWidget(preprocessing_cb)
        content_layout.addWidget(quality_cb)
        content_layout.addWidget(params_cb)
        content_layout.addWidget(warnings_cb)

        layout.addWidget(content_group)

        # Options group
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QFormLayout(options_group)

        decimal_spin = QtWidgets.QSpinBox()
        decimal_spin.setRange(1, 6)
        decimal_spin.setValue(3)

        timestamp_cb = QtWidgets.QCheckBox("Include timestamp in filenames")
        timestamp_cb.setChecked(True)

        options_layout.addRow("Decimal Places:", decimal_spin)
        options_layout.addRow("", timestamp_cb)

        layout.addWidget(options_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        export_btn = QtWidgets.QPushButton("Export")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        export_btn.setDefault(True)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            """
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """
        )

        button_layout.addStretch()
        button_layout.addWidget(export_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        def perform_export():
            """Perform the actual export operation with manual editing support"""
            # Validate format selection
            formats = {
                "pdf": pdf_cb.isChecked(),
                "csv": csv_cb.isChecked(),
                "spss": spss_cb.isChecked(),
                "audit_trail": audit_cb.isChecked(),
            }

            if not any(formats.values()):
                QtWidgets.QMessageBox.warning(
                    dialog, "Warning", "Please select at least one export format."
                )
                return

            # Get base file path
            default_name = "HRV_Analysis_Results"
            if self.current_bundle.source and hasattr(
                self.current_bundle.source, "path"
            ):
                source_path = Path(self.current_bundle.source.path)
                default_name = f"HRV_{source_path.stem}"

            # Add edited suffix if manual edits were applied
            if self.has_manual_edits:
                default_name += "_edited"

            base_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dialog, "Export HRV Results", default_name, "All Files (*)"
            )

            if not base_path:
                return  # User cancelled

            # Remove any extension from base_path
            base_path = str(Path(base_path).with_suffix(""))

            try:
                # Show progress dialog
                progress = QtWidgets.QProgressDialog(
                    "Exporting results...", "Cancel", 0, 100, dialog
                )
                progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
                progress.setAutoClose(True)
                progress.setAutoReset(True)
                progress.show()

                # Process Qt events to show progress dialog
                QtWidgets.QApplication.processEvents()

                # Prepare export settings with manual editing info
                export_settings = {
                    "formats": formats,
                    "content": {
                        "include_plots": plots_cb.isChecked(),
                        "include_preprocessing": preprocessing_cb.isChecked(),
                        "include_quality": quality_cb.isChecked(),
                        "include_parameters": params_cb.isChecked(),
                        "include_warnings": warnings_cb.isChecked(),
                        "include_editing": editing_cb.isChecked(),
                    },
                    "options": {
                        "decimal_places": decimal_spin.value(),
                        "include_timestamp": timestamp_cb.isChecked(),
                    },
                }

                progress.setValue(20)
                QtWidgets.QApplication.processEvents()

                # ENHANCED: Analysis parameters with manual editing information
                enhanced_parameters = self.analysis_parameters.copy()
                enhanced_parameters["manual_editing"] = {
                    "edits_applied": self.has_manual_edits,
                    "edit_count": len(
                        [
                            e
                            for e in self.edit_history
                            if e["action"] not in ["undo", "reset"]
                        ]
                    ),
                    "edit_history_summary": (
                        self.edit_history[-20:] if self.edit_history else []
                    ),
                    "srs_compliance": "FR-10 to FR-14 - Manual editing with audit trail",
                }

                # Get detailed audit trail from signal viewer
                if self.signal_viewer and hasattr(
                    self.signal_viewer, "get_audit_trail"
                ):
                    enhanced_parameters["detailed_audit_trail"] = (
                        self.signal_viewer.get_audit_trail()
                    )

                progress.setValue(50)
                QtWidgets.QApplication.processEvents()

                # Perform export
                exporter = HRVExporter(
                    self.current_bundle, self.current_results, self.analysis_parameters
                )

                exported_files = exporter.export_all(export_settings, base_path)

                progress.setValue(100)
                QtWidgets.QApplication.processEvents()

                # Show success message
                if exported_files:
                    file_list = "\n".join(
                        [
                            f"• {fmt}: {Path(path).name}"
                            for fmt, path in exported_files.items()
                        ]
                    )

                    # Add manual editing info to success message
                    edit_info = ""
                    if self.has_manual_edits:
                        edit_count = len(
                            [
                                e
                                for e in self.edit_history
                                if e["action"] not in ["undo", "reset"]
                            ]
                        )
                        edit_info = f"\n\n✏️ Manual Editing: {edit_count} edits applied and logged in audit trail (SRS FR-13)"

                    success_msg = (
                        f"Export completed successfully!\n\n"
                        f"Files created:\n{file_list}\n\n"
                        f"Location: {Path(base_path).parent}\n\n"
                        f"Compliance: SRS requirements FR-24 to FR-30 with manual editing audit trail"
                    )

                    QtWidgets.QMessageBox.information(
                        dialog, "Export Successful", success_msg
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        dialog,
                        "Export Warning",
                        "No files were exported. Please check your selections.",
                    )

                progress.close()
                dialog.accept()

            except Exception as e:
                progress.close()
                error_msg = f"Export failed with error:\n\n{str(e)}\n\nPlease check file permissions and try again."
                QtWidgets.QMessageBox.critical(dialog, "Export Error", error_msg)

        def validate_and_enable_export():
            """Enable/disable export button based on format selection"""
            formats_selected = any(
                [
                    pdf_cb.isChecked(),
                    csv_cb.isChecked(),
                    spss_cb.isChecked(),
                    audit_cb.isChecked(),
                ]
            )
            export_btn.setEnabled(formats_selected)

            # Enable/disable plot option based on PDF selection
            plots_cb.setEnabled(pdf_cb.isChecked())
            if not pdf_cb.isChecked():
                plots_cb.setChecked(False)

        # Connect signals
        export_btn.clicked.connect(perform_export)
        cancel_btn.clicked.connect(dialog.reject)

        # Connect format checkboxes to validation
        pdf_cb.toggled.connect(validate_and_enable_export)
        csv_cb.toggled.connect(validate_and_enable_export)
        spss_cb.toggled.connect(validate_and_enable_export)
        audit_cb.toggled.connect(validate_and_enable_export)

        # Initial validation
        validate_and_enable_export()

        # Show dialog
        dialog.exec()

    def save_session(self):
        """Enhanced session save with full state preservation"""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Session", str(Path.cwd()), "Session Files (*.json)"
        )

        if not file_path:
            return

        try:
            # Get detailed audit trail from signal viewer
            signal_viewer_data = None
            if hasattr(self.signal_viewer, "get_audit_trail"):
                signal_viewer_data = self.signal_viewer.get_audit_trail()

            success = SessionManager.save_session(
                file_path,
                self.analysis_parameters,
                self.edit_history,
                (
                    str(self.current_bundle.source.path)
                    if self.current_bundle and self.current_bundle.source
                    else ""
                ),
                self.current_results,
                self.current_bundle,  # Include current data state
                signal_viewer_data,
            )

            if success:
                edit_info = (
                    f" (with {len([e for e in self.edit_history if e['action'] not in ['undo', 'reset']])} manual edits)"
                    if self.has_manual_edits
                    else ""
                )
                self.statusBar().showMessage(
                    f"Session saved{edit_info}: {Path(file_path).name}"
                )

                QtWidgets.QMessageBox.information(
                    self,
                    "Session Saved",
                    f"Session saved successfully!\n\n"
                    f"File: {Path(file_path).name}\n"
                    f"Manual edits: {'Yes' if self.has_manual_edits else 'No'}\n"
                    f"Parameters: Preserved\n"
                    f"Data state: Preserved",
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Save Failed",
                    "Failed to save session. Check file permissions.",
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Error saving session:\n{str(e)}"
            )

    def load_session(self):
        """Enhanced session load with full state restoration"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Session", str(Path.cwd()), "Session Files (*.json)"
        )

        if not file_path:
            return

        try:
            session = SessionManager.load_session(file_path)
            if not session:
                QtWidgets.QMessageBox.warning(
                    self, "Load Failed", "Invalid or corrupted session file."
                )
                return

            # Show session info dialog
            self._show_session_info_dialog(session, file_path)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Load Error", f"Error loading session:\n{str(e)}"
            )

    def _show_session_info_dialog(self, session: Dict, file_path: str):
        """Show session information and options for loading"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Load Session")
        dialog.setModal(True)
        dialog.resize(500, 400)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Session info display
        info_text = self._format_session_info(session)
        info_label = QtWidgets.QTextEdit()
        info_label.setPlainText(info_text)
        info_label.setReadOnly(True)
        info_label.setMaximumHeight(250)

        layout.addWidget(QtWidgets.QLabel("Session Information:"))
        layout.addWidget(info_label)

        # Loading options
        options_group = QtWidgets.QGroupBox("What to load:")
        options_layout = QtWidgets.QVBoxLayout(options_group)

        load_params_cb = QtWidgets.QCheckBox("Analysis Parameters")
        load_params_cb.setChecked(True)

        load_data_cb = QtWidgets.QCheckBox("Data and Manual Edits")
        load_data_cb.setChecked("rr_data" in session)
        load_data_cb.setEnabled("rr_data" in session)

        load_file_cb = QtWidgets.QCheckBox("Reload Original File (if available)")
        load_file_cb.setChecked(False)

        options_layout.addWidget(load_params_cb)
        options_layout.addWidget(load_data_cb)
        options_layout.addWidget(load_file_cb)

        layout.addWidget(options_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        load_btn = QtWidgets.QPushButton("Load Session")
        load_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        cancel_btn = QtWidgets.QPushButton("Cancel")

        button_layout.addStretch()
        button_layout.addWidget(load_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        # Connect signals
        def perform_load():
            try:
                self._apply_session_data(
                    session,
                    load_params_cb.isChecked(),
                    load_data_cb.isChecked(),
                    load_file_cb.isChecked(),
                )
                dialog.accept()

                self.statusBar().showMessage(f"Session loaded: {Path(file_path).name}")

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    dialog, "Load Error", f"Failed to apply session data:\n{str(e)}"
                )

        load_btn.clicked.connect(perform_load)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def _format_session_info(self, session: Dict) -> str:
        """Format session information for display"""
        info_lines = []

        # Basic info
        session_info = session.get("session_info", {})
        info_lines.append(f"Version: {session_info.get('version', 'Unknown')}")
        info_lines.append(f"Created: {session_info.get('timestamp', 'Unknown')}")
        info_lines.append("")

        # Source file
        source_file = session.get("source_file", {})
        info_lines.append(f"Original File: {source_file.get('path', 'Unknown')}")
        info_lines.append("")

        # Manual editing info
        manual_editing = session.get("manual_editing", {})
        has_edits = manual_editing.get("has_edits", False)
        edit_count = manual_editing.get("edit_count", 0)

        info_lines.append(f"Manual Edits: {'Yes' if has_edits else 'No'}")
        if has_edits:
            info_lines.append(f"Edit Count: {edit_count}")
            last_edit = manual_editing.get("last_edit_timestamp")
            if last_edit:
                info_lines.append(f"Last Edit: {last_edit}")
        info_lines.append("")

        # Data info
        rr_data = session.get("rr_data", {})
        if rr_data:
            info_lines.append(f"RR Intervals: {rr_data.get('sample_count', 0)} samples")
            info_lines.append(f"Duration: {rr_data.get('duration_s', 0):.1f} seconds")
        else:
            info_lines.append("RR Data: Not saved in session")
        info_lines.append("")

        # Analysis parameters
        params = session.get("analysis_parameters", {})
        if params:
            info_lines.append("Analysis Parameters:")
            for section, settings in params.items():
                if isinstance(settings, dict):
                    info_lines.append(f"  {section}: {len(settings)} settings")
                else:
                    info_lines.append(f"  {section}: {settings}")

        return "\n".join(info_lines)

    def _apply_session_data(
        self, session: Dict, load_params: bool, load_data: bool, reload_file: bool
    ):
        """Apply loaded session data to restore application state"""

        # Load analysis parameters
        if load_params and "analysis_parameters" in session:
            self.analysis_parameters = session["analysis_parameters"].copy()
            self.params_widget.set_parameters(self.analysis_parameters)
            self.statusBar().showMessage("Analysis parameters restored")

        # Load data and manual edits
        if load_data and "rr_data" in session:
            try:
                # Create a data bundle from session data
                from hrvlib.data_handler import DataBundle, SourceInfo

                rr_data = session["rr_data"]
                rr_intervals = rr_data.get("rr_intervals_ms", [])

                if rr_intervals:
                    # Create source info
                    source_path = session.get("source_file", {}).get(
                        "path", "Session Data"
                    )
                    source_info = SourceInfo(
                        path=source_path,
                        filetype="session",
                        device="restored_from_session",
                        acquisition_date=session.get("session_info", {}).get(
                            "timestamp", ""
                        ),
                    )

                    # Create bundle
                    bundle = DataBundle(source=source_info)
                    bundle.rri_ms = rr_intervals
                    bundle.meta = rr_data.get("metadata", {})
                    bundle.meta["restored_from_session"] = True
                    bundle.meta["session_edit_count"] = session.get(
                        "manual_editing", {}
                    ).get("edit_count", 0)

                    # Set as current bundle
                    self.current_bundle = bundle

                    # Restore edit history
                    self.edit_history = session.get("manual_editing", {}).get(
                        "edit_history", []
                    )
                    self.has_manual_edits = session.get("manual_editing", {}).get(
                        "has_edits", False
                    )

                    # Update displays
                    self.meta_panel.update_meta_from_bundle(bundle)
                    self.analyze_btn.setEnabled(True)

                    # Update status
                    edit_info = (
                        f" with {len(self.edit_history)} manual edits"
                        if self.has_manual_edits
                        else ""
                    )
                    self.statusBar().showMessage(
                        f"Data restored from session{edit_info}"
                    )

            except Exception as e:
                raise Exception(f"Failed to restore data: {str(e)}")

        # Reload original file if requested
        if reload_file:
            source_path = session.get("source_file", {}).get("path")
            if source_path and Path(source_path).exists():
                try:
                    bundle = load_rr_file(source_path)
                    self.current_bundle = bundle

                    # Clear edit history since we're loading fresh file
                    self.has_manual_edits = False
                    self.edit_history.clear()

                    self.meta_panel.update_meta_from_bundle(bundle)
                    self.analyze_btn.setEnabled(True)
                    self.statusBar().showMessage(
                        f"Original file reloaded: {Path(source_path).name}"
                    )

                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "File Load Warning",
                        f"Could not reload original file:\n{str(e)}\n\nUsing session data instead.",
                    )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "File Not Found",
                    f"Original file not found:\n{source_path}\n\nUsing session data instead.",
                )

    def undo_edit(self):
        self.undo_beat_edit()

    def reset_edits(self):
        self.reset_beat_edits()

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
