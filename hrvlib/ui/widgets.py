"""
Enhanced widgets.py
Comprehensive widget collection for HRV analysis GUI
Separated from app.py for better code organization and maintainability
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSignal, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from hrvlib.data_handler import DataBundle
from hrvlib.pipeline import HRVAnalysisResults
from hrvlib.ui.plots import plot_pipeline_results


class MetaPanel(QtWidgets.QFrame):
    """Shows source metadata (file, device, format, acquisition_date, fs)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QFormLayout(self)
        self.labels = {}

        # Create labels for metadata fields
        for key in (
            "path",
            "filetype",
            "device",
            "acquisition_date",
            "fs",
            "duration",
            "samples",
        ):
            lbl = QtWidgets.QLabel("-")
            lbl.setStyleSheet(
                "QLabel { color: #ffffff; font-weight: bold; background-color: gray; padding: 2px; }"
            )
            self.layout.addRow(key.replace("_", " ").title() + ":", lbl)
            self.labels[key] = lbl

        # Style the frame
        self.setFrameStyle(QtWidgets.QFrame.Shape.StyledPanel)
        self.setMaximumHeight(200)

    def update_meta(self, source_dict):
        """Update metadata display with source information"""
        if not source_dict:
            for lbl in self.labels.values():
                lbl.setText("-")
            return

        # Handle SourceInfo objects
        if hasattr(source_dict, "__dict__"):
            source_dict = vars(source_dict)

        for k, lbl in self.labels.items():
            v = source_dict.get(k)
            if v is not None:
                # Format different types appropriately
                if k == "path":
                    # Show just the filename, not the full path
                    lbl.setText(str(Path(v).name) if v else "-")
                elif k == "duration":
                    # Format duration nicely
                    if isinstance(v, (int, float)):
                        lbl.setText(f"{v:.1f} s")
                    else:
                        lbl.setText(str(v))
                elif k == "fs":
                    # Format sampling frequency
                    if isinstance(v, (int, float)):
                        lbl.setText(f"{v:.1f} Hz")
                    else:
                        lbl.setText(str(v))
                elif k == "samples":
                    # Format sample count
                    if isinstance(v, (int, float)):
                        lbl.setText(f"{int(v):,}")
                    else:
                        lbl.setText(str(v))
                elif k == "acquisition_date":
                    # Format date/time
                    if isinstance(v, str):
                        lbl.setText(v)
                    else:
                        lbl.setText(str(v))
                else:
                    # Default formatting
                    lbl.setText(str(v))
            else:
                lbl.setText("-")

    def update_meta_from_bundle(self, bundle: DataBundle):
        """Update metadata from DataBundle with additional calculated info"""
        meta_dict = {}

        if bundle.source:
            meta_dict.update(vars(bundle.source))

        # Add calculated metadata
        if bundle.rri_ms:
            rri_array = np.array(bundle.rri_ms)
            meta_dict["samples"] = len(rri_array)
            meta_dict["duration"] = np.sum(rri_array) / 1000.0  # seconds

            # Add basic stats
            meta_dict["mean_rr"] = np.mean(rri_array)
            meta_dict["std_rr"] = np.std(rri_array)

        # Add fs from meta if available
        if "fs" in bundle.meta:
            meta_dict["fs"] = bundle.meta["fs"]
        elif bundle.ecg or bundle.ppg:
            # Get fs from first waveform if available
            if bundle.ecg:
                meta_dict["fs"] = bundle.ecg[0].fs
            elif bundle.ppg:
                meta_dict["fs"] = bundle.ppg[0].fs

        # Add format info
        meta_dict["filetype"] = bundle.meta.get(
            "format", meta_dict.get("filetype", "unknown")
        )

        self.update_meta(meta_dict)


class ResultsPanel(QtWidgets.QTextEdit):
    """Enhanced read-only area to show metrics & warnings with better formatting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)

        # Set monospace font for better alignment
        font = QtGui.QFont("Consolas", 10)
        if not font.exactMatch():
            font = QtGui.QFont("Courier", 10)
        self.setFont(font)

        # Style the text area
        self.setStyleSheet(
            """
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                color: #000000; /* Text color */
            }
        """
        )

    def show_results(self, results: dict):
        """Display results with enhanced formatting"""
        lines = []

        # Header with timestamp
        lines.append("=" * 60)
        lines.append("HRV ANALYSIS RESULTS")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Time-domain metrics
        td = results.get("time_domain")
        if td:
            lines.append("TIME DOMAIN METRICS")
            lines.append("-" * 30)
            for k, v in td.items():
                if isinstance(v, (int, float)):
                    lines.append(f"{k:15}: {v:8.3f}")
                else:
                    lines.append(f"{k:15}: {v}")
            lines.append("")

        # Frequency-domain metrics
        fd = results.get("frequency_domain") or results.get("freq_domain")
        if fd:
            lines.append("FREQUENCY DOMAIN METRICS")
            lines.append("-" * 30)
            for k, v in fd.items():
                if isinstance(v, (int, float)):
                    lines.append(f"{k:15}: {v:8.3f}")
                else:
                    lines.append(f"{k:15}: {v}")
            lines.append("")

        # Nonlinear metrics
        nl = results.get("nonlinear")
        if nl:
            lines.append("NONLINEAR METRICS")
            lines.append("-" * 30)
            for k, v in nl.items():
                if isinstance(v, (int, float)):
                    lines.append(f"{k:15}: {v:8.3f}")
                else:
                    lines.append(f"{k:15}: {v}")
            lines.append("")

        # Respiratory metrics
        resp = results.get("respiratory")
        if resp:
            lines.append("RESPIRATORY METRICS")
            lines.append("-" * 30)
            for k, v in resp.items():
                if isinstance(v, (int, float)):
                    lines.append(f"{k:15}: {v:8.3f}")
                elif isinstance(v, dict):
                    lines.append(f"{k:15}: {v}")
                else:
                    lines.append(f"{k:15}: {v}")
            lines.append("")

        # Quality assessment
        qa = results.get("quality_assessment")
        if qa:
            lines.append("QUALITY ASSESSMENT")
            lines.append("-" * 30)
            for k, v in qa.items():
                lines.append(f"{k:15}: {v}")
            lines.append("")

        # Preprocessing statistics
        ps = results.get("preprocessing_stats")
        if ps:
            lines.append("PREPROCESSING SUMMARY")
            lines.append("-" * 30)
            for k, v in ps.items():
                lines.append(f"{k:15}: {v}")
            lines.append("")

        # Warnings
        warnings = results.get("warnings", [])
        if warnings:
            lines.append("WARNINGS & ALERTS")
            lines.append("-" * 30)
            for w in warnings:
                lines.append(f"⚠ {w}")
            lines.append("")

        self.setPlainText("\n".join(lines))


class AnalysisParametersWidget(QtWidgets.QWidget):
    """Widget for configuring analysis parameters according to SRS requirements"""

    parameters_changed = pyqtSignal()  # Emit when parameters change

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Analysis window group
        window_group = QtWidgets.QGroupBox("Analysis Window")
        window_layout = QtWidgets.QFormLayout(window_group)

        self.analysis_window_start = QtWidgets.QSpinBox()
        self.analysis_window_start.setRange(0, 3600)
        self.analysis_window_start.setValue(0)
        self.analysis_window_start.setSuffix(" s")
        self.analysis_window_start.valueChanged.connect(self.parameters_changed.emit)

        self.analysis_window_duration = QtWidgets.QSpinBox()
        self.analysis_window_duration.setRange(
            10, 3600
        )  # SRS: minimum 10s with warnings
        self.analysis_window_duration.setValue(300)
        self.analysis_window_duration.setSuffix(" s")
        self.analysis_window_duration.valueChanged.connect(self.parameters_changed.emit)
        self.analysis_window_duration.valueChanged.connect(self._check_window_duration)

        window_layout.addRow("Start Time:", self.analysis_window_start)
        window_layout.addRow("Duration:", self.analysis_window_duration)

        # Preprocessing group
        preprocess_group = QtWidgets.QGroupBox("Preprocessing Settings")
        preprocess_layout = QtWidgets.QFormLayout(preprocess_group)

        self.artifact_correction = QtWidgets.QCheckBox(
            "Enable automatic artifact correction"
        )
        self.artifact_correction.setChecked(True)
        self.artifact_correction.toggled.connect(self.parameters_changed.emit)

        self.interpolation_method = QtWidgets.QComboBox()
        self.interpolation_method.addItems(["cubic_spline", "linear"])
        self.interpolation_method.currentTextChanged.connect(
            self.parameters_changed.emit
        )

        self.correction_threshold = QtWidgets.QDoubleSpinBox()
        self.correction_threshold.setRange(0.01, 0.10)  # 1% to 10% range per SRS
        self.correction_threshold.setValue(0.05)  # SRS: default 5%
        self.correction_threshold.setDecimals(2)
        self.correction_threshold.setSuffix("%")
        self.correction_threshold.valueChanged.connect(self.parameters_changed.emit)

        # Update tooltip to match SRS FR-14
        self.correction_threshold.setToolTip(
            "Configurable threshold for quality warnings when % corrected beats exceeds this value (SRS FR-14). "
            "Kubios standard artifact detection (20% ectopic threshold) is used regardless of this setting."
        )

        preprocess_layout.addRow("Artifact Correction:", self.artifact_correction)
        preprocess_layout.addRow("Interpolation:", self.interpolation_method)
        preprocess_layout.addRow(
            "Quality Warning Threshold:", self.correction_threshold
        )  # Updated label per SRS

        # Detrending group
        detrend_group = QtWidgets.QGroupBox("Detrending")
        detrend_layout = QtWidgets.QFormLayout(detrend_group)

        self.detrending_method = QtWidgets.QComboBox()
        self.detrending_method.addItems(["smoothness_priors", "linear", "none"])
        self.detrending_method.currentTextChanged.connect(self.parameters_changed.emit)

        self.detrending_lambda = QtWidgets.QDoubleSpinBox()
        self.detrending_lambda.setRange(1, 10000)
        self.detrending_lambda.setValue(500)
        self.detrending_lambda.valueChanged.connect(self.parameters_changed.emit)

        detrend_layout.addRow("Method:", self.detrending_method)
        detrend_layout.addRow("Lambda:", self.detrending_lambda)

        # Frequency domain group
        freq_group = QtWidgets.QGroupBox("Frequency Analysis")
        freq_layout = QtWidgets.QFormLayout(freq_group)

        self.psd_method = QtWidgets.QComboBox()
        self.psd_method.addItems(["welch", "lomb_scargle", "fft"])
        self.psd_method.currentTextChanged.connect(self.parameters_changed.emit)

        self.window_function = QtWidgets.QComboBox()
        self.window_function.addItems(["hann", "hamming", "blackman", "bartlett"])
        self.window_function.currentTextChanged.connect(self.parameters_changed.emit)

        freq_layout.addRow("PSD Method:", self.psd_method)
        freq_layout.addRow("Window:", self.window_function)

        # Add all groups to main layout
        layout.addWidget(window_group)
        layout.addWidget(preprocess_group)
        layout.addWidget(detrend_group)
        layout.addWidget(freq_group)
        layout.addStretch()

    def _check_window_duration(self):
        """Check for short window duration and show warning"""
        duration = self.analysis_window_duration.value()
        if duration < 30:
            QtWidgets.QToolTip.showText(
                self.analysis_window_duration.mapToGlobal(QtCore.QPoint(0, 0)),
                "Warning: Analysis windows shorter than 30s may have limited physiological relevance",
            )

    def get_parameters(self) -> Dict[str, Any]:
        """Return current parameter settings as dictionary"""
        return {
            "analysis_window": {
                "start_sec": self.analysis_window_start.value(),
                "duration_sec": self.analysis_window_duration.value(),
            },
            "preprocessing": {
                "artifact_correction": self.artifact_correction.isChecked(),
                "interpolation_method": self.interpolation_method.currentText(),
                "correction_threshold": self.correction_threshold.value()
                / 100.0,  # Convert to fraction
            },
            "detrending": {
                "method": self.detrending_method.currentText(),
                "lambda": self.detrending_lambda.value(),
            },
            "frequency_domain": {
                "psd_method": self.psd_method.currentText(),
                "window_function": self.window_function.currentText(),
            },
        }

    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from dictionary"""
        if "analysis_window" in params:
            aw = params["analysis_window"]
            self.analysis_window_start.setValue(aw.get("start_sec", 0))
            self.analysis_window_duration.setValue(aw.get("duration_sec", 300))

        if "preprocessing" in params:
            pp = params["preprocessing"]
            self.artifact_correction.setChecked(pp.get("artifact_correction", True))
            if "interpolation_method" in pp:
                self.interpolation_method.setCurrentText(pp["interpolation_method"])
            self.correction_threshold.setValue(
                pp.get("correction_threshold", 0.05) * 100
            )

        if "detrending" in params:
            dt = params["detrending"]
            if "method" in dt:
                self.detrending_method.setCurrentText(dt["method"])
            self.detrending_lambda.setValue(dt.get("lambda", 500))

        if "frequency_domain" in params:
            fd = params["frequency_domain"]
            if "psd_method" in fd:
                self.psd_method.setCurrentText(fd["psd_method"])
            if "window_function" in fd:
                self.window_function.setCurrentText(fd["window_function"])


class QualityAssessmentWidget(QtWidgets.QWidget):
    """Widget for displaying quality assessment and warnings according to SRS requirements"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Quality metrics group
        quality_group = QtWidgets.QGroupBox("Quality Metrics")
        quality_layout = QtWidgets.QFormLayout(quality_group)

        self.corrected_beats_label = QtWidgets.QLabel("0%")
        self.signal_quality_label = QtWidgets.QLabel("Unknown")
        self.rhythm_type_label = QtWidgets.QLabel("Unknown")
        self.artifact_density_label = QtWidgets.QLabel("0%")

        quality_layout.addRow("Corrected Beats:", self.corrected_beats_label)
        quality_layout.addRow("Signal Quality:", self.signal_quality_label)
        quality_layout.addRow("Rhythm Type:", self.rhythm_type_label)
        quality_layout.addRow("Artifact Density:", self.artifact_density_label)

        # Warnings group
        warnings_group = QtWidgets.QGroupBox("Warnings & Alerts")
        warnings_layout = QtWidgets.QVBoxLayout(warnings_group)

        self.warnings_list = QtWidgets.QListWidget()
        self.warnings_list.setMaximumHeight(120)
        warnings_layout.addWidget(self.warnings_list)

        # Status indicator
        self.status_indicator = QtWidgets.QLabel("● Ready")
        self.status_indicator.setStyleSheet("color: gray; font-weight: bold;")

        layout.addWidget(quality_group)
        layout.addWidget(warnings_group)
        layout.addWidget(self.status_indicator)
        layout.addStretch()

    def update_quality_assessment(self, results: HRVAnalysisResults):
        """Update quality assessment display with results"""
        if not results:
            self._reset_display()
            return

        # Extract quality info from preprocessing stats first
        preprocessing_stats = results.preprocessing_stats or {}
        quality_assessment = results.quality_assessment or {}

        # Get corrected beats percentage from preprocessing stats
        corrected_pct = 0.0
        if preprocessing_stats:
            # Try different possible keys for artifact percentage
            corrected_pct = (
                preprocessing_stats.get("corrected_beats_percentage", 0)
                or preprocessing_stats.get("artifact_percentage", 0)
                or preprocessing_stats.get("artifacts_corrected", 0)
            )

        # Also check quality assessment
        if quality_assessment and corrected_pct == 0:
            corrected_pct = quality_assessment.get(
                "corrected_beats_percentage", 0
            ) or quality_assessment.get("artifact_percentage", 0)

        self.corrected_beats_label.setText(f"{corrected_pct:.1f}%")

        # Color code based on SRS threshold (5%)
        if corrected_pct > 5.0:
            self.corrected_beats_label.setStyleSheet("color: red; font-weight: bold;")
            self.status_indicator.setText("● Poor Quality")
            self.status_indicator.setStyleSheet("color: red; font-weight: bold;")
        elif corrected_pct > 2.0:
            self.corrected_beats_label.setStyleSheet(
                "color: orange; font-weight: bold;"
            )
            self.status_indicator.setText("● Moderate Quality")
            self.status_indicator.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.corrected_beats_label.setStyleSheet("color: green; font-weight: bold;")
            self.status_indicator.setText("● Good Quality")
            self.status_indicator.setStyleSheet("color: green; font-weight: bold;")

        # Update other quality metrics
        signal_quality = quality_assessment.get("signal_quality", "Unknown")
        self.signal_quality_label.setText(signal_quality)

        rhythm_type = quality_assessment.get("rhythm_type", "Unknown")
        self.rhythm_type_label.setText(rhythm_type)

        artifact_density = quality_assessment.get("artifact_density", 0)
        self.artifact_density_label.setText(f"{artifact_density:.1f}%")

        # Update warnings from multiple sources
        self.warnings_list.clear()
        warnings = []

        # Get warnings from results
        if hasattr(results, "warnings") and results.warnings:
            warnings.extend(results.warnings)

        # Get warnings from quality assessment
        if quality_assessment.get("warnings"):
            warnings.extend(quality_assessment["warnings"])

        # Get recommendations as warnings
        if quality_assessment.get("recommendations"):
            warnings.extend(quality_assessment["recommendations"])

        # Add automatic warning for high correction percentage (SRS requirement)
        if corrected_pct > 5.0:
            warnings.insert(
                0,
                f"High correction rate ({corrected_pct:.1f}%) exceeds recommended 5% threshold",
            )

        for warning in warnings:
            item = QtWidgets.QListWidgetItem(str(warning))

            # Color code warnings based on severity
            warning_str = str(warning).lower()
            if (
                "threshold exceeded" in warning_str
                or "high" in warning_str
                or "poor" in warning_str
            ):
                item.setBackground(QtGui.QColor(255, 200, 200))  # Light red
                item.setIcon(
                    self.style().standardIcon(
                        QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical
                    )
                )
            elif (
                "overlap" in warning_str
                or "respiration" in warning_str
                or "consider" in warning_str
            ):
                item.setBackground(QtGui.QColor(255, 255, 200))  # Light yellow
                item.setIcon(
                    self.style().standardIcon(
                        QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
                    )
                )
            else:
                item.setBackground(QtGui.QColor(200, 200, 255))  # Light blue
                item.setIcon(
                    self.style().standardIcon(
                        QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation
                    )
                )

            self.warnings_list.addItem(item)

    def _reset_display(self):
        """Reset display to default state"""
        self.corrected_beats_label.setText("0%")
        self.corrected_beats_label.setStyleSheet("")
        self.signal_quality_label.setText("Unknown")
        self.rhythm_type_label.setText("Unknown")
        self.artifact_density_label.setText("0%")
        self.warnings_list.clear()
        self.status_indicator.setText("● Ready")
        self.status_indicator.setStyleSheet("color: gray; font-weight: bold;")


class SignalViewerWidget(QtWidgets.QWidget):
    """Interactive signal viewer with editing capabilities according to SRS requirements"""

    beat_edited = pyqtSignal(int, str)  # beat_index, action

    def __init__(self):
        super().__init__()
        self.bundle = None
        self.results = None
        self.selected_beat_index = None
        self.edit_mode = None
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Editing toolbar
        toolbar_layout = QtWidgets.QHBoxLayout()

        # Edit mode selection
        self.edit_mode_group = QtWidgets.QButtonGroup()

        self.select_mode_btn = QtWidgets.QRadioButton("Select")
        self.select_mode_btn.setChecked(True)
        self.delete_mode_btn = QtWidgets.QRadioButton("Delete")
        self.move_mode_btn = QtWidgets.QRadioButton("Move")
        self.interpolate_mode_btn = QtWidgets.QRadioButton("Interpolate")

        self.edit_mode_group.addButton(self.select_mode_btn, 0)
        self.edit_mode_group.addButton(self.delete_mode_btn, 1)
        self.edit_mode_group.addButton(self.move_mode_btn, 2)
        self.edit_mode_group.addButton(self.interpolate_mode_btn, 3)

        # Action buttons
        self.apply_edit_btn = QtWidgets.QPushButton("Apply Edit")
        self.apply_edit_btn.setEnabled(False)
        self.undo_btn = QtWidgets.QPushButton("Undo Last")
        self.undo_btn.setEnabled(False)
        self.reset_btn = QtWidgets.QPushButton("Reset All")

        # Add to toolbar
        toolbar_layout.addWidget(QtWidgets.QLabel("Mode:"))
        toolbar_layout.addWidget(self.select_mode_btn)
        toolbar_layout.addWidget(self.delete_mode_btn)
        toolbar_layout.addWidget(self.move_mode_btn)
        toolbar_layout.addWidget(self.interpolate_mode_btn)
        toolbar_layout.addWidget(QtWidgets.QFrame())  # Separator
        toolbar_layout.addWidget(self.apply_edit_btn)
        toolbar_layout.addWidget(self.undo_btn)
        toolbar_layout.addWidget(self.reset_btn)
        toolbar_layout.addStretch()

        # Matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.navbar = NavigationToolbar(self.canvas, self)

        # Status line
        self.status_label = QtWidgets.QLabel(
            "Click on RR intervals to select beats for editing"
        )
        self.status_label.setStyleSheet("color: gray; font-style: italic;")

        layout.addLayout(toolbar_layout)
        layout.addWidget(self.navbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.status_label)

        # Connect signals
        self.edit_mode_group.buttonClicked.connect(self._on_mode_changed)
        self.apply_edit_btn.clicked.connect(self._apply_current_edit)
        self.undo_btn.clicked.connect(self._undo_last_edit)
        self.reset_btn.clicked.connect(self._reset_all_edits)

    def update_display(self, bundle: DataBundle, results: HRVAnalysisResults):
        """Update signal display with new data"""
        self.bundle = bundle
        self.results = results
        self.selected_beat_index = None
        self.apply_edit_btn.setEnabled(False)

        self.figure.clear()

        # Use enhanced plotting function
        plot_pipeline_results(self.figure, bundle, results)

        # Connect pick event for beat selection
        self.canvas.mpl_connect("pick_event", self._on_beat_selected)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.draw()

        self.status_label.setText(
            "Signal loaded. Click on RR intervals to select beats for editing."
        )

    def _on_mode_changed(self, button):
        """Handle edit mode change"""
        modes = ["select", "delete", "move", "interpolate"]
        self.edit_mode = modes[self.edit_mode_group.id(button)]

        if self.edit_mode == "select":
            self.status_label.setText("Click on RR intervals to select beats")
        else:
            self.status_label.setText(
                f"Click on beats to {self.edit_mode}. Click 'Apply Edit' to confirm."
            )

    def _on_beat_selected(self, event):
        """Handle beat selection for editing"""
        if event.mouseevent.inaxes and hasattr(event, "ind"):
            self.selected_beat_index = event.ind[0] if event.ind else None
            if self.selected_beat_index is not None:
                self.apply_edit_btn.setEnabled(self.edit_mode != "select")
                self.status_label.setText(
                    f"Selected beat {self.selected_beat_index}. Mode: {self.edit_mode}"
                )

    def _on_canvas_click(self, event):
        """Handle general canvas clicks"""
        if event.inaxes and self.bundle and self.bundle.rri_ms:
            # Find closest beat to click location
            if hasattr(event, "xdata") and event.xdata is not None:
                rri = np.array(self.bundle.rri_ms)
                t = np.cumsum(rri) / 1000.0

                # Find closest time point
                closest_idx = np.argmin(np.abs(t - event.xdata))
                self.selected_beat_index = closest_idx

                if self.edit_mode != "select":
                    self.apply_edit_btn.setEnabled(True)
                    self.status_label.setText(
                        f"Selected beat {closest_idx} at t={t[closest_idx]:.1f}s for {self.edit_mode}"
                    )

    def _apply_current_edit(self):
        """Apply the current edit operation"""
        if self.selected_beat_index is not None and self.edit_mode:
            self.beat_edited.emit(self.selected_beat_index, self.edit_mode)
            self.undo_btn.setEnabled(True)
            self.status_label.setText(
                f"Applied {self.edit_mode} to beat {self.selected_beat_index}"
            )
            self.apply_edit_btn.setEnabled(False)

    def _undo_last_edit(self):
        """Undo the last edit operation"""
        # This would need to be implemented with proper edit history
        self.status_label.setText("Undo functionality not yet implemented")

    def _reset_all_edits(self):
        """Reset all edits"""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Reset All Edits",
            "Are you sure you want to reset all manual edits?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.status_label.setText(
                "All edits reset (functionality not yet implemented)"
            )


class ExportDialog(QtWidgets.QDialog):
    """Dialog for configuring export options according to SRS requirements"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Results - SRS Compliant")
        self.setModal(True)
        self.resize(400, 500)
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Export format selection
        format_group = QtWidgets.QGroupBox("Export Formats (SRS FR-24 to FR-30)")
        format_layout = QtWidgets.QVBoxLayout(format_group)

        self.pdf_checkbox = QtWidgets.QCheckBox("PDF Report (Comprehensive)")
        self.csv_checkbox = QtWidgets.QCheckBox("CSV (Metrics Only)")
        self.spss_checkbox = QtWidgets.QCheckBox("SPSS-compatible CSV")
        self.audit_checkbox = QtWidgets.QCheckBox("Audit Trail Log (JSON)")

        self.pdf_checkbox.setChecked(True)

        # Add descriptions
        self.pdf_checkbox.setToolTip(
            "Generate comprehensive PDF report with plots, metrics, and quality assessment"
        )
        self.csv_checkbox.setToolTip(
            "Export metrics in standard CSV format for analysis"
        )
        self.spss_checkbox.setToolTip(
            "Export in SPSS-compatible format with proper variable naming"
        )
        self.audit_checkbox.setToolTip(
            "Export complete audit trail of all preprocessing actions"
        )

        format_layout.addWidget(self.pdf_checkbox)
        format_layout.addWidget(self.csv_checkbox)
        format_layout.addWidget(self.spss_checkbox)
        format_layout.addWidget(self.audit_checkbox)

        # Export content options
        content_group = QtWidgets.QGroupBox("Content Options")
        content_layout = QtWidgets.QVBoxLayout(content_group)

        self.include_plots = QtWidgets.QCheckBox("Include plots in PDF")
        self.include_plots.setChecked(True)

        self.include_preprocessing = QtWidgets.QCheckBox(
            "Include preprocessing details"
        )
        self.include_preprocessing.setChecked(True)

        self.include_quality = QtWidgets.QCheckBox("Include quality assessment")
        self.include_quality.setChecked(True)

        self.include_parameters = QtWidgets.QCheckBox("Include analysis parameters")
        self.include_parameters.setChecked(True)

        self.include_warnings = QtWidgets.QCheckBox("Include warnings and annotations")
        self.include_warnings.setChecked(True)

        content_layout.addWidget(self.include_plots)
        content_layout.addWidget(self.include_preprocessing)
        content_layout.addWidget(self.include_quality)
        content_layout.addWidget(self.include_parameters)
        content_layout.addWidget(self.include_warnings)

        # Advanced options
        advanced_group = QtWidgets.QGroupBox("Advanced Options")
        advanced_layout = QtWidgets.QFormLayout(advanced_group)

        self.decimal_places = QtWidgets.QSpinBox()
        self.decimal_places.setRange(1, 6)
        self.decimal_places.setValue(3)

        self.include_timestamp = QtWidgets.QCheckBox("Include timestamp in filenames")
        self.include_timestamp.setChecked(True)

        advanced_layout.addRow("Decimal Places:", self.decimal_places)
        advanced_layout.addRow("", self.include_timestamp)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.setDefault(True)
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )

        self.cancel_btn = QtWidgets.QPushButton("Cancel")

        button_layout.addStretch()
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.cancel_btn)

        # Add all groups to layout
        layout.addWidget(format_group)
        layout.addWidget(content_group)
        layout.addWidget(advanced_group)
        layout.addLayout(button_layout)

        # Connect signals
        self.export_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        # Enable/disable options based on format selection
        self.pdf_checkbox.toggled.connect(self._update_content_options)

    def _update_content_options(self):
        """Enable/disable content options based on PDF selection"""
        pdf_enabled = self.pdf_checkbox.isChecked()
        self.include_plots.setEnabled(pdf_enabled)

    def get_export_settings(self) -> Dict[str, Any]:
        """Return export settings as dictionary"""
        return {
            "formats": {
                "pdf": self.pdf_checkbox.isChecked(),
                "csv": self.csv_checkbox.isChecked(),
                "spss": self.spss_checkbox.isChecked(),
                "audit_trail": self.audit_checkbox.isChecked(),
            },
            "content": {
                "include_plots": self.include_plots.isChecked(),
                "include_preprocessing": self.include_preprocessing.isChecked(),
                "include_quality": self.include_quality.isChecked(),
                "include_parameters": self.include_parameters.isChecked(),
                "include_warnings": self.include_warnings.isChecked(),
            },
            "options": {
                "decimal_places": self.decimal_places.value(),
                "include_timestamp": self.include_timestamp.isChecked(),
            },
        }


class HelpDialog(QtWidgets.QDialog):
    """Comprehensive help dialog implementing SRS FR-31 requirements"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HRV Analysis Help - User Manual")
        self.resize(800, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Create tabbed help content
        help_tabs = QtWidgets.QTabWidget()

        # Getting Started tab
        getting_started = QtWidgets.QTextEdit()
        getting_started.setReadOnly(True)
        getting_started.setHtml(self._get_getting_started_content())
        help_tabs.addTab(getting_started, "Getting Started")

        # File Formats tab
        file_formats = QtWidgets.QTextEdit()
        file_formats.setReadOnly(True)
        file_formats.setHtml(self._get_file_formats_content())
        help_tabs.addTab(file_formats, "File Formats")

        # Analysis Features tab
        analysis_features = QtWidgets.QTextEdit()
        analysis_features.setReadOnly(True)
        analysis_features.setHtml(self._get_analysis_features_content())
        help_tabs.addTab(analysis_features, "Analysis Features")

        # Quality Assessment tab
        quality_assessment = QtWidgets.QTextEdit()
        quality_assessment.setReadOnly(True)
        quality_assessment.setHtml(self._get_quality_assessment_content())
        help_tabs.addTab(quality_assessment, "Quality Assessment")

        # Troubleshooting tab
        troubleshooting = QtWidgets.QTextEdit()
        troubleshooting.setReadOnly(True)
        troubleshooting.setHtml(self._get_troubleshooting_content())
        help_tabs.addTab(troubleshooting, "Troubleshooting")

        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        layout.addWidget(help_tabs)
        layout.addWidget(close_btn)

    def _get_getting_started_content(self) -> str:
        return """
        <h2>Getting Started with HRV Analysis</h2>
        
        <h3>Basic Workflow</h3>
        <ol>
        <li><b>Load Data:</b> Use File → Open to load your HRV data file</li>
        <li><b>Configure Parameters:</b> Adjust analysis settings in the left panel</li>
        <li><b>Run Analysis:</b> Click the "Run Analysis" button</li>
        <li><b>Review Results:</b> Check the Signal Viewer, Metrics Summary, and Analysis Plots tabs</li>
        <li><b>Assess Quality:</b> Monitor the Quality Assessment panel for warnings</li>
        <li><b>Export Results:</b> Use File → Export Results to save your analysis</li>
        </ol>
        
        <h3>Key Features</h3>
        <ul>
        <li><b>Interactive Signal Editing:</b> Click on RR intervals to select and edit beats</li>
        <li><b>Real-time Quality Assessment:</b> Continuous monitoring of data quality</li>
        <li><b>Comprehensive Metrics:</b> Time-domain, frequency-domain, and nonlinear analysis</li>
        <li><b>Professional Reports:</b> PDF and CSV export for research and clinical use</li>
        </ul>
        
        <h3>Quick Tips</h3>
        <ul>
        <li>Use analysis windows of at least 5 minutes for reliable frequency-domain metrics</li>
        <li>Monitor the "Corrected Beats" percentage - values >5% indicate poor data quality</li>
        <li>Check warnings for respiratory frequency overlap with LF/HF bands</li>
        <li>Save your session to preserve manual edits and parameter settings</li>
        </ul>
        """

    def _get_file_formats_content(self) -> str:
        return """
        <h2>Supported File Formats</h2>
        
        <h3>RR Interval Data</h3>
        <ul>
        <li><b>CSV (.csv):</b> Comma-separated values with RR intervals in milliseconds</li>
        <li><b>Text (.txt):</b> Plain text files with one RR interval per line</li>
        <li><b>Polar HRM (.hrm):</b> Polar heart rate monitor format</li>
        <li><b>Garmin FIT (.fit):</b> Garmin activity files</li>
        <li><b>Suunto SML (.sml):</b> Suunto training files</li>
        </ul>
        
        <h3>Waveform Data</h3>
        <ul>
        <li><b>EDF (.edf):</b> European Data Format for ECG/PPG waveforms</li>
        <li><b>Movesense JSON:</b> Movesense sensor data</li>
        <li><b>Biopac/AcqKnowledge:</b> Laboratory acquisition formats</li>
        </ul>
        
        <h3>Data Requirements</h3>
        <ul>
        <li><b>Minimum Duration:</b> 2 minutes for basic metrics, 5+ minutes recommended</li>
        <li><b>Sampling Rate:</b> Variable, automatically detected</li>
        <li><b>Quality:</b> Less than 5% artifacts recommended for reliable analysis</li>
        </ul>
        
        <h3>Format Examples</h3>
        <h4>CSV Format:</h4>
        <pre>
        RR_ms
        856
        924
        812
        ...
        </pre>
        
        <h4>Text Format:</h4>
        <pre>
        856
        924
        812
        ...
        </pre>
        """

    def _get_analysis_features_content(self) -> str:
        return """
        <h2>Analysis Features</h2>
        
        <h3>Time Domain Metrics</h3>
        <ul>
        <li><b>SDNN:</b> Standard deviation of NN intervals (overall HRV)</li>
        <li><b>RMSSD:</b> Root mean square of successive differences (short-term HRV)</li>
        <li><b>pNN50:</b> Percentage of intervals differing >50ms from previous</li>
        <li><b>CVNN:</b> Coefficient of variation of NN intervals</li>
        <li><b>HRV Triangular Index:</b> Geometric measure of HRV</li>
        </ul>
        
        <h3>Frequency Domain Analysis</h3>
        <ul>
        <li><b>VLF Power:</b> Very low frequency (0.0033-0.04 Hz)</li>
        <li><b>LF Power:</b> Low frequency (0.04-0.15 Hz) - sympathetic activity</li>
        <li><b>HF Power:</b> High frequency (0.15-0.4 Hz) - parasympathetic activity</li>
        <li><b>LF/HF Ratio:</b> Balance between sympathetic and parasympathetic</li>
        <li><b>Normalized Units:</b> LF and HF in normalized units</li>
        </ul>
        
        <h3>Nonlinear Metrics</h3>
        <ul>
        <li><b>Poincaré Plot:</b> SD1 (short-term variability), SD2 (long-term variability)</li>
        <li><b>DFA:</b> Detrended fluctuation analysis (α1, α2)</li>
        <li><b>Sample Entropy:</b> Regularity and complexity of RR intervals</li>
        <li><b>Multiscale Entropy:</b> Complexity across multiple time scales</li>
        </ul>
        
        <h3>Respiratory Analysis</h3>
        <ul>
        <li><b>EDR-AM:</b> ECG-derived respiration using amplitude modulation</li>
        <li><b>Respiratory Rate:</b> Estimated breathing rate in breaths per minute</li>
        <li><b>RSA Coherence:</b> Respiratory sinus arrhythmia coherence</li>
        <li><b>LF/HF Overlap Detection:</b> Warnings when respiration overlaps frequency bands</li>
        </ul>
        """

    def _get_quality_assessment_content(self) -> str:
        return """
        <h2>Quality Assessment Guidelines</h2>
        
        <h3>Quality Metrics</h3>
        <ul>
        <li><b>Corrected Beats (%):</b> Percentage of beats that required correction</li>
        <li><b>Signal Quality:</b> Overall assessment of signal integrity</li>
        <li><b>Rhythm Type:</b> Normal sinus rhythm vs. irregular rhythms</li>
        <li><b>Artifact Density:</b> Rate of detected artifacts per minute</li>
        </ul>
        
        <h3>Quality Thresholds (SRS Compliant)</h3>
        <ul>
        <li><b>Excellent:</b> <2% corrected beats, clean signal</li>
        <li><b>Good:</b> 2-5% corrected beats, minor artifacts</li>
        <li><b>Poor:</b> >5% corrected beats, significant artifacts</li>
        </ul>
        
        <h3>Warning Indicators</h3>
        <ul>
        <li><b>Red Status:</b> >5% corrected beats - consider excluding analysis</li>
        <li><b>Orange Status:</b> 2-5% corrected beats - interpret with caution</li>
        <li><b>Green Status:</b> <2% corrected beats - reliable analysis</li>
        </ul>
        
        <h3>Common Quality Issues</h3>
        <ul>
        <li><b>Missed Beats:</b> Long RR intervals due to undetected beats</li>
        <li><b>Extra Beats:</b> Short intervals from false beat detection</li>
        <li><b>Ectopic Beats:</b> Abnormal beats (PVCs, PACs)</li>
        <li><b>Noise:</b> Motion artifacts, electrode issues</li>
        <li><b>Arrhythmias:</b> Atrial fibrillation, frequent ectopy</li>
        </ul>
        
        <h3>Respiratory Warnings</h3>
        <ul>
        <li><b>LF Band Overlap:</b> Respiration frequency overlaps with LF band (0.04-0.15 Hz)</li>
        <li><b>HF Band Overlap:</b> Respiration frequency overlaps with HF band (0.15-0.4 Hz)</li>
        <li><b>Low RSA Coherence:</b> Weak coupling between heart rate and respiration</li>
        </ul>
        """

    def _get_troubleshooting_content(self) -> str:
        return """
        <h2>Troubleshooting Guide</h2>
        
        <h3>File Loading Issues</h3>
        <ul>
        <li><b>Unsupported Format:</b> Check that your file is in a supported format</li>
        <li><b>Empty File:</b> Ensure the file contains valid RR interval or waveform data</li>
        <li><b>Encoding Errors:</b> Try saving the file in UTF-8 encoding</li>
        <li><b>Large Files:</b> Files >100MB may require additional processing time</li>
        </ul>
        
        <h3>Analysis Problems</h3>
        <ul>
        <li><b>No Results:</b> Check that you have sufficient data (minimum 2 minutes)</li>
        <li><b>Poor Quality Warnings:</b> Consider manual editing or using a different data segment</li>
        <li><b>Frequency Analysis Fails:</b> Ensure analysis window is at least 2 minutes</li>
        <li><b>Memory Errors:</b> Try analyzing shorter segments or reducing sampling rate</li>
        </ul>
        
        <h3>Export Issues</h3>
        <ul>
        <li><b>PDF Generation Fails:</b> Check file permissions and available disk space</li>
        <li><b>Large Export Files:</b> Disable plot inclusion for smaller file sizes</li>
        <li><b>Permission Errors:</b> Choose a different export location</li>
        </ul>
        
        <h3>Performance Optimization</h3>
        <ul>
        <li><b>Slow Analysis:</b> Reduce analysis window duration or use fewer metrics</li>
        <li><b>High Memory Usage:</b> Close other applications and restart the software</li>
        <li><b>Visualization Lag:</b> Reduce plot complexity in preferences</li>
        </ul>
        
        <h3>Getting Help</h3>
        <ul>
        <li><b>User Manual:</b> Comprehensive documentation available online</li>
        <li><b>Support Forum:</b> Community-driven help and discussion</li>
        <li><b>Bug Reports:</b> Report issues through the support system</li>
        <li><b>Feature Requests:</b> Suggest improvements for future versions</li>
        </ul>
        """


class SessionManager:
    """Utility class for managing session save/load functionality"""

    @staticmethod
    def save_session(
        file_path: str,
        analysis_parameters: Dict,
        edit_history: list,
        file_info: str,
        results: Optional[HRVAnalysisResults] = None,
    ) -> bool:
        """Save current session to JSON file"""
        try:
            session_data = {
                "version": "2.1.1",
                "timestamp": datetime.now().isoformat(),
                "file_info": file_info,
                "analysis_parameters": analysis_parameters,
                "edit_history": edit_history,
                "results_summary": (
                    SessionManager._serialize_results(results) if results else None
                ),
            }

            with open(file_path, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
            return True

        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    @staticmethod
    def load_session(file_path: str) -> Optional[Dict[str, Any]]:
        """Load session from JSON file"""
        try:
            with open(file_path, "r") as f:
                session_data = json.load(f)
            return session_data

        except Exception as e:
            print(f"Error loading session: {e}")
            return None

    @staticmethod
    def _serialize_results(results: HRVAnalysisResults) -> Dict[str, Any]:
        """Convert results to serializable format"""
        serialized = {}

        if results.time_domain:
            serialized["time_domain"] = dict(results.time_domain)
        if results.frequency_domain:
            serialized["frequency_domain"] = dict(results.frequency_domain)
        if results.nonlinear:
            serialized["nonlinear"] = dict(results.nonlinear)
        if results.respiratory:
            serialized["respiratory"] = dict(results.respiratory)
        if results.quality_assessment:
            serialized["quality_assessment"] = dict(results.quality_assessment)
        if results.preprocessing_stats:
            serialized["preprocessing_stats"] = dict(results.preprocessing_stats)

        return serialized
