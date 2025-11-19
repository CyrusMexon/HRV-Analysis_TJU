"""
Enhanced widgets.py - FIXED VERSION
Only the specific widgets with issues have been modified
All original classes and logic preserved
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


### FIXED Results Panel and Sub-widgets ###
class MetricSectionWidget(QtWidgets.QWidget):
    """Base widget for displaying a section of HRV metrics"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Section header
        self.header = QtWidgets.QLabel(self.title)
        self.header.setStyleSheet(
            """
            QLabel {
                background-color: #d0d0d0;
                padding: 12px 18px;
                font-weight: 600;
                color: #000000;
                font-size: 14px;
                border: 1px solid #cccccc;
                border-bottom: 1px solid #bbbbbb;
            }
        """
        )

        # Content area
        self.content_widget = QtWidgets.QWidget()
        self.content_widget.setStyleSheet(
            """
            QWidget {
                background-color: grey;
                border: none;
                border-top: none;
            }
        """
        )

        layout.addWidget(self.header)
        layout.addWidget(self.content_widget)

    def set_content_layout(self, layout):
        """Set the layout for the content area"""
        self.content_widget.setLayout(layout)


class TimeDomainWidget(MetricSectionWidget):
    """Widget for displaying time domain metrics"""

    def __init__(self, parent=None):
        super().__init__("Time Domain Analysis", parent)
        self.setup_content()

    def setup_content(self):
        layout = QtWidgets.QVBoxLayout()

        # Create table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setShowGrid(False)  # Disable grid lines
        self.table.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)  # Remove frame

        # FIXED: Style the table with black text and larger rows
        self.table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: transparent;
                background-color: grey;
                border: none;
                color: #000000;
            }
            QTableWidget::item {
                padding: 12px 8px;
                border-bottom: 1px solid #e0e0e0;
                color: #000000;
                min-height: 20px;
            }
            QTableWidget::item:hover {
                background-color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #eeeeee;
                padding: 12px 8px;
                border: none;
                border-bottom: 1px solid #dddddd;
                font-weight: 600;
                color: #000000;
                min-height: 20px;
            }
        """
        )

        # FIXED: Set row height
        self.table.verticalHeader().setDefaultSectionSize(40)

        layout.addWidget(self.table)
        self.set_content_layout(layout)

    def update_metrics(self, time_domain_data):
        """Update the table with time domain metrics"""
        if not time_domain_data:
            self.table.setRowCount(0)
            return

        metrics = [
            ("sdnn", "SDNN", "ms"),
            ("rmssd", "RMSSD", "ms"),
            ("pnn50", "pNN50", "%"),
            ("pnn20", "pNN20", "%"),
            ("mean_rr", "Mean RR", "ms"),
            ("mean_hr", "Mean HR", "bpm"),
            ("cvnn", "CV", ""),
            ("hrv_triangular_index", "HRV Triangular Index", ""),
            ("tinn", "TINN", "ms"),
        ]

        # Filter only available metrics
        available_metrics = [
            (key, name, unit)
            for key, name, unit in metrics
            if key in time_domain_data
            and isinstance(time_domain_data[key], (int, float))
        ]

        self.table.setRowCount(len(available_metrics))

        for row, (key, name, unit) in enumerate(available_metrics):
            value = time_domain_data[key]

            # Metric name
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

            # Value
            value_item = QtWidgets.QTableWidgetItem(f"{value:.2f}")
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            value_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )

            # Unit
            unit_item = QtWidgets.QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, value_item)
            self.table.setItem(row, 2, unit_item)

        self.table.resizeColumnsToContents()
        # FIXED: Set minimum height to show all rows without scrolling
        row_height = 40
        header_height = 40
        min_height = len(available_metrics) * row_height + header_height + 20
        self.table.setMinimumHeight(min_height)


class FrequencyDomainWidget(MetricSectionWidget):
    """Widget for displaying frequency domain metrics - FIXED to show all metrics like Kubios"""

    def __init__(self, parent=None):
        super().__init__("Frequency Domain Analysis", parent)
        self.setup_content()

    def setup_content(self):
        layout = QtWidgets.QVBoxLayout()

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setShowGrid(False)
        self.table.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)

        # Style the table
        self.table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: transparent;
                background-color: grey;
                border: none;
                color: #000000;
            }
            QTableWidget::item {
                padding: 12px 8px;
                border-bottom: 1px solid #e0e0e0;
                color: #000000;
                min-height: 20px;
            }
            QTableWidget::item:hover {
                background-color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #eeeeee;
                padding: 12px 8px;
                border: none;
                border-bottom: 1px solid #dddddd;
                font-weight: 600;
                color: #000000;
                min-height: 20px;
            }
        """
        )

        self.table.verticalHeader().setDefaultSectionSize(40)

        layout.addWidget(self.table)
        self.set_content_layout(layout)

    def update_metrics(self, frequency_data):
        """Update the table with frequency domain metrics - COMPLETE VERSION"""
        if not frequency_data:
            self.table.setRowCount(0)
            return

        # Define ALL metrics to display (matching Kubios output)
        metrics = [
            # Absolute powers
            ("VLF_power", "VLF Power", "ms²", 2),
            ("vlf_power", "VLF Power", "ms²", 2),  # Alternative key
            ("LF_power", "LF Power", "ms²", 2),
            ("lf_power", "LF Power", "ms²", 2),  # Alternative key
            ("HF_power", "HF Power", "ms²", 2),
            ("hf_power", "HF Power", "ms²", 2),  # Alternative key
            ("total_power", "Total Power", "ms²", 2),
            # Peak frequencies
            ("peak_freq_lf", "LF Peak Frequency", "Hz", 4),
            ("peak_freq_hf", "HF Peak Frequency", "Hz", 4),
            # Relative powers (percentage of total)
            ("VLF_power_nu", "VLF Power (%)", "%", 2),
            ("vlf_power_nu", "VLF Power (%)", "%", 2),
            ("LF_power_nu", "LF Power (%)", "%", 2),
            ("lf_power_nu", "LF Power (%)", "%", 2),
            ("HF_power_nu", "HF Power (%)", "%", 2),
            ("hf_power_nu", "HF Power (%)", "%", 2),
            # Normalized units (LF and HF as % of LF+HF)
            ("relative_lf_power", "LF (n.u.)", "%", 2),
            ("relative_hf_power", "HF (n.u.)", "%", 2),
            # LF/HF ratio
            ("LF_HF_ratio", "LF/HF Ratio", "", 3),
            ("lf_hf_ratio", "LF/HF Ratio", "", 3),
        ]

        # Collect available metrics (avoid duplicates)
        available_metrics = []
        seen_names = set()

        for key, name, unit, precision in metrics:
            if key in frequency_data and name not in seen_names:
                value = frequency_data[key]
                if isinstance(value, (int, float)):
                    # Skip NaN or inf values for cleaner display
                    if not (np.isnan(value) or np.isinf(value)):
                        available_metrics.append((key, name, unit, value, precision))
                        seen_names.add(name)

        if not available_metrics:
            # Show message if no metrics available
            self.table.setRowCount(1)
            msg_item = QtWidgets.QTableWidgetItem(
                "No frequency domain metrics available"
            )
            msg_item.setFlags(msg_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(0, 0, msg_item)
            self.table.setSpan(0, 0, 1, 3)
            return

        self.table.setRowCount(len(available_metrics))

        for row, (key, name, unit, value, precision) in enumerate(available_metrics):
            # Metric name
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

            # Value with appropriate precision
            value_item = QtWidgets.QTableWidgetItem(f"{value:.{precision}f}")
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            value_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )

            # Unit
            unit_item = QtWidgets.QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, value_item)
            self.table.setItem(row, 2, unit_item)

        self.table.resizeColumnsToContents()

        # Set minimum height to show all rows
        row_height = 40
        header_height = 40
        min_height = len(available_metrics) * row_height + header_height + 20
        self.table.setMinimumHeight(min_height)


class NonlinearWidget(MetricSectionWidget):
    """Widget for displaying nonlinear metrics with subsections"""

    def __init__(self, parent=None):
        super().__init__("Nonlinear Analysis", parent)
        self.setup_content()

    def setup_content(self):
        self.main_layout = QtWidgets.QVBoxLayout()
        self.set_content_layout(self.main_layout)

    def update_metrics(self, nonlinear_data):
        """Update with nonlinear metrics data"""
        # Clear existing widgets
        for i in reversed(range(self.main_layout.count())):
            self.main_layout.itemAt(i).widget().setParent(None)

        if not nonlinear_data:
            return

        # Poincare Plot
        if "poincare" in nonlinear_data and isinstance(
            nonlinear_data["poincare"], dict
        ):
            poincare_widget = self.create_subsection(
                "Poincaré Plot",
                nonlinear_data["poincare"],
                [
                    ("sd1", "SD1", "ms"),
                    ("sd2", "SD2", "ms"),
                    ("sd1_sd2_ratio", "SD1/SD2 Ratio", ""),
                    ("ellipse_area", "Ellipse Area", "ms²"),
                ],
            )
            self.main_layout.addWidget(poincare_widget)

        # DFA
        if "dfa" in nonlinear_data and isinstance(nonlinear_data["dfa"], dict):
            dfa_widget = self.create_subsection(
                "Detrended Fluctuation Analysis",
                nonlinear_data["dfa"],
                [
                    ("alpha1", "DFA α1", ""),
                    ("alpha2", "DFA α2", ""),
                ],
            )
            self.main_layout.addWidget(dfa_widget)

        # Sample Entropy
        if "sample_entropy" in nonlinear_data:
            entropy_data = {"sample_entropy": nonlinear_data["sample_entropy"]}
            entropy_widget = self.create_subsection(
                "Entropy Analysis",
                entropy_data,
                [
                    ("sample_entropy", "Sample Entropy", ""),
                ],
            )
            self.main_layout.addWidget(entropy_widget)

    def create_subsection(self, title, data, metrics):
        """Create a subsection widget for nonlinear metrics"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Subsection title
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet(
            """
            QLabel {
                background-color: #e6e6e6;
                padding: 8px 18px;
                font-weight: 600;
                color: #000000;
                font-size: 12px;
                text-transform: uppercase;
                border-bottom: 1px solid #dddddd;
            }
        """
        )

        # Table
        table = QtWidgets.QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setDefaultSectionSize(35)

        # FIXED: Updated styling
        table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: #C9CDCF;
                border: none;
                color: #000000;
            }
            QTableWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #e0e0e0;
                color: #000000;
                min-height: 18px;
            }
            QTableWidget::item:hover {
                background-color: #f0f0f0;
            }
        """
        )

        # Populate table
        available_metrics = [
            (key, name, unit)
            for key, name, unit in metrics
            if key in data and isinstance(data[key], (int, float))
        ]

        table.setRowCount(len(available_metrics))

        for row, (key, name, unit) in enumerate(available_metrics):
            value = float(data[key])

            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

            # Use more precision for some metrics
            precision = 3 if "alpha" in key or "entropy" in key else 2
            value_item = QtWidgets.QTableWidgetItem(f"{value:.{precision}f}")
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            value_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )

            unit_item = QtWidgets.QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

            table.setItem(row, 0, name_item)
            table.setItem(row, 1, value_item)
            table.setItem(row, 2, unit_item)

        table.resizeColumnsToContents()
        # FIXED: Set height to fit content
        row_height = 35
        min_height = len(available_metrics) * row_height + 10
        table.setMaximumHeight(min_height)
        table.setMinimumHeight(min_height)

        layout.addWidget(title_label)
        layout.addWidget(table)

        return widget


class QualityAssessmentSectionWidget(MetricSectionWidget):
    """Widget for displaying quality assessment"""

    def __init__(self, parent=None):
        super().__init__("Quality Assessment", parent)
        self.setup_content()

    def setup_content(self):
        layout = QtWidgets.QVBoxLayout()

        # Quality badge
        self.badge_widget = QtWidgets.QWidget()
        badge_layout = QtWidgets.QHBoxLayout(self.badge_widget)
        badge_layout.setContentsMargins(18, 15, 18, 15)

        self.quality_badge = QtWidgets.QLabel("UNKNOWN")
        self.quality_badge.setStyleSheet(
            """
            QLabel {
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                background-color: #C9CDCF;
                color: #000000;
                border: 1px solid #dee2e6;
            }
        """
        )

        badge_layout.addWidget(self.quality_badge)
        badge_layout.addStretch()

        # Metrics table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(40)

        # FIXED: Updated styling
        self.table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: #C9CDCF;
                border: none;
                color: #000000;
            }
            QTableWidget::item {
                padding: 12px 8px;
                border-bottom: 1px solid #e0e0e0;
                color: #000000;
                min-height: 20px;
            }
            QTableWidget::item:hover {
                background-color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #eeeeee;
                padding: 12px 8px;
                border: none;
                border-bottom: 1px solid #dddddd;
                font-weight: 600;
                color: #000000;
                min-height: 20px;
            }
        """
        )

        layout.addWidget(self.badge_widget)
        layout.addWidget(self.table)
        self.set_content_layout(layout)

    def update_quality(self, quality_data):
        """Update quality assessment display with proper quality determination"""
        if not quality_data:
            self.quality_badge.setText("UNKNOWN")
            self._set_badge_style("unknown")
            return

        # Determine overall quality based on multiple factors
        overall_quality = self._determine_overall_quality(quality_data)

        # Update badge
        self.quality_badge.setText(overall_quality.upper())
        self._set_badge_style(overall_quality)

        # Update metrics table
        metrics = [
            ("artifact_percentage", "Corrected Beats", "%"),
            ("duration_s", "Recording Duration", "s"),
        ]

        available_metrics = [
            (key, name, unit)
            for key, name, unit in metrics
            if key in quality_data and isinstance(quality_data[key], (int, float))
        ]

        self.table.setRowCount(len(available_metrics))

        for row, (key, name, unit) in enumerate(available_metrics):
            value = quality_data[key]

            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

            value_item = QtWidgets.QTableWidgetItem(f"{value:.1f}")
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            value_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )

            unit_item = QtWidgets.QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, value_item)
            self.table.setItem(row, 2, unit_item)

        self.table.resizeColumnsToContents()

        # Set table height
        row_height = 40
        header_height = 40
        min_height = len(available_metrics) * row_height + header_height + 20
        self.table.setMinimumHeight(min_height)

    def _determine_overall_quality(self, quality_data):
        """Determine overall quality based on multiple quality indicators (SRS compliant)"""

        # Primary factor: artifact/correction percentage
        artifact_pct = quality_data.get("artifact_percentage", 0)

        # Check for manual editing data from preprocessing stats
        if artifact_pct == 0:
            # Try alternative keys for artifact percentage
            artifact_pct = (
                quality_data.get("corrected_beats_percentage", 0)
                or quality_data.get("artifacts_corrected", 0)
                or quality_data.get("correction_percentage", 0)
            )

        # Duration factor
        duration_s = quality_data.get("duration_s", 0)

        # Explicit quality assessment if available
        explicit_quality = quality_data.get("overall_quality")
        if explicit_quality and explicit_quality.lower() in [
            "good",
            "fair",
            "poor",
            "excellent",
        ]:
            return explicit_quality.lower()

        # Signal quality factor
        signal_quality = quality_data.get("signal_quality", "").lower()

        # Determine quality based on SRS FR-14 thresholds
        if artifact_pct > 10.0:
            return "poor"
        elif artifact_pct > 5.0:
            # Additional checks for poor quality
            if duration_s < 120 or signal_quality in ["poor", "bad", "noisy"]:
                return "poor"
            else:
                return "fair"
        elif artifact_pct > 2.0:
            # Additional checks for fair quality
            if duration_s < 300:  # Less than 5 minutes
                return "fair"
            elif signal_quality in ["good", "excellent", "clean"]:
                return "good"
            else:
                return "fair"
        else:
            # Low artifact percentage
            if duration_s >= 300 and signal_quality in ["good", "excellent", "clean"]:
                return "excellent"
            elif duration_s >= 120:
                return "good"
            else:
                return "fair"  # Short duration limits quality

    def _set_badge_style(self, quality_level):
        """Set badge styling based on quality level"""
        style_map = {
            "excellent": "background-color: #4caf50; color: white; border: 1px solid #45a049;",
            "good": "background-color: #8bc34a; color: white; border: 1px solid #7cb342;",
            "fair": "background-color: #ff9800; color: white; border: 1px solid #f57c00;",
            "poor": "background-color: #f44336; color: white; border: 1px solid #d32f2f;",
            "unknown": "background-color: #9e9e9e; color: white; border: 1px solid #757575;",
        }

        style = style_map.get(quality_level, style_map["unknown"])

        self.quality_badge.setStyleSheet(
            f"""
            QLabel {{
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                {style}
            }}
        """
        )


class WarningsWidget(MetricSectionWidget):
    """Widget for displaying warnings and analysis notes"""

    def __init__(self, parent=None):
        super().__init__("Analysis Notes & Warnings", parent)
        self.setup_content()

    def setup_content(self):
        layout = QtWidgets.QVBoxLayout()

        self.warnings_list = QtWidgets.QListWidget()

        self.warnings_list.setStyleSheet(
            """
            QListWidget {
                background-color: #C9CDCF;
                border: none;
                padding: 8px;
                color: #000000;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #e0e0e0;
                color: #cc0000;
                font-size: 12px;
                min-height: 20px;
            }
            QListWidget::item:last-child {
                border-bottom: none;
            }
        """
        )

        layout.addWidget(self.warnings_list)
        self.set_content_layout(layout)

    def update_warnings(self, warnings_data, quality_data=None):
        """Update warnings display"""
        self.warnings_list.clear()

        all_warnings = []

        # Add general warnings
        if warnings_data:
            all_warnings.extend(warnings_data)

        # Add quality recommendations
        if quality_data and "recommendations" in quality_data:
            all_warnings.extend(quality_data["recommendations"])

        if not all_warnings:
            item = QtWidgets.QListWidgetItem("No warnings or notes")
            item.setForeground(QtGui.QColor("#666666"))  # Gray color for "no warnings"
            self.warnings_list.addItem(item)
            return

        for warning in all_warnings:
            item = QtWidgets.QListWidgetItem(f"• {warning}")
            self.warnings_list.addItem(item)


# Updated ResultsPanel - UNCHANGED except for styling fixes
class ResultsPanel(QtWidgets.QScrollArea):
    """Enhanced results panel using modular widgets"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.setup_ui()

    def setup_ui(self):
        # Main widget
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Header
        header = QtWidgets.QLabel("HRV Analysis Report")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            """
            QLabel {
                background-color: #C9CDCF;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 30px;
                font-size: 18px;
                font-weight: 600;
                color: #000000;
            }
        """
        )
        main_layout.addWidget(header)

        # Create section widgets
        self.time_domain_widget = TimeDomainWidget()
        self.frequency_domain_widget = FrequencyDomainWidget()
        self.nonlinear_widget = NonlinearWidget()
        self.quality_widget = QualityAssessmentSectionWidget()
        self.warnings_widget = WarningsWidget()

        main_layout.addWidget(self.time_domain_widget)
        main_layout.addWidget(self.frequency_domain_widget)
        main_layout.addWidget(self.nonlinear_widget)
        main_layout.addWidget(self.quality_widget)
        main_layout.addWidget(self.warnings_widget)
        main_layout.addStretch()

        self.setWidget(main_widget)

        # Style the scroll area
        self.setStyleSheet(
            """
            QScrollArea {
                background-color: #C9CDCF;
                border: none;
            }
        """
        )

    def show_results(self, results: dict):
        """FIXED: Update all section widgets with results data"""

        # Update each section with debugging
        time_domain = results.get("time_domain")
        self.time_domain_widget.update_metrics(time_domain)

        # Try multiple possible keys for frequency domain data
        freq_data = (
            results.get("frequency_domain")
            or results.get("freq_domain")
            or results.get("frequency")
            or results.get("freq")
        )

        self.frequency_domain_widget.update_metrics(freq_data)

        nonlinear_data = results.get("nonlinear")
        self.nonlinear_widget.update_metrics(nonlinear_data)

        quality_data = results.get("quality_assessment")
        self.quality_widget.update_quality(quality_data)

        self.warnings_widget.update_warnings(results.get("warnings"), quality_data)


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
        """Update quality assessment display with results - PIPELINE AWARE"""
        if not results:
            self._reset_display()
            return

        # Extract data from your pipeline's structure
        preprocessing_stats = results.preprocessing_stats or {}
        quality_assessment = results.quality_assessment or {}

        # 1. CORRECTED BEATS PERCENTAGE - from preprocessing_stats
        corrected_pct = preprocessing_stats.get("artifact_percentage", 0.0)
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

        # 2. SIGNAL QUALITY - from quality_assessment.overall_quality
        signal_quality = quality_assessment.get("overall_quality", "Unknown").title()
        self.signal_quality_label.setText(signal_quality)

        # 3. RHYTHM TYPE - derive from quality flags and warnings
        rhythm_type = "Normal Sinus"  # Default assumption

        # Check quality flags from preprocessing
        quality_flags = preprocessing_stats.get("quality_flags", {})
        if quality_flags.get("excessive_artifacts", False):
            rhythm_type = "Ectopic Present"
        elif quality_flags.get("poor_signal_quality", False):
            rhythm_type = "Poor Signal"

        # Check warnings for rhythm information
        warnings = results.warnings or []
        warning_text = " ".join(warnings).lower()
        if any(
            term in warning_text for term in ["atrial fibrillation", "af", "irregular"]
        ):
            rhythm_type = "Irregular"
        elif any(term in warning_text for term in ["arrhythmia", "ectopic"]):
            rhythm_type = "Ectopic Present"

        self.rhythm_type_label.setText(rhythm_type)

        # 4. ARTIFACT DENSITY - calculate from your pipeline data
        artifact_density = 0.0

        artifacts_detected = preprocessing_stats.get("artifacts_detected", 0)
        duration_s = quality_assessment.get("duration_s", 0)

        if duration_s > 0 and artifacts_detected > 0:
            # Artifacts per minute
            artifact_density = (artifacts_detected / duration_s) * 60.0

        self.artifact_density_label.setText(f"{artifact_density:.1f}/min")

        # 5. UPDATE WARNINGS - from multiple sources in your pipeline
        self.warnings_list.clear()
        warnings = []

        # Get warnings from results.warnings
        if results.warnings:
            warnings.extend(results.warnings)

        # Get recommendations from quality_assessment
        if quality_assessment.get("recommendations"):
            warnings.extend(quality_assessment["recommendations"])

        # Add SRS-compliant warnings based on your data
        if corrected_pct > 5.0:
            warnings.insert(
                0,
                f"High correction rate ({corrected_pct:.1f}%) exceeds recommended 5% threshold (SRS FR-14)",
            )
        elif corrected_pct > 2.0:
            warnings.insert(
                0,
                f"Moderate correction rate ({corrected_pct:.1f}%) - monitor data quality",
            )

        # Add duration warnings
        if duration_s < 120:
            warnings.append(
                "Recording duration < 2 minutes may limit metric reliability"
            )
        elif duration_s < 300:
            warnings.append(
                "Consider longer recordings (≥5 minutes) for more stable metrics"
            )

        # Add respiratory warnings if available
        if results.respiratory:
            resp_warnings = results.respiratory.get("warnings", [])
            warnings.extend([f"Respiratory: {w}" for w in resp_warnings])

        # Display warnings with appropriate icons and colors
        for warning in warnings[:8]:  # Limit to prevent overflow
            item = QtWidgets.QListWidgetItem(str(warning))

            # Color code warnings
            warning_lower = str(warning).lower()
            if any(
                term in warning_lower
                for term in ["threshold exceeded", "high", "poor", "exceeds"]
            ):
                item.setBackground(QtGui.QColor(255, 200, 200, 150))  # Light red
                item.setIcon(
                    self.style().standardIcon(
                        QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical
                    )
                )
            elif any(
                term in warning_lower
                for term in ["moderate", "monitor", "consider", "may limit"]
            ):
                item.setBackground(QtGui.QColor(255, 255, 200, 150))  # Light yellow
                item.setIcon(
                    self.style().standardIcon(
                        QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
                    )
                )
            else:
                item.setBackground(QtGui.QColor(200, 200, 255, 150))  # Light blue
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
    """
    Enhanced signal viewer that shows original plots AND provides editing capabilities
    Combines the original functionality with manual beat editing (SRS FR-10 to FR-14)
    """

    # Signals for communication with main application
    beat_edited = pyqtSignal(str, dict)  # action, edit_info
    quality_updated = pyqtSignal(float)  # corrected_beats_percentage
    reanalysis_requested = pyqtSignal()  # Request reanalysis after edits

    def __init__(self):
        super().__init__()
        self.bundle: Optional[DataBundle] = None
        self.results: Optional[HRVAnalysisResults] = None
        self.beat_editor: Optional["BeatEditor"] = None

        # Edit state
        self.selected_beat_index = None
        self.edit_mode = "select"
        self.interpolation_method = None
        self.editing_enabled = False  # Track if editing is available

        # Plot mode toggle
        self.current_plot_mode = "overview"  # "overview" or "editing"

        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Top toolbar - combines view toggle + editing controls
        self.create_main_toolbar(layout)

        # Matplotlib figure with navigation
        self.figure = Figure(figsize=(14, 12))
        self.canvas = FigureCanvas(self.figure)
        self.navbar = NavigationToolbar(self.canvas, self)

        # Status panel
        self.create_status_panel(layout)

        # Add to layout
        layout.addWidget(self.navbar)
        layout.addWidget(self.canvas)

        # Connect matplotlib events
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def create_main_toolbar(self, layout):
        """Create main toolbar with view toggle and editing controls"""
        main_toolbar = QtWidgets.QFrame()
        main_toolbar.setFrameStyle(QtWidgets.QFrame.Shape.StyledPanel)
        main_toolbar.setMaximumHeight(80)  # Limit toolbar height
        toolbar_layout = QtWidgets.QHBoxLayout(main_toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # View Mode Toggle
        view_group = QtWidgets.QGroupBox("View Mode")
        view_group.setMaximumHeight(60)
        view_layout = QtWidgets.QHBoxLayout(view_group)
        view_layout.setContentsMargins(5, 5, 5, 5)

        self.view_toggle = QtWidgets.QButtonGroup()

        self.overview_btn = QtWidgets.QRadioButton("Analysis Overview")
        self.overview_btn.setChecked(True)
        self.overview_btn.setToolTip("Show comprehensive analysis plots")

        self.editing_btn = QtWidgets.QRadioButton("Beat Editing")
        self.editing_btn.setToolTip("Interactive RR interval editing mode")

        self.view_toggle.addButton(self.overview_btn, 0)
        self.view_toggle.addButton(self.editing_btn, 1)

        view_layout.addWidget(self.overview_btn)
        view_layout.addWidget(self.editing_btn)

        # Editing Controls (initially hidden)
        self.editing_controls = self.create_editing_controls()
        self.editing_controls.setVisible(False)
        self.editing_controls.setMaximumHeight(60)

        toolbar_layout.addWidget(view_group)
        toolbar_layout.addWidget(self.editing_controls)
        toolbar_layout.addStretch()

        layout.addWidget(main_toolbar)

        # Connect view toggle
        self.view_toggle.buttonClicked.connect(self._on_view_mode_changed)

    def create_editing_controls(self):
        """Create editing controls widget"""
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)

        # Edit Mode Selection
        mode_group = QtWidgets.QGroupBox("Edit Mode")
        mode_group.setMaximumHeight(60)
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        mode_layout.setContentsMargins(3, 3, 3, 3)

        self.mode_group = QtWidgets.QButtonGroup()

        self.select_btn = QtWidgets.QRadioButton("Select")
        self.select_btn.setChecked(True)
        self.delete_btn = QtWidgets.QRadioButton("Delete")
        self.move_btn = QtWidgets.QRadioButton("Move")
        self.interpolate_btn = QtWidgets.QRadioButton("Interpolate")
        self.insert_btn = QtWidgets.QRadioButton("Insert")

        # Make buttons smaller
        for btn in [
            self.select_btn,
            self.delete_btn,
            self.move_btn,
            self.interpolate_btn,
            self.insert_btn,
        ]:
            btn.setStyleSheet("QRadioButton { font-size: 9px; }")

        self.mode_group.addButton(self.select_btn, 0)
        self.mode_group.addButton(self.delete_btn, 1)
        self.mode_group.addButton(self.move_btn, 2)
        self.mode_group.addButton(self.interpolate_btn, 3)
        self.mode_group.addButton(self.insert_btn, 4)

        mode_layout.addWidget(self.select_btn)
        mode_layout.addWidget(self.delete_btn)
        mode_layout.addWidget(self.move_btn)
        mode_layout.addWidget(self.interpolate_btn)
        mode_layout.addWidget(self.insert_btn)

        # Interpolation Method
        interp_group = QtWidgets.QGroupBox("Method")
        interp_group.setMaximumHeight(60)
        interp_layout = QtWidgets.QHBoxLayout(interp_group)
        interp_layout.setContentsMargins(3, 3, 3, 3)

        self.linear_rb = QtWidgets.QRadioButton("Linear")
        self.cubic_rb = QtWidgets.QRadioButton("Cubic")
        self.cubic_rb.setChecked(True)

        # Make smaller
        self.linear_rb.setStyleSheet("QRadioButton { font-size: 9px; }")
        self.cubic_rb.setStyleSheet("QRadioButton { font-size: 9px; }")

        interp_layout.addWidget(self.linear_rb)
        interp_layout.addWidget(self.cubic_rb)

        # Action Buttons
        action_group = QtWidgets.QGroupBox("Actions")
        action_group.setMaximumHeight(60)
        action_layout = QtWidgets.QHBoxLayout(action_group)
        action_layout.setContentsMargins(3, 3, 3, 3)

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 9px; padding: 2px; }"
        )

        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.setStyleSheet("QPushButton { font-size: 9px; padding: 2px; }")

        self.reset_btn = QtWidgets.QPushButton("Reset All")
        self.reset_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-size: 9px; padding: 2px; s}"
        )

        self.reanalyze_btn = QtWidgets.QPushButton("Reanalyze")
        self.reanalyze_btn.setEnabled(False)
        self.reanalyze_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-size: 9px; padding: 2px;}"
        )

        action_layout.addWidget(self.apply_btn)
        action_layout.addWidget(self.undo_btn)
        action_layout.addWidget(self.reset_btn)
        action_layout.addWidget(self.reanalyze_btn)

        # Assemble controls
        controls_layout.addWidget(mode_group)
        controls_layout.addWidget(interp_group)
        controls_layout.addWidget(action_group)

        # Connect signals
        self.mode_group.buttonClicked.connect(self._on_edit_mode_changed)
        self.linear_rb.toggled.connect(self._on_interpolation_changed)
        self.cubic_rb.toggled.connect(self._on_interpolation_changed)
        self.apply_btn.clicked.connect(self._apply_edit)
        self.undo_btn.clicked.connect(self._undo_edit)
        self.reset_btn.clicked.connect(self._reset_edits)
        self.reanalyze_btn.clicked.connect(self._request_reanalysis)

        return controls_widget

    def create_status_panel(self, layout):
        """Create status panel"""
        status_frame = QtWidgets.QFrame()
        status_frame.setFrameStyle(QtWidgets.QFrame.Shape.StyledPanel)
        status_frame.setMaximumHeight(35)  # Limit status panel height
        status_layout = QtWidgets.QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)  # Smaller margins

        self.status_label = QtWidgets.QLabel("Load data to begin")
        self.status_label.setStyleSheet("font-size: 9px;")

        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setStyleSheet("font-size: 9px; font-weight: bold;")

        self.quality_label = QtWidgets.QLabel("")
        self.quality_label.setStyleSheet("font-size: 9px; font-weight: bold;")

        status_layout.addWidget(QtWidgets.QLabel("Status:"))
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.stats_label)
        status_layout.addStretch()
        status_layout.addWidget(self.quality_label)

        layout.addWidget(status_frame)

    def update_display(self, bundle: DataBundle, results: HRVAnalysisResults):
        """Update display with new data"""
        self.bundle = bundle
        self.results = results

        # Try to initialize beat editor
        if bundle and bundle.rri_ms:
            try:
                from hrvlib.beat_editor import BeatEditor, InterpolationMethod

                self.beat_editor = BeatEditor(bundle.rri_ms, user_id="gui_user")
                self.editing_enabled = True
                self.editing_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)

                # Set interpolation method
                if self.cubic_rb.isChecked():
                    self.interpolation_method = InterpolationMethod.CUBIC_SPLINE
                else:
                    self.interpolation_method = InterpolationMethod.LINEAR

            except ImportError:
                self.editing_enabled = False
                self.editing_btn.setEnabled(False)
                self.editing_btn.setToolTip(
                    "Beat editing not available - beat_editor module not found"
                )

        # Update plot based on current mode
        self._update_plot()
        self._update_status_display()

    def _on_view_mode_changed(self, button):
        """Handle view mode toggle"""
        if self.view_toggle.id(button) == 0:  # Overview
            self.current_plot_mode = "overview"
            self.editing_controls.setVisible(False)
        else:  # Editing
            self.current_plot_mode = "editing"
            if self.editing_enabled:
                self.editing_controls.setVisible(True)
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Editing Not Available",
                    "Beat editing is not available. The beat_editor module may not be installed.",
                )
                self.overview_btn.setChecked(True)
                return

        self._update_plot()

    def _update_plot(self):
        """Update plot based on current mode"""
        if not self.bundle:
            return

        if self.current_plot_mode == "overview":
            self._plot_overview()
        else:
            self._plot_editing_view()

    def _plot_overview(self):
        """Plot the comprehensive analysis overview with proper spacing and fixed band powers"""
        if not self.bundle or not self.results:
            return

        # Create a more comprehensive overview with better spacing
        self.figure.clear()

        # Create a 3x3 grid with better spacing
        gs = self.figure.add_gridspec(
            3, 3, hspace=0.6, wspace=0.4, top=0.96, bottom=0.04, left=0.06, right=0.97
        )

        try:
            # Row 1: RR intervals (spans 2 columns) + HR distribution
            ax1 = self.figure.add_subplot(gs[0, :2])  # RR intervals - spans 2 columns
            ax2 = self.figure.add_subplot(gs[0, 2])  # HR distribution

            # Row 2: PSD (spans 2 columns) + Band powers
            ax3 = self.figure.add_subplot(gs[1, :2])  # PSD - spans 2 columns
            ax4 = self.figure.add_subplot(gs[1, 2])  # Band powers

            # Row 3: Poincare + DFA + Quality
            ax5 = self.figure.add_subplot(gs[2, 0])  # Poincare
            ax6 = self.figure.add_subplot(gs[2, 1])  # DFA
            ax7 = self.figure.add_subplot(gs[2, 2])  # Quality

            # Plot 1: RR intervals over time
            if self.bundle.rri_ms:
                rr_data = np.array(self.bundle.rri_ms)
                time_data = np.cumsum(rr_data) / 1000.0
                ax1.plot(time_data, rr_data, "b-", linewidth=1.2, alpha=0.7)
                ax1.set_xlabel("Time (s)", fontsize=8)
                ax1.set_ylabel("RR Interval (ms)", fontsize=8)
                ax1.set_title("RR Intervals Over Time", fontsize=9, pad=8, loc="left")
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(labelsize=7)

                # Add preprocessing info if available
                if self.results.preprocessing_stats:
                    corrected = self.results.preprocessing_stats.get(
                        "artifacts_corrected", 0
                    )
                    if corrected > 0:
                        ax1.text(
                            0.02,
                            0.98,
                            f"Corrected: {corrected} beats",
                            transform=ax1.transAxes,
                            fontsize=7,
                            verticalalignment="top",
                            bbox=dict(
                                boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6
                            ),
                        )

            # Plot 2: HR Distribution
            if self.bundle.rri_ms:
                hr_data = 60000.0 / np.array(self.bundle.rri_ms)
                ax2.hist(hr_data, bins=15, alpha=0.7, color="red", edgecolor="black")
                ax2.set_xlabel("HR (bpm)", fontsize=8)
                ax2.set_ylabel("Frequency", fontsize=8)
                ax2.set_title("HR Distribution", fontsize=9, pad=8)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(labelsize=7)

                # Add statistics - positioned better
                mean_hr = np.mean(hr_data)
                std_hr = np.std(hr_data)
                ax2.text(
                    0.98,
                    0.98,
                    f"μ: {mean_hr:.1f}\nσ: {std_hr:.1f}",
                    transform=ax2.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.6
                    ),
                )

            # Plot 3: Power Spectral Density
            if (
                self.results.frequency_domain
                and "psd_frequencies" in self.results.frequency_domain
                and "psd_power" in self.results.frequency_domain
            ):

                freqs = self.results.frequency_domain["psd_frequencies"]
                power = self.results.frequency_domain["psd_power"]

                if (
                    freqs is not None
                    and power is not None
                    and len(freqs) > 0
                    and len(power) > 0
                ):
                    ax3.semilogy(freqs, power, "b-", linewidth=1.5)
                    ax3.axvspan(0.04, 0.15, alpha=0.2, color="green", label="LF")
                    ax3.axvspan(0.15, 0.4, alpha=0.2, color="red", label="HF")
                    ax3.set_xlabel("Frequency (Hz)", fontsize=8)
                    ax3.set_ylabel("Power (ms²/Hz)", fontsize=8)
                    ax3.set_title("Power Spectral Density", fontsize=9, pad=8)
                    ax3.legend(fontsize=7, loc="upper right")
                    ax3.grid(True, alpha=0.3)
                    ax3.tick_params(labelsize=7)

            # Plot 4: Band Powers - FIXED
            if self.results.frequency_domain:

                # Check for different possible key names
                band_data = []
                for band_key, label, color in [
                    # Try multiple possible key variations
                    ("VLF_power", "VLF", "purple"),
                    ("vlf_power", "VLF", "purple"),
                    ("LF_power", "LF", "green"),
                    ("lf_power", "LF", "green"),
                    ("HF_power", "HF", "red"),
                    ("hf_power", "HF", "red"),
                    ("total_power", "Total", "blue"),
                ]:
                    if band_key in self.results.frequency_domain:
                        value = self.results.frequency_domain[band_key]
                        if isinstance(value, (int, float)) and value > 0:
                            band_data.append((label, value, color))

                if band_data:
                    labels, powers, colors = zip(*band_data)
                    bars = ax4.bar(labels, powers, color=colors, alpha=0.7)
                    ax4.set_ylabel("Power (ms²)", fontsize=8)
                    ax4.set_title("Frequency Bands", fontsize=9, pad=8)
                    ax4.grid(True, alpha=0.3, axis="y")
                    ax4.tick_params(labelsize=7)

                    # Add values on bars with better positioning
                    max_power = max(powers)
                    for bar, power in zip(bars, powers):
                        height = bar.get_height()
                        ax4.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + max_power * 0.02,
                            f"{power:.0f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                else:
                    ax4.text(
                        0.5,
                        0.5,
                        "No frequency\nband data\navailable",
                        ha="center",
                        va="center",
                        transform=ax4.transAxes,
                        fontsize=9,
                    )
                    ax4.set_title("Frequency Bands", fontsize=10, pad=10)

            # Plot 5: Poincare Plot
            if self.bundle.rri_ms and len(self.bundle.rri_ms) > 1:
                rr_data = np.array(self.bundle.rri_ms)
                x = rr_data[:-1]
                y = rr_data[1:]
                ax5.scatter(x, y, alpha=0.6, s=6, c="blue")
                ax5.set_xlabel("RRₙ (ms)", fontsize=8)
                ax5.set_ylabel("RRₙ₊₁ (ms)", fontsize=8)
                ax5.set_title("Poincaré Plot", fontsize=9, pad=8)
                ax5.tick_params(labelsize=7)

                # Add SD1/SD2 info if available - better positioning
                if (
                    self.results.nonlinear
                    and "poincare" in self.results.nonlinear
                    and self.results.nonlinear["poincare"]
                ):
                    poincare = self.results.nonlinear["poincare"]
                    sd1 = poincare.get("sd1", 0)
                    sd2 = poincare.get("sd2", 0)
                    ax5.text(
                        0.02,
                        0.02,
                        f"SD1: {sd1:.1f}\nSD2: {sd2:.1f}",
                        transform=ax5.transAxes,
                        fontsize=7,
                        verticalalignment="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.6
                        ),
                    )

                ax5.set_xlabel("RRₙ (ms)", fontsize=9)
                ax5.set_ylabel("RRₙ₊₁ (ms)", fontsize=9)
                ax5.set_title("Poincaré Plot", fontsize=10, pad=10)
                ax5.grid(True, alpha=0.3)
                ax5.tick_params(labelsize=8)

            # Plot 6: DFA Analysis - better layout
            if self.results.nonlinear and "dfa" in self.results.nonlinear:
                dfa_data = self.results.nonlinear["dfa"]

                # Get raw DFA data first
                box_sizes = dfa_data.get("box_sizes")
                fluctuations = dfa_data.get("fluctuations")
                alpha1 = dfa_data.get("alpha1")
                alpha2 = dfa_data.get("alpha2")

                plotted_raw_data = False

                # Plot the actual DFA curve (blue points) if available
                if (
                    box_sizes is not None
                    and fluctuations is not None
                    and len(box_sizes) == len(fluctuations)
                    and len(box_sizes) > 0
                ):

                    # Convert to numpy arrays and filter valid data
                    box_sizes = np.array(box_sizes)
                    fluctuations = np.array(fluctuations)

                    valid_mask = (
                        (box_sizes > 0)
                        & (fluctuations > 0)
                        & np.isfinite(box_sizes)
                        & np.isfinite(fluctuations)
                    )

                    if np.any(valid_mask):
                        valid_boxes = box_sizes[valid_mask]
                        valid_flucts = fluctuations[valid_mask]

                        # Plot the blue curve (same as PDF)
                        ax6.loglog(
                            valid_boxes,
                            valid_flucts,
                            "bo-",
                            markersize=3,
                            linewidth=1,
                            alpha=0.7,
                            label="DFA curve",
                        )
                        plotted_raw_data = True

                # Add scaling exponent lines (same as before but conditional)
                if alpha1 is not None and not np.isnan(alpha1):
                    short_scales = np.logspace(np.log10(4), np.log10(16), 20)
                    if plotted_raw_data:
                        # If we have raw data, fit the line to it
                        short_mask = (valid_boxes >= 4) & (valid_boxes <= 16)
                        if np.any(short_mask):
                            # Use actual data range for scaling
                            short_flucts = short_scales**alpha1
                            # Scale to match data
                            if len(valid_flucts[short_mask]) > 0:
                                scale_factor = np.mean(
                                    valid_flucts[short_mask]
                                ) / np.mean(short_flucts)
                                short_flucts *= scale_factor
                        else:
                            short_flucts = short_scales**alpha1 * 10
                    else:
                        short_flucts = short_scales**alpha1 * 10

                    ax6.loglog(
                        short_scales,
                        short_flucts,
                        "g-",
                        linewidth=2,
                        label=f"α₁={alpha1:.3f}",
                        alpha=0.8,
                    )

                if alpha2 is not None and not np.isnan(alpha2):
                    long_scales = np.logspace(np.log10(16), np.log10(64), 20)
                    if plotted_raw_data:
                        # Connect with actual data
                        long_mask = valid_boxes >= 16
                        if np.any(long_mask):
                            long_flucts = long_scales**alpha2
                            # Scale to match data
                            if len(valid_flucts[long_mask]) > 0:
                                scale_factor = np.mean(
                                    valid_flucts[long_mask]
                                ) / np.mean(long_flucts)
                                long_flucts *= scale_factor
                        else:
                            long_flucts = long_scales**alpha2 * 20
                    else:
                        # No raw data, create synthetic
                        if alpha1 is not None:
                            connection_point = (16**alpha1) * 10
                            long_flucts = (long_scales**alpha2) * (
                                connection_point / (16**alpha2)
                            )
                        else:
                            long_flucts = long_scales**alpha2 * 20

                    ax6.loglog(
                        long_scales,
                        long_flucts,
                        "r-",
                        linewidth=2,
                        label=f"α₂={alpha2:.3f}",
                        alpha=0.8,
                    )

                # Set labels and formatting
                ax6.set_xlabel("Window size", fontsize=8)
                ax6.set_ylabel("Fluctuation", fontsize=8)
                ax6.set_title("DFA Analysis", fontsize=9, pad=8)
                if plotted_raw_data or alpha1 is not None or alpha2 is not None:
                    ax6.legend(fontsize=7, loc="lower right")
                ax6.grid(True, alpha=0.3)
                ax6.tick_params(labelsize=7)

                # Debug print to see what data is available
                print(f"DFA Debug - box_sizes available: {box_sizes is not None}")
                print(f"DFA Debug - fluctuations available: {fluctuations is not None}")
                if box_sizes is not None:
                    print(f"DFA Debug - box_sizes length: {len(box_sizes)}")
                    print(
                        f"DFA Debug - fluctuations length: {len(fluctuations) if fluctuations is not None else 'None'}"
                    )

            else:
                ax6.text(
                    0.5,
                    0.5,
                    "DFA data\nnot available",
                    ha="center",
                    va="center",
                    transform=ax6.transAxes,
                    fontsize=8,
                )
                ax6.set_title("DFA Analysis", fontsize=9, pad=8)

            # Plot 7: Quality Assessment - improved layout
            quality_data = self.results.quality_assessment or {}
            preprocessing_stats = self.results.preprocessing_stats or {}

            # Collect quality metrics
            metrics_data = {}

            corrected_pct = (
                preprocessing_stats.get("corrected_beats_percentage", 0)
                or preprocessing_stats.get("artifact_percentage", 0)
                or quality_data.get("artifact_percentage", 0)
            )
            if corrected_pct > 0:
                metrics_data["Corrected\n(%)"] = corrected_pct

            duration = quality_data.get("duration_s")
            if duration:
                metrics_data["Duration\n(min)"] = duration / 60.0

            # Plot quality metrics
            if metrics_data:
                labels = list(metrics_data.keys())
                values = list(metrics_data.values())

                # Color code based on quality thresholds
                colors = []
                for label, value in zip(labels, values):
                    if "Corrected" in label:
                        if value > 5:
                            colors.append("red")
                        elif value > 2:
                            colors.append("orange")
                        else:
                            colors.append("green")
                    else:
                        colors.append("blue")

                bars = ax7.bar(labels, values, color=colors, alpha=0.7)
                ax7.set_title("Quality Metrics", fontsize=10, pad=10)
                ax7.grid(True, alpha=0.3, axis="y")
                ax7.tick_params(labelsize=8)

                # Add values on bars with better spacing
                max_val = max(values) if values else 1
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax7.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max_val * 0.02,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            else:
                ax7.text(
                    0.5,
                    0.5,
                    "Quality metrics\nnot available",
                    ha="center",
                    va="center",
                    transform=ax7.transAxes,
                    fontsize=9,
                )
                ax7.set_title("Quality Assessment", fontsize=10, pad=10)

            # Add main title with better positioning
            self.figure.suptitle(
                "Comprehensive HRV Analysis Overview",
                fontsize=12,
                fontweight="bold",
                y=1,
            )

            self.canvas.draw()

        except Exception as e:
            # Show the full error for debugging
            import traceback

            print(f"Comprehensive plotting failed: {e}")
            print("Full traceback:")
            traceback.print_exc()

            # Fallback to simple plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            if self.bundle.rri_ms:
                rr_data = np.array(self.bundle.rri_ms)
                time_data = np.cumsum(rr_data) / 1000.0
                ax.plot(time_data, rr_data, "b-", linewidth=1.2, label="RR intervals")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("RR Interval (ms)")
                ax.set_title("RR Intervals Over Time", fontsize=9, pad=8, loc="left")
                ax.grid(True, alpha=0.3)
                ax.legend()

            self.figure.tight_layout()
            self.canvas.draw()

    def _plot_editing_view(self):
        """Plot the RR intervals for editing"""
        if not self.beat_editor:
            self._plot_overview()  # Fallback
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Get current RR data
        rr_data = self.beat_editor.get_current_rr_intervals()
        time_data = np.cumsum(rr_data) / 1000.0

        # Plot RR intervals with interactive markers
        (line,) = ax.plot(
            time_data,
            rr_data,
            "b-",
            linewidth=1.2,
            marker="o",
            markersize=4,
            alpha=0.8,
            picker=True,
            pickradius=8,
            label="RR intervals",
        )

        # Highlight edited beats
        if self.beat_editor.edit_history:
            self._highlight_edits(ax, time_data, rr_data)

        # Mark selected beat
        if self.selected_beat_index is not None and self.selected_beat_index < len(
            rr_data
        ):
            selected_time = time_data[self.selected_beat_index]
            selected_rr = rr_data[self.selected_beat_index]
            ax.plot(
                selected_time,
                selected_rr,
                "ro",
                markersize=10,
                markeredgewidth=2,
                markerfacecolor="none",
                markeredgecolor="red",
                label="Selected",
            )

        # Formatting
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RR Interval (ms)")
        ax.set_title(f"Interactive RR Editor - Mode: {self.edit_mode.title()}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add instructions
        mode_text = f"Mode: {self.edit_mode.title()}"
        if self.edit_mode == "interpolate" and self.interpolation_method:
            method = (
                "Cubic"
                if hasattr(self.interpolation_method, "value")
                and self.interpolation_method.value == "cubic_spline"
                else "Linear"
            )
            mode_text += f" ({method})"

        ax.text(
            0.02,
            0.98,
            mode_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        self.figure.tight_layout()
        self.canvas.draw()

    def _highlight_edits(self, ax, time_data, rr_data):
        """Highlight recently edited beats"""
        if len(self.beat_editor.edit_history) > 0:
            # Show last few edits
            recent_edits = self.beat_editor.edit_history[-5:]
            edit_indices = set()

            for edit in recent_edits:
                if edit.action.value != "delete":  # Skip deleted beats
                    idx = min(edit.beat_index, len(time_data) - 1)
                    edit_indices.add(idx)

            if edit_indices:
                edit_times = [time_data[i] for i in edit_indices if i < len(time_data)]
                edit_rrs = [rr_data[i] for i in edit_indices if i < len(rr_data)]
                ax.scatter(
                    edit_times,
                    edit_rrs,
                    c="yellow",
                    s=60,
                    alpha=0.7,
                    edgecolors="orange",
                    linewidth=2,
                    label="Recently Edited",
                )

    def _on_edit_mode_changed(self, button):
        """Handle edit mode change"""
        if not self.editing_enabled:
            return

        modes = ["select", "delete", "move", "interpolate", "insert"]
        self.edit_mode = modes[self.mode_group.id(button)]

        self.selected_beat_index = None
        self.apply_btn.setEnabled(False)

        if self.current_plot_mode == "editing":
            self._plot_editing_view()

        self._update_status(
            "Click on RR intervals to select beats"
            if self.edit_mode == "select"
            else f"Click on beats to {self.edit_mode}"
        )

    def _on_interpolation_changed(self):
        """Handle interpolation method change"""
        if not self.editing_enabled:
            return

        try:
            from hrvlib.beat_editor import InterpolationMethod

            if self.cubic_rb.isChecked():
                self.interpolation_method = InterpolationMethod.CUBIC_SPLINE
            else:
                self.interpolation_method = InterpolationMethod.LINEAR

            if self.current_plot_mode == "editing":
                self._plot_editing_view()
        except ImportError:
            pass

    def _on_canvas_click(self, event):
        """Handle canvas clicks for beat selection"""
        if (
            self.current_plot_mode != "editing"
            or not self.beat_editor
            or not event.inaxes
            or self.edit_mode == "select"
        ):
            return

        # Find closest beat
        rr_data = self.beat_editor.get_current_rr_intervals()
        time_data = np.cumsum(rr_data) / 1000.0

        if event.xdata is not None and len(time_data) > 0:
            distances = np.abs(time_data - event.xdata)
            closest_idx = np.argmin(distances)

            # Select if click is close enough
            if distances[closest_idx] < (time_data[-1] - time_data[0]) * 0.02:
                self.selected_beat_index = closest_idx
                self.apply_btn.setEnabled(True)

                self._update_status(
                    f"Selected beat {closest_idx} at t={time_data[closest_idx]:.1f}s"
                )
                self._plot_editing_view()

    def _apply_edit(self):
        """Apply current edit"""
        if not self.beat_editor or self.selected_beat_index is None:
            return

        success = False
        edit_info = {"beat_index": self.selected_beat_index, "action": self.edit_mode}

        try:
            if self.edit_mode == "delete":
                success = self.beat_editor.delete_beat(self.selected_beat_index)
            elif self.edit_mode == "move":
                target_index, ok = QtWidgets.QInputDialog.getInt(
                    self,
                    "Move Beat",
                    f"Move beat from index {self.selected_beat_index} to index:",
                    self.selected_beat_index,
                    0,
                    len(self.beat_editor.get_current_rr_intervals()) - 1,
                )
                if ok:
                    success = self.beat_editor.move_beat(
                        self.selected_beat_index, target_index
                    )
                    edit_info["target_index"] = target_index
            elif self.edit_mode == "interpolate":
                success = self.beat_editor.interpolate_beat(
                    self.selected_beat_index, self.interpolation_method
                )
                edit_info["interpolation_method"] = (
                    self.interpolation_method.value
                    if self.interpolation_method
                    else "linear"
                )
            elif self.edit_mode == "insert":
                success = self.beat_editor.insert_beat(
                    self.selected_beat_index, self.interpolation_method
                )
                edit_info["interpolation_method"] = (
                    self.interpolation_method.value
                    if self.interpolation_method
                    else "linear"
                )

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Edit Error", f"Failed to {self.edit_mode}:\n{str(e)}"
            )
            return

        if success:
            self.selected_beat_index = None
            self.apply_btn.setEnabled(False)
            self.undo_btn.setEnabled(True)
            self.reanalyze_btn.setEnabled(True)

            self._plot_editing_view()
            self._update_status_display()

            # Emit signals
            self.beat_edited.emit(self.edit_mode, edit_info)

            # Check quality threshold
            corrected_pct = self.beat_editor.get_corrected_beats_percentage()
            self.quality_updated.emit(corrected_pct)

            if corrected_pct > 5.0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Quality Warning",
                    f"Corrected beats: {corrected_pct:.1f}% exceeds 5% threshold.\n"
                    "Consider using different data or reviewing signal quality.",
                )

    def _undo_edit(self):
        """Undo last edit"""
        if not self.beat_editor:
            return

        if self.beat_editor.undo_last_edit():
            self.selected_beat_index = None
            self.apply_btn.setEnabled(False)

            if not self.beat_editor.edit_history:
                self.undo_btn.setEnabled(False)
                self.reanalyze_btn.setEnabled(False)

            self._plot_editing_view()
            self._update_status_display()
            self.beat_edited.emit("undo", {})
        else:
            self._update_status("No edits to undo")

    def _reset_edits(self):
        """Reset all edits"""
        if not self.beat_editor or not self.beat_editor.edit_history:
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Reset All Edits",
            f"Reset all {len(self.beat_editor.edit_history)} edits?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.beat_editor.reset_all_edits()
            self.selected_beat_index = None
            self.apply_btn.setEnabled(False)
            self.undo_btn.setEnabled(False)
            self.reanalyze_btn.setEnabled(False)

            self._plot_editing_view()
            self._update_status_display()
            self.beat_edited.emit("reset", {})

    def _request_reanalysis(self):
        """Request reanalysis"""
        if self.beat_editor and self.beat_editor.edit_history:
            edited_rr = self.beat_editor.get_current_rr_intervals()
            if self.bundle:
                self.bundle.rri_ms = edited_rr.tolist()
            self.reanalysis_requested.emit()

    def _update_status(self, message: str):
        """Update status label"""
        self.status_label.setText(message)

    def _update_status_display(self):
        """Update all status displays"""
        if self.editing_enabled and self.beat_editor:
            stats = self.beat_editor.get_edit_statistics()

            # Update statistics
            if stats["total_edits"] > 0:
                stats_text = (
                    f"Edits: {stats['total_edits']} "
                    f"(Del: {stats['deleted_beats']}, Int: {stats['interpolated_beats']})"
                )
                self.stats_label.setText(stats_text)
            else:
                self.stats_label.setText("")

            # Update quality
            corrected_pct = stats["corrected_beats_percentage"]
            if corrected_pct > 5.0:
                color, text = "#f44336", f"Quality: POOR ({corrected_pct:.1f}%)"
            elif corrected_pct > 2.0:
                color, text = "#ff9800", f"Quality: FAIR ({corrected_pct:.1f}%)"
            elif corrected_pct > 0:
                color, text = "#4caf50", f"Quality: GOOD ({corrected_pct:.1f}%)"
            else:
                color, text = "", ""

            self.quality_label.setText(text)
            self.quality_label.setStyleSheet(
                f"color: {color}; font-weight: bold;" if color else ""
            )
        else:
            if self.current_plot_mode == "overview":
                self._update_status("Showing analysis overview")
            else:
                self._update_status("Beat editing not available")

    # Interface methods for compatibility
    def get_audit_trail(self) -> Dict[str, Any]:
        """Get audit trail for export"""
        if self.beat_editor:
            return {
                "audit_trail": self.beat_editor.get_audit_trail(),
                "edit_statistics": self.beat_editor.get_edit_statistics(),
                "original_rr_count": len(self.beat_editor.get_original_rr_intervals()),
                "current_rr_count": len(self.beat_editor.get_current_rr_intervals()),
            }
        return {}

    def get_edited_rr_intervals(self) -> Optional[np.ndarray]:
        """Get edited RR intervals"""
        if self.beat_editor:
            return self.beat_editor.get_current_rr_intervals()
        return None

    def has_edits(self) -> bool:
        """Check if edits were made"""
        return self.beat_editor is not None and len(self.beat_editor.edit_history) > 0


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
    """Enhanced session manager for SRS compliance with manual editing support"""

    @staticmethod
    def save_session(
        file_path: str,
        analysis_parameters: Dict,
        edit_history: list,
        file_info: str,
        results: Optional[HRVAnalysisResults] = None,
        bundle: Optional[DataBundle] = None,
        signal_viewer_data: Optional[Dict] = None,
    ) -> bool:
        """
        Save comprehensive session including manual edits (SRS compliant)

        Args:
            file_path: Path to save session file
            analysis_parameters: Analysis configuration
            edit_history: List of manual edits performed
            file_info: Source file information
            results: Analysis results (optional)
            bundle: Data bundle with current state (optional)
            signal_viewer_data: Editing state from signal viewer (optional)
        """
        try:
            # Core session data
            session_data = {
                "session_info": {
                    "version": "2.1.1",
                    "timestamp": datetime.now().isoformat(),
                    "srs_compliance": "Manual editing with audit trail (FR-10 to FR-14)",
                },
                "source_file": {
                    "path": file_info,
                    "loaded_timestamp": datetime.now().isoformat(),
                },
                "analysis_parameters": analysis_parameters,
                "manual_editing": {
                    "has_edits": len(edit_history) > 0,
                    "edit_count": len(
                        [
                            e
                            for e in edit_history
                            if e.get("action") not in ["undo", "reset"]
                        ]
                    ),
                    "edit_history": edit_history,
                    "last_edit_timestamp": (
                        edit_history[-1].get("timestamp") if edit_history else None
                    ),
                },
            }

            # Add current RR intervals if bundle available (including edits)
            if bundle and bundle.rri_ms:
                session_data["rr_data"] = {
                    "rr_intervals_ms": bundle.rri_ms,
                    "sample_count": len(bundle.rri_ms),
                    "duration_s": sum(bundle.rri_ms) / 1000.0 if bundle.rri_ms else 0,
                }

                # Include original metadata
                if hasattr(bundle, "meta") and bundle.meta:
                    session_data["rr_data"]["metadata"] = bundle.meta.copy()

            # Add detailed audit trail from signal viewer
            if signal_viewer_data:
                session_data["detailed_audit_trail"] = signal_viewer_data

            # Add analysis results summary (not full results to keep file size manageable)
            if results:
                session_data["results_summary"] = (
                    SessionManager._serialize_results_summary(results)
                )

            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, default=str)

            return True

        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    @staticmethod
    def load_session(file_path: str) -> Optional[Dict[str, Any]]:
        """Load session with validation"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # Validate session format
            if not SessionManager._validate_session(session_data):
                print("Invalid session format")
                return None

            return session_data

        except Exception as e:
            print(f"Error loading session: {e}")
            return None

    @staticmethod
    def _validate_session(session_data: Dict) -> bool:
        """Validate session data structure"""
        required_keys = ["session_info", "analysis_parameters"]
        return all(key in session_data for key in required_keys)

    @staticmethod
    def _serialize_results_summary(results: HRVAnalysisResults) -> Dict[str, Any]:
        """Create compact summary of results for session storage"""
        summary = {
            "computed_domains": {
                "time_domain": results.time_domain is not None,
                "frequency_domain": results.frequency_domain is not None,
                "nonlinear": results.nonlinear is not None,
            },
            "warnings_count": len(results.warnings) if results.warnings else 0,
            "quality_summary": (
                results.quality_assessment.get("overall_quality")
                if results.quality_assessment
                else "unknown"
            ),
        }

        # Add key metrics for validation
        if results.time_domain:
            summary["key_metrics"] = {
                "sdnn": results.time_domain.get("sdnn"),
                "rmssd": results.time_domain.get("rmssd"),
                "mean_rr": results.time_domain.get("mean_rr"),
            }

        return summary
