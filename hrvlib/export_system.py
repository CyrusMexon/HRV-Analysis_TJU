"""
Complete Export System - SRS Compliant (FR-24 to FR-30)
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Try pandas import with fallback
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from hrvlib.data_handler import DataBundle
from hrvlib.pipeline import HRVAnalysisResults
from hrvlib.ui.plots import create_summary_figure


class HRVExporter:
    """Complete HRV export system implementing SRS requirements FR-24 to FR-30"""

    def __init__(
        self,
        bundle: DataBundle,
        results: HRVAnalysisResults,
        analysis_parameters: Dict[str, Any],
    ):
        self.bundle = bundle
        self.results = results
        self.analysis_parameters = analysis_parameters
        self.export_timestamp = datetime.now()

    def export_all(
        self, export_settings: Dict[str, Any], base_path: str
    ) -> Dict[str, str]:
        """Export results in all requested formats"""
        exported_files = {}

        # Add timestamp to filename if requested
        if export_settings.get("options", {}).get("include_timestamp", True):
            timestamp_str = self.export_timestamp.strftime("%Y%m%d_%H%M%S")
            base_path = f"{base_path}_{timestamp_str}"

        formats = export_settings.get("formats", {})

        try:
            # PDF Report (FR-24)
            if formats.get("pdf", False):
                pdf_path = f"{base_path}.pdf"
                self._export_pdf_report(pdf_path, export_settings)
                exported_files["PDF Report"] = pdf_path

            # CSV Metrics (FR-25)
            if formats.get("csv", False):
                csv_path = f"{base_path}_metrics.csv"
                self._export_csv_metrics(csv_path, export_settings)
                exported_files["CSV Metrics"] = csv_path

            # SPSS-compatible CSV (FR-25)
            if formats.get("spss", False):
                spss_path = f"{base_path}_SPSS.csv"
                self._export_spss_csv(spss_path, export_settings)
                exported_files["SPSS CSV"] = spss_path

            # Audit Trail Log (FR-30)
            if formats.get("audit_trail", False):
                audit_path = f"{base_path}_audit_trail.json"
                self._export_audit_trail(audit_path, export_settings)
                exported_files["Audit Trail"] = audit_path

        except Exception as e:
            raise Exception(f"Export failed: {str(e)}")

        return exported_files

    def _export_pdf_report(
        self, file_path: str, export_settings: Dict[str, Any]
    ) -> None:
        """Export comprehensive PDF report (FR-24)"""
        content_options = export_settings.get("content", {})

        with PdfPages(file_path) as pdf:
            # Page 1: Summary figure with plots
            if content_options.get("include_plots", True):
                try:
                    summary_fig = create_summary_figure(self.bundle, self.results)
                    pdf.savefig(summary_fig, bbox_inches="tight")
                    plt.close(summary_fig)
                except Exception as e:
                    print(f"Warning: Could not create summary figure: {e}")
                    # Create a simple text page instead
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.text(
                        0.5,
                        0.5,
                        "HRV Analysis Results\n\nPlots could not be generated.\nSee detailed metrics on next page.",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=16,
                    )
                    ax.axis("off")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            # Page 2: Detailed metrics
            metrics_fig = self._create_metrics_page(content_options)
            pdf.savefig(metrics_fig, bbox_inches="tight")
            plt.close(metrics_fig)

            # Page 3: Quality assessment (if requested)
            if content_options.get(
                "include_preprocessing", True
            ) or content_options.get("include_quality", True):
                quality_fig = self._create_quality_page(content_options)
                pdf.savefig(quality_fig, bbox_inches="tight")
                plt.close(quality_fig)

    def _create_metrics_page(self, content_options: Dict[str, Any]) -> plt.Figure:
        """Create detailed metrics page for PDF"""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("HRV Analysis - Detailed Metrics", fontsize=16, fontweight="bold")

        axes = axes.flatten()

        # Time domain metrics
        axes[0].axis("off")
        axes[0].set_title("Time Domain Metrics", fontweight="bold", pad=20)

        if self.results.time_domain:
            time_metrics = []
            for metric, value in self.results.time_domain.items():
                if isinstance(value, (int, float)) and not metric.startswith(
                    "analysis_"
                ):
                    unit = self._get_metric_unit(metric)
                    time_metrics.append(
                        [self._format_metric_name(metric), f"{value:.2f}", unit]
                    )

            if time_metrics:
                table1 = axes[0].table(
                    cellText=time_metrics[:10],
                    colLabels=["Metric", "Value", "Unit"],
                    cellLoc="left",
                    loc="center",
                )
                table1.auto_set_font_size(False)
                table1.set_fontsize(9)
                table1.scale(1, 1.2)

        # Frequency domain metrics
        axes[1].axis("off")
        axes[1].set_title("Frequency Domain Metrics", fontweight="bold", pad=20)

        if self.results.frequency_domain:
            freq_metrics = []
            for metric, value in self.results.frequency_domain.items():
                if isinstance(value, (int, float)) and not metric.startswith(
                    ("analysis_", "psd_")
                ):
                    unit = self._get_metric_unit(metric)
                    freq_metrics.append(
                        [self._format_metric_name(metric), f"{value:.3f}", unit]
                    )

            if freq_metrics:
                table2 = axes[1].table(
                    cellText=freq_metrics[:10],
                    colLabels=["Metric", "Value", "Unit"],
                    cellLoc="left",
                    loc="center",
                )
                table2.auto_set_font_size(False)
                table2.set_fontsize(9)
                table2.scale(1, 1.2)

        # Nonlinear metrics
        axes[2].axis("off")
        axes[2].set_title("Nonlinear Metrics", fontweight="bold", pad=20)

        if self.results.nonlinear:
            nonlinear_metrics = []

            # Poincaré metrics
            if (
                "poincare" in self.results.nonlinear
                and self.results.nonlinear["poincare"]
            ):
                poincare = self.results.nonlinear["poincare"]
                for metric, value in poincare.items():
                    if isinstance(value, (int, float)):
                        unit = self._get_metric_unit(metric)
                        nonlinear_metrics.append(
                            [self._format_metric_name(metric), f"{value:.3f}", unit]
                        )

            # DFA metrics
            if "dfa" in self.results.nonlinear and self.results.nonlinear["dfa"]:
                dfa = self.results.nonlinear["dfa"]
                for metric, value in dfa.items():
                    if isinstance(value, (int, float)):
                        nonlinear_metrics.append([f"DFA {metric}", f"{value:.3f}", ""])

            # Entropy metrics
            if "sample_entropy" in self.results.nonlinear:
                value = self.results.nonlinear["sample_entropy"]
                if isinstance(value, (int, float)):
                    nonlinear_metrics.append(["Sample Entropy", f"{value:.3f}", ""])

            if "approximate_entropy" in self.results.nonlinear:
                value = self.results.nonlinear["approximate_entropy"]
                if isinstance(value, (int, float)):
                    nonlinear_metrics.append(["Approximate Entropy", f"{value:.3f}", ""])

            if nonlinear_metrics:
                table3 = axes[2].table(
                    cellText=nonlinear_metrics[:8],
                    colLabels=["Metric", "Value", "Unit"],
                    cellLoc="left",
                    loc="center",
                )
                table3.auto_set_font_size(False)
                table3.set_fontsize(9)
                table3.scale(1, 1.2)

        # Quality summary
        axes[3].axis("off")
        axes[3].set_title("Quality Summary", fontweight="bold", pad=20)

        quality_info = []
        if self.results.preprocessing_stats:
            quality_info.append(
                [
                    "Artifacts Corrected",
                    f"{self.results.preprocessing_stats.get('artifacts_corrected', 0)}",
                ]
            )
            quality_info.append(
                [
                    "Artifact %",
                    f"{self.results.preprocessing_stats.get('artifact_percentage', 0):.1f}%",
                ]
            )

        if self.results.quality_assessment:
            quality_info.append(
                [
                    "Overall Quality",
                    str(
                        self.results.quality_assessment.get(
                            "overall_quality", "Unknown"
                        )
                    ),
                ]
            )
            quality_info.append(
                [
                    "Duration",
                    f"{self.results.quality_assessment.get('duration_s', 0):.1f} s",
                ]
            )

        if quality_info:
            table4 = axes[3].table(
                cellText=quality_info,
                colLabels=["Parameter", "Value"],
                cellLoc="left",
                loc="center",
            )
            table4.auto_set_font_size(False)
            table4.set_fontsize(10)
            table4.scale(1, 1.5)

        plt.tight_layout()
        return fig

    def _create_quality_page(self, content_options: Dict[str, Any]) -> plt.Figure:
        """Create quality assessment page"""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(
            "Quality Assessment & Analysis Details", fontsize=16, fontweight="bold"
        )

        axes = axes.flatten()

        # Preprocessing details
        axes[0].axis("off")
        axes[0].set_title("Preprocessing Summary", fontweight="bold", pad=20)

        if self.results.preprocessing_stats:
            prep_text = [
                f"Artifacts Detected: {self.results.preprocessing_stats.get('artifacts_detected', 0)}",
                f"Artifacts Corrected: {self.results.preprocessing_stats.get('artifacts_corrected', 0)}",
                f"Artifact Percentage: {self.results.preprocessing_stats.get('artifact_percentage', 0):.1f}%",
                f"Correction Method: {self.results.preprocessing_stats.get('correction_method', 'N/A')}",
                f"Noise Segments: {self.results.preprocessing_stats.get('noise_segments', 0)}",
            ]
            axes[0].text(
                0.05,
                0.95,
                "\n".join(prep_text),
                transform=axes[0].transAxes,
                fontsize=11,
                verticalalignment="top",
            )

        # Quality assessment
        axes[1].axis("off")
        axes[1].set_title("Quality Assessment", fontweight="bold", pad=20)

        if self.results.quality_assessment:
            quality_text = [
                f"Overall Quality: {self.results.quality_assessment.get('overall_quality', 'Unknown')}",
                f"Duration: {self.results.quality_assessment.get('duration_s', 0):.1f} seconds",
                f"Data Adequate: {self.results.quality_assessment.get('data_length_adequate', False)}",
            ]
            axes[1].text(
                0.05,
                0.95,
                "\n".join(quality_text),
                transform=axes[1].transAxes,
                fontsize=11,
                verticalalignment="top",
            )

        # Warnings
        axes[2].axis("off")
        axes[2].set_title("Warnings & Recommendations", fontweight="bold", pad=20)

        warnings_text = []
        if self.results.warnings:
            warnings_text.extend([f"• {w}" for w in self.results.warnings])

        if (
            self.results.quality_assessment
            and "recommendations" in self.results.quality_assessment
        ):
            warnings_text.extend(
                [f"• {r}" for r in self.results.quality_assessment["recommendations"]]
            )

        if warnings_text:
            warnings_str = "\n".join(warnings_text[:10])  # Limit to 10 warnings
        else:
            warnings_str = "No warnings or recommendations"

        axes[2].text(
            0.05,
            0.95,
            warnings_str,
            transform=axes[2].transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        # Analysis parameters summary
        axes[3].axis("off")
        axes[3].set_title("Analysis Parameters", fontweight="bold", pad=20)

        param_text = []
        for section, params in self.analysis_parameters.items():
            if isinstance(params, dict):
                param_text.append(f"{section.replace('_', ' ').title()}:")
                for key, value in list(params.items())[
                    :3
                ]:  # Limit to first 3 per section
                    param_text.append(f"  {key}: {value}")
            else:
                param_text.append(f"{section}: {params}")

        axes[3].text(
            0.05,
            0.95,
            "\n".join(param_text[:15]),
            transform=axes[3].transAxes,
            fontsize=9,
            verticalalignment="top",
        )

        plt.tight_layout()
        return fig

    def _export_csv_metrics(
        self, file_path: str, export_settings: Dict[str, Any]
    ) -> None:
        """Export metrics in standard CSV format (FR-25)"""
        decimal_places = export_settings.get("options", {}).get("decimal_places", 3)

        # Collect all metrics
        metrics_data = {}

        # Add metadata
        metrics_data["export_timestamp"] = self.export_timestamp.isoformat()
        if self.bundle.source:
            metrics_data["source_file"] = str(Path(self.bundle.source.path).name)

        # Time domain metrics
        if self.results.time_domain:
            for key, value in self.results.time_domain.items():
                if isinstance(value, (int, float)) and not key.startswith("analysis_"):
                    metrics_data[f"time_{key}"] = round(value, decimal_places)

        # Frequency domain metrics
        if self.results.frequency_domain:
            for key, value in self.results.frequency_domain.items():
                if isinstance(value, (int, float)) and not key.startswith(
                    ("analysis_", "psd_")
                ):
                    metrics_data[f"freq_{key}"] = round(value, decimal_places)

        # Nonlinear metrics - flattened
        if self.results.nonlinear:
            if (
                "poincare" in self.results.nonlinear
                and self.results.nonlinear["poincare"]
            ):
                for key, value in self.results.nonlinear["poincare"].items():
                    if isinstance(value, (int, float)):
                        metrics_data[f"poincare_{key}"] = round(value, decimal_places)

            if "dfa" in self.results.nonlinear and self.results.nonlinear["dfa"]:
                for key, value in self.results.nonlinear["dfa"].items():
                    if isinstance(value, (int, float)):
                        metrics_data[f"dfa_{key}"] = round(value, decimal_places)

            if "sample_entropy" in self.results.nonlinear:
                value = self.results.nonlinear["sample_entropy"]
                if isinstance(value, (int, float)):
                    metrics_data["sample_entropy"] = round(value, decimal_places)

            if "approximate_entropy" in self.results.nonlinear:
                value = self.results.nonlinear["approximate_entropy"]
                if isinstance(value, (int, float)):
                    metrics_data["approximate_entropy"] = round(value, decimal_places)

        # Quality metrics
        if self.results.preprocessing_stats:
            metrics_data["artifacts_detected"] = self.results.preprocessing_stats.get(
                "artifacts_detected", 0
            )
            metrics_data["artifacts_corrected"] = self.results.preprocessing_stats.get(
                "artifacts_corrected", 0
            )
            metrics_data["artifact_percentage"] = round(
                self.results.preprocessing_stats.get("artifact_percentage", 0), 2
            )

        # Write CSV
        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            if metrics_data:
                writer = csv.DictWriter(csvfile, fieldnames=metrics_data.keys())
                writer.writeheader()
                writer.writerow(metrics_data)

    def _export_spss_csv(self, file_path: str, export_settings: Dict[str, Any]) -> None:
        """Export SPSS-compatible CSV (FR-25)"""
        decimal_places = export_settings.get("options", {}).get("decimal_places", 3)

        # Create SPSS-compatible data
        spss_data = {}

        # Basic metadata
        spss_data["SubjectID"] = "Subject_001"
        spss_data["RecordingDate"] = self.export_timestamp.strftime("%Y%m%d")

        # Time domain with SPSS names
        if self.results.time_domain:
            spss_mapping = {
                "sdnn": "SDNN_ms",
                "rmssd": "RMSSD_ms",
                "pnn50": "pNN50_pct",
                "nn50": "NN50_count",
                "mean_rr": "MeanRR_ms",
                "mean_hr": "MeanHR_bpm",
                "std_hr": "STDHR_bpm",
            }

            for key, value in self.results.time_domain.items():
                if isinstance(value, (int, float)) and key in spss_mapping:
                    spss_data[spss_mapping[key]] = round(value, decimal_places)

        # Frequency domain with SPSS names
        if self.results.frequency_domain:
            freq_mapping = {
                "lf_power": "LF_Power_ms2",
                "hf_power": "HF_Power_ms2",
                "lf_hf_ratio": "LF_HF_Ratio",
            }

            for key, value in self.results.frequency_domain.items():
                if isinstance(value, (int, float)) and key in freq_mapping:
                    spss_data[freq_mapping[key]] = round(value, decimal_places)

        # Nonlinear metrics with SPSS names
        if self.results.nonlinear:
            if (
                "poincare" in self.results.nonlinear
                and self.results.nonlinear["poincare"]
            ):
                poincare = self.results.nonlinear["poincare"]
                if "sd1" in poincare and isinstance(poincare["sd1"], (int, float)):
                    spss_data["Poincare_SD1_ms"] = round(
                        poincare["sd1"], decimal_places
                    )
                if "sd2" in poincare and isinstance(poincare["sd2"], (int, float)):
                    spss_data["Poincare_SD2_ms"] = round(
                        poincare["sd2"], decimal_places
                    )

            if "dfa" in self.results.nonlinear and self.results.nonlinear["dfa"]:
                dfa = self.results.nonlinear["dfa"]
                if "alpha1" in dfa and isinstance(dfa["alpha1"], (int, float)):
                    spss_data["DFA_Alpha1"] = round(dfa["alpha1"], decimal_places)
                if "alpha2" in dfa and isinstance(dfa["alpha2"], (int, float)):
                    spss_data["DFA_Alpha2"] = round(dfa["alpha2"], decimal_places)

        # Quality metrics
        if self.results.preprocessing_stats:
            spss_data["Artifact_Percentage"] = round(
                self.results.preprocessing_stats.get("artifact_percentage", 0), 2
            )

        # Write SPSS CSV
        if HAS_PANDAS:
            df = pd.DataFrame([spss_data])
            df.to_csv(file_path, index=False, encoding="utf-8", na_rep=".")
        else:
            # Fallback without pandas
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                if spss_data:
                    writer = csv.DictWriter(csvfile, fieldnames=spss_data.keys())
                    writer.writeheader()
                    writer.writerow(spss_data)

    def _export_audit_trail(
        self, file_path: str, export_settings: Dict[str, Any]
    ) -> None:
        """Export audit trail log with manual editing information (FR-30)"""
        audit_data = {
            "export_info": {
                "timestamp": self.export_timestamp.isoformat(),
                "software_version": "HRV Analysis Software v2.1.1",
                "srs_version": "v2.1.1",
                "export_settings": export_settings,
            },
            "source_info": {
                "file_path": (
                    str(self.bundle.source.path) if self.bundle.source else "Unknown"
                ),
                "file_type": getattr(self.bundle.source, "filetype", "Unknown"),
                "device": getattr(self.bundle.source, "device", "Unknown"),
            },
            "analysis_parameters": self.analysis_parameters,
            "preprocessing_log": {
                "artifacts_detected": (
                    self.results.preprocessing_stats.get("artifacts_detected", 0)
                    if self.results.preprocessing_stats
                    else 0
                ),
                "artifacts_corrected": (
                    self.results.preprocessing_stats.get("artifacts_corrected", 0)
                    if self.results.preprocessing_stats
                    else 0
                ),
                "quality_assessment": (
                    self.results.quality_assessment.get("overall_quality")
                    if self.results.quality_assessment
                    else "unknown"
                ),
            },
            "manual_editing_log": self.analysis_parameters.get(
                "detailed_audit_trail", {}
            ),
            "analysis_results_summary": {
                "time_domain_computed": self.results.time_domain is not None,
                "frequency_domain_computed": self.results.frequency_domain is not None,
                "nonlinear_computed": self.results.nonlinear is not None,
                "total_warnings": (
                    len(self.results.warnings) if self.results.warnings else 0
                ),
            },
            "validation": {
                "srs_compliance": "FR-24 to FR-30 with manual editing (FR-10 to FR-14)",
                "export_completeness": "Complete with audit trail",
            },
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2, default=str)

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for metric"""
        metric_lower = metric_name.lower()
        if any(
            x in metric_lower for x in ["rr", "sdnn", "rmssd", "tinn", "sd1", "sd2"]
        ):
            return "ms"
        elif any(x in metric_lower for x in ["hr", "heart_rate"]):
            return "bpm"
        elif any(x in metric_lower for x in ["pnn", "percentage", "pct"]):
            return "%"
        elif "power" in metric_lower:
            return "ms²"
        elif any(x in metric_lower for x in ["freq", "frequency"]):
            return "Hz"
        return ""

    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display"""
        formatted = metric_name.replace("_", " ").title()

        replacements = {
            "Sdnn": "SDNN",
            "Rmssd": "RMSSD",
            "Pnn50": "pNN50",
            "Pnn20": "pNN20",
            "Cvnn": "CVNN",
            "Tinn": "TINN",
            "Hrv": "HRV",
            "Vlf": "VLF",
            "Lf": "LF",
            "Hf": "HF",
            "Apen": "ApEn",
            "Sampen": "SampEn",
        }

        for old, new in replacements.items():
            formatted = formatted.replace(old, new)

        return formatted
