"""
Unified HRV Analysis Pipeline
Integrates preprocessing → time_domain → freq_domain → nonlinear → respiratory
Follows the SRS requirements and maintains data consistency across modules
"""

import warnings
from typing import Dict, Optional, Tuple, Union, List
import numpy as np
from dataclasses import dataclass

# Import all integrated modules
from hrvlib.data_handler import DataBundle, TimeSeries, load_rr_file
from hrvlib.preprocessing import preprocess_rri, PreprocessingResult
from hrvlib.metrics.time_domain import (
    create_time_domain_analysis,
    HRVTimeDomainAnalysis,
)
from hrvlib.metrics.freq_domain import (
    create_freq_domain_analysis,
    HRVFreqDomainAnalysis,
)
from hrvlib.metrics.nonlinear import create_nonlinear_analysis, NonlinearHRVAnalysis

from hrvlib.metrics.respiratory import (
    analyze_respiratory_metrics,
    add_respiratory_metrics_to_bundle,
)


@dataclass
class HRVAnalysisResults:
    """
    Complete HRV analysis results from unified pipeline
    """

    # Core analysis results
    time_domain: Optional[Dict] = None
    frequency_domain: Optional[Dict] = None
    nonlinear: Optional[Dict] = None
    respiratory: Optional[Dict] = None

    # Data quality and preprocessing info
    preprocessing_stats: Optional[Dict] = None
    quality_assessment: Optional[Dict] = None

    # Analysis metadata
    analysis_info: Dict = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.analysis_info is None:
            self.analysis_info = {}


class UnifiedHRVPipeline:
    """
    Unified HRV analysis pipeline that orchestrates all analysis modules
    Ensures consistent preprocessing and data flow across all components
    Follows SRS requirements FR-16 through FR-30
    """

    def __init__(
        self,
        bundle: DataBundle,
        preprocessing_config: Optional[Dict] = None,
        analysis_config: Optional[Dict] = None,
    ):
        """
        Initialize unified HRV pipeline

        Args:
            bundle: DataBundle containing physiological data
            preprocessing_config: Configuration for preprocessing step
            analysis_config: Configuration for analysis modules
        """
        self.bundle = bundle
        self.preprocessing_config = preprocessing_config or {}
        self.analysis_config = analysis_config or self._get_default_analysis_config()

        #
