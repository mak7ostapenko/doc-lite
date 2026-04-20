"""
Pipeline stage implementations for the universal pipeline.

Stages are composable units that perform specific operations in the pipeline.
"""

from .base import PipelineStage
from .preprocessing import DocPreprocessingStage
from .detection import LayoutDetectionStage, RegionDetectionStage
from .recognition import (
    FormulaRecognitionStage,
    TableRecognitionStage,
    SealRecognitionStage,
    ChartRecognitionStage,
)

__all__ = [
    "PipelineStage",
    "DocPreprocessingStage",
    "LayoutDetectionStage",
    "RegionDetectionStage",
    "FormulaRecognitionStage",
    "TableRecognitionStage",
    "SealRecognitionStage",
    "ChartRecognitionStage",
]
