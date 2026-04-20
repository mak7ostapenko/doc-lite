"""
OCR Core - Generic OCR primitives and utilities.

This module provides model-agnostic data structures and utilities for OCR:
- Coordinate system tracking (BBox, BPolygon, CoordinateSystem)
- Text primitives (TextDetection, TextRecognition, TextBlock, Page)
- Result container (GeneralOCRResultV2, LayoutBlockData)
- Export formatters (Image, JSON, Markdown, HTML, XLSX)
- Visualization utilities
"""

# Coordinate primitives
from .coordinates import (
    BBox,
    BPolygon,
    CoordinateSystem,
    CoordinateTransform,
    SpatialOps,
    ConversionUtils
)

# Text primitives
from .structures import (
    TextDetection,
    TextRecognition,
    TextBlock,
    Page
)

# Result containers
from .result import (
    LayoutBlockData,
    GeneralOCRResultV2
)

# Exporters
from .exporters import (
    BaseExporter,
    ImageExporter,
    JSONExporter,
    MarkdownExporter,
    HTMLExporter,
    XLSXExporter
)

__all__ = [
    # Coordinates
    'BBox',
    'BPolygon',
    'CoordinateSystem',
    'CoordinateTransform',
    'SpatialOps',
    'ConversionUtils',
    # Text primitives
    'TextDetection',
    'TextRecognition',
    'TextBlock',
    'Page',
    # Results
    'LayoutBlockData',
    'GeneralOCRResultV2',
    # Exporters
    'BaseExporter',
    'ImageExporter',
    'JSONExporter',
    'MarkdownExporter',
    'HTMLExporter',
    'XLSXExporter',
]
