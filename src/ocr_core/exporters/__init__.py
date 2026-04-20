"""
Modular exporters for GeneralOCRResult.

Each exporter is independent and produces specific output format.
All exporters maintain 100% backward compatibility with original outputs.
"""

from .base import BaseExporter
from .image import ImageExporter
from .json_exporter import JSONExporter
from .markdown import MarkdownExporter
from .html import HTMLExporter
from .xlsx import XLSXExporter

__all__ = [
    'BaseExporter',
    'ImageExporter',
    'JSONExporter',
    'MarkdownExporter',
    'HTMLExporter',
    'XLSXExporter',
]
