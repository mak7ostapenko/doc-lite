"""
Core OCR data structures.

Generic primitives that work with any OCR model.
"""

from .text_primitives import (
    TextDetection,
    TextRecognition,
    TextBlock,
    Page
)

__all__ = [
    'TextDetection',
    'TextRecognition',
    'TextBlock',
    'Page',
]
