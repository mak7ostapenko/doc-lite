"""
Adapters for converting framework-specific outputs to OCR core structures.

These adapters bridge between specific OCR frameworks (PaddleOCR, EasyOCR, etc.)
and our generic OCR core primitives.
"""

from .paddle_adapter import PaddleDetResultAdapter, PaddleOCRResultAdapter

__all__ = [
    'PaddleDetResultAdapter',
    'PaddleOCRResultAdapter',
]
