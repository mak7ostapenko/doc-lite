"""
OCR Strategy implementations for the universal pipeline.

Strategies define how OCR is performed and how results are standardized.
"""

from .base import OCRStrategy
from .full_image_ocr import FullImageOCRStrategy
from .direct_block_ocr import DirectBlockOCRStrategy

__all__ = [
    "OCRStrategy",
    "FullImageOCRStrategy",
    "DirectBlockOCRStrategy",
]
