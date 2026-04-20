"""
Text primitive data structures for OCR.

Generic structures that work with any OCR model, regardless of recognition granularity.
Recognition can be at character, word, phrase, line, or any other level - we don't enforce structure.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.ocr_core.coordinates import BBox


@dataclass
class TextDetection:
    """
    Raw text detection result - just a bounding box, no recognized text yet.

    Output from detection-only models (e.g., CRAFT, DBNet, etc.)
    """
    bbox: BBox
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence
        }


@dataclass
class TextRecognition:
    """
    Generic text recognition result.

    The text can represent anything the model returns:
    - Single character (Tesseract char-level)
    - Single word (some word-level models)
    - Multiple words / phrase (EasyOCR)
    - Full line (PaddleOCR, most line-level models)
    - Paragraph (some document understanding models)

    Granularity is implicit - we don't force a specific structure.
    """
    bbox: BBox
    text: str
    confidence: float
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bbox": self.bbox.to_list(),
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language
        }


@dataclass
class TextBlock:
    """
    Logical block of text (e.g., paragraph, column, section).

    Contains multiple text recognition results that form a semantic unit.
    The internal structure (lines, words, etc.) is flexible.
    """
    bbox: BBox
    label: str  # e.g., "text", "title", "paragraph", "caption", etc.
    text: str   # Combined text from all recognitions
    confidence: float
    recognitions: List[TextRecognition] = field(default_factory=list)
    language: Optional[str] = None

    @property
    def recognition_count(self) -> int:
        """Number of individual text recognitions in this block."""
        return len(self.recognitions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bbox": self.bbox.to_list(),
            "label": self.label,
            "text": self.text,
            "confidence": self.confidence,
            "recognitions": [rec.to_dict() for rec in self.recognitions],
            "language": self.language
        }


@dataclass
class Page:
    """
    Complete page/document with all text blocks.

    Represents the final structured output from OCR processing.
    """
    page_number: int
    image_path: str
    image_size: tuple  # (width, height)
    blocks: List[TextBlock] = field(default_factory=list)
    detections: List[TextDetection] = field(default_factory=list)  # Raw detections before recognition
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def block_count(self) -> int:
        """Number of text blocks on this page."""
        return len(self.blocks)

    @property
    def total_text(self) -> str:
        """All text on the page concatenated."""
        return "\n\n".join(block.text for block in self.blocks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "image_size": self.image_size,
            "blocks": [block.to_dict() for block in self.blocks],
            "detections": [det.to_dict() for det in self.detections],
            "metadata": self.metadata
        }
