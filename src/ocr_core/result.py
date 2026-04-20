"""
New modular GeneralOCRResult - loosely coupled, BBox format, no Paddle dependencies.

This implementation:
- Uses BBox format internally
- Separates data from export logic
- Maintains 100% backward compatibility with outputs
- No inheritance from Paddle classes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from src.ocr_core.coordinates import BBox, CoordinateSystem


@dataclass
class LayoutBlockData:
    """
    Single layout block with BBox format.

    Replaces the old LayoutBlock but uses BBox for coordinates.
    Generic structure that works with any OCR model.
    """
    bbox: BBox  # Explicit coordinate system tracking
    label: str
    content: str
    order_index: Optional[int] = None
    image: Optional[Dict[str, Any]] = None  # {"path": str, "img": PIL.Image}
    confidence: float = 1.0

    # Computed properties for generic use
    @property
    def width(self) -> float:
        """Block width."""
        return self.bbox.width

    @property
    def height(self) -> float:
        """Block height."""
        return self.bbox.height

    @property
    def area(self) -> float:
        """Block area."""
        return self.bbox.area

    @property
    def num_of_lines(self) -> int:
        """Estimated number of text lines in the block."""
        # Simple heuristic: count newlines in content
        return max(1, self.content.count('\n') + 1)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        return {
            "block_label": self.label,
            "block_content": self.content,
            "block_bbox": self.bbox.to_list(),  # Convert BBox to [x1,y1,x2,y2]
        }


@dataclass
class GeneralOCRResultV2:
    """
    Modular, loosely-coupled OCR result container.

    Design principles:
    - Simple dataclass, no inheritance
    - BBox format throughout
    - Export logic separated into external classes
    - Model-agnostic structure

    Maintains 100% backward compatibility with old outputs.
    """
    # Input metadata
    input_path: str
    page_index: int

    # Preprocessed image
    preprocessed_image: np.ndarray

    # Layout blocks (using BBox internally)
    layout_blocks: List[LayoutBlockData] = field(default_factory=list)

    # Detection results (BBox format)
    layout_det_bboxes: List[BBox] = field(default_factory=list)
    region_det_bboxes: List[BBox] = field(default_factory=list)

    # OCR results
    ocr_results: Dict[str, Any] = field(default_factory=dict)  # OCRResult data

    # Recognition results (kept as dicts for flexibility)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    seals: List[Dict[str, Any]] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)

    # Document preprocessing info
    doc_preprocessor_info: Dict[str, Any] = field(default_factory=dict)

    # Images in document (charts, figures, etc.)
    imgs_in_doc: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration
    model_settings: Dict[str, bool] = field(default_factory=dict)

    # Visualization data (extracted from Paddle model outputs via adapters)
    visualizations: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Structure: {
    #   "layout_det": {...},        # From PaddleDetResultAdapter
    #   "region_det": {...},        # From PaddleDetResultAdapter
    #   "ocr": {...},               # From PaddleOCRResultAdapter
    #   "doc_preprocessor": {...}   # From doc preprocessor pipeline
    # }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to simple dictionary for JSON serialization.

        This format is NEW and clean, not trying to match old format.
        Use exporters for backward-compatible outputs.
        """
        return {
            "input_path": self.input_path,
            "page_index": self.page_index,
            "layout_blocks": [block.to_legacy_dict() for block in self.layout_blocks],
            "layout_detection": [
                {"bbox": bbox.to_list(), "label": bbox.label, "confidence": bbox.confidence}
                for bbox in self.layout_det_bboxes
            ],
            "region_detection": [
                {"bbox": bbox.to_list(), "label": bbox.label, "confidence": bbox.confidence}
                for bbox in self.region_det_bboxes
            ],
            "tables": self.tables,
            "formulas": self.formulas,
            "seals": self.seals,
            "charts": self.charts,
            "model_settings": self.model_settings,
        }

    def save_to_img(self, output_dir: str | Path):
        """Export images using ImageExporter."""
        from src.ocr_core.exporters.image import ImageExporter
        exporter = ImageExporter()
        exporter.export(self, Path(output_dir))

    def save_to_json(self, output_dir: str | Path):
        """Export JSON using JSONExporter."""
        from src.ocr_core.exporters.json_exporter import JSONExporter
        exporter = JSONExporter()
        exporter.export(self, Path(output_dir))

    def save_to_markdown(self, output_dir: str | Path):
        """Export Markdown using MarkdownExporter."""
        from src.ocr_core.exporters.markdown import MarkdownExporter
        exporter = MarkdownExporter()
        exporter.export(self, Path(output_dir))

    def save_to_html(self, output_dir: str | Path):
        """Export HTML using HTMLExporter."""
        from src.ocr_core.exporters.html import HTMLExporter
        exporter = HTMLExporter()
        exporter.export(self, Path(output_dir))

    def save_to_xlsx(self, output_dir: str | Path):
        """Export XLSX using XLSXExporter."""
        from src.ocr_core.exporters.xlsx import XLSXExporter
        exporter = XLSXExporter()
        exporter.export(self, Path(output_dir))
