"""
PipelineContext - Shared state container for pipeline execution.

This dataclass holds all intermediate results as the pipeline progresses through stages.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.pp.models.object_detection.result import DetResult
from src.pp.pipelines.ocr.result import OCRResult
from src.pp.pipelines.layout_parsing.layout_objects import LayoutBlock


@dataclass
class PipelineContext:
    """
    Mutable state passed through pipeline stages.

    Each stage reads inputs from the context, performs its operation,
    and writes outputs back to the context for the next stage.

    Attributes:
        input_images: Raw input images (numpy arrays)
        config: Pipeline configuration dict

        # Stage outputs (populated as pipeline progresses)
        preprocessed_images: Images after doc preprocessing
        layout_det_results: Layout detection results (blocks)
        region_det_results: Region detection results (sub-blocks)
        formula_res_lists: Formula recognition results
        overall_ocr_results: OCR results (text detection + recognition)
        table_res_lists: Table recognition results
        seal_res_lists: Seal recognition results
        imgs_in_doc: Extracted images from document

        # Strategy-specific data
        block_to_ocr_maps: Direct mapping from blocks to OCR (for DirectBlockOCR strategy)

        # Final output
        layout_blocks: Parsed layout blocks
        final_results: List of LayoutParsingResultV2 objects

        # Metadata
        batch_input_paths: Input file paths
        batch_page_indexes: Page indexes for multi-page documents
    """

    # Required inputs
    input_images: List[np.ndarray]
    config: Dict[str, Any]

    # Stage outputs (optional, populated during execution)
    preprocessed_images: Optional[List[np.ndarray]] = None
    doc_preprocessor_results: Optional[List[Dict]] = None
    layout_det_results: Optional[List[DetResult]] = None
    region_det_results: Optional[List[DetResult]] = None
    formula_res_lists: Optional[List[List[dict]]] = None
    overall_ocr_results: Optional[List[OCRResult]] = None
    table_res_lists: Optional[List[List[dict]]] = None
    seal_res_lists: Optional[List[List[dict]]] = None
    imgs_in_doc: Optional[List[List[dict]]] = None

    # Strategy-specific
    block_to_ocr_maps: Optional[List[Optional[dict]]] = None

    # Final outputs
    layout_blocks: Optional[List[List[LayoutBlock]]] = None
    final_results: Optional[List[Any]] = None

    # Metadata
    batch_input_paths: Optional[List[str]] = None
    batch_page_indexes: Optional[List[int]] = None

    # Additional parameters (passed to various stages)
    params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """String representation showing which stages have been completed."""
        completed_stages = []
        if self.preprocessed_images is not None:
            completed_stages.append("preprocessing")
        if self.layout_det_results is not None:
            completed_stages.append("layout_detection")
        if self.region_det_results is not None:
            completed_stages.append("region_detection")
        if self.formula_res_lists is not None:
            completed_stages.append("formula_recognition")
        if self.overall_ocr_results is not None:
            completed_stages.append("ocr")
        if self.table_res_lists is not None:
            completed_stages.append("table_recognition")
        if self.seal_res_lists is not None:
            completed_stages.append("seal_recognition")
        if self.layout_blocks is not None:
            completed_stages.append("standardization")
        if self.final_results is not None:
            completed_stages.append("result_generation")

        return (
            f"PipelineContext("
            f"images={len(self.input_images)}, "
            f"completed={completed_stages})"
        )
