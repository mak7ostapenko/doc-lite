"""
Base class for OCR strategies.

OCR strategies define how text recognition is performed and how
results are matched to layout blocks.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.pp.pipelines.universal_pipeline.context import PipelineContext


class OCRStrategy(ABC):
    """
    Abstract base class for OCR strategies.

    Strategies encapsulate two key operations:
    1. OCR execution - how to perform text recognition
    2. Data standardization - how to match OCR results to layout blocks
    """

    def __init__(self, pipeline: Any):
        """
        Initialize strategy with reference to parent pipeline.

        Args:
            pipeline: The parent _LayoutParsingPipelineV2 instance.
                     Provides access to models and configuration.
        """
        self.pipeline = pipeline

    @abstractmethod
    def execute_ocr(self, context: PipelineContext) -> PipelineContext:
        """
        Execute OCR and populate context.overall_ocr_results.

        Args:
            context: Pipeline context with preprocessed images and layout results

        Returns:
            Updated context with overall_ocr_results populated
        """
        pass

    @abstractmethod
    def standardize_data(
        self,
        context: PipelineContext,
        image_index: int,
    ) -> tuple:
        """
        Standardize OCR data for a single image.

        This method takes OCR results and layout blocks, and produces
        a mapping between them. The approach depends on the strategy:
        - FullImageOCR: Complex spatial matching required
        - DirectBlockOCR: Direct 1:1 mapping already available

        Args:
            context: Pipeline context with all recognition results
            image_index: Index of the image to process

        Returns:
            Tuple of (region_block_ocr_idx_map, region_det_res, layout_det_res)
        """
        pass

    @property
    def name(self) -> str:
        """Get strategy name for logging and configuration."""
        return self.__class__.__name__
