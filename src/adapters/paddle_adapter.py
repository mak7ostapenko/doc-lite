"""
Adapters for converting PaddlePaddle OCR outputs to core structures.

These adapters bridge between PaddleOCR/PaddleX outputs and our generic OCR core.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.ocr_core.coordinates import BBox, CoordinateSystem, ConversionUtils


class PaddleDetResultAdapter:
    """
    Adapter for PaddlePaddle detection results (DetResult).

    Converts DetResult dict format to List[BBox] and extracts visualizations.
    """

    @staticmethod
    def convert_to_bboxes(
        det_result: Dict[str, Any],
        coord_system: CoordinateSystem = CoordinateSystem.PAGE_ABSOLUTE
    ) -> List[BBox]:
        """
        Convert PaddlePaddle DetResult to List[BBox].

        Args:
            det_result: DetResult dict with 'boxes' field
            coord_system: Coordinate system for the bboxes

        Returns:
            List of BBox objects
        """
        return ConversionUtils.bboxes_from_det_result(det_result, coord_system)

    @staticmethod
    def extract_visualization(det_result: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract visualization images from DetResult.

        Args:
            det_result: DetResult dict (may have 'img' attribute if it's an object)

        Returns:
            Dictionary of visualization images, or None
        """
        # Check if det_result has img attribute (when it's a result object)
        if hasattr(det_result, 'img'):
            return det_result.img
        elif isinstance(det_result, dict) and 'img' in det_result:
            return det_result['img']
        return None


class PaddleOCRResultAdapter:
    """
    Adapter for PaddlePaddle OCR results (OCRResult).

    Converts OCRResult dict format to List[BBox] and extracts visualizations.
    """

    @staticmethod
    def convert_to_bboxes(
        ocr_result: Dict[str, Any],
        coord_system: CoordinateSystem = CoordinateSystem.PAGE_ABSOLUTE
    ) -> List[BBox]:
        """
        Convert PaddlePaddle OCRResult to List[BBox].

        Args:
            ocr_result: OCRResult dict with 'rec_boxes' field
            coord_system: Coordinate system for the bboxes

        Returns:
            List of BBox objects
        """
        return ConversionUtils.bboxes_from_ocr_result(ocr_result, coord_system)

    @staticmethod
    def extract_visualization(ocr_result: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract visualization images from OCRResult.

        Args:
            ocr_result: OCRResult dict (may have 'img' attribute if it's an object)

        Returns:
            Dictionary of visualization images, or None
        """
        # Check if ocr_result has img attribute (when it's a result object)
        if hasattr(ocr_result, 'img'):
            return ocr_result.img
        elif isinstance(ocr_result, dict) and 'img' in ocr_result:
            return ocr_result['img']
        return None
