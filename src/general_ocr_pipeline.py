import copy
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.pp.utils import logging
from src.pp.common.image_batch_sampler import ImageBatchSampler
from src.pp.common.image_reader import ReadImage
from src.pp.models.object_detection.result import DetResult
from src.pp.pipelines.base import BasePipeline
from src.pp.pipelines.ocr.result import OCRResult
from src.pp.pipelines.layout_parsing.layout_objects import LayoutBlock, LayoutBlock, LayoutRegion
from src.pp.pipelines.layout_parsing.setting import BLOCK_LABEL_MAP, BLOCK_SETTINGS, REGION_SETTINGS
from src.pp.pipelines.layout_parsing.utils import (
    convert_formula_res_to_ocr_format,
    gather_imgs,
    get_sub_regions_ocr_res,
    remove_overlap_blocks,
)
from src.pp.pipelines.layout_parsing.xycut_enhanced.xycuts import xycut_enhanced
from src.ocr_core.result import GeneralOCRResultV2, LayoutBlockData
from src.ocr_core.coordinates import (
    BBox,
    CoordinateSystem,
    SpatialOps,
    ConversionUtils
)
from src.adapters.paddle_adapter import PaddleDetResultAdapter, PaddleOCRResultAdapter


class _GeneralOCRPipeline(BasePipeline):
    """Layout Parsing Pipeline V2"""

    def __init__(
        self,
        config: dict,
        device: str = None,
    ) -> None:
        """Initializes the layout parsing pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
        """

        super().__init__(device=device)

        self.inintial_predictor(config)

        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))
        self.img_reader = ReadImage(format="BGR")

    def inintial_predictor(self, config: dict) -> None:
        """Initializes the predictor based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the configuration for the predictor.

        Returns:
            None
        """

        if (
            config.get("use_doc_preprocessor", True)
            or config.get("use_doc_orientation_classify", True)
            or config.get("use_doc_unwarping", True)
        ):
            self.use_doc_preprocessor = True
        else:
            self.use_doc_preprocessor = False
        self.use_table_recognition = config.get("use_table_recognition", True)
        self.use_seal_recognition = config.get("use_seal_recognition", True)
        self.use_region_detection = config.get(
            "use_region_detection",
            True,
        )
        self.use_formula_recognition = config.get(
            "use_formula_recognition",
            True,
        )
        self.use_chart_recognition = config.get(
            "use_chart_recognition",
            False,
        )

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config.get("SubPipelines", {}).get(
                "DocPreprocessor",
                {
                    "pipeline_config_error": "config error for doc_preprocessor_pipeline!",
                },
            )
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config,
            )
        if self.use_region_detection:
            region_detection_config = config.get("SubModules", {}).get(
                "RegionDetection",
                {
                    "model_config_error": "config error for block_region_detection_model!"
                },
            )
            self.region_detection_model = self.create_model(
                region_detection_config,
            )

        layout_det_config = config.get("SubModules", {}).get(
            "LayoutDetection",
            {"model_config_error": "config error for layout_det_model!"},
        )
        layout_kwargs = {}
        if (threshold := layout_det_config.get("threshold", None)) is not None:
            layout_kwargs["threshold"] = threshold
        if (layout_nms := layout_det_config.get("layout_nms", None)) is not None:
            layout_kwargs["layout_nms"] = layout_nms
        if (
            layout_unclip_ratio := layout_det_config.get("layout_unclip_ratio", None)
        ) is not None:
            layout_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
        if (
            layout_merge_bboxes_mode := layout_det_config.get(
                "layout_merge_bboxes_mode", None
            )
        ) is not None:
            layout_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
        self.layout_det_model = self.create_model(layout_det_config, **layout_kwargs)

        general_ocr_config = config.get("SubPipelines", {}).get(
            "GeneralOCR",
            {"pipeline_config_error": "config error for general_ocr_pipeline!"},
        )
        self.general_ocr_pipeline = self.create_pipeline(
            general_ocr_config,
        )

        if self.use_seal_recognition:
            seal_recognition_config = config.get("SubPipelines", {}).get(
                "SealRecognition",
                {
                    "pipeline_config_error": "config error for seal_recognition_pipeline!",
                },
            )
            self.seal_recognition_pipeline = self.create_pipeline(
                seal_recognition_config,
            )

        if self.use_table_recognition:
            table_recognition_config = config.get("SubPipelines", {}).get(
                "TableRecognition",
                {
                    "pipeline_config_error": "config error for table_recognition_pipeline!",
                },
            )
            self.table_recognition_pipeline = self.create_pipeline(
                table_recognition_config,
            )

        if self.use_formula_recognition:
            formula_recognition_config = config.get("SubPipelines", {}).get(
                "FormulaRecognition",
                {
                    "pipeline_config_error": "config error for formula_recognition_pipeline!",
                },
            )
            self.formula_recognition_pipeline = self.create_pipeline(
                formula_recognition_config,
            )

        # Chart recognition model - only init if enabled
        if self.use_chart_recognition:
            chart_recognition_config = config.get("SubModules", {}).get(
                "ChartRecognition",
                {"model_config_error": "config error for chart_recognition_model!"},
            )
            self.chart_recognition_model = self.create_model(
                chart_recognition_config,
            )
        else:
            self.chart_recognition_model = None

        return

    def get_text_paragraphs_ocr_res(
        self,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
    ) -> OCRResult:
        """
        Retrieves the OCR results for text paragraphs, excluding those of formulas, tables, and seals.

        Args:
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            layout_det_res (DetResult): The detection result containing the layout information of the document.

        Returns:
            OCRResult: The OCR result for text paragraphs after excluding formulas, tables, and seals.
        """
        object_boxes = []
        for box_info in layout_det_res["boxes"]:
            if box_info["label"].lower() in ["formula", "table", "seal"]:
                object_boxes.append(box_info["coordinate"])
        object_boxes = np.array(object_boxes)
        sub_regions_ocr_res = get_sub_regions_ocr_res(
            overall_ocr_res, object_boxes, flag_within=False
        )
        return sub_regions_ocr_res

    def check_model_settings_valid(self, input_params: dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if input_params["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized.",
            )
            return False

        if input_params["use_seal_recognition"] and not self.use_seal_recognition:
            logging.error(
                "Set use_seal_recognition, but the models for seal recognition are not initialized.",
            )
            return False

        if input_params["use_table_recognition"] and not self.use_table_recognition:
            logging.error(
                "Set use_table_recognition, but the models for table recognition are not initialized.",
            )
            return False

        return True

    # ========================================================================
    # COORDINATE CONVERSION HELPER METHODS
    # ========================================================================

    def _convert_det_results_to_bbox(
        self,
        det_results: List[dict],
        coord_system: CoordinateSystem = CoordinateSystem.PAGE_ABSOLUTE
    ) -> List[dict]:
        """
        Convert detection results from legacy dict format to BBox format.

        Adds 'bboxes' field with List[BBox] and preserves original 'boxes' as 'boxes_legacy'.

        Args:
            det_results: List of detection result dicts with 'boxes' field
            coord_system: Coordinate system for the bboxes

        Returns:
            Same list of dicts with added 'bboxes' and 'boxes_legacy' fields
        """
        for det_res in det_results:
            det_res["bboxes"] = ConversionUtils.bboxes_from_det_result(
                det_res, coord_system
            )
            det_res["boxes_legacy"] = det_res["boxes"]
        return det_results

    def _convert_ocr_results_to_bbox(
        self,
        ocr_results: List[dict],
        coord_system: CoordinateSystem = CoordinateSystem.PAGE_ABSOLUTE
    ) -> List[dict]:
        """
        Convert OCR results from legacy format to BBox format.

        Adds 'rec_bboxes' field with List[BBox] to OCR results.

        Args:
            ocr_results: List of OCR result dicts with 'rec_boxes' field
            coord_system: Coordinate system for the bboxes

        Returns:
            Same list of dicts with added 'rec_bboxes' field
        """
        for ocr_res in ocr_results:
            ocr_res["rec_bboxes"] = ConversionUtils.bboxes_from_ocr_result(
                ocr_res, coord_system
            )
        return ocr_results

    def _create_empty_region_results(
        self,
        count: int
    ) -> List[dict]:
        """
        Create empty region detection results for when region detection is disabled.

        Args:
            count: Number of empty results to create

        Returns:
            List of empty result dicts
        """
        return [
            {"boxes": [], "bboxes": [], "boxes_legacy": []}
            for _ in range(count)
        ]

    # ========================================================================
    # HELPER METHODS FOR STANDARDIZED_DATA (Refactored Coordinate Handling)
    # ========================================================================

    def _correct_layout_labels(
        self,
        layout_bboxes: List[BBox]
    ) -> Tuple[List[BBox], float]:
        """
        Correct layout labels based on spatial heuristics.

        1. Footnotes above text → relabel as "text"
        2. Single paragraph_title with no doc_title → relabel as "doc_title" (if large enough)

        Returns:
            (corrected_layout_bboxes, max_block_area)
        """
        footnote_indices = []
        paragraph_title_indices = []
        bottom_text_y_max = 0
        max_block_area = 0.0
        doc_title_count = 0

        # First pass: collect info
        for idx, bbox in enumerate(layout_bboxes):
            max_block_area = max(max_block_area, bbox.area)

            if bbox.label == "footnote":
                footnote_indices.append(idx)
            elif bbox.label == "paragraph_title":
                paragraph_title_indices.append(idx)
            elif bbox.label == "text":
                bottom_text_y_max = max(bottom_text_y_max, bbox.y_max)
            elif bbox.label == "doc_title":
                doc_title_count += 1

        # Fix footnotes above text
        corrected_bboxes = list(layout_bboxes)
        for idx in footnote_indices:
            if layout_bboxes[idx].y_max < bottom_text_y_max:
                # Footnote is above text blocks → relabel as text
                corrected_bboxes[idx] = BBox(
                    layout_bboxes[idx].x_min, layout_bboxes[idx].y_min,
                    layout_bboxes[idx].x_max, layout_bboxes[idx].y_max,
                    layout_bboxes[idx].coord_system,
                    label="text",  # Changed from "footnote"
                    confidence=layout_bboxes[idx].confidence
                )

        # Fix single paragraph_title without doc_title
        if len(paragraph_title_indices) == 1 and doc_title_count == 0:
            title_idx = paragraph_title_indices[0]
            title_bbox = corrected_bboxes[title_idx]
            title_area_threshold = BLOCK_SETTINGS.get("title_conversion_area_ratio_threshold", 0.3)

            if title_bbox.area > max_block_area * title_area_threshold:
                corrected_bboxes[title_idx] = BBox(
                    title_bbox.x_min, title_bbox.y_min,
                    title_bbox.x_max, title_bbox.y_max,
                    title_bbox.coord_system,
                    label="doc_title",  # Changed from "paragraph_title"
                    confidence=title_bbox.confidence
                )

        return corrected_bboxes, max_block_area

    def _match_ocr_to_blocks(
        self,
        layout_bboxes: List[BBox],
        ocr_bboxes: List[BBox]
    ) -> Tuple[dict, dict]:
        """
        Match OCR boxes to layout blocks using spatial overlap.

        Returns:
            (block_to_ocr_map, ocr_to_blocks_map)
            - block_to_ocr_map: {block_idx: [ocr_idx1, ocr_idx2, ...]}
            - ocr_to_blocks_map: {ocr_idx: [block_idx1, block_idx2, ...]}
        """
        block_to_ocr_map = {}
        ocr_to_blocks_map = {}

        for block_idx, layout_bbox in enumerate(layout_bboxes):
            # Skip non-text blocks (formulas, tables, seals handled separately)
            if layout_bbox.label in ["formula", "table", "seal"]:
                continue

            # Find overlapping OCR boxes
            matched_ocr_indices = SpatialOps.filter_by_overlap(
                ocr_bboxes,
                layout_bbox,
                threshold=0.1,  # Keep threshold low to catch all candidates
                mode="small"
            )

            block_to_ocr_map[block_idx] = matched_ocr_indices

            # Build reverse mapping
            for ocr_idx in matched_ocr_indices:
                if ocr_idx not in ocr_to_blocks_map:
                    ocr_to_blocks_map[ocr_idx] = [block_idx]
                else:
                    ocr_to_blocks_map[ocr_idx].append(block_idx)

        return block_to_ocr_map, ocr_to_blocks_map

    def _split_multiblock_text(
        self,
        layout_bboxes: List[BBox],
        ocr_bboxes: List[BBox],
        overall_ocr_res: dict,
        block_to_ocr_map: dict,
        ocr_to_blocks_map: dict,
        image: np.ndarray,
        text_rec_model: Any,
        text_rec_score_thresh: float
    ) -> Tuple[dict, dict]:
        """
        Handle OCR text spanning multiple layout blocks.

        For each OCR box matched to 2+ blocks:
        1. Crop text to each block's intersection
        2. Re-run text recognition on each crop
        3. Update OCR results and mappings

        Returns:
            (updated_block_to_ocr_map, updated_overall_ocr_res)
        """
        # Find OCR boxes spanning multiple blocks
        multiblock_ocr_indices = [
            ocr_idx for ocr_idx, block_list in ocr_to_blocks_map.items()
            if len(block_list) > 1
        ]

        for ocr_idx in multiblock_ocr_indices:
            matched_blocks = ocr_to_blocks_map[ocr_idx]
            ocr_bbox = ocr_bboxes[ocr_idx]

            for block_idx in matched_blocks:
                layout_bbox = layout_bboxes[block_idx]

                # Calculate intersection
                inter_bbox = SpatialOps.intersection(ocr_bbox, layout_bbox)
                if inter_bbox is None:
                    continue  # No overlap (shouldn't happen, but be safe)

                # Clear existing low-quality OCR results in this intersection
                for existing_ocr_idx in block_to_ocr_map.get(block_idx, []):
                    if existing_ocr_idx >= len(ocr_bboxes):
                        continue  # skip newly appended entries not in original ocr_bboxes
                    existing_bbox = ocr_bboxes[existing_ocr_idx]
                    if SpatialOps.iou(existing_bbox, inter_bbox, "small") > 0.8:
                        overall_ocr_res["rec_texts"][existing_ocr_idx] = ""

                # Crop image and re-recognize
                x1, y1, x2, y2 = map(int, inter_bbox.to_list())
                crop_img = np.array(image)[y1:y2, x1:x2]
                crop_rec_res = list(text_rec_model([crop_img]))[0]

                if crop_rec_res["rec_score"] >= text_rec_score_thresh:
                    # Add new OCR result
                    crop_poly = inter_bbox.to_list()  # Use intersection bbox as polygon
                    crop_poly_points = [
                        [crop_poly[0], crop_poly[1]],
                        [crop_poly[2], crop_poly[1]],
                        [crop_poly[2], crop_poly[3]],
                        [crop_poly[0], crop_poly[3]]
                    ]

                    if matched_blocks.index(block_idx) == 0:
                        # Replace first OCR result
                        overall_ocr_res["dt_polys"][ocr_idx] = crop_poly_points
                        overall_ocr_res["rec_boxes"][ocr_idx] = np.array(crop_poly)
                        overall_ocr_res["rec_polys"][ocr_idx] = crop_poly_points
                        overall_ocr_res["rec_scores"][ocr_idx] = crop_rec_res["rec_score"]
                        overall_ocr_res["rec_texts"][ocr_idx] = crop_rec_res["rec_text"]
                    else:
                        # Append new OCR result
                        overall_ocr_res["dt_polys"].append(crop_poly_points)
                        if len(overall_ocr_res["rec_boxes"]) == 0:
                            overall_ocr_res["rec_boxes"] = np.array([crop_poly])
                        else:
                            overall_ocr_res["rec_boxes"] = np.vstack(
                                (overall_ocr_res["rec_boxes"], crop_poly)
                            )
                        overall_ocr_res["rec_polys"].append(crop_poly_points)
                        overall_ocr_res["rec_scores"].append(crop_rec_res["rec_score"])
                        overall_ocr_res["rec_texts"].append(crop_rec_res["rec_text"])
                        overall_ocr_res["rec_labels"].append("text")

                        # Update mapping
                        new_ocr_idx = len(overall_ocr_res["rec_texts"]) - 1
                        if ocr_idx in block_to_ocr_map[block_idx]:
                            block_to_ocr_map[block_idx].remove(ocr_idx)
                        block_to_ocr_map[block_idx].append(new_ocr_idx)

        return block_to_ocr_map, overall_ocr_res

    def _fallback_ocr_for_blocks(
        self,
        layout_bboxes: List[BBox],
        block_to_ocr_map: dict,
        overall_ocr_res: dict,
        image: np.ndarray,
        text_rec_model: Any,
        text_rec_score_thresh: float
    ) -> Tuple[dict, dict]:
        """
        Run OCR on blocks with no matched text (critical for image-heavy pages).

        For each block without text:
        1. Crop block from image
        2. Run text recognition
        3. Add result to overall_ocr_res

        Returns:
            (updated_block_to_ocr_map, updated_overall_ocr_res)
        """
        vision_labels = BLOCK_LABEL_MAP.get("vision_labels", [])

        for block_idx, ocr_indices in block_to_ocr_map.items():
            # Check if block has any non-empty text
            has_text = any(
                overall_ocr_res["rec_texts"][idx] != ""
                for idx in ocr_indices
            )

            if not has_text:
                layout_bbox = layout_bboxes[block_idx]

                # Skip vision blocks (images, charts, etc.)
                if layout_bbox.label in vision_labels:
                    continue

                # Crop and recognize
                x1, y1, x2, y2 = map(int, layout_bbox.to_list())
                crop_img = np.array(image)[y1:y2, x1:x2]
                crop_rec_res = list(text_rec_model([crop_img]))[0]

                if crop_rec_res["rec_score"] >= text_rec_score_thresh:
                    # Add new OCR result
                    crop_poly = layout_bbox.to_list()
                    crop_poly_points = [
                        [crop_poly[0], crop_poly[1]],
                        [crop_poly[2], crop_poly[1]],
                        [crop_poly[2], crop_poly[3]],
                        [crop_poly[0], crop_poly[3]]
                    ]

                    if len(overall_ocr_res["rec_boxes"]) == 0:
                        overall_ocr_res["rec_boxes"] = np.array([crop_poly])
                    else:
                        overall_ocr_res["rec_boxes"] = np.vstack(
                            (overall_ocr_res["rec_boxes"], crop_poly)
                        )

                    overall_ocr_res["rec_polys"].append(crop_poly_points)
                    overall_ocr_res["rec_scores"].append(crop_rec_res["rec_score"])
                    overall_ocr_res["rec_texts"].append(crop_rec_res["rec_text"])
                    overall_ocr_res["rec_labels"].append("text")

                    # Update mapping
                    new_ocr_idx = len(overall_ocr_res["rec_texts"]) - 1
                    block_to_ocr_map[block_idx].append(new_ocr_idx)

        return block_to_ocr_map, overall_ocr_res

    def standardized_data(
        self,
        image: list,
        region_det_res: DetResult,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        formula_res_list: list,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> list:
        """
        REFACTORED: Retrieves layout parsing result with explicit coordinate tracking.

        Uses new BBox/BPolygon classes for coordinate transformations.

        Args:
            image: Input image array
            region_det_res: Region detection results (DetResult format)
            layout_det_res: Layout detection results (DetResult format)
            overall_ocr_res: OCR results (OCRResult format)
            formula_res_list: Formula recognition results
            text_rec_model: Text recognition model for fallback OCR
            text_rec_score_thresh: Threshold for text recognition confidence

        Returns:
            (region_block_ocr_idx_map, region_det_res, layout_det_res)
        """
        # ====================================================================
        # STEP 0: Preprocessing
        # ====================================================================
        # Remove overlapping layout blocks
        layout_det_res = remove_overlap_blocks(
            layout_det_res,
            threshold=0.5,
            smaller=True,
        )

        # Convert formula results to OCR format (modifies overall_ocr_res in-place)
        convert_formula_res_to_ocr_format(formula_res_list, overall_ocr_res)

        # Set default threshold
        if text_rec_score_thresh is None:
            text_rec_score_thresh = self.general_ocr_pipeline.text_rec_score_thresh

        # ====================================================================
        # STEP 1: Use BBox format (already converted in predict method)
        # ====================================================================
        # Always re-derive from "boxes" after remove_overlap_blocks, since the
        # "bboxes" field may be stale (it is not updated by remove_overlap_blocks).
        layout_bboxes = ConversionUtils.bboxes_from_det_result(
            layout_det_res, CoordinateSystem.PAGE_ABSOLUTE
        )
        ocr_bboxes = overall_ocr_res.get("rec_bboxes")

        if ocr_bboxes is None:
            ocr_bboxes = ConversionUtils.bboxes_from_ocr_result(
                overall_ocr_res, CoordinateSystem.PAGE_ABSOLUTE
            )

        # ====================================================================
        # STEP 2: Correct layout labels (footnotes, titles)
        # ====================================================================
        layout_bboxes, max_block_area = self._correct_layout_labels(layout_bboxes)

        # Update layout_det_res with corrected labels
        for idx, bbox in enumerate(layout_bboxes):
            layout_det_res["boxes"][idx]["label"] = bbox.label

        # ====================================================================
        # STEP 3: Match OCR to layout blocks
        # ====================================================================
        block_to_ocr_map, ocr_to_blocks_map = self._match_ocr_to_blocks(
            layout_bboxes, ocr_bboxes
        )

        # ====================================================================
        # STEP 4: Handle multi-block text spans
        # ====================================================================
        block_to_ocr_map, overall_ocr_res = self._split_multiblock_text(
            layout_bboxes, ocr_bboxes, overall_ocr_res,
            block_to_ocr_map, ocr_to_blocks_map,
            image, text_rec_model, text_rec_score_thresh
        )

        # ====================================================================
        # STEP 5: Fallback OCR for blocks without text
        # ====================================================================
        block_to_ocr_map, overall_ocr_res = self._fallback_ocr_for_blocks(
            layout_bboxes, block_to_ocr_map, overall_ocr_res,
            image, text_rec_model, text_rec_score_thresh
        )

        # ====================================================================
        # STEP 6: Handle edge case - no layout but OCR exists
        # ====================================================================
        if len(layout_bboxes) == 0 and len(ocr_bboxes) > 0:
            # Convert OCR boxes to layout boxes
            for idx, ocr_bbox in enumerate(ocr_bboxes):
                layout_bboxes.append(
                    BBox(
                        ocr_bbox.x_min, ocr_bbox.y_min,
                        ocr_bbox.x_max, ocr_bbox.y_max,
                        CoordinateSystem.PAGE_ABSOLUTE,
                        label="text",
                        confidence=overall_ocr_res["rec_scores"][idx]
                    )
                )
                block_to_ocr_map[idx] = [idx]

        # ====================================================================
        # STEP 7: Region-to-block mapping (refactored with BBox)
        # ====================================================================
        mask_labels = (
            BLOCK_LABEL_MAP.get("unordered_labels", [])
            + BLOCK_LABEL_MAP.get("header_labels", [])
            + BLOCK_LABEL_MAP.get("footer_labels", [])
        )

        # Get region bboxes (already converted to BBox in predict)
        region_bboxes = region_det_res.get("bboxes")
        if region_bboxes is None:
            region_bboxes = ConversionUtils.bboxes_from_det_result(
                region_det_res, CoordinateSystem.PAGE_ABSOLUTE
            )

        # Sort regions by area
        region_bboxes = sorted(region_bboxes, key=lambda bbox: bbox.area)

        region_to_block_map = {}

        if len(region_bboxes) == 0:
            # No regions detected - create single supplementary region
            if len(layout_bboxes) > 0:
                # Create enclosing region for all layout blocks
                enclosing_region = SpatialOps.union_bbox(layout_bboxes)
                enclosing_region = BBox(
                    enclosing_region.x_min, enclosing_region.y_min,
                    enclosing_region.x_max, enclosing_region.y_max,
                    CoordinateSystem.PAGE_ABSOLUTE,
                    label="SupplementaryRegion",
                    confidence=1.0
                )
                region_bboxes = [enclosing_region]
                region_to_block_map[0] = list(range(len(layout_bboxes)))
        else:
            # Match blocks to detected regions (using BBox)
            block_idxes_set = set(range(len(layout_bboxes)))

            for region_idx, region_bbox in enumerate(region_bboxes):
                matched_idxes = []
                region_to_block_map[region_idx] = []

                # Find blocks overlapping with region
                for block_idx in block_idxes_set:
                    if layout_bboxes[block_idx].label in mask_labels:
                        continue
                    overlap_ratio = SpatialOps.iou(
                        region_bbox, layout_bboxes[block_idx], mode="small"
                    )
                    if overlap_ratio > REGION_SETTINGS.get(
                        "match_block_overlap_ratio_threshold", 0.8
                    ):
                        matched_idxes.append(block_idx)

                # Iteratively expand region to include adjacent blocks
                if len(matched_idxes) > 0:
                    old_matched_idxes = []
                    while len(old_matched_idxes) != len(matched_idxes):
                        old_matched_idxes = copy.deepcopy(matched_idxes)
                        matched_idxes = []
                        matched_bboxes = [
                            layout_bboxes[idx] for idx in old_matched_idxes
                        ]
                        expanded_bbox = SpatialOps.union_bbox(matched_bboxes)

                        # Preserve original region's label and confidence
                        new_region_bbox = BBox(
                            expanded_bbox.x_min, expanded_bbox.y_min,
                            expanded_bbox.x_max, expanded_bbox.y_max,
                            CoordinateSystem.PAGE_ABSOLUTE,
                            label=region_bbox.label,  # Keep original label
                            confidence=region_bbox.confidence  # Keep original confidence
                        )

                        for block_idx in block_idxes_set:
                            if layout_bboxes[block_idx].label in mask_labels:
                                continue
                            overlap_ratio = SpatialOps.iou(
                                new_region_bbox, layout_bboxes[block_idx], mode="small"
                            )
                            if overlap_ratio > REGION_SETTINGS.get(
                                "match_block_overlap_ratio_threshold", 0.8
                            ):
                                matched_idxes.append(block_idx)

                    for block_idx in matched_idxes:
                        block_idxes_set.remove(block_idx)
                    region_to_block_map[region_idx] = matched_idxes
                    # Update region bbox (preserving label and confidence)
                    region_bboxes[region_idx] = new_region_bbox

            # Create supplementary regions for unmatched blocks
            while len(block_idxes_set) > 0:
                unmatched_bboxes = [layout_bboxes[idx] for idx in block_idxes_set]
                if len(unmatched_bboxes) == 0:
                    break

                supplement_region_bbox = SpatialOps.union_bbox(unmatched_bboxes)
                matched_idxes = []

                # Shrink supplementary region if it overlaps with existing regions
                # Note: Keeping shrink logic simple for BBox - complex shrinking deferred
                for region_idx, existing_region_bbox in enumerate(region_bboxes[:len(region_to_block_map)]):
                    if len(region_to_block_map.get(region_idx, [])) == 0:
                        continue
                    overlap_ratio = SpatialOps.iou(
                        supplement_region_bbox, existing_region_bbox, mode="union"
                    )
                    if overlap_ratio > 0:
                        # Simple approach: find blocks that don't overlap with existing region
                        for block_idx in block_idxes_set:
                            block_bbox = layout_bboxes[block_idx]
                            if SpatialOps.iou(block_bbox, existing_region_bbox, mode="small") < 0.5:
                                matched_idxes.append(block_idx)
                        if matched_idxes:
                            break

                matched_idxes = [
                    idx for idx in matched_idxes
                    if layout_bboxes[idx].label not in mask_labels
                ]
                if len(matched_idxes) == 0:
                    matched_idxes = [
                        idx for idx in block_idxes_set
                        if layout_bboxes[idx].label not in mask_labels
                    ]
                    if len(matched_idxes) == 0:
                        break

                matched_bboxes = [layout_bboxes[idx] for idx in matched_idxes]
                supplement_region_bbox = SpatialOps.union_bbox(matched_bboxes)

                region_idx = len(region_bboxes)
                region_to_block_map[region_idx] = list(matched_idxes)
                for block_idx in matched_idxes:
                    block_idxes_set.remove(block_idx)

                # Add supplementary region to list
                supplement_bbox = BBox(
                    supplement_region_bbox.x_min, supplement_region_bbox.y_min,
                    supplement_region_bbox.x_max, supplement_region_bbox.y_max,
                    CoordinateSystem.PAGE_ABSOLUTE,
                    label="SupplementaryRegion",
                    confidence=1.0
                )
                region_bboxes.append(supplement_bbox)

            # Add supplementary regions for mask labels (headers/footers)
            mask_idxes = [
                idx for idx in range(len(layout_bboxes))
                if layout_bboxes[idx].label in mask_labels
            ]
            for idx in mask_idxes:
                bbox = layout_bboxes[idx]
                region_idx = len(region_bboxes)
                region_to_block_map[region_idx] = [idx]

                supplement_bbox = BBox(
                    bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max,
                    CoordinateSystem.PAGE_ABSOLUTE,
                    label="SupplementaryRegion",
                    confidence=1.0
                )
                region_bboxes.append(supplement_bbox)

        region_block_ocr_idx_map = dict(
            region_to_block_map=region_to_block_map,
            block_to_ocr_map=block_to_ocr_map,
        )

        # Store BBox lists back into result dicts for passing through
        layout_det_res["bboxes"] = layout_bboxes
        region_det_res["bboxes"] = region_bboxes

        return region_block_ocr_idx_map, region_det_res, layout_det_res

    def get_layout_parsing_objects(
        self,
        image: list,
        region_block_ocr_idx_map: dict,
        region_det_res: DetResult,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
        table_res_list: list,
        seal_res_list: list,
        chart_res_list: list,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> list:
        """
        Extract structured information from OCR and layout detection results.

        Args:
            image (list): The input image.
            overall_ocr_res (OCRResult): An object containing the overall OCR results, including detected text boxes and recognized text. The structure is expected to have:
                - "input_img": The image on which OCR was performed.
                - "dt_boxes": A list of detected text box coordinates.
                - "rec_texts": A list of recognized text corresponding to the detected boxes.

            layout_det_res (DetResult): An object containing the layout detection results, including detected layout boxes and their labels. The structure is expected to have:
                - "boxes": A list of dictionaries with keys "coordinate" for box coordinates and "block_label" for the type of content.

            table_res_list (list): A list of table detection results, where each item is a dictionary containing:
                - "block_bbox": The bounding box of the table layout.
                - "pred_html": The predicted HTML representation of the table.

            seal_res_list (List): A list of seal detection results. The details of each item depend on the specific application context.
            text_rec_model (Any): A model for text recognition.
            text_rec_score_thresh (Union[float, None]): The minimum score required for a recognized character to be considered valid. If None, use the default value specified during initialization. Default is None.

        Returns:
            list: A list of structured boxes where each item is a dictionary containing:
                - "block_label": The label of the content (e.g., 'table', 'chart', 'image').
                - The label as a key with either table HTML or image data and text.
                - "block_bbox": The coordinates of the layout box.
        """

        table_index = 0
        seal_index = 0
        chart_index = 0
        layout_parsing_blocks: List[LayoutBlock] = []

        # Use BBox format (converted in standardized_data)
        layout_bboxes = layout_det_res.get("bboxes", [])

        for box_idx, bbox in enumerate(layout_bboxes):

            label = bbox.label
            block_bbox_legacy = bbox.to_list()  # Convert to list for LayoutBlock
            rec_res = {"boxes": [], "rec_texts": [], "rec_labels": []}

            block = LayoutBlock(label=label, bbox=block_bbox_legacy)

            if label == "table" and len(table_res_list) > 0:
                block.content = table_res_list[table_index]["pred_html"]
                table_index += 1
            elif label == "seal" and len(seal_res_list) > 0:
                block.content = "\n".join(seal_res_list[seal_index]["rec_texts"])
                seal_index += 1
            elif label == "chart" and len(chart_res_list) > 0:
                block.content = chart_res_list[chart_index]
                chart_index += 1
            else:
                if label == "formula":
                    _, ocr_idx_list = get_sub_regions_ocr_res(
                        overall_ocr_res, [block_bbox_legacy], return_match_idx=True
                    )
                    region_block_ocr_idx_map["block_to_ocr_map"][box_idx] = ocr_idx_list
                else:
                    ocr_idx_list = region_block_ocr_idx_map["block_to_ocr_map"].get(
                        box_idx, []
                    )
                for box_no in ocr_idx_list:
                    rec_res["boxes"].append(overall_ocr_res["rec_boxes"][box_no])
                    rec_res["rec_texts"].append(
                        overall_ocr_res["rec_texts"][box_no],
                    )
                    rec_res["rec_labels"].append(
                        overall_ocr_res["rec_labels"][box_no],
                    )
                block.update_text_content(
                    image=image,
                    ocr_rec_res=rec_res,
                    text_rec_model=text_rec_model,
                    text_rec_score_thresh=text_rec_score_thresh,
                )

            if (
                label
                in ["seal", "table", "formula", "chart"]
                + BLOCK_LABEL_MAP["image_labels"]
            ):
                x_min, y_min, x_max, y_max = list(map(int, block_bbox_legacy))
                img_path = (
                    f"imgs/img_in_{block.label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                )
                img = Image.fromarray(image[y_min:y_max, x_min:x_max, ::-1])
                block.image = {"path": img_path, "img": img}

            layout_parsing_blocks.append(block)

        # Use BBox format for regions
        region_bboxes = region_det_res.get("bboxes", [])

        page_region_bboxes = []
        layout_parsing_regions: List[LayoutRegion] = []
        for region_idx, region_bbox in enumerate(region_bboxes):
            region_bbox_legacy = np.array(region_bbox.to_list()).astype("int")
            region_blocks = [
                layout_parsing_blocks[idx]
                for idx in region_block_ocr_idx_map["region_to_block_map"][region_idx]
            ]
            if region_blocks:
                page_region_bboxes.append(region_bbox)
                region = LayoutRegion(bbox=region_bbox_legacy, blocks=region_blocks)
                layout_parsing_regions.append(region)

        # Calculate enclosing bbox for page
        if page_region_bboxes:
            page_bbox = SpatialOps.union_bbox(page_region_bboxes)
            page_bbox_legacy = np.array(page_bbox.to_list()).astype("int")
        else:
            page_bbox_legacy = np.array([0, 0, 0, 0]).astype("int")

        layout_parsing_page = LayoutRegion(
            bbox=page_bbox_legacy, blocks=layout_parsing_regions
        )

        return layout_parsing_page

    def sort_layout_parsing_blocks(
        self, layout_parsing_page: LayoutRegion
    ) -> List[LayoutBlock]:
        layout_parsing_regions = xycut_enhanced(layout_parsing_page)
        parsing_res_list = []
        for region in layout_parsing_regions:
            layout_parsing_blocks = xycut_enhanced(region)
            parsing_res_list.extend(layout_parsing_blocks)

        return parsing_res_list

    def get_layout_parsing_res(
        self,
        image: list,
        region_det_res: DetResult,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        table_res_list: list,
        seal_res_list: list,
        chart_res_list: list,
        formula_res_list: list,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> list:
        """
        Retrieves the layout parsing result based on the layout detection result, OCR result, and other recognition results.
        Args:
            image (list): The input image.
            layout_det_res (DetResult): The detection result containing the layout information of the document.
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            table_res_list (list): A list of table recognition results.
            seal_res_list (list): A list of seal recognition results.
            formula_res_list (list): A list of formula recognition results.
            text_rec_score_thresh (Optional[float], optional): The score threshold for text recognition. Defaults to None.
        Returns:
            list: A list of dictionaries representing the layout parsing result.
        """

        # Standardize data
        region_block_ocr_idx_map, region_det_res, layout_det_res = (
            self.standardized_data(
                image=image,
                region_det_res=region_det_res,
                layout_det_res=layout_det_res,
                overall_ocr_res=overall_ocr_res,
                formula_res_list=formula_res_list,
                text_rec_model=self.general_ocr_pipeline.text_rec_model,
                text_rec_score_thresh=text_rec_score_thresh,
            )
        )

        # Format layout parsing block
        layout_parsing_page = self.get_layout_parsing_objects(
            image=image,
            region_block_ocr_idx_map=region_block_ocr_idx_map,
            region_det_res=region_det_res,
            overall_ocr_res=overall_ocr_res,
            layout_det_res=layout_det_res,
            table_res_list=table_res_list,
            seal_res_list=seal_res_list,
            chart_res_list=chart_res_list,
            text_rec_model=self.general_ocr_pipeline.text_rec_model,
            text_rec_score_thresh=self.general_ocr_pipeline.text_rec_score_thresh,
        )

        parsing_res_list = self.sort_layout_parsing_blocks(layout_parsing_page)

        index = 1
        for block in parsing_res_list:
            if block.label in BLOCK_LABEL_MAP["visualize_index_labels"]:
                block.order_index = index
                index += 1

        return parsing_res_list

    def get_model_settings(
        self,
        use_doc_orientation_classify: Union[bool, None],
        use_doc_unwarping: Union[bool, None],
        use_seal_recognition: Union[bool, None],
        use_table_recognition: Union[bool, None],
        use_formula_recognition: Union[bool, None],
        use_chart_recognition: Union[bool, None],
        use_region_detection: Union[bool, None],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Union[bool, None]): Enables document orientation classification if True. Defaults to system setting if None.
            use_doc_unwarping (Union[bool, None]): Enables document unwarping if True. Defaults to system setting if None.
            use_seal_recognition (Union[bool, None]): Enables seal recognition if True. Defaults to system setting if None.
            use_table_recognition (Union[bool, None]): Enables table recognition if True. Defaults to system setting if None.
            use_formula_recognition (Union[bool, None]): Enables formula recognition if True. Defaults to system setting if None.

        Returns:
            dict: A dictionary containing the model settings.

        """
        if use_doc_orientation_classify is None and use_doc_unwarping is None:
            use_doc_preprocessor = self.use_doc_preprocessor
        else:
            if use_doc_orientation_classify is True or use_doc_unwarping is True:
                use_doc_preprocessor = True
            else:
                use_doc_preprocessor = False

        if use_seal_recognition is None:
            use_seal_recognition = self.use_seal_recognition

        if use_table_recognition is None:
            use_table_recognition = self.use_table_recognition

        if use_formula_recognition is None:
            use_formula_recognition = self.use_formula_recognition

        if use_region_detection is None:
            use_region_detection = self.use_region_detection

        if use_chart_recognition is None:
            use_chart_recognition = self.use_chart_recognition

        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_region_detection=use_region_detection,
        )

    def predict(
        self,
        input: Union[str, list[str], np.ndarray, list[np.ndarray]],
        use_doc_orientation_classify: Union[bool, None] = None,
        use_doc_unwarping: Union[bool, None] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Union[bool, None] = None,
        use_table_recognition: Union[bool, None] = None,
        use_formula_recognition: Union[bool, None] = None,
        use_chart_recognition: Union[bool, None] = None,
        use_region_detection: Union[bool, None] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        text_det_limit_side_len: Union[int, None] = None,
        text_det_limit_type: Union[str, None] = None,
        text_det_thresh: Union[float, None] = None,
        text_det_box_thresh: Union[float, None] = None,
        text_det_unclip_ratio: Union[float, None] = None,
        text_rec_score_thresh: Union[float, None] = None,
        seal_det_limit_side_len: Union[int, None] = None,
        seal_det_limit_type: Union[str, None] = None,
        seal_det_thresh: Union[float, None] = None,
        seal_det_box_thresh: Union[float, None] = None,
        seal_det_unclip_ratio: Union[float, None] = None,
        seal_rec_score_thresh: Union[float, None] = None,
        use_wired_table_cells_trans_to_html: bool = False,
        use_wireless_table_cells_trans_to_html: bool = False,
        use_table_orientation_classify: bool = True,
        use_ocr_results_with_table_cells: bool = True,
        use_e2e_wired_table_rec_model: bool = False,
        use_e2e_wireless_table_rec_model: bool = True,
        lang: Optional[str] = None,
    ) -> GeneralOCRResultV2:
        """
        Predicts the layout parsing result for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): Input image path, list of image paths,
                                                                        numpy array of an image, or list of numpy arrays.
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_textline_orientation (Optional[bool]): Whether to use textline orientation prediction.
            use_seal_recognition (Optional[bool]): Whether to use seal recognition.
            use_table_recognition (Optional[bool]): Whether to use table recognition.
            use_formula_recognition (Optional[bool]): Whether to use formula recognition.
            use_region_detection (Optional[bool]): Whether to use region detection.
            layout_threshold (Optional[float]): The threshold value to filter out low-confidence predictions. Default is None.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to False.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
                Defaults to None.
                If it's a single number, then both width and height are used.
                If it's a tuple of two numbers, then they are used separately for width and height respectively.
                If it's None, then no unclipping will be performed.
            layout_merge_bboxes_mode (Optional[str], optional): The mode for merging bounding boxes. Defaults to None.
            text_det_limit_side_len (Optional[int]): Maximum side length for text detection.
            text_det_limit_type (Optional[str]): Type of limit to apply for text detection.
            text_det_thresh (Optional[float]): Threshold for text detection.
            text_det_box_thresh (Optional[float]): Threshold for text detection boxes.
            text_det_unclip_ratio (Optional[float]): Ratio for unclipping text detection boxes.
            text_rec_score_thresh (Optional[float]): Score threshold for text recognition.
            seal_det_limit_side_len (Optional[int]): Maximum side length for seal detection.
            seal_det_limit_type (Optional[str]): Type of limit to apply for seal detection.
            seal_det_thresh (Optional[float]): Threshold for seal detection.
            seal_det_box_thresh (Optional[float]): Threshold for seal detection boxes.
            seal_det_unclip_ratio (Optional[float]): Ratio for unclipping seal detection boxes.
            seal_rec_score_thresh (Optional[float]): Score threshold for seal recognition.
            use_wired_table_cells_trans_to_html (bool): Whether to use wired table cells trans to HTML.
            use_wireless_table_cells_trans_to_html (bool): Whether to use wireless table cells trans to HTML.
            use_table_orientation_classify (bool): Whether to use table orientation classification.
            use_ocr_results_with_table_cells (bool): Whether to use OCR results processed by table cells.
            use_e2e_wired_table_rec_model (bool): Whether to use end-to-end wired table recognition model.
            use_e2e_wireless_table_rec_model (bool): Whether to use end-to-end wireless table recognition model.
            lang (Optional[str]): Language code for text recognition (e.g., 'uk', 'en', 'ru').
                                 Automatically selects the appropriate recognition model for the language.
            **kwargs (Any): Additional settings to extend functionality.

        Returns:
            GeneralOCRResult: The predicted layout parsing result.
        """
        model_settings = self.get_model_settings(
            use_doc_orientation_classify,
            use_doc_unwarping,
            use_seal_recognition,
            use_table_recognition,
            use_formula_recognition,
            use_chart_recognition,
            use_region_detection,
        )

        if not self.check_model_settings_valid(model_settings):
            yield {"error": "the input params for model settings are invalid!"}

        for batch_data in self.batch_sampler(input):
            image_arrays = self.img_reader(batch_data.instances)

            if model_settings["use_doc_preprocessor"]:
                doc_preprocessor_results = list(
                    self.doc_preprocessor_pipeline(
                        image_arrays,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                    )
                )
            else:
                doc_preprocessor_results = [{"output_img": arr} for arr in image_arrays]

            doc_preprocessor_images = [
                item["output_img"] for item in doc_preprocessor_results
            ]

            layout_det_results = list(
                self.layout_det_model(
                    doc_preprocessor_images,
                    threshold=layout_threshold,
                    layout_nms=layout_nms,
                    layout_unclip_ratio=layout_unclip_ratio,
                    layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                )
            )

            # ================================================================
            # COORDINATE CONVERSION: Layout Detection → BBox Format
            # ================================================================
            self._convert_det_results_to_bbox(layout_det_results)

            imgs_in_doc = [
                gather_imgs(img, res["boxes_legacy"])
                for img, res in zip(doc_preprocessor_images, layout_det_results)
            ]

            if model_settings["use_region_detection"]:
                region_det_results = list(
                    self.region_detection_model(
                        doc_preprocessor_images,
                        layout_nms=True,
                        layout_merge_bboxes_mode="small",
                    ),
                )
                # ============================================================
                # COORDINATE CONVERSION: Region Detection → BBox Format
                # ============================================================
                self._convert_det_results_to_bbox(region_det_results)
            else:
                region_det_results = self._create_empty_region_results(
                    len(doc_preprocessor_images)
                )

            if model_settings["use_formula_recognition"]:
                formula_res_all = list(
                    self.formula_recognition_pipeline(
                        doc_preprocessor_images,
                        use_layout_detection=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        layout_det_res=layout_det_results,
                    ),
                )
                formula_res_lists = [
                    item["formula_res_list"] for item in formula_res_all
                ]
            else:
                formula_res_lists = [[] for _ in doc_preprocessor_images]

            for doc_preprocessor_image, formula_res_list in zip(
                doc_preprocessor_images, formula_res_lists
            ):
                for formula_res in formula_res_list:
                    x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                    doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = 255.0

            overall_ocr_results = list(
                self.general_ocr_pipeline(
                    doc_preprocessor_images,
                    use_textline_orientation=use_textline_orientation,
                    text_det_limit_side_len=text_det_limit_side_len,
                    text_det_limit_type=text_det_limit_type,
                    text_det_thresh=text_det_thresh,
                    text_det_box_thresh=text_det_box_thresh,
                    text_det_unclip_ratio=text_det_unclip_ratio,
                    text_rec_score_thresh=text_rec_score_thresh,
                    lang=lang,
                ),
            )

            for overall_ocr_res in overall_ocr_results:
                overall_ocr_res["rec_labels"] = ["text"] * len(
                    overall_ocr_res["rec_texts"]
                )

            # ================================================================
            # COORDINATE CONVERSION: OCR Results → BBox Format
            # ================================================================
            self._convert_ocr_results_to_bbox(overall_ocr_results)

            if model_settings["use_table_recognition"]:
                table_res_lists = []
                for (
                    layout_det_res,
                    doc_preprocessor_image,
                    overall_ocr_res,
                    formula_res_list,
                    imgs_in_doc_for_img,
                ) in zip(
                    layout_det_results,
                    doc_preprocessor_images,
                    overall_ocr_results,
                    formula_res_lists,
                    imgs_in_doc,
                ):
                    table_contents_for_img = copy.deepcopy(overall_ocr_res)
                    for formula_res in formula_res_list:
                        x_min, y_min, x_max, y_max = list(
                            map(int, formula_res["dt_polys"])
                        )
                        poly_points = [
                            (x_min, y_min),
                            (x_max, y_min),
                            (x_max, y_max),
                            (x_min, y_max),
                        ]
                        table_contents_for_img["dt_polys"].append(poly_points)
                        rec_formula = formula_res["rec_formula"]
                        if not rec_formula.startswith("$") or not rec_formula.endswith(
                            "$"
                        ):
                            rec_formula = f"${rec_formula}$"
                        table_contents_for_img["rec_texts"].append(f"{rec_formula}")
                        if table_contents_for_img["rec_boxes"].size == 0:
                            table_contents_for_img["rec_boxes"] = np.array(
                                [formula_res["dt_polys"]]
                            )
                        else:
                            table_contents_for_img["rec_boxes"] = np.vstack(
                                (
                                    table_contents_for_img["rec_boxes"],
                                    [formula_res["dt_polys"]],
                                )
                            )
                        table_contents_for_img["rec_polys"].append(poly_points)
                        table_contents_for_img["rec_scores"].append(1)

                    for img in imgs_in_doc_for_img:
                        img_path = img["path"]
                        x_min, y_min, x_max, y_max = img["coordinate"]
                        poly_points = [
                            (x_min, y_min),
                            (x_max, y_min),
                            (x_max, y_max),
                            (x_min, y_max),
                        ]
                        table_contents_for_img["dt_polys"].append(poly_points)
                        table_contents_for_img["rec_texts"].append(
                            f'<div style="text-align: center;"><img src="{img_path}" alt="Image" /></div>'
                        )
                        if table_contents_for_img["rec_boxes"].size == 0:
                            table_contents_for_img["rec_boxes"] = np.array(
                                [img["coordinate"]]
                            )
                        else:
                            table_contents_for_img["rec_boxes"] = np.vstack(
                                (table_contents_for_img["rec_boxes"], img["coordinate"])
                            )
                        table_contents_for_img["rec_polys"].append(poly_points)
                        table_contents_for_img["rec_scores"].append(img["score"])

                    table_res_all = list(
                        self.table_recognition_pipeline(
                            doc_preprocessor_image,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_layout_detection=False,
                            use_ocr_model=False,
                            overall_ocr_res=table_contents_for_img,
                            layout_det_res=layout_det_res,
                            cell_sort_by_y_projection=True,
                            use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
                            use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
                            use_table_orientation_classify=use_table_orientation_classify,
                            use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
                            use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
                            use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
                        ),
                    )
                    single_table_res_lists = [
                        item["table_res_list"] for item in table_res_all
                    ]
                    table_res_lists.extend(single_table_res_lists)
            else:
                table_res_lists = [[] for _ in doc_preprocessor_images]

            if model_settings["use_seal_recognition"]:
                seal_res_all = list(
                    self.seal_recognition_pipeline(
                        doc_preprocessor_images,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_layout_detection=False,
                        layout_det_res=layout_det_results,
                        seal_det_limit_side_len=seal_det_limit_side_len,
                        seal_det_limit_type=seal_det_limit_type,
                        seal_det_thresh=seal_det_thresh,
                        seal_det_box_thresh=seal_det_box_thresh,
                        seal_det_unclip_ratio=seal_det_unclip_ratio,
                        seal_rec_score_thresh=seal_rec_score_thresh,
                    ),
                )
                seal_res_lists = [item["seal_res_list"] for item in seal_res_all]
            else:
                seal_res_lists = [[] for _ in doc_preprocessor_images]

            for (
                input_path,
                page_index,
                doc_preprocessor_image,
                doc_preprocessor_res,
                layout_det_res,
                region_det_res,
                overall_ocr_res,
                table_res_list,
                seal_res_list,
                formula_res_list,
                imgs_in_doc_for_img,
            ) in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                doc_preprocessor_images,
                doc_preprocessor_results,
                layout_det_results,
                region_det_results,
                overall_ocr_results,
                table_res_lists,
                seal_res_lists,
                formula_res_lists,
                imgs_in_doc,
            ):
                chart_res_list = []
                if model_settings["use_chart_recognition"]:
                    chart_imgs_list = []
                    for bbox in layout_det_res["boxes"]:
                        if bbox["label"] == "chart":
                            x_min, y_min, x_max, y_max = bbox["coordinate"]
                            chart_img = doc_preprocessor_image[
                                int(y_min) : int(y_max), int(x_min) : int(x_max), :
                            ]
                            chart_imgs_list.append({"image": chart_img})

                    for chart_res_batch in self.chart_recognition_model(
                        input=chart_imgs_list
                    ):
                        chart_res_list.append(chart_res_batch["result"])

                parsing_res_list = self.get_layout_parsing_res(
                    doc_preprocessor_image,
                    region_det_res=region_det_res,
                    layout_det_res=layout_det_res,
                    overall_ocr_res=overall_ocr_res,
                    table_res_list=table_res_list,
                    seal_res_list=seal_res_list,
                    chart_res_list=chart_res_list,
                    formula_res_list=formula_res_list,
                    text_rec_score_thresh=text_rec_score_thresh,
                )

                for formula_res in formula_res_list:
                    x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                    doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = formula_res[
                        "input_img"
                    ]

                # ============================================================
                # BUILD RESULT: Direct initialization with BBox-based structures
                # ============================================================

                # Convert LayoutBlock (legacy) to LayoutBlockData (BBox-based)
                layout_blocks = []
                for block in parsing_res_list:
                    # Convert legacy bbox list [x1,y1,x2,y2] to BBox object
                    bbox_list = block.bbox
                    bbox = BBox(
                        x_min=float(bbox_list[0]),
                        y_min=float(bbox_list[1]),
                        x_max=float(bbox_list[2]),
                        y_max=float(bbox_list[3]),
                        coord_system=CoordinateSystem.PAGE_ABSOLUTE,
                        label=block.label,
                        confidence=getattr(block, 'confidence', 1.0)
                    )

                    layout_block_data = LayoutBlockData(
                        bbox=bbox,
                        label=block.label,
                        content=getattr(block, 'content', ''),
                        order_index=getattr(block, 'order_index', 0),
                        image=getattr(block, 'image', None),
                        confidence=getattr(block, 'confidence', 1.0)
                    )
                    layout_blocks.append(layout_block_data)

                # ============================================================
                # EXTRACT VISUALIZATIONS: Use adapters to extract from Paddle outputs
                # ============================================================
                visualizations = {}

                # Extract layout detection visualizations
                layout_viz = PaddleDetResultAdapter.extract_visualization(layout_det_res)
                if layout_viz:
                    visualizations["layout_det"] = layout_viz

                # Extract region detection visualizations
                if model_settings["use_region_detection"]:
                    region_viz = PaddleDetResultAdapter.extract_visualization(region_det_res)
                    if region_viz:
                        visualizations["region_det"] = region_viz

                # Extract OCR visualizations
                ocr_viz = PaddleOCRResultAdapter.extract_visualization(overall_ocr_res)
                if ocr_viz:
                    visualizations["ocr"] = ocr_viz

                # Extract doc preprocessor visualizations
                if model_settings.get("use_doc_preprocessor", False):
                    if "doc_preprocessor_res" in doc_preprocessor_res:
                        doc_prep_obj = doc_preprocessor_res.get("doc_preprocessor_res")
                        if hasattr(doc_prep_obj, 'img'):
                            visualizations["doc_preprocessor"] = doc_prep_obj.img

                # Direct initialization with BBox-based data
                yield GeneralOCRResultV2(
                    input_path=input_path,
                    page_index=page_index,
                    preprocessed_image=doc_preprocessor_res["output_img"],
                    layout_blocks=layout_blocks,
                    layout_det_bboxes=layout_det_res["bboxes"],
                    region_det_bboxes=region_det_res["bboxes"],
                    ocr_results=overall_ocr_res,
                    tables=table_res_list,
                    formulas=formula_res_list,
                    seals=seal_res_list,
                    charts=chart_res_list,
                    doc_preprocessor_info=doc_preprocessor_res,
                    imgs_in_doc=imgs_in_doc_for_img,
                    model_settings=model_settings,
                    visualizations=visualizations
                )


class GeneralOCRPipeline(_GeneralOCRPipeline):
    pass
