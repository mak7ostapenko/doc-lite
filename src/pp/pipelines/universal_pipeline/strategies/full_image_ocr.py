"""
Full-Image OCR Strategy.

Traditional approach: OCR entire image, then spatially match results to layout blocks.
"""

from typing import Any

from src.pp.pipelines.universal_pipeline.context import PipelineContext
from src.pp.pipelines.universal_pipeline.strategies.base import OCRStrategy


class FullImageOCRStrategy(OCRStrategy):
    """
    Full-image OCR followed by spatial matching.

    Flow:
    1. Run text detection on entire preprocessed image
    2. Run text recognition on detected boxes
    3. Spatially match OCR boxes to layout blocks (complex!)
    4. Handle text spanning multiple blocks
    5. Run fallback OCR on blocks without text

    Pros:
    - Can detect text not found by layout detection
    - Robust to layout detection errors

    Cons:
    - Requires ~300 lines of complex matching logic
    - Text boxes may span multiple layout blocks
    - Spatial ambiguity when text is between blocks
    """

    def execute_ocr(self, context: PipelineContext) -> PipelineContext:
        """
        Run full-image OCR on preprocessed images.

        Args:
            context: Pipeline context with preprocessed_images

        Returns:
            Updated context with overall_ocr_results populated
        """
        # Extract parameters from context
        params = context.params
        use_textline_orientation = params.get("use_textline_orientation")
        text_det_limit_side_len = params.get("text_det_limit_side_len")
        text_det_limit_type = params.get("text_det_limit_type")
        text_det_thresh = params.get("text_det_thresh")
        text_det_box_thresh = params.get("text_det_box_thresh")
        text_det_unclip_ratio = params.get("text_det_unclip_ratio")
        text_rec_score_thresh = params.get("text_rec_score_thresh")
        lang = params.get("lang")

        print("[Full-Image OCR] Running OCR on entire images")

        # Run general OCR pipeline on all images
        overall_ocr_results = list(
            self.pipeline.general_ocr_pipeline(
                context.preprocessed_images,
                use_textline_orientation=use_textline_orientation,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_limit_type=text_det_limit_type,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                text_rec_score_thresh=text_rec_score_thresh,
                lang=lang,
            )
        )

        # Add labels (all recognized as generic "text")
        for overall_ocr_res in overall_ocr_results:
            overall_ocr_res["rec_labels"] = ["text"] * len(
                overall_ocr_res["rec_texts"]
            )

        context.overall_ocr_results = overall_ocr_results
        context.block_to_ocr_maps = [None] * len(overall_ocr_results)  # No pre-computed mapping

        print(f"[Full-Image OCR] Detected {sum(len(r['rec_texts']) for r in overall_ocr_results)} text boxes across {len(overall_ocr_results)} images")

        return context

    def standardize_data(
        self,
        context: PipelineContext,
        image_index: int,
    ) -> tuple:
        """
        Standardize using complex spatial matching logic.

        This calls the pipeline's standardized_data() with block_to_ocr_map=None,
        which triggers the full matching logic path.

        Args:
            context: Pipeline context
            image_index: Index of image to process

        Returns:
            Tuple of (region_block_ocr_idx_map, region_det_res, layout_det_res)
        """
        print(f"[Full-Image OCR Path] Using spatial matching for image {image_index}")

        # Call standardized_data with NO pre-computed mapping
        # This triggers the full matching logic (lines 550-573 in pipeline_v2.py)
        return self.pipeline.standardized_data(
            image=context.preprocessed_images[image_index],
            region_det_res=context.region_det_results[image_index],
            layout_det_res=context.layout_det_results[image_index],
            overall_ocr_res=context.overall_ocr_results[image_index],
            formula_res_list=context.formula_res_lists[image_index],
            text_rec_model=self.pipeline.general_ocr_pipeline.text_rec_model,
            text_rec_score_thresh=context.params.get("text_rec_score_thresh"),
            block_to_ocr_map=None,  # None = full matching logic
        )
