"""
Direct Block OCR Strategy.

Efficient approach: Detect layout first, crop text blocks, OCR each block individually.
"""

from typing import Any

from src.pp.pipelines.universal_pipeline.context import PipelineContext
from src.pp.pipelines.universal_pipeline.strategies.base import OCRStrategy


class DirectBlockOCRStrategy(OCRStrategy):
    """
    Direct block-level OCR with 1:1 mapping.

    Flow:
    1. Detect layout blocks
    2. Crop text-related blocks (text, paragraph_title, doc_title, etc.)
    3. Run text recognition on each cropped block
    4. Create direct 1:1 mapping (block_idx → ocr_idx)
    5. Skip complex matching logic (not needed!)

    Pros:
    - ~47% faster (only OCRs text regions)
    - Eliminates ~300 lines of matching code
    - No spatial ambiguity (each block → one OCR result)
    - No multi-block text handling needed

    Cons:
    - Relies on accurate layout detection
    - Won't find text outside detected blocks
    """

    def execute_ocr(self, context: PipelineContext) -> PipelineContext:
        """
        Run direct block-level OCR.

        For each image:
        1. Loop through layout blocks
        2. Crop text-related blocks
        3. Run OCR on crops
        4. Store direct mapping

        Args:
            context: Pipeline context with preprocessed_images and layout_det_results

        Returns:
            Updated context with overall_ocr_results and block_to_ocr_maps populated
        """
        # Extract parameters from context
        params = context.params
        text_rec_score_thresh = params.get("text_rec_score_thresh") or \
                               self.pipeline.general_ocr_pipeline.text_rec_score_thresh
        lang = params.get("lang")

        print("[Direct Block OCR] Processing text blocks directly")

        overall_ocr_results = []
        block_to_ocr_maps = []

        for img_idx, (doc_preprocessor_image, layout_det_res) in enumerate(
            zip(context.preprocessed_images, context.layout_det_results)
        ):
            # Check if we have text blocks detected
            text_labels = [
                "text", "paragraph_title", "doc_title",
                "footnote", "header", "footer", "page_number",
            ]
            has_text_blocks = any(
                box["label"].lower() in text_labels
                for box in layout_det_res["boxes"]
            )

            if has_text_blocks:
                # Use pipeline's _ocr_text_blocks_directly method with FULL OCR parameters
                overall_ocr_res, block_to_ocr_map = self.pipeline._ocr_text_blocks_directly(
                    doc_preprocessor_image,
                    layout_det_res,
                    self.pipeline.general_ocr_pipeline.text_rec_model,
                    text_rec_score_thresh,
                    lang=lang,
                    use_textline_orientation=params.get("use_textline_orientation"),
                    text_det_limit_side_len=params.get("text_det_limit_side_len"),
                    text_det_limit_type=params.get("text_det_limit_type"),
                    text_det_thresh=params.get("text_det_thresh"),
                    text_det_box_thresh=params.get("text_det_box_thresh"),
                    text_det_unclip_ratio=params.get("text_det_unclip_ratio"),
                )
                overall_ocr_results.append(overall_ocr_res)
                block_to_ocr_maps.append(block_to_ocr_map)

                num_blocks = len(block_to_ocr_map)
                num_ocr = len(overall_ocr_res["rec_texts"])
                print(f"[Direct Block OCR] Image {img_idx}: {num_blocks} blocks → {num_ocr} text lines")
            else:
                # Fallback: Use general OCR if no layout blocks (rare edge case)
                print(f"[Direct Block OCR] Image {img_idx}: No text blocks, using fallback OCR")
                ocr_res = list(
                    self.pipeline.general_ocr_pipeline(
                        [doc_preprocessor_image],
                        use_textline_orientation=params.get("use_textline_orientation"),
                        text_det_limit_side_len=params.get("text_det_limit_side_len"),
                        text_det_limit_type=params.get("text_det_limit_type"),
                        text_det_thresh=params.get("text_det_thresh"),
                        text_det_box_thresh=params.get("text_det_box_thresh"),
                        text_det_unclip_ratio=params.get("text_det_unclip_ratio"),
                        text_rec_score_thresh=text_rec_score_thresh,
                        lang=lang,
                    )
                )[0]
                # Add labels for fallback case
                ocr_res["rec_labels"] = ["text"] * len(ocr_res["rec_texts"])
                overall_ocr_results.append(ocr_res)
                block_to_ocr_maps.append(None)  # No direct mapping for fallback

        context.overall_ocr_results = overall_ocr_results
        context.block_to_ocr_maps = block_to_ocr_maps

        total_ocr = sum(len(r["rec_texts"]) for r in overall_ocr_results)
        print(f"[Direct Block OCR] Total: {total_ocr} OCR results across {len(overall_ocr_results)} images")

        return context

    def standardize_data(
        self,
        context: PipelineContext,
        image_index: int,
    ) -> tuple:
        """
        Standardize using pre-computed direct mapping.

        This calls the pipeline's standardized_data() WITH block_to_ocr_map,
        which triggers the simplified direct path (skips matching logic).

        Args:
            context: Pipeline context
            image_index: Index of image to process

        Returns:
            Tuple of (region_block_ocr_idx_map, region_det_res, layout_det_res)
        """
        block_to_ocr_map = context.block_to_ocr_maps[image_index]

        if block_to_ocr_map is not None:
            print(f"[Direct Block OCR Path] Using pre-computed mapping for image {image_index} - skipping ~300 lines of matching logic")
        else:
            print(f"[Direct Block OCR Path] No mapping for image {image_index} (fallback case) - using matching logic")

        # Call standardized_data WITH pre-computed mapping
        # This triggers the direct path (lines 516-549 in pipeline_v2.py)
        return self.pipeline.standardized_data(
            image=context.preprocessed_images[image_index],
            region_det_res=context.region_det_results[image_index],
            layout_det_res=context.layout_det_results[image_index],
            overall_ocr_res=context.overall_ocr_results[image_index],
            formula_res_list=context.formula_res_lists[image_index],
            text_rec_model=self.pipeline.general_ocr_pipeline.text_rec_model,
            text_rec_score_thresh=context.params.get("text_rec_score_thresh"),
            block_to_ocr_map=block_to_ocr_map,  # Pre-computed mapping!
        )
