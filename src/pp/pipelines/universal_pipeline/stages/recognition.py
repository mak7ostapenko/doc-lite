"""
Recognition stages for formulas, tables, seals, and charts.

Each recognition stage handles a specific type of content.
"""

import copy

import numpy as np

from src.pp.pipelines.universal_pipeline.context import PipelineContext
from src.pp.pipelines.universal_pipeline.stages.base import PipelineStage


class FormulaRecognitionStage(PipelineStage):
    """
    Formula recognition stage.

    Recognizes mathematical formulas and converts to LaTeX.

    Inputs (from context):
    - preprocessed_images: Preprocessed images
    - layout_det_results: Layout detection results

    Outputs (to context):
    - formula_res_lists: Formula recognition results

    Side effects:
    - Masks formula regions in preprocessed_images (sets to white)
    """

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute formula recognition.

        Args:
            context: Pipeline context

        Returns:
            Updated context with formula_res_lists
        """
        if self.pipeline.use_formula_recognition:
            print(f"[FormulaRecognition] Recognizing formulas on {len(context.preprocessed_images)} images")

            formula_res_all = list(
                self.pipeline.formula_recognition_pipeline(
                    context.preprocessed_images,
                    use_layout_detection=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    layout_det_res=context.layout_det_results,
                ),
            )
            formula_res_lists = [
                item["formula_res_list"] for item in formula_res_all
            ]

            # Mask formula regions (set to white) to avoid OCR conflicts
            for doc_preprocessor_image, formula_res_list in zip(
                context.preprocessed_images, formula_res_lists
            ):
                for formula_res in formula_res_list:
                    x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                    doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = 255.0

            context.formula_res_lists = formula_res_lists

            total_formulas = sum(len(f_list) for f_list in formula_res_lists)
            print(f"[FormulaRecognition] Recognized {total_formulas} formulas")
        else:
            formula_res_lists = [[] for _ in context.preprocessed_images]
            context.formula_res_lists = formula_res_lists

            print(f"[FormulaRecognition] Skipped (disabled)")

        return context


class TableRecognitionStage(PipelineStage):
    """
    Table recognition stage.

    Recognizes table structure and extracts cell content.

    Inputs (from context):
    - preprocessed_images: Preprocessed images
    - layout_det_results: Layout detection results
    - overall_ocr_results: OCR results
    - formula_res_lists: Formula recognition results
    - imgs_in_doc: Extracted images

    Outputs (to context):
    - table_res_lists: Table recognition results
    """

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute table recognition.

        Args:
            context: Pipeline context

        Returns:
            Updated context with table_res_lists
        """
        params = context.params

        if self.pipeline.use_table_recognition:
            print(f"[TableRecognition] Recognizing tables on {len(context.preprocessed_images)} images")

            table_res_lists = []
            for (
                layout_det_res,
                doc_preprocessor_image,
                overall_ocr_res,
                formula_res_list,
                imgs_in_doc_for_img,
            ) in zip(
                context.layout_det_results,
                context.preprocessed_images,
                context.overall_ocr_results,
                context.formula_res_lists,
                context.imgs_in_doc,
            ):
                # Prepare table contents (OCR + formulas + images)
                table_contents_for_img = copy.deepcopy(overall_ocr_res)

                # Add formulas to table contents
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
                    if not rec_formula.startswith("$") or not rec_formula.endswith("$"):
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

                # Add images to table contents
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
                    table_contents_for_img["rec_texts"].append(f"![]({img_path})")
                    if table_contents_for_img["rec_boxes"].size == 0:
                        table_contents_for_img["rec_boxes"] = np.array(
                            [[x_min, y_min, x_max, y_max]]
                        )
                    else:
                        table_contents_for_img["rec_boxes"] = np.vstack(
                            (
                                table_contents_for_img["rec_boxes"],
                                [[x_min, y_min, x_max, y_max]],
                            )
                        )
                    table_contents_for_img["rec_polys"].append(poly_points)
                    table_contents_for_img["rec_scores"].append(img["score"])

                # Run table recognition
                table_res_all = list(
                    self.pipeline.table_recognition_pipeline(
                        doc_preprocessor_image,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_layout_detection=False,
                        use_ocr_model=False,
                        overall_ocr_res=table_contents_for_img,
                        layout_det_res=layout_det_res,
                        cell_sort_by_y_projection=True,
                        use_wired_table_cells_trans_to_html=params.get("use_wired_table_cells_trans_to_html", False),
                        use_wireless_table_cells_trans_to_html=params.get("use_wireless_table_cells_trans_to_html", False),
                        use_table_orientation_classify=params.get("use_table_orientation_classify", True),
                        use_ocr_results_with_table_cells=params.get("use_ocr_results_with_table_cells", True),
                        use_e2e_wired_table_rec_model=params.get("use_e2e_wired_table_rec_model", False),
                        use_e2e_wireless_table_rec_model=params.get("use_e2e_wireless_table_rec_model", True),
                    ),
                )
                single_table_res_lists = [
                    item["table_res_list"] for item in table_res_all
                ]
                table_res_lists.extend(single_table_res_lists)

            context.table_res_lists = table_res_lists

            total_tables = sum(len(t_list) for t_list in table_res_lists)
            print(f"[TableRecognition] Recognized {total_tables} tables")
        else:
            table_res_lists = [[] for _ in context.preprocessed_images]
            context.table_res_lists = table_res_lists

            print(f"[TableRecognition] Skipped (disabled)")

        return context


class SealRecognitionStage(PipelineStage):
    """
    Seal recognition stage.

    Recognizes seal/stamp text.

    Inputs (from context):
    - preprocessed_images: Preprocessed images
    - layout_det_results: Layout detection results

    Outputs (to context):
    - seal_res_lists: Seal recognition results
    """

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute seal recognition.

        Args:
            context: Pipeline context

        Returns:
            Updated context with seal_res_lists
        """
        params = context.params

        if self.pipeline.use_seal_recognition:
            print(f"[SealRecognition] Recognizing seals on {len(context.preprocessed_images)} images")

            seal_res_all = list(
                self.pipeline.seal_recognition_pipeline(
                    context.preprocessed_images,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_layout_detection=False,
                    layout_det_res=context.layout_det_results,
                    seal_det_limit_side_len=params.get("seal_det_limit_side_len"),
                    seal_det_limit_type=params.get("seal_det_limit_type"),
                    seal_det_thresh=params.get("seal_det_thresh"),
                    seal_det_box_thresh=params.get("seal_det_box_thresh"),
                    seal_det_unclip_ratio=params.get("seal_det_unclip_ratio"),
                    seal_rec_score_thresh=params.get("seal_rec_score_thresh"),
                ),
            )
            seal_res_lists = [item["seal_res_list"] for item in seal_res_all]

            context.seal_res_lists = seal_res_lists

            total_seals = sum(len(s_list) for s_list in seal_res_lists)
            print(f"[SealRecognition] Recognized {total_seals} seals")
        else:
            seal_res_lists = [[] for _ in context.preprocessed_images]
            context.seal_res_lists = seal_res_lists

            print(f"[SealRecognition] Skipped (disabled)")

        return context


class ChartRecognitionStage(PipelineStage):
    """
    Chart recognition stage.

    Recognizes charts and converts to tables.

    Note: This is processed per-image in the main loop,
    not as a batch operation.
    """

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Chart recognition is handled in the main loop.
        This is a placeholder for future batch processing.
        """
        # Chart recognition is currently done per-image in predict()
        # Could be refactored here in the future
        return context
