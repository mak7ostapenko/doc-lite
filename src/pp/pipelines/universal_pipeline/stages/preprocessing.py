"""
Document preprocessing stages.

Handles document orientation classification and unwarping.
"""

from src.pp.pipelines.universal_pipeline.context import PipelineContext
from src.pp.pipelines.universal_pipeline.stages.base import PipelineStage


class DocPreprocessingStage(PipelineStage):
    """
    Document preprocessing stage.

    Applies document orientation classification and/or unwarping
    to correct document orientation and perspective distortion.

    Inputs (from context):
    - input_images: Raw input images

    Outputs (to context):
    - preprocessed_images: Corrected images
    - doc_preprocessor_results: Preprocessing results (orientation, etc.)
    """

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute document preprocessing.

        Args:
            context: Pipeline context with input_images

        Returns:
            Updated context with preprocessed_images
        """
        params = context.params
        use_doc_orientation_classify = params.get("use_doc_orientation_classify")
        use_doc_unwarping = params.get("use_doc_unwarping")

        if self.pipeline.use_doc_preprocessor:
            print(f"[DocPreprocessing] Running with orientation={use_doc_orientation_classify}, unwarping={use_doc_unwarping}")

            doc_preprocessor_results = list(
                self.pipeline.doc_preprocessor_pipeline(
                    context.input_images,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_doc_unwarping=use_doc_unwarping,
                )
            )

            preprocessed_images = [
                item["output_img"] for item in doc_preprocessor_results
            ]

            context.doc_preprocessor_results = doc_preprocessor_results
            context.preprocessed_images = preprocessed_images

            print(f"[DocPreprocessing] Processed {len(preprocessed_images)} images")
        else:
            # No preprocessing - use raw images
            doc_preprocessor_results = [
                {"output_img": arr} for arr in context.input_images
            ]
            context.doc_preprocessor_results = doc_preprocessor_results
            context.preprocessed_images = context.input_images

            print(f"[DocPreprocessing] Skipped (disabled)")

        return context
