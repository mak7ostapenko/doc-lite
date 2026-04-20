"""
Layout and region detection stages.

Detects document structure elements (blocks, regions).
"""

from src.pp.pipelines.universal_pipeline.context import PipelineContext
from src.pp.pipelines.universal_pipeline.stages.base import PipelineStage
from src.pp.pipelines.layout_parsing.utils import gather_imgs


class LayoutDetectionStage(PipelineStage):
    """
    Layout detection stage.

    Detects document layout blocks (text, tables, images, formulas, etc.).

    Inputs (from context):
    - preprocessed_images: Preprocessed images

    Outputs (to context):
    - layout_det_results: Detected layout blocks
    - imgs_in_doc: Extracted images from document
    """

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute layout detection.

        Args:
            context: Pipeline context with preprocessed_images

        Returns:
            Updated context with layout_det_results
        """
        params = context.params
        layout_threshold = params.get("layout_threshold")
        layout_nms = params.get("layout_nms")
        layout_unclip_ratio = params.get("layout_unclip_ratio")
        layout_merge_bboxes_mode = params.get("layout_merge_bboxes_mode")

        print(f"[LayoutDetection] Detecting layout on {len(context.preprocessed_images)} images")

        layout_det_results = list(
            self.pipeline.layout_det_model(
                context.preprocessed_images,
                threshold=layout_threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            )
        )

        # Extract images from document
        imgs_in_doc = [
            gather_imgs(img, res["boxes"])
            for img, res in zip(context.preprocessed_images, layout_det_results)
        ]

        context.layout_det_results = layout_det_results
        context.imgs_in_doc = imgs_in_doc

        total_blocks = sum(len(res["boxes"]) for res in layout_det_results)
        print(f"[LayoutDetection] Detected {total_blocks} total blocks across {len(layout_det_results)} images")

        return context


class RegionDetectionStage(PipelineStage):
    """
    Region detection stage.

    Detects finer-grained regions within layout blocks.

    Inputs (from context):
    - preprocessed_images: Preprocessed images

    Outputs (to context):
    - region_det_results: Detected regions
    """

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute region detection.

        Args:
            context: Pipeline context with preprocessed_images

        Returns:
            Updated context with region_det_results
        """
        if self.pipeline.use_region_detection:
            print(f"[RegionDetection] Detecting regions on {len(context.preprocessed_images)} images")

            region_det_results = list(
                self.pipeline.region_detection_model(
                    context.preprocessed_images,
                    layout_nms=True,
                    layout_merge_bboxes_mode="small",
                ),
            )

            context.region_det_results = region_det_results

            total_regions = sum(len(res["boxes"]) for res in region_det_results)
            print(f"[RegionDetection] Detected {total_regions} total regions")
        else:
            # No region detection - empty results
            region_det_results = [{"boxes": []} for _ in context.preprocessed_images]
            context.region_det_results = region_det_results

            print(f"[RegionDetection] Skipped (disabled)")

        return context
