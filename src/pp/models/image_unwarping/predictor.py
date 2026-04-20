from typing import Any, Dict, List, Tuple, Union

import numpy as np

from src.pp.common.image_batch_sampler import ImageBatchSampler
from src.pp.common.image_reader import ReadImage
from src.pp.models.base_predictor import BasePredictor
from src.pp.models.common.vision_processors import Normalize, ToBatch, ToCHWImage
from src.pp.models.image_unwarping.processors import DocTrPostProcess
from src.pp.models.image_unwarping.result import DocTrResult


class WarpPredictor(BasePredictor):
    """WarpPredictor that inherits from BasePredictor."""


    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes WarpPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.preprocessors, self.infer, self.postprocessors = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        """Builds and returns an ImageBatchSampler instance.

        Returns:
            ImageBatchSampler: An instance of ImageBatchSampler.
        """
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        """Returns the warpping result, DocTrResult.

        Returns:
            type: The DocTrResult.
        """
        return DocTrResult

    def _build(self) -> Tuple:
        """Build the preprocessors, inference engine, and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors, inference engine, and postprocessors.
        """
        preprocessors = {"Read": ReadImage(format="BGR")}
        preprocessors["Normalize"] = Normalize(mean=0.0, std=1.0, scale=1.0 / 255)
        preprocessors["ToCHW"] = ToCHWImage()
        preprocessors["ToBatch"] = ToBatch()

        infer = self.create_static_infer()

        postprocessors = {"DocTrPostProcess": DocTrPostProcess()}
        return preprocessors, infer, postprocessors

    def process(self, batch_data: List[Union[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data.instances)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_raw_imgs)
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        x = self.preprocessors["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x=x)
        batch_warp_preds = self.postprocessors["DocTrPostProcess"](batch_preds)

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "doctr_img": batch_warp_preds,
        }
