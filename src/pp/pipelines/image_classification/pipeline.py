from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.pp.models.image_classification.result import TopkResult
from src.pp.utils.pp_option import PaddlePredictorOption
from src.pp.pipelines.base import BasePipeline


class _ImageClassificationPipeline(BasePipeline):
    """Image Classification Pipeline"""

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.

        """
        super().__init__(
            device=device, pp_option=pp_option
        )

        image_classification_model_config = config["SubModules"]["ImageClassification"]
        model_kwargs = {}
        if (topk := image_classification_model_config.get("topk", None)) is not None:
            model_kwargs = {"topk": topk}
        self.image_classification_model = self.create_model(
            image_classification_model_config, **model_kwargs
        )
        self.topk = image_classification_model_config.get("topk", 5)

    def predict(
        self, input: Union[str, List[str], np.ndarray, List[np.ndarray]], **kwargs
    ) -> TopkResult:
        """Predicts image classification results for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            TopkResult: The predicted top k results.
        """

        topk = kwargs.pop("topk", self.topk)
        yield from self.image_classification_model(input, topk=topk)


class ImageClassificationPipeline(_ImageClassificationPipeline):
    pass
