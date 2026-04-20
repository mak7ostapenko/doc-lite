import torch
import pyiqa
from PIL import Image
import numpy as np
from pyiqa.models.inference_model import InferenceModel as PyIQAInferenceModel
from pyiqa.default_model_configs import DEFAULT_CONFIGS as pyiqa_models_configs

from src.image_quality.consts import MODELS_THRESHOLDS
from src.image_quality import utils

# NOTE: you can use any model from pyiqa. This variable is consist of model names which were tested 
__supported_models__ = (
    'brisque', 'hyperiqa', 'ilniqe', 'liqe', 'musiq', 'topiq_nr'
)

class ImgQualityAssessmentModel:
    def __init__(
        self, 
        model_name: str = 'topiq_nr',
        path_to_model: str = None,
    ):
        assert model_name in __supported_models__, f"Unknown model_name: {model_name}"

        self.model_name = model_name
        self.path_to_model = path_to_model
        
        self.thr_1 = MODELS_THRESHOLDS[model_name]
        self.thr_2 = MODELS_THRESHOLDS[model_name]

        (
            self._score_range, 
            self._lower_better, 
            self._left_approx, 
            self._right_approx
        ) = utils.parse_score_range(
            model_name=model_name,
            pyiqa_models_configs=pyiqa_models_configs,
        )

        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._model: PyIQAInferenceModel = pyiqa.create_metric(
            model_name, 
            device=self._device,
            pretrained=(False if self.path_to_model else True),
        )
        if self.path_to_model is not None:
            self._model.load_weights(
                weights_path=path_to_model,
            )
            

    def predict(self, img_bgr: np.ndarray):
        img_bgr = Image.fromarray(img_bgr)
        score = float(self._model(img_bgr).cpu().numpy())
        score = utils.check_and_normalise_score(
            score=score, 
            score_range=self._score_range, 
            lower_better=self._lower_better
        )
        return score
        