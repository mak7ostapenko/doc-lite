from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.pp.utils.official_models import official_models

from src.pp import model_lists

from src.pp.models.base_predictor import BasePredictor

from .formula_recognition import FormulaRecPredictor
from .image_classification import ClasPredictor
from .image_unwarping import WarpPredictor
from .object_detection import DetPredictor
from .table_structure_recognition import TablePredictor
from .text_detection import TextDetPredictor
from .text_recognition import TextRecPredictor


# Explicit predictor registry - replaces metaclass auto-registration
# Each model name maps to its predictor class
PREDICTOR_REGISTRY = {}

# Build registry from MODELS lists (flattened structure)
# TextDetPredictor: 8 models
for model_name in model_lists.TEXT_DET_MODELS:
    PREDICTOR_REGISTRY[model_name] = TextDetPredictor

# TextRecPredictor: 27 models
for model_name in model_lists.TEXT_REC_MODELS:
    PREDICTOR_REGISTRY[model_name] = TextRecPredictor

# ClasPredictor: 100 models
for model_name in model_lists.CLAS_MODELS:
    PREDICTOR_REGISTRY[model_name] = ClasPredictor

# DetPredictor: 86 models
for model_name in model_lists.DET_MODELS:
    PREDICTOR_REGISTRY[model_name] = DetPredictor

# FormulaRecPredictor: 7 models
for model_name in model_lists.FORMULA_MODELS:
    PREDICTOR_REGISTRY[model_name] = FormulaRecPredictor

# TablePredictor: 4 models
for model_name in model_lists.TABLE_MODELS:
    PREDICTOR_REGISTRY[model_name] = TablePredictor

# WarpPredictor: 1 model
for model_name in model_lists.UNWARP_MODELS:
    PREDICTOR_REGISTRY[model_name] = WarpPredictor


def create_predictor(
    model_name: str,
    model_dir: Optional[str] = None,
    device=None,
    pp_option=None,
    *args,
    **kwargs,
) -> BasePredictor:
    
    if model_dir is None:
        assert (
            model_name in official_models
        ), f"The model ({model_name}) is not supported! Please using directory of local model files or model name supported by PaddleX!"
        model_dir = official_models[model_name]
    else:
        assert Path(model_dir).exists(), f"{model_dir} is not exists!"
        model_dir = Path(model_dir)

    config = BasePredictor.load_config(model_dir)
    assert (
        model_name == config["Global"]["model_name"]
    ), f"Model name mismatch，please input the correct model dir."

    # Lookup predictor class from explicit registry
    if model_name not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Check if model is supported or provide a valid model_dir path."
        )

    predictor_class = PREDICTOR_REGISTRY[model_name]
    return predictor_class(
        model_dir=model_dir,
        config=config,
        device=device,
        pp_option=pp_option,
        *args,
        **kwargs,
    )
