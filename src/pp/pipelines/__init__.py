
from typing import Any, Dict, Optional

from src.pp.utils import logging
from src.pp.utils.config import parse_config
from src.pp.utils.pp_option import PaddlePredictorOption
from src.pp.pipelines.base import BasePipeline
from src.pp.pipelines.doc_preprocessor.pipeline import DocPreprocessorPipeline
from src.pp.pipelines.formula_recognition.pipeline import FormulaRecognitionPipeline
from src.pp.pipelines.image_classification.pipeline import ImageClassificationPipeline
from src.pp.pipelines.ocr.pipeline import OCRPipeline
from src.pp.pipelines.seal_recognition.pipeline import SealRecognitionPipeline
from src.pp.pipelines.table_recognition.pipeline_v2 import TableRecognitionPipelineV2
from src.pp.pipelines.layout_parsing.pipeline_v2 import LayoutParsingPipelineV2
from src.pp.pipelines.universal_pipeline import LayoutParsingPipelineV2 as UniversalPipeline

# Note: GeneralOCRPipeline uses lazy import to avoid circular dependency
# (it imports from src.pp.pipelines.base)

PIPELINE_REGISTRY = {
    "OCR": OCRPipeline,
    "PP-StructureV3": "src.general_ocr_pipeline:GeneralOCRPipeline",  # Lazy import
    "table_recognition_v2": TableRecognitionPipelineV2,
    "doc_preprocessor": DocPreprocessorPipeline,
    "formula_recognition": FormulaRecognitionPipeline,
    "image_classification": ImageClassificationPipeline,
    "seal_recognition": SealRecognitionPipeline,
    "flat_universal": UniversalPipeline,
}


def load_pipeline_config(pipeline: str) -> Dict[str, Any]:
    """
    Load the pipeline configuration.

    Args:
        pipeline (str): The name of the pipeline or the path to the config file.

    Returns:
        Dict[str, Any]: The parsed pipeline configuration.

    Raises:
        Exception: If the config file of pipeline does not exist.
    """

    if not (pipeline.endswith(".yml") or pipeline.endswith(".yaml")):
        pipeline_path = f"configs/pipelines/{pipeline}.yaml"
        if pipeline_path is None:
            raise Exception(
                f"The pipeline ({pipeline}) does not exist! Please use a pipeline name or a config file path!"
            )
    else:
        pipeline_path = pipeline
    config = parse_config(pipeline_path)
    return config


def create_pipeline(
    pipeline: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    *args: Any,
    **kwargs: Any,
) -> BasePipeline:
    """
    Create a pipeline instance based on the provided parameters.

    If the input parameter config is not provided, it is obtained from the
    default config corresponding to the pipeline name.

    Args:
        pipeline (Optional[str], optional): The name of the pipeline to
            create, or the path to the config file. Defaults to None.
        config (Optional[Dict[str, Any]], optional): The pipeline configuration.
            Defaults to None.
        device (Optional[str], optional): The device to run the pipeline on.
            Defaults to None.
        pp_option (Optional[PaddlePredictorOption], optional): The options for
            the PaddlePredictor. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        BasePipeline: The created pipeline instance.
    """
    
    if config is None:
        config = load_pipeline_config(pipeline)
    else:
        if pipeline is not None and config["pipeline_name"] != pipeline:
            logging.warning(
                "The pipeline name in the config (%r) is different from the specified pipeline name (%r). %r will be used.",
                config["pipeline_name"],
                pipeline,
                config["pipeline_name"],
            )
        config = config.copy()
    pipeline_name = config["pipeline_name"]

    pipeline_class_or_str = PIPELINE_REGISTRY[pipeline_name]

    # Handle lazy import (string format: "module.path:ClassName")
    if isinstance(pipeline_class_or_str, str):
        module_path, class_name = pipeline_class_or_str.split(":")
        from importlib import import_module
        module = import_module(module_path)
        pipeline_class = getattr(module, class_name)
    else:
        pipeline_class = pipeline_class_or_str

    pipeline = pipeline_class(
        config=config,
        device=device,
        *args,
        **kwargs,
    )
    return pipeline


