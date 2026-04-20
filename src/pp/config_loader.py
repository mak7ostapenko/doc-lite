"""
Configuration loading for PP-StructureV3 pipeline.

Loads YAML configurations and resolves model directories.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_pp_structure_config() -> Dict[str, Any]:
    """
    Load all configurations for PP-StructureV3 pipeline from YAML.

    Reads the main PP-StructureV3.yaml and extracts all nested SubModule configs.
    Follows the original pipeline's approach of extracting dictionaries from YAML.

    Returns:
        Dict containing all loaded configurations with descriptive keys
    """
    # Get PaddleX config directory (in this repository)
    current_file = Path(__file__).resolve()
    repo_root = current_file.parent.parent  # src/ -> repo root
    paddlex_config_dir = repo_root / "paddlex" / "configs"
    main_config_path = paddlex_config_dir / "pipelines/PP-StructureV3.yaml"

    # Load main config
    with open(main_config_path, "r") as f:
        main_config = yaml.safe_load(f)

    # Extract SubModules configs (directly in YAML, not separate files)
    sub_modules = main_config.get("SubModules", {})
    sub_pipelines = main_config.get("SubPipelines", {})

    # Main pipeline models (directly in SubModules)
    layout_det_config = sub_modules.get("LayoutDetection", {})
    region_det_config = sub_modules.get("RegionDetection", {})
    chart_rec_config = sub_modules.get("ChartRecognition", {})

    # Doc Preprocessor pipeline
    doc_preprocessor_config = sub_pipelines.get("DocPreprocessor", {})
    doc_preprocessor_sub = doc_preprocessor_config.get("SubModules", {})
    doc_ori_config = doc_preprocessor_sub.get("DocOrientationClassify", {})
    doc_unwarp_config = doc_preprocessor_sub.get("DocUnwarping", {})

    # General OCR pipeline
    general_ocr_config = sub_pipelines.get("GeneralOCR", {})
    ocr_sub = general_ocr_config.get("SubModules", {})
    text_det_config = ocr_sub.get("TextDetection", {})
    textline_ori_config = ocr_sub.get("TextLineOrientation", {})
    text_rec_config = ocr_sub.get("TextRecognition", {})

    # Table Recognition pipeline
    table_rec_config = sub_pipelines.get("TableRecognition", {})
    table_rec_sub = table_rec_config.get("SubModules", {})
    table_cls_config = table_rec_sub.get("TableClassification", {})
    wired_table_structure_config = table_rec_sub.get("WiredTableStructureRecognition", {})
    wireless_table_structure_config = table_rec_sub.get("WirelessTableStructureRecognition", {})
    wired_cells_det_config = table_rec_sub.get("WiredTableCellsDetection", {})
    wireless_cells_det_config = table_rec_sub.get("WirelessTableCellsDetection", {})
    table_ori_config = table_rec_sub.get("TableOrientationClassify", {})

    # Table OCR (nested in TableRecognition SubPipelines)
    table_ocr_config = table_rec_config.get("SubPipelines", {}).get("GeneralOCR", {})
    table_ocr_sub = table_ocr_config.get("SubModules", {})
    table_ocr_text_det_config = table_ocr_sub.get("TextDetection", {})
    table_ocr_text_rec_config = table_ocr_sub.get("TextRecognition", {})

    # Seal Recognition pipeline
    seal_rec_config = sub_pipelines.get("SealRecognition", {})
    seal_ocr_config = seal_rec_config.get("SubPipelines", {}).get("SealOCR", {})
    seal_ocr_sub = seal_ocr_config.get("SubModules", {})
    seal_text_det_config = seal_ocr_sub.get("TextDetection", {})

    # Formula Recognition pipeline
    formula_rec_config = sub_pipelines.get("FormulaRecognition", {})
    formula_rec_sub = formula_rec_config.get("SubModules", {})
    formula_rec_model_config = formula_rec_sub.get("FormulaRecognition", {})

    # Return all configs with descriptive keys
    return {
        "main_config": main_config,
        "layout_detection_config": layout_det_config,
        "region_detection_config": region_det_config,
        "chart_recognition_config": chart_rec_config,
        "doc_orientation_config": doc_ori_config,
        "doc_unwarping_config": doc_unwarp_config,
        "text_detection_config": text_det_config,
        "textline_orientation_config": textline_ori_config,
        "text_recognition_config": text_rec_config,
        "table_classification_config": table_cls_config,
        "wired_table_structure_config": wired_table_structure_config,
        "wireless_table_structure_config": wireless_table_structure_config,
        "wired_table_cells_detection_config": wired_cells_det_config,
        "wireless_table_cells_detection_config": wireless_cells_det_config,
        "table_orientation_config": table_ori_config,
        "table_ocr_text_detection_config": table_ocr_text_det_config,
        "table_ocr_text_recognition_config": table_ocr_text_rec_config,
        "seal_text_detection_config": seal_text_det_config,
        "formula_recognition_config": formula_rec_model_config,
    }


def get_language_specific_model_name(model_name: str, lang: str = "en") -> str:
    """
    Transform model name for language-specific variants.

    Args:
        model_name: Base model name (e.g., "PP-OCRv5_server_rec")
        lang: Language code ("en", "uk", "ru")

    Returns:
        Transformed model name:
        - English: PP-OCRv5_server_rec (no transformation, uses server model)
        - Ukrainian/Russian: eslav_PP-OCRv5_mobile_rec (uses mobile model)

    Raises:
        ValueError: If unsupported language is provided
    """
    # Only transform recognition models
    if "rec" not in model_name:
        return model_name

    # English: Use server model as-is (no transformation)
    if lang == "en" or not lang:
        return model_name  # PP-OCRv5_server_rec

    # Ukrainian/Russian: Transform to eslav mobile model
    if lang in ("uk", "ru"):
        # Convert server to mobile
        if "server" in model_name:
            model_name = model_name.replace("server", "mobile")

        # Add eslav prefix if not already present
        if not model_name.startswith("eslav_"):
            model_name = f"eslav_{model_name}"

        return model_name  # eslav_PP-OCRv5_mobile_rec

    # Unsupported language
    raise ValueError(f"Unsupported language: {lang}. Supported languages: en, uk, ru")


def get_model_dir(model_name: str) -> str:
    """
    Get the full path to a model directory.

    Args:
        model_name: Name of the model (e.g., "PP-OCRv5_server_det", "eslav_PP-OCRv5_mobile_rec")
                    Should already be transformed by get_language_specific_model_name() if needed.

    Returns:
        Full path to the model directory
    """
    checkpoints_dir = os.path.expanduser("~/.paddlex/official_models")
    return os.path.join(checkpoints_dir, model_name)
