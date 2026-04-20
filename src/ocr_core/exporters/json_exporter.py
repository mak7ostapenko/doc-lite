"""
JSONExporter - Exports OCR results to JSON format.

Maintains 100% backward compatibility with original _to_json() method.
"""

import json
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
import numpy as np

from .base import BaseExporter

from src.ocr_core.result import GeneralOCRResultV2



class JSONExporter(BaseExporter):
    """
    Export OCR results as JSON.

    Exact copy of logic from original _to_json() method (lines 245-293).
    """

    def export(self, result: 'GeneralOCRResultV2', output_path: Path) -> Dict[str, Any]:
        """
        Generate JSON representation.

        Args:
            result: GeneralOCRResultV2 instance
            output_path: Directory to save JSON file

        Returns:
            Dictionary ready for JSON serialization
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        data = {}
        data["input_path"] = result.input_path
        data["page_index"] = result.page_index
        model_settings = result.model_settings
        data["model_settings"] = model_settings

        # Convert layout blocks (using BBox format)
        parsing_res_list = [
            {
                "block_label": block.label,
                "block_content": block.content,
                "block_bbox": block.bbox.to_list(),  # Convert BBox to list
            }
            for block in result.layout_blocks
        ]
        data["parsing_res_list"] = parsing_res_list

        # Doc preprocessor data
        if model_settings.get("use_doc_preprocessor", False):
            if "doc_preprocessor_res" in result.doc_preprocessor_info:
                doc_prep_obj = result.doc_preprocessor_info.get("doc_preprocessor_res")
                if hasattr(doc_prep_obj, 'json'):
                    data["doc_preprocessor_res"] = doc_prep_obj.json["res"]

        # Layout detection data (from BBox format)
        data["layout_det_res"] = {
            "boxes": [
                {
                    "coordinate": bbox.to_list(),
                    "label": bbox.label,
                    "score": bbox.confidence
                }
                for bbox in result.layout_det_bboxes
            ]
        }

        # OCR data (from ocr_results)
        if result.ocr_results:
            # Store OCR result data (already has rec_texts, rec_boxes, etc.)
            data["overall_ocr_res"] = {
                "rec_texts": result.ocr_results.get("rec_texts", []),
                "rec_scores": result.ocr_results.get("rec_scores", []),
                "dt_polys": result.ocr_results.get("dt_polys", []),
            }

        # Table recognition data
        if model_settings.get("use_table_recognition", False) and len(result.tables) > 0:
            data["table_res_list"] = []
            for sno in range(len(result.tables)):
                table_res = result.tables[sno]
                if hasattr(table_res, 'json'):
                    data["table_res_list"].append(table_res.json["res"])
                else:
                    data["table_res_list"].append(table_res)

        # Seal recognition data
        if model_settings.get("use_seal_recognition", False) and len(result.seals) > 0:
            data["seal_res_list"] = []
            for sno in range(len(result.seals)):
                seal_res = result.seals[sno]
                if hasattr(seal_res, 'json'):
                    data["seal_res_list"].append(seal_res.json["res"])
                else:
                    data["seal_res_list"].append(seal_res)

        # Formula recognition data
        if model_settings.get("use_formula_recognition", False) and len(result.formulas) > 0:
            data["formula_res_list"] = []
            for sno in range(len(result.formulas)):
                formula_res = result.formulas[sno]
                if hasattr(formula_res, 'json'):
                    data["formula_res_list"].append(formula_res.json["res"])
                else:
                    data["formula_res_list"].append(formula_res)

        # Save to file
        json_file = output_path / f"{Path(result.input_path).stem}_res.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=self._json_default)

        return data

    @staticmethod
    def _json_default(obj):
        """Handle non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
