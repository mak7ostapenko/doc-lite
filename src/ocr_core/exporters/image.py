"""
ImageExporter - Creates visualization images from OCR results.

Maintains 100% backward compatibility with original _to_img() method.
"""

import copy
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.ocr_core.visualisation.fonts import PINGFANG_FONT
from src.pp.pipelines.layout_parsing.utils import get_show_color

from .base import BaseExporter

from src.ocr_core.result import GeneralOCRResultV2


class ImageExporter(BaseExporter):
    """
    Export OCR results as visualization images.

    Creates multiple images showing:
    - Layout detection boxes
    - Region detection boxes
    - OCR results
    - Table cells
    - Seal detection
    - Layout ordering

    Exact copy of logic from original _to_img() method (lines 141-203).
    """

    def export(self, result: 'GeneralOCRResultV2', output_path: Path) -> Dict[str, np.ndarray]:
        """
        Generate all visualization images.

        Args:
            result: GeneralOCRResultV2 instance
            output_path: Directory to save images

        Returns:
            Dictionary mapping image names to numpy arrays
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        res_img_dict = {}
        model_settings = result.model_settings
        viz = result.visualizations

        # Doc preprocessor visualizations
        if model_settings.get("use_doc_preprocessor", False):
            if "doc_preprocessor" in viz:
                for key, value in viz["doc_preprocessor"].items():
                    res_img_dict[key] = value

        # Layout detection visualization
        if "layout_det" in viz and "res" in viz["layout_det"]:
            res_img_dict["layout_det_res"] = viz["layout_det"]["res"]

        # Region detection visualization
        if model_settings.get("use_region_detection", False):
            if "region_det" in viz and "res" in viz["region_det"]:
                res_img_dict["region_det_res"] = viz["region_det"]["res"]

        # OCR visualization
        if "ocr" in viz and "ocr_res_img" in viz["ocr"]:
            res_img_dict["overall_ocr_res"] = viz["ocr"]["ocr_res_img"]

        # Table cell visualization
        if model_settings.get("use_table_recognition", False) and len(result.tables) > 0:
            table_cell_img = Image.fromarray(
                copy.deepcopy(result.preprocessed_image[:, :, ::-1])
            )
            table_draw = ImageDraw.Draw(table_cell_img)
            rectangle_color = (255, 0, 0)
            for sno in range(len(result.tables)):
                table_res = result.tables[sno]
                cell_box_list = table_res.get("cell_box_list", [])
                for box in cell_box_list:
                    x1, y1, x2, y2 = [int(pos) for pos in box]
                    table_draw.rectangle(
                        [x1, y1, x2, y2], outline=rectangle_color, width=2
                    )
            res_img_dict["table_cell_img"] = table_cell_img

        # Seal visualization
        if model_settings.get("use_seal_recognition", False) and len(result.seals) > 0:
            for sno in range(len(result.seals)):
                seal_res = result.seals[sno]
                seal_region_id = seal_res.get("seal_region_id")
                if hasattr(seal_res, 'img'):
                    sub_seal_res_dict = seal_res.img
                    key = f"seal_res_region{seal_region_id}"
                    res_img_dict[key] = sub_seal_res_dict["ocr_res_img"]

        # Layout ordering visualization (using BBox format)
        image = Image.fromarray(result.preprocessed_image[:, :, ::-1])
        draw = ImageDraw.Draw(image, "RGBA")
        font_size = int(0.018 * int(image.width)) + 2
        font = ImageFont.truetype(PINGFANG_FONT.path, font_size, encoding="utf-8")

        for block in result.layout_blocks:
            bbox = block.bbox.to_list()  # Convert BBox to [x1, y1, x2, y2]
            index = block.order_index
            label = block.label
            fill_color = get_show_color(label, False)
            draw.rectangle(bbox, fill=fill_color)
            if index is not None:
                text_position = (bbox[2] + 2, bbox[1] - font_size // 2)
                if int(image.width) - bbox[2] < font_size:
                    text_position = (
                        int(bbox[2] - font_size * 1.1),
                        bbox[1] - font_size // 2,
                    )
                draw.text(text_position, str(index), font=font, fill="red")

        res_img_dict["layout_order_res"] = image

        # Save all images to files
        for img_name, img_data in res_img_dict.items():
            if isinstance(img_data, Image.Image):
                img_data.save(output_path / f"{img_name}.jpg")
            elif isinstance(img_data, np.ndarray):
                Image.fromarray(img_data).save(output_path / f"{img_name}.jpg")

        return res_img_dict
