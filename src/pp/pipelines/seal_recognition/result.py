
from typing import Dict

import numpy as np

from src.pp.common import BaseCVResult, JsonMixin


class SealRecognitionResult(BaseCVResult):
    """Seal Recognition Result"""

    def _to_img(self) -> Dict[str, np.ndarray]:
        res_img_dict = {}
        layout_det_res = self["layout_det_res"]
        if len(layout_det_res) > 0:
            res_img_dict["layout_det_res"] = layout_det_res.img["res"]

        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            res_img_dict.update(**self["doc_preprocessor_res"].img)

        for sno in range(len(self["seal_res_list"])):
            seal_res = self["seal_res_list"][sno]
            seal_region_id = seal_res["seal_region_id"]
            sub_seal_res_dict = seal_res.img
            key = f"seal_res_region{seal_region_id}"
            res_img_dict[key] = sub_seal_res_dict["ocr_res_img"]
        return res_img_dict

    def _to_str(self, *args, **kwargs) -> Dict[str, str]:
        """Converts the instance's attributes to a dictionary and then to a string.

        Args:
            *args: Additional positional arguments passed to the base class method.
            **kwargs: Additional keyword arguments passed to the base class method.

        Returns:
            Dict[str, str]: A dictionary with the instance's attributes converted to strings.
        """
        data = {}
        data["input_path"] = self["input_path"]
        data["model_settings"] = self["model_settings"]
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].str["res"]
        if len(self["layout_det_res"]) > 0:
            data["layout_det_res"] = self["layout_det_res"].str["res"]
        data["seal_res_list"] = []
        for sno in range(len(self["seal_res_list"])):
            seal_res = self["seal_res_list"][sno]
            data["seal_res_list"].append(seal_res.str["res"])
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs) -> Dict[str, str]:
        """
        Converts the object's data to a JSON dictionary.

        Args:
            *args: Positional arguments passed to the JsonMixin._to_json method.
            **kwargs: Keyword arguments passed to the JsonMixin._to_json method.

        Returns:
            Dict[str, str]: A dictionary containing the object's data in JSON format.
        """
        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        data["model_settings"] = self["model_settings"]
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
        if len(self["layout_det_res"]) > 0:
            data["layout_det_res"] = self["layout_det_res"].json["res"]
        data["seal_res_list"] = []
        for sno in range(len(self["seal_res_list"])):
            seal_res = self["seal_res_list"][sno]
            data["seal_res_list"].append(seal_res.json["res"])
        return JsonMixin._to_json(data, *args, **kwargs)
