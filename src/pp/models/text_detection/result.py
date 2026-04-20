import copy

import numpy as np

from src.pp.common import BaseCVResult, JsonMixin

import cv2


class TextDetResult(BaseCVResult):

    def _to_img(self):
        """draw rectangle"""
        boxes = self["dt_polys"]
        image = self["input_img"]
        for box in boxes:
            box = np.reshape(np.array(box).astype(int), [-1, 1, 2]).astype(np.int64)
            cv2.polylines(image, [box], True, (0, 0, 255), 2)
        return {"res": image[:, :, ::-1]}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
