import copy

import numpy as np

from src.pp.common import BaseCVResult, JsonMixin

import cv2


class TableRecResult(BaseCVResult):
    """SaveTableResults"""

    def _to_img(self):
        image = self["input_img"]
        bbox_res = self["bbox"]
        if len(bbox_res) > 0 and len(bbox_res[0]) == 4:
            vis_img = self.draw_rectangle(image, bbox_res)
        else:
            vis_img = self.draw_bbox(image, bbox_res)
        return {"res": vis_img}

    def draw_rectangle(self, image, boxes):
        """draw_rectangle"""
        boxes = np.array(boxes)
        img_show = image.copy()
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img_show

    def draw_bbox(self, image, boxes):
        """draw_bbox"""
        for box in boxes:
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
        return image

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
