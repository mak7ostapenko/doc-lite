"""
Bounding box operations and utilities for OCR pipelines.

This module consolidates bbox/point processing functionality including:
- Base operator class for transformations
- Point/box conversion utilities
- Box sorting algorithms (quad and polygon boxes)
- OCR word box calculation
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

__all__ = [
    "BaseOperator",
    "convert_points_to_boxes",
    "SortQuadBoxes",
    "SortPolyBoxes",
    "cal_ocr_word_box",
]


# ============================================================================
# Base Operator Class
# ============================================================================


class BaseOperator(ABC):
    """Base Operator for transformations"""

    def __init__(self):
        """Initializes an instance of base operator."""
        super().__init__()

    @abstractmethod
    def __call__(self):
        """
        Declaration of an abstract method. Subclasses are expected to
        provide a concrete implementation of call method.
        """
        raise NotImplementedError(
            "The component method `__call__` has not been implemented yet."
        )


# ============================================================================
# Point/Box Conversion
# ============================================================================


def convert_points_to_boxes(dt_polys: list) -> np.ndarray:
    """
    Converts a list of polygons to a numpy array of bounding boxes.

    Args:
        dt_polys (list): A list of polygons, where each polygon is represented
                        as a list of (x, y) points.

    Returns:
        np.ndarray: A numpy array of bounding boxes, where each box is represented
                    as [left, top, right, bottom].
                    If the input list is empty, returns an empty numpy array.
    """

    if len(dt_polys) > 0:
        dt_polys_tmp = dt_polys.copy()
        dt_polys_tmp = np.array(dt_polys_tmp)
        boxes_left = np.min(dt_polys_tmp[:, :, 0], axis=1)
        boxes_right = np.max(dt_polys_tmp[:, :, 0], axis=1)
        boxes_top = np.min(dt_polys_tmp[:, :, 1], axis=1)
        boxes_bottom = np.max(dt_polys_tmp[:, :, 1], axis=1)
        dt_boxes = np.array([boxes_left, boxes_top, boxes_right, boxes_bottom])
        dt_boxes = dt_boxes.T
    else:
        dt_boxes = np.array([])
    return dt_boxes


# ============================================================================
# Box Sorting Classes
# ============================================================================


class SortQuadBoxes(BaseOperator):
    """SortQuadBoxes Operator."""

    def __init__(self):
        """Initializes the class."""
        super().__init__()

    def __call__(self, dt_polys: List[np.ndarray]) -> np.ndarray:
        """
        Sort quad boxes in order from top to bottom, left to right
        args:
            dt_polys(ndarray):detected quad boxes with shape [4, 2]
        return:
            sorted boxes(ndarray) with shape [4, 2]
        """
        dt_boxes = np.array(dt_polys)
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes


class SortPolyBoxes(BaseOperator):
    """SortPolyBoxes Operator."""

    def __init__(self):
        """Initializes the class."""
        super().__init__()

    def __call__(self, dt_polys: List[np.ndarray]) -> np.ndarray:
        """
        Sort poly boxes in order from top to bottom, left to right
        args:
            dt_polys(ndarray):detected poly boxes with a [N, 2] np.ndarray list
        return:
            sorted boxes(ndarray) with [N, 2] np.ndarray list
        """
        num_boxes = len(dt_polys)
        if num_boxes == 0:
            return dt_polys
        else:
            y_min_list = []
            for bno in range(num_boxes):
                y_min_list.append(min(dt_polys[bno][:, 1]))
            rank = np.argsort(np.array(y_min_list))
            dt_polys_rank = []
            for no in range(num_boxes):
                dt_polys_rank.append(dt_polys[rank[no]])
            return dt_polys_rank


# ============================================================================
# OCR Word Box Calculation
# ============================================================================


def _sort_word_boxes(boxes, y_thresh=10):
    """
    Internal helper to sort word boxes by reading order.

    Args:
        boxes: List of boxes to sort
        y_thresh: Y-axis threshold for grouping boxes into lines

    Returns:
        Sorted list of boxes
    """
    box_centers = [np.mean(box, axis=0) for box in boxes]
    items = list(zip(boxes, box_centers))
    items.sort(key=lambda x: x[1][1])

    lines = []
    current_line = []
    last_y = None
    for box, center in items:
        if last_y is None or abs(center[1] - last_y) < y_thresh:
            current_line.append((box, center))
        else:
            lines.append(current_line)
            current_line = [(box, center)]
        last_y = center[1]
    if current_line:
        lines.append(current_line)

    final_box = []
    for line in lines:
        line = sorted(line, key=lambda x: x[1][0])
        final_box.extend(box for box, center in line)

    return final_box


def cal_ocr_word_box(rec_str, box, rec_word_info):
    """
    Calculate the detection frame for each word based on the results of
    recognition and detection of OCR.

    Args:
        rec_str: Recognition string
        box: Detection box coordinates
        rec_word_info: Word information tuple (col_num, word_list, word_col_list, state_list)

    Returns:
        Tuple of (word_box_content_list, word_box_list)
    """
    col_num, word_list, word_col_list, state_list = rec_word_info
    box = box.tolist()
    bbox_x_start = box[0][0]
    bbox_x_end = box[1][0]
    bbox_y_start = box[0][1]
    bbox_y_end = box[2][1]

    cell_width = (bbox_x_end - bbox_x_start) / col_num

    word_box_list = []
    word_box_content_list = []
    cn_width_list = []
    cn_col_list = []
    for word, word_col, state in zip(word_list, word_col_list, state_list):
        if state == "cn":
            if len(word_col) != 1:
                char_seq_length = (word_col[-1] - word_col[0] + 1) * cell_width
                char_width = char_seq_length / (len(word_col) - 1)
                cn_width_list.append(char_width)
            cn_col_list += word_col
            word_box_content_list += word
        else:
            cell_x_start = bbox_x_start + int(word_col[0] * cell_width)
            cell_x_end = bbox_x_start + int((word_col[-1] + 1) * cell_width)
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)
            word_box_content_list.append("".join(word))
    if len(cn_col_list) != 0:
        if len(cn_width_list) != 0:
            avg_char_width = np.mean(cn_width_list)
        else:
            avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_str)
        for center_idx in cn_col_list:
            center_x = (center_idx + 0.5) * cell_width
            cell_x_start = max(int(center_x - avg_char_width / 2), 0) + bbox_x_start
            cell_x_end = (
                min(int(center_x + avg_char_width / 2), bbox_x_end - bbox_x_start)
                + bbox_x_start
            )
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)
    word_box_list = _sort_word_boxes(word_box_list, y_thresh=12)
    return word_box_content_list, word_box_list
