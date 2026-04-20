__all__ = ["get_neighbor_boxes_idx"]

import numpy as np


def get_neighbor_boxes_idx(src_boxes: np.ndarray, ref_box: np.ndarray) -> list:
    """
    Retrieve indices of source boxes that are neighbors to the reference box.

    Parameters:
    src_boxes (np.ndarray): An array of bounding boxes with shape (N, 4),
                            where N is the number of boxes and each box is represented
                            by [x1, y1, x2, y2].
    ref_box (np.ndarray): A single bounding box represented by [x1, y1, x2, y2].

    Returns:
    list: A list of indices of the source boxes that are close to the
          reference box based on the intersection area.
    """
    match_idx_list = []
    if len(src_boxes) > 0:
        x1 = np.maximum(ref_box[0], src_boxes[:, 0])
        y1 = np.maximum(ref_box[1], src_boxes[:, 1])
        x2 = np.minimum(ref_box[2], src_boxes[:, 2])
        y2 = np.minimum(ref_box[3], src_boxes[:, 3])
        pub_w = x2 - x1
        pub_h = y2 - y1
        match_idx = np.where((pub_w > 0) & (pub_h < 3) & (pub_h > -15))[0]
        match_idx_list.extend(match_idx)
    return match_idx_list
