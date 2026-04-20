import numpy as np

import cv2


def rotate_image(image, angle):
    if angle < 0 or angle >= 360:
        raise ValueError("`angle` should be in range [0, 360)")

    if angle < 1e-7:
        return image

    # Should we align corners?
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    mat[0, 2] += (new_w - w) / 2
    mat[1, 2] += (new_h - h) / 2
    dst_size = (new_w, new_h)

    rotated = cv2.warpAffine(
        image,
        mat,
        dst_size,
        flags=cv2.INTER_CUBIC,
    )
    return rotated
