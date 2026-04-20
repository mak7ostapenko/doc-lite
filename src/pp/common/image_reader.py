import numpy as np
from src.pp.utils.io.readers import ImageReader

import cv2

class ReadImage:
    """Load image from the file."""

    def __init__(self, format="BGR"):
        """
        Initialize the instance.

        Args:
            format (str, optional): Target color format to convert the image to.
                Choices are 'BGR', 'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__()
        self.format = format
        flags = {
            "BGR": cv2.IMREAD_COLOR,
            "RGB": cv2.IMREAD_COLOR,
            "GRAY": cv2.IMREAD_GRAYSCALE,
        }[self.format]
        self._img_reader = ImageReader(flags=flags)

    def __call__(self, imgs):
        """apply"""
        return [self.read(img) for img in imgs]

    def read(self, img):
        if isinstance(img, np.ndarray):
            if self.format == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif isinstance(img, str):
            blob = self._img_reader.read(img)
            if blob is None:
                raise Exception(f"Image read Error: {img}")

            if self.format == "RGB":
                if blob.ndim != 3:
                    raise RuntimeError("Array is not 3-dimensional.")
                # BGR to RGB
                blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            return blob
        else:
            raise TypeError(
                f"ReadImage only supports the following types:\n"
                f"1. str, indicating a image file path or a directory containing image files.\n"
                f"2. numpy.ndarray.\n"
                f"However, got type: {type(img).__name__}."
            )
