import math

import numpy as np
from PIL import Image
from src.pp.models.common import vision_funcs as F

import cv2


class _BaseResize:
    def __init__(self, size_divisor, interp, backend="cv2"):
        _CV2_INTERP_DICT = {
            "NEAREST": cv2.INTER_NEAREST,
            "LINEAR": cv2.INTER_LINEAR,
            "BICUBIC": cv2.INTER_CUBIC,
            "AREA": cv2.INTER_AREA,
            "LANCZOS4": cv2.INTER_LANCZOS4,
        }
        _PIL_INTERP_DICT = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "BOX": Image.BOX,
            "LANCZOS4": Image.LANCZOS,
        }

        super().__init__()

        if size_divisor is not None:
            assert isinstance(
                size_divisor, int
            ), "`size_divisor` should be None or int."
        self.size_divisor = size_divisor

        try:
            interp = interp.upper()
            if backend == "cv2":
                interp = _CV2_INTERP_DICT[interp]
            elif backend == "pil":
                interp = _PIL_INTERP_DICT[interp]
            else:
                raise ValueError("backend must be `cv2` or `pil`")
        except KeyError:
            raise ValueError(
                "For backend '{}', `interp` should be one of {}. Please ensure the interpolation method matches the selected backend.".format(
                    backend,
                    (
                        _CV2_INTERP_DICT.keys()
                        if backend == "cv2"
                        else _PIL_INTERP_DICT.keys()
                    ),
                )
            )
        self.interp = interp
        self.backend = backend

    @staticmethod
    def _rescale_size(img_size, target_size):
        """rescale size"""
        scale = min(max(target_size) / max(img_size), min(target_size) / min(img_size))
        rescaled_size = [round(i * scale) for i in img_size]
        return rescaled_size, scale


class Resize(_BaseResize):
    """Resize the image."""

    def __init__(
        self,
        target_size,
        keep_ratio=False,
        size_divisor=None,
        interp="LINEAR",
        backend="cv2",
    ):
        """
        Initialize the instance.

        Args:
            target_size (list|tuple|int): Target width and height.
            keep_ratio (bool, optional): Whether to keep the aspect ratio of resized
                image. Default: False.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp, backend=backend)

        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        F.check_image_size(target_size)
        self.target_size = target_size

        self.keep_ratio = keep_ratio

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        target_size = self.target_size
        original_size = img.shape[:2][::-1]

        if self.keep_ratio:
            h, w = img.shape[0:2]
            target_size, _ = self._rescale_size((w, h), self.target_size)

        if self.size_divisor:
            target_size = [
                math.ceil(i / self.size_divisor) * self.size_divisor
                for i in target_size
            ]
        img = F.resize(img, target_size, interp=self.interp, backend=self.backend)
        return img


class ResizeByLong(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    longest side.
    """

    def __init__(
        self, target_long_edge, size_divisor=None, interp="LINEAR", backend="cv2"
    ):
        """
        Initialize the instance.

        Args:
            target_long_edge (int): Target length of the longest side of image.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp, backend=backend)
        self.target_long_edge = target_long_edge

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        h, w = img.shape[:2]
        scale = self.target_long_edge / max(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize / self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize / self.size_divisor) * self.size_divisor

        img = F.resize(
            img, (w_resize, h_resize), interp=self.interp, backend=self.backend
        )
        return img


class ResizeByShort(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    shortest side.
    """

    def __init__(
        self, target_short_edge, size_divisor=None, interp="LINEAR", backend="cv2"
    ):
        """
        Initialize the instance.

        Args:
            target_short_edge (int): Target length of the shortest side of image.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp, backend=backend)
        self.target_short_edge = target_short_edge

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        h, w = img.shape[:2]
        scale = self.target_short_edge / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize / self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize / self.size_divisor) * self.size_divisor

        img = F.resize(
            img, (w_resize, h_resize), interp=self.interp, backend=self.backend
        )
        return img

class Normalize:
    """Normalize the three-channel image."""

    def __init__(self, scale=1.0 / 255, mean=0.5, std=0.5):
        """
        Initialize the instance.

        Args:
            scale (float, optional): Scaling factor to apply to the image before
                applying normalization. Default: 1/255.
            mean (float|tuple|list, optional): Means for each channel of the image.
                Default: 0.5.
            std (float|tuple|list|np.ndarray, optional): Standard deviations for each channel
                of the image. Default: 0.5.
        """
        super().__init__()

        if isinstance(mean, float):
            mean = [mean] * 3
        elif len(mean) != 3:
            raise ValueError(
                f"Expected `mean` to be a tuple or list of length 3, but got {len(mean)} elements."
            )
        if isinstance(std, float):
            std = [std] * 3
        elif len(std) != 3:
            raise ValueError(
                f"Expected `std` to be a tuple or list of length 3, but got {len(std)} elements."
            )

        self.alpha = [scale / std[i] for i in range(len(std))]
        self.beta = [-mean[i] / std[i] for i in range(len(std))]

    def norm(self, img):
        split_im = list(cv2.split(img))

        for c in range(img.shape[2]):
            split_im[c] = split_im[c].astype(np.float32)
            split_im[c] *= self.alpha[c]
            split_im[c] += self.beta[c]

        res = cv2.merge(split_im)
        return res

    def __call__(self, imgs):
        """apply"""
        return [self.norm(img) for img in imgs]


class ToCHWImage:
    """Reorder the dimensions of the image from HWC to CHW."""

    def __call__(self, imgs):
        """apply"""
        return [img.transpose((2, 0, 1)) for img in imgs]


class ToBatch:
    def __call__(self, imgs):
        return [np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)]
