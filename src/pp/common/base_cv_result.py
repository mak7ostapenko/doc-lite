from pathlib import Path

from src.pp.common.base_result import BaseResult
from src.pp.common.mixin import ImgMixin


class BaseCVResult(BaseResult, ImgMixin):
    """Base class for computer vision results."""

    def __init__(self, data: dict) -> None:
        """
        Initialize the BaseCVResult.

        Args:
            data (dict): The initial data.
        """
        super().__init__(data)
        ImgMixin.__init__(self, "pillow")

    def _get_input_fn(self):
        fn = super()._get_input_fn()
        if (page_idx := self.get("page_index", None)) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            fn = f"{stem}_{page_idx}{suffix}"
        return fn
