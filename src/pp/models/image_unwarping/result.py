import copy

import numpy as np

from src.pp.common import BaseCVResult, JsonMixin


class DocTrResult(BaseCVResult):
    """
    Result class for DocTr, encapsulating the output of a document image processing task.

    Attributes:
        (inherited from BaseCVResult): Any attributes defined in the base class.

    Methods:
        _to_img(self) -> np.ndarray:
            Converts the stored image result to a numpy array.
    """

    def _to_img(self) -> np.ndarray:
        result = np.array(self["doctr_img"])
        return {"res": result}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        data["doctr_img"] = "..."
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
