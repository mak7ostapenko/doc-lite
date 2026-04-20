
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.pp import constants
from src.pp.utils import logging
from src.pp.utils.device import get_default_device, parse_device

from src.pp.common.base_batch_sampler import BaseBatchSampler
from src.pp.utils.io.readers import YAMLReader
from src.pp.utils.pp_option import PaddlePredictorOption
from src.pp.models.common.static_infer import ONNXRuntimeInfer


class PredictionWrap:
    """Wraps the prediction data and supports get by index."""

    def __init__(self, data: Dict[str, List[Any]], num: int) -> None:
        """Initializes the PredictionWrap with prediction data.

        Args:
            data (Dict[str, List[Any]]): A dictionary where keys are string identifiers and values are lists of predictions.
            num (int): The number of predictions, that is length of values per key in the data dictionary.

        Raises:
            AssertionError: If the length of any list in data does not match num.
        """
        assert isinstance(data, dict), "data must be a dictionary"
        for k in data:
            assert len(data[k]) == num, f"{len(data[k])} != {num} for key {k}!"
        self._data = data
        self._keys = data.keys()

    def get_by_idx(self, idx: int) -> Dict[str, Any]:
        """Get the prediction by specified index.

        Args:
            idx (int): The index to get predictions from.

        Returns:
            Dict[str, Any]: A dictionary with the same keys as the input data, but with the values at the specified index.
        """
        return {key: self._data[key][idx] for key in self._keys}


class BasePredictor(ABC):
    MODEL_FILE_PREFIX = constants.MODEL_FILE_PREFIX

    def __init__(
        self,
        model_dir: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        device: Optional[str] = None,
        batch_size: int = 1,
        pp_option: Optional[PaddlePredictorOption] = None,
    ) -> None:
        """Initializes the BasePredictor.

        Args:
            model_dir (str): The directory where the model files are stored.
            config (Optional[Dict[str, Any]], optional): The model configuration
                dictionary. Defaults to None.
            device (Optional[str], optional): The device to run the inference
                engine on. Defaults to None.
            batch_size (int, optional): The batch size to predict.
                Defaults to 1.
            pp_option (Optional[PaddlePredictorOption], optional): The inference
                engine options. Defaults to None.

        """
        super().__init__()

        self.model_dir = Path(model_dir)
        self.config = config if config else self.load_config(self.model_dir)
        self.batch_sampler = self._build_batch_sampler()
        self.result_class = self._get_result_class()

        # alias predict() to the __call__()
        self.predict = self.__call__

        self.batch_sampler.batch_size = batch_size
        self._pp_option = self._prepare_pp_option(pp_option, device)

        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")

    @property
    def config_path(self) -> str:
        """
        Get the path to the configuration file.

        Returns:
            str: The path to the configuration file.
        """
        return self.get_config_path(self.model_dir)

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            str: The model name.
        """
        return self.config["Global"]["model_name"]

    @property
    def pp_option(self) -> PaddlePredictorOption:
        if not hasattr(self, "_pp_option"):
            raise AttributeError(f"{repr(self)} has no attribute 'pp_option'.")
        return self._pp_option

    def __call__(
        self,
        input: Any,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Predict with the input data.

        Args:
            input (Any): The input data to be predicted.
            batch_size (int, optional): The batch size to use. Defaults to None.
            **kwargs (Dict[str, Any]): Additional keyword arguments to set up predictor.

        Returns:
            Iterator[Any]: An iterator yielding the prediction output.
        """
        self.set_predictor(batch_size)
        yield from self.apply(input, **kwargs)

    def set_predictor(
        self,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Sets the predictor configuration.

        Args:
            batch_size (Optional[int], optional): The batch size to use. Defaults to None.

        Returns:
            None
        """
        if batch_size:
            self.batch_sampler.batch_size = batch_size

    def create_static_infer(self):
        return ONNXRuntimeInfer(
            self.model_name, self.model_dir, self.MODEL_FILE_PREFIX, self._pp_option
        )

    def apply(self, input: Any, **kwargs) -> Iterator[Any]:
        """
        Do predicting with the input data and yields predictions.

        Args:
            input (Any): The input data to be predicted.

        Yields:
            Iterator[Any]: An iterator yielding prediction results.
        """

        batches = self.batch_sampler(input)
        for batch_data in batches:
            prediction = self.process(batch_data, **kwargs)
            prediction = PredictionWrap(prediction, len(batch_data))
            for idx in range(len(batch_data)):
                yield self.result_class(prediction.get_by_idx(idx))

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        """process the batch data sampled from BatchSampler and return the prediction result.

        Args:
            batch_data (List[Any]): The batch data sampled from BatchSampler.

        Returns:
            Dict[str, List[Any]]: The prediction result.
        """
        raise NotImplementedError

    @classmethod
    def get_config_path(cls, model_dir) -> str:
        """Get the path to the configuration file for the given model directory.

        Args:
            model_dir (Path): The directory where the static model files is stored.

        Returns:
            Path: The path to the configuration file.
        """
        return model_dir / f"{cls.MODEL_FILE_PREFIX}.yml"

    @classmethod
    def load_config(cls, model_dir) -> Dict:
        """Load the configuration from the specified model directory.

        Args:
            model_dir (Path): The where the static model files is stored.

        Returns:
            dict: The loaded configuration dictionary.
        """
        yaml_reader = YAMLReader()
        return yaml_reader.read(cls.get_config_path(model_dir))

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        """Build batch sampler.

        Returns:
            BaseBatchSampler: batch sampler object.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        """Get the result class.

        Returns:
            type: The result class.
        """
        raise NotImplementedError

    def _prepare_pp_option(
        self,
        pp_option: Optional[PaddlePredictorOption],
        device: Optional[str],
    ) -> PaddlePredictorOption:
        if pp_option is None or device is not None:
            device_info = self._get_device_info(device)
        else:
            device_info = None
        if pp_option is None:
            pp_option = PaddlePredictorOption()
        if device_info:
            pp_option.device_type = device_info[0]
            pp_option.device_id = device_info[1]
        pp_option.setdefault_by_model_name(model_name=self.model_name)
        
        return pp_option

    # Should this be static?
    def _get_device_info(self, device):
        if device is None:
            device = get_default_device()
        device_type, device_ids = parse_device(device)
        if device_ids is not None:
            device_id = device_ids[0]
        else:
            device_id = None
        if device_ids and len(device_ids) > 1:
            logging.debug("Got multiple device IDs. Using the first one: %d", device_id)
        return device_type, device_id
