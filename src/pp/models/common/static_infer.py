import abc
import subprocess
from os import PathLike
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import onnxruntime as ort

from src.pp.utils import logging
from src.pp.utils.model_paths import get_model_paths
from src.pp.utils.pp_option import PaddlePredictorOption


CACHE_DIR = ".cache"

def _sort_inputs(inputs, names):
    # NOTE: Adjust input tensors to match the sorted sequence.
    indices = sorted(range(len(names)), key=names.__getitem__)
    inputs = [inputs[indices.index(i)] for i in range(len(inputs))]
    return inputs


class StaticInfer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError


class ONNXRuntimeInfer(StaticInfer):
    """ONNX Runtime-based inference to replace ONNXRuntimeInfer"""
    
    def __init__(
        self,
        model_name: str,
        model_dir: Union[str, PathLike],
        model_file_prefix: str,
        option: PaddlePredictorOption,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_file_prefix = model_file_prefix
        self._option = option
        self._session = None
        self._use_paddle_fallback = False

    @property
    def session(self):
        if self._session is None:
            self._session = self._create_onnx_session()
        return self._session
        
    def _create_onnx_session(self):
        """Create ONNX Runtime session"""
        model_paths = get_model_paths(self.model_dir, self.model_file_prefix)
        
        # First try to find ONNX model
        onnx_model_path = None
        if "onnx" in model_paths:
            onnx_path = model_paths["onnx"]
            if isinstance(onnx_path, dict) and "model" in onnx_path:
                onnx_model_path = onnx_path["model"]
            elif isinstance(onnx_path, (str, Path)):
                onnx_model_path = str(onnx_path)
        
        if onnx_model_path is None:
            # Look for inference.onnx directly
            inference_onnx = self.model_dir / "inference.onnx"
            if inference_onnx.exists():
                onnx_model_path = str(inference_onnx)
        
        if onnx_model_path is None:
            # Fall back to PaddlePaddle if no ONNX model found
            logging.warning(f"No ONNX model found in {self.model_dir}, falling back to PaddlePaddle inference")
            # Delegate to original ONNXRuntimeInfer implementation
            self._use_paddle_fallback = True
            return None
        
        self._use_paddle_fallback = False
            
        logging.info(f"Loading ONNX model: {onnx_model_path}")
        
        # Set up ONNX Runtime options
        providers = []
        if self._option.device_type == "gpu":
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        sess_options = ort.SessionOptions()
        if self._option.device_type == "cpu":
            # Use cpu_threads from PaddlePredictorOption
            cpu_threads = getattr(self._option, 'cpu_threads', 10)
            sess_options.intra_op_num_threads = cpu_threads
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(
            onnx_model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        logging.info(f"ONNX Runtime session created successfully with providers: {session.get_providers()}")
        return session

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Run inference using ONNX Runtime or PaddlePaddle fallback"""
        
        input_names = [input.name for input in self.session.get_inputs()]
        
        if len(input_names) != len(x):
            raise ValueError(
                f"The number of inputs does not match the model: {len(input_names)} vs {len(x)}"
            )
            
        # Sort inputs to match model input order
        x = _sort_inputs(x, input_names)
        
        # Prepare input dictionary
        input_dict = {}
        for name, input_array in zip(input_names, x):
            input_dict[name] = np.ascontiguousarray(input_array)
            
        # Run inference
        outputs = self.session.run(None, input_dict)
        return outputs
        

