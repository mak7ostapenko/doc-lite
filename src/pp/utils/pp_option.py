"""
PaddleInfer predictor options for ONNX Runtime inference.

This module configures the PaddleInfer wrapper around ONNX Runtime backend.
All models run using ONNX Runtime for inference.
"""

import os
from copy import deepcopy
from typing import Dict, List

from src.pp.utils import logging
from src.pp.utils.device import get_default_device, parse_device, set_env_for_device_type


def get_default_run_mode(model_name=None, device_type=None):
    """Get default run mode - always returns 'paddle' (PaddleInfer wrapper for ONNX)"""
    return "paddle"

# Models that require legacy IR (PIR disabled)
NEWIR_BLOCKLIST = [
    "PP-YOLOE_seg-S",
    "PatchTST_ad",
    "Nonstationary_ad",
    "DLinear_ad",
    "Co-Deformable-DETR-R50",
    "Co-Deformable-DETR-Swin-T",
    "Co-DINO-R50",
    "Co-DINO-Swin-L",
    "LaTeX_OCR_rec",
    "BEVFusion",
    "GroundingDINO-T",
]


class PaddlePredictorOption(object):
    """PaddleInfer configuration for ONNX Runtime backend"""

    SUPPORT_RUN_MODE = (
        "paddle",        # PaddleInfer wrapper for ONNX Runtime
        "paddle_fp32",   # FP32 precision
        "paddle_fp16",   # FP16 precision
    )
    SUPPORT_DEVICE = ("gpu", "cpu")

    def __init__(self, **kwargs):
        super().__init__()
        self._cfg = {}
        self._init_option(**kwargs)

    def copy(self):
        obj = type(self)()
        obj._cfg = deepcopy(self._cfg)
        if hasattr(self, "trt_cfg_setting"):
            obj.trt_cfg_setting = self.trt_cfg_setting
        return obj

    def _init_option(self, **kwargs):
        for k, v in kwargs.items():
            if self._has_setter(k):
                setattr(self, k, v)
            else:
                raise Exception(
                    f"{k} is not supported to set! The supported option is: {self._get_settable_attributes()}"
                )

    def setdefault_by_model_name(self, model_name):
        for k, v in self._get_default_config(model_name).items():
            self._cfg.setdefault(k, v)

    def _get_default_config(self, model_name):
        """Get default ONNX Runtime inference configuration"""
        if self.device_type is None:
            device_type, device_ids = parse_device(get_default_device())
            device_id = None if device_ids is None else device_ids[0]
        else:
            device_type, device_id = self.device_type, self.device_id

        default_config = {
            "run_mode": get_default_run_mode(),
            "device_type": device_type,
            "device_id": device_id,
            "cpu_threads": 10,
            "delete_pass": [],
            "enable_new_ir": model_name not in NEWIR_BLOCKLIST,
            "mkldnn_cache_capacity": 10,
        }
        return default_config

    def _update(self, k, v):
        self._cfg[k] = v

    @property
    def run_mode(self):
        return self._cfg.get("run_mode")

    @run_mode.setter
    def run_mode(self, run_mode: str):
        """set run mode"""
        if run_mode not in self.SUPPORT_RUN_MODE:
            support_run_mode_str = ", ".join(self.SUPPORT_RUN_MODE)
            raise ValueError(
                f"`run_mode` must be {support_run_mode_str}, but received {repr(run_mode)}."
            )

        self._update("run_mode", run_mode)

    @property
    def device_type(self):
        return self._cfg.get("device_type")

    @device_type.setter
    def device_type(self, device_type):
        if device_type not in self.SUPPORT_DEVICE:
            support_run_mode_str = ", ".join(self.SUPPORT_DEVICE)
            raise ValueError(
                f"The device type must be one of {support_run_mode_str}, but received {repr(device_type)}."
            )
        self._update("device_type", device_type)
        set_env_for_device_type(device_type)
        # XXX(gaotingquan): set flag to accelerate inference in paddle 3.0b2
        if device_type in ("gpu", "cpu"):
            os.environ["FLAGS_enable_pir_api"] = "1"

    @property
    def device_id(self):
        return self._cfg.get("device_id")

    @device_id.setter
    def device_id(self, device_id):
        self._update("device_id", device_id)

    @property
    def cpu_threads(self):
        return self._cfg.get("cpu_threads")

    @cpu_threads.setter
    def cpu_threads(self, cpu_threads):
        """set cpu threads"""
        if not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise Exception()
        self._update("cpu_threads", cpu_threads)

    @property
    def delete_pass(self):
        return self._cfg.get("delete_pass")

    @delete_pass.setter
    def delete_pass(self, delete_pass):
        self._update("delete_pass", delete_pass)

    @property
    def enable_new_ir(self):
        return self._cfg.get("enable_new_ir")

    @enable_new_ir.setter
    def enable_new_ir(self, enable_new_ir: bool):
        """Enable/disable new IR (PIR) - some models require legacy IR"""
        self._update("enable_new_ir", enable_new_ir)

    def set_device(self, device: str):
        """set device"""
        if not device:
            return
        device_type, device_ids = parse_device(device)
        self.device_type = device_type
        device_id = device_ids[0] if device_ids is not None else None
        self.device_id = device_id
        if device_ids is None or len(device_ids) > 1:
            logging.debug(f"The device ID has been set to {device_id}.")

    def get_support_run_mode(self):
        """get supported run mode"""
        return self.SUPPORT_RUN_MODE

    def get_support_device(self):
        """get supported device"""
        return self.SUPPORT_DEVICE

    def __str__(self):
        return ",  ".join([f"{k}: {v}" for k, v in self._cfg.items()])

    def __getattr__(self, key):
        if key not in self._cfg:
            raise Exception(f"The key ({key}) is not found in cfg: \n {self._cfg}")
        return self._cfg.get(key)

    def __eq__(self, obj):
        if isinstance(obj, PaddlePredictorOption):
            return obj._cfg == self._cfg
        return False

    def _has_setter(self, attr):
        prop = getattr(self.__class__, attr, None)
        return isinstance(prop, property) and prop.fset is not None

    def _get_settable_attributes(self):
        return [
            name
            for name, prop in vars(self.__class__).items()
            if isinstance(prop, property) and prop.fset is not None
        ]
