import os
from contextlib import ContextDecorator

from . import logging


SUPPORTED_DEVICE_TYPE = ["cpu", "gpu", "xpu", "npu", "mlu", "gcu", "dcu"]


def set_env_for_device_type(device_type):
    """Set environment variables for specific device type"""
    # Stub function - environment variables can be set here if needed
    pass


def get_default_device():
    """Get default device - returns 'cpu' by default"""
    return "cpu"

def constr_device(device_type, device_ids):
    if device_type == "cpu" and device_ids is not None:
        raise ValueError("`device_ids` must be None for CPUs")
    if device_ids:
        device_ids = ",".join(map(str, device_ids))
        return f"{device_type}:{device_ids}"
    else:
        return f"{device_type}"

def parse_device(device):
    """parse_device"""
    # According to https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html
    parts = device.split(":")
    if len(parts) > 2:
        raise ValueError(f"Invalid device: {device}")
    if len(parts) == 1:
        device_type, device_ids = parts[0], None
    else:
        device_type, device_ids = parts
        device_ids = device_ids.split(",")
        for device_id in device_ids:
            if not device_id.isdigit():
                raise ValueError(
                    f"Device ID must be an integer. Invalid device ID: {device_id}"
                )
        device_ids = list(map(int, device_ids))
    device_type = device_type.lower()
    # raise_unsupported_device_error(device_type, SUPPORTED_DEVICE_TYPE)
    assert device_type.lower() in SUPPORTED_DEVICE_TYPE
    if device_type == "cpu" and device_ids is not None:
        raise ValueError("No Device ID should be specified for CPUs")
    return device_type, device_ids


