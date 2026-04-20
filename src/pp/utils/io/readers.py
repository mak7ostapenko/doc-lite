


import enum


import numpy as np
import pandas as pd
import yaml


import cv2

__all__ = [
    "ReaderType",
    "ImageReader",
    "CSVReader",
    "YAMLReader",
]


class ReaderType(enum.Enum):
    """ReaderType"""

    IMAGE = 1
    GENERATIVE = 2
    JSON = 4
    YAML = 8
    MARKDOWN = 9
    TXT = 10


class _BaseReader(object):
    """_BaseReader"""

    def read(self, in_path):
        """read file from path"""
        raise NotImplementedError

    def get_type(self):
        """get type"""
        raise NotImplementedError


class ImageReader(_BaseReader):
    """ImageReader - Direct OpenCV implementation (no backend abstraction)"""

    def __init__(self, flags=None):
        """
        Initialize ImageReader with OpenCV.

        Args:
            flags: OpenCV imread flags (default: cv2.IMREAD_COLOR)
        """
        super().__init__()
        if flags is None:
            flags = cv2.IMREAD_COLOR
        self.flags = flags

    def read(self, in_path):
        """Read image file using OpenCV"""
        return cv2.imread(str(in_path), flags=self.flags)

    def get_type(self):
        """get type"""
        return ReaderType.IMAGE


class _GenerativeReader(_BaseReader):
    """_GenerativeReader"""

    def get_type(self):
        """get type"""
        return ReaderType.GENERATIVE


def is_generative_reader(reader):
    """is_generative_reader"""
    return isinstance(reader, _GenerativeReader)



class YAMLReader(_BaseReader):

    def __init__(self):
        super().__init__()

    def read(self, in_path):
        with open(in_path, "r", encoding="utf-8") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return data

    def get_type(self):
        return ReaderType.YAML


class MarkDownReader(_BaseReader):

    def __init__(self):
        super().__init__()

    def read(self, in_path):
        with open(in_path, "r") as f:
            data = f.read()
        return data

    def get_type(self):
        return ReaderType.MARKDOWN


class TXTReader(_BaseReader):
    """TXTReader"""

    def __init__(self):
        super().__init__()

    def read(self, in_path):
        with open(in_path, "r") as f:
            data = f.read()
        return data

    def get_type(self):
        return ReaderType.TXT


class CSVReader(_BaseReader):
    """CSVReader"""

    def __init__(self):
        super().__init__()

    def read(self, in_path):
        """read CSV file"""
        return pd.read_csv(str(in_path))

    def get_type(self):
        """get type"""
        return ReaderType.TS
