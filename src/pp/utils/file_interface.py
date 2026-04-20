import logging
import os

import chardet
import ruamel.yaml
import yaml
from filelock import FileLock

try:
    import ujson as json
except:
    logging.warning("failed to import ujson, using json instead")
    import json

from contextlib import contextmanager


@contextmanager
def custom_open(file_path, mode):
    """
    自定义打开文件函数

    Args:
        file_path (str): 文件路径
        mode (str): 文件打开模式，'r'，'w' 或 'a'

    Returns:
        Any: 返回文件对象

    Raises:
        FileNotFoundError: 当文件不存在时，raise FileNotFoundError
        ValueError: 当 mode 参数不是 'r'， 'w' 和 'a' 时，raise ValueError
    """
    if mode == "r":
        if not os.path.exists(file_path):
            raise FileNotFoundError("file {} not found".format(file_path))
        file = open(file_path, "r", encoding="utf-8")
        try:
            file.read()
            file.seek(0)
            yield file
        except UnicodeDecodeError:
            file = open(file_path, "r", encoding="gbk")
            try:
                file.read()
                file.seek(0)
                yield file
            except UnicodeDecodeError:
                with open(file_path, "rb") as f:
                    encoding = chardet.detect(f.read())["encoding"]
                file = open(file_path, "r", encoding=encoding)
                yield file
        finally:
            file.close()
    elif mode == "w":
        file = open(file_path, "w", encoding="utf-8")
        yield file
        file.close()
    elif mode == "a":
        encoding = "utf-8"
        if os.path.exists(file_path):
            file = open(file_path, "r", encoding=encoding)
            try:
                file.read()
                file.seek(0)
            except UnicodeDecodeError:
                encoding = "gbk"
                file = open(file_path, "r", encoding=encoding)
                try:
                    file.read()
                    file.seek(0)
                except UnicodeDecodeError:
                    with open(file_path, "rb") as f:
                        encoding = chardet.detect(f.read())["encoding"]
            finally:
                file.close()

        file = open(file_path, "a", encoding=encoding)
        yield file
        file.close()
    else:
        raise ValueError("mode must be 'r', 'w' or 'a', but got {}".format(mode))
