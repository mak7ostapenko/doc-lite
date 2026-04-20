from functools import wraps

from src.pp.utils import logging


class FuncRegister(object):
    def __init__(self, register_map):
        assert isinstance(register_map, dict)
        self._register_map = register_map

    def __call__(self, key=None):
        """register the decoratored func as key in dict"""

        def decorator(func):
            actual_key = key if key is not None else func.__name__
            self._register_map[actual_key] = func
            logging.debug(
                f"The func ({func.__name__}) has been registered as key ({actual_key})."
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator
