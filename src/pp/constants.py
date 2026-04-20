import os

from typing import Final

DEFAULT_CACHE_DIR = os.path.abspath(os.path.join(os.path.expanduser("~"), ".paddlex"))
# Use project root directory for models
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LOCAL_FONT_FILE_PATH = None
MODEL_SOURCE = os.environ.get("PADDLE_PDX_MODEL_SOURCE", "huggingface").lower()

MODEL_FILE_PREFIX: Final[str] = "inference"
