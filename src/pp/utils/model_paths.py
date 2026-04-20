from os import PathLike
from pathlib import Path
from typing import Tuple, TypedDict, Union

from src.pp.constants import MODEL_FILE_PREFIX


class ModelPaths(TypedDict, total=False):
    paddle: Tuple[Path, Path]
    onnx: Path
    om: Path


def get_model_paths(
    model_dir: Union[str, PathLike],
    model_file_prefix: str = MODEL_FILE_PREFIX,
) -> ModelPaths:
    model_dir = Path(model_dir)
    model_paths: ModelPaths = {}
    pd_model_path = None
    if (model_dir / f"{model_file_prefix}.json").exists():
        pd_model_path = model_dir / f"{model_file_prefix}.json"
    elif (model_dir / f"{model_file_prefix}.pdmodel").exists():
        pd_model_path = model_dir / f"{model_file_prefix}.pdmodel"
    if pd_model_path and (model_dir / f"{model_file_prefix}.pdiparams").exists():
        model_paths["paddle"] = (
            pd_model_path,
            model_dir / f"{model_file_prefix}.pdiparams",
        )
    if (model_dir / f"{model_file_prefix}.onnx").exists():
        model_paths["onnx"] = model_dir / f"{model_file_prefix}.onnx"
    if (model_dir / f"{model_file_prefix}.om").exists():
        model_paths["om"] = model_dir / f"{model_file_prefix}.om"
    return model_paths
