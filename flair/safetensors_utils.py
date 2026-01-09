import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if _is_tensor(value):
            tensors[new_key] = value
            metadata[key] = {"__tensor_key__": new_key}
        elif isinstance(value, dict):
            nested_tensors, nested_metadata = _flatten_dict(value, new_key, sep)
            tensors.update(nested_tensors)
            metadata[key] = nested_metadata
        elif isinstance(value, list):
            processed_list, list_tensors = _process_list(value, new_key, sep)
            tensors.update(list_tensors)
            metadata[key] = processed_list
        else:
            metadata[key] = value

    return tensors, metadata


def _process_list(
    lst: list, parent_key: str, sep: str
) -> tuple[list, dict[str, torch.Tensor]]:
    tensors: dict[str, torch.Tensor] = {}
    processed: list = []

    for i, item in enumerate(lst):
        item_key = f"{parent_key}{sep}{i}"

        if _is_tensor(item):
            tensors[item_key] = item
            processed.append({"__tensor_key__": item_key})
        elif isinstance(item, dict):
            nested_tensors, nested_metadata = _flatten_dict(item, item_key, sep)
            tensors.update(nested_tensors)
            processed.append(nested_metadata)
        elif isinstance(item, list):
            nested_list, nested_tensors = _process_list(item, item_key, sep)
            tensors.update(nested_tensors)
            processed.append(nested_list)
        else:
            processed.append(item)

    return processed, tensors


def _unflatten_dict(
    metadata: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in metadata.items():
        if isinstance(value, dict):
            if "__tensor_key__" in value:
                tensor_key = value["__tensor_key__"]
                result[key] = tensors[tensor_key]
            else:
                result[key] = _unflatten_dict(value, tensors)
        elif isinstance(value, list):
            result[key] = _unflatten_list(value, tensors)
        else:
            result[key] = value

    return result


def _unflatten_list(lst: list, tensors: dict[str, torch.Tensor]) -> list:
    result: list = []

    for item in lst:
        if isinstance(item, dict):
            if "__tensor_key__" in item:
                tensor_key = item["__tensor_key__"]
                result.append(tensors[tensor_key])
            else:
                result.append(_unflatten_dict(item, tensors))
        elif isinstance(item, list):
            result.append(_unflatten_list(item, tensors))
        else:
            result.append(item)

    return result


def separate_tensors_and_metadata(
    state_dict: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return _flatten_dict(state_dict)


def combine_tensors_and_metadata(
    tensors: dict[str, torch.Tensor], metadata: dict[str, Any]
) -> dict[str, Any]:
    return _unflatten_dict(metadata, tensors)


def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, BytesIO):
        return {"__bytesio__": base64.b64encode(obj.getvalue()).decode("ascii")}
    if isinstance(obj, torch.dtype):
        return {"__torch_dtype__": str(obj)}
    if isinstance(obj, torch.device):
        return {"__torch_device__": str(obj)}
    if isinstance(obj, type):
        return {"__class__": f"{obj.__module__}.{obj.__name__}"}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_deserializer(obj: dict) -> Any:
    if "__bytes__" in obj:
        return base64.b64decode(obj["__bytes__"])
    if "__bytesio__" in obj:
        return BytesIO(base64.b64decode(obj["__bytesio__"]))
    if "__torch_dtype__" in obj:
        dtype_str = obj["__torch_dtype__"]
        dtype_map = {
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.int16": torch.int16,
            "torch.int8": torch.int8,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
        }
        return dtype_map.get(dtype_str, torch.float32)
    if "__torch_device__" in obj:
        return torch.device(obj["__torch_device__"])
    return obj


class SafetensorsSerializer:
    TENSORS_FILENAME = "model.safetensors"
    METADATA_FILENAME = "model_metadata.json"

    @classmethod
    def save(cls, state_dict: dict[str, Any], path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        tensors, metadata = separate_tensors_and_metadata(state_dict)

        tensors_path = path / cls.TENSORS_FILENAME
        if tensors:
            safetensors_save_file(tensors, str(tensors_path))
        else:
            safetensors_save_file({}, str(tensors_path))

        metadata_path = path / cls.METADATA_FILENAME
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=_json_serializer)

    @classmethod
    def load(cls, path: Union[str, Path]) -> dict[str, Any]:
        path = Path(path)

        tensors_path = path / cls.TENSORS_FILENAME
        tensors = safetensors_load_file(str(tensors_path))

        metadata_path = path / cls.METADATA_FILENAME
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f, object_hook=_json_deserializer)

        return combine_tensors_and_metadata(tensors, metadata)

    @classmethod
    def is_safetensors_model(cls, path: Union[str, Path]) -> bool:
        path = Path(path)
        if path.is_dir():
            return (path / cls.TENSORS_FILENAME).exists() and (path / cls.METADATA_FILENAME).exists()
        return False