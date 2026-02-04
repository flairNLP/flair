import pytest
import torch

from flair.safetensors_utils import SafetensorsSerializer


@pytest.mark.parametrize(
    "state_dict",
    [
        {"weight": torch.randn(3, 3)},
        {"a": torch.randn(2), "b": {"nested": torch.randn(4)}},
        {"config": "value", "tensor": torch.randn(5)},
    ],
)
def test_save_load_roundtrip(tmp_path, state_dict):
    """Verify state dict keys are preserved through save/load cycle."""
    SafetensorsSerializer.save(state_dict, tmp_path / "model")
    loaded = SafetensorsSerializer.load(tmp_path / "model")
    assert loaded.keys() == state_dict.keys()


@pytest.mark.parametrize(
    "state_dict",
    [
        {"weight": torch.randn(3, 3)},
        {"nested": {"deep": {"tensor": torch.randn(2)}}},
    ],
)
def test_tensors_preserved(tmp_path, state_dict):
    """Verify tensor values remain numerically identical after serialization."""
    SafetensorsSerializer.save(state_dict, tmp_path / "model")
    loaded = SafetensorsSerializer.load(tmp_path / "model")
    original_tensor = list(state_dict.values())[0]
    loaded_tensor = list(loaded.values())[0]
    while isinstance(original_tensor, dict):
        original_tensor = list(original_tensor.values())[0]
        loaded_tensor = list(loaded_tensor.values())[0]
    assert torch.allclose(original_tensor, loaded_tensor)


def test_is_safetensors_model(tmp_path):
    """Verify safetensors format detection distinguishes valid models from missing paths."""
    assert not SafetensorsSerializer.is_safetensors_model(tmp_path / "nonexistent")
    SafetensorsSerializer.save({"w": torch.randn(2)}, tmp_path / "model")
    assert SafetensorsSerializer.is_safetensors_model(tmp_path / "model")
