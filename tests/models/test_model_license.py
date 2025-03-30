from flair.nn import Model


def test_model_license_persistence(tmp_path):
    """Test setting and persisting license information for a model."""
    # Create temporary file path using pytest's tmp_path fixture
    model_path = tmp_path / "test_model_license.pt"

    # Load a base model
    model = Model.load("ner-fast")

    # Check initial license (should be none/default)
    assert model.license_info == "No license information available"

    # Set a new license
    test_license = "MIT License - Copyright (c) 2024"
    model.license_info = test_license
    assert model.license_info == test_license

    # Save the model with the new license
    model.save(str(model_path))

    # Load the saved model and check license persists
    loaded_model = Model.load(model_path)
    assert loaded_model.license_info == test_license
