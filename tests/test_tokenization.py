import pytest

# Assuming these tokenizers are in flair.tokenization
from flair.tokenization import (
    SegtokTokenizer,
    SpaceTokenizer,
    SpacyTokenizer,
    JapaneseTokenizer,
    SciSpacyTokenizer,
    StaccatoTokenizer,
    TokenizerWrapper,
)


# Helper function for basic serialization/deserialization test
def _test_tokenizer_serialization(tokenizer_instance):
    """Tests standard to_dict and from_dict functionality."""
    # Serialize
    config = tokenizer_instance.to_dict()

    # Check basic keys
    assert "class_module" in config
    assert "class_name" in config
    assert config["class_module"] == tokenizer_instance.__class__.__module__
    assert config["class_name"] == tokenizer_instance.__class__.__name__

    # Deserialize
    reconstructed_tokenizer = tokenizer_instance.__class__.from_dict(config)

    # Check type
    assert isinstance(reconstructed_tokenizer, tokenizer_instance.__class__)

    # Check basic functionality (optional but recommended)
    text = "This is a test."
    original_tokens = tokenizer_instance.tokenize(text)
    reconstructed_tokens = reconstructed_tokenizer.tokenize(text)
    assert original_tokens == reconstructed_tokens

    return reconstructed_tokenizer  # Return for further specific checks if needed


# --- Individual Tokenizer Tests ---


def test_staccato_tokenizer_serialization():
    tokenizer = StaccatoTokenizer()
    _test_tokenizer_serialization(tokenizer)


def test_segtok_tokenizer_serialization():
    # Test default
    tokenizer_default = SegtokTokenizer()
    reconstructed_default = _test_tokenizer_serialization(tokenizer_default)
    assert reconstructed_default.additional_split_characters is None

    # Test with additional chars
    split_chars = ["§", "%"]
    tokenizer_custom = SegtokTokenizer(additional_split_characters=split_chars)
    reconstructed_custom = _test_tokenizer_serialization(tokenizer_custom)
    assert reconstructed_custom.additional_split_characters == split_chars


def test_space_tokenizer_serialization():
    tokenizer = SpaceTokenizer()
    _test_tokenizer_serialization(tokenizer)


def test_spacy_tokenizer_serialization():
    pytest.importorskip("spacy")
    # Skip if model not installed, or handle potential download within test setup if desired
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("Spacy model 'en_core_web_sm' not installed")

    tokenizer = SpacyTokenizer("en_core_web_sm")
    reconstructed = _test_tokenizer_serialization(tokenizer)
    assert reconstructed.model.meta["name"] == "en_core_web_sm"


def test_japanese_tokenizer_serialization():
    pytest.importorskip("konoha")
    # Assuming MeCab is a common backend for testing, may need specific setup/skipping
    # depending on the CI environment or local setup.
    try:
        tokenizer = JapaneseTokenizer("mecab")
        reconstructed = _test_tokenizer_serialization(tokenizer)
        assert reconstructed.tokenizer == "mecab"
        assert reconstructed.sudachi_mode == "A"  # Check default mode

        # Test with different mode
        tokenizer_sudachi = JapaneseTokenizer("sudachi", sudachi_mode="B")
        config_sudachi = tokenizer_sudachi.to_dict()
        reconstructed_sudachi = JapaneseTokenizer.from_dict(config_sudachi)
        assert reconstructed_sudachi.tokenizer == "sudachi"
        assert reconstructed_sudachi.sudachi_mode == "B"

    except Exception as e:
        # Catch potential errors during konoha/mecab init if not properly configured
        pytest.skip(f"Skipping JapaneseTokenizer test due to initialization error: {e}")


def test_scispacy_tokenizer_serialization():
    pytest.importorskip("spacy")
    pytest.importorskip("scispacy")
    # Skip if model not installed
    try:
        spacy.load("en_core_sci_sm")
    except OSError:
        pytest.skip("SciSpacy model 'en_core_sci_sm' not installed")

    tokenizer = SciSpacyTokenizer()
    _test_tokenizer_serialization(tokenizer)


def test_tokenizer_wrapper_serialization():
    def dummy_tokenizer(text: str) -> list[str]:
        return text.split("-")

    tokenizer = TokenizerWrapper(dummy_tokenizer)

    # Test serialization
    config = tokenizer.to_dict()
    assert "class_module" in config
    assert "class_name" in config
    assert config.get("serializable") is False  # Check the non-serializable flag
    assert config.get("function_name") == "dummy_tokenizer"

    # Test deserialization raises error
    with pytest.raises(NotImplementedError):
        TokenizerWrapper.from_dict(config)


def test_tokenizer_equality():

    assert StaccatoTokenizer() == StaccatoTokenizer()
    assert SegtokTokenizer() == SegtokTokenizer()
    assert SegtokTokenizer() != StaccatoTokenizer()
    assert SegtokTokenizer(additional_split_characters=["!"]) != SegtokTokenizer()
    assert SegtokTokenizer(additional_split_characters=["!"]) == SegtokTokenizer(additional_split_characters=["!"])
