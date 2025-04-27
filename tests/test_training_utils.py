import torch

from flair.data import Token, Sentence, DataPair, Span, DataTriple, Image
from flair.training_utils import identify_dynamic_embeddings

# Helper tensors
dynamic_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
static_tensor = torch.tensor([3.0, 4.0], requires_grad=False)


def test_identify_dynamic_embeddings_empty_list():
    """Test with an empty list of data points."""
    assert identify_dynamic_embeddings([]) is None


def test_identify_dynamic_embeddings_no_embeddings():
    """Test with data points that have no embeddings at all."""
    token = Token("hello")
    sentence = Sentence("world")
    pair = DataPair(Token("a"), Token("b"))
    assert identify_dynamic_embeddings([token]) is None
    assert identify_dynamic_embeddings([sentence]) is None  # Sentence itself has no embedding yet
    assert identify_dynamic_embeddings([pair]) is None
    assert identify_dynamic_embeddings([token, sentence, pair]) is None


def test_identify_dynamic_embeddings_only_static():
    """Test with data points having only static embeddings."""
    token = Token("hello")
    token.set_embedding("static_tok", static_tensor.clone())

    sentence = Sentence("world .")  # Creates tokens
    sentence.set_embedding("static_sent", static_tensor.clone())
    sentence.tokens[0].set_embedding("static_sent_tok", static_tensor.clone())

    pair_tok1 = Token("a")
    pair_tok1.set_embedding("static_pair_tok1", static_tensor.clone())
    pair_tok2 = Token("b")
    pair_tok2.set_embedding("static_pair_tok2", static_tensor.clone())
    pair = DataPair(pair_tok1, pair_tok2)
    pair.set_embedding("static_pair", static_tensor.clone())

    assert identify_dynamic_embeddings([token]) == []
    assert identify_dynamic_embeddings([sentence]) == []
    assert identify_dynamic_embeddings([pair]) == []
    assert identify_dynamic_embeddings([token, sentence, pair]) == []


def test_identify_dynamic_embeddings_token():
    """Test with a single Token having mixed embeddings."""
    token = Token("test")
    token.set_embedding("dynamic_1", dynamic_tensor.clone())
    token.set_embedding("static_1", static_tensor.clone())
    result = identify_dynamic_embeddings([token])
    assert isinstance(result, list)
    assert set(result) == {"dynamic_1"}


def test_identify_dynamic_embeddings_sentence_direct():
    """Test with a Sentence having direct mixed embeddings (no token embeddings)."""
    sentence = Sentence("test sentence")
    sentence.set_embedding("dynamic_sent", dynamic_tensor.clone())
    sentence.set_embedding("static_sent", static_tensor.clone())
    # Note: sentence.tokens exist but don't have embeddings set here
    result = identify_dynamic_embeddings([sentence])
    assert isinstance(result, list)
    assert set(result) == {"dynamic_sent"}


def test_identify_dynamic_embeddings_sentence_with_tokens():
    """Test with a Sentence and its Tokens having mixed embeddings."""
    sentence = Sentence("test sentence")  # Creates tokens
    sentence.set_embedding("dynamic_sent", dynamic_tensor.clone())
    sentence.set_embedding("static_sent", static_tensor.clone())
    sentence.tokens[0].set_embedding("dynamic_tok_0", dynamic_tensor.clone())
    sentence.tokens[0].set_embedding("static_tok_0", static_tensor.clone())
    sentence.tokens[1].set_embedding("static_tok_1", static_tensor.clone())

    result = identify_dynamic_embeddings([sentence])
    assert isinstance(result, list)
    assert set(result) == {"dynamic_sent", "dynamic_tok_0"}


def test_identify_dynamic_embeddings_span():
    """Test with a Span containing tokens with mixed embeddings."""
    sentence = Sentence("This is a span test")  # Creates tokens
    sentence.tokens[2].set_embedding("dynamic_tok_2", dynamic_tensor.clone())  # Token within span
    sentence.tokens[3].set_embedding("static_tok_3", static_tensor.clone())  # Token within span
    sentence.tokens[0].set_embedding("static_tok_0", static_tensor.clone())  # Token outside span

    span = sentence[2:4]  # Span over "a span"

    result = identify_dynamic_embeddings([span])  # Test span directly (depends on how user might use it)
    assert isinstance(result, list)
    # Should find dynamic embeddings on the span AND its constituent tokens
    assert set(result) == {"dynamic_tok_2"}

    # More typical use case: check the sentence containing the span
    result_sent = identify_dynamic_embeddings([sentence])
    assert isinstance(result_sent, list)
    assert set(result_sent) == {"dynamic_tok_2"}


def test_identify_dynamic_embeddings_datapair():
    """Test with a DataPair containing Tokens with mixed embeddings."""
    tok1 = Token("first")
    tok1.set_embedding("dynamic_tok1", dynamic_tensor.clone())
    tok1.set_embedding("static_tok1", static_tensor.clone())

    tok2 = Token("second")
    tok2.set_embedding("static_tok2", static_tensor.clone())

    pair = DataPair(tok1, tok2)
    pair.set_embedding("dynamic_pair", dynamic_tensor.clone())
    pair.set_embedding("static_pair", static_tensor.clone())

    result = identify_dynamic_embeddings([pair])
    assert isinstance(result, list)
    assert set(result) == {"dynamic_tok1", "dynamic_pair"}


def test_identify_dynamic_embeddings_datatriple():
    """Test with a DataTriple containing Tokens with mixed embeddings."""
    tok1 = Token("first")
    tok1.set_embedding("dynamic_tok1", dynamic_tensor.clone())

    tok2 = Token("second")
    tok2.set_embedding("static_tok2", static_tensor.clone())

    tok3 = Token("third")
    tok3.set_embedding("dynamic_tok3", dynamic_tensor.clone())

    triple = DataTriple(tok1, tok2, tok3)
    triple.set_embedding("dynamic_triple", dynamic_tensor.clone())
    triple.set_embedding("static_triple", static_tensor.clone())

    result = identify_dynamic_embeddings([triple])
    assert isinstance(result, list)
    assert set(result) == {"dynamic_tok1", "dynamic_tok3", "dynamic_triple"}


def test_identify_dynamic_embeddings_image():
    """Test with an Image data point."""
    image = Image()
    image.set_embedding("dynamic_img", dynamic_tensor.clone())
    image.set_embedding("static_img", static_tensor.clone())

    result = identify_dynamic_embeddings([image])
    assert isinstance(result, list)
    assert set(result) == {"dynamic_img"}


def test_identify_dynamic_embeddings_mixed_list():
    """Test with a list containing various data point types."""
    token = Token("just a token")
    token.set_embedding("dynamic_mixed_tok", dynamic_tensor.clone())

    sentence = Sentence("a sentence")
    sentence.set_embedding("dynamic_mixed_sent", dynamic_tensor.clone())
    sentence.tokens[0].set_embedding("dynamic_mixed_sent_tok", dynamic_tensor.clone())

    pair_tok1 = Token("pair_a")
    pair_tok1.set_embedding("dynamic_mixed_pair_tok", dynamic_tensor.clone())
    pair_tok2 = Token("pair_b")
    pair = DataPair(pair_tok1, pair_tok2)
    pair.set_embedding("dynamic_mixed_pair", dynamic_tensor.clone())

    image = Image()
    image.set_embedding("dynamic_mixed_img", dynamic_tensor.clone())

    # Add one with only static to ensure it doesn't get picked up
    static_token = Token("static only")
    static_token.set_embedding("static_mixed", static_tensor.clone())

    # Add one with no embeddings
    empty_token = Token("empty")

    data_points = [token, sentence, pair, image, static_token, empty_token]
    result = identify_dynamic_embeddings(data_points)

    expected = {
        "dynamic_mixed_tok",
        "dynamic_mixed_sent",
        "dynamic_mixed_sent_tok",
        "dynamic_mixed_pair_tok",
        "dynamic_mixed_pair",
        "dynamic_mixed_img",
    }

    assert isinstance(result, list)
    assert set(result) == expected
