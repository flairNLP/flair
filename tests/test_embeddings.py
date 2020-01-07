import pytest
import torch

from flair.embeddings import (
    WordEmbeddings,
    TokenEmbeddings,
    StackedEmbeddings,
    DocumentPoolEmbeddings,
    FlairEmbeddings,
    DocumentRNNEmbeddings,
    DocumentLMEmbeddings,
)

from flair.data import Sentence, Dictionary
from flair.models import LanguageModel


def test_loading_not_existing_embedding():
    with pytest.raises(ValueError):
        WordEmbeddings("other")

    with pytest.raises(ValueError):
        WordEmbeddings("not/existing/path/to/embeddings")


def test_loading_not_existing_char_lm_embedding():
    with pytest.raises(ValueError):
        FlairEmbeddings("other")


def test_keep_batch_order():
    sentence, glove, charlm = init_document_embeddings()
    embeddings = DocumentRNNEmbeddings([glove])
    sentences_1 = [Sentence("First sentence"), Sentence("This is second sentence")]
    sentences_2 = [Sentence("This is second sentence"), Sentence("First sentence")]

    embeddings.embed(sentences_1)
    embeddings.embed(sentences_2)

    assert sentences_1[0].to_original_text() == "First sentence"
    assert sentences_1[1].to_original_text() == "This is second sentence"

    assert torch.norm(sentences_1[0].embedding - sentences_2[1].embedding) == 0.0
    assert torch.norm(sentences_1[0].embedding - sentences_2[1].embedding) == 0.0
    del embeddings


@pytest.mark.integration
def test_stacked_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: StackedEmbeddings = StackedEmbeddings([glove, charlm])

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 1074

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings


@pytest.mark.integration
def test_fine_tunable_flair_embedding():
    language_model_forward = LanguageModel(
        Dictionary.load("chars"), is_forward_lm=True, hidden_size=32, nlayers=1
    )

    embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [FlairEmbeddings(language_model_forward, fine_tune=True)],
        hidden_size=128,
        bidirectional=False,
    )

    sentence: Sentence = Sentence("I love Berlin.")

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 128
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0

    embeddings: DocumentLMEmbeddings = DocumentLMEmbeddings(
        [FlairEmbeddings(language_model_forward, fine_tune=True)]
    )

    sentence: Sentence = Sentence("I love Berlin.")

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 32
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


@pytest.mark.integration
def test_document_lstm_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [glove, charlm], hidden_size=128, bidirectional=False
    )

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 128
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


@pytest.mark.integration
def test_document_bidirectional_lstm_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [glove, charlm], hidden_size=128, bidirectional=True
    )

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 512
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


@pytest.mark.integration
def test_document_pool_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    for mode in ["mean", "max", "min"]:
        embeddings: DocumentPoolEmbeddings = DocumentPoolEmbeddings(
            [glove, charlm], pooling=mode, fine_tune_mode="none"
        )

        embeddings.embed(sentence)

        assert len(sentence.get_embedding()) == 1074

        sentence.clear_embeddings()

        assert len(sentence.get_embedding()) == 0
        del embeddings


@pytest.mark.integration
def test_document_pool_embeddings_nonlinear():
    sentence, glove, charlm = init_document_embeddings()

    for mode in ["mean", "max", "min"]:
        embeddings: DocumentPoolEmbeddings = DocumentPoolEmbeddings(
            [glove, charlm], pooling=mode, fine_tune_mode="nonlinear"
        )

        embeddings.embed(sentence)

        assert len(sentence.get_embedding()) == 1074

        sentence.clear_embeddings()

        assert len(sentence.get_embedding()) == 0
        del embeddings


def init_document_embeddings():
    text = "I love Berlin. Berlin is a great place to live."
    sentence: Sentence = Sentence(text)

    glove: TokenEmbeddings = WordEmbeddings("turian")
    charlm: TokenEmbeddings = FlairEmbeddings("news-forward-fast")

    return sentence, glove, charlm


def load_and_apply_word_embeddings(emb_type: str):
    text = "I love Berlin."
    sentence: Sentence = Sentence(text)
    embeddings: TokenEmbeddings = WordEmbeddings(emb_type)
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) != 0

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings


def load_and_apply_char_lm_embeddings(emb_type: str):
    text = "I love Berlin."
    sentence: Sentence = Sentence(text)
    embeddings: TokenEmbeddings = FlairEmbeddings(emb_type)
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) != 0

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings
