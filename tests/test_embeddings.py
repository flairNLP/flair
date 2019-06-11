import pytest

from flair.embeddings import (
    WordEmbeddings,
    TokenEmbeddings,
    StackedEmbeddings,
    DocumentPoolEmbeddings,
    FlairEmbeddings,
    DocumentRNNEmbeddings,
)

from flair.data import Sentence


def test_loading_not_existing_embedding():
    with pytest.raises(ValueError):
        WordEmbeddings("other")

    with pytest.raises(ValueError):
        WordEmbeddings("not/existing/path/to/embeddings")


def test_loading_not_existing_char_lm_embedding():
    with pytest.raises(ValueError):
        FlairEmbeddings("other")


@pytest.mark.integration
def test_stacked_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: StackedEmbeddings = StackedEmbeddings([glove, charlm])

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 1074

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0


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


def load_and_apply_char_lm_embeddings(emb_type: str):
    text = "I love Berlin."
    sentence: Sentence = Sentence(text)
    embeddings: TokenEmbeddings = FlairEmbeddings(emb_type)
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) != 0

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
