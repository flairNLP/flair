import pytest
import torch

from flair.embeddings import (
    WordEmbeddings,
    TokenEmbeddings,
    StackedEmbeddings,
    DocumentPoolEmbeddings,
    FlairEmbeddings,
    DocumentRNNEmbeddings,
    DocumentLMEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings,
)

from flair.data import Sentence, Dictionary
from flair.models import LanguageModel

glove: TokenEmbeddings = WordEmbeddings("turian")
flair_embedding: TokenEmbeddings = FlairEmbeddings("news-forward-fast")


def test_load_non_existing_embedding():
    with pytest.raises(ValueError):
        WordEmbeddings("other")

    with pytest.raises(ValueError):
        WordEmbeddings("not/existing/path/to/embeddings")


def test_load_non_existing_flair_embedding():
    with pytest.raises(ValueError):
        FlairEmbeddings("other")


def test_keep_batch_order():

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


def test_stacked_embeddings():

    embeddings: StackedEmbeddings = StackedEmbeddings([glove, flair_embedding])

    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 1074

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings


def test_transformer_word_embeddings():

    embeddings = TransformerWordEmbeddings('distilbert-base-uncased')

    sentence: Sentence = Sentence("I love Berlin")
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 3072

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0

    embeddings = TransformerWordEmbeddings('distilbert-base-uncased', layers='all')

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 5376

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings

    embeddings = TransformerWordEmbeddings('distilbert-base-uncased', layers='all', use_scalar_mix=True)

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 768

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings


def test_transformer_weird_sentences():

    embeddings = TransformerWordEmbeddings('distilbert-base-uncased', layers='all', use_scalar_mix=True)

    sentence = Sentence("Hybrid mesons , qq ̄ states with an admixture")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("typical proportionalities of ∼ 1nmV − 1 [ 3,4 ] .")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟 🤟  🤟 hüllo")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟hallo 🤟 🤟 🤟 🤟")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟hallo 🤟 🤟 🤟 🤟")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟")
    sentence_2 = Sentence("second sentence")
    embeddings.embed([sentence, sentence_2])
    for token in sentence:
        assert len(token.get_embedding()) == 768
    for token in sentence_2:
        assert len(token.get_embedding()) == 768

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


def test_document_lstm_embeddings():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [glove, flair_embedding], hidden_size=128, bidirectional=False
    )

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 128
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


def test_document_bidirectional_lstm_embeddings():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [glove, flair_embedding], hidden_size=128, bidirectional=True
    )

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 512
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


def test_document_pool_embeddings():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    for mode in ["mean", "max", "min"]:
        embeddings: DocumentPoolEmbeddings = DocumentPoolEmbeddings(
            [glove, flair_embedding], pooling=mode, fine_tune_mode="none"
        )

        embeddings.embed(sentence)

        assert len(sentence.get_embedding()) == 1074

        sentence.clear_embeddings()

        assert len(sentence.get_embedding()) == 0
        del embeddings


def test_document_pool_embeddings_nonlinear():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    for mode in ["mean", "max", "min"]:
        embeddings: DocumentPoolEmbeddings = DocumentPoolEmbeddings(
            [glove, flair_embedding], pooling=mode, fine_tune_mode="nonlinear"
        )

        embeddings.embed(sentence)

        assert len(sentence.get_embedding()) == 1074

        sentence.clear_embeddings()

        assert len(sentence.get_embedding()) == 0
        del embeddings


def test_transformer_document_embeddings():

    embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased')

    sentence: Sentence = Sentence("I love Berlin")
    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 768

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0

    embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', layers='all')

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 5376

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0

    embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', layers='all', use_scalar_mix=True)

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 768

    sentence.clear_embeddings()

    del embeddings