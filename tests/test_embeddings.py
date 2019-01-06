import pytest
import os

from flair.embeddings import WordEmbeddings, TokenEmbeddings, CharLMEmbeddings, StackedEmbeddings, \
    DocumentLSTMEmbeddings, DocumentMeanEmbeddings, DocumentPoolEmbeddings

from flair.data import Sentence


@pytest.mark.slow
def test_glove():
    load_and_apply_word_embeddings('glove')


@pytest.mark.slow
def test_extvec():
    load_and_apply_word_embeddings('extvec')


@pytest.mark.slow
def test_crawl():
    load_and_apply_word_embeddings('crawl')


@pytest.mark.slow
def test_news():
    load_and_apply_word_embeddings('news')


@pytest.mark.slow
def test_fr():
    load_and_apply_word_embeddings('fr')


@pytest.mark.slow
def test_it():
    load_and_apply_word_embeddings('it')

@pytest.mark.slow
def test_it():
    load_and_apply_word_embeddings('it-wiki')

@pytest.mark.slow
def test_it():
    load_and_apply_word_embeddings('it-crawl')

@pytest.mark.slow
def test_news_forward():
    load_and_apply_char_lm_embeddings('news-forward')


@pytest.mark.slow
def test_news_backward():
    load_and_apply_char_lm_embeddings('news-backward')


@pytest.mark.slow
def test_mix_forward():
    load_and_apply_char_lm_embeddings('mix-forward')


@pytest.mark.slow
def test_mix_backward():
    load_and_apply_char_lm_embeddings('mix-backward')


@pytest.mark.slow
def test_german_forward():
    load_and_apply_char_lm_embeddings('german-forward')


@pytest.mark.slow
def test_german_backward():
    load_and_apply_char_lm_embeddings('german-backward')


def test_loading_not_existing_embedding():
    with pytest.raises(ValueError):
        WordEmbeddings('other')

    with pytest.raises(ValueError):
        WordEmbeddings('not/existing/path/to/embeddings')


def test_loading_not_existing_char_lm_embedding():
    with pytest.raises(ValueError):
        CharLMEmbeddings('other')


@pytest.mark.integration
def test_stacked_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: StackedEmbeddings = StackedEmbeddings([glove, charlm])

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert(len(token.get_embedding()) == 1124)

        token.clear_embeddings()

        assert(len(token.get_embedding()) == 0)


@pytest.mark.integration
def test_document_lstm_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove, charlm], hidden_size=128,
                                                                bidirectional=False)

    embeddings.embed(sentence)

    assert (len(sentence.get_embedding()) == 128)
    assert (len(sentence.get_embedding()) == embeddings.embedding_length)

    sentence.clear_embeddings()

    assert (len(sentence.get_embedding()) == 0)


@pytest.mark.integration
def test_document_bidirectional_lstm_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove, charlm], hidden_size=128,
                                                                bidirectional=True)

    embeddings.embed(sentence)

    assert (len(sentence.get_embedding()) == 512)
    assert (len(sentence.get_embedding()) == embeddings.embedding_length)

    sentence.clear_embeddings()

    assert (len(sentence.get_embedding()) == 0)


@pytest.mark.integration
def test_document_pool_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    for mode in ['mean', 'max', 'min']:
        embeddings: DocumentPoolEmbeddings = DocumentPoolEmbeddings([glove, charlm], mode=mode)

        embeddings.embed(sentence)

        assert (len(sentence.get_embedding()) == 1124)

        sentence.clear_embeddings()

        assert (len(sentence.get_embedding()) == 0)


def init_document_embeddings():
    text = 'I love Berlin. Berlin is a great place to live.'
    sentence: Sentence = Sentence(text)

    glove: TokenEmbeddings = WordEmbeddings('en-glove')
    charlm: TokenEmbeddings = CharLMEmbeddings('news-forward-fast')

    return sentence, glove, charlm


def load_and_apply_word_embeddings(emb_type: str):
    text = 'I love Berlin.'
    sentence: Sentence = Sentence(text)
    embeddings: TokenEmbeddings = WordEmbeddings(emb_type)
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert(len(token.get_embedding()) != 0)

        token.clear_embeddings()

        assert(len(token.get_embedding()) == 0)


def load_and_apply_char_lm_embeddings(emb_type: str):
    text = 'I love Berlin.'
    sentence: Sentence = Sentence(text)
    embeddings: TokenEmbeddings = CharLMEmbeddings(emb_type)
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert(len(token.get_embedding()) != 0)

        token.clear_embeddings()

        assert(len(token.get_embedding()) == 0)
