import pytest

from flair.embeddings import WordEmbeddings, TokenEmbeddings, CharLMEmbeddings, StackedEmbeddings, \
    DocumentLSTMEmbeddings, DocumentMeanEmbeddings

from flair.data import Sentence


def test_en_glove():
    load_and_apply_word_embeddings('en-glove')


def test_en_numberbatch():
    load_and_apply_word_embeddings('en-numberbatch')


def test_en_extvec():
    load_and_apply_word_embeddings('en-extvec')


def test_en_crawl():
    load_and_apply_word_embeddings('en-crawl')


def test_en_news():
    load_and_apply_word_embeddings('en-news')


def test_de_fasttext():
    load_and_apply_word_embeddings('de-fasttext')


def test_de_numberbatch():
    load_and_apply_word_embeddings('de-numberbatch')


def test_sv_fasttext():
    load_and_apply_word_embeddings('sv-fasttext')


def test_news_forward():
    load_and_apply_char_lm_embeddings('news-forward')


def test_news_backward():
    load_and_apply_char_lm_embeddings('news-backward')


def test_mix_forward():
    load_and_apply_char_lm_embeddings('mix-forward')


def test_mix_backward():
    load_and_apply_char_lm_embeddings('mix-backward')


def test_german_forward():
    load_and_apply_char_lm_embeddings('german-forward')


def test_german_backward():
    load_and_apply_char_lm_embeddings('german-backward')


def test_stacked_embeddings():
    text = 'I love Berlin.'
    sentence: Sentence = Sentence(text)

    glove: TokenEmbeddings = WordEmbeddings('en-glove')
    news: TokenEmbeddings = WordEmbeddings('en-news')
    charlm: TokenEmbeddings = CharLMEmbeddings('mix-backward')

    embeddings: StackedEmbeddings = StackedEmbeddings([glove, news, charlm])

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert(len(token.get_embedding()) != 0)

        token.clear_embeddings()

        assert(len(token.get_embedding()) == 0)


@pytest.fixture
def init_document_embeddings():
    text = 'I love Berlin. Berlin is a great place to live.'
    sentence: Sentence = Sentence(text)

    glove: TokenEmbeddings = WordEmbeddings('en-glove')
    charlm: TokenEmbeddings = CharLMEmbeddings('mix-backward')

    return sentence, glove, charlm


def test_document_lstm_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove, charlm], hidden_states=128,
                                                                bidirectional=False, use_first_representation=False)

    embeddings.embed(sentence)

    assert (len(sentence.get_embedding()) != 0)
    assert (sentence.get_embedding().shape[1] == embeddings.embedding_length)

    sentence.clear_embeddings()

    assert (len(sentence.get_embedding()) == 0)


def test_document_bidirectional_lstm_embeddings():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove, charlm], hidden_states=128,
                                                                bidirectional=True, use_first_representation=False)

    embeddings.embed(sentence)

    assert (len(sentence.get_embedding()) != 0)
    assert (sentence.get_embedding().shape[1] == embeddings.embedding_length)

    sentence.clear_embeddings()

    assert (len(sentence.get_embedding()) == 0)


def test_document_bidirectional_lstm_embeddings_using_first_representation():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove, charlm], hidden_states=128,
                                                                bidirectional=True, use_first_representation=True)

    embeddings.embed(sentence)

    assert (len(sentence.get_embedding()) != 0)
    assert (sentence.get_embedding().shape[1] == embeddings.embedding_length)

    sentence.clear_embeddings()

    assert (len(sentence.get_embedding()) == 0)


def test_document_lstm_embeddings_using_first_representation():
    sentence, glove, charlm = init_document_embeddings()

    embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings([glove, charlm], hidden_states=128,
                                                                bidirectional=False, use_first_representation=True)

    embeddings.embed(sentence)

    assert (len(sentence.get_embedding()) != 0)
    assert (sentence.get_embedding().shape[1] == embeddings.embedding_length)

    sentence.clear_embeddings()

    assert (len(sentence.get_embedding()) == 0)


def test_document_mean_embeddings():
    text = 'I love Berlin. Berlin is a great place to live.'
    sentence: Sentence = Sentence(text)

    glove: TokenEmbeddings = WordEmbeddings('en-glove')
    charlm: TokenEmbeddings = CharLMEmbeddings('mix-backward')

    embeddings: DocumentMeanEmbeddings = DocumentMeanEmbeddings([glove, charlm])

    embeddings.embed(sentence)

    assert (len(sentence.get_embedding()) != 0)

    sentence.clear_embeddings()

    assert (len(sentence.get_embedding()) == 0)


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