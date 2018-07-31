from flair.embeddings import WordEmbeddings, TokenEmbeddings, CharLMEmbeddings, StackedEmbeddings

from flair.data import Sentence


def test_en_glove():
    load_and_apply_embeddings('en-glove')


def test_en_numberbatch():
    load_and_apply_embeddings('en-numberbatch')


def test_en_extvec():
    load_and_apply_embeddings('en-extvec')


def test_en_crawl():
    load_and_apply_embeddings('en-crawl')


def test_en_news():
    load_and_apply_embeddings('en-news')


def test_de_fasttext():
    load_and_apply_embeddings('de-fasttext')


def test_de_numberbatch():
    load_and_apply_embeddings('de-numberbatch')


def test_sv_fasttext():
    load_and_apply_embeddings('sv-fasttext')


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

    embeddings: StackedEmbeddings = StackedEmbeddings([glove, news])

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert(len(token.get_embedding()) != 0)

        token.clear_embeddings()

        assert(len(token.get_embedding()) == 0)


def load_and_apply_embeddings(emb_type: str):
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