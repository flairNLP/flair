from flair.data import Sentence
from flair.embeddings import (
    FlairEmbeddings,
    StackedEmbeddings,
    TokenEmbeddings,
    WordEmbeddings,
)
from flair.embeddings.base import load_embeddings


def test_stacked_embeddings():
    glove: TokenEmbeddings = WordEmbeddings("turian")
    flair_embedding: TokenEmbeddings = FlairEmbeddings("news-forward-fast")
    embeddings: StackedEmbeddings = StackedEmbeddings([glove, flair_embedding])

    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 1074

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings


def test_stacked_embeddings_stay_the_same_after_saving_and_loading():
    glove: TokenEmbeddings = WordEmbeddings("turian")
    flair_embedding: TokenEmbeddings = FlairEmbeddings("news-forward-fast")
    embeddings: StackedEmbeddings = StackedEmbeddings([glove, flair_embedding])

    assert not embeddings.training

    sentence_old: Sentence = Sentence("I love Berlin")
    embeddings.embed(sentence_old)
    names_old = embeddings.get_names()
    embedding_length_old = embeddings.embedding_length

    save_data = embeddings.save_embeddings(use_state_dict=True)
    new_embeddings = load_embeddings(save_data)

    sentence_new: Sentence = Sentence("I love Berlin")
    new_embeddings.embed(sentence_new)
    names_new = new_embeddings.get_names()
    embedding_length_new = new_embeddings.embedding_length

    assert not new_embeddings.training
    assert names_old == names_new
    assert embedding_length_old == embedding_length_new

    for token_old, token_new in zip(sentence_old, sentence_new):
        assert (token_old.get_embedding(names_old) == token_new.get_embedding(names_new)).all()
