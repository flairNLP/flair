from flair.data import Sentence
from flair.embeddings import (
    FlairEmbeddings,
    StackedEmbeddings,
    TokenEmbeddings,
    WordEmbeddings,
)


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
