from flair.visual import tSNE
from flair.data import Sentence
from flair.embeddings import CharLMEmbeddings, StackedEmbeddings
import unittest
import numpy


class TesttSNE(unittest.TestCase):
    def test(self):
        with open('resources/data/snippet.txt') as f:
            sentences = [x for x in f.read().split('\n') if x]

        sentences = [Sentence(x) for x in sentences[:100]]

        charlm_embedding_forward = CharLMEmbeddings('news-forward')
        charlm_embedding_backward = CharLMEmbeddings('news-backward')

        embeddings = StackedEmbeddings(
            [charlm_embedding_backward, charlm_embedding_forward]
        )

        trans_ = tSNE(embeddings)

        embeddings = trans_.fit(sentences)

        numpy.save(embeddings, 'resources/data/embeddings.npy')


if __name__ == '__main__':
    unittest.main()




