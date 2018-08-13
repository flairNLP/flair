from flair.visual import *
from flair.data import Sentence
from flair.embeddings import CharLMEmbeddings, StackedEmbeddings
import unittest
import numpy


class Test(unittest.TestCase):
    def test_prepare(self):
        with open('resources/data/snippet.txt') as f:
            sentences = [x for x in f.read().split('\n') if x]

        sentences = [Sentence(x) for x in sentences[:100]]

        charlm_embedding_forward = CharLMEmbeddings('news-forward')
        charlm_embedding_backward = CharLMEmbeddings('news-backward')

        embeddings = StackedEmbeddings(
            [charlm_embedding_backward, charlm_embedding_forward]
        )

        X = prepare_word_embeddings(embeddings, sentences)
        contexts = word_contexts(sentences)

        numpy.save('resources/data/embeddings', X)

        with open('resources/data/contexts.txt', 'w') as f:
            f.write('\n'.join(contexts))

    def test_tSNE(self):

        X = numpy.load('resources/data/embeddings.npy')
        trans_ = tSNE()
        reduced = trans_.fit(X)

        numpy.save('resources/data/tsne', reduced)

    def test__prepare_char(self):

        with open('resources/data/snippet.txt') as f:
            sentences = [x for x in f.read().split('\n') if x]

        sentences = [Sentence(x) for x in sentences[:100]]

        embeddings = CharLMEmbeddings('news-forward')

        X_forward = prepare_char_embeddings(embeddings, sentences)

        embeddings = CharLMEmbeddings('news-backward')

        X_backward = prepare_char_embeddings(embeddings, sentences)

        X = numpy.concatenate([X_forward, X_backward], axis=1)

        numpy.save('resources/data/char_embeddings', X)

    def test_tSNE_char(self):

        X = numpy.load('resources/data/char_embeddings.npy')
        trans_ = tSNE()
        reduced = trans_.fit(X)

        numpy.save('resources/data/char_tsne', reduced)

    def test_prepare_char_uni(self):

        with open('resources/data/snippet.txt') as f:
            sentences = [x for x in f.read().split('\n') if x]

        sentences = [Sentence(x) for x in sentences[:100]]

        embeddings = CharLMEmbeddings('news-forward')

        X = prepare_char_embeddings(embeddings, sentences)

        numpy.save('resources/data/uni_embeddings', X)

    def test_tSNE_char_uni(self):

        X = numpy.load('resources/data/uni_embeddings.npy')
        trans_ = tSNE()
        reduced = trans_.fit(X)

        numpy.save('resources/data/uni_tsne', reduced)

    def test_char_contexts(self):

        with open('resources/data/snippet.txt') as f:
            sentences = [x for x in f.read().split('\n') if x]

        sentences = [Sentence(x) for x in sentences[:100]]

        contexts = char_contexts(sentences)

        with open('resources/data/char_contexts.txt', 'w') as f:
            f.write('\n'.join(contexts))

    def test_benchmark(self):

        import time

        with open('resources/data/snippet.txt') as f:
            sentences = [x for x in f.read().split('\n') if x]

        sentences = [Sentence(x) for x in sentences[:10]]


        charlm_embedding_forward = CharLMEmbeddings('news-forward')
        charlm_embedding_backward = CharLMEmbeddings('news-backward')

        embeddings = StackedEmbeddings(
            [charlm_embedding_backward, charlm_embedding_forward]
        )

        tic = time.time()

        prepare_word_embeddings(embeddings, sentences)

        current_elaped = time.time() - tic

        print('current implementation: {} sec/ sentence'.format(current_elaped / 10))

        embeddings_f = CharLMEmbeddings('news-forward')
        embeddings_b = CharLMEmbeddings('news-backward')

        tic = time.time()

        prepare_char_embeddings(embeddings_f, sentences)
        prepare_char_embeddings(embeddings_b, sentences)

        current_elaped = time.time() - tic

        print('pytorch implementation: {} sec/ sentence'.format(current_elaped / 10))


class Test_show(unittest.TestCase):
    def test_word(self):

        reduced = numpy.load('resources/data/tsne.npy')

        with open('resources/data/contexts.txt') as f:
            contexts = f.read().split('\n')

        show(reduced, contexts)

    def test_char(self):

        reduced = numpy.load('resources/data/char_tsne.npy')

        with open('resources/data/char_contexts.txt') as f:
            contexts = f.read().split('\n')

        show(reduced, contexts)

    def test_uni_sentence(self):

        reduced = numpy.load('resources/data/uni_tsne.npy')

        with open('resources/data/snippet.txt') as f:
            sentences = [x for x in f.read().split('\n') if x]

        l = len(sentences[0])

        with open('resources/data/char_contexts.txt') as f:
            contexts = f.read().split('\n')

        show(reduced[:l], contexts[:l])

    def test_uni(self):

        reduced = numpy.load('resources/data/uni_tsne.npy')

        with open('resources/data/char_contexts.txt') as f:
            contexts = f.read().split('\n')

        show(reduced, contexts)


class TestHighlighter(unittest.TestCase):
    def test(self):

        i = numpy.random.choice(2048)

        with open('resources/data/snippet.txt') as f:
            sentences = [x for x in f.read().split('\n') if x]

        embeddings = CharLMEmbeddings('news-forward')

        features = embeddings.lm.get_representation(sentences[0]).squeeze()

        Highlighter().highlight(features[:, i], sentences[0])



if __name__ == '__main__':
    unittest.main()




