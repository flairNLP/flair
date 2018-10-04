import os
import shutil

import pytest

from flair.visual import *
from flair.data import Sentence
from flair.embeddings import CharLMEmbeddings, StackedEmbeddings
import numpy

from flair.visual.training_curves import Plotter


@pytest.mark.skip(reason='Skipping test by default due to long execution time.')
def test_benchmark():
    import time

    with open('./resources/visual/snippet.txt') as f:
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


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
def test_show_word_embeddings():

    with open('./resources/visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    sentences = [Sentence(x) for x in sentences]

    charlm_embedding_forward = CharLMEmbeddings('news-forward')
    charlm_embedding_backward = CharLMEmbeddings('news-backward')

    embeddings = StackedEmbeddings([charlm_embedding_backward, charlm_embedding_forward])

    X = prepare_word_embeddings(embeddings, sentences)
    contexts = word_contexts(sentences)

    trans_ = tSNE()
    reduced = trans_.fit(X)

    visualize(reduced, contexts, './resources/visual/sentence_embeddings.html')

    # clean up directory
    os.remove('./resources/visual/sentence_embeddings.html')


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
def test_show_char_embeddings():

    with open('./resources/visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    sentences = [Sentence(x) for x in sentences]

    embeddings = CharLMEmbeddings('news-forward')

    X_forward = prepare_char_embeddings(embeddings, sentences)

    embeddings = CharLMEmbeddings('news-backward')

    X_backward = prepare_char_embeddings(embeddings, sentences)

    X = numpy.concatenate([X_forward, X_backward], axis=1)

    contexts = char_contexts(sentences)

    trans_ = tSNE()
    reduced = trans_.fit(X)

    visualize(reduced, contexts, './resources/visual/char_embeddings.html')

    # clean up directory
    os.remove('./resources/visual/char_embeddings.html')


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
def test_show_uni_sentence_embeddings():

    with open('./resources/visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    sentences = [Sentence(x) for x in sentences]

    embeddings = CharLMEmbeddings('news-forward')

    X = prepare_char_embeddings(embeddings, sentences)

    trans_ = tSNE()
    reduced = trans_.fit(X)

    l = len(sentences[0])

    contexts = char_contexts(sentences)

    visualize(reduced[:l], contexts[:l], './resources/visual/uni_sentence_embeddings.html')

    # clean up directory
    os.remove('./resources/visual/uni_sentence_embeddings.html')


def test_highlighter():
    with open('./resources/visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    embeddings = CharLMEmbeddings('news-forward')

    features = embeddings.lm.get_representation(sentences[0]).squeeze()

    Highlighter().highlight_selection(features, sentences[0], n=1000, file_='./resources/visual/highligh.html')

    # clean up directory
    os.remove('./resources/visual/highligh.html')


def test_plotting_training_curves_and_weights():
    plotter = Plotter()
    plotter.plot_training_curves('./resources/visual/loss.tsv')
    plotter.plot_weights('./resources/visual/weights.txt')

    # clean up directory
    os.remove('./resources/visual/weights.png')
    os.remove('./resources/visual/training.png')