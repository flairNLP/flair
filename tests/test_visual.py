import os
import shutil

import pytest

from flair.visual import *
from flair.data import Sentence
from flair.embeddings import CharLMEmbeddings, StackedEmbeddings
import numpy

from flair.visual.manifold import Visualizer, tSNE
from flair.visual.training_curves import Plotter


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
def test_visualize_word_emeddings():

    with open('./resources/visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    sentences = [Sentence(x) for x in sentences]

    charlm_embedding_forward = CharLMEmbeddings('news-forward')
    charlm_embedding_backward = CharLMEmbeddings('news-backward')

    embeddings = StackedEmbeddings([charlm_embedding_backward, charlm_embedding_forward])

    visualizer = Visualizer()
    visualizer.visualize_word_emeddings(embeddings, sentences, './resources/visual/sentence_embeddings.html')

    # clean up directory
    os.remove('./resources/visual/sentence_embeddings.html')


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
def test_visualize_word_emeddings():

    with open('./resources/visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    sentences = [Sentence(x) for x in sentences]

    charlm_embedding_forward = CharLMEmbeddings('news-forward')

    visualizer = Visualizer()
    visualizer.visualize_char_emeddings(charlm_embedding_forward, sentences, './resources/visual/sentence_embeddings.html')

    # clean up directory
    os.remove('./resources/visual/sentence_embeddings.html')


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")
def test_visualize():

    with open('./resources/visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    sentences = [Sentence(x) for x in sentences]

    embeddings = CharLMEmbeddings('news-forward')

    visualizer = Visualizer()

    X_forward = visualizer.prepare_char_embeddings(embeddings, sentences)

    embeddings = CharLMEmbeddings('news-backward')

    X_backward = visualizer.prepare_char_embeddings(embeddings, sentences)

    X = numpy.concatenate([X_forward, X_backward], axis=1)

    contexts = visualizer.char_contexts(sentences)

    trans_ = tSNE()
    reduced = trans_.fit(X)

    visualizer.visualize(reduced, contexts, './resources/visual/char_embeddings.html')

    # clean up directory
    os.remove('./resources/visual/char_embeddings.html')


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