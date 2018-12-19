import os

import pytest

from flair.visual import *
from flair.data import Sentence
from flair.embeddings import CharLMEmbeddings, StackedEmbeddings
import numpy

from flair.visual.manifold import Visualizer, tSNE
from flair.visual.training_curves import Plotter


@pytest.mark.slow
def test_visualize_word_emeddings(resources_path):

    with open(resources_path / 'visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    sentences = [Sentence(x) for x in sentences]

    charlm_embedding_forward = CharLMEmbeddings('news-forward')
    charlm_embedding_backward = CharLMEmbeddings('news-backward')

    embeddings = StackedEmbeddings([charlm_embedding_backward, charlm_embedding_forward])

    visualizer = Visualizer()
    visualizer.visualize_word_emeddings(embeddings, sentences, str(resources_path / 'visual/sentence_embeddings.html'))

    # clean up directory
    (resources_path / 'visual/sentence_embeddings.html').unlink()


@pytest.mark.slow
def test_visualize_word_emeddings(resources_path):

    with open(resources_path / 'visual/snippet.txt') as f:
        sentences = [x for x in f.read().split('\n') if x]

    sentences = [Sentence(x) for x in sentences]

    charlm_embedding_forward = CharLMEmbeddings('news-forward')

    visualizer = Visualizer()
    visualizer.visualize_char_emeddings(charlm_embedding_forward, sentences, str(resources_path / 'visual/sentence_embeddings.html'))

    # clean up directory
    (resources_path / 'visual/sentence_embeddings.html').unlink()


@pytest.mark.slow
def test_visualize(resources_path):

    with open(resources_path / 'visual/snippet.txt') as f:
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

    visualizer.visualize(reduced, contexts, str(resources_path / 'visual/char_embeddings.html'))

    # clean up directory
    (resources_path / 'visual/char_embeddings.html').unlink()


def test_highlighter(resources_path):
    with (resources_path / 'visual/snippet.txt').open() as f:
        sentences = [x for x in f.read().split('\n') if x]

    embeddings = CharLMEmbeddings('news-forward')

    features = embeddings.lm.get_representation(sentences[0]).squeeze()

    Highlighter().highlight_selection(features, sentences[0], n=1000, file_=str(resources_path / 'visual/highligh.html'))

    # clean up directory
    (resources_path / 'visual/highligh.html').unlink()


def test_plotting_training_curves_and_weights(resources_path):
    plotter = Plotter()
    plotter.plot_training_curves(resources_path / 'visual/loss.tsv')
    plotter.plot_weights(resources_path / 'visual/weights.txt')

    # clean up directory
    (resources_path / 'visual/weights.png').unlink()
    (resources_path / 'visual/training.png').unlink()
