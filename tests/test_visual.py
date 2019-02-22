from flair.visual import *
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

from flair.visual.training_curves import Plotter


def test_highlighter(resources_path):
    with (resources_path / 'visual/snippet.txt').open() as f:
        sentences = [x for x in f.read().split('\n') if x]

    embeddings = FlairEmbeddings('news-forward')

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
