from unittest.mock import MagicMock

from flair.data import Sentence, Span
from flair.embeddings import FlairEmbeddings
from flair.visual import *
from flair.visual.html_templates import HTML_PAGE, TAGGED_ENTITY
from flair.visual.training_curves import Plotter


def test_highlighter(resources_path):
    with (resources_path / "visual/snippet.txt").open() as f:
        sentences = [x for x in f.read().split("\n") if x]

    embeddings = FlairEmbeddings("news-forward")

    features = embeddings.lm.get_representation(sentences[0]).squeeze()

    Highlighter().highlight_selection(
        features,
        sentences[0],
        n=1000,
        file_=str(resources_path / "visual/highligh.html"),
    )

    # clean up directory
    (resources_path / "visual/highligh.html").unlink()


def test_plotting_training_curves_and_weights(resources_path):
    plotter = Plotter()
    plotter.plot_training_curves(resources_path / "visual/loss.tsv")
    plotter.plot_weights(resources_path / "visual/weights.txt")

    # clean up directory
    (resources_path / "visual/weights.png").unlink()
    (resources_path / "visual/training.png").unlink()


def mock_ner_span(tag, start, end):
    span = Span([])
    span.tag = tag
    span.start_pos = start
    span.end_pos = end
    return span


def test_html_rendering():
    text = (
        "Boris Johnson has been elected new Conservative leader in a ballot of party members and will become the "
        "next UK prime minister. &"
    )
    sent = Sentence()
    sent.get_spans = MagicMock()
    sent.get_spans.return_value = [
        mock_ner_span("PER", 0, 13),
        mock_ner_span("MISC", 35, 47),
        mock_ner_span("LOC", 109, 111),
    ]
    sent.to_original_text = MagicMock()
    sent.to_original_text.return_value = text
    settings = {"colors": {"LOC": "yellow"}, "labels": {"LOC": "location"}}
    actual = Visualizer.render_ner_html([sent], settings=settings)

    expected_res = HTML_PAGE.format(
        text=TAGGED_ENTITY.format(color="#F7FF53", entity="Boris Johnson", label="PER")
        + " has been elected new "
        + TAGGED_ENTITY.format(color="#4647EB", entity="Conservative", label="MISC")
        + " leader in a ballot of party members and will become the next "
        + TAGGED_ENTITY.format(color="yellow", entity="UK", label="location")
        + " prime minister. &amp;"
    )

    assert expected_res == actual
