from flair.data import Sentence, Span, Token
from flair.embeddings import FlairEmbeddings
from flair.visual import Highlighter
from flair.visual.ner_html import HTML_PAGE, PARAGRAPH, TAGGED_ENTITY, render_ner_html
from flair.visual.training_curves import Plotter


def test_highlighter(resources_path):
    with (resources_path / "visual/snippet.txt").open() as f:
        sentences = [x for x in f.read().split("\n") if x]

    embeddings = FlairEmbeddings("news-forward")

    features = embeddings.lm.get_representation(sentences[0], "", "").squeeze()

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


def mock_ner_span(text, tag, start, end):
    span = Span([]).set_label("class", tag)
    span.start_pos = start
    span.end_pos = end
    span.tokens = [Token(text[start:end])]
    return span


def test_html_rendering():
    text = (
        "Boris Johnson has been elected new Conservative leader in "
        "a ballot of party members and will become the "
        "next UK prime minister. &"
    )
    sentence = Sentence(text)

    print(sentence[0:2].add_label("ner", "PER"))
    print(sentence[6:7].add_label("ner", "MISC"))
    print(sentence[19:20].add_label("ner", "LOC"))
    colors = {
        "PER": "#F7FF53",
        "ORG": "#E8902E",
        "LOC": "yellow",
        "MISC": "#4647EB",
        "O": "#ddd",
    }
    actual = render_ner_html([sentence], colors=colors)

    expected_res = HTML_PAGE.format(
        text=PARAGRAPH.format(
            sentence=TAGGED_ENTITY.format(color="#F7FF53", entity="Boris Johnson", label="PER")
            + " has been elected new "
            + TAGGED_ENTITY.format(color="#4647EB", entity="Conservative", label="MISC")
            + " leader in a ballot of party members and will become the next "
            + TAGGED_ENTITY.format(color="yellow", entity="UK", label="LOC")
            + " prime minister. &amp;"
        ),
        title="Flair",
    )

    assert expected_res == actual
