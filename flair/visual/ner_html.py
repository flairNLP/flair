import html
from typing import Union

from flair.data import Sentence

TAGGED_ENTITY = """
<mark class="entity" style="background: {color}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 3; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    {entity}
    <span style="font-size: 0.8em; font-weight: bold; line-height: 3; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">{label}</span>
</mark>
"""

PARAGRAPH = """<p>{sentence}</p>"""

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>{title}</title>
    </head>

    <body style="font-size: 16px; font-family: 'Segoe UI'; padding: 4rem 2rem">{text}</body>
</html>
"""


def split_to_spans(s: Sentence, label_name="ner"):
    orig = s.to_original_text()
    last_idx = 0
    spans = []
    tagged_ents = s.get_labels(label_name)
    for ent in tagged_ents:
        if last_idx != ent.data_point.start_position:
            spans.append((orig[last_idx : ent.data_point.start_position], None))
        spans.append((ent.data_point.text, ent.value))
        assert ent.data_point.end_position is not None
        last_idx = ent.data_point.end_position
    if last_idx < len(orig) - 1:
        spans.append((orig[last_idx : len(orig)], None))
    return spans


def render_ner_html(
    sentences: Union[list[Sentence], Sentence],
    title: str = "Flair",
    colors={
        "PER": "#F7FF53",
        "ORG": "#E8902E",
        "LOC": "#FF40A3",
        "MISC": "#4647EB",
        "O": "#ddd",
    },
    default_color: str = "#ddd",
    wrap_page=True,
    label_name="ner",
) -> str:
    """Create the html code to visualize some sentences.

    Args:
        sentences: single sentence or list of sentences to convert to HTML
        title: title of the HTML page
        colors: dict where keys are tags and values are color HTML codes
        default_color: color to use if colors parameter is missing a tag
        wrap_page: if True method returns result of processing sentences wrapped by &lt;html&gt; and &lt;body&gt; tags, otherwise - without these tags
        label_name: the label name to specify which labels of the sentence are visualized.

    Returns: HTML as a string
    """
    if isinstance(sentences, Sentence):
        sentences = [sentences]
    sentences_html = []
    for s in sentences:
        spans = split_to_spans(s, label_name=label_name)
        spans_html = []
        for fragment, tag in spans:
            escaped_fragment = html.escape(fragment).replace("\n", "<br/>")
            if tag:
                escaped_fragment = TAGGED_ENTITY.format(
                    entity=escaped_fragment,
                    label=tag,
                    color=colors.get(tag, default_color),
                )
            spans_html.append(escaped_fragment)
        line = PARAGRAPH.format(sentence="".join(spans_html))
        sentences_html.append(line)

    final_text = "".join(sentences_html)

    if wrap_page:
        return HTML_PAGE.format(text=final_text, title=title)
    else:
        return final_text
