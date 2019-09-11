import html
from typing import Union, List

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


def split_to_spans(s: Sentence):
    orig = s.to_original_text()
    last_idx = 0
    spans = []
    tagged_ents = s.get_spans("ner")
    for ent in tagged_ents:
        if last_idx != ent.start_pos:
            spans.append((orig[last_idx : ent.start_pos], None))
        spans.append((ent.text, ent.tag))
        last_idx = ent.end_pos
    if last_idx < len(orig) - 1:
        spans.append((orig[last_idx : len(orig)], None))
    return spans


def render_ner_html(
    sentences: Union[List[Sentence], Sentence],
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
) -> str:
    """
    :param sentences: single sentence or list of sentences to convert to HTML
    :param title: title of the HTML page
    :param colors: dict where keys are tags and values are color HTML codes
    :param default_color: color to use if colors parameter is missing a tag
    :param wrap_page: if True method returns result of processing sentences wrapped by &lt;html&gt; and &lt;body&gt; tags, otherwise - without these tags
    :return: HTML as a string
    """
    if isinstance(sentences, Sentence):
        sentences = [sentences]
    sentences_html = []
    for s in sentences:
        spans = split_to_spans(s)
        spans_html = list()
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
