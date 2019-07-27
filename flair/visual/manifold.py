import html
from typing import Union, List

from sklearn.manifold import TSNE
import tqdm
import numpy

from flair.data import Sentence
from flair.visual.html_templates import TAGGED_ENTITY, HTML_PAGE


class _Transform:
    def __init__(self):
        pass

    def fit(self, X):
        return self.transform.fit_transform(X)


class tSNE(_Transform):
    def __init__(self):
        super().__init__()

        self.transform = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)


def split_to_spans(s: Sentence):
    orig = s.to_original_text()
    last_idx = 0
    spans = []
    tagged_ents = s.get_spans('ner')
    for ent in tagged_ents:
        if last_idx != ent.start_pos:
            spans.append((orig[last_idx:ent.start_pos], None))
        spans.append((orig[ent.start_pos:ent.end_pos], ent.tag))
        last_idx = ent.end_pos
    if last_idx < len(orig) - 1:
        spans.append((orig[last_idx:len(orig)], None))
    return spans


class Visualizer(object):
    def visualize_word_emeddings(self, embeddings, sentences, output_file):
        X = self.prepare_word_embeddings(embeddings, sentences)
        contexts = self.word_contexts(sentences)

        trans_ = tSNE()
        reduced = trans_.fit(X)

        self.visualize(reduced, contexts, output_file)

    def visualize_char_emeddings(self, embeddings, sentences, output_file):
        X = self.prepare_char_embeddings(embeddings, sentences)
        contexts = self.char_contexts(sentences)

        trans_ = tSNE()
        reduced = trans_.fit(X)

        self.visualize(reduced, contexts, output_file)

    @staticmethod
    def prepare_word_embeddings(embeddings, sentences):
        X = []

        for sentence in tqdm.tqdm(sentences):
            embeddings.embed(sentence)

            for i, token in enumerate(sentence):
                X.append(token.embedding.detach().numpy()[None, :])

        X = numpy.concatenate(X, 0)

        return X

    @staticmethod
    def word_contexts(sentences):
        contexts = []

        for sentence in sentences:

            strs = [x.text for x in sentence.tokens]

            for i, token in enumerate(strs):
                prop = '<b><font color="red"> {token} </font></b>'.format(token=token)

                prop = " ".join(strs[max(i - 4, 0): i]) + prop
                prop = prop + " ".join(strs[i + 1: min(len(strs), i + 5)])

                contexts.append("<p>" + prop + "</p>")

        return contexts

    @staticmethod
    def prepare_char_embeddings(embeddings, sentences):
        X = []

        for sentence in tqdm.tqdm(sentences):
            sentence = " ".join([x.text for x in sentence])

            hidden = embeddings.lm.get_representation([sentence])
            X.append(hidden.squeeze().detach().numpy())

        X = numpy.concatenate(X, 0)

        return X

    @staticmethod
    def char_contexts(sentences):
        contexts = []

        for sentence in sentences:
            sentence = " ".join([token.text for token in sentence])

            for i, char in enumerate(sentence):
                context = '<span style="background-color: yellow"><b>{}</b></span>'.format(
                    char
                )
                context = "".join(sentence[max(i - 30, 0): i]) + context
                context = context + "".join(
                    sentence[i + 1: min(len(sentence), i + 30)]
                )

                contexts.append(context)

        return contexts

    @staticmethod
    def visualize(X, contexts, file):
        import matplotlib.pyplot
        import mpld3

        fig, ax = matplotlib.pyplot.subplots()

        ax.grid(True, alpha=0.3)

        points = ax.plot(
            X[:, 0], X[:, 1], "o", color="b", mec="k", ms=5, mew=1, alpha=0.6
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Hover mouse to reveal context", size=20)

        tooltip = mpld3.plugins.PointHTMLTooltip(
            points[0], contexts, voffset=10, hoffset=10
        )

        mpld3.plugins.connect(fig, tooltip)

        mpld3.save_html(fig, file)

    @staticmethod
    def render_ner_html(sentences: Union[List[Sentence], Sentence], settings=None, wrap_page=True) -> str:
        """
        :param sentences: single sentence or list of sentences to convert to HTML
        :param settings: overrides and completes default settings; includes colors and labels dictionaries
        :param wrap_page: if True method returns result of processing sentences wrapped by &lt;html&gt; and &lt;body&gt; tags, otherwise - without these tags
        :return: HTML as a string
        """
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        colors = {
            "PER": "#F7FF53",
            "ORG": "#E8902E",
            "LOC": "#FF40A3",
            "MISC": "#4647EB",
            "O": "#ddd",
        }

        if settings and "colors" in settings:
            colors.update(settings["colors"])

        labels = {
            "PER": "PER",
            "ORG": "ORG",
            "LOC": "LOC",
            "MISC": "MISC",
            "O": "O",
        }

        if settings and "labels" in settings:
            labels.update(settings["labels"])

        tagged_html = []
        for s in sentences:
            spans = split_to_spans(s)

            for fragment, tag in spans:
                escaped_fragment = html.escape(fragment).replace('\n', '<br/>')
                if tag:
                    escaped_fragment = TAGGED_ENTITY.format(entity=escaped_fragment,
                                                            label=labels.get(tag, "O"),
                                                            color=colors.get(tag, "#ddd"))
                tagged_html.append(escaped_fragment)

        final_text = ''.join(tagged_html)

        if wrap_page:
            return HTML_PAGE.format(text=final_text)
        else:
            return final_text
