from sklearn.manifold import TSNE
import tqdm
import numpy


class _Transform:
    def __init__(self):
        pass

    def fit(self, X):
        return self.transform.fit_transform(X)


class tSNE(_Transform):
    def __init__(self):
        super().__init__()

        self.transform = \
            TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)


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

        print('computing embeddings')
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
                prop = '<b><font color="red"> {token} </font></b>'.format(
                    token=token)

                prop = ' '.join(strs[max(i - 4, 0):i]) + prop
                prop = prop + ' '.join(strs[i + 1:min(len(strs), i + 5)])

                contexts.append('<p>' + prop + '</p>')

        return contexts

    @staticmethod
    def prepare_char_embeddings(embeddings, sentences):
        X = []

        print('computing embeddings')
        for sentence in tqdm.tqdm(sentences):

            sentence = ' '.join([x.text for x in sentence])

            hidden = embeddings.lm.get_representation([sentence])
            X.append(hidden.squeeze().detach().numpy())

        X = numpy.concatenate(X, 0)

        return X

    @staticmethod
    def char_contexts(sentences):
        contexts = []

        for sentence in sentences:
            sentence = ' '.join([token.text for token in sentence])

            for i, char in enumerate(sentence):

                context = '<span style="background-color: yellow"><b>{}</b></span>'.format(char)
                context = ''.join(sentence[max(i - 30, 0):i]) + context
                context = context + ''.join(sentence[i + 1:min(len(sentence), i + 30)])

                contexts.append(context)

        return contexts

    @staticmethod
    def visualize(X, contexts, file):
        import matplotlib.pyplot
        import mpld3

        fig, ax = matplotlib.pyplot.subplots()

        ax.grid(True, alpha=0.3)

        points = ax.plot(X[:, 0], X[:, 1], 'o', color='b',
                         mec='k', ms=5, mew=1, alpha=.6)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Hover mouse to reveal context', size=20)

        tooltip = mpld3.plugins.PointHTMLTooltip(
            points[0],
            contexts,
            voffset=10,
            hoffset=10
        )

        mpld3.plugins.connect(fig, tooltip)

        mpld3.save_html(fig, file)
