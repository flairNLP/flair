from sklearn.manifold import TSNE
import tqdm
import numpy


class tSNE:
    def __init__(self, embeddings):

        self.embeddings = embeddings

        self.transform = \
            TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

    def _prepare(self, sentences):

        X = []

        print('computing embeddings')
        for sentence in tqdm.tqdm(sentences):
            self.embeddings.embed(sentence)

            for token in sentence:
                X.append(token.embedding.detach().numpy()[None, :])

        X = numpy.concatenate(X, 0)

        return X

    def fit(self, sentences):

        return self.transform.fit_transform(self._prepare(sentences))
