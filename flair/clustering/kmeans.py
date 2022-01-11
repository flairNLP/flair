import logging

import torch

from flair.data import Corpus
from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings, SentenceTransformerDocumentEmbeddings

from .base import ClusteringModel
from .distance import get_cosine_distance

log = logging.getLogger("flair")


class KMeans(ClusteringModel):

    def __init__(self, n_clusters, embeddings: DocumentEmbeddings, corpus: Corpus, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.centroids = dict
        self.embeddings = embeddings
        self.corpus = corpus
        self.max_iter = max_iter

    def fit(self, batch_size: int = 64, p_norm: int = 2) -> list:
        log.info("k-means clustering started with n_clusters: " + str(self.n_clusters))

        self._embed(batch_size=batch_size)

        vectors = torch.stack([sentence.embedding for sentence in self.corpus.train])
        self.centroids = self._init_centroids(vectors)

        for iter in range(self.max_iter):
            assignments = torch.argmin(torch.cdist(vectors, self.centroids, p=p_norm), dim=1)
            self._adjust_centroids(vectors=vectors, assignments=assignments)
            if iter > 0 and all(assignments == previous_assignments):
                log.info("k-means converged. exiting...")
                break
            else:
                previous_assignments = assignments

        print()

    def _embed(self, batch_size: int):
        for batch in DataLoader(self.corpus.train, batch_size=batch_size):
            self.embeddings.embed(batch)

    def _init_centroids(self, vectors: torch.tensor):
        return vectors[torch.randint(high=vectors.shape[0], size=(self.n_clusters,))]


    def is_algorithm_converged(self, old_points: list) -> bool:
        return bool(torch.all(torch.eq(torch.stack(self.k_points), torch.stack(old_points))))


    def _adjust_centroids(self, vectors: torch.tensor, assignments: torch.tensor) -> torch.tensor:
        self.centroids = torch.stack([torch.mean(vectors[assignments == cluster], dim=0) for cluster in range(self.n_clusters)])
