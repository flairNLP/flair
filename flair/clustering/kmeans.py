import logging
import pickle
from typing import List, Union

import torch

from flair.data import Corpus, Sentence, Dataset
from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings

from tqdm import tqdm

from .base import ClusteringModel

log = logging.getLogger("flair")


class KMeans(ClusteringModel):

    def __init__(self, n_clusters: int, embeddings: DocumentEmbeddings, corpus: Corpus, centroids: torch.tensor = None):
        self.n_clusters = n_clusters
        self.centroids = centroids
        self.embeddings = embeddings
        self.corpus = corpus

    def fit(self, max_iter: int = 100, batch_size: int = 64, p_norm: int = 2):
        log.info("k-means clustering started with n_clusters: " + str(self.n_clusters))

        self._embed(self.corpus.train, batch_size=batch_size)

        vectors = torch.stack([sentence.embedding for sentence in self.corpus.train])
        self.centroids = self._init_centroids(vectors)

        for iter in range(max_iter):
            assignments = torch.argmin(torch.cdist(vectors, self.centroids, p=p_norm), dim=1)
            self._adjust_centroids(vectors=vectors, assignments=assignments)
            if iter > 0 and all(assignments == previous_assignments):
                log.info(f"k-means converged after {iter} iterations.")
                break
            else:
                previous_assignments = assignments

        self.save()

    def _embed(self, sentences: Union[List[Sentence], Sentence], batch_size: int):
        log.info("Embed sentences...")
        for batch in tqdm(DataLoader(sentences, batch_size=batch_size)):
            self.embeddings.embed(batch)

    def _init_centroids(self, vectors: torch.tensor) -> torch.tensor:
        return vectors[torch.randint(high=vectors.shape[0], size=(self.n_clusters,))]

    def _adjust_centroids(self, vectors: torch.tensor, assignments: torch.tensor) -> torch.tensor:
        self.centroids = torch.stack([torch.mean(vectors[assignments == cluster], dim=0) for cluster in range(self.n_clusters)])

    def predict(self, sentences: Union[List[Sentence], Sentence], batch_size: int = 64, p_norm: int = 2):
        if not isinstance(sentences, list) and not isinstance(sentences, Dataset):
            sentences = [sentences]

        self._embed(sentences=sentences, batch_size=batch_size)
        vectors = torch.stack([sentence.embedding for sentence in sentences])
        return torch.argmin(torch.cdist(vectors, self.centroids, p=p_norm), dim=1)

    def save(self, path: str = "kmeans.pt"):
        state_dict = {
            "embeddings": self.embeddings.__class__,
            "corpus": self.corpus.__class__,
            "centroids": self.centroids,
            "n_clusters": self.n_clusters
        }

        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)

    @staticmethod
    def load(path="kmeans.pt"):
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)

        model = KMeans(
            n_clusters=state_dict["n_clusters"],
            embeddings=state_dict["embeddings"](),
            corpus=state_dict["corpus"](),
            centroids=state_dict["centroids"]
        )
        return model
