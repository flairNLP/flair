import copy
import logging

from torch import tensor

import flair
import numpy as np
import torch

from flair.datasets import DataLoader
from flair.models.clustering.clustering import Clustering
from flair.models.clustering.distance.distance import get_cosine_distance

log = logging.getLogger("flair")


class KMeans(Clustering):
    def __init__(self, k, embeddings: flair.embeddings.DocumentEmbeddings = None):
        self.k = k
        self.embeddings = embeddings
        self.k_points = []
        self.predict = []
        self.max_iteration = 100

    def cluster(self, sentences: list, batch_size: int = 64) -> list:
        log.debug("k Means cluster start with k: " + str(self.k))

        for batch in DataLoader(sentences, batch_size=batch_size):
            self.embeddings.embed(batch)

        vectors = [sentence.embedding for sentence in sentences]
        cluster_result = self.cluster_vectors(vectors)

        for idx, sentence in enumerate(sentences):
            sentence.set_label("cluster", str(self.predict[idx]))

        return cluster_result

    def cluster_vectors(self, vectors: list) -> list:
        self.predict = [0] * len(vectors)

        # init the k centroids
        idxs = torch.from_numpy(np.random.choice(len(vectors), self.k))
        self.k_points = [vectors[i] for i in idxs]

        for i in range(self.max_iteration):
            clusters = self.assign_clusters(vectors)
            old_points = copy.copy(self.k_points)
            self.adjust_centroids(clusters)

            if self.is_algorithm_converged(old_points):
                log.debug("Finished the k-Means with: " + str(i) + " iterations. ")
                return clusters

        log.debug("Finished the k-Means with maxItertation: " + str(self.max_iteration))
        return clusters

    def is_algorithm_converged(self, old_points: list) -> bool:
        return bool(torch.all(torch.eq(torch.stack(self.k_points), torch.stack(old_points))))

    def assign_clusters(self, vectors: list) -> list:
        cluster = []
        for i in range(self.k):
            cluster.append([])

        for idx, vector in enumerate(vectors):
            cluster_index = self.assign(vector)

            cluster[cluster_index].append(vector)
            self.predict[idx] = cluster_index

        return cluster

    def assign(self, vector: tensor):
        distance_to_centroid = get_cosine_distance(vector.repeat([self.k, 1]).t(), torch.stack(self.k_points).t())
        index = torch.argmin(distance_to_centroid)

        return int(index)

    def adjust_centroids(self, clusters: list):
        # TODO: ZeroDivisionError: division by zero
        for idx, cluster in enumerate(clusters):
            if len(cluster) != 0:
                self.k_points[idx] = 1 / cluster.__len__() * sum(cluster)
            else:
                log.error("No element found in Cluster: ERROR")
