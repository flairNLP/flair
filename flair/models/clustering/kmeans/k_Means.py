import copy

import numpy as np
import torch

from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings
from flair.models.clustering.clustering import Clustering
from flair.models.clustering.distance.distance import get_cosine_distance


class KMeans(Clustering):
    def __init__(self, k, embeddings: DocumentEmbeddings = None):
        self.k = k
        self.embeddings = embeddings
        self.k_points = []
        self.predict = []
        self.max_iteration = 100

    def cluster(self, sentences: list, batch_size: int = 64) -> list:
        print("k Means cluster start with k: " + str(self.k))

        for batch in DataLoader(sentences, batch_size=batch_size):
            self.embeddings.embed(batch)

        vectors = [sentence.embedding for sentence in sentences]
        cluster_result = self.cluster_vectors(vectors)

        for idx, sentence in enumerate(sentences):
            sentence.set_label("cluster", str(self.predict[idx]))

        return cluster_result

    def cluster_vectors(self, vectors: list) -> list:
        self.predict = [0] * len(vectors)

        for i in range(self.k):
            idxs = torch.from_numpy(np.random.choice(len(vectors), self.k, replace=False))
            self.k_points = [vectors[i] for i in idxs]

        for i in range(0, self.max_iteration):
            clusters = self.iterate_and_assign(vectors)
            old_points = copy.copy(self.k_points)
            self.recalculate_clusters(clusters)

            if self.is_algorithm_converged(old_points):
                print("Finished the k-Means with: " + str(i) + " iterations. ")
                return clusters

        print("Finished the k-Means with maxItertation: " + str(self.max_iteration))
        return clusters

    def is_algorithm_converged(self, old_points: list) -> bool:
        bools = []

        for idx, point in enumerate(self.k_points):
            bools.append(bool(torch.all(torch.eq(self.k_points[idx], old_points[idx]))))
        return all(bools)

    def iterate_and_assign(self, vectors: list) -> list:
        cluster = []
        for i in range(self.k):
            cluster.append([])

        for idx, vector in enumerate(vectors):
            cluster_index = -1
            min_distance = 1000

            for i in range(self.k):
                distance_to_centroid = get_cosine_distance(vector, self.k_points[i])
                if distance_to_centroid < min_distance:
                    min_distance = distance_to_centroid
                    cluster_index = i

            cluster[cluster_index].append(vector)
            self.predict[idx] = cluster_index

        return cluster

    def recalculate_clusters(self, clusters: list):
        # TODO: ZeroDivisionError: division by zero
        for idx, cluster in enumerate(clusters):
            if cluster.__len__() != 0:
                self.k_points[idx] = 1 / cluster.__len__() * sum(cluster)
            else:
                print("No element found in Cluster: ERROR")
