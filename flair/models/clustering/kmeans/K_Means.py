import copy
import random

import numpy as np
import torch
from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings

from Clustering import Clustering
from distance import Distance


class KMeans(Clustering):
    def __init__(self, k, embeddings: DocumentEmbeddings = None):
        self.k = k
        self.embeddings = embeddings
        self.kPoints = []
        self.predict = []
        self.maxIteration = 100

    def cluster(self, sentences: list, batchSize: int = 64) -> list:
        print('k Means cluster start with k: ' + str(self.k))

        for batch in DataLoader(sentences, batch_size=batchSize):
            self.embeddings.embed(batch)

        vectors = [sentence.embedding for sentence in sentences]
        clusterResult = self.clusterVectors(vectors)

        for idx, sentence in enumerate(sentences):
            sentence.set_label('cluster', str(self.predict[idx]))

        return clusterResult

    def clusterVectors(self, vectors: list) -> list:
        self.predict = [0] * len(vectors)

        for i in range(self.k):
            idxs = torch.from_numpy(np.random.choice(len(vectors), self.k, replace=False))
            self.kPoints = [vectors[i] for i in idxs]

        for i in range(0, self.maxIteration):
            clusters = self.iterateAndAssign(vectors)
            oldPoints = copy.copy(self.kPoints)
            self.recalculateClusters(clusters)

            if self.isAlgorithmConvered(oldPoints):
                print("Finished the k-Means with: " + str(i) + " iterations. ")
                return clusters

        print("Finished the k-Means with maxItertation: " + str(self.maxIteration))
        return clusters

    def isAlgorithmConvered(self, oldPoints: list) -> bool:
        bools = []

        for idx, point in enumerate(self.kPoints):
            bools.append(bool(torch.all(torch.eq(self.kPoints[idx], oldPoints[idx]))))
        return all(bools)

    def iterateAndAssign(self, vectors: list) -> list:
        cluster = []
        for i in range(self.k):
            cluster.append([])

        for idx, item in enumerate(vectors):
            clusterIndex = -1
            maxi = 0

            for i in range(self.k):
                distance = Distance.getCosineDistance(item, self.kPoints[i])
                if distance > maxi and distance != 1:
                    maxi = distance
                    clusterIndex = i

            cluster[clusterIndex].append(vectors[idx])
            self.predict[idx] = clusterIndex

        return cluster

    def recalculateClusters(self, clusters: list):
        # TODO: ZeroDivisionError: division by zero
        for idx, cluster in enumerate(clusters):
            if cluster.__len__() != 0:
                self.kPoints[idx] = 1 / cluster.__len__() * sum(cluster)
