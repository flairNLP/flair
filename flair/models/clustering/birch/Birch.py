from flair.embeddings import DocumentEmbeddings

from Clustering import Clustering
from birch.model.CfTree import CfTree
from birch.model.ClusteringFeature import ClusteringFeature
from flair.datasets import DataLoader

from kmeans.K_Means import KMeans

branchingFactorNonLeaf = 0
branchingFactorLeaf = 0
distanceMax = 1000000000
threshold = 0


class Birch(Clustering):
    def __init__(self, thresholds: float, embeddings: DocumentEmbeddings, B: int, L: int):
        global threshold
        threshold = thresholds
        global branchingFactorLeaf
        branchingFactorLeaf = L
        global branchingFactorNonLeaf
        branchingFactorNonLeaf = B
        global distanceMax

        self.embeddings = embeddings
        self.cfTree = CfTree()
        self.predict = []

    def cluster(self, vectors: list, batchSize: int = 64):
        print("Starting BIRCH clustering with threshold: " + str(threshold))
        self.predict = [0] * len(vectors)

        for batch in DataLoader(vectors, batch_size=batchSize):
            self.embeddings.embed(batch)

        for idx, vector in enumerate(vectors):
            self.cfTree.insertCf(ClusteringFeature(vector.embedding, idx=idx))
            self.cfTree.validate()

        cfs = self.cfTree.getLeafCfs()
        cfVectors = self.cfTree.getVectorsFromCf(cfs)

        kMeans = KMeans(3)
        kMeans.clusterVectors(cfVectors)

        for idx, cf in enumerate(cfs):
            for cfIndex in cf.indices:
                self.predict[cfIndex] = kMeans.predict[idx]

        return self.cfTree
