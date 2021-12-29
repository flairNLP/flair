from flair.datasets import DataLoader

from flair.embeddings import DocumentEmbeddings
from flair.models.clustering.Clustering import Clustering
from flair.models.clustering.birch.model.CfTree import CfTree
from flair.models.clustering.birch.model.ClusteringFeature import ClusteringFeature
from flair.models.clustering.kmeans.K_Means import KMeans


class Birch(Clustering):
    def __init__(self, thresholds: float, embeddings: DocumentEmbeddings, b: int, l: int):
        global threshold
        threshold = thresholds
        global branching_factor_leaf
        branching_factor_leaf = l
        global branching_factor_non_leaf
        branching_factor_non_leaf = b
        global distance_max

        self.embeddings = embeddings
        self.cf_tree = CfTree()
        self.predict = []

    def cluster(self, vectors: list, batch_size: int = 64):
        print("Starting BIRCH clustering with threshold: " + str(threshold))
        self.predict = [0] * len(vectors)

        for batch in DataLoader(vectors, batch_size=batch_size):
            self.embeddings.embed(batch)

        for idx, vector in enumerate(vectors):
            self.cf_tree.insert_cf(ClusteringFeature(vector.embedding, idx=idx))
            self.cf_tree.validate()

        cfs = self.cf_tree.get_leaf_cfs()
        cf_vectors = self.cf_tree.get_vectors_from_cf(cfs)

        k_means = KMeans(3)
        k_means.cluster_vectors(cf_vectors)

        for idx, cf in enumerate(cfs):
            for cf_index in cf.indices:
                self.predict[cf_index] = k_means.predict[idx]

        return self.cf_tree
