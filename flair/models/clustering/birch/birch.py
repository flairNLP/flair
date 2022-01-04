import logging

from flair.datasets import DataLoader

from flair.embeddings import DocumentEmbeddings
from flair.models.clustering.clustering import Clustering
from flair.models.clustering.birch.model.cfTree import CfTree
from flair.models.clustering.birch.model.clusteringFeature import ClusteringFeature
from flair.models.clustering.kmeans.k_Means import KMeans

log = logging.getLogger("flair")


class Birch(Clustering):
    def __init__(self, thresholds: float, embeddings: DocumentEmbeddings, b: int, l: int, clusters: int):
        Birch.threshold = thresholds
        Birch.branching_factor_leaf = l
        Birch.branching_factor_non_leaf = b

        self.embeddings = embeddings
        self.cf_tree = CfTree()
        self.predict = []
        self.k = clusters

    def cluster(self, vectors: list, batch_size: int = 64):
        log.debug(
            "Starting BIRCH clustering with threshold: "
            + str(Birch.threshold)
            + " with Branchingfactor: "
            + str(Birch.branching_factor_leaf)
        )
        self.predict = [0] * len(vectors)

        for batch in DataLoader(vectors, batch_size=batch_size):
            self.embeddings.embed(batch)

        for idx, vector in enumerate(vectors):
            self.cf_tree.insert_cf(ClusteringFeature(vector.embedding, idx=idx))

        self.cf_tree.validate(vectors)
        cfs = self.cf_tree.get_leaf_cfs()
        cf_vectors = self.cf_tree.get_vectors_from_cf(cfs)

        k_means = KMeans(self.k)
        k_means.cluster_vectors(cf_vectors)

        for idx, cf in enumerate(cfs):
            for cf_index in cf.indices:
                self.predict[cf_index] = k_means.predict[idx]

        return self.cf_tree
