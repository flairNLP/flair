from flair.models.clustering.birch.model.ClusteringFeature import ClusteringFeature


class CfNode:
    def __init__(self):
        self.cfs = []
        self.is_leaf = False
        self.parent = None

    def sum_all_cfs(self) -> ClusteringFeature:
        result = ClusteringFeature()

        for cf in self.cfs:
            result.absorb_cf(cf)

        return result
