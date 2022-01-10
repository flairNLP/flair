from torch import tensor

from flair.models.clustering.birch import distance_max, branching_factor_leaf
from flair.models.clustering.birch.model.cfNode import CfNode
from flair.models.clustering.birch.model.clusteringFeature import ClusteringFeature


class LeafNode(CfNode):
    def __init__(self, init_cfs: list = None, parent=None):
        super().__init__()
        if init_cfs is None:
            self.cfs = [ClusteringFeature()]
        else:
            self.cfs = init_cfs
        self.parent = parent
        self.is_leaf = True
        self.prev = None
        self.next = None

    def add_cf(self, cf: ClusteringFeature):
        self.cfs.append(cf)

    def can_add_new_cf(self):
        return len(self.cfs) <= branching_factor_leaf

    def get_closest_cf(self, vector: tensor) -> ClusteringFeature:
        min_distance = distance_max
        cf_result = None

        for cf in self.cfs:
            distance = cf.calculate_distance(vector)

            if distance < min_distance:
                min_distance = distance
                cf_result = cf

        return cf_result
