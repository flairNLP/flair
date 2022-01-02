import torch

from flair.models.clustering.birch import branching_factor_non_leaf, distance_max
from flair.models.clustering.birch.model.cfNode import CfNode
from flair.models.clustering.birch.model.clusteringFeature import ClusteringFeature
from flair.models.clustering.birch.model.leafNode import LeafNode


class NonLeafNode(CfNode):
    def __init__(self, parent=None):
        super().__init__()
        if parent is not None:
            self.parent = parent
        self.entries = [LeafNode(parent=self)]
        self.cfs = [ClusteringFeature()]

    def can_add_node(self) -> bool:
        return self.entries.__len__() < branching_factor_non_leaf

    def add_node(self, cf_node: CfNode):
        cf_node.parent = self
        self.entries.append(cf_node)
        self.cfs.append(cf_node.sum_all_cfs())

    def get_child_index(self, child: CfNode) -> int:
        for idx, entry in enumerate(self.entries):
            if entry is child:
                return idx

        if child is self:
            raise Exception("Child is the same as self.")
        else:
            raise Exception("No Child found.")

    def get_closest_child_index(self, vector: ClusteringFeature) -> int:
        min_distance = torch.empty(1, device="cuda").fill_(distance_max)
        index = None

        for idx, cf in enumerate(self.cfs):
            distance = cf.calculate_distance(vector)

            if distance < min_distance:
                min_distance = distance
                index = idx

        return index

    def get_closest_child(self, cf: ClusteringFeature) -> CfNode:
        return self.entries[self.get_closest_child_index(cf)]
