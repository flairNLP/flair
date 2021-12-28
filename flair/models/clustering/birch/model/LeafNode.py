from torch import Tensor

import birch.Birch
from birch.model.ClusteringFeature import ClusteringFeature
from birch.model.CfNode import CfNode


class LeafNode(CfNode):
    def __init__(self, initCfs: list = None, parent=None):
        super().__init__()
        if initCfs is None:
            self.cfs = [ClusteringFeature()]
        else:
            self.cfs = initCfs
        self.parent = parent
        self.isLeaf = True
        self.prev = None
        self.next = None

    def addCF(self, cf: ClusteringFeature):
        self.cfs.append(cf)

    def canAddNewCf(self):
        return self.cfs.__len__() < birch.Birch.branchingFactorLeaf

    def getClosestCF(self, vector: Tensor) -> ClusteringFeature:
        minDistance = birch.Birch.distanceMax
        cfResult = None

        for cf in self.cfs:
            distance = cf.calcualteDistance(vector)

            if distance < minDistance:
                minDistance = distance
                cfResult = cf

        return cfResult


