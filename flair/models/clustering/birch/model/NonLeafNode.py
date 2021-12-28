from torch import Tensor

import birch.Birch
from birch.model.CfNode import CfNode
from birch.model.ClusteringFeature import ClusteringFeature
from birch.model.LeafNode import LeafNode


class NonLeafNode(CfNode):
    def __init__(self, parent=None):
        super().__init__()
        if parent is not None:
            self.parent = parent
        self.entries = [LeafNode(parent=self)]
        self.cfs = [ClusteringFeature()]

    def canAddNode(self) -> bool:
        return self.entries.__len__() < birch.Birch.branchingFactorNonLeaf

    def addNode(self, cfNode: CfNode):
        cfNode.parent = self
        self.entries.append(cfNode)
        self.cfs.append(cfNode.sumAllCfs())

    def getChildIndex(self, child: CfNode) -> int:
        for idx, entry in enumerate(self.entries):
            if entry is child:
                return idx

        if child is self:
            raise Exception("Child is the same as self.")
        else:
            raise Exception("No Child found.")

    def getClosestChildIndex(self, vector: ClusteringFeature) -> int:
        minDistance = Tensor([birch.Birch.distanceMax])
        index = None

        for idx, cf in enumerate(self.cfs):
            distance = cf.calcualteDistance(vector)

            if distance < minDistance:
                minDistance = distance
                index = idx

        return index

    def getClosestChild(self, cf: ClusteringFeature) -> CfNode:
        return self.entries[self.getClosestChildIndex(cf)]
