import torch
from torch import Tensor

from birch import Birch
from distance import Distance


class ClusteringFeature:
    def __init__(self, tensor: Tensor = None, idx: int = None):
        if tensor is None:
            self.N = 0
            self.SS = None
            self.LS = None
        else:
            self.N = 1
            self.SS = tensor
            self.LS = tensor * tensor
        if idx is None:
            self.indices = []
        else:
            self.indices = [idx]

    def absorbCf(self, cf):
        self.N += cf.N
        self.indices.extend(cf.indices)
        if self.LS is None:
            self.LS = cf.LS
        else:
            self.LS += cf.LS
        if self.SS is None:
            self.SS = cf.SS
        else:
            self.SS *= cf.SS

    def getCenter(self) -> Tensor:
        return self.LS / self.N

    def calcualteDistance(self, vector) -> Tensor:
        if self.LS is None:
            return Tensor([Birch.distanceMax - 100])
        else:
            return Distance.getCosineDistance(self.getCenter(), vector.getCenter())

    def canAbsorbCf(self, cf) -> bool:
        if self.LS is None:
            return True

        distance = Distance.getCosineDistance(self.getCenter(), cf.getCenter())
        return distance <= Birch.threshold
