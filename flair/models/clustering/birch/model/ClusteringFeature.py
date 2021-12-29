import torch
from torch import Tensor

from flair.models.clustering.birch import distance_max, threshold
from flair.models.clustering.distance import Distance


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

    def absorb_cf(self, cf):
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

    def get_center(self) -> Tensor:
        return self.LS / self.N

    def calculate_distance(self, vector) -> Tensor:
        if self.LS is None:
            return torch.empty(1, device="cuda").fill_(distance_max - 100)
        else:
            return Distance.get_cosine_distance(self.get_center(), vector.get_center())

    def can_absorb_cf(self, cf) -> bool:
        if self.LS is None:
            return True

        distance = Distance.get_cosine_distance(self.get_center(), cf.get_center())
        return distance <= threshold
