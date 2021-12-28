import torch
from torch import Tensor


def getCosineDistance(vector1: Tensor, vector2: Tensor) -> Tensor:
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return 1 - cos(vector1, vector2)


def getFurthest2Points(cfs: list) -> list:
    candidates = []

    for index, cf in enumerate(cfs):
        distances = list(map(lambda e: e.calcualteDistance(cf).item(), cfs))
        if index != int(max(distances)):
            candidates.append([index, int(max(distances))])

    distances = list(map(lambda e: e.calcualteDistance(cf).item(), cfs))
    return candidates[int(max(distances))]
