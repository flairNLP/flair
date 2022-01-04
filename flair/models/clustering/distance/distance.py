import torch
from torch import tensor


def get_cosine_distance(vector1: tensor, vector2: tensor) -> tensor:
    cos = torch.nn.functional.cosine_similarity(vector1, vector2, dim=0)
    return 1 - cos


def get_furthest_2_points(cfs: list) -> list:
    if len(cfs) == 2:
        return [0, 1]
    candidates = []

    for index, cf in enumerate(cfs):
        distances = list(map(lambda e: e.calculate_distance(cf).item(), cfs))
        if index != int(max(distances)):
            candidates.append([index, int(max(distances))])

    distances = list(map(lambda e: e.calculate_distance(cf).item(), cfs))
    return candidates[int(max(distances))]
