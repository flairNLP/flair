"""
This module was copied from the repository the following repository:
https://github.com/asappresearch/dynamic-classification

It contains the code from the paper "Metric Learning for Dynamic Text
Classification".

https://arxiv.org/abs/1911.01026

In case this file is modified, please consider contributing to the original
repository.

It was published under MIT License:
https://github.com/asappresearch/dynamic-classification/blob/master/LICENSE.md

Source: https://github.com/asappresearch/dynamic-classification/blob/55beb5a48406c187674bea40487c011e8fa45aab/distance/euclidean.py
"""


import torch
import torch.nn as nn
from torch import Tensor


class EuclideanDistance(nn.Module):
    """Implement a EuclideanDistance object."""

    def forward(self, mat_1: Tensor, mat_2: Tensor) -> Tensor:  # type: ignore
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        _dist = [torch.sum((mat_1 - mat_2[i]) ** 2, dim=1) for i in range(mat_2.size(0))]
        dist = torch.stack(_dist, dim=1)
        return dist


class EuclideanMean(nn.Module):
    """Implement a EuclideanMean object."""

    def forward(self, data: Tensor) -> Tensor:  # type: ignore
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        return data.mean(0)
