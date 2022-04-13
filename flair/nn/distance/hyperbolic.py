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

Source: https://github.com/asappresearch/dynamic-classification/blob/55beb5a48406c187674bea40487c011e8fa45aab/distance/hyperbolic.py
"""

import torch
import torch.nn as nn
from torch import Tensor

EPSILON = 1e-5


def arccosh(x):
    """Compute the arcosh, numerically stable."""
    x = torch.clamp(x, min=1 + EPSILON)
    a = torch.log(x)
    b = torch.log1p(torch.sqrt(x * x - 1) / x)
    return a + b


def mdot(x, y):
    """Compute the inner product."""
    m = x.new_ones(1, x.size(1))
    m[0, 0] = -1
    return torch.sum(m * x * y, 1, keepdim=True)


def dist(x, y):
    """Get the hyperbolic distance between x and y."""
    return arccosh(-mdot(x, y))


def project(x):
    """Project onto the hyeprboloid embedded in in n+1 dimensions."""
    return torch.cat([torch.sqrt(1.0 + torch.sum(x * x, 1, keepdim=True)), x], 1)


def log_map(x, y):
    """Perform the log step."""
    d = dist(x, y)
    return (d / torch.sinh(d)) * (y - torch.cosh(d) * x)


def norm(x):
    """Compute the norm"""
    n = torch.sqrt(torch.abs(mdot(x, x)))
    return n


def exp_map(x, y):
    """Perform the exp step."""
    n = torch.clamp(norm(y), min=EPSILON)
    return torch.cosh(n) * x + (torch.sinh(n) / n) * y


def loss(x, y):
    """Get the loss for the optimizer."""
    return torch.sum(dist(x, y) ** 2)


class HyperbolicDistance(nn.Module):
    """Implement a HyperbolicDistance object."""

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
        # Get projected 1st dimension
        mat_1_x_0 = torch.sqrt(1 + mat_1.pow(2).sum(dim=1, keepdim=True))
        mat_2_x_0 = torch.sqrt(1 + mat_2.pow(2).sum(dim=1, keepdim=True))

        # Compute bilinear form
        left = mat_1_x_0.mm(mat_2_x_0.t())  # n_1 x n_2
        right = mat_1[:, 1:].mm(mat_2[:, 1:].t())  # n_1 x n_2

        # Arcosh
        return arccosh(left - right).pow(2)


class HyperbolicMean(nn.Module):
    """Compute the mean point in the hyperboloid model."""

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
        n_iter = 5 if self.training else 100

        # Project the input data to n+1 dimensions
        projected = project(data)

        mean = torch.mean(projected, 0, keepdim=True)
        mean = mean / norm(mean)

        r = 1e-2
        for i in range(n_iter):
            g = -2 * torch.mean(log_map(mean, projected), 0, keepdim=True)
            mean = exp_map(mean, -r * g)
            mean = mean / norm(mean)

        # The first dimension, is recomputed in the distance module
        return mean.squeeze()[1:]
