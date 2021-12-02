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

Source: https://github.com/asappresearch/dynamic-classification/blob/55beb5a48406c187674bea40487c011e8fa45aab/distance/__init__.py
"""


from .euclidean import EuclideanDistance, EuclideanMean
from .hyperbolic import HyperbolicDistance, HyperbolicMean
from .cosine import CosineDistance, LogitCosineDistance
from .euclidean import EuclideanMean as CosineMean

__all__ = ['EuclideanDistance', 'EuclideanMean', 'HyperbolicDistance', 'HyperbolicMean', 'CosineDistance', 'LogitCosineDistance', 'CosineMean']
