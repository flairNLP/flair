import logging
from typing import Union, List

import numpy as np
import torch
from torch import logsumexp, Tensor
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings

from .base import ClusteringModel
from flair.data import Sentence, Corpus

log = logging.getLogger("flair")


class ExpectationMaximization(ClusteringModel):

    def __init__(self, k: int, embeddings: DocumentEmbeddings, corpus: Corpus):
        self.embeddings = embeddings
        self.corpus = corpus

        self.models = []
        self.lower_log_bounds = []
        self.k = k
        self.data_points = 0
        self.dimension = 0
        self.pi = []
        self.probabilities = []

    def fit(self, max_iter: int = 100, batch_size: int = 64, r_tol: float = 1e-3):

        self._embed(self.corpus.train, batch_size=batch_size)
        vectors = torch.stack([sentence.embedding for sentence in self.corpus.train])

        self.init_params_random(vectors)
        prev_lls = None
        data = torch.stack([i.embedding for i in vectors])

        for i in range(max_iteration):
            self.e_step(data)
            self.m_step(data)
            current_lls = self.calculate_lower_bound(data)

            if prev_lls and torch.abs((current_lls - prev_lls) / prev_lls) < r_tol:
                print(self.lower_log_bounds)
                log.debug("Finished the EM Clustering with: " + str(i) + " iterations. ")
                return self.probabilities

            prev_lls = current_lls
            self.lower_log_bounds.append(current_lls)

        log.debug("Finished the EM Clustering with maxItertation: " + str(max_iteration))
        return self.probabilities

    def _embed(self, sentences: Union[List[Sentence], Sentence], batch_size: int):
        log.info("Embed sentences...")
        for batch in tqdm(DataLoader(sentences, batch_size=batch_size)):
            self.embeddings.embed(batch)

    def init_params_random(self, vectors: list):
        self.data_points = len(vectors)
        self.dimension = len(vectors[0].embedding)
        self.pi = torch.empty(self.k, device="cuda").fill_(1.0 / self.k).log()
        self.probabilities = torch.empty((self.data_points, self.k), device="cuda").fill_(0)

        idxs = torch.from_numpy(np.random.choice(self.data_points, self.k, replace=False))
        means = [vectors[i].embedding for i in idxs]
        variance = torch.eye(self.dimension, device="cuda")
        for i in range(self.k):
            self.models.append(MultivariateNormal(means[i], variance))

    def e_step(self, data: Tensor):
        for idx, model in enumerate(self.models):
            ll_hood = model.log_prob(data)
            ll_hood_weight = ll_hood * self.pi[idx]
            self.probabilities[:, idx] = ll_hood_weight - logsumexp(ll_hood_weight, dim=0, keepdim=True)

        self.probabilities /= self.probabilities.sum(axis=1, keepdims=True)

    def m_step(self, data: Tensor):

        for i in range(self.k):
            # probability of points to belong to cluster c
            r_c = (
                self.probabilities[:, i]
                .reshape(self.data_points, 1)
                .repeat_interleave(self.dimension)
                .reshape(self.data_points, self.dimension)
            )

            # total responsibility allocated to cluster i
            m_c = torch.sum(self.probabilities[:, i])

            # fraction of total assigned to cluster i
            self.pi[i] = m_c / self.data_points

            data_with_weights = data * r_c
            # weighted mean of assigned data
            mu = data_with_weights.sum(dim=0) / m_c

            # weighted covariance of assigned data
            var = (data - mu).t() @ ((data - mu) * r_c) / m_c

            self.models[i] = MultivariateNormal(mu, scale_tril=var, validate_args=False)

    def calculate_lower_bound(self, data):
        ll = torch.empty(self.data_points, self.k, device="cuda")

        for i in range(self.k):
            ll[:, i] = self.models[i].log_prob(data)
        return torch.log(torch.sum(ll * self.pi))

    def get_discrete_result(self):
        return torch.max(self.probabilities, dim=1).indices.cpu().detach().numpy()
