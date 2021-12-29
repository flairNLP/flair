import numpy as np
import torch
from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings
from torch import logsumexp, Tensor
from torch.distributions import MultivariateNormal

from flair.models.clustering.clustering import Clustering


class EM_Clustering(Clustering):
    def __init__(self, k: int, embeddings: DocumentEmbeddings):
        self.models = []
        self.lls = []
        self.k = k
        self.embeddings = embeddings
        self.maxIteration = 10

    def cluster(self, vectors: list, batch_size: int = 64, rtol=1e-3):

        try:
            for batch in DataLoader(vectors, batch_size=batch_size):
                self.embeddings.embed(batch)
        except RuntimeError as e:
            print("Please lower the batchsize of the cluster method.")

        self.datapoints = vectors.__len__()
        self.dimension = vectors[0].embedding.__len__()
        idxs = torch.from_numpy(np.random.choice(self.datapoints, self.k, replace=False))
        means = [vectors[i].embedding for i in idxs]
        variance = torch.eye(self.dimension, device="cuda")
        data = torch.stack([i.embedding for i in vectors])
        self.pi = torch.empty(self.k, device="cuda").fill_(1.0 / self.k).log()
        self.z = torch.empty((self.datapoints, self.k), device="cuda").fill_(0)

        for i in range(self.k):
            self.models.append(MultivariateNormal(means[i], variance))

        for i in range(0, self.maxIteration):
            self.e_step(data)
            self.m_step(data)

            self.lls.append(self.lower_bound(data, self.z))

            if (
                self.lls.__len__() >= 2
                and torch.abs(
                    (self.lls[self.lls.__len__() - 1] - self.lls[self.lls.__len__() - 2])
                    / self.lls[self.lls.__len__() - 2]
                )
                < rtol
            ):
                print("Break")
                break

        print(self.lls)
        return []

    def e_step(self, data: Tensor):
        for idx, model in enumerate(self.models):
            llhood = model.log_prob(data)
            llhoodWeight = llhood * self.pi[idx]
            self.z[:, idx] = llhoodWeight - logsumexp(llhoodWeight, dim=0, keepdim=True)

        self.posterior = self.z
        self.z /= self.z.sum(axis=1, keepdims=True)

    def m_step(self, data: Tensor):

        for i in range(self.k):
            m_c = torch.sum(self.z[:, i])
            self.pi[i] = m_c / self.datapoints
            test = data * self.z[:, i].reshape(self.datapoints, 1).repeat_interleave(self.dimension).reshape(
                self.datapoints, self.dimension
            )
            mu = 1 / m_c * test.sum(dim=0)
            var = m_c * (torch.t(data - mu) @ (data - mu))

            self.models[i] = MultivariateNormal(mu, scale_tril=var, validate_args=False)

    def lower_bound(self, data, q):
        ll = torch.empty(self.datapoints, self.k, device="cuda")

        for c in range(self.k):
            ll[:, c] = self.models[c].log_prob(data)
        return torch.sum(q * (ll + torch.log(self.pi) - torch.log(q)))

    def get_diskret_result(self):
        return torch.max(self.z, dim=1).indices.cpu().detach().numpy()
