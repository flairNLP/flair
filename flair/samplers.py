import logging
from collections import defaultdict

from torch.utils.data.sampler import Sampler
import random, torch

import flair
import itertools
import numpy as np
import sklearn.cluster

from flair.data import FlairDataset

log = logging.getLogger("flair")


class ImbalancedClassificationDatasetSampler(Sampler):
    """Use this to upsample rare classes and downsample common classes in your unbalanced classification dataset.
    """

    def __init__(self, data_source: FlairDataset):
        """
        Initialize by passing a classification dataset with labels, i.e. either TextClassificationDataSet or
        :param data_source:
        """
        super().__init__(data_source)

        self.indices = list(range(len(data_source)))
        self.num_samples = len(data_source)

        # first determine the distribution of classes in the dataset
        label_count = defaultdict(int)
        for sentence in data_source:
            for label in sentence.get_label_names():
                label_count[label] += 1

        # weight for each sample
        offset = 0
        weights = [
            1.0 / (offset + label_count[data_source[idx].get_label_names()[0]])
            for idx in self.indices
        ]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class ChunkSampler(Sampler):
    """Splits data into blocks and randomizes them before sampling. This causes some order of the data to be preserved,
    while still shuffling the data.
    """

    def __init__(self, data_source, block_size=5, plus_window=5):
        """Initialize by passing a block_size and a plus_window parameter.
        :param data_source: dataset to sample from
        :param block_size: minimum size of each block
        :param plus_window: randomly adds between 0 and this value to block size at each epoch
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(self.data_source)

        self.block_size = block_size
        self.plus_window = plus_window

    def __iter__(self):
        data = list(range(len(self.data_source)))

        blocksize = self.block_size + random.randint(0, self.plus_window)

        log.info(
            f"Chunk sampling with blocksize = {blocksize} ({self.block_size} + {self.plus_window})"
        )

        # Create blocks
        blocks = [data[i : i + blocksize] for i in range(0, len(data), blocksize)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]
        return iter(data)

    def __len__(self):
        return self.num_samples


class ExpandingChunkSampler(Sampler):
    """Splits data into blocks and randomizes them before sampling. Block size grows with each epoch.
    This causes some order of the data to be preserved, while still shuffling the data.
    """

    def __init__(self, data_source, step=3):
        """Initialize by passing a block_size and a plus_window parameter.
        :param data_source: dataset to sample from
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(self.data_source)

        self.block_size = 1
        self.epoch_count = 0
        self.step = step

    def __iter__(self):
        self.epoch_count += 1

        data = list(range(len(self.data_source)))

        log.info(f"Chunk sampling with blocksize = {self.block_size}")

        # Create blocks
        blocks = [
            data[i : i + self.block_size] for i in range(0, len(data), self.block_size)
        ]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]

        if self.epoch_count % self.step == 0:
            self.block_size += 1

        return iter(data)

    def __len__(self):
        return self.num_samples


def sample_multinomial(parms):

    return torch.sum(torch.cumsum(parms, dim=0).to(flair.device) - torch.rand(1,).to(flair.device) < 0).item()


class HardNegativeIterator:

    def __init__(self,
                 dataset,
                 model,
                 chunk_size=4096,
                 batch_size=128,
                 clique_size=8,
                 n_batches=128,
                 cluster_sampling_probabilities=[0.35, 0.5, 0.15]):
        self.dataset = dataset
        self.model = model
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.clique_size = clique_size
        self.n_batches = n_batches

        self.chunk_start = 0
        self.indices = torch.arange(start=0, end=len(self.dataset))
        self.indices = self.indices[torch.randperm(len(self.dataset))]

        self.cluster_by = 'both'
        self.cluster_sampling_probabilities = torch.Tensor(cluster_sampling_probabilities)
        self.n_clusters = self.cluster_sampling_probabilities.shape[0]

        self.__mine_batches()


    def __sample(self, cluster_indices, clusters_sorted, sampling_matrix, batch_indices, clique_indices):
        # if clique indices is empty then sample cluster, and then sample a pair from the cluster based sampling matrix
        cluster_id = clusters_sorted[sample_multinomial(self.cluster_sampling_probabilities)]

        if clique_indices:
            # if clique is not empty sampled based on sampling multinomial that is computed from rows
            # corresponding to clique elements in sampling matrix
            sampling_multinomial = sampling_matrix[torch.LongTensor(clique_indices)].sum(dim=0)
            sampling_multinomial[torch.LongTensor(clique_indices + batch_indices)] = 0
            sampling_indices = torch.LongTensor(np.where(cluster_indices == cluster_id)[0])
            sampling_multinomial = sampling_multinomial[sampling_indices]
        else:
            # if the clique is empty sample based on the p_x
            sampling_indices = torch.LongTensor(np.setdiff1d(np.where(cluster_indices == cluster_id)[0], batch_indices))
            sampling_multinomial = sampling_matrix[sampling_indices].sum(dim=1)

        if sampling_multinomial.sum() > 0:
            retval = sampling_indices[sample_multinomial(sampling_multinomial / sampling_multinomial.sum())].item()
        else:
            retval = None

        return retval

    def __mine_batches(self):
        # GENERAL IDEA:
        # 1. take a chunk of data points (chunk is much bigger than a batch)
        # 2. embed each modality
        # 3. compute the loss on the whole chunk given the current model
        # 4. sample the pairs based on loss, i.e. convert loss into sampling probs

        # == 1 ==
        chunk_end = self.chunk_start + self.chunk_size
        chunk_end = chunk_end if chunk_end < len(self.dataset) else len(self.dataset)
        chunk_indices = self.indices[self.chunk_start:chunk_end]
        chunk_data = [self.dataset[i] for i in chunk_indices]

        # print(f'Start: {self.chunk_start}, End: {chunk_end}')

        # == 2 ==
        with torch.no_grad():
            chunk_mapped_source_embeddings = self.model._embed_source(chunk_data)
            chunk_mapped_target_embeddings = self.model._embed_target(chunk_data)

            # == 3 ==
            similarity_matrix = self.model.similarity_measure.forward(
                (chunk_mapped_source_embeddings, chunk_mapped_target_embeddings)
            )

            # part from SimilarityLearner.forward_loss
            def add_to_index_map(hashmap, key, val):
                if key not in hashmap:
                    hashmap[key] = [val]
                else:
                    hashmap[key] += [val]

            index_map = {'first': {}, 'second': {}}
            for data_point_id, data_point in enumerate(chunk_data):
                add_to_index_map(index_map['first'], str(data_point.first), data_point_id)
                add_to_index_map(index_map['second'], str(data_point.second), data_point_id)

            targets = torch.zeros_like(similarity_matrix).to(flair.device)

            for data_point in chunk_data:
                first_indices = index_map['first'][str(data_point.first)]
                second_indices = index_map['second'][str(data_point.second)]
                for first_index, second_index in itertools.product(first_indices, second_indices):
                    targets[first_index, second_index] = 1.

            similarity_loss_matrix = self.model.similarity_loss(similarity_matrix, targets)
            n_cols = similarity_loss_matrix.shape[1]
            similarity_loss_matrix *= torch.ones_like(similarity_loss_matrix) - targets

        similarity_loss_matrix = similarity_loss_matrix.detach().cpu()

        # == 4 ===
        # this is the most critical part
        # Ideas:
        # strategy a) take pairs with top loss
        # strategy b) for each row take x pairs with highest loss
        # strategy c) for each row sample x pairs depending on the loss,
        # the c) can have two parameters: threshold and clip/threshold
        # the threshold is a number
        # the clip/threshold is an option to clip the loss values to a number or to set loss above threshold to 0
        # here we implement a slightly more complex strategy:

        alpha = 1.0
        sampling_matrix = 1.0 - torch.exp(-alpha * similarity_loss_matrix)
        sampling_matrix /= sampling_matrix.sum()

        p_y_x = sampling_matrix / sampling_matrix.sum(dim=1, keepdim=True)
        p_x = sampling_matrix.sum(dim=1)
        p_y_x[p_x == 0] = 0
        H_y_x = -p_y_x * torch.log(p_y_x)
        H_y_x[p_y_x == 0] = 0
        H_y_x = H_y_x.sum(dim=1)

        if self.cluster_by == 'p_x':
            cluster_vals = torch.unsqueeze(p_x, dim=0)
        elif self.cluster_by == 'H_y_x':
            cluster_vals = torch.unsqueeze(H_y_x, dim=0)
        elif self.cluster_by == 'both':
            cluster_vals = torch.stack([p_x, H_y_x]).t()

        cluster_vals -= cluster_vals.mean(dim=0, keepdim=True)
        cluster_vals /= cluster_vals.std(dim=0, keepdim=True)

        clusterer = sklearn.cluster.KMeans(self.n_clusters)
        cluster_indices = clusterer.fit_predict(cluster_vals.numpy())
        clusters_sorted = np.argsort(clusterer.cluster_centers_[:,0])

        batches = []
        while len(batches) < self.n_batches:
            batch_indices = []
            backoff_counter = 0
            while len(batch_indices) < self.batch_size:
                # print(batch_indices)
                # print(backoff_counter)
                clique_indices = []
                first_index = self.__sample(cluster_indices, clusters_sorted, sampling_matrix, batch_indices, clique_indices)
                if first_index:
                    clique_indices.append(first_index)
                    while len(clique_indices) < self.clique_size:
                        clique_member_index = self.__sample(cluster_indices, clusters_sorted, sampling_matrix, batch_indices, clique_indices)
                        if clique_member_index:
                            clique_indices.append(clique_member_index)
                        else:
                            break
                    if len(clique_indices) > 1:
                        if len(clique_indices) + len(batch_indices) <= self.batch_size:
                            batch_indices.extend(clique_indices)
                        else:
                            break
                    else:
                        backoff_counter += 1
                else:
                    backoff_counter += 1
                if backoff_counter > 10:
                    break
            if batch_indices:
                # remove all pairs of sampled batch indices from the sampling matrix
                batch_pairs = torch.LongTensor([[i1, i2] for i1 in batch_indices for i2 in batch_indices if i1 != i2])
                sampling_matrix_update = torch.ones_like(sampling_matrix)
                sampling_matrix_update[batch_pairs[:, 0], batch_pairs[:, 1]] = 0
                sampling_matrix *= sampling_matrix_update
                batches.append([chunk_indices[i].item() for i in batch_indices])
            else:
                break

        self.batches_iter = itertools.chain(*batches)

    def __iter__(self):
        return self

    def __next__(self):
        return self.batches_iter.__next__()

    def __len__(self):
        return len(self.dataset)


class HardNegativesSampler(Sampler):

    def __init__(self,
                 dataset,
                 model,
                 chunk_size=4096,
                 batch_size=128,
                 clique_size=8,
                 n_batches=256,
                 cluster_sampling_probabilities=[0.35, 0.5, 0.15]):
        self.dataset = dataset
        self.model = model
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.clique_size = clique_size
        self.n_batches = n_batches
        self.cluster_sampling_probabilities = cluster_sampling_probabilities

    def __iter__(self):
        return HardNegativeIterator(self.dataset,
                                    self.model,
                                    self.chunk_size,
                                    self.batch_size,
                                    self.clique_size,
                                    self.n_batches,
                                    self.cluster_sampling_probabilities)

    def __len__(self):
        return len(self.dataset)
