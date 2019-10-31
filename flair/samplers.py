import logging
from abc import abstractmethod
from collections import defaultdict

from torch.utils.data.sampler import Sampler
import random, torch

import itertools

import flair
from flair.data import FlairDataset

log = logging.getLogger("flair")


class FlairSampler(Sampler):
    def set_dataset(self, data_source):
        """Initialize by passing a block_size and a plus_window parameter.
        :param data_source: dataset to sample from
        """
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def __len__(self):
        return self.num_samples


class ImbalancedClassificationDatasetSampler(FlairSampler):
    """Use this to upsample rare classes and downsample common classes in your unbalanced classification dataset.
    """

    def __init__(self):
        super(ImbalancedClassificationDatasetSampler, self).__init__(None)

    def set_dataset(self, data_source: FlairDataset):
        """
        Initialize by passing a classification dataset with labels, i.e. either TextClassificationDataSet or
        :param data_source:
        """
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.indices = list(range(len(data_source)))

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


class ChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling. This causes some order of the data to be preserved,
    while still shuffling the data.
    """

    def __init__(self, block_size=5, plus_window=5):
        super(ChunkSampler, self).__init__(None)
        self.block_size = block_size
        self.plus_window = plus_window
        self.data_source = None

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


class ExpandingChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling. Block size grows with each epoch.
    This causes some order of the data to be preserved, while still shuffling the data.
    """

    def __init__(self, step=3):
        """Initialize by passing a block_size and a plus_window parameter.
        :param data_source: dataset to sample from
        """
        super(ExpandingChunkSampler, self).__init__(None)
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


class HardNegativesSampler(Sampler):

    def __init__(self,
                 model,
                 chunk_size=4096,
                 p_t=1e-2,
                 n_neg_per_pos=7,
                 batch_size=64):
        self.model = model
        self.chunk_size = chunk_size
        self.p_t = p_t
        self.n_neg_per_pos = n_neg_per_pos
        self.batch_size = batch_size

        self.chunk_begin = 0

    def set_dataset(self, data_source):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.indices = torch.randperm(len(self.data_source))

    def __sample_batches(self, similarity_loss_matrix):
        chunk_size = similarity_loss_matrix.shape[0]

        nonzero_loss = (similarity_loss_matrix > 0).float()
        # mean of non-zero elements, that are assumed to be distributed as exponential
        # TODO: handle it as a mixture of delta at 0 and exponential
        means = similarity_loss_matrix.sum(dim=1) / nonzero_loss.sum(dim=1)
        means = means.unsqueeze(dim=1)

        similarity_loss_matrix = means * torch.exp(-means * similarity_loss_matrix)
        similarity_loss_matrix = similarity_loss_matrix * (similarity_loss_matrix >= self.p_t).float() + \
                                 self.p_t * (similarity_loss_matrix < self.p_t).float()
        similarity_loss_matrix = 1. / similarity_loss_matrix
        similarity_loss_matrix *= nonzero_loss

        indices = []
        visited_points = set([])
        for row_id in range(chunk_size):
            similarity_loss_matrix_row = similarity_loss_matrix[row_id]
            if row_id not in visited_points:
                # in theory, we could add this only if there is any loss incured for this point,
                # here we add it in any case, to ensure we're exactly the permutation of what we'd have
                # if we did random sampling
                sampled_indices = [torch.tensor(row_id)]
                similarity_loss_matrix[:,row_id] = 0
                for _ in range(self.n_neg_per_pos):
                    similarity_loss_matrix_row_sum = similarity_loss_matrix_row.sum()
                    if similarity_loss_matrix_row_sum == 0:
                        break
                    similarity_loss_matrix_row /= similarity_loss_matrix_row_sum
                    cumsum = torch.cumsum(similarity_loss_matrix_row, dim=0)
                    cumsum[-1] = 1.0
                    sampled_index = (cumsum - torch.rand(1) <=0).sum()
                    sampled_indices.append(sampled_index)
                    similarity_loss_matrix[:,sampled_index] = 0
                sampled_indices = torch.stack(sampled_indices)
                indices.append(sampled_indices)
                visited_points.update(set(sampled_indices.tolist()))
        # for _ in range(self.n_neg_per_pos):
        #     similarity_loss_matrix /= similarity_loss_matrix.sum(dim=1, keepdim=True)
        #     cumsum = torch.cumsum(similarity_loss_matrix, dim=1)
        #     cumsum[:,-1] = 1.0  # this is neccessary to fix numerical problems
        #     sampled_indices = torch.sum(cumsum - torch.rand(chunk_size, 1) <= 0, dim=1)
        #     similarity_loss_matrix[torch.arange(chunk_size), sampled_indices] = 0
        #     indices.append(sampled_indices)
        # indices = torch.stack(indices)
        # indices = torch.cat([torch.arange(chunk_size).unsqueeze(0), indices], dim=0).T
        # indices = indices.flatten()

        indices = torch.cat(indices)

        return indices

    def __mine_batches(self):
        # == 1: take a chunk of data points (chunk is much bigger than a batch) ==
        chunk_data = [self.data_source[i] for i in self.chunk_indices]

        with torch.no_grad():
            # == 2: embed each modality ==
            chunk_borders = list(range(0, len(self.chunk_indices), self.batch_size))
            if chunk_borders[-1] != len(self.chunk_indices):
                chunk_borders.append(len(self.chunk_indices))
            batched_chunk_data = [chunk_data[batch_begin:batch_end] for batch_begin, batch_end in zip(chunk_borders[:-1], chunk_borders[1:])]
            chunk_mapped_source_embeddings = []
            chunk_mapped_target_embeddings = []
            for batch in batched_chunk_data:
                chunk_mapped_source_embeddings.append(self.model._embed_source(batch))
                chunk_mapped_target_embeddings.append(self.model._embed_target(batch))
            chunk_mapped_source_embeddings = torch.cat(chunk_mapped_source_embeddings, dim=0).cpu()
            chunk_mapped_target_embeddings = torch.cat(chunk_mapped_target_embeddings, dim=0).cpu()

            # == 3: compute the loss on the whole chunk given the current model ==
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

            targets = torch.zeros_like(similarity_matrix).to('cpu') # flair.device)

            for data_point in chunk_data:
                first_indices = index_map['first'][str(data_point.first)]
                second_indices = index_map['second'][str(data_point.second)]
                for first_index, second_index in itertools.product(first_indices, second_indices):
                    targets[first_index, second_index] = 1.

            similarity_loss_matrix = self.model.similarity_loss(similarity_matrix, targets)
            # We're sampling just positive-negative DataPairs
            similarity_loss_matrix *= torch.ones_like(similarity_loss_matrix) - targets

        similarity_loss_matrix = similarity_loss_matrix.detach() # .cpu()
        # print('Computed the loss sampling matrix')

        # == 4: sample the pairs based on loss, i.e. convert loss into sampling probs ==
        sampled_indices = self.chunk_indices[self.__sample_batches(similarity_loss_matrix)]

        return sampled_indices

    def __iter__(self):
        chunk_end = self.chunk_begin + self.chunk_size
        chunk_end = len(self.data_source) if chunk_end > len(self.data_source) else chunk_end
        if (chunk_end - self.chunk_begin) < 2:
            self.chunk_begin = 0
            chunk_end = self.chunk_begin + self.chunk_size
            self.indices = torch.randperm(len(self.data_source))

        self.chunk_indices = self.indices[self.chunk_begin:chunk_end]
        self.chunk_begin = chunk_end

        self.__mine_batches()

        sampled_indices = self.__mine_batches()

        return iter(sampled_indices)

    def __len__(self):
        return self.num_samples
