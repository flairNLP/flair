import logging
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler

import random

import numpy as np

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
            for label in sentence.labels:
                label_count[label.value] += 1

        # weight for each sample
        offset = 0
        weights = [
            1.0 / (offset + label_count[data_source[idx].labels[0].value])
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

# == similarity batch sampler helpers ==


def sample_multinomial(parms):
    return ((np.cumsum(parms) - np.random.random()) < 0).sum()


def sample_columns(matrix, squashing_function, threshold, topk):
    cols = []
    row_thresholds = []
    for row_id in range(matrix.shape[0]):
        row = matrix[row_id]
        over_threshold_indicators = row >= threshold
        n_over_threshold = over_threshold_indicators.sum()
        if n_over_threshold > 0:
            # If we can sample some positive pairs
            vals_indices = over_threshold_indicators.nonzero()[1]
            vals = np.array(row[over_threshold_indicators])[0]
            row_topk = np.min([n_over_threshold, topk])
            row_threshold = np.max([np.sort(vals)[-row_topk], threshold])
            row_thresholds.append(row_threshold)
            vals = squashing_function(vals)
            sampled_vals_indices = sample_multinomial(vals / vals.sum())
            cols.append(vals_indices[sampled_vals_indices])
        else:
            # ... otherwise just sample randomly, it's gonna be either
            # negative (zero score) or ignored (non-zero score,
            # but under the threshold)
            row_thresholds.append(threshold)
            cols.append(np.random.choice(matrix.shape[1]))

    return np.array(cols), np.array(row_thresholds)[None].T


class SimilarityBatchSampler:

    def __init__(self,
                 data_source,
                 batch_size,
                 shuffle,
                 squashing_function,
                 score_threshold,
                 topk):
        self.source_dataset = data_source.datasets[0]
        self.destination_dataset = data_source.datasets[1]
        self.score_matrix = data_source.score_tensor
        # batch sampling parms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.squashing_function = squashing_function
        self.score_threshold = score_threshold
        self.topk = topk
        #
        self.n_source = len(self.source_dataset)
        self.source_indices = list(range(self.n_source))
        if self.shuffle:
            random.shuffle(self.source_indices)
        self.batch_begin = 0
        self.end_of_epoch = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_of_epoch:
            raise StopIteration
        else:
            batch_end = self.batch_begin + self.batch_size
            if batch_end >= self.n_source:
                batch_end = self.n_source
                self.end_of_epoch = True
            else:
                self.end_of_epoch = False
            batch_source_ids = self.source_indices[self.batch_begin:batch_end]
            if len(batch_source_ids) < 2:
                raise StopIteration
            score_matrix_slice = self.score_matrix[np.array(batch_source_ids)]
            # sample one column per row
            batch_destination_ids, score_thresholds = sample_columns(score_matrix_slice, self.squashing_function, self.score_threshold, self.topk)
            # determine label matrix
            batch_label_matrix = score_matrix_slice[:, batch_destination_ids].toarray()
            batch_label_matrix[batch_label_matrix == 0] = -1
            batch_label_matrix[np.logical_and(batch_label_matrix > 0, batch_label_matrix < score_thresholds)] = 0
            batch_label_matrix[batch_label_matrix >= score_thresholds] = 1

            batch_sources = [self.source_dataset[source_id] for source_id in batch_source_ids]
            batch_destinations = [self.destination_dataset[destination_id] for destination_id in batch_destination_ids]
            batch_label_matrix = torch.from_numpy(batch_label_matrix).to(flair.device)

            self.batch_begin = batch_end

            return batch_sources, batch_destinations, batch_label_matrix

    def __len__(self):
        return self.n_source // self.batch_size + int(self.n_source % self.batch_size != 0)