import logging
from collections import defaultdict

from torch.utils.data.sampler import Sampler
import random, torch

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
