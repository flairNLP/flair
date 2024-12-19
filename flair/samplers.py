import logging
import random
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler

log = logging.getLogger("flair")


class FlairSampler(Sampler):
    def set_dataset(self, data_source):
        """Initialize the data source for the FlairSampler.

        Args:
            data_source: dataset to sample from.
        """
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def __len__(self) -> int:
        return self.num_samples


class ImbalancedClassificationDatasetSampler(FlairSampler):
    """Use this to upsample rare classes and downsample common classes in your unbalanced classification dataset."""

    def __init__(self) -> None:
        super().__init__(None)

    def set_dataset(self, data_source):
        """Initialize the dataset used for sampling."""
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.indices = list(range(len(data_source)))

        # first determine the distribution of classes in the dataset
        label_count: dict[str, int] = defaultdict(int)
        for sentence in data_source:
            for label in sentence.labels:
                label_count[label.value] += 1

        # weight for each sample
        offset = 0
        weights = [1.0 / (offset + label_count[data_source[idx].labels[0].value]) for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))


class ChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling.

    This causes some order of the data to be preserved, while still shuffling the data.
    """

    def __init__(self, block_size=5, plus_window=5) -> None:
        super().__init__(None)
        self.block_size = block_size
        self.plus_window = plus_window
        self.data_source = None

    def __iter__(self):
        data = list(range(len(self.data_source)))

        blocksize = self.block_size + random.randint(0, self.plus_window)

        log.info(f"Chunk sampling with blocksize = {blocksize} ({self.block_size} + {self.plus_window})")

        # Create blocks
        blocks = [data[i : i + blocksize] for i in range(0, len(data), blocksize)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]
        return iter(data)


class ExpandingChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling.

    Block size grows with each epoch.
    This causes some order of the data to be preserved, while still shuffling the data.
    """

    def __init__(self, step=3) -> None:
        """Initialize the ExpandingChunkSampler.

        Args:
            step: every *step* epochs the block size increments by one.
        """
        super().__init__(None)
        self.block_size = 1
        self.epoch_count = 0
        self.step = step

    def __iter__(self):
        self.epoch_count += 1

        data = list(range(len(self.data_source)))

        log.info(f"Chunk sampling with blocksize = {self.block_size}")

        # Create blocks
        blocks = [data[i : i + self.block_size] for i in range(0, len(data), self.block_size)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]

        if self.epoch_count % self.step == 0:
            self.block_size += 1

        return iter(data)
