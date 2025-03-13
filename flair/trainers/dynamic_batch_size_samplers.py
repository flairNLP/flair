from collections import defaultdict
from torch.utils.data import BatchSampler
import torch.distributed
import logging

logger = logging.getLogger("flair")


NUM_STEPS_PER_BATCH = 16

class SingleLengthBatchSampler(BatchSampler):
    def __init__(self, max_tokens_per_batch_step=4096, min_sentences_per_batch_step=1):
        self.max_tokens_per_batch_step = max_tokens_per_batch_step
        self.min_sentences_per_batch_step = min_sentences_per_batch_step
        self.batches = []
 
    def set_dataset(self, dataset):
        self.dataset = dataset
        self._generate_batches()

    def _generate_batches(self):
        length_buckets = defaultdict(list)
        for idx, datapoint in enumerate(self.dataset):
            length = len(datapoint)
            length_buckets[length].append(idx)

        sorted_lengths = sorted(length_buckets.keys())
        self.batches = []
        for length in sorted_lengths:
            num_sentences_per_batch_step = max(self.max_tokens_per_batch_step // length, 1)
            if num_sentences_per_batch_step < self.min_sentences_per_batch_step:
                logger.warning(f"all datapoints of at least {length} tokens will be skipped")
                break

            all_indices = length_buckets[length]
            num_sentences = len(all_indices)

            num_sentences_per_batch = num_sentences_per_batch_step * NUM_STEPS_PER_BATCH
            for i in range(0, num_sentences, num_sentences_per_batch):
                batch_indices = all_indices[i:i + num_sentences_per_batch]
                if len(batch_indices) >= self.min_sentences_per_batch_step * NUM_STEPS_PER_BATCH:
                    self.batches.append(batch_indices)
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


class MultiGPUSingleLengthBatchSampler(BatchSampler):
    def __init__(self, max_tokens_per_batch_step=4096, min_sentences_per_batch_step=1):
        self.max_tokens_per_batch_step = max_tokens_per_batch_step
        self.min_sentences_per_batch_step = min_sentences_per_batch_step
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.batches = []

    def set_dataset(self, dataset):
        self.dataset = dataset
        self._generate_batches()

    def _generate_batches(self):
        length_buckets = defaultdict(list)
        for idx, datapoint in enumerate(self.dataset):
            length = len(datapoint)
            length_buckets[length].append(idx)

        sorted_lengths = sorted(length_buckets.keys())
        self.batches = []
        for length in sorted_lengths:
            num_sentences_per_batch_step = max(self.max_tokens_per_batch_step // length, 1)
            if num_sentences_per_batch_step < self.min_sentences_per_batch_step:
                logger.warning(f"all datapoints of at least {length} tokens will be skipped")
                break

            indices_all_GPU = length_buckets[length]
            num_sentences_all_GPU = len(indices_all_GPU)
            num_sentences_per_GPU = num_sentences_all_GPU // self.num_replicas # drop last few sentences
            indices_this_GPU = indices_all_GPU[self.rank : num_sentences_per_GPU * self.num_replicas : self.num_replicas]

            num_sentences_per_batch = num_sentences_per_batch_step * NUM_STEPS_PER_BATCH
            for i in range(0, num_sentences_per_GPU, num_sentences_per_batch):
                batch_indices = indices_this_GPU[i:i + num_sentences_per_batch]
                if len(batch_indices) >= self.min_sentences_per_batch_step * NUM_STEPS_PER_BATCH:
                    self.batches.append(batch_indices)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)