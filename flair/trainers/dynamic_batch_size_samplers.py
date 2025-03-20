from collections import defaultdict
from torch.utils.data import BatchSampler
import torch.distributed
import logging
import pickle
import torch
import boto3
import random
from typing import Optional

logger = logging.getLogger("flair")


class SingleLengthBatchSampler(BatchSampler):
    def __init__(
        self, 
        max_tokens_per_batch_step: int = 4096,
        max_sentences_per_batch_step: int = 128,
        min_sentences_per_batch_step: int = 2,
        skip_ratio: Optional[float] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.max_tokens_per_batch_step = max_tokens_per_batch_step
        self.max_sentences_per_batch_step = max_sentences_per_batch_step
        self.min_sentences_per_batch_step = min_sentences_per_batch_step
        self.skip_ratio = skip_ratio
        logger.warning(f"max_tokens_per_batch_step: {max_tokens_per_batch_step}")
        logger.warning(f"max_sentences_per_batch_step: {max_sentences_per_batch_step}")
        logger.warning(f"min_sentences_per_batch_step: {min_sentences_per_batch_step}")
        logger.warning(f"skip_ratio: {skip_ratio}")
        self.shuffle = shuffle
        self.seed = seed
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        self.epoch = 0
        self.batches = []

    def set_dataset(self, dataset, sampler_cache_s3_bucket, sampler_cache_s3_folder):
        self.dataset = dataset
        self.sampler_cache_s3_bucket = sampler_cache_s3_bucket
        self.sampler_cache_s3_folder = sampler_cache_s3_folder
        cache_file_name = f"{sampler_cache_s3_folder}/rank_{self.rank}_batches.pkl"

        try:
            s3_client = boto3.client("s3")
            self.batches = pickle.loads(s3_client.get_object(Bucket=sampler_cache_s3_bucket, Key=cache_file_name)['Body'].read())
            logger.warning(f"Load self.batches from s3://{sampler_cache_s3_bucket}/{cache_file_name}")
            
        except:
            logger.warning(f"Not found s3://{sampler_cache_s3_bucket}/{cache_file_name}")
            logger.warning("Generate batches")
            self._generate_batches()
            # cache batches to save time for next epoch
            s3_resource = boto3.resource("s3")
            pickle_bytes = pickle.dumps(self.batches)
            s3_resource.Object(sampler_cache_s3_bucket, cache_file_name).put(Body=pickle_bytes)
            logger.warning(f"Dump self.batches to s3://{sampler_cache_s3_bucket}/{cache_file_name}")

    def _generate_batches(self):
        try:
            # If an execution using an identical dataset (order of the datapoints in the dataset must be identical)
            # has been triggered before, change length_buckets_file_name to the file saved from the previous execution.
            # This saves a lot of time when a corpus is not in memory (e.g., ICQ_Augmented).
            s3_client = boto3.client("s3")
            length_buckets_file_name = f"flyte/   /length_buckets.pkl"
            length_buckets = pickle.loads(s3_client.get_object(Bucket=self.sampler_cache_s3_bucket, Key=length_buckets_file_name)['Body'].read())
            logger.warning(f"Load length_buckets from s3://{self.sampler_cache_s3_bucket}/{length_buckets_file_name}")
            # check if length_buckets matches dataset (high probability but not guaranteed)
            for length, indices in length_buckets.items():
                if len(indices) == 1:
                    continue
                index_1, index_2 = random.sample(indices, 2)
                if len(self.dataset[index_1]) != len(self.dataset[index_2]):
                    raise Exception("Incorrect length_buckets_file_name loaded")

        except:
            logger.warning("Generate length_buckets")
            length_buckets = defaultdict(list)
            for idx, datapoint in enumerate(self.dataset):
                length = len(datapoint)
                length_buckets[length].append(idx)

            s3_resource = boto3.resource("s3")
            pickle_bytes = pickle.dumps(length_buckets)
            s3_resource.Object(self.sampler_cache_s3_bucket, f"{self.sampler_cache_s3_folder}/length_buckets.pkl").put(Body=pickle_bytes)
            logger.warning(f"Dump length_buckets to s3://{self.sampler_cache_s3_bucket}/{self.sampler_cache_s3_folder}/length_buckets.pkl")

        sorted_lengths = sorted(length_buckets.keys())
        self.batches = []
        for length in sorted_lengths:
            num_sentences_per_batch_step = min(self.max_tokens_per_batch_step // length, self.max_sentences_per_batch_step)
            num_sentences_per_batch_step = max(num_sentences_per_batch_step, self.min_sentences_per_batch_step)

            indices_all_GPU = length_buckets[length]
            num_sentences_all_GPU = len(indices_all_GPU)
            num_sentences_per_GPU = num_sentences_all_GPU // self.num_replicas # drop last few sentences
            indices_this_GPU = indices_all_GPU[self.rank : num_sentences_per_GPU * self.num_replicas : self.num_replicas]

            # num_sentences_per_batch = num_sentences_per_batch_step * NUM_STEPS_PER_BATCH
            for i in range(0, num_sentences_per_GPU, num_sentences_per_batch_step):
                batch_indices = indices_this_GPU[i:i + num_sentences_per_batch_step]
                if self.skip_ratio is not None:
                    threshold = num_sentences_per_batch_step * self.skip_ratio
                    if len(batch_indices) < threshold:
                        logger.warning(f"{len(batch_indices)} sentences of length {length} are skipped")
                        break
                self.batches.append(batch_indices)

        num_indices = sum(len(batch) for batch in self.batches)
        logger.warning(f"num datapoint: {num_indices}")

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            logger.warning(f"shuffle begin from rank {self.rank}")
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_indices = torch.randperm(len(self.batches), generator=g).tolist()
            # logger.warning(f"batch_indices of rank {self.rank}: {str(batch_indices)}")
            shuffle_batches = [self.batches[batch_index] for batch_index in batch_indices]
            logger.warning(f"shuffle end from rank {self.rank}")
            return iter(shuffle_batches)
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
