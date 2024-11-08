import logging
import os
import random
from multiprocessing.connection import Connection
from typing import Callable

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import Dataset

import flair
from flair.data import Corpus, _len_dataset

log = logging.getLogger("flair")


def launch_distributed(fn, *args, **kwargs):
    """Executes the function fn(*args, **kwargs) on multiple processes (one for each local GPU).

    If training with multi_gpu=True, launch_distributed should wrap your code that calls .train or .fine_tune.

    Returns: the return value of the function fp(*args, **kwargs) from the rank 0 process
    """
    world_size = torch.cuda.device_count()
    log.info(f"Launching {world_size} processes")
    parent_conn, child_conn = mp.Pipe()
    mp.spawn(_process_entrypoint, args=(world_size, child_conn, fn, args, kwargs), nprocs=world_size)
    return_value = parent_conn.recv()
    return return_value


def _process_entrypoint(
    rank: int, world_size: int, child_conn: Connection, fn: Callable, args: tuple, kwargs: dict
) -> None:
    """Lifecycle of a distributed process -- setup, run, cleanup."""
    log.info(f"Started process on rank={rank}")
    try:
        _ddp_setup(rank, world_size)
        return_value = fn(*args, **kwargs)
        if is_main_process():
            child_conn.send(return_value)
    finally:
        destroy_process_group()


def _ddp_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    flair.device = torch.device(rank)
    torch.cuda.set_device(flair.device)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def is_main_process() -> bool:
    """True for exactly 1 process, regardless of whether being run on CPU/single-GPU/multi-gpu."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


def aggregate(value, aggregation_fn=np.mean):
    """Gather `value` from all processes and send to `aggregation_fn` to get a single return value."""
    if torch.distributed.is_initialized():
        gathered_values = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_values, value)
    else:
        gathered_values = [value]
    return aggregation_fn(gathered_values)


def validate_corpus_same_across_processes(corpus: Corpus) -> None:
    for dataset in [corpus.train, corpus.dev, corpus.test]:
        if dataset is not None:
            validate_dataset_same_across_processes(dataset)


def validate_dataset_same_across_processes(dataset: Dataset, sample_size: int = 10) -> None:
    """Sanity checks a few examples to catch datasets that are obviously different, but not exhaustive to save time."""
    random_indices = random.sample(range(_len_dataset(dataset)), min(sample_size, _len_dataset(dataset)))
    for i in random_indices:
        example = dataset[i]
        examples = aggregate(example, list)
        if not all(example == examples[0] for example in examples):
            raise ValueError("Dataset must be the same on each process")
