import logging
import os
import random
from multiprocessing.connection import Connection
from typing import Callable, Collection, Iterable, TypeVar

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import Dataset

import flair
from flair.data import Corpus, _len_dataset

log = logging.getLogger("flair")

T = TypeVar("T")


def launch_distributed(fn: Callable, *args, **kwargs):
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


def validate_corpus_same_each_process(corpus: Corpus) -> None:
    """Catches most cases in which a corpus is not the same on each process.

    However, there is no guarantee for two reasons:
    1) It uses a sample for speed
    2) It compares strings to avoid requiring the datasets to be serializable
    """
    for dataset in [corpus.train, corpus.dev, corpus.test]:
        if dataset is not None:
            _validate_dataset_same_each_process(dataset)


def _validate_dataset_same_each_process(dataset: Dataset, sample_size: int = 10) -> None:
    """:raises: ValueError if the dataset is not the same on each process."""
    random_indices = random.sample(range(_len_dataset(dataset)), min(sample_size, _len_dataset(dataset)))
    for i in random_indices:
        example = str(dataset[i])
        examples = aggregate(example, list)
        if not all(example == examples[0] for example in examples):
            raise ValueError("Dataset must be the same on each process")


def gather(value: T) -> list[T]:
    """Gather `value` from all processes and return a list of values."""
    if torch.distributed.is_initialized():
        gathered_values = [value for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_values, value)
    else:
        gathered_values = [value]
    return gathered_values


def aggregate(value: T, aggregation_fn: Callable):
    """Gather `value` from all processes and send to `aggregation_fn` to get a single return value."""
    gathered_values = gather(value)
    return aggregation_fn(gathered_values)


def broadcast_value(value: T, src: int = 0) -> T:
    """
    Broadcasts a Python object from the source process (src) to all other processes.
    Every process returns the same object.
    """
    obj_list = [value]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# aggregation functions
def flatten(l: Iterable[Iterable[T]]) -> list[T]:
    """Flattens all elements in an iterable, such as a list, of iterables into a single list."""
    return [x for s in l for x in s]


def flatten_set(list_of_sets: Iterable[Iterable[T]]) -> set[T]:
    """Flattens all elements in an iterable, such as a list, of iterables into a single set."""
    return {x for subset in list_of_sets for x in subset}


def merge_sets(list_of_sets: Collection[set[T]]) -> set[T]:
    """Merges a collection of sets into a single set."""
    merged_set = set()
    for s in list_of_sets:
        merged_set.update(s)
    return merged_set


def flatten_dicts(list_of_dicts: list[dict[str, list[T]]]) -> dict[str, list[T]]:
    """This function merges a list of dictionaries with list values into a single dictionary with merged list values."""
    merged_dict: dict[str, list[T]] = {}
    for d in list_of_dicts:
        for k, v in d.items():
            if k not in merged_dict:
                merged_dict[k] = []
            merged_dict[k].extend(v)
    return merged_dict


def aggregate_tensor_sum(list_of_tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Custom aggregation function to sum loss values from all processes.
    Moves all tensors to CPU and converts them to Python scalars before summing.
    Returns a single tensor containing the summed loss.
    """
    total = sum(t.cpu().item() for t in list_of_tensors)
    return torch.tensor(total)
