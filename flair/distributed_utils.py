import logging
import os
from multiprocessing.connection import Connection
from typing import Callable

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group

import flair
from flair.class_utils import T

log = logging.getLogger("flair")


def launch_distributed(fn, *args, **kwargs):
    """Executes the function fn(*args, **kwargs) on multiple processes (one for each local GPU).

    Returns: the return value of the function fp(*args, **kwargs) from the rank 0 process
    """
    world_size = torch.cuda.device_count()
    log.info(f"Launching {world_size} processes")
    parent_conn, child_conn = mp.Pipe()
    mp.spawn(_entrypoint, args=(world_size, child_conn, fn, args, kwargs), nprocs=world_size)
    return_value = parent_conn.recv()
    return return_value


def _entrypoint(rank: int, world_size: int, child_conn: Connection, fn: Callable, args: tuple, kwargs: dict) -> None:
    """Lifecycle of a process -- setup, run, cleanup."""
    log.info(f"Started process on rank={rank}")
    _ddp_setup(rank, world_size)
    return_value = fn(*args, **kwargs)
    if is_main_process():
        child_conn.send(return_value)
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


def aggregate_if_distributed(value: T, aggregation_fn: Callable = np.mean) -> T:
    """Gathers value from each process and returns the aggregated value according to the supplied function."""
    if torch.distributed.is_initialized():
        gathered_values = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_values, value)
        return aggregation_fn(gathered_values)
    else:
        return value
