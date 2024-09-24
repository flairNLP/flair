import logging
import os

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group

import flair

log = logging.getLogger("flair")


def launch_distributed(fp, *args):
    """Executes the function fp(*args) on multiple GPUs (all local GPUs)"""
    world_size = torch.cuda.device_count()
    log.info(f"Launching {world_size} distributed processes")
    mp.spawn(entrypoint, args=(world_size, fp, *args), nprocs=world_size)


def entrypoint(rank, world_size, fp, *args):
    ddp_setup(rank, world_size)
    fp(*args)
    destroy_process_group()


def ddp_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    flair.distributed = True
    flair.device = torch.device(rank)
    torch.cuda.set_device(flair.device)


def is_main_process() -> bool:
    """True for exactly 1 process, regardless of whether being run on CPU/single-GPU/multi-gpu"""
    if flair.distributed:
        return flair.device.index == 0
    else:
        return True


class DistributedModel(torch.nn.parallel.DistributedDataParallel):
    """DistributedDataParallel, but redirects access to methods and attributes to the original Model"""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
