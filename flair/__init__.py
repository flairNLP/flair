import logging.config
import os
from pathlib import Path

import torch
from transformers import set_seed as hf_set_seed

# global variable: cache_root
cache_root = Path(os.getenv("FLAIR_CACHE_ROOT", Path(Path.home(), ".flair")))

# global variable: device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# global variable: version
__version__ = "0.11"

# global variable: arrow symbol
_arrow = " → "

from . import (  # noqa: E402 import after setting device
    data,
    models,
    nn,
    trainers,
    visual,
)

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"flair": {"handlers": ["console"], "level": "INFO", "propagate": False}},
    }
)

logger = logging.getLogger("flair")


def set_seed(seed: int):
    hf_set_seed(seed)


__all__ = [
    "cache_root",
    "device",
    "__version__",
    "logger",
    "set_seed",
    "data",
    "models",
    "nn",
    "trainers",
    "visual",
    "datasets",
]
