import logging.config
import os
from pathlib import Path

import torch
from transformers import set_seed as hf_set_seed

# global variable: cache_root
from .file_utils import set_proxies

cache_root = Path(os.getenv("FLAIR_CACHE_ROOT", Path(Path.home(), ".flair")))
"""The path to the cache folder Flair is using.

This value defaults to `<Home Directory>/.flair`.
You can choose the path by setting the `FLAIR_CACHE_ROOT` environment variable.
"""

device: torch.device
"""Flair is using a single device for everything. You can set this device by overwriting this variable.

This value will be automatically set to the first found GPU if available and to CPU otherwise.
You can choose a specific GPU, by setting the `FLAIR_DEVICE` environment variable to its index.
"""


# global variable: device
if torch.cuda.is_available():
    device_id = os.environ.get("FLAIR_DEVICE")

    # No need for correctness checks, torch is doing it
    device = torch.device(f"cuda:{device_id}") if device_id else torch.device("cuda:0")
else:
    device = torch.device("cpu")

# global variable: version
__version__ = "0.12.2"
"""The current version of the flair library installed."""

# global variable: arrow symbol
_arrow = " → "

# global variable: debug
debug = False

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
"""The logger used by Flair.

You can reconfigure it to change the log output to your likings.
"""


def set_seed(seed: int):
    """Set the seed for all random generators used in training.

    Use this method to guarantee reproducibility of experiments.

    Args:
        seed: any value you want

    """
    hf_set_seed(seed)


__all__ = [
    "cache_root",
    "device",
    "__version__",
    "debug",
    "logger",
    "set_seed",
    "data",
    "models",
    "nn",
    "trainers",
    "visual",
    "datasets",
    "set_proxies",
]
