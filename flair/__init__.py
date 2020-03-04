import os
import torch
from pathlib import Path

# global variable: cache_root
cache_root = os.getenv('FLAIR_CACHE_ROOT', Path(Path.home(), ".flair"))

# global variable: device
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# global variable: embedding_storage_mode
embedding_storage_mode = "default"

from . import data
from . import models
from . import visual
from . import trainers
from . import nn

import logging.config

__version__ = "0.4.5"

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
        "loggers": {
            "flair": {"handlers": ["console"], "level": "INFO", "propagate": False}
        },
    }
)

logger = logging.getLogger("flair")
