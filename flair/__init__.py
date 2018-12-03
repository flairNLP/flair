from . import data
from . import models
from . import visual
from . import trainers

import logging.config
import yaml


with open('logging-config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger('flair')
