from . import data
from . import models
from . import visual

import sys
import logging

logger = logging.getLogger(__name__)

FORMAT = '%(asctime)-15s %(message)s'

logging.basicConfig(level=logging.WARNING, format=FORMAT, stream=sys.stdout)
logging.getLogger('flair').setLevel(logging.INFO)