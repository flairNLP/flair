from . import data
from . import models
from . import visual
from . import trainers

import sys
import logging
import warnings

logger = logging.getLogger(__name__)

FORMAT = '%(asctime)-15s %(message)s'

logging.basicConfig(level=logging.WARNING, format=FORMAT, stream=sys.stdout)
logging.getLogger('flair').setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)