import torch

import flair
from flair.models import TARSClassifier

flair.device = torch.device("cpu")
m = TARSClassifier.load("tars-base")
m.save("local-tars-base.pt")
m = TARSClassifier.load("local-tars-base.pt")
breakpoint()
