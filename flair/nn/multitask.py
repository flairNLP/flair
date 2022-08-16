from typing import Iterable, Tuple

from flair.data import Corpus, MultiCorpus
from flair.models import MultitaskModel
from flair.nn import Model


def make_multitask_model_and_corpus(mapping: Iterable[Tuple[Model, Corpus]]) -> Tuple[Model, Corpus]:
    models = []
    corpora = []
    ids = []

    for task_id, map in enumerate(mapping):
        models.append(map[0])
        corpora.append(map[1])
        ids.append(f"Task_{task_id}")

    return MultitaskModel(models, ids), MultiCorpus(corpora, ids)
