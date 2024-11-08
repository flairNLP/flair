from collections.abc import Iterable
from typing import Union

from flair.data import Corpus, MultiCorpus
from flair.models import MultitaskModel
from flair.nn import Classifier, Model


def make_multitask_model_and_corpus(
    mapping: Iterable[Union[tuple[Classifier, Corpus], tuple[Classifier, Corpus, float]]]
) -> tuple[Model, Corpus]:
    models = []
    corpora = []
    loss_factors = []
    ids = []

    for task_id, _map in enumerate(mapping):
        models.append(_map[0])
        corpora.append(_map[1])
        if len(_map) == 3:
            loss_factors.append(_map[2])
        else:
            loss_factors.append(1.0)

        ids.append(f"Task_{task_id}")

    return MultitaskModel(models=models, task_ids=ids, loss_factors=loss_factors), MultiCorpus(corpora, ids)
