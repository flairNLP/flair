from typing import Iterable, Tuple, Union

from flair.data import Corpus, MultiCorpus
from flair.models import MultitaskModel
from flair.nn import Classifier, Model


def make_multitask_model_and_corpus(
    mapping: Iterable[Union[Tuple[Classifier, Corpus], Tuple[Classifier, Corpus, float]]]
) -> Tuple[Model, Corpus]:
    models = []
    corpora = []
    loss_factors = []
    ids = []

    for task_id, map in enumerate(mapping):
        models.append(map[0])
        corpora.append(map[1])
        if len(map) == 3:
            loss_factors.append(map[2])
        else:
            loss_factors.append(1.0)

        ids.append(f"Task_{task_id}")

    return MultitaskModel(models=models, task_ids=ids, loss_factors=loss_factors), MultiCorpus(corpora, ids)
