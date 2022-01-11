from collections import OrderedDict

from flair.data import Corpus
from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings
from tqdm import tqdm

import logging

log = logging.getLogger("flair")


def to_clustering_dataset(corpus: Corpus, embeddings: DocumentEmbeddings, label_type: str, batch_size: int = 32, return_label_dict: bool = False, **kwargs):
    """
    Turns the corpora into X, y datasets as required for most sklearn clustering models.
    Ref.: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
    """
    sentences = corpus.get_all_sentences()

    log.info("Embed sentences...")
    for batch in tqdm(DataLoader(sentences, batch_size=batch_size)):
        embeddings.embed(batch)

    X = [sentence.embedding for sentence in sentences]
    labels = [sentence.get_labels(label_type)[0].value for sentence in sentences]
    label_dict = {v: k for k, v in enumerate(OrderedDict.fromkeys(labels))}
    y = [label_dict.get(label) for label in labels]

    if return_label_dict:
        return X, y, label_dict

    return X, y
