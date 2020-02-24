import pytest
import flair.datasets
from typing import Tuple

from flair.data import Dictionary, Corpus
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models.text_classification_model import TextClassifier


def init(tasks_base_path) -> Tuple[Corpus, Dictionary, TextClassifier]:
    # get training, test and dev data
    corpus = flair.datasets.ClassificationCorpus(tasks_base_path / "ag_news")
    label_dict = corpus.make_label_dictionary()

    glove_embedding: WordEmbeddings = WordEmbeddings("turian")
    document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [glove_embedding], 128, 1, False, 64, False, False
    )

    model = TextClassifier(document_embeddings, label_dict, multi_label=False)

    return corpus, label_dict, model


def test_labels_to_indices(tasks_base_path):
    corpus, label_dict, model = init(tasks_base_path)

    result = model._labels_to_indices(corpus.train)

    for i in range(len(corpus.train)):
        expected = label_dict.get_idx_for_item(corpus.train[i].labels[0].value)
        actual = result[i].item()

        assert expected == actual


def test_labels_to_one_hot(tasks_base_path):
    corpus, label_dict, model = init(tasks_base_path)

    result = model._labels_to_one_hot(corpus.train)

    for i in range(len(corpus.train)):
        expected = label_dict.get_idx_for_item(corpus.train[i].labels[0].value)
        actual = result[i]

        for idx in range(len(label_dict)):
            if idx == expected:
                assert actual[idx] == 1
            else:
                assert actual[idx] == 0
