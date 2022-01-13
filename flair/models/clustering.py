import pickle
from pathlib import Path
from typing import Union, List

import torch
from tqdm import tqdm
from collections import OrderedDict

from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.model_selection import train_test_split

from flair.data import Corpus, Sentence
from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings

import logging

log = logging.getLogger("flair")


class ClusteringModel:
    def __init__(
        self, model: Union[ClusterMixin, BaseEstimator], corpus: Corpus, label_type: str, embeddings: DocumentEmbeddings
    ):
        self.model = model
        self.corpus = corpus
        self.label_type = label_type
        self.embeddings = embeddings

    def fit(self, **kwargs):
        """
        Trains the model.
        """
        X, y = self._convert_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.model.fit(X_train, **kwargs)

    def predict(self, sentences: Union[List[Sentence], Sentence]):
        """
        Predict labels given a list of sentences and returns the respective class indices.
        """
        X = [sentence.embedding.cpu().detach().numpy() for sentence in sentences]
        return self.model.predict(X)

    def save(self, model_file: Union[str, Path]):
        """
        Saves current model.
        """
        binary_result = pickle.dumps(self.model)
        torch.save(binary_result, str(model_file), pickle_protocol=4)

        log.info("Saved model to: " + str(model_file))

    def load(self, model_file: Union[str, Path]):
        """
        Loads a model.
        """
        state = torch.load(model_file)
        self.model = pickle.loads(state)

        log.info("loaded model from: " + str(model_file))

    def _convert_dataset(self, batch_size: int = 32, return_label_dict: bool = False):
        """
        Turns the corpora into X, y datasets as required for most sklearn clustering models.
        Ref.: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
        """
        sentences = self.corpus.get_all_sentences()

        log.info("Embed sentences...")
        for batch in tqdm(DataLoader(sentences, batch_size=batch_size)):
            self.embeddings.embed(batch)

        X = [sentence.embedding.cpu().detach().numpy() for sentence in sentences]
        labels = [sentence.get_labels(self.label_type)[0].value for sentence in sentences]
        label_dict = {v: k for k, v in enumerate(OrderedDict.fromkeys(labels))}
        y = [label_dict.get(label) for label in labels]

        if return_label_dict:
            return X, y, label_dict

        return X, y
