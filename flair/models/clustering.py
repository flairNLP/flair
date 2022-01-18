import pickle
from pathlib import Path
from typing import Union, List

import torch
from sklearn.metrics import normalized_mutual_info_score, silhouette_samples
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
    """
    A wrapper class for the sklearn clustering models. With this class clustering with the library 'flair' can be done.
    """

    def __init__(
        self, model: Union[ClusterMixin, BaseEstimator], corpus: Corpus, label_type: str, embeddings: DocumentEmbeddings
    ):
        """
          :param model: the clustering algortihm from sklearn this wrapper will use.
          :param corpus: the flair corpus this wrapper will use for clustering.
          :param label_type: the label from the sentence will be used for the evaluation.
          :param embeddings: the flair DocumentEmbedding this wrapper uses to calculate a vector for each sentence.
        """

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
        predict = self.model.predict(X)

        for idx, sentence in enumerate(sentences):
            sentence.set_label("cluster", str(predict[idx]))

        return predict

    def save(self, model_file: Union[str, Path]):
        """
        Saves current model.
        """
        dump = pickle.dumps(self.model)
        torch.save(dump, str(model_file), pickle_protocol=4)

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

    def evaluate(self):
        """
        This method calculates some evaluation metrics for the clustering.
        Also, the result of the evaluation is logged.
        """
        sentences = self.corpus.get_all_sentences()
        X = [sentence.embedding.cpu().detach().numpy() for sentence in sentences]
        labels = [sentence.get_labels(self.label_type)[0].value for sentence in sentences]

        predict = self.model.predict(X)

        log.info("NMI - Score: " + str(normalized_mutual_info_score(predict, labels)))
