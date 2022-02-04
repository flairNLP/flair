import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Union

import joblib
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

from flair.data import Corpus
from flair.datasets import DataLoader
from flair.embeddings import DocumentEmbeddings

log = logging.getLogger("flair")


class ClusteringModel:
    """
    A wrapper class for the sklearn clustering models. With this class clustering with the library 'flair' can be done.
    """

    def __init__(self, model: Union[ClusterMixin, BaseEstimator], embeddings: DocumentEmbeddings):
        """
        :param model: the clustering algortihm from sklearn this wrapper will use.
        :param embeddings: the flair DocumentEmbedding this wrapper uses to calculate a vector for each sentence.
        """
        self.model = model
        self.embeddings = embeddings

    def fit(self, corpus: Corpus, **kwargs):
        """
        Trains the model.
        :param corpus: the flair corpus this wrapper will use for fitting the model.
        """
        X = self._convert_dataset(corpus)

        log.info("Start clustering " + str(self.model) + " with " + str(len(X)) + " Datapoints.")
        self.model.fit(X, **kwargs)
        log.info("Finished clustering.")

    def predict(self, corpus: Corpus):
        """
        Predict labels given a list of sentences and returns the respective class indices.

        :param corpus: the flair corpus this wrapper will use for predicting the labels.
        """

        X = self._convert_dataset(corpus)
        log.info("Start the prediction " + str(self.model) + " with " + str(len(X)) + " Datapoints.")
        predict = self.model.predict(X)

        for idx, sentence in enumerate(corpus.get_all_sentences()):
            sentence.set_label("cluster", str(predict[idx]))

        log.info("Finished prediction and labeled all sentences.")
        return predict

    def save(self, model_file: Union[str, Path]):
        """
        Saves current model.

        :param model_file: path where to save the model.
        """
        joblib.dump(pickle.dumps(self), str(model_file))

        log.info("Saved the model to: " + str(model_file))

    @staticmethod
    def load(model_file: Union[str, Path]):
        """
        Loads a model from a given path.

        :param model_file: path to the file where the model is saved.
        """
        log.info("Loading model from: " + str(model_file))
        return pickle.loads(joblib.load(str(model_file)))

    def _convert_dataset(self, corpus, label_type: str = None, batch_size: int = 32, return_label_dict: bool = False):
        """
        Turns the corpora into X, y datasets as required for most sklearn clustering models.
        Ref.: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster

        :param label_type: the label from sentences will be extracted. If the value is none this will be skipped.
        """

        log.info("Embed sentences...")
        sentences = []
        for batch in tqdm(DataLoader(corpus.get_all_sentences(), batch_size=batch_size)):
            self.embeddings.embed(batch)
            sentences.extend(batch)

        X = [sentence.embedding.cpu().detach().numpy() for sentence in sentences]

        if label_type is None:
            return X

        labels = [sentence.get_labels(label_type)[0].value for sentence in sentences]
        label_dict = {v: k for k, v in enumerate(OrderedDict.fromkeys(labels))}
        y = [label_dict.get(label) for label in labels]

        if return_label_dict:
            return X, y, label_dict

        return X, y

    def evaluate(self, corpus: Corpus, label_type: str):
        """
        This method calculates some evaluation metrics for the clustering.
        Also, the result of the evaluation is logged.

        :param corpus: the flair corpus this wrapper will use for evaluation.
        :param label_type: the label from the sentence will be used for the evaluation.
        """
        X, Y = self._convert_dataset(corpus, label_type=label_type)
        predict = self.model.predict(X)
        log.info("NMI - Score: " + str(normalized_mutual_info_score(predict, Y)))
