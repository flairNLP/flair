from typing import Optional, Union, List

from pathlib import Path

import logging
from tqdm import tqdm

import torch
from torch.nn import Module

import flair
from flair.data import DataPoint, Sentence, Dictionary, Dataset, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.nn import Classifier
from flair.training_utils import Result, store_embeddings
from flair.embeddings import TokenEmbeddings

from .distance import HyperbolicMean, HyperbolicDistance, EuclideanMean, EuclideanDistance


log = logging.getLogger("flair.PrototypicalTagger")


class PrototypicalSequenceTagger(Classifier):
    def __init__(self,
                 sorted_samples,
                 embeddings : TokenEmbeddings,
                 tag_type : str, support_size : int,
                 hyperbolic : Optional[bool] = True,
                 embedding_to_metric_space : Optional[Module] = None,
                 require_double_eval : Optional[bool] = False,
                 ):
        """
        Prototypical model to tag tokens in a sentence using an embedding and
        an euclidean or hyperbolic distance metric.

        :param train_data: Train data used to compute prototypes on model.eval().
        :param embeddings: Embedding for the sentences the tokens occur in.
        The embedding should contain information about the sentence
        (otherwise token tagging done this way becomes pointless).
        :param tag_type: The tag to predict.
        :param support_size: The size of the support set (over all classes).
        This number can be retrieved from the episodic sampler.
        :param hyperbolic: Whether to use euclidean or hyperbolic distance.
        :param require_double_eval: Prototypes are only computed on the second
        call of eval (or train(false)) - if used with train_with_dev=True,
        prototypes are only computed once at the end of training.
        :param embedding_to_metric_space: Funktion to apply after the embedding.
        """

        super().__init__()
        self.embeddings = embeddings
        self.support_size = support_size
        self.tag_type = tag_type
        self.prototype_vectors = None
        self.prototype_labels = None

        self.sorted_samples = sorted_samples

        self.require_double_eval = require_double_eval

        self._hyperbolic = hyperbolic


        self.embedding_to_metric_space = embedding_to_metric_space

        self.loss = torch.nn.CrossEntropyLoss()

        if hyperbolic:
            self.distance = HyperbolicDistance()
            self.mean = HyperbolicMean()
        else:
            self.distance = EuclideanDistance()
            self.mean = EuclideanMean()

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def encode_tokens(self, tokens):
        names = self.embeddings.get_names()

        self.embeddings.embed([
            token.sentence for token in tokens])

        embedded = torch.stack([
            torch.cat(token.get_each_embedding(names))
            for token in tokens
        ], dim=0)

        if self.embedding_to_metric_space is not None:
            return self.embedding_to_metric_space(embedded)
        else:
            return embedded

    def compute_prototypes(self, support_set, sorted=False):
        # list for prototypes
        labels, vectors = list(), list()

        if not sorted:
            classes = dict()

            for token in support_set:
                cls = token.get_tag(self.tag_type).value
                classes.setdefault(cls, list()).append(token)

            for cls, tokens in classes.items():
                labels.append(cls)
                vectors.append(
                    self.mean(self.encode_tokens(tokens))
                )
        else:
            for bucket in support_set:
                current_class = None
                current_tokens = list()

                for token in bucket:
                    cls = token.get_tag(self.tag_type).value

                    if current_class is None:
                        current_class = cls
                    else:
                        assert cls == current_class

                    current_tokens.append(token)

                labels.append(current_class)
                vectors.append(
                    self.mean(self.encode_tokens(current_tokens))
                )

        self.prototype_labels = labels
        self.prototype_vectors = torch.stack(vectors, dim=0)

    def forward_loss(self, data):
        feature = self.forward(data)
        query_set = data[self.support_size:]
        return self._calculate_loss(feature, query_set)

    def _get_label_index(self, label):
        try:
            return self.prototype_labels.index(label)
        except ValueError:
            return 0

    def _calculate_loss(self, feature, tokens):
        true_class = torch.tensor([
            self._get_label_index(
                token.get_tag(self.tag_type).value
            )
            for token in tokens
        ]).to(flair.device)

        return self.loss(feature, true_class)

    def forward(self, data):
        if self.training:
            support_set = data[:self.support_size]
            query_set = data[self.support_size:]
            self.compute_prototypes(support_set)

        else:
            query_set = data

        assert self.prototype_vectors is not None

        query_set = self.encode_tokens(query_set)

        return -self.distance(query_set, self.prototype_vectors)


    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size=32,
            all_tag_prob: bool = False,
            verbose: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none"
    ):
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a List of Sentence
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name is None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences


             # read Dataset into data loader (if list of sentences passed, make Dataset first)
            if not isinstance(sentences, Dataset):
                sentences = SentenceDataset(sentences)

            dataloader = DataLoader(sentences, batch_size=mini_batch_size, )

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                tokens = [
                    token
                    for sentence in sentences
                    for token in sentence
                ]

                feature = self.forward(tokens)

                if return_loss:
                    overall_loss += self._calculate_loss(feature, tokens)

                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    get_all_tags=all_tag_prob,
                )

                for (token, tag) in zip(tokens, tags):
                    token.add_tag_label(label_name, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (token, token_all_tags) in zip(batch, all_tags):
                    token.add_tags_proba_dist(label_name, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def _obtain_labels(
            self,
            feature: torch.Tensor,
            get_all_tags: bool,
    ) -> (List[List[Label]], List[List[List[Label]]]):
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """
        tags = []
        all_tags = []

        softmax_batch = torch.nn.functional.softmax(feature, dim=-1)

        softmax_batch = softmax_batch.cpu()

        probs_batch, prediction_batch = torch.max(softmax_batch, dim=-1)


        for all_probs, prob, pred in zip(softmax_batch, probs_batch, prediction_batch):
            tags.append(
                Label(self.prototype_labels[pred], prob)
            )

            if get_all_tags:
                all_tags.append([
                    Label(
                        self.prototype_labels[idx], idx_prob
                    )
                    for idx, idx_prob in enumerate(all_probs)
                ])

        return tags, all_tags

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "tag_type": self.tag_type,
            "support_size": self.support_size,
            "hyperbolic": self._hyperbolic,
            "prototype_labels": self.prototype_labels,
            "prototype_vectors": self.prototype_vectors
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = PrototypicalTagger(
            embeddings=state["embeddings"],
            tag_type=state["tag_type"],
            support_size=state["support_size"],
            hyperbolic=state["hyperbolic"],
        )
        model.load_state_dict(state["state_dict"])
        model.prototype_labels = state["prototype_labels"]
        model.prototype_vectors = state["prototype_vectors"]
        return model

    def train(self, mode=True):
        _last_mode = self.training

        super().train(mode)


        if not self.require_double_eval or (not mode and not _last_mode):
            with torch.no_grad():
                # compute prototypes for evaluation
                self.compute_prototypes(self.sorted_samples(), sorted=True)

    @property
    def label_type(self):
        return self.tag_type
