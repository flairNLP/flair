import logging
from collections import Counter
from typing import Optional

import torch
from tqdm import tqdm

import flair
from flair.datasets import DataLoader
from flair.nn.distance import (
    CosineDistance,
    EuclideanDistance,
    HyperbolicDistance,
    LogitCosineDistance,
    NegativeScaledDotProduct,
)

logger = logging.getLogger("flair")


class PrototypicalDecoder(torch.nn.Module):
    def __init__(
            self,
            num_prototypes: int,
            embeddings_size: int,
            prototype_size: Optional[int] = None,
            distance_function: str = "euclidean",
            normal_distributed_initial_prototypes: bool = False,
            running_prototypes: bool = True,
    ):
        super().__init__()

        if not prototype_size:
            prototype_size = embeddings_size

        self.prototype_size = prototype_size

        # optional metric space decoder if prototypes have different length than embedding
        self.metric_space_decoder: Optional[torch.nn.Linear] = None
        if prototype_size != embeddings_size:
            self.metric_space_decoder = torch.nn.Linear(embeddings_size, prototype_size)
            torch.nn.init.xavier_uniform_(self.metric_space_decoder.weight)

        # create initial prototypes for all classes (all initial prototypes are a vector of all 1s)
        self.prototype_vectors = torch.nn.Parameter(torch.ones(num_prototypes, prototype_size), requires_grad=True)

        # if set, create initial prototypes from normal distribution
        if normal_distributed_initial_prototypes:
            self.prototype_vectors = torch.nn.Parameter(torch.normal(torch.zeros(num_prototypes, prototype_size)))

        self._distance_function = distance_function

        self.distance: Optional[torch.nn.Module] = None
        if distance_function.lower() == "hyperbolic":
            self.distance = HyperbolicDistance()
        elif distance_function.lower() == "cosine":
            self.distance = CosineDistance()
        elif distance_function.lower() == "logit_cosine":
            self.distance = LogitCosineDistance()
        elif distance_function.lower() == "euclidean":
            self.distance = EuclideanDistance()
        elif distance_function.lower() == "dot_product":
            self.distance = NegativeScaledDotProduct()
        else:
            raise KeyError(f"Distance function {distance_function} not found.")

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    @property
    def num_prototypes(self):
        return self.prototype_vectors.size(0)

    def forward(self, embedded, return_prototypes: bool = False):
        # decode embeddings into prototype space
        if self.metric_space_decoder is not None:
            encoded = self.metric_space_decoder(embedded)
        else:
            encoded = embedded

        if return_prototypes:
            return encoded

        distance = self.distance(encoded, self.prototype_vectors)

        scores = -distance

        return scores

    def calculate_first_prototypes(self, classifier, sentences, mini_batch_size=16):
        """
        Function that calclues a prototype for each class based on the first embedding in the whole dataset
        :param dataset: dataset for which to calculate prototypes
        :param mini_batch_size: number of sentences to embed at same time
        :return:
        """
        self.prototype_vectors = torch.nn.Parameter(torch.normal(torch.zeros(self.num_prototypes, self.prototype_size)))
        self.prototype_vectors.requires_grad = False

        from flair.nn import DefaultClassifier
        classifier: DefaultClassifier = classifier

        label_dictionary = classifier.label_dictionary
        handled = set()

        representative_sentences = []
        for sentence in tqdm(sentences):
            labels = [label.value for label in sentence.get_labels(classifier.label_type)]

            if all(label in handled for label in labels):
                continue
            else:
                representative_sentences.append(sentence)
                for label in labels:
                    handled.add(label)

            if len(handled) == len(label_dictionary):
                break
        print(f"Found {len(representative_sentences)} (of {len(sentences)}) sentences for {len(handled)} tags")

        handled = set()
        dataloader = DataLoader(sentences, batch_size=mini_batch_size)

        for batch in tqdm(dataloader):

            # get the data points for which to predict labels
            data_points = classifier._get_data_points_for_batch(batch)

            # pass data points through network to get encoded data point tensor
            data_point_tensor = classifier._encode_data_points(batch, data_points)
            data_point_tensor = data_point_tensor.detach()

            for idx, data_point in enumerate(data_points):
                label_value = data_point.get_label(classifier.label_type).value
                if label_value not in handled:
                    self.prototype_vectors[label_dictionary.get_idx_for_item(label_value)] \
                        = data_point_tensor[idx]
                    handled.add(label_value)

        self.prototype_vectors.requires_grad = True

    def calculate_average_prototypes(self, classifier, sentences):
        """
        Function that calclues a prototype for each class based on the first embedding in the whole dataset
        :param dataset: dataset for which to calculate prototypes
        :param mini_batch_size: number of sentences to embed at same time
        :return:
        """
        # reset prototypes for all classes
        self.prototype_vectors = torch.nn.Parameter(torch.normal(torch.zeros(self.num_prototypes, self.prototype_size)))
        self.prototype_vectors.requires_grad = False

        from flair.nn import DefaultClassifier
        classifier: DefaultClassifier = classifier

        label_dictionary = classifier.label_dictionary

        counter = Counter()
        for sentence in sentences:
            counter.update([token.get_label(classifier.label_type).value for token in sentence])

        dataloader = DataLoader(sentences, batch_size=32)

        for batch in tqdm(dataloader):

            data_points = classifier._get_data_points_for_batch(batch)

            # pass data points through network to get encoded data point tensor
            data_point_tensor = classifier._encode_data_points(batch, data_points)

            for idx, data_point in enumerate(data_points):
                label = data_point.get_label(classifier.label_type).value

                # add up prototypes
                self.prototype_vectors[label_dictionary.get_idx_for_item(label)] \
                    += data_point_tensor[idx]

        for label, count in counter.most_common():
            # print(label, count, self.prototype_vectors[label_dictionary.get_idx_for_item(label)][:4])
            average_prototype = self.prototype_vectors[label_dictionary.get_idx_for_item(label)] / count
            self.prototype_vectors[label_dictionary.get_idx_for_item(label)] = average_prototype
            # print(self.prototype_vectors[label_dictionary.get_idx_for_item(label)][:4])

        # ads
        self.prototype_vectors.requires_grad = True
