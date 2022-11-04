import logging
import itertools
from pathlib import Path
from typing import List, Tuple, Union

import torch

import flair.embeddings
import flair.nn
from flair.data import Corpus, Sentence, Dictionary
from flair.embeddings import Embeddings
from flair.file_utils import cached_path
from flair.datasets import DataLoader

from collections import Counter

log = logging.getLogger("flair")


class EmbeddingAlignmentlassifier(flair.nn.Classifier[Sentence]):
    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        corpus: Corpus,
        **classifierargs,
    ):
        """
        Class-driven Embedding Alignment Model (CEA)
        The main idea is to learn embeddings that maximize the similarity between two
        textual documents if they share the same class label, and minimize it if they do not.
        This model does not have any learnable parameters, thus it supports only Transformer Embeddings
        because they can be fine-tuned. The model embeds text documents, creates text pairs, and aligns
        two documents closer to each other (by optimizing cosine similarity to be equal to 1).
        Prediction phase uses learned embeddings and K-Nearest Neighbors to classify documents.
        :param document_embeddings: Embedding used to encode sentence (transformer document embeddings for now)
        :param corpus: The dataset used to train the model. The model uses the training set for the KNN algorithm.
        :param label_type: Name of the gold labels to use.

        # TODO: overwrite evaluate method
        # 1) Training set has to be re-embedded after each epoch (main reason to overwrite evaluate)
        # 2) KNN does not have a loss value (as expected in other classifier models)
        """

        super(EmbeddingAlignmentlassifier, self).__init__(**classifierargs)

        # only document embeddings so far
        self.document_embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings

        # corpus is required to make KNN predictions
        self._corpus = corpus

        self._label_type = label_type

        self._cos_similarity = torch.nn.CosineSimilarity(dim=0)

        self.loss_function = torch.nn.MSELoss(reduction="sum")

        # auto-spawn on GPU if available
        self.to(flair.device)

    def label_type(self):
        return self._label_type

    def _create_sentence_pair_label_map(self, sentences: List[Sentence]) -> List[Tuple[Tuple[int, int], int]]:

        sentence_idx = range(len(sentences))
        sentence_pair_idx = list(itertools.combinations(sentence_idx, 2))

        # "zero" labels (similarity of 0) if two documents don't belong to the same class
        labels = [0] * len(sentence_pair_idx)

        # "one" labels (similarity of 1) if two documents share the same class label
        for pair_id, sentence_pair in enumerate(sentence_pair_idx):
            if sentences[sentence_pair[0]].get_label(self._label_type).value ==\
                    sentences[sentence_pair[1]].get_label(self._label_type).value:
                labels[pair_id] = 1

        sentence_pair_label_map = list(zip(sentence_pair_idx, labels))
        return sentence_pair_label_map

    def _calculate_loss(self, sentences: List[Sentence], sentence_pair_label_map) -> Tuple[torch.Tensor, int]:
        """loss is calculated only during model training and not during prediction"""

        similarities = []
        embedding_names = self.document_embeddings.get_names()

        # TODO: calculating cos similarity for a full batch should be faster than for each pair individually
        for sentence_pair, label in sentence_pair_label_map:
            first_embedding = sentences[sentence_pair[0]].get_embedding(embedding_names)
            second_embedding = sentences[sentence_pair[1]].get_embedding(embedding_names)

            # calculate similarity between two embeddings
            similarities.append(self._cos_similarity(first_embedding, second_embedding))

        similarities = torch.stack(similarities)
        labels = torch.FloatTensor([sentence_pair[1] for sentence_pair in sentence_pair_label_map])

        # loss is the mean squared error between pairwise cosine similarity and 0 and 1 values
        loss = self.loss_function(similarities, labels)

        return loss, len(sentence_pair_label_map)

    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:

        # embed all sentences
        self.document_embeddings.embed(sentences)

        # create sentence pairs and 0 or 1 labels
        sentence_pair_label_map = self._create_sentence_pair_label_map(sentences)

        return self._calculate_loss(sentences, sentence_pair_label_map)

    def _embed_training_set(self, mini_batch_size: int = 32):
        loader = DataLoader(self._corpus.train, batch_size=mini_batch_size, num_workers=0)
        for batch in loader:
            self.document_embeddings.embed(batch)

    def _k_nearest_neighbor(self,
                            sentence: Sentence,
                            n_neighbors: int = 5,
                            label_name: str = "predicted"):
        """Perform KNN classification for a single sentence"""

        embedding_names = self.document_embeddings.get_names()
        sentence_embedding = sentence.get_embedding(embedding_names)

        similarity_label_pairs = []

        for training_sample in self._corpus.train:
            embedding_from_train_set = training_sample.get_embedding(embedding_names)
            similarity = self._cos_similarity(embedding_from_train_set, sentence_embedding).squeeze()
            label = training_sample.get_label(self._label_type).value
            similarity_label_pairs.append((similarity, label))

        # sort and take n most similar documents
        similarity_label_pairs = sorted(similarity_label_pairs, key=lambda tup: tup[0], reverse=True)[:n_neighbors]
        n_closest_labels = [text_pair[1] for text_pair in similarity_label_pairs]

        # majority vote
        predicted_label = Counter(n_closest_labels).most_common(1)[0][0]

        # add predicted label to the sentence
        sentence.add_label(typename=label_name, value=predicted_label)

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        label_name: str = "predicted",
        **kwargs,
    ):

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # filter empty sentences
        sentences = [sentence for sentence in sentences if len(sentence) > 0]
        if len(sentences) == 0:
            return sentences

        # Step 1: embed full training set if needed
        # TODO issue: training set has to be re-embedded after each epoch (fine-tunable transformer model)
        # TODO: this has to be done by overwriting an existing evaluate method and not in predict method

        embedding_names = self.document_embeddings.get_names()
        is_train_set_already_embedded = len(self._corpus.train[0].get_embedding(embedding_names)) != 0
        if not is_train_set_already_embedded:
            self._embed_training_set(mini_batch_size=mini_batch_size)

        # Step 2: embed sentences to be predicted
        loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=0)
        for batch in loader:
            self.document_embeddings.embed(batch)

        # Step 3: perform KNN based on cosine similarity to assign each sentence a class
        for sentence in sentences:
            self._k_nearest_neighbor(sentence, label_name=label_name)
