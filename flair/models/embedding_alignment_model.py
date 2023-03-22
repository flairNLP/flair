import logging
from typing import List, Tuple, Union
from collections import Counter
import random

import torch

import flair.embeddings
import flair.nn
from flair.data import Corpus, Sentence, Dictionary
from flair.datasets import DataLoader

log = logging.getLogger("flair")


class EmbeddingAlignmentClassifier(flair.nn.Classifier[Sentence]):
    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        label_dictionary: Dictionary,
        corpus: Corpus,
        knn: int = 5,
        **classifierargs,
    ):
        """
        Class-driven Embedding Alignment Model (CEA)
        The main idea is to learn embeddings that maximize the similarity between two
        textual documents if they share the same class label, and minimize it if they do not.
        This model does not have any learnable parameters, thus it supports only Transformer Embeddings
        because they can be fine-tuned. The model embeds text documents, creates text pairs, and aligns
        two documents closer to each other (by optimizing cosine similarity to be equal to 1) if they belong
        to the same class and moves two documents apart (similarity to be equal to 0) if they don't.
        Prediction phase uses learned embeddings and K-Nearest Neighbors to classify documents.

        :param document_embeddings: Embedding used to encode sentence (transformer document embeddings for now)
        :param label_type: Name of the gold labels to use.
        :param corpus: The dataset used to train the model. The model uses the training set for the KNN algorithm.
        :param knn: number of neighbours for KNN predictions

        # Update: this model works and was tested on TREC_6 & TREC_50
        # TODO: Storing + loading is not implemented yet (don't store the corpus), + minor todo with replacing 'random
        """

        super(EmbeddingAlignmentClassifier, self).__init__(**classifierargs)

        # only document embeddings so far
        self.embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings

        # corpus is required to make KNN predictions
        self._corpus: Corpus = corpus

        self._label_type: str = label_type

        self.label_dictionary: Dictionary = label_dictionary

        # number of neighbours for K-NN predictions
        self.knn = knn

        # loss function: MSE between cosine similarities and 0s (pairs of different class) and 1s (pairs of same class)
        self.loss_function = torch.nn.MSELoss(reduction="sum")

        # auto-spawn on GPU if available
        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    @property
    def corpus(self):
        return self._corpus

    def _create_sentence_pair_label_map(self, sentences: List[Sentence]) -> List[Tuple[Tuple[int, int], int]]:

        sentence_pair_label_map = []  # maps a pair of two sentence ids to 0 or 1 label

        # group sentences by their labels and add sentence ids for each label
        sentence_label_map = {label: [] for label in self.label_dictionary.get_items()}
        sentence_label_map.pop("<unk>", None)  # disregard unk label
        for sentence_id, sentence in enumerate(sentences):
            sentence_label_map[sentence.get_label(self._label_type).value].append(sentence_id)

        # for each sentence sample one pair from the same class (positives) and one from another (negatives)
        for sentence_id, sentence in enumerate(sentences):
            label = sentence.get_label(self._label_type).value
            positive_sentences = [sent_id for sent_id in sentence_label_map[label] if sent_id != sentence_id]
            negative_sentences = [sent_id for sent_id in list(range(len(sentences)))
                                  if sent_id not in positive_sentences and sent_id != sentence_id]

            # add sentence ids as pairs: positive pair ids and label 1
            if positive_sentences:
                positive_sentence_pair = random.choice(positive_sentences)
                sentence_pair_label_map.append(((sentence_id, positive_sentence_pair), 1))

            # add sentence ids as pairs: negative pair ids and label 0
            if negative_sentences:
                negative_sentence_pair = random.choice(negative_sentences)
                sentence_pair_label_map.append(((sentence_id, negative_sentence_pair), 0))

        return sentence_pair_label_map

    def _calculate_loss(self, sentences: List[Sentence], sentence_pair_label_map) -> Tuple[torch.Tensor, int]:
        """loss is calculated only during model training and not during prediction"""

        embedding_names = self.embeddings.get_names()

        # gather embeddings for each sentence and its pair
        first_sentences = [sentences[sentence_pair[0]].get_embedding(embedding_names)
                           for sentence_pair, _ in sentence_pair_label_map]
        second_sentences = [sentences[sentence_pair[1]].get_embedding(embedding_names)
                            for sentence_pair, _ in sentence_pair_label_map]

        # put to gpu
        first_sentences = torch.stack(first_sentences).to(flair.device)
        second_sentences = torch.stack(second_sentences).to(flair.device)

        # calculate cosine similarities for a full batch
        similarities = torch.nn.functional.cosine_similarity(first_sentences, second_sentences, dim=1)
        labels = torch.FloatTensor([label for _, label in sentence_pair_label_map]).to(flair.device)

        loss = self.loss_function(similarities, labels)

        return loss, len(sentence_pair_label_map)

    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:

        # embed all sentences
        self.embeddings.embed(sentences)

        # create sentence pairs and 0 or 1 labels
        sentence_pair_label_map = self._create_sentence_pair_label_map(sentences)

        # calculate MSE loss between sentence pair similarities and 0s and 1s
        return self._calculate_loss(sentences, sentence_pair_label_map)

    def _embed_training_set(self, mini_batch_size: int = 32):
        loader = DataLoader(self._corpus.train, batch_size=mini_batch_size, num_workers=0)
        for batch in loader:
            self.embeddings.embed(batch)

    def _k_nearest_neighbor(self, sentence: Sentence) -> str:
        """KNN classification for a single sentence"""

        # get embedding for a given sentence
        embedding_names = self.embeddings.get_names()
        sentence_embedding = sentence.get_embedding(embedding_names)

        # gather embeddings for all training instances
        train_set_embeddings = [training_sample.get_embedding(embedding_names)
                                for training_sample in self._corpus.train]
        train_set_embeddings = torch.stack(train_set_embeddings)

        # calculate cos similarity between given sentence and all training instances
        similarities = torch.nn.functional.cosine_similarity(sentence_embedding, train_set_embeddings, dim=1)

        # sort by top k (nearest neighbours) and get their labels
        _, top_k_idx = similarities.topk(k=self.knn)
        closest_labels = [self._corpus.train[sentence_id].get_label(self._label_type).value
                          for sentence_id in top_k_idx]

        # majority vote
        predicted_label = Counter(closest_labels).most_common(1)[0][0]

        return predicted_label

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        label_name: str = "predicted",
        return_loss: bool = False,
        **kwargs,
    ):
        """Prediction phase uses K-Nearest Neighbor thus needs embeddings of the training set"""

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # filter empty sentences
        sentences = [sentence for sentence in sentences if len(sentence) > 0]
        if len(sentences) == 0:
            return sentences

        # Step 1: embed full training set if needed
        embedding_names = self.embeddings.get_names()
        is_train_set_already_embedded = len(self._corpus.train[0].get_embedding(embedding_names)) != 0
        if not is_train_set_already_embedded:
            self._embed_training_set(mini_batch_size=mini_batch_size)

        # Step 2: embed sentences to be predicted
        loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=0)
        for batch in loader:
            self.embeddings.embed(batch)

        # Step 3: perform KNN to assign each new sentence a class
        for sentence in sentences:
            predicted_label = self._k_nearest_neighbor(sentence)
            sentence.add_label(typename=label_name, value=predicted_label)

        # KNN predictions do not have loss value
        if return_loss:
            return 0

    def evaluate(self, data_points, mini_batch_size: int, **kwargs):
        result = super().evaluate(data_points, mini_batch_size=mini_batch_size, **kwargs)

        # clear embeddings of the training set after each epoch
        embedding_names = self.embeddings.get_names()
        for sentence in self._corpus.train:
            sentence.clear_embeddings(embedding_names)

        return result
