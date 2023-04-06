import logging
from typing import List, Tuple, Union, Optional
from collections import Counter
import random

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence, Dictionary
from flair.datasets import DataLoader

log = logging.getLogger("flair")


class EmbeddingAlignmentClassifier(flair.nn.Classifier[Sentence]):
    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        label_dictionary: Dictionary,
        train_corpus: List[Sentence],  # just training set corpus.train
        use_memory: bool = False,
        mix_memory: bool = False,
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
        :param train_corpus: corpus.train set from the corpus. The model uses training set for the KNN algorithm.
        :param knn: number of neighbours for KNN predictions
        :param use_memory: (experimental) store a single embedding from a previous batch and detach
        :param mix_memory: (experimental) use embedding from memory only if no sentence pair was found in the current batch
        """

        super(EmbeddingAlignmentClassifier, self).__init__(**classifierargs)

        # only document embeddings so far
        self.embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings

        # corpus is required to make KNN predictions
        self._train_corpus: List[Sentence] = train_corpus

        self._label_type: str = label_type

        self.label_dictionary: Dictionary = label_dictionary

        # number of neighbours for K-NN predictions
        self.knn = knn

        # store previous embedding for each label (ensures that we always find positive and negative pairs)
        self.use_memory = use_memory
        self.mix_memory = mix_memory
        if self.use_memory:
            self.memory = {label: None for label in self.label_dictionary.get_items()}
            self.memory.pop("<unk>", None)  # disregard unk label

        # loss function: MSE between cosine similarities and 0s (pairs of different class) and 1s (pairs of same class)
        self.loss_function = torch.nn.MSELoss(reduction="sum")

        # auto-spawn on GPU if available
        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    @property
    def train_corpus(self):
        return self._train_corpus

    def _find_embedding_pair_in_batch(self,
                                      sentence: Sentence,
                                      mini_batch: List[Sentence],
                                      sample: str = "positive",  # 'positive' or 'negative' sample
                                      ) -> torch.FloatTensor:  # returns single embedding

        label = sentence.get_label(self._label_type).value
        embedding_names = self.embeddings.get_names()

        # positive sentences
        positive_sentences = [sentence_pair for sentence_pair in mini_batch
                              if sentence_pair.get_label(self._label_type).value == label
                              and sentence_pair != sentence]

        # negative sentences
        negative_sentences = [sentence_pair for sentence_pair in mini_batch
                              if sentence_pair not in positive_sentences
                              and sentence_pair != sentence]

        sentence_pair_embedding: Optional[torch.tensor] = None
        if sample == "positive" and positive_sentences:
            sentence_pair_embedding = random.choice(positive_sentences).get_embedding(embedding_names)
        if sample == "negative" and negative_sentences:
            sentence_pair_embedding = random.choice(negative_sentences).get_embedding(embedding_names)

        return sentence_pair_embedding

    def _find_embedding_pair_in_memory(self,
                                       sentence: Sentence,
                                       sample: str = "positive",  # 'positive' or 'negative' sample
                                       ) -> torch.FloatTensor:

        label = sentence.get_label(self._label_type).value

        positive_sentence = self.memory[label]
        negative_labels = [negative_label for negative_label in list(self.memory.keys()) if negative_label != label]
        negative_sentences = [self.memory[negative_label] for negative_label in negative_labels]

        sentence_pair_embedding: Optional[torch.FloatTensor] = None
        if sample == "positive" and torch.is_tensor(positive_sentence):
            sentence_pair_embedding = positive_sentence
        if sample == "negative" and negative_sentences:
            sentence_pair_embedding = random.choice(negative_sentences)

        # return raw embedding
        return sentence_pair_embedding

    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:

        # embed all sentences
        self.embeddings.embed(sentences)
        embedding_names = self.embeddings.get_names()

        embedding_pairs: List[Tuple[torch.tensor, torch.tensor]] = []
        labels: List[torch.tensor] = []

        for sentence in sentences:
            positive_sample, negative_sample = None, None
            if self.use_memory:
                positive_sample = self._find_embedding_pair_in_memory(sentence, sample="positive")
                negative_sample = self._find_embedding_pair_in_memory(sentence, sample="negative")

            # if no samples found in memory, take it from the current batch
            # this is necessary for the first forward pass even if using the memory
            if not torch.is_tensor(positive_sample):
                positive_sample = self._find_embedding_pair_in_batch(sentence, mini_batch=sentences, sample="positive")
            if not torch.is_tensor(negative_sample):
                negative_sample = self._find_embedding_pair_in_batch(sentence, mini_batch=sentences, sample="negative")

            # if no samples found
            if self.mix_memory:
                if not torch.is_tensor(positive_sample):
                    positive_sample = self._find_embedding_pair_in_memory(sentence, sample="positive")
                if not torch.is_tensor(negative_sample):
                    negative_sample = self._find_embedding_pair_in_memory(sentence, sample="negative")

            # add sentence pair from the same class and label 1
            if torch.is_tensor(positive_sample):
                embedding_pairs.append((sentence.get_embedding(embedding_names), positive_sample))
                labels.append(1)

            # add sentence pair from a different class and label 0
            if torch.is_tensor(negative_sample):
                embedding_pairs.append((sentence.get_embedding(embedding_names), negative_sample))
                labels.append(0)

        # refresh memory after each forward pass
        if self.use_memory or self.mix_memory:
            for sentence in sentences:
                # detach (no gradient updates) and store embeddings in memory
                self.memory[sentence.get_label().value] = sentence.embedding.clone().detach()

        # return MSE loss between sentence pair similarities and 0s and 1s
        return self._calculate_loss(embedding_pairs, torch.FloatTensor(labels))

    def _calculate_loss(self,
                        embedding_pairs: Tuple[torch.tensor, torch.tensor],
                        labels: torch.FloatTensor,
                        ) -> Tuple[torch.Tensor, int]:

        # put to gpu
        first_sentences = torch.stack([embedding_pair[0] for embedding_pair in embedding_pairs]).to(flair.device)
        second_sentences = torch.stack([embedding_pair[1] for embedding_pair in embedding_pairs]).to(flair.device)
        labels = labels.to(flair.device)

        # calculate cosine similarities for a full batch
        similarities = torch.nn.functional.cosine_similarity(first_sentences, second_sentences, dim=1)

        # MSE loss between cosine similarities and 0s (pairs of different class) and 1s (pairs of same class)
        loss = self.loss_function(similarities, labels)

        return loss, first_sentences.shape[0]

    def _embed_training_set(self, mini_batch_size: int = 32):
        loader = DataLoader(self.train_corpus, batch_size=mini_batch_size, num_workers=0)
        for batch in loader:
            self.embeddings.embed(batch)

    def _k_nearest_neighbor(self, sentence: Sentence) -> str:
        """KNN classification for a single sentence"""

        # get embedding for a given sentence
        embedding_names = self.embeddings.get_names()
        sentence_embedding = sentence.get_embedding(embedding_names)

        # gather embeddings for all training instances
        train_set_embeddings = [training_sample.get_embedding(embedding_names)
                                for training_sample in self.train_corpus]
        train_set_embeddings = torch.stack(train_set_embeddings)

        # calculate cos similarity between given sentence and all training instances
        similarities = torch.nn.functional.cosine_similarity(sentence_embedding, train_set_embeddings, dim=1)

        # sort by top k (nearest neighbours) and get their labels
        _, top_k_idx = similarities.topk(k=self.knn)
        closest_labels = [self.train_corpus[sentence_id].get_label(self._label_type).value
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

        # embed full training set if needed
        embedding_names = self.embeddings.get_names()
        is_train_set_already_embedded = len(self.train_corpus[0].get_embedding(embedding_names)) != 0
        if not is_train_set_already_embedded:
            self._embed_training_set(mini_batch_size=mini_batch_size)

        # embed sentences to be predicted
        loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=0)
        for batch in loader:
            self.embeddings.embed(batch)

        # perform KNN to assign each new sentence a class
        for sentence in sentences:
            predicted_label = self._k_nearest_neighbor(sentence)
            sentence.add_label(typename=label_name, value=predicted_label)

        # KNN predictions do not have loss value
        if return_loss:
            return 0

    def evaluate(self, data_points, mini_batch_size: int, **kwargs):
        result = super().evaluate(data_points, mini_batch_size=mini_batch_size, **kwargs)

        # clear embeddings from the training set after each epoch
        embedding_names = self.embeddings.get_names()
        for sentence in self.train_corpus:
            sentence.clear_embeddings(embedding_names)

        return result
