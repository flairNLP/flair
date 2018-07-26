import warnings

import torch.autograd as autograd
import torch.nn as nn
import torch
import os
import numpy as np

from flair.file_utils import cached_path
from flair.data import Dictionary, Sentence, Token
from flair.embeddings import TextEmbeddings, TextLSTMEmbedder

from typing import List, Tuple, Union

from flair.trainers.util import convert_labels_to_one_hot


class TextClassifier(nn.Module):

    def __init__(self,
                 word_embeddings: List[TextEmbeddings],
                 hidden_states: int,
                 num_layers: int,
                 reproject_words: bool,
                 bidirectional: bool,
                 label_dictionary: Dictionary,
                 multi_label: bool):

        super(TextClassifier, self).__init__()

        self.word_embeddings = word_embeddings
        self.hidden_states = hidden_states
        self.num_layers = num_layers
        self.reproject_words = reproject_words
        self.bidirectional = bidirectional
        self.label_dictionary = label_dictionary
        self.multi_label = multi_label

        self.text_embeddings: TextLSTMEmbedder = TextLSTMEmbedder(word_embeddings, hidden_states, num_layers, reproject_words, bidirectional)

        self.decoder = nn.Linear(self.text_embeddings.embedding_length, len(self.label_dictionary))

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, sentences):
        self.text_embeddings.embed(sentences)

        text_embedding_list = [sentence.get_embedding().unsqueeze(0) for sentence in sentences]
        text_embedding_tensor = torch.cat(text_embedding_list, 0)

        if torch.cuda.is_available():
            text_embedding_tensor = text_embedding_tensor.cuda()

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def save(self, model_file: str):
        model_state = {
            'state_dict': self.state_dict(),
            'word_embeddings': self.word_embeddings,
            'hidden_states': self.hidden_states,
            'num_layers': self.num_layers,
            'reproject_words': self.reproject_words,
            'bidirectional': self.bidirectional,
            'label_dictionary': self.label_dictionary,
            'multi_label': self.multi_label,
        }
        torch.save(model_state, model_file, pickle_protocol=4)

    @classmethod
    def load_from_file(cls, model_file):

        # ATTENTION: suppressing torch serialization warnings. This needs to be taken out once we sort out recursive
        # serialization of torch objects
        warnings.filterwarnings("ignore")
        state = torch.load(model_file, map_location={'cuda:0': 'cpu'})
        warnings.filterwarnings("default")

        model = TextClassifier(
            word_embeddings=state['word_embeddings'],
            hidden_states=state['hidden_states'],
            num_layers=state['num_layers'],
            reproject_words=state['reproject_words'],
            bidirectional=state['bidirectional'],
            label_dictionary=state['label_dictionary'],
            multi_label=state['multi_label']
        )

        model.load_state_dict(state['state_dict'])
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def obtain_labels_and_loss(self, sentences: List[Sentence]) -> (List[List[str]], float):
        label_scores = self.forward(sentences)

        if torch.cuda.is_available():
            label_scores = label_scores.cuda()

        if self.multi_label:
            pred_labels = [self._get_multi_label(scores) for scores in label_scores]
            loss = self._calculate_multi_label_loss(label_scores, sentences)
        else:
            pred_labels = [self._get_single_label(scores) for scores in label_scores]
            loss = self._calculate_single_label_loss(label_scores, sentences)

        return pred_labels, loss

    def predict(self, sentences: List[Sentence], mini_batch_size=32) -> List[Sentence]:

        if type(sentences) is Sentence:
            sentences = [sentences]

        # make mini-batches
        batches = [sentences[x:x + mini_batch_size] for x in range(0, len(sentences), mini_batch_size)]

        for batch in batches:
            batch_labels = self._predict_labels(batch)

            for (sentence, labels) in zip(batch, batch_labels):
                sentence.labels = labels

        return sentences

    def _predict_labels(self, sentences: List[Sentence]) -> List[List[str]]:
        label_scores = self.forward(sentences)

        if torch.cuda.is_available():
            label_scores = label_scores.cuda()

        if self.multi_label:
            return [self._get_multi_label(scores) for scores in label_scores]

        return [self._get_single_label(scores) for scores in label_scores]

    def _get_multi_label(self, label_scores):
        labels = []

        sigmoid = torch.nn.Sigmoid()

        results = list(map(lambda x: sigmoid(x), label_scores[0]))
        for idx, conf in enumerate(results):
            label = self.label_dictionary.get_item_for_index(idx)
            labels.append(label)

        return labels

    def _get_single_label(self, label_scores):
        conf, idx = torch.max(label_scores[0], 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [label]

    def calculate_loss(self, sentences: List[Sentence]):
        label_scores = self.forward(sentences)

        if torch.cuda.is_available():
            label_scores = label_scores.cuda()

        if self.multi_label:
            return self._calculate_multi_label_loss(label_scores, sentences)

        return self._calculate_single_label_loss(label_scores, sentences)

    def _calculate_multi_label_loss(self, label_scores, batch):
        loss_function = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        return loss_function(sigmoid(label_scores), self._labels_to_one_hot(batch))

    def _calculate_single_label_loss(self, label_scores, sentences):
        loss_function = nn.CrossEntropyLoss()
        return loss_function(label_scores, self._labels_to_indices(sentences))

    def _labels_to_one_hot(self, batch):
        one_hot = convert_labels_to_one_hot(batch, self.label_dictionary)
        return torch.from_numpy(one_hot)

    def _labels_to_indices(self, sentences):

        indices = [
            torch.LongTensor([self.label_dictionary.get_idx_for_item(label) for label in sentence.labels])
            for sentence in sentences
        ]

        return torch.cat(indices, 0)