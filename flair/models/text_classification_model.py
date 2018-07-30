import warnings
from typing import List

import torch
import torch.nn as nn

import flair.embeddings
from flair.data import Dictionary, Sentence
from flair.training_utils import convert_labels_to_one_hot, clear_embeddings


class TextClassifier(nn.Module):
    """
    Text Classification Model
    The model takes word embeddings, puts them into an LSTM to obtain a text representation, and puts the
    text representation in the end into a linear layer to get the actual class label.
    The model can handle single and multi class data sets.
    """

    def __init__(self,
                 token_embeddings: List[flair.embeddings.TokenEmbeddings],
                 hidden_states: int,
                 num_layers: int,
                 reproject_words: bool,
                 bidirectional: bool,
                 label_dictionary: Dictionary,
                 multi_label: bool):

        super(TextClassifier, self).__init__()

        self.hidden_states = hidden_states
        self.num_layers = num_layers
        self.reproject_words = reproject_words
        self.bidirectional = bidirectional
        self.label_dictionary: Dictionary = label_dictionary
        self.multi_label = multi_label

        self.document_embeddings: flair.embeddings.DocumentLSTMEmbeddings = flair.embeddings.DocumentLSTMEmbeddings(
            token_embeddings, hidden_states, num_layers, reproject_words, bidirectional)

        self.decoder = nn.Linear(self.document_embeddings.embedding_length, len(self.label_dictionary))

        self._init_weights()

        if torch.cuda.is_available():
            self.cuda()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, sentences):
        self.document_embeddings.embed(sentences)

        text_embedding_list = [sentence.get_embedding().unsqueeze(0) for sentence in sentences]
        text_embedding_tensor = torch.cat(text_embedding_list, 0)

        if torch.cuda.is_available():
            text_embedding_tensor = text_embedding_tensor.cuda()

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def save(self, model_file: str):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = {
            'state_dict': self.state_dict(),
            'document_embeddings': self.document_embeddings,
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
        """
        Loads the model from the given file.
        :param model_file: the model file
        :return: the loaded text classifier model
        """

        # ATTENTION: suppressing torch serialization warnings. This needs to be taken out once we sort out recursive
        # serialization of torch objects
        warnings.filterwarnings("ignore")
        state = torch.load(model_file, map_location={'cuda:0': 'cpu'})
        warnings.filterwarnings("default")

        model = TextClassifier(
            token_embeddings=state['document_embeddings'],
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

    def predict(self, sentences: List[Sentence], mini_batch_size: int = 32, embeddings_in_memory: bool = True) -> List[Sentence]:
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :return: the list of sentences containing the labels
        """
        if type(sentences) is Sentence:
            sentences = [sentences]

        batches = [sentences[x:x + mini_batch_size] for x in range(0, len(sentences), mini_batch_size)]

        for batch in batches:
            batch_labels, _ = self.get_labels_and_loss(batch)

            for (sentence, labels) in zip(batch, batch_labels):
                sentence.labels = labels

            if not embeddings_in_memory:
                clear_embeddings(batch)

        return sentences

    def get_labels_and_loss(self, sentences: List[Sentence]) -> (List[List[str]], float):
        """
        Predicts the labels of sentences and calculates the loss.
        :param sentences: list of sentences
        :return: list of predicted labels and the loss value
        """
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

    def _get_multi_label(self, label_scores) -> List[str]:
        labels = []

        sigmoid = torch.nn.Sigmoid()

        results = list(map(lambda x: sigmoid(x), label_scores))
        for idx, conf in enumerate(results):
            label = self.label_dictionary.get_item_for_index(idx)
            labels.append(label)

        return labels

    def _get_single_label(self, label_scores) -> List[str]:
        conf, idx = torch.max(label_scores[0], 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [label]

    def _calculate_multi_label_loss(self, label_scores, sentences: List[Sentence]) -> float:
        loss_function = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        return loss_function(sigmoid(label_scores), self._labels_to_one_hot(sentences))

    def _calculate_single_label_loss(self, label_scores, sentences: List[Sentence]) -> float:
        loss_function = nn.CrossEntropyLoss()
        return loss_function(label_scores, self._labels_to_indices(sentences))

    def _labels_to_one_hot(self, sentences: List[Sentence]):
        label_list = [sentence.labels for sentence in sentences]
        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [
            torch.LongTensor([self.label_dictionary.get_idx_for_item(label) for label in sentence.labels])
            for sentence in sentences
        ]

        return torch.cat(indices, 0)