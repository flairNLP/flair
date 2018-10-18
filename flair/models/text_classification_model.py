import warnings
import logging
from typing import List, Union

import torch
import torch.nn as nn

import flair.embeddings
from flair.data import Dictionary, Sentence, Label
from flair.training_utils import convert_labels_to_one_hot, clear_embeddings


log = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    """
    Text Classification Model
    The model takes word embeddings, puts them into an LSTM to obtain a text representation, and puts the
    text representation in the end into a linear layer to get the actual class label.
    The model can handle single and multi class data sets.
    """

    def __init__(self,
                 document_embeddings: flair.embeddings.DocumentEmbeddings,
                 label_dictionary: Dictionary,
                 multi_label: bool):

        super(TextClassifier, self).__init__()

        self.document_embeddings = document_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.multi_label = multi_label

        self.document_embeddings: flair.embeddings.DocumentLSTMEmbeddings = document_embeddings

        self.decoder = nn.Linear(self.document_embeddings.embedding_length, len(self.label_dictionary))

        self._init_weights()

        if multi_label:
            self.loss_function = nn.BCELoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()

        # auto-spawn on GPU if available
        if torch.cuda.is_available():
            self.cuda()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, sentences) -> List[List[float]]:
        self.document_embeddings.embed(sentences)

        text_embedding_list = [sentence.get_embedding() for sentence in sentences]
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
        if torch.cuda.is_available():
            state = torch.load(model_file)
        else:
            state = torch.load(model_file, map_location={'cuda:0': 'cpu'})
        warnings.filterwarnings("default")

        model = TextClassifier(
            document_embeddings=state['document_embeddings'],
            label_dictionary=state['label_dictionary'],
            multi_label=state['multi_label']
        )

        model.load_state_dict(state['state_dict'])
        model.eval()
        return model

    def predict(self, sentences: Union[Sentence, List[Sentence]], mini_batch_size: int = 32) -> List[Sentence]:
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :return: the list of sentences containing the labels
        """
        if type(sentences) is Sentence:
            sentences = [sentences]

        filtered_sentences = self._filter_empty_sentences(sentences)

        batches = [filtered_sentences[x:x + mini_batch_size] for x in range(0, len(filtered_sentences), mini_batch_size)]

        for batch in batches:
            scores = self.forward(batch)
            predicted_labels = self.obtain_labels(scores)

            for (sentence, labels) in zip(batch, predicted_labels):
                sentence.labels = labels

            clear_embeddings(batch)

        return sentences

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning('Ignore {} sentence(s) with no tokens.'.format(len(sentences) - len(filtered_sentences)))
        return filtered_sentences

    def calculate_loss(self, scores: List[List[float]], sentences: List[Sentence]) -> float:
        """
        Calculates the loss.
        :param scores: the prediction scores from the model
        :param sentences: list of sentences
        :return: loss value
        """
        if self.multi_label:
            return self._calculate_multi_label_loss(scores, sentences)

        return self._calculate_single_label_loss(scores, sentences)

    def obtain_labels(self, scores: List[List[float]]) -> List[List[Label]]:
        """
        Predicts the labels of sentences.
        :param scores: the prediction scores from the model
        :return: list of predicted labels
        """

        if self.multi_label:
            return [self._get_multi_label(s) for s in scores]

        return [self._get_single_label(s) for s in scores]

    def _get_multi_label(self, label_scores) -> List[Label]:
        labels = []

        sigmoid = torch.nn.Sigmoid()

        results = list(map(lambda x: sigmoid(x), label_scores))
        for idx, conf in enumerate(results):
            if conf > 0.5:
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))

        return labels

    def _get_single_label(self, label_scores) -> List[Label]:
        conf, idx = torch.max(label_scores, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [Label(label, conf.item())]

    def _calculate_multi_label_loss(self, label_scores, sentences: List[Sentence]) -> float:
        sigmoid = nn.Sigmoid()
        return self.loss_function(sigmoid(label_scores), self._labels_to_one_hot(sentences))

    def _calculate_single_label_loss(self, label_scores, sentences: List[Sentence]) -> float:
        return self.loss_function(label_scores, self._labels_to_indices(sentences))

    def _labels_to_one_hot(self, sentences: List[Sentence]):
        label_list = [sentence.get_label_names() for sentence in sentences]
        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [
            torch.LongTensor([self.label_dictionary.get_idx_for_item(label.value) for label in sentence.labels])
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0)
        if torch.cuda.is_available():
            vec = vec.cuda()

        return vec