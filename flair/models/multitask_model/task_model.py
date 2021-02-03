from pathlib import Path
import numpy as np
from typing import Union, List, Optional, Dict

import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm

import flair.nn
from flair.data import Sentence, Dictionary, Dataset, DataPoint, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings, DocumentEmbeddings
from flair.training_utils import Metric, Result, store_embeddings, convert_labels_to_one_hot
from flair.models.sequence_tagger_model import START_TAG, STOP_TAG

from .crf import CRF
from .viterbi import ViterbiLoss, ViterbiDecoder
from .utils import init_stop_tag_embedding, get_tags_tensor

class SequenceTaggerTask(torch.nn.Module):

    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            use_rnn: bool = True,
            rnn: Optional[torch.nn.Module] = None,
            rnn_type: str = "LSTM",
            hidden_size: int = 256,
            rnn_layers: int = 1,
            bidirectional: bool = True,
            use_crf: bool = True,
            reproject_embeddings: bool = True,
            dropout: float = 0.0,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.5,
            beta: float = 1.0,
    ):
        super(SequenceTaggerTask, self).__init__()

        # Multitask logging info
        self.name = f"{self._get_name()} - Task: {tag_type}"

        # Embedding specific
        self.embeddings = embeddings
        embedding_dim: int = embeddings.embedding_length
        self.stop_token_emb = init_stop_tag_embedding(embedding_dim)
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.tag_type = tag_type

        # Evalulation specific attributes
        self.metric = Metric("Evaluation", beta=beta)
        self.beta = beta

        # RNN specific attributes
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type if not rnn else rnn._get_name()
        self.hidden_size = hidden_size if not rnn else rnn.hidden_size
        self.rnn_layers = rnn_layers if not rnn else rnn.num_layers
        self.bidirectional = bidirectional if not rnn else rnn.bidirectional

        # Dropouts
        # Dropout specific attributes
        self.use_dropout = True if dropout > 0.0 else False
        self.use_word_dropout = True if word_dropout > 0.0 else False
        self.use_locked_dropout = True if locked_dropout > 0.0 else False

        # Model layers
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            self.embedding2nn = torch.nn.Linear(embedding_dim, embedding_dim)

        if self.use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        if self.use_word_dropout:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if self.use_locked_dropout:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        # CRF specific
        self.use_crf = use_crf
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        # Model layers
        if use_rnn:
            if not rnn:
                self.rnn = self.RNN(rnn_type, rnn_layers,  hidden_size, bidirectional, rnn_input_dim=embedding_dim)
            else:
                self.rnn = rnn

            num_directions = 2 if self.bidirectional else 1
            hidden_output_dim = self.rnn.hidden_size * num_directions
        else:
            self.linear = torch.nn.Linear(embedding_dim, embedding_dim)
            hidden_output_dim = embedding_dim

        if use_crf:
            self.crf = CRF(hidden_output_dim, self.tagset_size)
            self.viterbi_loss = ViterbiLoss(tag_dictionary)
            self.viterbi_decoder = ViterbiDecoder(tag_dictionary)
        else:
            self.linear2tag = torch.nn.Linear(hidden_output_dim, self.tagset_size)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    @staticmethod
    def RNN(
            rnn_type: str,
            rnn_layers: int,
            hidden_size: int,
            bidirectional: bool,
            rnn_input_dim: int
    ):
        """
        Static wrapper function returning an RNN instance from PyTorch
        :param rnn_type: Type of RNN from torch.nn
        :param rnn_layers: number of layers to include
        :param hidden_size: hidden size of RNN cell
        :param bidirectional: If True, RNN cell is bidirectional
        :param rnn_input_dim: Input dimension to RNN cell
        """
        if rnn_type in ["LSTM", "GRU", "RNN"]:
            RNN = getattr(torch.nn, rnn_type)(
                rnn_input_dim,
                hidden_size,
                num_layers=rnn_layers,
                dropout=0.0 if rnn_layers == 1 else 0.5,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise Exception(f"Unknown RNN type: {rnn_type}. Please use either LSTM, GRU or RNN.")

        return RNN

    def forward_loss(self, sentences) -> torch.Tensor:
        """
        Forward loss function from abstract base class in flair
        :param sentences: list of sentences
        """
        features, lengths = self.forward(sentences)
        return self.loss(features, sentences, lengths)

    def forward(self, sentences) -> (torch.Tensor, torch.Tensor):
        """
        Forward method of base multitask model
        :param sentences: list of sentences
        """
        self.embeddings.embed(sentences)

        # Prepare sentence tensor - extract embeddings
        tensor_list = list(map(lambda sent: torch.cat((sent.get_sequence_tensor(), self.stop_token_emb.unsqueeze(0)), dim=0), sentences))
        sentence_tensor = pad_sequence(tensor_list, batch_first=True)

        lengths = torch.LongTensor([len(sentence) + 1 for sentence in sentences])
        lengths = lengths.sort(dim=0, descending=True)
        sentence_tensor = sentence_tensor[lengths.indices]

        # Feed-forward
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = pack_padded_sequence(sentence_tensor, list(lengths.values), batch_first=True)
            rnn_output, hidden = self.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)
        else:
            sentence_tensor = self.linear(sentence_tensor)

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.use_crf:
            features = self.crf(sentence_tensor)
        else:
            features = self.linear2tag(sentence_tensor)

        return features, lengths

    def loss(self, features: torch.Tensor, sentences: Union[List[Sentence], Sentence], lengths) -> torch.Tensor:
        """
        Loss function of multitask base model.
        :param features: Output features / CRF scores from feed-forward function
        :param sentences: list of sentences
        """

        # Preparation for loss function
        tags_tensor = get_tags_tensor(sentences, self.tag_dictionary, self.tag_type)
        # Sort tag tensor same order as features in decreasing order by length
        tags_tensor = tags_tensor[lengths.indices]

        if self.use_crf:
            loss = self.viterbi_loss(features, tags_tensor, lengths.values)
        else:
            loss = self.cross_entropy_loss(features.permute(0,2,1), tags_tensor)

        return loss

    def evaluate(
        self,
        sentences,
        embedding_storage_mode: str = "none",
    ) -> (Result, float):

        with torch.no_grad():

            loss = self.predict(sentences,
                                embedding_storage_mode=embedding_storage_mode,
                                label_name='predicted',
                                return_loss=True)

            self.calculate_metric(sentences)

            self.store_result()

        return loss

    def predict(
            self,
            sentences,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none",
    ):
        if label_name == None:
            label_name = self.tag_type

        features, lengths = self.forward(sentences)

        # features und lengths in der forward sortiert
        tags = self.viterbi_decoder.decode(features, lengths)

        # sorted sentences to match tags from decoder
        sentences = [sentences[i] for i in lengths.indices]

        # Add predicted labels to sentences
        for (sentence, sent_tags) in zip(sentences, tags):
            for (token, tag) in zip(sentence.tokens, sent_tags):
                token.add_tag_label(label_name, tag)

        # clearing token embeddings to save memory
        store_embeddings(sentences, storage_mode=embedding_storage_mode)

        if return_loss:
            return self.loss(features, sentences, lengths)

    def calculate_metric(self, sentences):

        if self._requires_span_F1_evaluation():
            self._span_F1_evaluation(sentences)
        else:
            self._tag_F1_evaluation(sentences)

    def _span_F1_evaluation(self, sentences):
        # Add scores to metric
        for sentence in sentences:

            # make list of gold tags
            gold_spans = sentence.get_spans(self.tag_type)
            gold_tags = [(span.tag, repr(span)) for span in gold_spans]

            # make list of predicted tags
            predicted_spans = sentence.get_spans("predicted")
            predicted_tags = [(span.tag, repr(span)) for span in predicted_spans]

            # check for true positives, false positives and false negatives
            for tag, prediction in predicted_tags:
                if (tag, prediction) in gold_tags:
                    self.metric.add_tp(tag)
                else:
                    self.metric.add_fp(tag)

            for tag, gold in gold_tags:
                if (tag, gold) not in predicted_tags:
                    self.metric.add_fn(tag)

    def _tag_F1_evaluation(self, sentences):
        for sentence in sentences:

            for token in sentence:
                # add gold tag
                gold_tag = token.get_tag(self.tag_type).value
                predicted_tag = token.get_tag('predicted').value

                if gold_tag == predicted_tag:
                    self.metric.add_tp(predicted_tag)
                else:
                    self.metric.add_fp(predicted_tag)
                    self.metric.add_fn(gold_tag)

    def store_result(self):
        # Log results
        detailed_result = (
            "\nResults:"
            f"\n- F1-score (micro) {self.metric.micro_avg_f_score():.4f}"
            f"\n- F1-score (macro) {self.metric.macro_avg_f_score():.4f}"
            '\n\nBy class:'
        )

        for class_name in self.metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {self.metric.get_tp(class_name)} - fp: {self.metric.get_fp(class_name)} - "
                f"fn: {self.metric.get_fn(class_name)} - precision: "
                f"{self.metric.precision(class_name):.4f} - recall: {self.metric.recall(class_name):.4f} - "
                f"f1-score: "
                f"{self.metric.f_score(class_name):.4f}"
            )

        self.result = Result(
            main_score=self.metric.micro_avg_f_score(),
            log_line=f"{self.metric.precision():.4f}\t{self.metric.recall():.4f}\t{self.metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
            multitask_id=self.name,
        )

    def _requires_span_F1_evaluation(self) -> bool:
        span_F1 = False
        for item in self.tag_dictionary.get_items():
            if item.startswith('B-'):
                span_F1 = True
        return span_F1

    def _reset_eval_metrics(self):
        self.metric = Metric("Evaluation", beta=self.beta)
        self.result = None

class TextClassificationTask(torch.nn.Module):

    def __init__(
            self,
            document_embeddings: flair.embeddings.DocumentEmbeddings,
            label_dictionary: Dictionary,
            label_type: str = None,
            multi_label: bool = None,
            multi_label_threshold: float = 0.5,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
    ):
        super(TextClassificationTask, self).__init__()
        # Multitask logging info
        self.name = f"{self._get_name()} - Task: {label_type}"

        # Label information
        self.document_embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.label_type = label_type

        if multi_label is not None:
            self.multi_label = multi_label
        else:
            self.multi_label = self.label_dictionary.multi_label
        self.multi_label_threshold = multi_label_threshold

        # Metric specific
        self.metric = Metric("Evaluation", beta=beta)
        self.beta = beta

        # Initial loss weights
        self.weight_dict = loss_weights
        if loss_weights is not None:
            self.init_loss_weights(loss_weights)
        else:
            self.loss_weights = None

        # Model layers
        self.decoder = torch.nn.Linear(
            self.document_embeddings.embedding_length, len(self.label_dictionary)
        )

        torch.nn.init.xavier_uniform_(self.decoder.weight)

        if self.multi_label:
            self.loss_function = torch.nn.BCEWithLogitsLoss(weight=self.loss_weights)
        else:
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.loss_weights)

    def init_loss_weights(self, loss_weights: torch.Tensor):
        n_classes = len(self.label_dictionary)
        weight_list = [1. for i in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]
        self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)

    def forward_loss(
            self, sentences: Union[List[Sentence], Sentence]
    ) -> torch.tensor:

        scores = self.forward(sentences)

        return self.loss(scores, sentences)

    def forward(self, sentences):

        self.document_embeddings.embed(sentences)

        embedding_names = self.document_embeddings.get_names()

        text_embedding_list = [
            sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in sentences
        ]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def loss(self, scores, sentences):

        if self.multi_label:
            labels = self._labels_to_one_hot(sentences)
        else:
            labels = self._labels_to_indices(sentences)

        return self.loss_function(scores, labels)

    def evaluate(
            self,
            sentences: Union[List[DataPoint], Dataset],
            embedding_storage_mode: str = "none",
    ) -> (Result, float):

        with torch.no_grad():

            loss = self.predict(sentences,
                                embedding_storage_mode=embedding_storage_mode,
                                label_name='predicted',
                                return_loss=True)

            self.calculate_metric(sentences)

            self.store_result()

        return loss

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            multi_class_prob: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none",
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param multi_class_prob : return probability for all class for multiclass
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name == None:
            label_name = self.label_type if self.label_type is not None else 'label'

        features = self.forward(sentences)

        loss = self.loss(features, sentences)

        predicted_labels = self._obtain_labels(features, predict_prob=multi_class_prob)

        for (sentence, labels) in zip(sentences, predicted_labels):
            for label in labels:
                if self.multi_label or multi_class_prob:
                    sentence.add_label(label_name, label.value, label.score)
                else:
                    sentence.set_label(label_name, label.value, label.score)

        # clearing token embeddings to save memory
        store_embeddings(sentences, storage_mode=embedding_storage_mode)

        if return_loss:
            return loss

    def _obtain_labels(self, scores: List[List[float]], predict_prob: bool = False) -> List[List[Label]]:

        if self.multi_label:
            return [self._get_multi_label(s) for s in scores]

        elif predict_prob:
            return [self._predict_label_prob(s) for s in scores]

        return [self._get_single_label(s) for s in scores]

    def _get_single_label(self, label_scores) -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        conf, idx = torch.max(softmax, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [Label(label, conf.item())]

    def _get_multi_label(self, label_scores) -> List[Label]:
        labels = []

        sigmoid = torch.nn.Sigmoid()

        results = list(map(lambda x: sigmoid(x), label_scores))
        for idx, conf in enumerate(results):
            if conf > self.multi_label_threshold:
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))

        return labels

    def calculate_metric(self, sentences):
        predicted_labels_batch = list(map(lambda sentence: sentence.get_labels('predicted'), sentences))
        list(map(lambda sentence: sentence.remove_labels('predicted'), sentences))
        gold_labels_batch = list(map(lambda sentence: sentence.get_labels(self.label_type), sentences))

        for (gold_labels, predicted_labels) in zip(gold_labels_batch, predicted_labels_batch):
            gold_labels = [label.value for label in gold_labels]
            predicted_labels = [label.value for label in predicted_labels]

            for prediction in predicted_labels:
                if prediction in gold_labels:
                    self.metric.add_tp(prediction)
                else:
                    self.metric.add_fp(prediction)

            for gold_label in gold_labels:
                if gold_label not in predicted_labels:
                    self.metric.add_fn(gold_label)

    def store_result(self):
        detailed_result = (
                "\nResults:"
                f"\n- F-score (micro) {self.metric.micro_avg_f_score():.4f}"
                f"\n- F-score (macro) {self.metric.macro_avg_f_score():.4f}"
                '\n\nBy class:'
        )

        for class_name in self.metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {self.metric.get_tp(class_name)} - fp: {self.metric.get_fp(class_name)} - "
                f"fn: {self.metric.get_fn(class_name)} - precision: "
                f"{self.metric.precision(class_name):.4f} - recall: {self.metric.recall(class_name):.4f} - "
                f"f1-score: "
                f"{self.metric.f_score(class_name):.4f}"
            )

        if not self.multi_label:
            log_header = "ACCURACY"
            log_line = f"\t{self.metric.accuracy():.4f}"
        else:
            log_header = "PRECISION\tRECALL\tF1\tACCURACY"
            log_line = f"{self.metric.precision()}\t" \
                       f"{self.metric.recall()}\t" \
                       f"{self.metric.macro_avg_f_score()}\t" \
                       f"{self.metric.accuracy()}"

        self.result = Result(
            main_score=self.metric.f_score(),
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            multitask_id=self.name
        )

    def _labels_to_one_hot(self, sentences: List[Sentence]):

        label_list = []
        for sentence in sentences:
            label_list.append([label.value for label in sentence.get_labels(self.label_type)])

        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0).to(flair.device)
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):

        indices = [
            torch.LongTensor(
                [
                    self.label_dictionary.get_idx_for_item(label.value)
                    for label in sentence.get_labels(self.label_type)
                ]
            )
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0).to(flair.device)

        return vec

    def _reset_eval_metrics(self):
        self.metric = Metric("Evaluation", beta=self.beta)
        self.result = None