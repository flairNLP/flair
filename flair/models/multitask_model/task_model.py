from typing import Union, List, Optional
from pathlib import Path

import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import flair.nn
from flair.data import Sentence, Dictionary, Dataset
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings, DocumentEmbeddings
from flair.training_utils import Metric, Result, store_embeddings
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
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

        # Dropouts
        # Dropout specific attributes
        self.use_dropout = True if dropout > 0.0 else False
        self.use_word_dropout = True if word_dropout > 0.0 else False
        self.use_locked_dropout = True if locked_dropout > 0.0 else False

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
            self.rnn = self.RNN(rnn_type, rnn_layers,  hidden_size, bidirectional, rnn_input_dim=embedding_dim)
        else:
            self.linear = torch.nn.Linear(embedding_dim, embedding_dim)

        num_directions = 2 if bidirectional else 1
        if use_crf:
            self.crf = CRF(hidden_size * num_directions, self.tagset_size)
            self.viterbi_loss = ViterbiLoss(tag_dictionary)
            self.viterbi_decoder = ViterbiDecoder(tag_dictionary)
        else:
            self.linear2tag = torch.nn.Linear(hidden_size * num_directions, len(tag_dictionary))
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

        with torch.no_grad():

            features, lengths = self.forward(sentences)

            if return_loss:
                loss = self.loss(features, sentences, lengths)

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
            return loss

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
            embeddings: DocumentEmbeddings,
            label_dictionary: Dictionary
    ):
        super(TextClassificationTask, self).__init__()
        pass

    def forward(self):
        pass