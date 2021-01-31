from typing import Union, List, Optional
from pathlib import Path

import torch
import torch.nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from flair.data import Sentence, Dictionary, Dataset
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
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
            beta: float = 1.0,
    ):
        super(SequenceTaggerTask, self).__init__()
        # Embedding specific
        embedding_dim: int = embeddings.embedding_length
        self.stop_token_emb = init_stop_tag_embedding(embedding_dim)
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.tag_type = tag_type

        # Evalulation specific attributes
        self.beta = beta

        # RNN specific attributes
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

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

    def forward_loss(self, sentence_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward loss function from abstract base class in flair
        :param sentences: list of sentences
        """
        # Prepare sentence tensor - extract embeddings
        tensor_list = list(map(lambda sent:
                               sent.get_sequence_tensor(add_stop_tag_embedding=True, stop_tag_embedding=self.stop_token_emb)
                               , sentences))
        sentence_tensor = pad_sequence(tensor_list, batch_first=True)

        # Sort sentences after length in decreasing order
        # Add one since we're having an <STOP> tag append to our sequence
        lengths = torch.LongTensor([len(sentence) + 1 for sentence in sentences])
        lengths = lengths.sort(dim=0, descending=True)
        sentence_tensor = sentence_tensor[lengths.indices]
        features, lengths = self.forward(sentence_tensor)
        return self.loss(features, sentence_tensor, lengths)

    def forward(self, sentence_tensor: torch.Tensor, lengths) -> (torch.Tensor, torch.Tensor):
        """
        Forward method of base multitask model
        :param sentences: list of sentences
        """
        if self.use_rnn:
            packed = pack_padded_sequence(sentence_tensor, list(lengths.values), batch_first=True)
            rnn_output, hidden = self.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)
        else:
            pass

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.use_crf:
            features = self.crf(sentence_tensor)
        else:
            features = self.linear2tag(sentence_tensor)

        return features, lengths

    def loss(self, features: torch.Tensor, sentences: Union[List[Sentence], Sentence], lengths: tuple) -> torch.Tensor:
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
        sentences: Union[List[Sentence], Dataset],
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: int = 8
    ) -> (Result, float):

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        eval_loss = 0
        batch_no: int = 0

        # check if we need to evaluate span of tags
        if self._requires_span_F1_evaluation():
            metric = Metric("Evaluation", beta=self.beta)
        else:
            metric = Dictionary(add_unk=False)

        # Predict and adds predicted labels to each sentence in batch
        # performs forward(), loss(), ViterbiDecoder.decode() and adds predicted label to batch
        for batch in data_loader:

            loss = self.predict(batch,
                                embedding_storage_mode=embedding_storage_mode,
                                label_name='predicted',
                                return_loss=True)

            eval_loss += loss
            batch_no += 1

            # Add scores to metric
            for sentence in batch:

                # make list of gold tags
                gold_spans = sentence.get_spans(self.tag_type)
                gold_tags = [(span.tag, repr(span)) for span in gold_spans]

                # make list of predicted tags
                predicted_spans = sentence.get_spans("predicted")
                predicted_tags = [(span.tag, repr(span)) for span in predicted_spans]

                # check for true positives, false positives and false negatives
                for tag, prediction in predicted_tags:
                    if (tag, prediction) in gold_tags:
                        metric.add_tp(tag)
                    else:
                        metric.add_fp(tag)

                for tag, gold in gold_tags:
                    if (tag, gold) not in predicted_tags:
                        metric.add_fn(tag)

        eval_loss /= batch_no

        # Log results
        detailed_result = (
            "\nResults:"
            f"\n- F1-score (micro) {metric.micro_avg_f_score():.4f}"
            f"\n- F1-score (macro) {metric.macro_avg_f_score():.4f}"
            '\n\nBy class:'
        )

        for class_name in metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                f"fn: {metric.get_fn(class_name)} - precision: "
                f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                f"f1-score: "
                f"{metric.f_score(class_name):.4f}"
            )

        result = Result(
            main_score=metric.micro_avg_f_score(),
            log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )

        return result, eval_loss

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
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

    def _requires_span_F1_evaluation(self) -> bool:
        span_F1 = False
        for item in self.tag_dictionary.get_items():
            if item.startswith('B-'):
                span_F1 = True
        return span_F1

class TextClassificationTask(torch.nn.Module):

    def __init__(
            self,
            embedding_length: int,
            tag_dictionary: Dictionary,
            tag_type: str,
            use_crf: bool = True,
            shared_rnn: bool = False):
        super(TextClassificationTask, self).__init__()
        self.stop_token_emb = init_stop_tag_embedding(embedding_length) # seq label specific
        self.tag_type = tag_type # specific for seq label or none for tc
        if use_crf: # seq label specific
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        # CRF specific attributes
        self.use_crf = use_crf  # seq label specific

    def forward(self):
        pass