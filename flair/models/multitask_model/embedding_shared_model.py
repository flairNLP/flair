import logging
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional
from tqdm import tqdm

import torch.nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import Dataset

import flair.nn
from flair.data import Sentence, Dictionary, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Metric, Result, store_embeddings

from .task_model import SequenceTaggerTask, TextClassificationTask
from .utils import get_tags_tensor

log = logging.getLogger("flair")

class EmbeddingSharedModel(flair.nn.Model):
    """
    Basic multitask model.
    """

    def __init__(
        self,
        models: List[Dict[str:torch.nn.Module]],
        embeddings: TokenEmbeddings,
        reproject_embeddings: bool = True,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5
    ):
        """
        Initializes a base multitask model instance
        :param embeddings: embeddings which are used
        :param tag_dictionary: Dictionary of tags of task
        :param tag_type: Type of tag which is going to be predicted
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        :param use_rnn: if True, adds a RNN layer to the model. If False, simple linear layer.
        :param rnn_type: specifies the RNN type to use. Use "RNN", "GRU" or "LSTM". Default is "LSTM".
        :param hidden_size: hidden size of the rnn layer.
        :param rnn_layers: number of layers to use for RNN.
        :param bidirectional: If True, RNN layer is bidirectional. If False, single direction.
        :param use_crf: If True, use Conditonal Random Field. If False, use Dense Softmax layer for prediction.
        :param use_lm: If True, use additional language model during training for multitask purpose.
        :param dropout: Includes standard dropout, if provided attribute value is > 0.0
        :param word_dropout: Includes word_dropout, if provided attribute value is > 0.0
        :param locked_dropout: Includes locked_dropout, if provided attribute value is > 0.0
        :param beta: Parameter for F-beta score for evaluation and training annealing
        """

        super(EmbeddingSharedModel, self).__init__()

        # Embeddings and task specific attributes
        self.embeddings = embeddings
        embedding_dim: int = embeddings.embedding_length

        # Dropout specific attributes
        self.use_dropout = True if dropout > 0.0 else False
        self.use_word_dropout= True if word_dropout > 0.0 else False
        self.use_locked_dropout = True if locked_dropout > 0.0 else False

        if self.use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        if self.use_word_dropout:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if self.use_locked_dropout:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        # Model layers
        self.reproject_embeddings = reproject_embeddings
        if reproject_embeddings:
            if type(self.reproject_embeddings) == int:
                model_input_dim: int = self.reproject_embeddings
            self.embedding2nn = torch.nn.Linear(embedding_dim, model_input_dim)

        # Dynamically create task models from tag_spaces
        for model in models:
            self.__setattr__(model.get("task_id"), model.get("model"))

        self.to(flair.device)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Forward loss function from abstract base class in flair
        :param sentences: list of sentences
        """
        sentence_tensor, lengths = self.forward(sentences)
        features, lengths = self.model_forward(sentences)
        return self.loss(features, sentences, lengths)

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> (torch.Tensor, torch.Tensor):
        """
        Forward method of base multitask model
        :param sentences: list of sentences
        """

        # preparation of sentences and feed-forward part
        # embed sentences
        self.embeddings.embed(sentences)

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

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        return sentence_tensor, lengths

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
        out_path: Path = None,
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