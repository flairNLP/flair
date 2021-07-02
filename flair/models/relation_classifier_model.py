from itertools import compress
import logging
from pathlib import Path
from typing import List, Union, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import numpy as np

import sklearn.metrics as skmetrics
import flair.nn
import flair.embeddings
from flair.data import Dictionary, Sentence, DataPoint, RelationLabel, Span
from flair.datasets import SentenceDataset, DataLoader
from flair.training_utils import Result, store_embeddings

log = logging.getLogger("flair")


class RelationClassifierLinear(flair.nn.Model):

    def __init__(
            self,
            token_embeddings: flair.embeddings.TokenEmbeddings,
            label_dictionary: Dictionary,
            label_type: str = None,
            span_label_type: str = None,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
            use_gold_spans: bool = True,
            pooling_operation: str = "first_last",
            dropout_value: float = 0.5,
    ):
        """
        Initializes a RelationClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """

        super(RelationClassifierLinear, self).__init__()

        self.token_embeddings: flair.embeddings.TokenEmbeddings = token_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.label_dictionary.add_item('O')
        self.label_type = label_type
        self.span_label_type = span_label_type

        self.beta = beta
        self.use_gold_spans = use_gold_spans
        self.pooling_operation = pooling_operation

        self.dropout_value = dropout_value

        self.dropout = torch.nn.Dropout(dropout_value)

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.label_dictionary)
            weight_list = [1.0 for i in range(n_classes)]
            for i, tag in enumerate(self.label_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        relation_representation_length = 2 * token_embeddings.embedding_length
        if self.pooling_operation == 'first_last':
            relation_representation_length *= 2

        self.decoder = nn.Linear(relation_representation_length, len(self.label_dictionary))

        nn.init.xavier_uniform_(self.decoder.weight)

        self.loss_function = nn.CrossEntropyLoss(weight=self.loss_weights)

        # auto-spawn on GPU if available
        self.to(flair.device)

    def _internal_forward_scores_and_loss(self,
                                          sentences: Union[List[DataPoint], DataPoint],
                                          return_scores: bool = True,
                                          return_loss: bool = True):

        self.token_embeddings.embed(sentences)

        entity_pairs = []
        relation_embeddings = []
        indices = []

        for sentence in sentences:

            # super lame: make dictionary to find relation annotations for a given entity pair
            relation_dict = {}
            for relation_label in sentence.get_labels(self.label_type):
                relation_label: RelationLabel = relation_label
                relation_dict[create_position_string(relation_label.head, relation_label.tail)] = relation_label

            # get all entities
            spans = sentence.get_spans(self.span_label_type)

            # get embedding for each entity
            span_embeddings = []
            for span in spans:
                if self.pooling_operation == "first":
                    span_embeddings.append(span.tokens[0].get_embedding())
                if self.pooling_operation == "first_last":
                    span_embeddings.append(torch.cat([span.tokens[0].get_embedding(), span.tokens[-1].get_embedding()]))

            # go through cross product of entities, for each pair concat embeddings
            for span, embedding in zip(spans, span_embeddings):
                for span_2, embedding_2 in zip(spans, span_embeddings):
                    if span == span_2: continue

                    position_string = create_position_string(span, span_2)

                    # get gold label for this relation (if one exists)
                    if position_string in relation_dict:
                        relation_label: RelationLabel = relation_dict[position_string]
                        label = relation_label.value
                    # if using gold spans only, skip all entity pairs that are not in gold data
                    elif self.use_gold_spans:
                        continue
                    else:
                        # if no gold label exists, and all spans are used, label defaults to 'O' (no relation)
                        label = 'O'

                    indices.append(self.label_dictionary.get_idx_for_item(label))

                    relation_embeddings.append(torch.cat([embedding, embedding_2]))

                    entity_pairs.append((span, span_2))

        all_relations = torch.stack(relation_embeddings)

        all_relations = self.dropout(all_relations)

        sentence_relation_scores = self.decoder(all_relations)

        labels = torch.tensor(indices).to(flair.device)

        if return_loss:
            loss = self.loss_function(sentence_relation_scores, labels)

        if return_loss and not return_scores:
            return loss, len(labels)

        if return_scores and not return_loss:
            return sentence_relation_scores, entity_pairs

        if return_scores and return_loss:
            return sentence_relation_scores, entity_pairs, loss,

    def forward_loss(self, sentences: Union[List[DataPoint], DataPoint]) -> torch.tensor:
        return self._internal_forward_scores_and_loss(sentences, return_scores=False, return_loss=True)

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size: int = 32,
            multi_class_prob: bool = False,
            verbose: bool = False,
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
        if label_name is None:
            label_name = self.label_type if self.label_type is not None else "label"

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, DataPoint):
                sentences = [sentences]

            # filter empty sentences
            if isinstance(sentences[0], DataPoint):
                sentences = [sentence for sentence in sentences if len(sentence) > 0]
            if len(sentences) == 0:
                return sentences

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True)

            reordered_sentences: List[Union[DataPoint, str]] = [sentences[index] for index in rev_order_len_index]

            dataloader = DataLoader(dataset=SentenceDataset(reordered_sentences), batch_size=mini_batch_size)
            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                # stop if all sentences are empty
                if not batch:
                    continue

                scores, pairs, loss = self._internal_forward_scores_and_loss(batch,
                                                                             return_scores=True,
                                                                             return_loss=return_loss)

                if return_loss:
                    overall_loss += loss

                softmax = torch.nn.functional.softmax(scores, dim=-1)
                conf, idx = torch.max(softmax, dim=-1)

                for pair, c, i in zip(pairs, conf, idx):
                    label = self.label_dictionary.get_item_for_index(i.item())

                    sentence: Sentence = pair[0][0].sentence

                    relation_label = RelationLabel(value=label, score=c.item(), head=pair[0], tail=pair[1])
                    sentence.add_complex_label(label_name,
                                               relation_label)

                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def evaluate(
            self,
            sentences: Union[List[DataPoint], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
            return_predictions: bool = False,
    ) -> Result:

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # use scikit-learn to evaluate
        y_true = []
        y_pred = []

        with torch.no_grad():
            eval_loss = 0

            lines: List[str] = []
            batch_count: int = 0

            for batch in data_loader:
                batch_count += 1

                # remove previously predicted labels
                [sentence.remove_labels('predicted') for sentence in batch]

                # predict for batch
                loss = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    label_name="predicted",
                    return_loss=True,
                )

                eval_loss += loss

                # get the gold labels
                all_spans: List[str] = []
                true_values_for_batch = {}
                for s_id, sentence in enumerate(batch):
                    for relation_label in sentence.get_labels(self.label_type):
                        position_string = str(s_id) + ': ' + create_position_string(relation_label.head,
                                                                                    relation_label.tail)
                        true_values_for_batch[position_string] = relation_label
                        if position_string not in all_spans:
                            all_spans.append(position_string)

                # get the predicted labels
                predictions = {}
                for s_id, sentence in enumerate(batch):
                    for relation_label in sentence.get_labels("predicted"):
                        position_string = str(s_id) + ': ' + create_position_string(relation_label.head,
                                                                                    relation_label.tail)
                        predictions[position_string] = relation_label
                        if position_string not in all_spans:
                            all_spans.append(position_string)

                ordered_ground_truth = []
                ordered_predictions = []

                for span in all_spans:

                    true_value = true_values_for_batch[span] if span in true_values_for_batch else 'O'
                    prediction = predictions[span] if span in predictions else 'O'

                    ordered_ground_truth.append(true_value)
                    ordered_predictions.append(prediction)

                    eval_line = f"{span}\t{true_value.value}\t{prediction.value}\n"
                    lines.append(eval_line)

                    true_idx = self.label_dictionary.get_idx_for_item(true_value.value)
                    y_true_instance = np.zeros(len(self.label_dictionary), dtype=int)
                    for i in range(len(self.label_dictionary)):
                        y_true_instance[true_idx] = 1
                    y_true.append(y_true_instance.tolist())

                    pred_idx = self.label_dictionary.get_idx_for_item(prediction.value)
                    y_pred_instance = np.zeros(len(self.label_dictionary), dtype=int)
                    for i in range(len(self.label_dictionary)):
                        y_pred_instance[pred_idx] = 1
                    y_pred.append(y_pred_instance.tolist())

                store_embeddings(batch, embedding_storage_mode)

            if not return_predictions:
                for sentence in sentences:
                    for relation in sentence.relations:
                        relation.annotation_layers["predicted"] = []

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make "classification report"
            target_names = []
            labels = []
            for i in range(len(self.label_dictionary)):
                label_name = self.label_dictionary.get_item_for_index(i)
                if label_name == 'O': continue
                target_names.append(label_name)
                labels.append(i)

            classification_report = skmetrics.classification_report(
                y_true, y_pred, digits=4, target_names=target_names, zero_division=0, labels=labels,
            )

            classification_report_dict = skmetrics.classification_report(
                y_true, y_pred, digits=4, target_names=target_names, zero_division=0, output_dict=True, labels=labels,
            )

            # get scores
            accuracy_score = round(skmetrics.accuracy_score(y_true, y_pred), 4)

            precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
            recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
            micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            detailed_result = (
                    "\nResults:"
                    f"\n- F-score (micro) {micro_f_score}"
                    f"\n- F-score (macro) {macro_f_score}"
                    f"\n- Accuracy {accuracy_score}"
                    "\n\nBy class:\n" + classification_report
            )

            # line for log file
            log_header = "PRECISION\tRECALL\tF1\tACCURACY"
            log_line = f"{precision_score}\t" f"{recall_score}\t" f"{macro_f_score}\t" f"{accuracy_score}"

            eval_loss /= batch_count

            return Result(
                main_score=classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]],
                log_line=log_line,
                log_header=log_header,
                detailed_results=detailed_result,
                classification_report=classification_report_dict,
                loss=eval_loss,
            )

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "token_embeddings": self.token_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "span_label_type": self.span_label_type,
            "beta": self.beta,
            "loss_weights": self.loss_weights,
            "pooling_operation": self.pooling_operation,
            "dropout_value": self.dropout_value,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        model = RelationClassifierLinear(
            token_embeddings=state["token_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            span_label_type=state["span_label_type"],
            beta=state["beta"],
            loss_weights=state["loss_weights"],
            pooling_operation=state["pooling_operation"],
            dropout_value=state["dropout_value"],
        )

        model.load_state_dict(state["state_dict"])
        return model


def create_position_string(head: Span, tail: Span) -> str:
    return f"{head.id_text} -> {tail.id_text}"
