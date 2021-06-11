import logging
from pathlib import Path
from typing import List, Union, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import numpy as np

import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import flair.nn
import flair.embeddings
from flair.data import Dictionary, Sentence, Label, DataPoint, Relation
from flair.datasets import SentenceDataset, DataLoader
from flair.file_utils import cached_path
from flair.training_utils import convert_labels_to_one_hot, Result, store_embeddings

log = logging.getLogger("flair")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RelationClassifier(flair.nn.Model):
    """
    Text Classification Model
    The model takes word embeddings, puts them into an RNN to obtain a text representation, and puts the
    text representation in the end into a linear layer to get the actual class label.
    The model can handle single and multi class data sets.
    """

    def __init__(
            self,
            hidden_size: int,
            token_embeddings: flair.embeddings.TokenEmbeddings,
            label_dictionary: Dictionary,
            label_type: str = None,
            span_label_type: str = None,
            multi_label: bool = None,
            multi_label_threshold: float = 0.5,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
    ):
        """
        Initializes a RelationClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """

        super(RelationClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.token_embeddings: flair.embeddings.TokenEmbeddings = token_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.label_type = label_type
        self.span_label_type = span_label_type

        if multi_label is not None:
            self.multi_label = multi_label
        else:
            self.multi_label = self.label_dictionary.multi_label

        self.multi_label_threshold = multi_label_threshold

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.label_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.label_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        self.head_mlp = MLP(self.token_embeddings.embedding_length, hidden_dim=self.hidden_size, output_dim=self.hidden_size, num_layers=2)
        self.tail_mlp = MLP(self.token_embeddings.embedding_length, hidden_dim=self.hidden_size, output_dim=self.hidden_size, num_layers=2)

        self.decoder = nn.Linear(
            2*self.hidden_size, len(self.label_dictionary)
        )

        nn.init.xavier_uniform_(self.decoder.weight)

        if self.multi_label:
            self.loss_function = nn.BCEWithLogitsLoss(weight=self.loss_weights)
        else:
            self.loss_function = nn.CrossEntropyLoss(weight=self.loss_weights)

        # auto-spawn on GPU if available
        self.to(flair.device)

    def forward(self, sentences):

        self.token_embeddings.embed(sentences)

        relation_scores = []

        for sentence in sentences:
            spans = sentence.get_spans(self.span_label_type)

            span_embeddings = []
            for span in spans:
                span_embeddings.append(span.tokens[0].get_embedding().unsqueeze(0))

            span_embeddings = torch.cat(span_embeddings, dim=0)  # [num_rels_i x emb_dim]

            num_rels = span_embeddings.shape[0]
            head_embeddings = self.head_mlp(span_embeddings).unsqueeze(1).expand(num_rels, num_rels, self.hidden_size)  # [num_rels_i x num_rels_i x hidden_size]
            tail_embeddings = self.tail_mlp(span_embeddings).unsqueeze(0).expand(num_rels, num_rels, self.hidden_size)  # [num_rels_i x num_rels_i x hidden_size]

            head_tail_pairs = torch.cat([head_embeddings, tail_embeddings], dim=-1)  # [num_rels_i x num_rels_i x 2*hidden_size]

            sentence_relation_scores = self.decoder(head_tail_pairs)  # [num_rels_i x num_rels_i x num_labels]

            relation_scores.append(sentence_relation_scores)

        return relation_scores

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "token_embeddings": self.token_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "span_label_type": self.span_label_type,
            "multi_label": self.multi_label,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "hidden_size": self.hidden_size,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        label_type = None if "label_type" not in state.keys() else state["label_type"]
        span_label_type = None if "span_label_type" not in state.keys() else state["span_label_type"]

        model = RelationClassifier(
            hidden_size=state["hidden_size"],
            token_embeddings=state["token_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=label_type,
            span_label_type=span_label_type,
            multi_label=state["multi_label"],
            beta=beta,
            loss_weights=weights,
        )

        model.load_state_dict(state["state_dict"])
        return model

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence]
    ) -> torch.tensor:

        scores = self.forward(data_points)

        return self._calculate_loss(scores, data_points)

    def _calculate_loss(self, scores, data_points):
        labels = self._labels_to_one_hot(data_points) if self.multi_label \
            else self._labels_to_indices(data_points)

        scores_flattened = torch.cat([s.view(-1, len(self.label_dictionary)) for s in scores], dim=0)

        return self.loss_function(scores_flattened, labels)

    def _forward_scores_and_loss(
            self, data_points: Union[List[Sentence], Sentence], return_loss=False):
        scores = self.forward(data_points)

        loss = None
        if return_loss:
            loss = self._calculate_loss(scores, data_points)

        return scores, loss

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
            label_name = self.label_type if self.label_type is not None else 'label'

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
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )

            reordered_sentences: List[Union[DataPoint, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            dataloader = DataLoader(
                dataset=SentenceDataset(reordered_sentences), batch_size=mini_batch_size
            )
            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            for batch in dataloader:
                for sentence in batch:
                    relation_dict = {}
                    for relation in sentence.relations:
                        relation_dict[relation.span_indices] = relation

                    spans = sentence.get_spans(self.span_label_type)
                    new_relations = []
                    for i in range(len(spans)):
                        for j in range(len(spans)):
                            head = spans[i]
                            tail = spans[j]
                            span_indices = (head.tokens[0].idx, head.tokens[-1].idx, tail.tokens[0].idx, tail.tokens[-1].idx)

                            if span_indices in relation_dict:
                                relation = relation_dict[span_indices]
                            else:
                                relation = Relation(head, tail)
                                if relation_dict:
                                    relation.set_label(self.label_type, value="N")

                            new_relations.append(relation)

                    sentence.relations = new_relations

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                # stop if all sentences are empty
                if not batch:
                    continue

                scores, loss = self._forward_scores_and_loss(batch, return_loss)

                if return_loss:
                    overall_loss += loss

                predicted_labels = self._obtain_labels(scores, predict_prob=multi_class_prob)

                for (sentence, labels) in zip(batch, predicted_labels):
                    for relation, relation_labels in zip(sentence.relations, labels):
                        for label in relation_labels:
                            if self.multi_label or multi_class_prob:
                                relation.add_label(label_name, label.value, label.score)
                            else:
                                relation.set_label(label_name, label.value, label.score)

                # clearing token embeddings to save memory
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
            main_score_type: Tuple[str, str]=("micro avg", 'f1-score'),
            return_predictions: bool = False
    ) -> (Result, float):


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
                [relation.remove_labels('predicted') for sentence in batch for relation in sentence.relations]

                # predict for batch
                loss = self.predict(batch,
                                    embedding_storage_mode=embedding_storage_mode,
                                    mini_batch_size=mini_batch_size,
                                    label_name='predicted',
                                    return_loss=True)

                eval_loss += loss

                # get the gold labels
                true_values_for_batch = [relation.get_labels(self.label_type) for sentence in batch for relation in sentence.relations]

                # get the predicted labels
                predictions = [relation.get_labels('predicted') for sentence in batch for relation in sentence.relations]

                # for sentence, prediction, true_value in zip(
                #         sentences_for_batch,
                #         predictions,
                #         true_values_for_batch,
                # ):
                #     eval_line = "{}\t{}\t{}\n".format(
                #         sentence, true_value, prediction
                #     )
                #     lines.append(eval_line)


                for predictions_for_sentence, true_values_for_sentence in zip(
                        predictions, true_values_for_batch
                ):

                    true_values_for_sentence = [label.value for label in true_values_for_sentence]
                    predictions_for_sentence = [label.value for label in predictions_for_sentence]

                    y_true_instance = np.zeros(len(self.label_dictionary), dtype=int)
                    for i in range(len(self.label_dictionary)):
                        if self.label_dictionary.get_item_for_index(i) in true_values_for_sentence:
                            y_true_instance[i] = 1
                    y_true.append(y_true_instance.tolist())

                    y_pred_instance = np.zeros(len(self.label_dictionary), dtype=int)
                    for i in range(len(self.label_dictionary)):
                        if self.label_dictionary.get_item_for_index(i) in predictions_for_sentence:
                            y_pred_instance[i] = 1
                    y_pred.append(y_pred_instance.tolist())

                store_embeddings(batch, embedding_storage_mode)

            # remove predicted labels if return_predictions is False
            # Problem here: the predictions are only contained in sentences if it was chosen memory_mode="full" during
            # creation of the ClassificationDataset in the ClassificationCorpus creation. If the ClassificationCorpus has
            # memory mode "partial", then the predicted labels are not contained in sentences in any case so the following
            # optional removal has no effect. Predictions won't be accessible outside the eval routine in this case regardless
            # whether return_predictions is True or False. TODO: fix this

            if not return_predictions:
                for sentence in sentences:
                    for relation in sentence.relations:
                        relation.annotation_layers['predicted'] = []

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make "classification report"
            target_names = []
            for i in range(len(self.label_dictionary)):
                target_names.append(self.label_dictionary.get_item_for_index(i))

            classification_report = metrics.classification_report(y_true, y_pred, digits=4,
                                                                  target_names=target_names, zero_division=0)
            classification_report_dict = metrics.classification_report(y_true, y_pred, digits=4,
                                                                       target_names=target_names, zero_division=0, output_dict=True)

            # get scores
            micro_f_score = round(metrics.fbeta_score(y_true, y_pred, beta=self.beta, average='micro', zero_division=0),
                                  4)
            accuracy_score = round(metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(metrics.fbeta_score(y_true, y_pred, beta=self.beta, average='macro', zero_division=0),
                                  4)
            precision_score = round(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0), 4)
            recall_score = round(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0), 4)

            detailed_result = (
                    "\nResults:"
                    f"\n- F-score (micro) {micro_f_score}"
                    f"\n- F-score (macro) {macro_f_score}"
                    f"\n- Accuracy {accuracy_score}"
                    '\n\nBy class:\n' + classification_report
            )

            # line for log file
            if not self.multi_label:
                log_header = "ACCURACY"
                log_line = f"\t{accuracy_score}"
            else:
                log_header = "PRECISION\tRECALL\tF1\tACCURACY"
                log_line = f"{precision_score}\t" \
                           f"{recall_score}\t" \
                           f"{macro_f_score}\t" \
                           f"{accuracy_score}"

            result = Result(
                main_score=classification_report_dict[main_score_type[0]][main_score_type[1]],
                log_line=log_line,
                log_header=log_header,
                detailed_results=detailed_result,
                classification_report=classification_report_dict
            )

            eval_loss /= batch_count

            return result, eval_loss

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                "Ignore {} sentence(s) with no tokens.".format(
                    len(sentences) - len(filtered_sentences)
                )
            )
        return filtered_sentences

    def _obtain_labels(
            self, scores: List[List[float]], predict_prob: bool = False
    ) -> List[List[Label]]:
        """
        Predicts the labels of sentences.
        :param scores: the prediction scores from the model
        :return: list of predicted labels
        """
        if self.multi_label:
            return [self._get_multi_label(s) for s in scores]

        elif predict_prob:
            return [self._predict_label_prob(s) for s in scores]

        return [self._get_single_label(s) for s in scores]

    def _get_multi_label(self, label_scores) -> List[Label]:
        labels = []

        sigmoid = torch.nn.Sigmoid()

        results = list(map(lambda x: sigmoid(x), label_scores))
        for idx, conf in enumerate(results):
            if conf > self.multi_label_threshold:
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))

        return labels

    def _get_single_label(self, label_scores) -> List[Label]:
        num_relations = label_scores.shape[0]
        softmax = torch.nn.functional.softmax(label_scores.view(num_relations*num_relations, -1), dim=-1)
        conf, idx = torch.max(softmax, dim=-1)

        labels = []
        for c, i in zip(conf, idx):
            label = self.label_dictionary.get_item_for_index(i.item())
            labels.append([Label(label, c.item())])

        return labels

    def _predict_label_prob(self, label_scores) -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        label_probs = []
        for idx, conf in enumerate(softmax):
            label = self.label_dictionary.get_item_for_index(idx)
            label_probs.append(Label(label, conf.item()))
        return label_probs

    def _labels_to_one_hot(self, sentences: List[Sentence]):

        label_list = []
        for sentence in sentences:
            label_list.append([label.value for label in sentence.get_labels(self.label_type)])

        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0).to(flair.device)
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices: List[int] = []
        for sentence in sentences:
            relation_dict = {}
            for relation in sentence.relations:
                relation_dict[relation.span_indices] = relation

            spans = sentence.get_spans(self.span_label_type)
            for i in range(len(spans)):
                for j in range(len(spans)):
                    head = spans[i]
                    tail = spans[j]
                    span_indices = (head.tokens[0].idx, head.tokens[-1].idx, tail.tokens[0].idx, tail.tokens[-1].idx)

                    label = "N"
                    if span_indices in relation_dict:
                        relation = relation_dict[span_indices]
                        label = relation.get_labels(self.label_type)[0].value

                    indices.append(self.label_dictionary.get_idx_for_item(label))

        vec = torch.tensor(indices).to(flair.device)

        return vec

    @staticmethod
    def _fetch_model(model_name) -> str:
        model_map = {}

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name

    def __str__(self):
        return super(flair.nn.Model, self).__str__().rstrip(')') + \
               f'  (beta): {self.beta}\n' + \
               f'  (weights): {self.weight_dict}\n' + \
               f'  (weight_tensor) {self.loss_weights}\n)'
