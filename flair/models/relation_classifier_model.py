import logging
from typing import List, Union, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import flair.nn
import flair.embeddings
from flair.data import Dictionary, Sentence, DataPoint, RelationLabel, Span
from flair.datasets import SentenceDataset, DataLoader
from flair.training_utils import store_embeddings

log = logging.getLogger("flair")


class RelationClassifier(flair.nn.Classifier):

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
            dropout_value: float = 0.0,
    ):
        """
        Initializes a RelationClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """

        super(RelationClassifier, self).__init__()

        self.token_embeddings: flair.embeddings.TokenEmbeddings = token_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.label_dictionary.add_item('O')
        self._label_type = label_type
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
        # self.loss_function = flair.nn.FocalLoss(gamma=0.5, reduction='sum')
        # self.loss_function = flair.nn.DiceLoss(reduction='sum', with_logits=True, ohem_ratio=0.1)

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
            # print(sentence_relation_scores.size())
            # print(labels.size())
            # asd
            loss = self.loss_function(sentence_relation_scores, labels)
            # print(loss)

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

                scores_pairs_loss = self._internal_forward_scores_and_loss(batch,
                                                                           return_scores=True,
                                                                           return_loss=return_loss)
                scores = scores_pairs_loss[0]
                pairs = scores_pairs_loss[1]

                if return_loss:
                    overall_loss += scores_pairs_loss[2]

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
        model = RelationClassifier(
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

    @property
    def label_type(self):
        return self._label_type


def create_position_string(head: Span, tail: Span) -> str:
    return f"{head.id_text} -> {tail.id_text}"
