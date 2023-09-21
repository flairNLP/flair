import logging
import pwd
from typing import Tuple, Dict, List, Callable

import numpy as np
import os
import json

import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

import flair
from flair.data import DT, Dictionary, Optional, Sentence, Span, Union
from flair.embeddings import DocumentEmbeddings, TokenEmbeddings
from flair.training_utils import store_embeddings

from pathlib import Path

from flair.nn.wikidata_utils import get_wikidata_categories

log = logging.getLogger("flair")

class WikidataLabelVerbalizer:
    def __init__(self,
                 label_dictionary,
                 verbalizations_path: Path,
                 use_wikidata_description: bool = False,
                 use_wikidata_classes: bool = True,
                 max_verbalization_length: int = 16,
                 add_occupation: bool = True,
                 add_field_of_work: bool =False
                 ):

        self.label_dictionary = label_dictionary
        self.verbalizations_path = verbalizations_path
        self.use_wikidata_description = use_wikidata_description
        self.use_wikidata_classes = use_wikidata_classes
        self.add_occupation = add_occupation
        self.add_field_of_work = add_field_of_work

        self.verbalized_labels = self.verbalize_all_labels(self, label_dictionary=self.label_dictionary)

        self.max_verbalization_length = max_verbalization_length

        if self.max_verbalization_length:
            verbalized_labels_truncated = []
            for l in self.verbalized_labels:
                s = flair.data.Sentence(l)
                s.tokens = s.tokens[:self.max_verbalization_length]
                verbalized_labels_truncated.append(s.text)
            self.verbalized_labels = verbalized_labels_truncated

    @staticmethod
    def verbalize_all_labels(self, label_dictionary: Dictionary) -> List[Sentence]:

        if os.path.exists(self.verbalizations_path):
            with open(self.verbalizations_path, 'r') as f:
                self.verbalizations = json.load(f)
                print(f"Found verbalizations for {len(self.verbalizations)} labels at  {self.verbalizations_path}")
        else:
            with open(self.verbalizations_path, 'w') as file:
                json.dump({}, file)
                self.verbalizations = {}
                print(f"Created empty verbalization dictionary at {self.verbalizations_path}")

        counter_newly_verbalized = 0
        verbalized_labels = []
        for byte_label, idx in label_dictionary.item2idx.items():
            str_label = byte_label.decode("utf-8")
            entity = str_label

            if entity == "O":
                verbalized = "outside"
            elif entity == "<unk>":
                verbalized = "unknown"
            else:
                if self.verbalizations:
                    verbalized = self.verbalizations.get(entity, None)
                if not verbalized:
                    wikipedia_label = entity
                    wikidata_info = get_wikidata_categories(wikipedia_label, method ="only_one_level_up",
                                                            add_occupation=self.add_occupation,
                                                            add_field_of_work=self.add_field_of_work)
                    wikidata_classes = wikidata_info["class_names"]
                    wikidata_title = wikidata_info["wikidata_title"]
                    wikidata_description = wikidata_info["wikibase_description"]
                    verbalized = wikidata_title
                    if self.use_wikidata_description:
                        verbalized += ", " + wikidata_description
                    if self.use_wikidata_classes:
                        verbalized += ", " + ", ".join(wikidata_classes)
                    #self.verbalizations[entity] = verbalized
                    counter_newly_verbalized +=1

            verbalized_labels.append(verbalized)
            self.verbalizations[entity] = verbalized
            print("Verbalized id", idx, ":", str_label, "->", verbalized)

            # save every n entities and final
            if idx % 100 == 0 or idx >= len(label_dictionary.item2idx.items()) -1:
                if self.verbalizations_path:
                    verbalizations_dict = self.verbalizations
                    with open(self.verbalizations_path, "w") as file:
                        json.dump(verbalizations_dict, file)

        print(f"--- Created verbalized labels for {len(verbalized_labels)} labels")
        print(f"--- Thereof newly verbalized:", counter_newly_verbalized)

        return verbalized_labels

    def verbalize_list_of_labels(self, list_of_labels):
        label_idx = [self.label_dictionary.get_idx_for_item(idx) for idx in list_of_labels]
        return [self.verbalized_labels[idx] for idx in label_idx ]


class DualEncoderSimilarityLoss(flair.nn.Classifier[Sentence]):
    """This model uses a dual encoder architecture where both inputs and labels (verbalized) are encoded with separate
    Transformers. It uses a similarity loss (e.g. cosine similarity) to push datapoints and true labels nearer together,
    and performs KNN like inference.
    Label verbalizations should be plugged in.
    """

    def __init__(
        self,
        token_encoder: TokenEmbeddings,
        label_encoder: DocumentEmbeddings,
        label_dictionary: Dictionary,
        pooling_operation: str = "average",
        label_type: str = "nel",
        dropout: float = 0.1,
        linear_size: int = 128,
        use_span_width_embeddings: bool = True,
        max_span_width: int = 8,
        custom_label_verbalizations = None,
        train_only_with_positive_labels = True,
        negative_sampling_factor: [int, bool] = 1,
        negative_sampling_strategy: str = "random",
        margin_loss: float = 0.5,
        threshold_in_prediction: float = 0.5,

    ):
        super().__init__()

        self.token_encoder = token_encoder
        self.label_encoder = label_encoder
        self.label_dictionary = label_dictionary
        self._label_type = label_type

        self.dropout = torch.nn.Dropout(dropout)
        self.label_start_linear = torch.nn.Linear(self.label_encoder.embedding_length, linear_size)
        self.label_end_linear = torch.nn.Linear(self.label_encoder.embedding_length, linear_size)
        self.label_span_linear = torch.nn.Linear(self.label_encoder.embedding_length, linear_size)

        token_embedding_size = self.token_encoder.embedding_length // 2
        self.token_start_linear = torch.nn.Linear(token_embedding_size, linear_size)
        self.token_end_linear = torch.nn.Linear(token_embedding_size, linear_size)

        self.max_span_width = max_span_width
        if use_span_width_embeddings:
            self.token_span_linear = torch.nn.Linear(self.token_encoder.embedding_length + linear_size, linear_size)
            assert (
                self.token_encoder.model.config.max_position_embeddings
                == self.label_encoder.model.config.max_position_embeddings
            ), (
                "The maximum position embeddings for the token encoder and label encoder must be the same when using "
                "span width embeddings."
            )
            span_width = self.token_encoder.model.config.max_position_embeddings
            self.width_embeddings = torch.nn.Embedding(span_width, linear_size, padding_idx=0)
        else:
            self.token_span_linear = torch.nn.Linear(self.token_encoder.embedding_length, linear_size)
            self.width_embeddings = None

        self.pooling_operation = pooling_operation
        self._label_type = label_type
        self.custom_label_verbalizations = custom_label_verbalizations
        self.train_only_with_positive_labels = train_only_with_positive_labels
        self.negative_sampling_factor = negative_sampling_factor
        self.negative_sampling_strategy = negative_sampling_strategy
        self.current_label_embeddings = []
        if not self.train_only_with_positive_labels:
            self.negative_sampling_factor = False
        if isinstance(self.negative_sampling_factor, bool):
            if self.negative_sampling_factor:
                self.negative_sampling_factor = 1
        if negative_sampling_factor >=1:
            self.train_only_with_positive_labels = True
            assert self.negative_sampling_strategy in ["hard_negatives", "random"], \
                ("You need to choose between 'random' and 'hard_negatives' as a negative_sampling_strategy!")

        self.margin_loss = margin_loss
        self.loss_function = torch.nn.CosineEmbeddingLoss(margin=self.margin_loss,
                                                          #reduction="sum"
                                                          )
        self.threshold_in_prediction = threshold_in_prediction

        cases: Dict[str, Callable[[Span, List[str]], torch.Tensor]] = {
            "average": self.emb_mean,
            "first": self.emb_first,
            "last": self.emb_last,
            "first_last": self.emb_firstAndLast,
        }

        if pooling_operation not in cases:
            raise KeyError('pooling_operation has to be one of "average", "first", "last" or "first_last"')

        self.aggregated_embedding = cases[pooling_operation]

        self.to(flair.device)

        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    def emb_first(self, span: Span, embedding_names):
        return span.tokens[0].get_embedding(embedding_names)

    def emb_last(self, span: Span, embedding_names):
        return span.tokens[-1].get_embedding(embedding_names)

    def emb_firstAndLast(self, span: Span, embedding_names):
        return torch.cat(
            (span.tokens[0].get_embedding(embedding_names), span.tokens[-1].get_embedding(embedding_names)), 0
        )

    def emb_mean(self, span, embedding_names):
        return torch.mean(torch.stack([token.get_embedding(embedding_names) for token in span], 0), 0)

    def _get_data_points_from_sentence(self, sentence: Sentence) -> List[Span]:
        return sentence.get_spans(self.label_type)

    def _filter_data_point(self, data_point: Sentence) -> bool:
        return bool(data_point.get_labels(self.label_type))

    def _get_embedding_for_data_point(self, prediction_data_point: Span) -> torch.Tensor:
        return self.aggregated_embedding(prediction_data_point, self.token_encoder.get_names())

    def _embed_labels_batchwise(self, batch_size=128, cpu = False):
        if cpu:
            label_embeddings = torch.tensor([], device="cpu")
        else:
            label_embeddings = torch.tensor([], device=flair.device)

        indices = [idx for idx in range(len(self.label_dictionary))]
        labels = [byte_label.decode("utf-8") for byte_label, idx in self.label_dictionary.item2idx.items()]

        if self.custom_label_verbalizations:
            labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(labels)
            labels_verbalized = [Sentence(label) for label in labels_verbalized]
        else:
            labels_verbalized = [Sentence(label) for label in labels]

        for i in range(0, len(indices), batch_size):

            labels_batch = labels_verbalized[i:i + batch_size]

            self.label_encoder.embed(labels_batch)
            label_embeddings_batch = torch.stack([label.get_embedding() for label in labels_batch])
            if cpu:
                label_embeddings_batch = label_embeddings_batch.detach().cpu()
            label_embeddings = torch.cat((label_embeddings, label_embeddings_batch), dim = 0)

            for s in labels_batch:
                s.clear_embeddings()

        return label_embeddings


    def _encode_data_points(self, sentences): #-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        self.token_encoder.embed(sentences)

        datapoints = []
        for s in sentences:
            datapoints.extend(self._get_data_points_from_sentence(s))

        if len(datapoints) == 0:
            return [], [], [], []

        span_hidden_states = torch.stack([self.aggregated_embedding(d, self.token_encoder.get_names()) for d in datapoints])

        if self.training:
            if self.train_only_with_positive_labels:
                labels = set()
                for d in datapoints:
                    labels.add(d.get_label(self.label_type).value)
                labels = list(labels)

                labels_ids = [self.label_dictionary.get_idx_for_item(l) for l in labels]

                if self.custom_label_verbalizations:
                    labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(labels)
                    labels_verbalized = [Sentence(label) for label in labels_verbalized]
                else:
                    labels_verbalized = [Sentence(label) for label in labels]

                self.label_encoder.embed(labels_verbalized)
                label_hidden_states_batch = torch.stack([label.get_embedding() for label in labels_verbalized])

                label_hidden_states = torch.zeros(len(self.label_dictionary), self.label_encoder.embedding_length,
                                                  device=flair.device)
                label_hidden_states[torch.LongTensor(list(labels_ids)), :] = label_hidden_states_batch

            else:
                label_hidden_states = self._embed_labels_batchwise()

        else:
            # if it's the first batch right after training, embed all and save!
            if self.is_first_batch_in_evaluation:
                #self.label_encoder.embed(labels_verbalized)
                #label_hidden_states = torch.stack([label.get_embedding() for label in labels_verbalized])
                label_hidden_states = self._embed_labels_batchwise(cpu = True)
                self.current_label_embeddings = label_hidden_states
                self.is_first_batch_in_evaluation = False

            else:
                label_hidden_states = self.current_label_embeddings

        return span_hidden_states, label_hidden_states, datapoints, sentences


    def _get_random_negative_labels(self, labels, factor):
        import random
        random_idx = random.sample(range(len(self.label_dictionary)), len(labels)*factor)
        random_labels = [self.label_dictionary.get_items()[idx] for idx in random_idx]
        if self.custom_label_verbalizations:
            random_labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(random_labels)
            random_labels_verbalized = [Sentence(label) for label in random_labels_verbalized]

        else:
            random_labels_verbalized = [Sentence(label) for label in random_labels]
        self.label_encoder.embed(random_labels_verbalized)
        random_label_hidden_states = torch.stack([label.get_embedding() for label in random_labels_verbalized])

        return random_label_hidden_states

    def _get_hard_negative_labels(self, labels, factor):

        # first: just for getting similar labels, no gradient tracking / gpu necessary:
        all_labels_embeddings = self._embed_labels_batchwise(cpu=True)
        batch_labels_indices = [self.label_dictionary.get_idx_for_item(l) for l in labels]
        batch_labels_embeddings = all_labels_embeddings[batch_labels_indices]

        similarity_matrix = cosine_similarity(batch_labels_embeddings.detach().cpu(),
                                              all_labels_embeddings.detach().cpu())

        # to exclude the real label
        for i, label_idx in enumerate(batch_labels_indices):
            similarity_matrix[i, label_idx] = float("-inf")

        _, hard_negative_indices = torch.topk(torch.tensor(similarity_matrix), factor, dim=1)

        # important: We need to "enroll" the labels the right way: each span should get it's hard negative, spans are concatenated each factor!

        rt_hard_negatives_embeddings = []
        rt_hard_negatives_verbalized = [] # not used, but for debugging
        for f in range(factor):
            hard_negative_indices_f = hard_negative_indices[:,f]

            # now embed those 'for real', so that gradient tracking possible and most current version
            if self.custom_label_verbalizations:
                hard_negatives_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(
                                                [self.label_dictionary.get_items()[idx] for idx in hard_negative_indices_f]
                                                )
            else:
                hard_negatives_verbalized = [self.label_dictionary.get_items()[idx] for idx in hard_negative_indices_f]

            rt_hard_negatives_verbalized.extend(hard_negatives_verbalized)
            hard_negatives = [ flair.data.Sentence(n) for n in hard_negatives_verbalized ]
            self.label_encoder.embed(hard_negatives)

            hard_negatives_embeddings = [ n.get_embedding() for n in hard_negatives ]
            rt_hard_negatives_embeddings.extend(hard_negatives_embeddings)

        rt_hard_negatives_embeddings = torch.stack(rt_hard_negatives_embeddings, dim=0).to(flair.device)

        return rt_hard_negatives_embeddings


    def _calculate_loss(self, span_hidden_states, label_hidden_states, datapoints, sentences, label_name):
        gold_label_name = label_name
        # todo: This is not quite right. We're comparing the label embeddings of the gold labels to the span embeddings.
        # todo: Shouldn't we compare the labels? Or the embeddings of the predicted labels against the embeddings of the real label?

        gold_labels = [d.get_label(gold_label_name).value for d in datapoints]
        gold_labels_idx = [self.label_dictionary.get_idx_for_item(l) for l in gold_labels]

        if len(gold_labels) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True if self.training else False), 0

        if self.training and self.train_only_with_positive_labels:
            gold_labels_hidden_states = [label_hidden_states[i] for i in gold_labels_idx]
            gold_labels_hidden_states = torch.stack(gold_labels_hidden_states)
            y = torch.ones(len(gold_labels_idx), device=flair.device)

            if self.negative_sampling_factor:
                nr_datapoints = len(gold_labels_idx)

                if self.negative_sampling_strategy == "random":
                    some_random_label_hidden_states = self._get_random_negative_labels(gold_labels,
                                                                                       self.negative_sampling_factor)
                    gold_labels_hidden_states_with_negatives = torch.cat((gold_labels_hidden_states,
                                                                          some_random_label_hidden_states), dim = 0)
                if self.negative_sampling_strategy == "hard_negatives":
                    hard_negatives_label_hidden_states = self._get_hard_negative_labels(gold_labels,
                                                                                        self.negative_sampling_factor)
                    gold_labels_hidden_states_with_negatives = torch.cat((gold_labels_hidden_states,
                                                                          hard_negatives_label_hidden_states), dim = 0)

                gold_labels_hidden_states = gold_labels_hidden_states_with_negatives

                span_hidden_states_concat = span_hidden_states
                for i in range(self.negative_sampling_factor):
                    span_hidden_states_concat = torch.cat((span_hidden_states_concat, span_hidden_states), dim = 0)
                span_hidden_states = span_hidden_states_concat

                y = torch.cat((y, -torch.ones(nr_datapoints*self.negative_sampling_factor, device=flair.device)), dim = 0)


            loss = self.loss_function(span_hidden_states, gold_labels_hidden_states, y).unsqueeze(0)


        else:

            losses = []
            for i, sp in enumerate(datapoints):
                y = -torch.ones(label_hidden_states.shape[0], device=flair.device if self.training else "cpu")
                y[gold_labels_idx[i]] = 1.0
                # stack the span embedding, so that we can push all OTHER labels away from it
                span_repeated = span_hidden_states[i].repeat(label_hidden_states.shape[0],1)
                sp_loss = self.loss_function(span_repeated, label_hidden_states, y)
                losses.append(sp_loss)

            loss = torch.mean(torch.stack(losses))

        return loss, len(datapoints)

    def forward_loss(self, sentences) -> Tuple[torch.Tensor, int]:

        if self.training:
            self.is_first_batch_in_evaluation = True # set this to assure new embedding of labels in the following evaluation phase

        if not [spans for sentence in sentences for spans in sentence.get_spans(self.label_type)]:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        span_hidden_states, label_hidden_states, datapoints, sentences = self._encode_data_points(sentences)
        loss, nr_datapoints = self._calculate_loss(span_hidden_states, label_hidden_states, datapoints, sentences, label_name=self.label_type)

        return loss, nr_datapoints


    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        with torch.no_grad():
            span_hidden_states, label_hidden_states, datapoints, sentences = self._encode_data_points(sentences)
            if len(datapoints) != 0:

                # Compute cosine similarity and use threshold for prediction
                # Normalize the embeddings along the dimension D (this is crucial for cosine similarity)
                span_hidden_states = F.normalize(span_hidden_states, p=2, dim=1).detach().cpu()
                label_hidden_states = F.normalize(label_hidden_states, p=2, dim=1).detach().cpu()
                # The result will be of size (BS, N) where each row contains the similarities between a span and all labels
                similarity = torch.mm(span_hidden_states, label_hidden_states.t())

                # version 2:
                # Compute cosine similarity using torch.nn.CosineSimilarity
                #cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                #similarity = torch.zeros((len(datapoints), label_hidden_states.shape[0]))
                #for i in range(len(datapoints)):
                #    for j in range(label_hidden_states.shape[0]):
                #        similarity[i, j] = cosine_sim(span_hidden_states[i].unsqueeze(0), label_hidden_states[j].unsqueeze(0))

                # Check which similarities are above the threshold
                above_threshold = similarity > self.threshold_in_prediction

                # Get the label indices with maximum similarity for each span
                _, max_label_indices = torch.max(similarity, dim=1)

                # If the maximum similarity for a span is below the threshold, we'll set its label index to -1
                final_label_indices = torch.where(above_threshold[torch.arange(len(datapoints)), max_label_indices],
                                                  max_label_indices,
                                                  torch.tensor(-1).cpu())
                #print(final_label_indices)

                for i,d in enumerate(datapoints):
                    label_idx = final_label_indices[i]
                    if label_idx != -1:
                        label = self.label_dictionary.get_items()[label_idx]
                        conf = similarity[i, label_idx]
                        d.add_label(label_name,
                                    value=label,
                                    score=conf
                                    )

        if return_loss:
            if len(datapoints) == 0:
                return torch.tensor(0.0, dtype=torch.float, device=flair.device,
                                    requires_grad=False), 0
            return self._calculate_loss(
                span_hidden_states, label_hidden_states, datapoints, sentences, label_name = self.label_type # todo or use "predicted" (label_name)?
            )

        return None


    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            eval_line = f"\n{datapoint.to_original_text()}\n"

            for span in datapoint.get_spans(gold_label_type):
                symbol = "✓" if span.get_label(gold_label_type).value == span.get_label("predicted").value else "❌"
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {span.get_label("predicted").value} ({symbol})\n'
                )

            lines.append(eval_line)
        return lines
