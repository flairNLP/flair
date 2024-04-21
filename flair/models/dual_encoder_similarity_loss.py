import logging
import pwd
import random
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
from flair.models.entity_linker_model import CandidateGenerator

from pathlib import Path

from flair.nn.wikidata_utils import get_wikidata_categories, get_sitelinks_of_entity, get_pageviews_of_entity_precomputed, get_wikipedia_first_paragraph

log = logging.getLogger("flair")

class WikidataLabelVerbalizer:
    def __init__(self,
                 label_dictionary,
                 verbalizations_paths: List,
                 use_wikidata_description: bool = False,
                 use_wikidata_classes: bool = True,
                 use_wikipedia_first_paragraph: bool = False,
                 max_verbalization_length: int = 16,
                 add_occupation: bool = True,
                 add_field_of_work: bool =False,
                 delimiter: str = ";",
                 add_property_prefixes = True
                 ):

        self.label_dictionary = label_dictionary
        self.verbalizations_paths = verbalizations_paths
        self.use_wikidata_description = use_wikidata_description
        self.use_wikidata_classes = use_wikidata_classes
        self.use_wikipedia_first_paragraph = use_wikipedia_first_paragraph
        self.add_occupation = add_occupation
        self.add_field_of_work = add_field_of_work
        self.delimiter = delimiter
        self.add_property_prefixes = add_property_prefixes

        self.verbalized_labels = self.verbalize_all_labels(label_dictionary=self.label_dictionary, save_verbalizations=True)
        self.label_dict_list = self.label_dictionary.get_items()

        self.max_verbalization_length = max_verbalization_length

        if self.max_verbalization_length:
            verbalized_labels_truncated = []
            for l in self.verbalized_labels:
                s = flair.data.Sentence(l)
                if len(s.tokens) <= self.max_verbalization_length:
                    verbalized_labels_truncated.append(s.text)
                else:
                    for i in range(self.max_verbalization_length, len(s.tokens)-1):
                        if self.use_wikipedia_first_paragraph:
                            if s.tokens[i].text == ".":
                                s.tokens = s.tokens[:i]
                                break
                        else:
                            if f"{self.delimiter}" in s.tokens[i].text:
                                s.tokens = s.tokens[:i]
                                break
                    verbalized_labels_truncated.append(s.text)
            self.verbalized_labels = verbalized_labels_truncated

    def verbalize_entity(self, entity):
        wikipedia_label = entity
        if self.use_wikipedia_first_paragraph:
            title, wikipedia_first_paragraph = get_wikipedia_first_paragraph(entity)
        else:
            if self.use_wikidata_description and not self.use_wikidata_classes:
                wikidata_info = get_wikidata_categories(wikipedia_label,
                                                        method="with_property_labels" if self.add_property_prefixes else "only_one_level_up",
                                                        add_occupation=self.add_occupation,
                                                        add_field_of_work=self.add_field_of_work,
                                                        add_country=False,
                                                        only_description=True)
            else:
                wikidata_info = get_wikidata_categories(wikipedia_label,
                                                        method="with_property_labels" if self.add_property_prefixes else "only_one_level_up",
                                                        add_occupation=self.add_occupation,
                                                        add_field_of_work=self.add_field_of_work,
                                                        add_country = False)
            wikidata_classes = wikidata_info["class_names"]
            title = wikidata_info["wikidata_title"]
            wikidata_description = wikidata_info["wikibase_description"]

        verbalized = title
        if self.use_wikipedia_first_paragraph:
            verbalized += f"{self.delimiter} " + wikipedia_first_paragraph

        if self.use_wikidata_description:
            if len(wikidata_description) >0:
                verbalized += f"{self.delimiter} " + wikidata_description
        if self.use_wikidata_classes:
            if len(wikidata_classes) >0:
                if self.add_property_prefixes and isinstance(wikidata_classes[0], tuple): # has the property labels
                    #verbalized += f"{self.delimiter} " + f"{self.delimiter} ".join(["".join(t) for t in wikidata_classes])
                    last_p = "" # make sure the prefix is printed only once
                    for p,c in wikidata_classes:
                        if p != last_p:
                            verbalized += f"{self.delimiter} {p}{c}"
                        else:
                            verbalized += f"{f'{self.delimiter}' if p == '' else ' and'} {c}"
                        last_p = p
                else:
                    verbalized += f"{self.delimiter} " + f"{self.delimiter} ".join(wikidata_classes)
        return verbalized

    #@staticmethod
    def verbalize_all_labels(self, label_dictionary: Dictionary, save_verbalizations = True) -> List[Sentence]:

        if len(self.verbalizations_paths) > 0:
            print(f"Found {len(self.verbalizations_paths)} source paths of verbalizations:")
            print(self.verbalizations_paths)
            self.verbalizations = {}
            for p in self.verbalizations_paths:
                if os.path.exists(p):
                    with open(p, 'r') as f:
                        verbalizations_to_add = json.load(f)
                        self.verbalizations ={**self.verbalizations, **verbalizations_to_add}
                        print(f"Found verbalizations for {len(verbalizations_to_add)} labels at  {p}")

        else:
            self.verbalizations_paths = ["/tmp/verbalizations.json"]
            with open("/tmp/verbalizations.json", 'w') as file:
                json.dump({}, file)
            self.verbalizations = {}
            print(f"Created empty verbalization dictionary at {self.verbalizations_paths[0]}")

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
                verbalized = self.verbalizations.get(entity, None)
                if not verbalized:
                    verbalized = self.verbalize_entity(entity)
                    #self.verbalizations[entity] = verbalized
                    counter_newly_verbalized +=1
                    print(f"Not found {idx} {entity} --> {verbalized}")

            verbalized_labels.append(verbalized)
            self.verbalizations[entity] = verbalized
            #print("Verbalized id", idx, ":", str_label, "->", verbalized)

            # save every n entities and final
            if save_verbalizations:
                if counter_newly_verbalized > 0 and (idx % 500 == 0 or idx >= len(label_dictionary.item2idx.items()) -1):
                    verbalizations_dict = self.verbalizations
                    with open(f"{self.verbalizations_paths[0]}", "w") as file:
                        json.dump(verbalizations_dict, file, indent= 0, sort_keys= True)
                        file.flush()
                    print("saved new verbalizations at:", self.verbalizations_paths[0])

        print(f"--- Created verbalized labels for {len(verbalized_labels)} labels")
        print(f"--- Thereof newly verbalized:", counter_newly_verbalized)

        return verbalized_labels

    def verbalize_list_of_labels(self, list_of_labels,
                                       save_verbalizations = True):
        save = False
        if set(list_of_labels).issubset(self.label_dict_list):
            label_idx = [self.label_dictionary.get_idx_for_item(idx) for idx in list_of_labels]
            rt =  [self.verbalized_labels[idx] for idx in label_idx ]

        else:
            rt = []
            for l in list_of_labels:
                if l in self.verbalizations:
                    rt.append(self.verbalizations[l])
                else:
                    save = True
                    verbalization = self.verbalize_entity(l)
                    self.verbalizations[l] = verbalization
                    rt.append(verbalization)
                    print(f"Not found {l} --> {verbalization}")

            if save_verbalizations and save:
                verbalizations_dict = self.verbalizations
                with open(f"{self.verbalizations_paths[0]}", "w") as file:
                    json.dump(verbalizations_dict, file, indent=0, sort_keys=True)
                    file.flush()
                print("saved new verbalizations at:", self.verbalizations_paths[0])

        return rt

    def add_new_verbalization_path(self, json_path):
        self.verbalizations_paths.append(json_path)


class EuclideanEmbeddingLoss(torch.nn.Module):
    def __init__(self, margin=0.0,
                 reduction = "mean"):
        super(EuclideanEmbeddingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1, input2, target):
        euclidean_distance = torch.sqrt(torch.sum(torch.square(input1 - input2), dim=-1))
        loss_values = torch.where(target == 1, euclidean_distance,
                                  torch.maximum(torch.tensor(0.0), self.margin - euclidean_distance)
                                  )
        if self.reduction == "mean":
            return torch.mean(loss_values)
        elif self.reduction == "sum":
            return torch.sum(loss_values)
        else:
            return loss_values


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
        pooling_operation: str = "average", #todo "average",
        label_type: str = "nel",
        custom_label_verbalizations = None,
        max_verbalization_length = None,
        train_only_with_positive_labels = True,
        negative_sampling_factor: [int, bool] = 1,
        negative_sampling_strategy: str = "random",
        margin_loss: float = 0.5,
        weighted_loss = False,
        add_popularity = False,
        popularity_save_path: Path = None,
        label_embeddings_save_path: Path = None,
        distance_function: str = "euclidean", # "cosine", "mm"
        losses: list = ["triplet_loss"], # "binary_embedding_loss", "BCE_loss"
        candidates: Optional[CandidateGenerator] = None,
        predict_greedy: bool = False,
        predict_again_finally: bool = False,
        train_greedy: [bool, str] = False,

    ):
        super().__init__()

        self.token_encoder = token_encoder
        self.label_encoder = label_encoder
        self.label_dictionary = label_dictionary
        self.label_dict_list = self.label_dictionary.get_items()
        self._label_type = label_type

        self.pooling_operation = pooling_operation
        self._label_type = label_type
        self.custom_label_verbalizations = custom_label_verbalizations
        self.max_verbalization_length = self.custom_label_verbalizations.max_verbalization_length if self.custom_label_verbalizations else max_verbalization_length
        self.add_property_prefixes = self.custom_label_verbalizations.add_property_prefixes if self.custom_label_verbalizations else None
        self.train_only_with_positive_labels = train_only_with_positive_labels
        self.negative_sampling_factor = negative_sampling_factor
        self.negative_sampling_strategy = negative_sampling_strategy
        self.is_first_batch_in_evaluation = True
        self.is_first_batch_in_training = True
        if self.negative_sampling_strategy == "batch_negatives":
            self.negative_sampling_factor = False
        if not self.train_only_with_positive_labels:
            self.negative_sampling_factor = False
        if isinstance(self.negative_sampling_factor, bool):
            if self.negative_sampling_factor:
                self.negative_sampling_factor = 1
        if negative_sampling_factor >=1:
            self.train_only_with_positive_labels = True
            assert self.negative_sampling_strategy in ["hard_negatives", "random", "batch_negatives"], \
                ("You need to choose between 'random', 'hard_negatives' and 'batch_negatives' as a negative_sampling_strategy!")

        self.weighted_loss = weighted_loss
        self.add_popularity = add_popularity
        self.popularity_save_path = popularity_save_path
        if self.add_popularity:
            if not self.popularity_save_path:
                #self.popularity_save_path = "/tmp/wikipedia_sitelinks_dict.json"
                #self.popularity_save_path = "/vol/tmp/ruckersu/dual_encoder_entity_linking/sitelinks_dict.json"
                self.popularity_save_path = "/vol/tmp/ruckersu/dual_encoder_entity_linking/pageviews_dict.json"

            self._create_popularity_dict(self.label_dict_list)

        self.distance_function = distance_function
        assert self.distance_function in ["cosine", "euclidean", "mm"], \
            ("You need to choose between 'cosine', 'euclidean' and 'mm' for distance_function!")

        if self.distance_function == "cosine":
            def cosine_distance(x1, x2):
                x1_normalized = F.normalize(x1, p=2, dim=1)
                x2_normalized = F.normalize(x2, p=2, dim=1)
                similarity = torch.sum(x1_normalized * x2_normalized, dim=1)
                return 1 - similarity  # Convert similarity to dissimilarity

        self.losses = losses
        for l in self.losses:
            assert l in ["triplet_loss", "binary_embedding_loss", "BCE_loss"], \
                ("You need to choose between 'triplet_loss', 'binary_embedding_loss' and 'BCE_loss' for loss!")

        self.margin_loss = margin_loss
        if self.losses[0] == "BCE_loss":
            #self.loss_function = torch.nn.BCEWithLogitsLoss()
            self.loss_function = torch.nn.BCELoss()

        elif self.losses[0] == "triplet_loss":

            if self.distance_function == "euclidean":
                self.loss_function = torch.nn.TripletMarginLoss(margin=self.margin_loss, p=2, eps=1e-7,
                                                                reduction="mean",
                                                                #swap=True
                                                                )

            elif self.distance_function == "cosine":
                def cosine_distance(x1, x2):
                    x1_normalized = F.normalize(x1, p=2, dim=1)
                    x2_normalized = F.normalize(x2, p=2, dim=1)
                    similarity = torch.sum(x1_normalized * x2_normalized, dim=1)
                    return 1 - similarity  # Convert similarity to dissimilarity

                self.loss_function = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance,
                                                                            margin=self.margin_loss,
                                                                            reduction="mean",
                                                                            #swap=True
                                                                            )
            elif self.distance_function == "mm":
                def mm_distance(x1, x2):
                    return -torch.mm(x1, x2.t())

                self.loss_function = torch.nn.TripletMarginWithDistanceLoss(distance_function=mm_distance,
                                                                            margin=self.margin_loss,
                                                                            reduction="mean")

            else:
                raise NotImplementedError


        elif self.losses[0] == "binary_embedding_loss":

            if self.distance_function == "euclidean":
                self.loss_function = EuclideanEmbeddingLoss(margin=self.margin_loss,
                                                            reduction="mean")
            elif self.distance_function == "cosine":
                self.loss_function = torch.nn.CosineEmbeddingLoss(margin=self.margin_loss,
                                                                  reduction="mean",
                                                                  )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.predict_greedy = predict_greedy
        self.predict_again_finally = predict_again_finally
        self.train_greedy = train_greedy
        if self.train_greedy:
            self.predict_greedy = True
        self.label_embeddings_save_path = label_embeddings_save_path
        if self.label_embeddings_save_path:
            self.label_embeddings_save_path.mkdir(parents=True, exist_ok=True)

        #self.current_label_embeddings, _ = self._embed_labels_batchwise(cpu=True, max_limit=None if len(self.label_dict_list) < 6000 else 1000) # before training, necessary negative sampling

        self.candidates = candidates


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

    def _create_popularity_dict(self, labels):

        with open(self.popularity_save_path, "r") as handle:
            self.popularity_dict = json.load(handle)

        print(f"Found popularity_dict with {len(self.popularity_dict)} entries")
        use = "pageviews"  # "sitelinks"
        if not all(item in self.popularity_dict for item in labels):
            for i, e in enumerate(self.label_dict_list):
                if e not in self.popularity_dict:
                    if use == "sitelinks":
                        sitelinks = get_sitelinks_of_entity(e)
                        self.popularity_dict[e] = {"popularity": sitelinks}
                    if use == "pageviews":
                        pageviews = get_pageviews_of_entity_precomputed(e,
                                        "/vol/tmp/ruckersu/data/wikipedia_pageviews/en_wikipedia_ranking.txt"
                                        )
                        self.popularity_dict[e] = {"popularity": pageviews}

                    print(i, e, self.popularity_dict[e]["popularity"])

                if i % 100 == 0:
                    print("saving...")
                    with open(self.popularity_save_path, "w") as f:
                        json.dump(self.popularity_dict, f, indent= 0, sort_keys= True)

            with open(self.popularity_save_path, "w") as f:
                json.dump(self.popularity_dict, f, indent= 0, sort_keys= True)

        # get highest popularity value for normalizing
        # max_popularity = 0
        # for entity, info in self.popularity_dict.items():
        #     if "popularity" in info:
        #         popularity_count = int(info["popularity"])
        #         if popularity_count > max_popularity:
        #             max_popularity = popularity_count

        # or just use N as max, so everything above N gets 1.0
        if use == "sitelinks":
            max_popularity = 100
        if use == "pageviews":
            max_popularity = 10000000

        # now normalize:
        for entity, info in self.popularity_dict.items():
            if info["popularity"]:
                self.popularity_dict[entity]["popularity_normalized"] = min(
                    round(int(info["popularity"]) / max_popularity, 6), 1.0)
            else:
                self.popularity_dict[entity][
                    "popularity_normalized"] = 0.0  # Dummy, if popularity was None (e.g. article was deleted)

        if self.add_popularity == "concatenate":
            # add a dense layer on token embeddings, to match dimension (+1)
            new_dimension = self.token_encoder.embedding_length + 1
            self.token_dense = torch.nn.Linear(in_features=self.token_encoder.embedding_length,
                                               out_features=new_dimension)
            self.token_dense.to(flair.device)

            with torch.no_grad():
                # Initialize the weights to approximate an identity mapping:
                # self.token_dense.weight.copy_(torch.eye(self.token_encoder.embedding_length))
                torch.nn.init.eye_(self.token_dense.weight)
                self.token_dense.bias.zero_()

        if self.add_popularity == "verbalize":
            self.popularity_mapping = {(0, 0.25): "low popularity",
                                       (0.25, 0.5): "medium popularity",
                                       (0.5, 0.75): "high popularity",
                                       (0.75, 1.0): "very high popularity"
                                       }

            for entity, info in self.popularity_dict.items():
                for range_, label in self.popularity_mapping.items():
                    if range_[0] <= self.popularity_dict[entity]["popularity_normalized"] <= range_[1]:
                        verbalization = label
                self.popularity_dict[entity]["popularity_verbalized"] = verbalization

            print("Now adding popularity info to verbalizations...")
            if self.custom_label_verbalizations:
                self.custom_label_verbalizations.verbalized_labels = [self.custom_label_verbalizations.verbalized_labels[i] + f"{self.custom_label_verbalizations.delimiter} {self.popularity_dict[l]['popularity_verbalized']}" for i,l in enumerate(self.label_dict_list)]
            else:
                self.label_dict_list = [self.label_dict_list[i] + f"; {self.popularity_dict[l]['popularity_verbalized']}" for i,l in enumerate(self.label_dict_list)]

        print("done with preprocessing popularity info")

    def _embed_batchwise_and_return_hidden_states(self, encoder,
                                                  data_to_embed, # labels or sentences
                                                  data_to_get_embeddings, # spans
                                                  max_step_size : [None, int] = 32, clear_embeddings = True, return_stacked = True):

        #if not self.training:
        #    max_step_size = None

        if len(data_to_embed) != len(data_to_get_embeddings):
            max_step_size = None

        if max_step_size:
            step_size = len(data_to_embed) if len(data_to_embed) <= max_step_size else max_step_size

        if max_step_size == None:
            encoder.embed(data_to_embed)
            if isinstance(data_to_get_embeddings[0], flair.data.Sentence):
                final_embeddings = [d.get_embedding() for d in data_to_get_embeddings]
            elif isinstance(data_to_get_embeddings[0], flair.data.Span):
                final_embeddings = [self.aggregated_embedding(d, encoder.get_names()) for d in data_to_get_embeddings]
            if clear_embeddings:
                [d.clear_embeddings() for d in data_to_embed]

        else:
            final_embeddings = []
            for i in range(0, len(data_to_embed), step_size):
                encoder.embed(data_to_embed[i:i + step_size])
                if isinstance(data_to_get_embeddings[0], flair.data.Sentence):
                    final_embeddings.extend([d.get_embedding() for d in data_to_get_embeddings[i:i + step_size]])
                elif isinstance(data_to_get_embeddings[0], flair.data.Span):
                    final_embeddings.extend([self.aggregated_embedding(d, encoder.get_names()) for d in data_to_get_embeddings[i:i + step_size]])
                else:
                    raise NotImplementedError
                if clear_embeddings:
                    [d.clear_embeddings() for d in data_to_embed]

        if return_stacked:
            final_embeddings = torch.stack(final_embeddings)

        return final_embeddings

    def _embed_labels_batchwise(self, batch_size=128, cpu = False, max_limit = None):
        print(f"Now creating label embeddings with limit {max_limit}...")
        if cpu:
            label_embeddings = torch.tensor([], device="cpu")
        else:
            label_embeddings = torch.tensor([], device=flair.device)

        import random
        used_indices = sorted(random.sample(range(len(self.label_dictionary)), len(self.label_dictionary) if not max_limit else max_limit))
        used_labels = np.array(self.label_dict_list)[used_indices]

        if self.custom_label_verbalizations:
            used_labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(used_labels, save_verbalizations=False)
            used_labels_verbalized = [Sentence(label) for label in used_labels_verbalized]

        else:
            used_labels_verbalized = [Sentence(label) for label in used_labels]

        for i in range(0, len(used_indices), batch_size):
            #print(i)
            #print(torch.cuda.memory_summary(device=flair.device, abbreviated=False))
            labels_batch = used_labels_verbalized[i:i + batch_size]
            if cpu:
                self.label_encoder.embed(labels_batch)
                label_embeddings_batch = torch.stack([label.get_embedding() for label in labels_batch]).detach().cpu()

            else:
                self.label_encoder.embed(labels_batch)
                label_embeddings_batch = torch.stack([label.get_embedding() for label in labels_batch])

            label_embeddings = torch.cat((label_embeddings, label_embeddings_batch), dim = 0)
            del label_embeddings_batch

            for s in labels_batch:
                s.clear_embeddings()

        if self.add_popularity == "concatenate":
            popularities = torch.Tensor([self.popularity_dict[e]["popularity_normalized"] for e in used_labels])
            label_embeddings = torch.cat((label_embeddings, popularities.unsqueeze(1)), dim = 1)

        del used_labels
        print("Done with creating label embeddings.")
        return label_embeddings, used_indices


    def _encode_data_points(self, sentences): #-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        datapoints = []
        for s in sentences:
            datapoints.extend(self._get_data_points_from_sentence(s))

        #print(len(datapoints))
        if len(datapoints) == 0:
            return [], [], [], []

        if self.training and len(datapoints) >= len(sentences) * 20:
            print("Too many datapoints, so skipping:", len(datapoints))
            return [], [], [], []

        if self.training and self.train_greedy == "use_gold" and len(datapoints) <= 20:
            # add a random part of gold labels (verbalizations) to the sentences, similar to in prediction, then embed and use those
            datapoints_modified, new_sentences = self._add_already_predicted_label_verbalizations_to_sentences(sentences = sentences,
                                                                                                               datapoints=datapoints,
                                                                                                               use_gold=True,
                                                                                                               leave_out_datapoints_to_be_predicted=True)
            span_hidden_states = self._embed_batchwise_and_return_hidden_states(encoder = self.token_encoder,
                                                                                data_to_embed=new_sentences,
                                                                                data_to_get_embeddings=datapoints_modified,
                                                                                clear_embeddings=True,
                                                                                )


        else:
            span_hidden_states = self._embed_batchwise_and_return_hidden_states(encoder=self.token_encoder,
                                                                                data_to_embed=sentences,
                                                                                data_to_get_embeddings=datapoints,
                                                                                clear_embeddings=True,
                                                                                max_step_size = None
                                                                                )

        if self.add_popularity == "concatenate":
            # TODO is this good? set to non trainable?
            span_hidden_states = self.token_dense(span_hidden_states)

        if self.training:
            if self.train_only_with_positive_labels:

                labels = []
                for d in datapoints:
                    #labels.add(d.get_label(self.label_type).value)
                    labels.append(d.get_label(self.label_type).value)

                labels_ids = [self.label_dictionary.get_idx_for_item(l) for l in labels]

                if self.custom_label_verbalizations:
                    labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(labels)
                    labels_verbalized = [Sentence(label) for label in labels_verbalized]
                else:
                    labels_verbalized = [Sentence(label) for label in labels]

                label_hidden_states_batch = self._embed_batchwise_and_return_hidden_states(encoder=self.label_encoder,
                                                                                           data_to_embed=labels_verbalized,
                                                                                           data_to_get_embeddings=labels_verbalized,
                                                                                           clear_embeddings=True,
                                                                                           )

                if self.add_popularity == "concatenate":
                    popularities = torch.Tensor(
                        [self.popularity_dict[e]["popularity_normalized"] for e in labels]).to(flair.device)
                    label_hidden_states_batch = torch.cat((label_hidden_states_batch, popularities.unsqueeze(1)), dim=1)

                label_hidden_states = torch.zeros(len(self.label_dictionary), label_hidden_states_batch.shape[1],
                                                  device=flair.device)
                label_hidden_states[torch.LongTensor(list(labels_ids)), :] = label_hidden_states_batch

            else:
                label_hidden_states, _ = self._embed_labels_batchwise(max_limit=None)

        else:
            # if it's the first batch right after training, embed all and save!
            if self.is_first_batch_in_evaluation:
                label_hidden_states, _ = self._embed_labels_batchwise(cpu = True, max_limit=None)
                self.current_label_embeddings = label_hidden_states
                self.is_first_batch_in_evaluation = False
                self.is_first_batch_in_training = True # for next training epoch

                # save them to enable inspection (and for one of the negative sampling methods):
                save_label_embeddings = False
                if self.label_embeddings_save_path and save_label_embeddings:
                    print("\nSaving current embeddings here:")
                    print(self.label_embeddings_save_path)
                    label_tensor_path = self.label_embeddings_save_path / "label_embeddings.npy"
                    np.save(str(label_tensor_path), label_hidden_states.cpu().numpy())

                    # save as tsv as well, for enabling visualizations
                    label_tensor_tsv_path = self.label_embeddings_save_path / "label_embeddings.tsv"
                    np.savetxt(label_tensor_tsv_path, label_hidden_states.cpu().numpy(), delimiter='\t', fmt='%.8f')

                    label_names = [byte_label.decode("utf-8") for byte_label, idx in
                                   self.label_dictionary.item2idx.items()]
                    with open(self.label_embeddings_save_path / "label_embeddings_names.txt", 'w') as file:
                        for item in label_names:
                            file.write(item + '\n')

            else:
                label_hidden_states = self.current_label_embeddings

        return span_hidden_states, label_hidden_states, datapoints, sentences

    def _get_similarity_matrix(self, matrix1, matrix2,
                               activation : [list] = [None],#["softmax"],
                               normalize = False, #False, #True
                               ):

        if self.losses[0] == "BCE_loss":
            # for the BCE loss to work, we need to make sure there are no negative similarities
            #normalize = True
            #activation = ["min_max_normalization"]
            #activation = ["sigmoid"] # TODO what to chose here?
            activation = ["softmax"]


        if normalize:
            matrix1 = torch.nn.functional.normalize(matrix1)
            matrix2 = torch.nn.functional.normalize(matrix2)

        if self.distance_function == "mm":
            similarity = torch.mm(matrix1,
                                  matrix2.t())

        if self.distance_function == "cosine":
            #similarity = torch.Tensor(cosine_similarity(matrix1,matrix2)) # same but from numpy, so problems with needed detach().cpu()
            matrix1 = F.normalize(matrix1, p=2, dim=1)
            matrix2 = F.normalize(matrix2, p=2, dim=1)
            similarity = torch.mm(matrix1, matrix2.T)

        if self.distance_function == "euclidean":
            similarity = -torch.cdist(matrix1, matrix2)

        if "softmax" in activation:
            soft = torch.nn.Softmax(dim =1)
            similarity = soft(similarity) #, dim=1)
        if "sigmoid" in activation:
            sig = torch.nn.Sigmoid()
            similarity = sig(similarity)
        if "min_max_normalization" in activation:
            def min_max_normalize_row(matrix):
                min_vals = torch.min(matrix, dim=1)[0]
                max_vals = torch.max(matrix, dim=1)[0]
                # Normalize each row independently
                normalized_matrix = (matrix - min_vals.unsqueeze(1)) / (max_vals - min_vals).unsqueeze(1)
                return normalized_matrix
            similarity = min_max_normalize_row(similarity)

        return similarity


    def _get_random_negative_embeddings(self, span_hidden_states, labels, factor, datapoints, epoch_wise = False):
        import random
        rt_random_negatives_embeddings = []
        rt_random_negatives_labels = []
        popularities = []
        for f in range(factor):
            random_idx = random.sample(range(len(self.label_dictionary)), len(labels))
            random_labels = [self.label_dict_list[idx] for idx in random_idx]
            if self.add_popularity == "concatenate":
                popularities.extend([self.popularity_dict[e]["popularity_normalized"] for e in random_labels])

            if epoch_wise:
                random_label_hidden_states = self.current_label_embeddings[random_idx]

            else:

                if self.custom_label_verbalizations:
                    random_labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(random_labels)
                    random_labels_verbalized = [Sentence(label) for label in random_labels_verbalized]
                else:
                    random_labels_verbalized = [Sentence(label) for label in random_labels]

                random_label_hidden_states = self._embed_batchwise_and_return_hidden_states(encoder=self.label_encoder,
                                                                                            data_to_embed=random_labels_verbalized,
                                                                                            data_to_get_embeddings=random_labels_verbalized,
                                                                                            clear_embeddings=True,
                                                                                            )

            rt_random_negatives_embeddings.extend(random_label_hidden_states)
            rt_random_negatives_labels.extend(random_labels)

        rt_random_negatives_embeddings = torch.stack(rt_random_negatives_embeddings, dim=0).to(flair.device)

        if self.add_popularity == "concatenate":
            popularities = torch.Tensor(popularities)
            rt_random_negatives_embeddings = torch.cat((rt_random_negatives_embeddings, popularities.unsqueeze(1)), dim=1)

        return rt_random_negatives_embeddings, rt_random_negatives_labels

    def _get_hard_negative_embeddings(self, span_hidden_states, labels, factor, datapoints,
                                      epoch_wise = True,
                                      ):

        with torch.no_grad():

            #TODO: Do I want to choose by similarity to the gold LABEL or by similarity to the SPAN embedding?
            # i.e. span_hidden_states or (see above) batch_labels_verbalized?

            # first: just for getting indices of similar labels: no gradient tracking / gpu necessary:
            if epoch_wise:
                if self.candidates:
                    # get all candidates for mentions in batch
                    all_candidates = set()
                    for d in datapoints:
                        m_candidates = self.candidates.get_candidates(d.text)
                        for c in m_candidates:
                            all_candidates.add(c)
                    all_candidates = [ c for c in all_candidates if c not in labels]

                    if len(all_candidates) >0:
                        if self.custom_label_verbalizations:
                            all_candidates_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(all_candidates)
                            all_candidates_verbalized = [Sentence(label) for label in all_candidates_verbalized]
                        else:
                            all_candidates_verbalized = [Sentence(l) for l in all_candidates]

                        all_labels_embeddings = self._embed_batchwise_and_return_hidden_states(encoder=self.label_encoder,
                                                                       data_to_embed=all_candidates_verbalized,
                                                                       data_to_get_embeddings=all_candidates_verbalized,
                                                                       clear_embeddings=True,
                                                                       )
                        all_labels_indices = [self.label_dictionary.get_idx_for_item(c) for c in all_candidates]
                        all_labels = all_candidates
                    else: # if no candidates, use normal hard embedding method
                        all_labels_embeddings, all_labels_indices = self.current_label_embeddings, self.current_label_embeddings_indices
                        all_labels = [self.label_dict_list[idx] for idx in all_labels_indices]

                else:
                    all_labels_embeddings, all_labels_indices = self.current_label_embeddings, self.current_label_embeddings_indices
                    all_labels = [self.label_dict_list[idx] for idx in all_labels_indices]

            else:
                raise NotImplementedError
                # TODO embed and compare all labels

            all_labels_embeddings = all_labels_embeddings.detach().cpu()
            batch_labels_indices = [self.label_dictionary.get_idx_for_item(l) for l in labels]

            # Choose method to rank hard negative labels
            negative_mining_method = "by_similarity_to_span" # "by_similarity_to_label"

            if negative_mining_method == "by_similarity_to_label":

                if self.custom_label_verbalizations:
                    labels_batch_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(labels)
                    labels_batch_verbalized = [Sentence(label) for label in labels_batch_verbalized]
                else:
                    labels_batch_verbalized = [Sentence(self.label_dict_list[label_i]) for label_i in batch_labels_indices]

                batch_labels_embeddings = self._embed_batchwise_and_return_hidden_states(encoder=self.label_encoder,
                                                                                         data_to_embed=labels_batch_verbalized,
                                                                                         data_to_get_embeddings=labels_batch_verbalized,
                                                                                         clear_embeddings=False,
                                                                                         )

                if self.add_popularity == "concatenate":
                    popularities = torch.Tensor([self.popularity_dict[e]["popularity_normalized"] for e in labels]).to(flair.device)
                    batch_labels_embeddings = torch.cat((batch_labels_embeddings, popularities.unsqueeze(1)), dim=1)

                batch_labels_embeddings = batch_labels_embeddings.detach().cpu()

                similarity_matrix = self._get_similarity_matrix(batch_labels_embeddings, all_labels_embeddings)

            if negative_mining_method == "by_similarity_to_span":
                span_hidden_states = span_hidden_states.detach().cpu()
                similarity_matrix = self._get_similarity_matrix(span_hidden_states, all_labels_embeddings)

            # exclude any of the labels in batch (those would get this as negative as well!)
            if self.losses[0] == "BCE_loss":
                for label_idx in batch_labels_indices:
                    if label_idx in all_labels_indices:
                        similarity_matrix[:, all_labels_indices.index(label_idx)] = float("-inf")
            else:
                # exclude each real label
                for i, label_idx in enumerate(batch_labels_indices):
                    if label_idx in all_labels_indices:
                        similarity_matrix[i, all_labels_indices.index(label_idx)] = float("-inf")

            if factor > similarity_matrix.shape[1]: # in case not enough candidates were found to fill all negatives
                factor = similarity_matrix.shape[1]

            # only allow respective candidates per mention? or simply search over all?
            if self.candidates:
                similarity_matrix = self._mask_scores(similarity_matrix, datapoints, label_order = all_labels)

            _, hard_negative_indices_sample = torch.topk(similarity_matrix, factor, dim=1)


        if not self.candidates or len(all_candidates) == 0:
            #convert the indices back to the right ones (in case a sample of all labels were used!)
            map_indices = lambda x: all_labels_indices[x]
            hard_negative_indices_sample = hard_negative_indices_sample.apply_(map_indices)

        # now embed those negative labels 'for real', so that gradient tracking possible and most current version
        # important: We need to "enroll" the labels the right way: each span should get its hard negative, spans are concatenated each factor!

        hard_negative_indices_flat = hard_negative_indices_sample.t().flatten().tolist()

        # for efficiency: we do not embed duplicates multiple times!
        hard_negatives_indices_unique = sorted(list(set(hard_negative_indices_flat)))

        if self.candidates and len(all_candidates) >0:
            rt_hard_negative_labels_unique = [all_candidates[idx] for idx in hard_negatives_indices_unique]
            rt_hard_negative_labels_flat = [all_candidates[idx] for idx in hard_negative_indices_flat]
        else:
            rt_hard_negative_labels_unique = [self.label_dict_list[idx] for idx in hard_negatives_indices_unique]
            rt_hard_negative_labels_flat = [self.label_dict_list[idx] for idx in hard_negative_indices_flat]


        if self.custom_label_verbalizations:
            hard_negatives_verbalized_unique = self.custom_label_verbalizations.verbalize_list_of_labels(rt_hard_negative_labels_unique)
            hard_negatives_verbalized_unique = [Sentence(label) for label in hard_negatives_verbalized_unique]
        else:
            hard_negatives_verbalized_unique = [Sentence(label) for label in rt_hard_negative_labels_unique]


        rt_hard_negatives_embeddings_unique = self._embed_batchwise_and_return_hidden_states(encoder=self.label_encoder,
                                                                                             data_to_embed=hard_negatives_verbalized_unique,
                                                                                             data_to_get_embeddings=hard_negatives_verbalized_unique,
                                                                                             clear_embeddings=True,
                                                                                             return_stacked=True
                                                                                             )

        # TODO now we need to go back to not-unique!
        hard_negative_indices_flat_position_in_unique = torch.tensor([hard_negatives_indices_unique.index(idx) for idx in hard_negative_indices_flat]).to(flair.device)
        rt_hard_negatives_embeddings = torch.index_select(rt_hard_negatives_embeddings_unique, dim=0,
                                                          index=hard_negative_indices_flat_position_in_unique)

        if self.add_popularity == "concatenate":
            popularities = torch.Tensor([self.popularity_dict[e]["popularity_normalized"] for e in rt_hard_negative_labels_flat]).to(flair.device)
            rt_hard_negatives_embeddings = torch.cat((rt_hard_negatives_embeddings, popularities.unsqueeze(1)), dim=1)

        return rt_hard_negatives_embeddings, rt_hard_negative_labels_flat


    def _calculate_loss(self, span_hidden_states, label_hidden_states, datapoints, sentences, label_name):
        gold_label_name = label_name
        # todo: Is this right like so? We're comparing the label embeddings of the gold labels to the span embeddings.
        # todo: Shouldn't we compare the labels? Or the embeddings of the predicted labels against the embeddings of the real label?

        gold_labels = [d.get_label(gold_label_name).value for d in datapoints]
        gold_labels_idx = [self.label_dictionary.get_idx_for_item(l) for l in gold_labels]

        if len(gold_labels) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True if self.training else False), 0

        if self.training and self.train_only_with_positive_labels:
            gold_labels_hidden_states = [label_hidden_states[i] for i in gold_labels_idx]
            gold_labels_hidden_states = torch.stack(gold_labels_hidden_states)
            y = torch.ones(len(gold_labels_idx), device=flair.device)

            if self.negative_sampling_factor or self.negative_sampling_strategy == "batch_negatives":
                nr_datapoints = len(gold_labels_idx)

                if self.negative_sampling_strategy == "batch_negatives":
                    self.negative_sampling_factor = len(datapoints) - 1  # naming is confusing, but needed below
                    if len(datapoints) >1:
                        batch_negatives_hidden_states = torch.stack([hidden_state for i, (datapoint, real_hidden_state) in enumerate(zip(datapoints, gold_labels_hidden_states)) for
                                                                            j, hidden_state in enumerate(gold_labels_hidden_states) if i != j])
                    else: # fallback to random if only one datapoint in batch
                        batch_negatives_hidden_states, negative_labels = self._get_random_negative_embeddings(span_hidden_states,
                                                                                             gold_labels,
                                                                                             1,
                                                                                             datapoints)
                        self.negative_sampling_factor = 1

                    gold_labels_hidden_states_with_negatives = torch.cat((gold_labels_hidden_states,
                                                                          batch_negatives_hidden_states), dim=0)

                if self.negative_sampling_strategy == "random":
                    some_random_label_hidden_states, negative_labels = self._get_random_negative_embeddings(span_hidden_states,
                                                                                           gold_labels,
                                                                                           self.negative_sampling_factor,
                                                                                           datapoints)
                    gold_labels_hidden_states_with_negatives = torch.cat((gold_labels_hidden_states,
                                                                          some_random_label_hidden_states), dim = 0)
                if self.negative_sampling_strategy == "hard_negatives":
                    hard_negatives_label_hidden_states, negative_labels = self._get_hard_negative_embeddings(span_hidden_states,
                                                                                            gold_labels,
                                                                                            self.negative_sampling_factor,
                                                                                            datapoints)
                    gold_labels_hidden_states_with_negatives = torch.cat((gold_labels_hidden_states,
                                                                          hard_negatives_label_hidden_states), dim = 0)

                if self.losses[0] != "triplet_loss":
                    gold_labels_hidden_states = gold_labels_hidden_states_with_negatives

                if self.losses[0] in ["BCE_loss", "triplet_loss"]:
                    y = y

                else:
                    span_hidden_states_concat = span_hidden_states
                    for i in range(int(len(negative_labels)/len(datapoints))): #range(self.negative_sampling_factor):
                        span_hidden_states_concat = torch.cat((span_hidden_states_concat, span_hidden_states), dim = 0)
                    span_hidden_states = span_hidden_states_concat

                    y = torch.cat((y, torch.zeros(len(negative_labels), device=flair.device)), dim = 0)


            if self.losses[0] == "BCE_loss":
                logits = self._get_similarity_matrix(span_hidden_states, gold_labels_hidden_states)

                target = torch.zeros(span_hidden_states.shape[0], gold_labels_hidden_states.shape[0], device=flair.device)
                #target[torch.arange(span_hidden_states.shape[0]), torch.arange(len(datapoints))] = y # this wrongly assigns 0 in case the gold label is in there multiple times
                used_labels_batch = gold_labels + negative_labels
                for i,d in enumerate(datapoints):
                    for j,g_label in enumerate(used_labels_batch):
                        if d.get_label(self.label_type).value == g_label:
                            target[i,j] = 1.0
                # for i,d in enumerate(datapoints):
                #     for j in range(self.negative_sampling_factor):
                #         target[i,i+len(datapoints)+len(datapoints)*j] = 0.2
                loss = self.loss_function(logits, target)

            elif self.losses[0] == "triplet_loss":
                anchor = span_hidden_states
                positive = gold_labels_hidden_states
                losses = []
                for i in range(int(hard_negatives_label_hidden_states.shape[0]/len(datapoints))):
                    if self.negative_sampling_strategy == "hard_negatives":
                        negative = hard_negatives_label_hidden_states
                    elif self.negative_sampling_strategy == "random":
                        negative = some_random_label_hidden_states
                    elif self.negative_sampling_strategy == "batch_negatives":
                        negative = batch_negatives_hidden_states

                    negative_current = negative[i*len(datapoints):(i+1)*len(datapoints)]

                    losses.append(self.loss_function(anchor, positive, negative_current))

                #print([l.item() for l in losses])
                loss = torch.mean(torch.stack(losses))

                if "binary_embedding_loss" in self.losses:
                    euclidean_loss_function = EuclideanEmbeddingLoss(self.margin_loss)

                    method = "positives_and_random" #positives_and_random" #"only_positives" # "positives_and_random"
                    if method == "only_positives":
                        # a) also use Euclidean Loss, but only use the positives: #todo Attention! if not in combination with freezing span/label encoder, this results in trivial solution!
                        euclidean_loss = euclidean_loss_function(anchor, positive, torch.ones(len(anchor)).to(flair.device))

                    if method == "positives_and_random":
                        # b) also use the Euclidean loss, but with random negatives
                        random_embeddings, random_labels = self._get_random_negative_embeddings(span_hidden_states,
                                                                      gold_labels,
                                                                      self.negative_sampling_factor,
                                                                      datapoints)

                        span_hidden_states_concat = span_hidden_states
                        for i in range(self.negative_sampling_factor):
                            span_hidden_states_concat = torch.cat((span_hidden_states_concat, span_hidden_states), dim=0)
                        span_hidden_states = span_hidden_states_concat
                        positive_concat_random = torch.cat([positive, random_embeddings])
                        target = torch.zeros(len(span_hidden_states)).to(flair.device)
                        target[:len(positive)] = 1.0
                        euclidean_loss = euclidean_loss_function(span_hidden_states, positive_concat_random, target)

                    if method == "positives_and_negatives":
                        # c) also use the Euclidean loss
                        span_hidden_states_concat = span_hidden_states
                        for i in range(self.negative_sampling_factor):
                            span_hidden_states_concat = torch.cat((span_hidden_states_concat, span_hidden_states),
                                                                  dim=0)
                        span_hidden_states = span_hidden_states_concat
                        positive_concat_negative = torch.cat([positive, hard_negatives_label_hidden_states])
                        target = torch.zeros(len(span_hidden_states)).to(flair.device)
                        target[:len(positive)] = 1.0
                        euclidean_loss = euclidean_loss_function(span_hidden_states, positive_concat_negative, target)
                    loss += euclidean_loss

            elif self.losses[0] == "binary_embedding_loss": # normal CosineEmbeddingLoss or EuclideanEmbeddingLoss
                y = torch.where(y == 0, -1, 1)
                loss = self.loss_function(span_hidden_states, gold_labels_hidden_states, y).unsqueeze(0)

            else:
                raise NotImplementedError

        else:

            #if self.losses[0] == "BCE_loss":

            #    logits = self._get_similarity_matrix(span_hidden_states, label_hidden_states)

            #    target = torch.zeros(span_hidden_states.shape[0], label_hidden_states.shape[0],
            #                         device=flair.device if self.training else "cpu")

            #    target[torch.arange(span_hidden_states.shape[0]), gold_labels_idx] = 1

            #    loss = self.loss_function(logits, target)

            # else: # TODO this is taking cosine_loss everytime, need to adapt it to all different kinds of losses.
            # However, this has no effect as it's just the evaluation loss
            gold_labels_hidden_states = [self.current_label_embeddings[i] for i in gold_labels_idx]
            gold_labels_hidden_states = torch.stack(gold_labels_hidden_states).to(flair.device)
            y = torch.ones(len(gold_labels_idx), device=flair.device)
            if self.margin_loss == None:
                self.margin_loss = 1.0
            cosine_loss = torch.nn.CosineEmbeddingLoss(margin=self.margin_loss,
                                                       reduction= "mean",
                                                       )
            loss = cosine_loss(span_hidden_states.to(flair.device), gold_labels_hidden_states, y).unsqueeze(0)

        return loss, len(datapoints)

    def forward_loss(self, sentences) -> Tuple[torch.Tensor, int]:

        if self.training:
            self.is_first_batch_in_evaluation = True # set this to assure new embedding of labels in the following evaluation phase

        if self.is_first_batch_in_training:
            self.current_label_embeddings, self.current_label_embeddings_indices = self._embed_labels_batchwise(cpu=True, max_limit=None if len(
                self.label_dict_list) < 10000 else 10000)  # before training, necessary for negative sampling
            #self.current_label_embeddings, self.current_label_embeddings_indices = self._embed_labels_batchwise(cpu=True, max_limit=200) # only use random sample of labels so hard negatoves are not too hard
            self.is_first_batch_in_training = False

        if not [spans for sentence in sentences for spans in sentence.get_spans(self.label_type)]:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        span_hidden_states, label_hidden_states, datapoints, sentences = self._encode_data_points(sentences)

        if self.train_greedy == "use_predicted" or self.train_greedy == True:
            span_hidden_states, label_hidden_states = self.predict(sentences, return_span_and_label_hidden_states=True)

        loss, nr_datapoints = self._calculate_loss(span_hidden_states, label_hidden_states, datapoints, sentences, label_name=self.label_type)

        return loss, nr_datapoints

    def _mask_scores(self, scores: torch.Tensor, data_points: List[Span], label_order):
        if not self.candidates:
            return scores

        masked_scores = -torch.inf * torch.ones(scores.size(), requires_grad=True, device=scores.device)

        for idx, span in enumerate(data_points):
            # get the candidates
            candidate_set = self.candidates.get_candidates(span.text)
            # during training, add the gold value as candidate
            if self.training:
                candidate_set.add(span.get_label(self.label_type).value)
            candidate_set.add("<unk>")
            indices_of_candidates = [label_order.index(candidate) for candidate in candidate_set if candidate in label_order]
            masked_scores[idx, indices_of_candidates] = scores[idx, indices_of_candidates]

        return masked_scores

    def _add_already_predicted_label_verbalizations_to_sentences(self,
                                                                 sentences,
                                                                 datapoints,
                                                                 leave_out_datapoints_to_be_predicted: True,
                                                                 max_labels_to_add = 10, # Todo
                                                                 use_gold = False,
                                                                 use_gold_percentage = None, #0.9, #1.0, # todo
                                                                 also_add_in_context = True,          # TODO insert predicted verbalizations also in context sentences?
                                                                 method = "bracket_after_each_mention",#"concatenate_all_directly_after_mention", #"bracket_after_each_mention", #"concatenate_all_directly_after_mention", #"bracket_after_each_mention"
                                                                 cut_label_name=False,
                                                                 use_gibberish = False, # TDOO
                                                                 ):
        modified_sentences_per_datapoint = []
        datapoints_in_modified_sentences = [None for d in datapoints]
# schnupsi was HERE!!!
        for d_i, datapoint in enumerate(datapoints):
            s = datapoint.sentence
            datapoints_in_sentence = s.get_spans("nel")
            sentence_before = s.text
            sentence_with_verbalization = sentence_before
            if use_gold:
                labels_already = s.get_spans("nel")
                if leave_out_datapoints_to_be_predicted:
                    if datapoint in labels_already:
                        labels_already.remove(datapoint)
                if use_gold_percentage:
                    labels_already = random.sample(labels_already, int(len(labels_already)* use_gold_percentage)) # old
                else:
                    use_number_of_labels = random.randint(0, len(labels_already))
                    labels_already = random.sample(labels_already, use_number_of_labels)

            else:
                labels_already = s.get_spans("predicted")
                if leave_out_datapoints_to_be_predicted:
                    if datapoint in labels_already:
                        labels_already.remove(datapoint)

            if max_labels_to_add and len(labels_already) > max_labels_to_add:
                random.sample(labels_already, max_labels_to_add) # TODO: good? For sentences with huge number of mentions
            labels_already.sort(key=lambda a: a.start_position)
            added_characters = 0
            modified_datapoints_offsets = [[d.start_position, d.end_position] for d in datapoints_in_sentence]

            for i,l in enumerate(labels_already):
                if use_gold:
                    label = l.get_label("nel").value
                else:
                    label = l.get_label("predicted").value

                if method == "bracket_after_each_mention":
                    add_at_position = l.end_position + added_characters
                if method == "concatenate_all_directly_after_mention":
                    add_at_position = datapoint.end_position + added_characters

                if label not in self.label_dict_list:
                    if self.custom_label_verbalizations:
                        verbalization = self.custom_label_verbalizations.verbalizations[label]
                        if cut_label_name:
                            index_separator = verbalization.find(self.custom_label_verbalizations.delimiter)
                            if index_separator != -1:
                                verbalization = verbalization[index_separator + 2:]
                    else:
                        verbalization = label

                else:
                    label_idx = self.label_dictionary.get_idx_for_item(label)
                    if self.custom_label_verbalizations:
                        verbalization = self.custom_label_verbalizations.verbalized_labels[label_idx]
                        if cut_label_name:
                            index_separator = verbalization.find(self.custom_label_verbalizations.delimiter)
                            if index_separator != -1:
                                verbalization = verbalization[index_separator+2:]
                    else:
                        verbalization = self.label_dict_list[label_idx]

                if use_gibberish:
                    import string
                    verbalization = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))

                sentence_with_verbalization = sentence_before[:add_at_position] + \
                                              f'{" (" if method == "bracket_after_each_mention" or i == 0 else "| "}' \
                                              + verbalization + \
                                              f'{")" if method == "bracket_after_each_mention" or i == len(labels_already) - 1 else ""}' \
                                              + sentence_before[add_at_position:]
                added_just_now = len(sentence_with_verbalization) - len(sentence_before)
                added_characters += added_just_now
                sentence_before = sentence_with_verbalization

                for counter,d in enumerate(datapoints_in_sentence):
                    d_start_position = modified_datapoints_offsets[counter][0]
                    d_end_position = modified_datapoints_offsets[counter][1]

                    if d_start_position > add_at_position:
                        d_start_position += added_just_now
                        d_end_position += added_just_now
                        modified_datapoints_offsets[counter][0] = d_start_position
                        modified_datapoints_offsets[counter][1] = d_end_position

            new_sentence = flair.data.Sentence(sentence_with_verbalization)

            for i, datapoint_in_sentence in enumerate(datapoints_in_sentence):
                start, end = modified_datapoints_offsets[i]
                d_start_token = [i for i, t in enumerate(new_sentence.tokens) if t.start_position == start]
                d_end_token = [i for i, t in enumerate(new_sentence.tokens) if t.end_position == end]
                if datapoint_in_sentence == datapoint:
                    datapoints_in_modified_sentences[d_i] = flair.data.Span(new_sentence.tokens[d_start_token[0]:d_end_token[0] + 1])

            new_sentence.copy_context_from_sentence(s)

            if also_add_in_context:
                relevant_sentences_before = []
                at_sentence = new_sentence
                len_context_before = 0
                while len_context_before <= self.token_encoder.context_length:
                    if "DOCSTART" in at_sentence.text or not at_sentence._previous_sentence:
                        break
                    relevant_sentences_before.append(at_sentence._previous_sentence)
                    at_sentence = at_sentence._previous_sentence
                    len_context_before += len(at_sentence)

                relevant_sentences_after = []
                at_sentence = new_sentence
                len_context_after = 0
                while len_context_after <= self.token_encoder.context_length:
                    if "DOCSTART" in at_sentence.text or not at_sentence._next_sentence:
                        break
                    relevant_sentences_after.append(at_sentence._next_sentence)
                    at_sentence = at_sentence._next_sentence
                    len_context_after += len(at_sentence)

                context_sentences = relevant_sentences_before + relevant_sentences_after
                context_sentences_left_modified = []
                context_sentences_right_modified = []
                for i, s in enumerate(context_sentences):
                    datapoints_in_sentence = s.get_spans("nel")
                    sentence_before = s.text
                    sentence_with_verbalization = sentence_before
                    if use_gold:
                        labels_already = s.get_spans("nel")
                        if use_gold_percentage:
                            labels_already = random.sample(labels_already,
                                                           int(len(labels_already) * use_gold_percentage))  # old
                        else:
                            use_number_of_labels = random.randint(0, len(labels_already))
                            labels_already = random.sample(labels_already, use_number_of_labels)
                    else:
                        labels_already = s.get_spans("predicted")

                    if max_labels_to_add and len(labels_already) > max_labels_to_add:
                        random.sample(labels_already, max_labels_to_add)
                    labels_already.sort(key=lambda a: a.start_position)
                    added_characters = 0
                    modified_datapoints_offsets = [[d.start_position, d.end_position] for d in datapoints_in_sentence]

                    for l in labels_already:
                        if use_gold:
                            label = l.get_label("nel").value
                        else:
                            label = l.get_label("predicted").value
                        if label not in self.label_dict_list:
                            if self.custom_label_verbalizations:
                                verbalization = self.custom_label_verbalizations.verbalizations[label]
                                if cut_label_name:
                                    index_separator = verbalization.find(self.custom_label_verbalizations.delimiter)
                                    if index_separator != -1:
                                        verbalization = verbalization[index_separator + 2:]
                            else:
                                verbalization = label

                        else:
                            label_idx = self.label_dictionary.get_idx_for_item(label)
                            if self.custom_label_verbalizations:
                                verbalization = self.custom_label_verbalizations.verbalized_labels[label_idx]
                                if cut_label_name:
                                    index_separator = verbalization.find(self.custom_label_verbalizations.delimiter)
                                    if index_separator != -1:
                                        verbalization = verbalization[index_separator + 2:]
                            else:
                                verbalization = self.label_dict_list[label_idx]

                        if use_gibberish:
                            import string
                            verbalization = ''.join(
                                random.choice(string.ascii_letters + string.digits) for _ in range(15))


                        add_at_position = l.end_position + added_characters

                        sentence_with_verbalization = sentence_before[:add_at_position] + \
                                                      "(" + verbalization + ")" + \
                                                      sentence_before[add_at_position:]
                        added_just_now = len(sentence_with_verbalization) - len(sentence_before)
                        added_characters += added_just_now
                        sentence_before = sentence_with_verbalization

                    new_context_sentence = flair.data.Sentence(sentence_with_verbalization)
                    if i < len(relevant_sentences_before):
                        context_sentences_left_modified.append(new_context_sentence)
                    else:
                        context_sentences_right_modified.append(new_context_sentence)

                at_sentence = new_sentence
                for context_sentence in context_sentences_left_modified:
                    at_sentence._previous_sentence = context_sentence
                    at_sentence = context_sentence

                at_sentence = new_sentence
                for context_sentence in context_sentences_right_modified:
                    at_sentence._next_sentence = context_sentence
                    at_sentence = context_sentence

            modified_sentences_per_datapoint.append(new_sentence)

        if len(datapoints) != len(datapoints_in_modified_sentences):
            print("here not the same!")
        return datapoints_in_modified_sentences, modified_sentences_per_datapoint

    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
        return_span_and_label_hidden_states: bool = True,
    ):
        with torch.no_grad() if not self.training else torch.enable_grad():

            if self.training:
                label_name = "predicted"

            for s in sentences:
                [d.remove_labels("top_5") for d in s.get_spans("nel")]

            span_hidden_states, label_hidden_states, datapoints, sentences = self._encode_data_points(sentences)
            if len(datapoints) != 0:

                if self.candidates:
                    method = "add_all_candidates" #"add_all_candidates"
                    if method == "add_all_candidates":
                        # we need to: a) add the candidates that are not yet in there, b) mask the non-candidates for each span                        new_labels = set()
                        candidates_for_batch = set()
                        for d in datapoints:
                            candidate_set = self.candidates.get_candidates(d.text)
                            if len(candidate_set) >0:
                                candidates_for_batch.update(candidate_set)
                            else:
                                candidates_for_batch.add("<unk>")
                        candidates_for_batch = list(candidates_for_batch)
                        new_labels = [l for l in candidates_for_batch if l not in self.label_dict_list]

                        indices_of_candidates = [self.label_dict_list.index(c) for c in candidates_for_batch if
                                                 c in self.label_dict_list]

                        if len(indices_of_candidates) <= 1:
                            print("here, len of candidates:", len(indices_of_candidates))
                        label_hidden_states = label_hidden_states[[indices_of_candidates]] # no need to compare to all labels
                        label_ordering = [self.label_dict_list[i] for i in indices_of_candidates]

                        if len(new_labels) >0:
                            if self.custom_label_verbalizations:
                                new_labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(new_labels)
                                new_labels_verbalized = [Sentence(label) for label in new_labels_verbalized]
                            else:
                                new_labels_verbalized = [Sentence(label) for label in new_labels]

                            new_labels_hidden_states_batch = self._embed_batchwise_and_return_hidden_states(
                                encoder=self.label_encoder,
                                data_to_embed=new_labels_verbalized,
                                data_to_get_embeddings=new_labels_verbalized,
                                clear_embeddings=True,
                                )
                            new_labels_hidden_states_batch = new_labels_hidden_states_batch.detach().cpu()

                            label_hidden_states = torch.cat((label_hidden_states, new_labels_hidden_states_batch), dim = 0)

                            label_ordering = label_ordering + new_labels

                    if method == "only_if_in_label_dict":
                        label_ordering = self.label_dict_list # todo make more efficient: no need to compare to all labels! see above
                else:
                    label_ordering = self.label_dict_list

                if not self.training:
                    span_hidden_states = span_hidden_states.detach().cpu()
                    label_hidden_states = label_hidden_states.detach().cpu()

                if self.predict_greedy:
                    sentences_printable = [d.sentence.text for d in datapoints]

                similarity = self._get_similarity_matrix(span_hidden_states, label_hidden_states)

                if self.candidates:
                    similarity = self._mask_scores(similarity, data_points = datapoints, label_order = label_ordering)

                # just for inspection: get the top N (=5) most probable labels:
                top_confidences, top_indices = torch.topk(similarity, k=min(5, similarity.shape[1]), dim=1)
                top_confidences = [[round(float(tensor), 5) for tensor in inner_list] for inner_list in
                                  top_confidences]

                top_labels = [[label_ordering[index] for index in indices] for indices in top_indices]
                original_top_labels = top_labels
                # take the highest one
                final_label_indices = top_indices[:, 0]

                #print(final_label_indices)

                if not self.predict_greedy:
                    for i,d in enumerate(datapoints):
                        label_idx = final_label_indices[i]
                        conf = similarity[i, label_idx]

                        label = label_ordering[label_idx]

                        d.set_label(label_name,
                                    value=label,
                                    score=float(conf)
                                    )
                        d.set_label(typename = "top_5", value = " | ".join(f"{item1} {item2}" for item1, item2 in zip(top_labels[i], top_confidences[i])))

                # predict_greedy:
                else:
                    if not self.training or len(datapoints) > 1: # in training, if it's just one datapoint, not necessary

                        PREDICT_SIZE = int(len(datapoints)/3) if self.training else int(len(datapoints)/5) #TODO
                        PREDICT_SIZE = max(1, PREDICT_SIZE) # in case len(datapoints) is <=3, so it's not 0
                        #print("predict size:", PREDICT_SIZE)
                        #criterion = "similarity"      # the ones with the highest similarity scores
                        criterion = "clear_distance"  # the ones where the distance to the next probable one is very high

                        if criterion == "clear_distance":
                            def custom_sorting_key(list_of_scores):
                                first_score = list_of_scores[0]
                                if len(list_of_scores) > 1:
                                    mean_distance = sum(abs(first_score - score) for score in list_of_scores[1:]) / len(
                                        list_of_scores[1:])
                                    distance_to_next = first_score - list_of_scores[1]
                                else: mean_distance = 10
                                return mean_distance # distance_to_next

                        first = True
                        still_to_predict_indices = list(range(len(datapoints)))

                        while len(still_to_predict_indices) >0:
                            if first:
                                if criterion == "similarity":
                                    confidences = [similarity[i, final_label_indices[i]] for i in range(len(datapoints)) ]
                                    _, high_confidence_indices = torch.topk(torch.Tensor(confidences), min(PREDICT_SIZE, len(datapoints)), dim=0)
                                    high_confidence_indices = [int(tensor) for tensor in high_confidence_indices]

                                if criterion == "clear_distance":
                                    sorted_indices = sorted(range(len(top_confidences)),
                                                            key=lambda i: custom_sorting_key(top_confidences[i]),
                                                            reverse=True)
                                    high_confidence_indices = sorted_indices[:min(PREDICT_SIZE, len(datapoints))]

                                first = False

                            else:

                                datapoints_modified, new_sentences = self._add_already_predicted_label_verbalizations_to_sentences(
                                    sentences, datapoints, leave_out_datapoints_to_be_predicted=True,
                                    use_gold = False)

                                #print("Nr sentences to embed:", len(new_sentences))
                                still_to_predict_span_hidden_states = self._embed_batchwise_and_return_hidden_states(
                                    encoder=self.token_encoder,
                                    data_to_embed=[new_sentences[i] for i in still_to_predict_indices],
                                    data_to_get_embeddings=[datapoints_modified[i] for i in still_to_predict_indices],
                                    clear_embeddings=True,
                                    )

                                if not self.training:
                                    still_to_predict_span_hidden_states = still_to_predict_span_hidden_states.detach().cpu()

                                still_to_predict_similarity = self._get_similarity_matrix(still_to_predict_span_hidden_states, label_hidden_states)
                                if self.candidates:
                                    still_to_predict_datapoints = [datapoints[i] for i in still_to_predict_indices]
                                    still_to_predict_similarity = self._mask_scores(still_to_predict_similarity,
                                                                                    data_points=still_to_predict_datapoints,
                                                                                    label_order=label_ordering)

                                if still_to_predict_similarity.shape[1] < 5:
                                    print("here, len used labels:", len(label_ordering))
                                still_to_predict_top_confidences, still_to_predict_top_indices = torch.topk(still_to_predict_similarity, k=min(5, still_to_predict_similarity.shape[1]), dim=1)

                                still_to_predict_final_labels_indices = still_to_predict_top_indices[:, 0]
                                still_to_predict_confidences = [still_to_predict_similarity[i, still_to_predict_final_labels_indices[i]] for i in range(len(still_to_predict_indices)) ]

                                if criterion == "similarity":
                                    still_to_predict_high_confidences, still_to_predict_high_confidence_indices = torch.topk(torch.Tensor(still_to_predict_confidences),
                                                                                             k=min(PREDICT_SIZE, len(still_to_predict_indices)), # in case less than 5
                                                                                             dim=0)
                                    still_to_predict_high_confidence_indices = [int(tensor) for tensor in still_to_predict_high_confidence_indices]

                                if criterion == "clear_distance":
                                    sorted_indices = sorted(range(len(still_to_predict_top_confidences)),
                                                            key=lambda i: custom_sorting_key(
                                                                still_to_predict_top_confidences[i]),
                                                            reverse=True)
                                    still_to_predict_high_confidence_indices = sorted_indices[:min(PREDICT_SIZE, len(datapoints))]

                                # map indices back to original order in datapoints
                                high_confidence_indices = [still_to_predict_indices[i] for i in still_to_predict_high_confidence_indices]

                                # and write it into the global lists:
                                for nr, i_local in enumerate(still_to_predict_high_confidence_indices):
                                    i_global = still_to_predict_indices[i_local]
                                    final_label_indices[i_global] = still_to_predict_final_labels_indices[i_local]
                                    similarity[i_global] = still_to_predict_similarity[i_local, still_to_predict_final_labels_indices[i_local]]
                                    #similarity[i_global] = still_to_predict_high_confidences[nr]
                                    top_labels[i_global] = [label_ordering[i] for i in still_to_predict_top_indices[i_local]]
                                    top_confidences[i_global] = [round(float(i), 3) for i in still_to_predict_top_confidences[i_local]]
                                    sentences_printable[i_global] = datapoints_modified[i_global].sentence.text

                            for i, d in enumerate(datapoints):
                                if i not in high_confidence_indices:
                                    continue
                                label_idx = final_label_indices[i]
                                conf = similarity[i, label_idx]

                                label = label_ordering[label_idx]

                                d.set_label(label_name,
                                            value=label,
                                            score=float(conf)
                                            )

                                d.set_label(typename="top_5", value=" | ".join(
                                    f"{item1} {item2}" for item1, item2 in zip(top_labels[i], top_confidences[i])))

                                d._input_text = sentences_printable[i]

                            still_to_predict_indices = [i for i, d in enumerate(datapoints) if d.get_label(
                                "predicted").value == "O"]  # predict the ones that have not been predicted

        if not self.training and self.predict_again_finally:

            if len(datapoints) > 0:
                datapoints_modified, new_sentences = self._add_already_predicted_label_verbalizations_to_sentences(
                    sentences, datapoints, leave_out_datapoints_to_be_predicted=True, use_gold = False)

                span_hidden_states = self._embed_batchwise_and_return_hidden_states(encoder=self.token_encoder,
                                                                                    data_to_embed=new_sentences,
                                                                                    data_to_get_embeddings=datapoints_modified,
                                                                                    clear_embeddings=True,
                                                                                    )

                # self.token_encoder.embed(new_sentences)
                # span_hidden_states = torch.stack(
                #     [self.aggregated_embedding(d, self.token_encoder.get_names()) for d in datapoints_modified])
                # [s.clear_embeddings() for s in new_sentences]

                span_hidden_states = span_hidden_states.detach().cpu()
                label_hidden_states = label_hidden_states.detach().cpu()

                similarity = self._get_similarity_matrix(span_hidden_states, label_hidden_states)
                if self.candidates:
                    similarity = self._mask_scores(similarity, data_points=datapoints,
                                                               label_order=label_ordering)

                # just for inspection: get the top N (=5) most probable labels:
                top_confidences, top_indices = torch.topk(similarity, k=min(5, similarity.shape[1]), dim=1)
                top_confidences = [[round(float(tensor), 5) for tensor in inner_list] for inner_list in
                                   top_confidences]

                top_labels = [[label_ordering[index] for index in indices] for indices in top_indices]
                # take the highest one
                final_label_indices = top_indices[:, 0]

                for i, d in enumerate(datapoints):
                    label_idx = final_label_indices[i]
                    label = label_ordering[label_idx]
                    conf = similarity[i, label_idx]

                    d.set_label(label_name,
                                value=label,
                                score=float(conf)
                                )

                    d.set_label(typename="top_5", value=" | ".join(
                        f"{item1} {item2}" for item1, item2 in zip(top_labels[i], top_confidences[i])))

                    if datapoints_modified[i].sentence.text != new_sentences[i].text:
                        print("here")
                    d._input_text = datapoints_modified[i].sentence.text

        if return_loss:
            if len(datapoints) == 0:
                return torch.tensor(0.0, dtype=torch.float, device=flair.device,
                                    requires_grad=False), 0
            # TODO this is not right yet
            return self._calculate_loss(
                span_hidden_states, label_hidden_states, datapoints, sentences, label_name = self.label_type
            )

        if return_span_and_label_hidden_states:
            if len(datapoints) > 0:
                #embed again with now ALL predicted labels in sentence (all but the one in question)
                # and return those. And of course the label embeddings as well.
                datapoints_modified, new_sentences = self._add_already_predicted_label_verbalizations_to_sentences(
                    sentences, datapoints, leave_out_datapoints_to_be_predicted=True)

                #print(len(new_sentences))

                span_hidden_states = self._embed_batchwise_and_return_hidden_states(encoder = self.token_encoder,
                                                                                    data_to_embed=new_sentences,
                                                                                    data_to_get_embeddings=datapoints_modified,
                                                                                    clear_embeddings=True,
                                                                                    )

                if self.training:
                    [d.remove_labels("predicted") for d in datapoints] # necessary so that in epoch 2 the while loop condition takes effect!
                    [d.remove_labels("top_5") for d in datapoints]

        return span_hidden_states, label_hidden_states



    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            eval_line = f"\n{datapoint.to_original_text()}\n"

            for span in datapoint.get_spans(gold_label_type):
                symbol = "✓" if span.get_label(gold_label_type).value == span.get_label("predicted").value else "❌"
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {span.get_label("predicted").value} ({symbol}) top_5: {span.get_label("top_5").value}\n'
                )
                if self.predict_greedy:
                    eval_line += (
                    f'  <-- "{span._input_text}"\n\n'
                    )

            lines.append(eval_line)
        return lines

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "label_encoder": self.label_encoder,
            "token_encoder": self.token_encoder,
            "label_type": self.label_type,
            "label_dictionary": self.label_dictionary,
            "pooling_operation": self.pooling_operation,
            "custom_label_verbalizations": self.custom_label_verbalizations,
            "train_only_with_positive_labels" : self.train_only_with_positive_labels,
            "negative_sampling_factor": self.negative_sampling_factor,
            "negative_sampling_strategy": self.negative_sampling_strategy,
            "margin_loss": self.margin_loss,
            "losses": self.losses,
            "distance_function": self.distance_function,
            "label_embeddings_save_path": self.label_embeddings_save_path,
            "is_first_batch_in_evaluation": self.is_first_batch_in_evaluation,
            "weighted_loss": self.weighted_loss,
            "max_verbalization_length": self.max_verbalization_length,
            "add_popularity": self.add_popularity,
            "popularity_save_path": self.popularity_save_path,
            "predict_greedy": self.predict_greedy,
            "predict_again_finally": self.predict_again_finally,
            "add_property_prefixes": self.add_property_prefixes,
            #"candidates": self.candidates,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        return super()._init_model_with_state_dict(
            state,
            label_encoder = state.get("label_encoder"),
            token_encoder = state.get("token_encoder"),
            label_type = state.get("label_type"),
            label_dictionary = state.get("label_dictionary"),
            pooling_operation = state.get("pooling_operation"),
            custom_label_verbalizations = state.get("custom_label_verbalizations"),
            train_only_with_positive_labels = state.get("train_only_with_positive_labels"),
            negative_sampling_factor = state.get("negative_sampling_factor"),
            negative_sampling_strategy = state.get("negative_sampling_strategy"),
            margin_loss = state.get("margin_loss"),
            losses = state.get("losses"),
            distance_function=state.get("distance_function"),
            label_embeddings_save_path = state.get("label_embeddings_save_path"),
            weighted_loss = state.get("weighted_loss"),
            max_verbalization_length = state.get("max_verbalization_length"),
            add_popularity = state.get("add_popularity"),
            popularity_save_path = state.get("popularity_save_path"),
            predict_greedy = state.get("predict_greedy"),
            predict_again_finally = state.get("predict_again_finally"),
            #add_property_prefixes = state.get("add_property_prefixes"),
            #candidates = state.get("candidates")
            **kwargs,
        )

