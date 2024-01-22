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
from flair.models.entity_linker_model import CandidateGenerator

from pathlib import Path

from flair.nn.wikidata_utils import get_wikidata_categories, get_sitelinks_of_entity

log = logging.getLogger("flair")

class WikidataLabelVerbalizer:
    def __init__(self,
                 label_dictionary,
                 verbalizations_paths: List,
                 use_wikidata_description: bool = False,
                 use_wikidata_classes: bool = True,
                 max_verbalization_length: int = 16,
                 add_occupation: bool = True,
                 add_field_of_work: bool =False,
                 ):

        self.label_dictionary = label_dictionary
        self.verbalizations_paths = verbalizations_paths
        self.use_wikidata_description = use_wikidata_description
        self.use_wikidata_classes = use_wikidata_classes
        self.add_occupation = add_occupation
        self.add_field_of_work = add_field_of_work

        self.verbalized_labels = self.verbalize_all_labels(self, label_dictionary=self.label_dictionary, save_verbalizations=True)
        self.label_dict_list = self.label_dictionary.get_items()

        self.max_verbalization_length = max_verbalization_length

        if self.max_verbalization_length:
            verbalized_labels_truncated = []
            for l in self.verbalized_labels:
                s = flair.data.Sentence(l)
                s.tokens = s.tokens[:self.max_verbalization_length]
                verbalized_labels_truncated.append(s.text)
            self.verbalized_labels = verbalized_labels_truncated

    def verbalize_entity(self, entity):
        wikipedia_label = entity
        wikidata_info = get_wikidata_categories(wikipedia_label, method="only_one_level_up",
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

        return verbalized

    @staticmethod
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

            verbalized_labels.append(verbalized)
            self.verbalizations[entity] = verbalized
            print("Verbalized id", idx, ":", str_label, "->", verbalized)

            # save every n entities and final
            if save_verbalizations:
                if counter_newly_verbalized > 0 and (idx % 100 == 0 or idx >= len(label_dictionary.item2idx.items()) -1):
                    verbalizations_dict = self.verbalizations
                    with open(f"{self.verbalizations_paths[0]}", "w") as file:
                        json.dump(verbalizations_dict, file)
                    print("saved new verbalizations at:", self.verbalizations_paths[0])

        print(f"--- Created verbalized labels for {len(verbalized_labels)} labels")
        print(f"--- Thereof newly verbalized:", counter_newly_verbalized)

        return verbalized_labels

    def verbalize_list_of_labels(self, list_of_labels):
        if set(list_of_labels).issubset(self.label_dict_list):
            label_idx = [self.label_dictionary.get_idx_for_item(idx) for idx in list_of_labels]
            rt =  [self.verbalized_labels[idx] for idx in label_idx ]

        else:
            rt = []
            for l in list_of_labels:
                if l in self.verbalizations:
                    rt.append(self.verbalizations[l])
                else:
                    verbalization = self.verbalize_entity(l)
                    self.verbalizations[l] = verbalization
                    rt.append(verbalization)
        return rt

    def add_new_verbalization_path(self, json_path):
        self.verbalizations_paths.append(json_path)



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
        custom_label_verbalizations = None,
        train_only_with_positive_labels = True,
        negative_sampling_factor: [int, bool] = 1,
        negative_sampling_strategy: str = "random",
        margin_loss: float = 0.5,
        threshold_in_prediction: float = 0.5,
        weighted_loss = False,
        add_popularity = False,
        popularity_save_path: Path = None,
        label_embeddings_save_path: Path = None,
        BCE_loss: bool = False,
        candidates: Optional[CandidateGenerator] = None,

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
        self.train_only_with_positive_labels = train_only_with_positive_labels
        self.negative_sampling_factor = negative_sampling_factor
        self.negative_sampling_strategy = negative_sampling_strategy
        self.is_first_batch_in_evaluation = True
        if not self.train_only_with_positive_labels:
            self.negative_sampling_factor = False
        if isinstance(self.negative_sampling_factor, bool):
            if self.negative_sampling_factor:
                self.negative_sampling_factor = 1
        if negative_sampling_factor >=1:
            self.train_only_with_positive_labels = True
            assert self.negative_sampling_strategy in ["hard_negatives", "random"], \
                ("You need to choose between 'random' and 'hard_negatives' as a negative_sampling_strategy!")

        self.weighted_loss = weighted_loss
        self.add_popularity = add_popularity
        self.popularity_save_path = popularity_save_path
        if self.add_popularity:
            if not self.popularity_save_path:
                self.popularity_save_path = "/tmp/wikipedia_sitelinks_dict.json"
            self._create_popularity_dict(self.label_dict_list)

        self.BCE_loss = BCE_loss
        if self.BCE_loss:
            self.loss_function = torch.nn.BCEWithLogitsLoss(
                                                            #pos_weight=torch.Tensor(negative_sampling_factor)
                                                            )
            self.margin_loss = None
            self.threshold_in_prediction = None
        else:
            self.margin_loss = margin_loss
            self.loss_function = torch.nn.CosineEmbeddingLoss(margin=self.margin_loss,
                                                              reduction="none" if self.weighted_loss else "mean",
                                                              )
            self.threshold_in_prediction = threshold_in_prediction
        self.label_embeddings_save_path = label_embeddings_save_path
        if self.label_embeddings_save_path:
            self.label_embeddings_save_path.mkdir(parents=True, exist_ok=True)

        self.current_label_embeddings, _ = self._embed_labels_batchwise(cpu=True, max_limit=None) # before training

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
            self.sitelinks_dict = json.load(handle)

        print(f"Found sitelinks_dict with {len(self.sitelinks_dict)} entries")
        if not all(item in self.sitelinks_dict for item in labels):
            for i, e in enumerate(self.label_dict_list):
                if e not in self.sitelinks_dict:
                    sitelinks = get_sitelinks_of_entity(e)
                    self.sitelinks_dict[e] = {"sitelinks": sitelinks}
                    print(i, e, self.sitelinks_dict[e]["sitelinks"])

                if i % 100 == 0:
                    print("saving...")
                    with open(self.popularity_save_path, "w") as f:
                        json.dump(self.sitelinks_dict, f)

            with open(self.popularity_save_path, "w") as f:
                json.dump(self.sitelinks_dict, f)

        # get highest sitelinks value for normalizing
        # max_sitelinks = 0
        # for entity, info in self.sitelinks_dict.items():
        #     if "sitelinks" in info:
        #         sitelinks_count = int(info["sitelinks"])
        #         if sitelinks_count > max_sitelinks:
        #             max_sitelinks = sitelinks_count

        # or just use N as max, so everything above N gets 1.0
        max_sitelinks = 100

        # now normalize:
        for entity, info in self.sitelinks_dict.items():
            if info["sitelinks"]:
                self.sitelinks_dict[entity]["sitelinks_normalized"] = min(
                    round(int(info["sitelinks"]) / max_sitelinks, 6), 1.0)
            else:
                self.sitelinks_dict[entity][
                    "sitelinks_normalized"] = 0.0  # Dummy, if sitelinks was None (e.g. article was deleted)

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
            self.popularity_mapping = {(0, 0.25): "Low Popularity",
                                       (0.25, 0.5): "Medium Popularity",
                                       (0.5, 0.75): "High Popularity",
                                       (0.75, 1.0): "Very High Popularity"
                                       }

            for entity, info in self.sitelinks_dict.items():
                for range_, label in self.popularity_mapping.items():
                    if range_[0] <= self.sitelinks_dict[entity]["sitelinks_normalized"] <= range_[1]:
                        verbalization = label
                self.sitelinks_dict[entity]["sitelinks_verbalized"] = verbalization

        print("done with preprocessing sitelinks info")

    def _embed_labels_batchwise(self, batch_size=128, cpu = False, max_limit = None):
        print(f"Now creating label embeddings with limit {max_limit}...")
        if cpu:
            label_embeddings = torch.tensor([], device="cpu")
        else:
            label_embeddings = torch.tensor([], device=flair.device)

        import random
        used_indices = sorted(random.sample(range(len(self.label_dictionary)), len(self.label_dictionary) if not max_limit else max_limit))
        used_labels = np.array(self.label_dict_list)[used_indices]

        if self.add_popularity == "verbalize":
            popularities_verbalized = [self.sitelinks_dict[label]["sitelinks_verbalized"] for label in used_labels]

        if self.custom_label_verbalizations:
            used_labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(used_labels)
            if self.add_popularity == "verbalize":
                used_labels_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                          in zip(used_labels_verbalized, popularities_verbalized)]
            else:
                used_labels_verbalized = [Sentence(label) for label in used_labels_verbalized]
        else:
            if self.add_popularity == "verbalize":
                used_labels_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                          in zip(used_labels, popularities_verbalized)]
            else:
                used_labels_verbalized = [Sentence(label) for label in used_labels]

        for i in range(0, len(used_indices), batch_size):
            #print(i)
            #print(torch.cuda.memory_summary(device=flair.device, abbreviated=False))
            labels_batch = used_labels_verbalized[i:i + batch_size]
            if cpu:
                self.label_encoder.embed(labels_batch)
                label_embeddings_batch = torch.stack([label.get_embedding() for label in labels_batch]).detach().cpu()
                #label_embeddings_batch = torch.stack([label.get_embedding() for label in labels_batch])

            else:
                self.label_encoder.embed(labels_batch)
                label_embeddings_batch = torch.stack([label.get_embedding() for label in labels_batch])

            label_embeddings = torch.cat((label_embeddings, label_embeddings_batch), dim = 0)
            del label_embeddings_batch

            for s in labels_batch:
                s.clear_embeddings()

        if self.add_popularity == "concatenate":
            popularities = torch.Tensor([self.sitelinks_dict[e]["sitelinks_normalized"] for e in used_labels])
            label_embeddings = torch.cat((label_embeddings, popularities.unsqueeze(1)), dim = 1)

        del used_labels
        return label_embeddings, used_indices


    def _encode_data_points(self, sentences): #-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        self.token_encoder.embed(sentences)

        datapoints = []
        for s in sentences:
            datapoints.extend(self._get_data_points_from_sentence(s))

        if len(datapoints) == 0:
            return [], [], [], []

        span_hidden_states = torch.stack([self.aggregated_embedding(d, self.token_encoder.get_names()) for d in datapoints])

        if self.add_popularity == "concatenate":
            # TODO is this good? set to non trainable?
            span_hidden_states = self.token_dense(span_hidden_states)

        if self.training:
            if self.train_only_with_positive_labels:
                #labels = set()
                labels = []
                for d in datapoints:
                    #labels.add(d.get_label(self.label_type).value)
                    labels.append(d.get_label(self.label_type).value)
                #labels = list(labels)

                labels_ids = [self.label_dictionary.get_idx_for_item(l) for l in labels]

                if self.add_popularity == "verbalize":
                    popularities_verbalized = [self.sitelinks_dict[label]["sitelinks_verbalized"] for label in
                                               labels]

                if self.custom_label_verbalizations:
                    labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(labels)
                    if self.add_popularity == "verbalize":
                        labels_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                                  in zip(labels_verbalized, popularities_verbalized)]
                    else:
                        labels_verbalized = [Sentence(label) for label in labels_verbalized]
                else:
                    if self.add_popularity == "verbalize":
                        labels_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                                  in zip(labels, popularities_verbalized)]
                    else:
                        labels_verbalized = [Sentence(label) for label in labels]

                # if self.custom_label_verbalizations:
                #     labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(labels)
                #     labels_verbalized = [Sentence(label) for label in labels_verbalized]
                # else:
                #     labels_verbalized = [Sentence(label) for label in labels]
                #
                self.label_encoder.embed(labels_verbalized)
                label_hidden_states_batch = torch.stack([label.get_embedding() for label in labels_verbalized])

                if self.add_popularity == "concatenate":
                    popularities = torch.Tensor(
                        [self.sitelinks_dict[e]["sitelinks_normalized"] for e in labels]).to(flair.device)
                    label_hidden_states_batch = torch.cat((label_hidden_states_batch, popularities.unsqueeze(1)), dim=1)

                label_hidden_states = torch.zeros(len(self.label_dictionary), label_hidden_states_batch.shape[1],
                                                  device=flair.device)
                label_hidden_states[torch.LongTensor(list(labels_ids)), :] = label_hidden_states_batch

            else:
                label_hidden_states, _ = self._embed_labels_batchwise(max_limit=None)

        else:
            # if it's the first batch right after training, embed all and save!
            if self.is_first_batch_in_evaluation:
                #self.label_encoder.embed(labels_verbalized)
                #label_hidden_states = torch.stack([label.get_embedding() for label in labels_verbalized])
                label_hidden_states, _ = self._embed_labels_batchwise(cpu = True, max_limit=None)
                self.current_label_embeddings = label_hidden_states
                self.is_first_batch_in_evaluation = False

                # save them to enable inspection (and for one of the negative sampling methods):
                if self.training and self.label_embeddings_save_path:
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


    def _get_random_negative_labels(self, span_hidden_states, labels, factor, epoch_wise = False):
        import random
        rt_random_negatives_embeddings = []
        popularities = []
        for f in range(factor):
            random_idx = random.sample(range(len(self.label_dictionary)), len(labels))
            random_labels = [self.label_dict_list[idx] for idx in random_idx]
            if self.add_popularity == "concatenate":
                popularities.extend([self.sitelinks_dict[e]["sitelinks_normalized"] for e in random_labels])

            if epoch_wise:
                random_label_hidden_states = self.current_label_embeddings[random_idx]

            else:
                #if self.custom_label_verbalizations:
                #    random_labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(random_labels)
                #    random_labels_verbalized = [Sentence(label) for label in random_labels_verbalized]
                #
                #else:
                #    random_labels_verbalized = [Sentence(label) for label in random_labels]
                #print(f"Now embedding random labels:", len(random_labels_verbalized))

                if self.add_popularity == "verbalize":
                    popularities_verbalized = [self.sitelinks_dict[label]["sitelinks_verbalized"] for label in
                                               random_labels]

                if self.custom_label_verbalizations:
                    random_labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(random_labels)
                    if self.add_popularity == "verbalize":
                        random_labels_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                                  in zip(random_labels_verbalized, popularities_verbalized)]
                    else:
                        random_labels_verbalized = [Sentence(label) for label in random_labels_verbalized]
                else:
                    if self.add_popularity == "verbalize":
                        random_labels_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                                  in zip(random_labels, popularities_verbalized)]
                    else:
                        random_labels_verbalized = [Sentence(label) for label in random_labels]

                self.label_encoder.embed(random_labels_verbalized)
                random_label_hidden_states = torch.stack([label.get_embedding() for label in random_labels_verbalized])

            rt_random_negatives_embeddings.extend(random_label_hidden_states)

        rt_random_negatives_embeddings = torch.stack(rt_random_negatives_embeddings, dim=0).to(flair.device)

        if self.add_popularity == "concatenate":
            popularities = torch.Tensor(popularities)
            rt_random_negatives_embeddings = torch.cat((rt_random_negatives_embeddings, popularities.unsqueeze(1)), dim=1)

        return rt_random_negatives_embeddings

    def _get_hard_negative_labels(self, span_hidden_states, labels, factor,
                                  epoch_wise = True,
                                  ):

        with torch.no_grad():

            #TODO: Do I want to choose by similarity to the gold LABEL or by similarity to the SPAN embedding?
            # i.e. span_hidden_states or (see above) batch_labels_verbalized?

            # first: just for getting indices of similar labels: no gradient tracking / gpu necessary:
            if epoch_wise:
                all_labels_embeddings = self.current_label_embeddings
                all_labels_indices = range(len(self.current_label_embeddings))

            else:
                raise NotImplementedError
                # TODO embed and compare all labels

            batch_labels_indices = [self.label_dictionary.get_idx_for_item(l) for l in labels]

            #if self.custom_label_verbalizations:
            #    labels_batch_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(
            #        labels)
            #    labels_batch_verbalized = [Sentence(label) for label in labels_batch_verbalized]
            #else:
            #    labels_batch_verbalized = [Sentence(label) for label in labels]
            if self.add_popularity == "verbalize":
                popularities_verbalized = [self.sitelinks_dict[label]["sitelinks_verbalized"] for label in
                                           labels]

            if self.custom_label_verbalizations:
                labels_batch_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(labels)
                if self.add_popularity == "verbalize":
                    labels_batch_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                         in zip(labels_batch_verbalized, popularities_verbalized)]
                else:
                    labels_batch_verbalized = [Sentence(label) for label in labels_batch_verbalized]
            else:
                if self.add_popularity == "verbalize":
                    labels_batch_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                         in zip(labels, popularities_verbalized)]
                else:
                    labels_batch_verbalized = [Sentence(label) for label in labels]

            self.label_encoder.embed(labels_batch_verbalized)
            batch_labels_embeddings = torch.stack(
                [label.get_embedding() for label in labels_batch_verbalized])

            if self.add_popularity == "concatenate":
                popularities = torch.Tensor([self.sitelinks_dict[e]["sitelinks_normalized"] for e in labels]).to(flair.device)
                batch_labels_embeddings = torch.cat((batch_labels_embeddings, popularities.unsqueeze(1)), dim=1)


            #if self.add_popularity == "concatenate:
            #    all_labels_embeddings = all_labels_embeddings[:,:self.label_encoder.embedding_length] # don't use the popularity score for similarity

            if self.BCE_loss:

                # Compute cosine similarity
                #logits = torch.mm(batch_labels_embeddings.detach().cpu(),
                #                  all_labels_embeddings.t())
                #logits = torch.mm(F.normalize(span_hidden_states.detach().cpu(), p=2, dim=1),
                #                  F.normalize(all_labels_embeddings.detach().cpu().t(), p=2, dim=1))
                logits = torch.mm(span_hidden_states.detach().cpu(),
                                  all_labels_embeddings.detach().cpu().t())

                # exclude the real label
                for i, label_idx in enumerate(batch_labels_indices):
                    if label_idx in all_labels_indices:
                        logits[i, all_labels_indices.index(label_idx)] = float("-inf")

                _, hard_negative_indices_sample = torch.topk(logits, factor, dim=1)

            else:
                if self.add_popularity == "concatenate":
                    #batch_labels_embeddings = batch_labels_embeddings[:, :self.label_encoder.embedding_length]  # don't use the popularity score for similarity
                    #similarity_matrix = cosine_similarity(batch_labels_embeddings.detach().cpu(),
                    #                                     all_labels_embeddings.detach().cpu())

                    similarity_matrix = cosine_similarity(span_hidden_states.detach().cpu(),
                                                          all_labels_embeddings.detach().cpu())


                else:
                    # TODO see above
                    #similarity_matrix = cosine_similarity(batch_labels_embeddings.detach().cpu(),
                    #                                      all_labels_embeddings.detach().cpu())
                    similarity_matrix = cosine_similarity(span_hidden_states.detach().cpu(),
                                                          all_labels_embeddings.detach().cpu())

                # to exclude the real label
                for i, label_idx in enumerate(batch_labels_indices):
                    if label_idx in all_labels_indices:
                        similarity_matrix[i, all_labels_indices.index(label_idx)] = float("-inf")

                _, hard_negative_indices_sample = torch.topk(torch.tensor(similarity_matrix), factor, dim=1)

        # now embed those negative labels 'for real', so that gradient tracking possible and most current version
        # important: We need to "enroll" the labels the right way: each span should get its hard negative, spans are concatenated each factor!
        rt_hard_negatives_embeddings = []
        rt_hard_negatives_verbalized = [] # not used, but for debugging
        rt_hard_negatives_labels = []
        for f in range(factor):
            hard_negative_indices_f = hard_negative_indices_sample[:,f]
            labels_f = [self.label_dict_list[idx] for idx in hard_negative_indices_f]

            # if self.custom_label_verbalizations:
            #     hard_negatives_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(
            #                                     labels_f
            #                                     )
            # else:
            #     hard_negatives_verbalized = labels_f

            if self.add_popularity == "verbalize":
                popularities_verbalized = [self.sitelinks_dict[label]["sitelinks_verbalized"] for label in
                                           labels_f]

            if self.custom_label_verbalizations:
                hard_negatives_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(labels_f)
                if self.add_popularity == "verbalize":
                    hard_negatives_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                         in zip(hard_negatives_verbalized, popularities_verbalized)]
                else:
                    hard_negatives_verbalized = [Sentence(label) for label in hard_negatives_verbalized]
            else:
                if self.add_popularity == "verbalize":
                    hard_negatives_verbalized = [Sentence(label + f", {popularity}") for label, popularity
                                         in zip(labels_f, popularities_verbalized)]
                else:
                    hard_negatives_verbalized = [Sentence(label) for label in labels_f]

            rt_hard_negatives_verbalized.extend(hard_negatives_verbalized)
            self.label_encoder.embed(hard_negatives_verbalized)

            hard_negatives_embeddings = [ n.get_embedding() for n in hard_negatives_verbalized ]
            rt_hard_negatives_embeddings.extend(hard_negatives_embeddings)
            rt_hard_negatives_labels.extend(labels_f)

        rt_hard_negatives_embeddings = torch.stack(rt_hard_negatives_embeddings, dim=0).to(flair.device)

        if self.add_popularity == "concatenate":
            popularities = torch.Tensor([self.sitelinks_dict[e]["sitelinks_normalized"] for e in rt_hard_negatives_labels]).to(flair.device)
            rt_hard_negatives_embeddings = torch.cat((rt_hard_negatives_embeddings, popularities.unsqueeze(1)), dim=1)

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
                    some_random_label_hidden_states = self._get_random_negative_labels(span_hidden_states,
                                                                                       gold_labels,
                                                                                       self.negative_sampling_factor)
                    gold_labels_hidden_states_with_negatives = torch.cat((gold_labels_hidden_states,
                                                                          some_random_label_hidden_states), dim = 0)
                if self.negative_sampling_strategy == "hard_negatives":
                    hard_negatives_label_hidden_states = self._get_hard_negative_labels(span_hidden_states,
                                                                                        gold_labels,
                                                                                        self.negative_sampling_factor)
                    gold_labels_hidden_states_with_negatives = torch.cat((gold_labels_hidden_states,
                                                                          hard_negatives_label_hidden_states), dim = 0)

                gold_labels_hidden_states = gold_labels_hidden_states_with_negatives

                span_hidden_states_concat = span_hidden_states
                for i in range(self.negative_sampling_factor):
                    span_hidden_states_concat = torch.cat((span_hidden_states_concat, span_hidden_states), dim = 0)
                span_hidden_states = span_hidden_states_concat

                y = torch.cat((y, torch.zeros(nr_datapoints*self.negative_sampling_factor, device=flair.device)), dim = 0)


            if self.BCE_loss:
                logits = torch.mm(span_hidden_states, gold_labels_hidden_states.t())
                #logits = torch.mm(F.normalize(span_hidden_states, p=2, dim=1),
                #                  F.normalize(gold_labels_hidden_states.t(), p=2, dim=1))
                target = torch.zeros(span_hidden_states.shape[0], gold_labels_hidden_states.shape[0], device=flair.device)
                target[torch.arange(span_hidden_states.shape[0]), torch.arange(gold_labels_hidden_states.shape[0])] = y

                if self.weighted_loss:
                    real_gold_labels = gold_labels_hidden_states[:len(gold_labels)]
                    real_gold_labels_concat = torch.cat([real_gold_labels] * (1+self.negative_sampling_factor), dim=0)
                    weights = torch.Tensor(cosine_similarity(real_gold_labels_concat.detach().cpu(),
                                                             gold_labels_hidden_states.detach().cpu())
                                           ).to(flair.device)
                    target = weights

                    loss = self.loss_function(logits, target)
                else:
                    loss = self.loss_function(logits, target)

            else:
                y = torch.where(y == 0, -1, 1)
                if self.weighted_loss:
                    losses = self.loss_function(span_hidden_states, gold_labels_hidden_states, y)
                    ## A: using gold LABEL embeddings for weight calc:
                    real_gold_labels = gold_labels_hidden_states[:len(gold_labels)]
                    real_gold_labels_concat = torch.cat([real_gold_labels] * (1+self.negative_sampling_factor), dim=0)
                    weights = torch.Tensor(np.diagonal(cosine_similarity(real_gold_labels_concat.detach().cpu(),
                                                                         gold_labels_hidden_states.detach().cpu()))
                                           ).to(flair.device)

                    ## B: using SPAN embeddings for weight calc:
                    #weights = torch.Tensor(np.diagonal(cosine_similarity(span_hidden_states.detach().cpu(),
                    #                                                    gold_labels_hidden_states.detach().cpu()))
                    #                      ).to(flair.device)

                    loss = torch.sum(losses * weights)
                else:
                    loss = self.loss_function(span_hidden_states, gold_labels_hidden_states, y).unsqueeze(0)

        else:

            if self.BCE_loss:

                logits = torch.mm(span_hidden_states, label_hidden_states.t())
                #logits = torch.mm(F.normalize(span_hidden_states, p=2, dim=1),
                #                  F.normalize(label_hidden_states.t(), p=2, dim=1))

                target = torch.zeros(span_hidden_states.shape[0], label_hidden_states.shape[0],
                                     device=flair.device if self.training else "cpu")

                target[torch.arange(span_hidden_states.shape[0]), gold_labels_idx] = 1

                loss = self.loss_function(logits, target)


            else:
                losses = []
                for i, sp in enumerate(datapoints):
                    y = torch.zeros(label_hidden_states.shape[0], device=flair.device if self.training else "cpu")
                    y[gold_labels_idx[i]] = 1.0

                    # stack the span embedding, so that we can push all OTHER labels away from it
                    span_repeated = span_hidden_states[i].repeat(label_hidden_states.shape[0],1)

                    y = torch.where(y == 0, -1, 1)

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

    def _mask_scores(self, scores: torch.Tensor, data_points: List[Span], label_order):
        if not self.candidates:
            return scores

        masked_scores = -torch.inf * torch.ones(scores.size(), requires_grad=True, device=flair.device)

        for idx, span in enumerate(data_points):
            # get the candidates
            candidate_set = self.candidates.get_candidates(span.text)
            # during training, add the gold value as candidate
            if self.training:
                candidate_set.add(span.get_label(self.label_type).value)
            #candidate_set.add("<unk>")
            indices_of_candidates = [label_order.index(candidate) for candidate in candidate_set]
            masked_scores[idx, indices_of_candidates] = scores[idx, indices_of_candidates]

        return masked_scores


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

                if self.candidates:
                    # we need to: a) add the candidates that are not yet in there, b) mask the non-candidates for each span
                    new_labels = set()
                    for d in datapoints:
                        candidate_set = self.candidates.get_candidates(d.text)
                        new_labels.update({c for c in candidate_set if c not in self.label_dict_list})
                    new_labels = list(new_labels)

                    if self.custom_label_verbalizations:
                        new_labels_verbalized = self.custom_label_verbalizations.verbalize_list_of_labels(new_labels)
                        new_labels_verbalized = [Sentence(label) for label in new_labels_verbalized]
                    else:
                        new_labels_verbalized = [Sentence(label) for label in new_labels]

                    self.label_encoder.embed(new_labels_verbalized)
                    new_labels_hidden_states_batch = torch.stack([label.get_embedding() for label in new_labels_verbalized]).detach().cpu()

                    label_hidden_states = torch.cat((label_hidden_states, new_labels_hidden_states_batch), dim = 0)

                    label_ordering = self.label_dict_list + new_labels

                if self.BCE_loss:
                    span_hidden_states = span_hidden_states.detach().cpu()
                    label_hidden_states = label_hidden_states.detach().cpu()

                    # Compute cosine similarity
                    #logits = torch.mm(F.normalize(span_hidden_states, p=2, dim=1),
                    #                  F.normalize(label_hidden_states.t(), p=2, dim=1))

                    logits = torch.mm(span_hidden_states, label_hidden_states.t())
                    logits_sigmoided = torch.sigmoid(logits)

                    top_indices = torch.topk(logits_sigmoided, k=5, dim=1)[1]
                    top_labels = [[self.label_dict_list[index] for index in indices] for indices in top_indices]
                    # take the highest one
                    final_label_indices = top_indices[:, 0]

                else:

                    span_hidden_states = span_hidden_states.detach().cpu()
                    label_hidden_states = label_hidden_states.detach().cpu()

                    # version 1 (so far used)
                    #similarity = torch.mm(span_hidden_states, label_hidden_states.t())

                    # version 2:
                    similarity = torch.tensor(cosine_similarity(span_hidden_states, label_hidden_states))

                    # Get the label indices with maximum similarity for each span
                    _, max_label_indices = torch.max(similarity, dim=1)

                    # just for inspection: get the top N (=5) most probable labels:
                    top_indices = torch.topk(similarity, k=5, dim=1)[1]
                    top_labels = [[self.label_dict_list[index] for index in indices] for indices in top_indices]
                    # take the highest one
                    final_label_indices = top_indices[:, 0]


                #print(final_label_indices)

                for i,d in enumerate(datapoints):
                    if self.BCE_loss:
                        #label_idx = final_label_indices[i][0]
                        #conf = final_label_indices[i][1]
                        label_idx = final_label_indices[i].item()
                        conf = logits_sigmoided[i, label_idx].item()
                    else:
                        label_idx = final_label_indices[i]
                        conf = similarity[i, label_idx]

                    #if conf >= self.threshold_in_prediction:
                    if self.candidates:
                        label = label_ordering[label_idx]
                    else:
                        label = self.label_dict_list[label_idx]
                    d.set_label(label_name,
                                value=label,
                                score=conf
                                )
                    d.set_label(typename = "top_5", value = "|".join(top_labels[i]))

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
                    f' --> {span.get_label("predicted").value} ({symbol}) top_5: {span.get_label("top_5").value}\n'
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
            "BCE_loss": self.BCE_loss,
            "margin_loss": self.margin_loss,
            "threshold_in_prediction": self.threshold_in_prediction,
            "label_embeddings_save_path": self.label_embeddings_save_path,
            "is_first_batch_in_evaluation": self.is_first_batch_in_evaluation,
            "weighted_loss": self.weighted_loss,
            "add_popularity": self.add_popularity,
            "popularity_save_path": self.popularity_save_path,
            #"candidates": self.candidates
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
            threshold_in_prediction = state.get("threshold_in_prediction"),
            BCE_loss = state.get("BCE_loss"),
            label_embeddings_save_path = state.get("label_embeddings_save_path"),
            weighted_loss = state.get("weighted_loss"),
            add_popularity = state.get("add_popularity"),
            popularity_save_path = state.get("popularity_save_path"),
            #candidates = state.get("candidates")
            **kwargs,
        )

