import json
import sys
import os
import logging
import random
from math import ceil, floor
import time

from tqdm import tqdm
#from tqdm.auto import tqdm
from typing import Tuple, Dict, List, Callable, Literal
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import gc

import flair
from flair.data import DT, Dictionary, Optional, Sentence, Span, Union, Token
from flair.embeddings import DocumentEmbeddings, TokenEmbeddings

from openai import OpenAI
from google import genai
from google.genai import types
from gradio_client import Client as GradioClient

from multiprocessing import Process
import threading

import pickle
import re
from collections import defaultdict

from vllm import LLM, SamplingParams

log = logging.getLogger("flair")
# logging.getLogger("openai").setLevel(logging.WARNING)
# logging.getLogger("httpx").setLevel(logging.WARNING)
logging.disable(logging.INFO)


class SimilarityMetric:
    def __init__(self, metric_to_use):
        self.metric_to_use = metric_to_use

    #def __call__(self, tensor_a, tensor_b):
    #    return self.distance(tensor_a, tensor_b)

    def distance(self, tensor_a, tensor_b):
        sim = self.similarity(tensor_a, tensor_b)
        if self.metric_to_use == "cosine":
            return 1-sim
        else:
            return -sim

    def similarity(self, tensor_a, tensor_b):

        def chunked_cdist(small_tensor, big_tensor, chunk_size=500000):
            results = []
            small_len = small_tensor.size(0)
            big_len = big_tensor.size(0)
            # only process chunk_size entries at once
            # chunk_size = a_len * b_chunk_size
            # and b_chunk_size
            chunk_size = ceil(chunk_size / small_len)
            for j in range(0, big_len, chunk_size):
                chunk_b = big_tensor[j:j + chunk_size]
                results.append(torch.cdist(small_tensor, chunk_b, compute_mode = "donot_use_mm_for_euclid_dist"))
            return torch.cat(results, dim=1)

        if self.metric_to_use == "euclidean":
            # if we do not use compute_mode = "donot_use_mm_for_euclid_dist", numerical deviations are very high on gpu, see https://github.com/pytorch/pytorch/issues/42479 and https://github.com/pytorch/pytorch/issues/57690
            #return -torch.cdist(tensor_a, tensor_b, compute_mode = "donot_use_mm_for_euclid_dist")
            return -chunked_cdist(tensor_a, tensor_b)

        elif self.metric_to_use == "cosine":
            tensor_a_normalized = F.normalize(tensor_a, p=2, dim=-1)
            tensor_b_normalized = F.normalize(tensor_b, p=2, dim=-1)

            #if tensor_b_normalized.dim() == 2:
            return torch.matmul(tensor_a_normalized, tensor_b_normalized.transpose(-2,-1))
            #elif tensor_b_normalized.dim() == 3:
            #    return torch.matmul(tensor_a_normalized, tensor_b_normalized.transpose(1,2))

        elif self.metric_to_use == "mm":
            #if tensor_b.dim() == 2:
            #    return torch.mm(tensor_a, tensor_b.t())
            #elif tensor_b.dim() == 3:
            return torch.matmul(tensor_a, tensor_b.transpose(-2, -1))

        else:
            raise ValueError(f"Unsupported metric to use: {self.metric_to_use}")


# class DEEDTripletMarginLoss(torch.nn.TripletMarginLoss):
#     def __init__(self, similarity_metric: SimilarityMetric = SimilarityMetric("euclidean"), **kwargs):
#     #    kwargs["reduction"] = "none"
#         super(DEEDTripletMarginLoss, self).__init__(**kwargs)
#         self.similarity_metric = similarity_metric
#
#     def forward(self, anchor, positive, negative):
#
#         positive_negative = torch.cat([positive.unsqueeze(0), negative]).unsqueeze(2)
#         similarities = self.similarity_metric.similarity(anchor.unsqueeze(1), positive_negative).squeeze()
#         pos_cosine_sim, neg_cosine_sim = similarities[0].unsqueeze(0), similarities[1::]
#         losses = F.relu(neg_cosine_sim - pos_cosine_sim + self.margin)
#         return losses.mean()

class DEEDTripletMarginLoss(torch.nn.TripletMarginWithDistanceLoss):
    def __init__(self, similarity_metric: SimilarityMetric = SimilarityMetric("euclidean"), **kwargs):
        kwargs["reduction"] = "none"
        self.margin_step = 0.45
        self.margin_adjustment_frequency = 500
        super(DEEDTripletMarginLoss, self).__init__(distance_function=similarity_metric.distance, **kwargs)

    def forward(self, anchor, positive, negative):
        loss = super(DEEDTripletMarginLoss, self).forward(anchor.unsqueeze(1), positive.unsqueeze(1), negative.transpose(0,1))
        return loss.mean()

    def adjust_margin(self, increase = False):
        if increase:
            new_margin = self.margin + self.margin_step
            self.margin = min(new_margin, 10.0)  # Ensure margin does not go above
        else:
            new_margin = self.margin - self.margin_step
            self.margin = max(new_margin, 0.5)  # Ensure margin does not go below

        print("Adjusted margin to:", self.margin)


class DEEDEuclideanEmbeddingLoss(torch.nn.Module):
    def __init__(self, mode: str = "margin",
                       margin: float =5.0,
                       similarity_metric: SimilarityMetric = SimilarityMetric("euclidean")):
        """
        Similar to pytorch's CosineEmbeddingLoss (https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html) with negatives, but using euclidean distance.
        :param margin: Margin to push the negatives away from the anchor.
        :param mode: Using the margin as fixed ('margin') or considering margin from positive ('using_positive')
        """
        super().__init__()
        self.mode = mode
        self.margin = margin
        self.similarity_metric = similarity_metric # todo use this!
        if self.similarity_metric.metric_to_use != "euclidean":
            raise NotImplementedError

    def forward(self, anchor, positive,  negative):
        # handle positives
        # euclidean distance between anchor and positive embeddings
        positive_loss = torch.nn.functional.pairwise_distance(anchor, positive) # same as above

        # handle negatives
        # calculate the euclidean distance between the anchor and each batch of negatives
        dist = torch.nn.functional.pairwise_distance(anchor, negative)
        # loss is distance after margin is applied

        # a) using fixed margin:
        if self.mode == "margin":
            negative_loss = torch.max(torch.tensor(0.0), self.margin - dist)

        # b) using positive loss (--> similar to triplet loss, but with positive loss included)
        elif self.mode == "using_positive":
            negative_loss = torch.max(torch.tensor(0.0), positive_loss - dist) # no margin: negative must just be further than positive
            #negative_loss = torch.max(torch.tensor(0.0), positive_loss - dist + self.margin) # negative must be >= margin from positive
        else:
            raise ValueError

        # take mean over both losses
        # todo If negatives factor > 1, we weigh the negative losses more. Do we want that?
        # could instead do for example:
        # positive_loss = positive_loss.expand(negative_loss.shape)
        losses = torch.cat([positive_loss.unsqueeze(0), negative_loss])
        return torch.mean(losses)


class DEEDCrossEntropyLoss(torch.nn.CrossEntropyLoss):

    def __init__(self, similarity_metric: SimilarityMetric = SimilarityMetric("euclidean")):
        super().__init__()
        self.similarity_metric = similarity_metric

    def forward(self, anchor, positive, negative):
        #factor = negative.shape[0]

        ## version a) using all negatives as negatives for all spans
        #positive_negative = torch.cat([positive.unsqueeze(1), negative.permute(1,0,2)], dim=0).squeeze(1)
        #similarities = -torch.cdist(anchor, positive_negative)
        #similarities = self.similarity_metric.similarity(anchor.unsqueeze(1), positive_negative).squeeze(1)
        #target = torch.tensor(range(anchor.shape[0])).to(flair.device)

        ## version b) using only the correct negatives per span:
        positive_negative = torch.cat([positive.unsqueeze(1), negative.permute(1,0,2)], dim=1)
        #similarities = -torch.cdist(anchor.unsqueeze(1), positive_negative).squeeze(1)
        similarities = self.similarity_metric.similarity(anchor.unsqueeze(1), positive_negative).squeeze(1)
        target = torch.zeros(anchor.shape[0], dtype=torch.int64).to(flair.device)

        loss = super(DEEDCrossEntropyLoss, self).forward(similarities, target)

        return loss


class LabelList:
    def __init__(self):
        self._items = []
        self._item2idx: Dict[str, int] = {}
        self._item2sentence: Dict[str, flair.data.Sentence] = {}

    @property
    def items(self):
        return self._items.copy()

    def add(self, items: List[str]):
        self._items.extend(items)
        for item in items:
            if item in self._item2idx:
                raise ValueError("Duplicate Item found, not supported right now.")
            self._item2idx[item] = len(self._item2idx)

    def index_for(self, item: str):
        return self._item2idx.get(item, None)

    def sentence_object_for(self, item: str):
        return self._item2sentence.get(item, None)

    def add_sentence_object_for(self, item: str, sentence_object: flair.data.Sentence):
        self._item2sentence[item] = sentence_object

class DualEncoderEntityDisambiguation(flair.nn.Classifier[Sentence]):

    def __init__(self, token_encoder: TokenEmbeddings,
                 label_encoder: Union[DocumentEmbeddings, TokenEmbeddings],
                 known_labels: List[str], gold_labels: List[str] = [],
                 label_type: str = "nel", label_map: dict = {},
                 embedding_pooling: Literal["first", "last", "mean", "first_last"] = "mean",
                 negative_sampling_strategy: Literal["shift", "random", "hard", "hard_random"] = "hard", negative_sampling_factor: Union[int, str] = 1,
                 loss_function_name: Literal["triplet", "binary_embedding", "cross_entropy"] = "triplet",
                 similarity_metric_name: Literal ["euclidean", "cosine", "mm"] = "euclidean", constant_updating: bool = True,
                 label_embedding_batch_size: int = 128, label_embeddings_storage_device: torch.device = None, *args, **kwargs):
        """
        This model uses a dual encoder architecture where both inputs and labels (verbalized) are encoded with separate
        Transformers. It uses some kind of similarity loss to push datapoints and true labels nearer together while pushing negatives away
        and performs KNN like inference.
        More descriptive label verbalizations can be plugged in.
        :param token_encoder: Token embeddings to embed the spans in a sentence.
        :param label_encoder: Document embeddings to embed the label verbalizations.
        :param known_labels: List of all labels that the model can use, in addition to the gold labels.
        :param gold_labels: List of corpus specific gold labels that should be used during predictions.
        :param embedding_pooling: Pooling of both mention and label embeddings.
        :param label_type: Label type to predict (e.g. "nel").
        :param label_map: Mapping of label values to more descriptive verbalizations, used for embedding the labels.
        :param negative_sampling_strategy: Strategy to search for negative samples. Must be one of "hard", "shift", "random", "hard_random".
        :param negative_sampling_factor: Number of negatives per positive, e.g. 1 (one negative sample per positive), 2 (two negative samples per positive).
        :param loss_function_name: Loss funtion to use, must be one of "triplet", "binary_embedding", "cross_entropy".
        :param similarity_metric_name: Similarity metric to use, must be one of "euclidean", "cosine", "mm".
        :param constant_updating: Updating the label embeddings for every embedded label (positive and negative).
        :param label_embedding_batch_size: Batch size to use for embedding labels to avoid memory overflow.
        :param label_embeddings_storage_device: Device to store the sampled label embeddings on. If None, uses flair.device
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.token_encoder = token_encoder
        self.label_encoder = label_encoder
        self._label_type = label_type
        self.label_map = label_map
        self.known_labels = known_labels
        self.gold_labels = gold_labels
        self.embedding_pooling = embedding_pooling
        if isinstance(self.label_encoder, DocumentEmbeddings):
            if self.embedding_pooling != "mean" and self.label_encoder.cls_pooling == "mean":
                raise Warning("Pooling method is not congruent.")
            if self.embedding_pooling != "first" and self.label_encoder.cls_pooling == "first":
                raise Warning("Pooling method is not congruent.")
            if self.embedding_pooling == "first_last":
                raise Warning("Pooling method is not congruent.")

        self._label_embeddings = None
        self._next_prediction_needs_updated_label_embeddings = False
        self._label_embedding_batch_size = label_embedding_batch_size
        if not label_embeddings_storage_device:
            label_embeddings_storage_device = flair.device
        self._label_embeddings_storage_device = label_embeddings_storage_device
        if similarity_metric_name in ["euclidean", "cosine", "mm"]:
            self.similarity_metric = SimilarityMetric(metric_to_use = similarity_metric_name)
        else:
            raise ValueError(f"Similarity metric {similarity_metric_name} not recognized.")

        if loss_function_name == "triplet":
            self.loss_function = DEEDTripletMarginLoss(similarity_metric= self.similarity_metric, margin = 0.5 if similarity_metric_name == "cosine" else 3.0)
        elif loss_function_name == "binary_embedding":
            self.loss_function = DEEDEuclideanEmbeddingLoss(similarity_metric= self.similarity_metric)
        elif loss_function_name == "cross_entropy":
            self.loss_function = DEEDCrossEntropyLoss(similarity_metric= self.similarity_metric)
        else:
            raise ValueError(f"Loss {loss_function_name} not recognized.")

        self.constant_updating = constant_updating
        self.negative_sampling_strategy = negative_sampling_strategy
        if negative_sampling_strategy == "shift":
            self._negative_sampling_fn = self._negative_sampling_shift
        elif negative_sampling_strategy == "random":
            self._negative_sampling_fn = self._negative_sampling_random_over_all
        elif negative_sampling_strategy == "hard":
            self._negative_sampling_fn = self._negative_sampling_hard
        elif negative_sampling_strategy == "hard_random":
            self._negative_sampling_fn = self._negative_sampling_hard_and_random
        else:
            raise ValueError(f"Negative Sampling Strategy {negative_sampling_strategy} not supported.")
        self._negative_sampling_factor = negative_sampling_factor
        self._iteration_count = 0
        self._seen_spans = 0
        self._last_update = 0

        self._label_dict = None

        self._INDEX_NOT_FOUND = torch.tensor(-1, device=flair.device, dtype=torch.int64)

        self.to(flair.device)


    def _label_at(self, idx: int):
        """ Label at index in label_dict """
        return self._label_dict.items[idx]

    def _idx_for_label(self, label: str):
        """ Index of label in label_dict.items """
        idx = self._label_dict.index_for(label)
        if idx is None:
             return self._INDEX_NOT_FOUND # torch.tensor(-1, device=flair.device)
        return idx

    def _update_some_label_embeddings(self, labels, new_label_embeddings):
        if self._label_embeddings is None:
            # using this just to make sure self._label_dict and self._label_embeddings get created if not yet there
            _ = self.get_label_embeddings()

        with torch.no_grad():
            indices = [self._idx_for_label(label) for label in labels]
            invalid_items = [i for i, id in enumerate(indices) if id is None]
            valid_indices = [id for id in indices if id is not None]

            if len(invalid_items) !=0:
                mask = torch.ones(new_label_embeddings.size(0), dtype=torch.bool)
                for i in invalid_items:
                    mask[i] = False
                new_label_embeddings = new_label_embeddings[mask]

            if len(valid_indices) !=0:
                valid_indices = torch.tensor(valid_indices)
                self._label_embeddings[valid_indices] = new_label_embeddings

    @property
    def label_type(self):
        return self._label_type

    def update_labels(self, known: List[str], gold: List[str]):
        """
        Giving the model a new set on known or gold labels. E.g. when predicting on a new corpus.
        :param known: List of all labels the model should be aware of. Same as known_labels in init.
        :param gold: List of gold labels. Same as gold_labels in init.
        """
        self.known_labels = known
        self.gold_labels = gold
        self._label_dict = None
        self._create_label_dict()
        self._recompute_label_embeddings()

    def _create_label_dict(self):
        """
        Creates self._label_dict and sets self._label_embeddings == None (so they will be embedded)
        """
        if not self._label_dict:
            labels = self.gold_labels + self.known_labels
            labels = list(set(labels))
            print(f"Using a total of {len(labels)} labels ({len(self.gold_labels)} gold labels).")
            print("Need label embedding update")
            self._label_dict = LabelList()
            self._label_dict.add(labels)
        else:
            print("Already existing label dict.")

    def _recompute_label_embeddings(self):
        if not self._label_dict:
            self._create_label_dict()

        self._label_embeddings = None # delete the old ones, for memory reasons
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            print(f"After iteration {self._iteration_count} / {self._seen_spans} spans, updating label embeddings...")
            self._label_embeddings = self._embed_labels_batchwise_return_stacked_embeddings(labels = [l for l in self._label_dict.items],
                                                                                            update_these_embeddings = False,
                                                                                            device=self._label_embeddings_storage_device,
                                                                                            )
            self._last_update = self._seen_spans

    # Function to split sentences intelligently
    def _split_sentence(self, sentence,
                        max_characters: int,
                        max_spans_per_sentence: int,
                        respect_full_stops: bool):

        # Tokenize the sentence
        num_characters = len(sentence.text)
        num_spans = len(sentence.get_spans(self.label_type))

        # If the sentence is short enough and has not too many spans, return it as is
        if num_characters <= max_characters and num_spans <= max_spans_per_sentence:
            return [sentence]

        # Otherwise: split the sentence
        split_at = sentence.tokens[-1].start_position # start somewhere (when not length but span number is problem important)
        for t in sentence.tokens:
            if t.end_position >= max_characters:
                split_at = t.idx-1
                break

        # make sure that not more than max_spans_per_sentence is in there:
        span_counter = 0
        for tmp in sentence.get_spans(self.label_type):
            if (tmp[0].idx-1) < split_at and span_counter == max_spans_per_sentence:
                split_at = tmp[0].idx -2
                break
            if (tmp[0].idx-1) > split_at:
                break
            span_counter +=1

        if respect_full_stops:
           # Move to the nearest " ." before (as rule for more sentence-like splitting)
            period_indices = [i for i,t in enumerate(sentence.tokens[:split_at]) if t.text == "." ]

            if len(period_indices) >0:
                last_period_index = period_indices[-1]
                if split_at - last_period_index <= 50: # if close enough, use it
                    split_at = last_period_index +1

        # But make sure it is not split inside a span
        for tmp in reversed(sentence.get_spans(self.label_type)):
            if (tmp[0].idx-1) < split_at and tmp[-1].idx > split_at:
                split_at = tmp[0].idx -2

        first_half_tokens = sentence.tokens[:split_at]
        second_half_tokens = sentence.tokens[split_at:]

        # Create the first and second sentence
        first_half_sentence = Sentence([t.text for t in first_half_tokens])
        second_half_sentence = Sentence([t.text for t in second_half_tokens])

        # Adjust spans (annotations)
        for span in sentence.get_spans(self.label_type):
            start_token = span[0].idx-1
            end_token = span[-1].idx
            if end_token <= split_at:
                new_sp = Span(first_half_sentence.tokens[start_token:end_token])
                for k, labels in span.annotation_layers.items():
                    for l in labels:
                        new_sp.set_label(typename=k, value=l.value, score=l.score)
            elif start_token >= split_at:
                # Adjust indices for the second half
                new_start_token = start_token - len(first_half_tokens)
                new_end_token = end_token - len(first_half_tokens)
                new_tokens = second_half_sentence.tokens[new_start_token:new_end_token]
                new_sp = Span(new_tokens)
                for k, labels in span.annotation_layers.items():
                    for l in labels:
                        new_sp.set_label(typename=k, value=l.value, score=l.score)
            else:
                # Should not happen that spans that are split across but check here
                print("Split span problem encountered")

        # Add the first half as context to the second half and vice versa
        first_half_sentence._next_sentence = second_half_sentence
        second_half_sentence._previous_sentence = first_half_sentence
        # The second sentence could still be too long, so repeat
        rest_sentences = self._split_sentence(second_half_sentence, max_characters, max_spans_per_sentence, respect_full_stops)
        rest_sentences.insert(0, first_half_sentence)
        return rest_sentences


    def _custom_batching(self, sentences,
                         batch_size = None):

        batched_sentences, _ = self._prepare_sentences(sentences,
                                                       batch_size=batch_size)
        return batched_sentences

    def _prepare_sentences(self, sentences: List[Sentence],
                           max_characters_sentence = 2800,
                           max_spans_per_sentence = 100, #50, #100, #75,
                           max_spans_per_batch = 150, #100, #150,
                           max_characters_per_batch_with_context: Union[int, None] = 8000,
                           respect_full_stops = True,
                           batch_size: Union[int, None] = None,
                           batch_size_max: int = 128
                           ):

        """
        Prepares the sentences. In case some are too long, they get split up. Also, only the ones that have spans in them are kept.
        The original spans are returned (mainly for use during prediction).
        :param sentences: List of sentences to be embedded.
        :param max_characters_sentence: Maximum sentence length in characters. Sentences are split accordingly.
        :param max_spans_per_sentence: Maximum number of spans per sentence. Sentences are split accordingly.
        :param max_spans_per_batch: Maximum spans allowed to be in one batch. Sentences are split accordingly.
        :param max_characters_per_batch_with_context: Maximum characters allowed in one batch (counting context!). Sentences are split accordingly.
        :param batch_size: How many sentences are put together at max in a mini batch (if batch_size is given). If None, means no batching, all sentences as one batch.
        :return: Tupel: List of lists of sentences, list of lists of original span objects per batch (important for prediction).
        """
        # keep original span objects, not just the spans from the possibly split sentences (necessary for prediction!)
        original_spans = []
        for s in sentences:
            original_spans.extend(s.get_spans(self.label_type))

        split_sentences = []
        for s in sentences:
            if len(s.text) > max_characters_sentence or len(s.get_spans(self.label_type)) > max_spans_per_sentence:
                split_sentences.extend(self._split_sentence(s,
                                                            max_characters=max_characters_sentence,
                                                            max_spans_per_sentence=max_spans_per_sentence,
                                                            respect_full_stops=respect_full_stops))
            else:
                split_sentences.append(s)

        span_counter = 0
        token_counter = 0
        sentences_to_embed = []
        for s in split_sentences:
            spans = s.get_spans(self.label_type)
            if len(spans) > 0:
                span_counter += len(spans)
                token_counter += len(s)
                sentences_to_embed.append(s)

        if not batch_size:
            batch_size = batch_size_max # dummy batch size for loop below

        batched_sentences = []
        batched_original_spans = []
        current_batch_spans = []
        current_batch = []
        current_spans = 0
        spans_index = 0
        current_characters = 0
        for sentence in sentences_to_embed:
            num_spans = len(sentence.get_spans(self.label_type))

            if len(current_batch) >= batch_size or current_spans + num_spans > max_spans_per_batch or current_characters > max_characters_per_batch_with_context:
                batched_sentences.append(current_batch)
                batched_original_spans.append(current_batch_spans)
                current_batch_spans = []
                current_batch = []
                current_spans = 0
                current_characters = 0

            current_batch.append(sentence)
            current_batch_spans.extend(original_spans[spans_index:spans_index+num_spans])
            spans_index += num_spans
            current_spans += num_spans
            sentence_with_context, _ = self.token_encoder._expand_sentence_with_context(sentence)
            current_characters += sum([len(t.text) for t in sentence_with_context])

        if current_batch:
            batched_sentences.append(current_batch)
            batched_original_spans.append(current_batch_spans)

        return batched_sentences, batched_original_spans

    # def _add_special_tokens_around_spans(self, sentence: Sentence,
    #                                      #start_token = "[S]", end_token = "[E]"
    #                                      #start_token = "[", end_token = "]"
    #                                      start_token = "'", end_token = "'"
    #
    #     ):
    #
    #     start_verbalization = Sentence(start_token)
    #     end_verbalization = Sentence(end_token)
    #
    #     start_verbalization_token_text = [t.text for t in start_verbalization.tokens]
    #     end_verbalization_token_text = [t.text for t in end_verbalization.tokens]
    #
    #     len_start_verbalization_tokens = len(start_verbalization_token_text)
    #     len_end_verbalization_tokens = len(end_verbalization_token_text)
    #
    #     len_verbalization_tokens = len_start_verbalization_tokens + len_end_verbalization_tokens
    #
    #     spans = sentence.get_spans(self.label_type)
    #
    #     tokens_text = [t.text for t in sentence.tokens]
    #     added_tokens = 0
    #     token_indices = [[sp.tokens[0].idx, sp.tokens[-1].idx] for sp in spans]
    #
    #     for i, sp in enumerate(sentence.get_spans(self.label_type)):
    #
    #         add_start_at_position_in_tokens = sp.tokens[0].idx-1 + added_tokens
    #         add_end_at_position_in_tokens = sp.tokens[-1].idx + added_tokens
    #
    #         tokens_text = tokens_text[:add_start_at_position_in_tokens] \
    #                       + start_verbalization_token_text + tokens_text[add_start_at_position_in_tokens:add_end_at_position_in_tokens] + end_verbalization_token_text \
    #                       + tokens_text[add_end_at_position_in_tokens:]
    #
    #         added_tokens += len_verbalization_tokens
    #
    #         for j, d in enumerate(spans):
    #             s_start_idx = token_indices[j][0]
    #             s_end_idx = token_indices[j][1]
    #
    #             if s_start_idx > add_end_at_position_in_tokens:
    #                 s_start_idx += len_verbalization_tokens
    #                 s_end_idx += len_verbalization_tokens
    #                 token_indices[j][0] = s_start_idx
    #                 token_indices[j][1] = s_end_idx
    #
    #     # Cannot use new_sentence = Sentence(text), we needed to use tokens instead of working with the text directly because of the weird problem with the space in unicode, e.g. 0x200c
    #     new_sentence = Sentence(tokens_text)
    #
    #     for i, sp in enumerate(spans):
    #         start_token_index, end_token_index = token_indices[i]
    #         new_sp = Span(new_sentence.tokens[start_token_index -1 + len_start_verbalization_tokens:end_token_index+len_start_verbalization_tokens])
    #
    #         for k, labels in sp.annotation_layers.items():
    #             for l in labels:
    #                 new_sp.set_label(typename=k, value=l.value, score=l.score)
    #
    #     new_sentence._previous_sentence = sentence._previous_sentence
    #
    #     new_sentence._next_sentence = sentence._next_sentence
    #
    #     return new_sentence



    def _embed_spans(self, sentences: List[Sentence], clear_embeddings = True):
        """
        Embed sentences and get embeddings for their spans.
        :param sentences:
        :return:
        """

        original_spans = []
        for s in sentences:
            sentence_spans = s.get_spans(self.label_type)
            if sentence_spans:
                original_spans.extend(sentence_spans)

        if not original_spans:
            return None, None

        spans = []
        for s in sentences:
            sentence_spans = s.get_spans(self.label_type)
            if sentence_spans:
                spans.extend(sentence_spans)

        self.token_encoder.embed(sentences)

        if self.embedding_pooling == "first":
            span_embeddings = [span[0].get_embedding() for span in spans]
        if self.embedding_pooling == "last":
            span_embeddings = [span[-1].get_embedding() for span in spans]
        if self.embedding_pooling == "mean":
            span_embeddings = [torch.mean(torch.stack([token.get_embedding() for token in span], 0), 0) for span in spans]
        if self.embedding_pooling == "first_last":
            span_embeddings = [torch.cat([span[0].get_embedding(), span[-1].get_embedding()]) for span in spans]

        if clear_embeddings:
            for s in sentences:
                s.clear_embeddings()

        return original_spans, torch.stack(span_embeddings, dim=0)


    def _embed_labels_batchwise_return_stacked_embeddings(self, labels: List[str], clear_embeddings: bool = True, update_these_embeddings: bool = True,
                                                          use_tqdm: bool = True, device: torch.device = None, fixed_batch_size: int = None, step: int = 128, split_title_verbalization: bool = False,
                                                          ):


        if not fixed_batch_size:
            initial_batch_size = self._label_embedding_batch_size
            batch_size = initial_batch_size

        else:
            batch_size = fixed_batch_size

        unique_labels, inverse_indices = np.unique(labels, return_inverse=True)

        labels_sentence_objects = self.get_sentence_objects_for_labels(unique_labels, use_tqdm = use_tqdm)

        final_embeddings = []

        if use_tqdm:
            pbar = tqdm(total=len(labels_sentence_objects), position=0, leave=True, dynamic_ncols=False)

        i = 0

        while True:
            try:
                batch = labels_sentence_objects[i:i + batch_size]

                self.label_encoder.embed(batch)

                if isinstance(self.label_encoder, DocumentEmbeddings):
                    embeddings = [l.get_embedding() for l in batch]
                elif isinstance(self.label_encoder, TokenEmbeddings):
                    if self.embedding_pooling == "first_last":
                        #embeddings = [torch.cat([l[0].get_embedding(), l[-1].get_embedding()], 0) for l in batch] # using the whole verbalization as span
                        embeddings = [torch.cat([l[0].get_embedding(), l[int(l.get_label("last title token").value)].get_embedding()], 0) for l in batch] # using only the label title as span
                    if self.embedding_pooling == "first":
                        embeddings = [l[0].get_embedding() for l in batch]
                    if self.embedding_pooling == "mean":
                        #embeddings = [torch.mean(torch.stack([token.get_embedding() for token in l.tokens], 0), 0) for l in batch ] # using the whole verbalization as span
                        embeddings = [torch.mean(torch.stack([token.get_embedding() for token in l.tokens[:int(l.get_label("last title token").value)+1]], 0), 0) for l in batch]  # using only the label title as span
                    if split_title_verbalization:
                        title_embeddings = [torch.mean(torch.stack([token.get_embedding() for token in l.tokens[:int(l.get_label("last title token").value)+1]], 0), 0) for l in batch]  # using only the label title as span
                        verbalization_embeddings = [torch.mean(torch.stack([token.get_embedding() for token in l.tokens[int(l.get_label("last title token").value+2):]], 0), 0) if len(l) > int(l.get_label("last title token").value+1)
                                                    else torch.mean(torch.stack([token.get_embedding() for token in l.tokens[:int(l.get_label("last title token").value)+1]], 0), 0)
                                                    for l in batch ]  # using only the label title as span
                        embeddings = [torch.cat([title_embeddings[i], verbalization_embeddings[i]], dim = 0) for i in range(len(title_embeddings)) ]

                else:
                    raise ValueError("Label Encoder not of either type DocumenEmbedding nor TokenEmbedding")
                #if device:
                #    embeddings = embeddings.to(device)
                final_embeddings.extend(embeddings)
                if clear_embeddings:
                    for l in batch:
                        l.clear_embeddings()
                del embeddings

                if use_tqdm:
                    pbar.update(min(batch_size, len(labels_sentence_objects)-i))

                i += batch_size

                if i > len(labels_sentence_objects):
                    break

                # try increasing the batch size for the next iteration
                if not fixed_batch_size and batch_size < initial_batch_size:
                    batch_size += int(step/2)

            except:
                if not fixed_batch_size:
                    batch_size -= step
                else:
                    batch_size -= 1
                torch.cuda.empty_cache()
                if batch_size < 1:
                    raise RuntimeError("Batch size too small for processing")

        if use_tqdm:
            pbar.close()

        final_embeddings = torch.stack(final_embeddings, dim = 0) # correct? todo
        if device:
            final_embeddings.to(device)
        final_embeddings = final_embeddings[inverse_indices]

        if update_these_embeddings:
            if self.constant_updating:
                self._update_some_label_embeddings(labels=labels, new_label_embeddings=final_embeddings)

        return final_embeddings

    def _negative_sampling_shift(self, span_embeddings: torch.Tensor, batch_gold_labels: List[str]):
        """
        Shifting the labels to make them negatives for each other.
        :param span_embeddings: Not used in this strategy.
        :param batch_gold_labels: Gold labels of the spans in this batch. Get shifted.
        :return: Negative labels. A list of labels (note: if self._negative_sampling_factor >1 be careful with unrolling correctly)
        """
        negative_samples = []
        for i in range(self._negative_sampling_factor):
            negative_samples.extend(list(np.roll(batch_gold_labels, shift=1+i)))

        indices_where_gold_already_nearest = []
        ## TODO

        return negative_samples, indices_where_gold_already_nearest


    def _negative_sampling_random_over_all(self, span_embeddings: torch.Tensor, batch_gold_labels: List[str]):
         # todo: currently it's possibly that the gold label is samples as negative
         if self._label_dict is None:
             self._create_label_dict()
         negative_samples_indices = []
         for i in range(self._negative_sampling_factor):
             negative_samples_indices.extend(random.sample(range(len(self._label_dict.items)), len(batch_gold_labels)))

         negative_labels = [self._label_at(i) for i in negative_samples_indices]

         return negative_labels

    def _negative_sampling_hard(self, span_embeddings: torch.Tensor, batch_gold_labels: List[str], return_indices_where_gold_already_nearest: bool = True):
        """
        Look for difficult labels as negatives (i.e. similarity to mention embeddings).
        :param span_embeddings: Embeddings of the spans in this batch.
        :param batch_gold_labels: Gold labels of the spans in this batch.
        :return: Negative labels. A list of labels (note: if self._negative_sampling_factor >1 be careful with unrolling correctly)
        """
        if self._label_dict is None:
            self._create_label_dict()

        with torch.no_grad():

            indices_where_gold_already_nearest = []
            span_embeddings = span_embeddings.to(self._label_embeddings_storage_device)

            # reembed the labels every N step to have more recent embeddings
            if self.constant_updating and self._seen_spans - self._last_update >= 160000: # 40000
                self._recompute_label_embeddings()

            similarity_spans_labels = self.similarity_metric.similarity(span_embeddings, self.get_label_embeddings())

            gold_label_indices = [ self._idx_for_label(label) for label in batch_gold_labels ]
            gold_label_indices = torch.tensor(gold_label_indices).to(flair.device)

            # check which of the gold labels are in the label set (it is possible that some are not)
            gold_is_in_sample = gold_label_indices != self._INDEX_NOT_FOUND # torch.tensor(-1, device=flair.device)

            # only keep the indices where the gold label exists in label set (for assigning -inf later):
            spans_range = torch.arange(len(batch_gold_labels), device=flair.device)
            spans_range = spans_range[gold_is_in_sample]
            gold_label_similarities = [similarity_spans_labels[i, gold_label_indices[i]].item() if gold_label_indices[i] != self._INDEX_NOT_FOUND else -torch.inf for i in range(len(span_embeddings))]

            # set the similarity to the true gold label to -inf, so it will not be sampled as negative
            gold_label_indices = gold_label_indices[gold_is_in_sample]
            similarity_spans_labels[spans_range, gold_label_indices] = -torch.inf

            # Top K sampling (always the hardest)
            #_, most_similar_label_index = torch.topk(similarity_spans_labels, self._negative_sampling_factor, dim=1)

            # Multinomial sampling (with temperature)

            temperature = 0.05
            similarity_temperature = similarity_spans_labels.div(temperature)
            #
            # # Susanna's method:
            similarity_as_probabilities = torch.softmax(similarity_temperature, dim=1)
            #
            # # Alan's method:
            # # to prevent overflow problem with small temperature values, substract largest value from all
            # # this makes a vector in which the largest value is 0
            #max_values, _ = torch.max(similarity_temperature, dim=1, keepdim=True)
            #similarity_temperature = similarity_temperature - max_values
            #similarity_as_probabilities = similarity_temperature.exp()

            most_similar_label_index = torch.multinomial(similarity_as_probabilities, self._negative_sampling_factor)
            most_similar_label_index_best = most_similar_label_index[:,0]
            for i in range(len(span_embeddings)):
                if gold_label_similarities[i] > similarity_spans_labels[i, most_similar_label_index_best[i]]:# + self.loss_function.margin:
                    indices_where_gold_already_nearest.append(i)

        most_similar_label_index = most_similar_label_index.T.flatten()

        most_similar_labels = [self._label_at(i) for i in most_similar_label_index]

        if return_indices_where_gold_already_nearest:
            return most_similar_labels, indices_where_gold_already_nearest
        else:
            return most_similar_labels

    def _negative_sampling_hard_and_random(self, span_embeddings: torch.Tensor, batch_gold_labels: List[str]):
        hard_negatives, _ = self._negative_sampling_hard(span_embeddings, batch_gold_labels)
        random_negatives = self._negative_sampling_random_over_all(span_embeddings, batch_gold_labels)

        # chose randomly either the hard or the negative one per sample:
        return [random.choice([hard, rand]) for hard, rand in zip(hard_negatives, random_negatives)], _


    def get_label_embeddings(self):
        if self._label_dict is None:
            self._create_label_dict()

        if self._label_embeddings is None:
            self._recompute_label_embeddings()

        return self._label_embeddings


    def get_sentence_objects_for_labels(self, labels, use_tqdm: bool = False):
        if not self._label_dict:
            self._create_label_dict()

        sentence_objects = []

        label_iterator = range(0, len(labels))

        if use_tqdm:
            label_iterator = tqdm(label_iterator, position=0, leave=True)

        for i in label_iterator:
            l = labels[i]
            sentence_object = self._label_dict.sentence_object_for(l)
            if not sentence_object:
                sentence_object = flair.data.Sentence(self.label_map.get(l,l.replace("_", " ")).replace("ʼ", "'")) # see issue https://github.com/flairNLP/flair/issues/3594
                sentence_object.set_label("last title token", len(sentence_object)-1) # default is last token of whole verbalization
                for token in sentence_object:
                    if token.text == ";":
                        sentence_object.set_label("last title token", token.idx-2) # set to the last token of title
                        break
                self._label_dict.add_sentence_object_for(l, sentence_object)
            sentence_objects.append(sentence_object)

        return sentence_objects

    #@torch.compile
    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:
        """
        One forward pass through the model. Embed sentences, get span representations, get label representations, sample negative labels, compute loss.
        :param sentences: Sentences in batch.
        :return: Tuple(loss, number of spans)
        """

        # label samples will need updated embeddings in the prediction
        self._next_prediction_needs_updated_label_embeddings = True

        if len(sentences) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        (spans, span_embeddings) = self._embed_spans(sentences)
        if spans is None:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        nr_spans = len(spans)

        # get one embedding vector for each label
        labels = [sp.get_label(self.label_type).value for sp in spans]

        # sample negative labels

        negative_sampling_factor_before = self._negative_sampling_factor
        # and also, optionally then delete the samples where gold is already the closest one
        delete_where_gold_already_nearest = False
        if self._seen_spans <= 30000:
            if negative_sampling_factor_before == "dyn":
                self._negative_sampling_factor = 1
            # start with mix of random and hard labels
            #negative_labels, indices_where_gold_already_nearest = self._negative_sampling_hard_and_random(span_embeddings, labels)

        else:
            if negative_sampling_factor_before == "dyn":
                # after some steps, set the negative_sampling_factor dynamically, depending on the nr of spans in the batch:
                calculated_factor = min(10, int(100/nr_spans))
                if calculated_factor < 1:
                    calculated_factor = 1
                self._negative_sampling_factor = calculated_factor

            #delete_where_gold_already_nearest = True

        negative_labels, indices_where_gold_already_nearest = self._negative_sampling_fn(span_embeddings, labels)

        if delete_where_gold_already_nearest:
            indices_to_use = [i for i in range(len(labels)) if i not in indices_where_gold_already_nearest]
            labels = [labels[i] for i in indices_to_use]
            negative_indices_to_use = [[f * len(spans) + i for i in indices_to_use] for f in
                                       range(self._negative_sampling_factor)]
            negative_indices_to_use = [item for sublist in negative_indices_to_use for item in sublist]

            negative_labels = [negative_labels[i] for i in negative_indices_to_use]
            span_embeddings = span_embeddings[indices_to_use]

        if len(labels) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        # concatenate and embed together
        together = labels + negative_labels

        #print("Now embedding nr labels:", len(together))
        try:
            together_label_embeddings = self._embed_labels_batchwise_return_stacked_embeddings(labels = together, use_tqdm=False, fixed_batch_size=32)
        except Exception as e:
            print("Nr Spans", len(spans))
            print("Nr Labels", len(labels))
            print("Nr negative Labels", len(negative_labels))
            print("Nr sentences", len(sentences))
            print("Sentence lengths:")
            for s in sentences:
                print(len(s.text))
            raise e

        # divide into (gold) label and negative embeddings (negatives must be shaped as negative_factor x num_spans x embedding_size)
        label_embeddings = together_label_embeddings[:len(labels)]
        negative_label_embeddings = torch.reshape(together_label_embeddings[len(labels):], (self._negative_sampling_factor, len(labels), span_embeddings.shape[1]))

        # calculate loss
        loss = self.loss_function(span_embeddings, label_embeddings, negative_label_embeddings)

        self._negative_sampling_factor = negative_sampling_factor_before

        del together_label_embeddings, label_embeddings, negative_label_embeddings, span_embeddings, together, labels, negative_labels, spans
        #gc.collect()
        #torch.cuda.empty_cache()

        self._iteration_count += 1
        self._seen_spans += nr_spans

        # if isinstance(self.loss_function, DEEDTripletMarginLoss):
        #     if self._iteration_count % self.loss_function.margin_adjustment_frequency == 0 and self._iteration_count > 0:
        #         self.loss_function.adjust_margin(increase=True)

        return loss, nr_spans

    def predict(
            self,
            sentences: Union[List[DT], DT],
            mini_batch_size: int = 32,
            return_probabilities_for_all_classes: bool = False,
            verbose: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            top_k: int = 5,
            embedding_storage_mode="none",
            return_span_and_label_hidden_states: bool = True,
            **kwargs
    ):
        """
        Predicts labels for the spans in sentences. Adds them to the spans under label_name.
        :return:
        """

        with torch.no_grad():# if not self.training else torch.enable_grad():

            # After a forward loss the embeddings of the labels might be outdated because the weights of the label encoder have changed.
            # Also, after resampling of the labels the label embeddings might not yet exist
            # To avoid unnecessary work the labels only get embedded here (important for a large label set, might take very long)
            if self._next_prediction_needs_updated_label_embeddings:
                self._recompute_label_embeddings()
                self._next_prediction_needs_updated_label_embeddings = False
            
            batches, batches_original_spans = self._prepare_sentences(sentences,
                                                                      max_spans_per_sentence=200,
                                                                      max_spans_per_batch=400,
                                                                      )
            for batch, original_spans in zip(batches, batches_original_spans):

                if not original_spans:
                    continue
                (spans, span_embeddings) = self._embed_spans(batch)

                label_embeddings = self.get_label_embeddings().to(flair.device)
                # Choosing the most similar label from the set of labels (might not include the true gold label)
                similarity_span_all_labels = self.similarity_metric.similarity(span_embeddings, label_embeddings)

                most_similar_label_similarity, most_similar_label_index = torch.max(similarity_span_all_labels, dim=1)

                # for inspection (and for the experiment with a different criterion) save the top 5 predictions:
                top5_similarity, top5_index = torch.topk(similarity_span_all_labels, k=top_k, dim=1)
                
                for i, sp in enumerate(spans):
                    original_span = original_spans[i]
                    label_value = self._label_at(most_similar_label_index[i])
                    label_score = most_similar_label_similarity[i].item()

                    original_span.set_label(label_name, label_value, score = label_score)

                    if save_top_k:
                        top_k = zip(top_k_similarity[i], top_k_index[i])
                        for t_i, (t_sim, t_index) in enumerate(top_k):
                           original_span.set_label(typename=f"top_{t_i+1}", value=self._label_at(t_index.item()), score=t_sim.item())

                del label_embeddings, span_embeddings, similarity_span_all_labels

        if return_loss:
            # todo not yet implemented
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=False), sum([len(b) for b in batches_original_spans])

    def _print_predictions(self, batch, gold_label_type, print_top_k = True):
        lines = []
        for datapoint in batch:
            eval_line = f"\n{datapoint.to_original_text()}\n"

            for span in datapoint.get_spans(gold_label_type):
                pred = span.get_label("predicted").value
                symbol = "✓" if span.get_label(gold_label_type).value == pred else "❌"
                verbalization = self.label_map.get(pred,pred.replace("_", " "))
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {pred} ({symbol}) "{verbalization}"'
                    f' --> {[(span.get_label(f"top_{i}").value, span.get_label(f"top_{i}").score) for i in range(5)]}'
                    f' --> {span.start_position} - {span.end_position}\n'
                )

                if print_top_k:
                    top_k_labels = [l for l in span.annotation_layers if l.startswith("top_")]
                    if top_k_labels:
                        eval_line += "   Top K: "
                        for label_name in top_k_labels:
                            eval_line += f"{span.get_label(label_name).value}={span.get_label(label_name).score:.3f} "
                    eval_line += "\n"

            lines.append(eval_line)

        return lines


    def _get_state_dict(self):
        # todo Something missing here?
        model_state = {
            **super()._get_state_dict(),
            "label_encoder": self.label_encoder,
            "token_encoder": self.token_encoder,
            "label_type": self.label_type,
            "label_map": self.label_map,
            "known_labels": self.known_labels,
            #"negative_sampling_factor": self._negative_sampling_factor,
            "negative_sampling_strategy": self.negative_sampling_strategy,
            "loss_function": self.loss_function,
            "similarity_metric": self.similarity_metric

        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        return super()._init_model_with_state_dict(
            state,
            label_encoder = state.get("label_encoder"),
            token_encoder = state.get("token_encoder"),
            label_type = state.get("label_type"),
            label_map = state.get("label_map"),
            known_labels = state.get("known_labels"),
            #negative_sampling_factor = state.get("negative_sampling_factor"),
            negative_sampling_strategy = state.get("negative_sampling_strategy"),
            loss_function = state.get("loss_function"),
            similarity_metric = state.get("similarity_metric"),

            **kwargs,
        )



class GreedyDualEncoderEntityDisambiguation(DualEncoderEntityDisambiguation):
    """
    This is the greedy version of the DualEncoderEntityDisambiguation Class.
    During training, some of the gold labels get used for label verbalization insertion.
    During prediction, the most confident predicted labels get used for insertions, while the new sentences get
    re-embedded and predicted. This process is iterative until all spans have predicted labels.
    """

    def __init__(self, insert_in_context: Union[int, bool] = False, insert_which_labels: str = "gold_pred", **kwargs):
        super(GreedyDualEncoderEntityDisambiguation, self).__init__(**kwargs)
        if not insert_in_context:
            self.insert_in_context = 0
        elif insert_in_context == True:
            self.insert_in_context = 2
        else:
            self.insert_in_context = insert_in_context

        self.insert_which_labels = insert_which_labels


    def sample_spans_to_use_for_gold_label_verbalization(self, sentences, label_name: str = "nel", marker_name: str = "to_verbalize", search_context_window: int = 0):
        """
        Samples random spans with a label_type annotation that will be used for gold label verbalization insertion during training.
        :param sentences: Sentences too search for spans.
        :param search_context_window: Number of context sentences before and after to include in search. Set to 0 if only the current sentence should be used.
        :return: Spans that were chosen for label verbalization.
        """
        spans = []
        for s in sentences:
            spans.extend(s.get_spans(label_name))
            (previous, next) = (s._previous_sentence, s._next_sentence)
            for i in range(search_context_window):
                if previous:
                    spans.extend(previous.get_spans(label_name))
                    previous = previous._previous_sentence
                if next:
                    spans.extend(next.get_spans(label_name))
                    next = next._next_sentence
        # In case we do not shuffle and use search_context_window, the same spans would keep getting added. Use set() to only use them once.
        spans = list(set(spans))
        number_of_spans_to_verbalize = random.randint(0, len(spans))
        sampled_spans = random.sample(spans, number_of_spans_to_verbalize)

        # add a verbalization marker to the chosen spans:
        for sp in sampled_spans:
            label = sp.get_label(label_name)
            sp.set_label(marker_name, value=label.value, score=label.score)

        return sampled_spans

    def sample_predicted_spans_for_label_verbalization_in_training(self, sentences, label_name = "predicted_for_forward", marker_name: str = "to_verbalize"):

        if self._label_dict is None:
            self._create_label_dict()

        self._next_prediction_needs_updated_label_embeddings = False

        super(GreedyDualEncoderEntityDisambiguation, self).predict(sentences,
                                                                   label_name=label_name,
                                                                   return_loss=False,
                                                                   # embedding_storage_mode=embedding_storage_mode
                                                                   )

        spans = []
        for s in sentences:
            spans.extend(s.get_spans(self.label_type))

        predicted_spans = self.select_predicted_spans_to_use_for_label_verbalization(sentences,
                                                                                    label_name=label_name,
                                                                                    nr_steps=3) # so roughly the best 1/3 of predicted spans

        number_of_spans_to_verbalize = random.randint(0, len(predicted_spans))

        sampled_spans = random.sample(predicted_spans, number_of_spans_to_verbalize)

        for sp in sampled_spans:
            label = sp.get_label(label_name)
            sp.set_label(marker_name, value=label.value, score=label.score)

        for s in sentences:
            s.remove_labels(label_name)

        return sampled_spans


    def select_predicted_spans_to_use_for_label_verbalization(self, sentences, label_name, nr_steps: int):
        """
        From all spans with label_tape (e.g. "nel") and label_name (e.g. "predicted") annotation, take the n spans with highest score.
        :param sentences: Sentences to select spans from.
        :param label_name: Label type that is storing the scores of predictions (e.g. "predicted").
        :param nr_steps: Number of iterations (roughly).
        :return: n or less spans.
        """

        if nr_steps < 1:
            nr_steps = 1
        spans = []
        for s in sentences:
            spans.extend([sp for sp in s.get_spans(label_name) if sp.has_label(self.label_type)])

        # sequential (natural order)
        # chosen = []
        # for s in sentences:
        #     spans_in_sentence = [sp for sp in s.get_spans(label_name) if sp.has_label(self.label_type)]
        #     if len(spans_in_sentence) > 0:
        #         chosen.extend(spans_in_sentence[:ceil(len(spans_in_sentence)/nr_steps)])

        # first method: choose n most confident per batch
        # sorted_spans = sorted(spans, key = lambda sp: sp.get_label(label_name).score, reverse = True)
        # chosen = sorted_spans[:ceil(len(sorted_spans)/nr_steps)]

        # now: chose the most confident per sentence:
        chosen = []
        for s in sentences:
            spans_in_sentence = [sp for sp in s.get_spans(label_name) if sp.has_label(self.label_type)]
            if len(spans_in_sentence) > 0:
                sorted_spans_in_sentence = sorted(spans_in_sentence, key=lambda sp: sp.get_label(label_name).score, reverse=True)
                # only use them if the score is better than the prediction from previous iteration (if any)
                if nr_steps > 1:
                    sorted_spans_only_better_score = []
                    for sp in sorted_spans_in_sentence:
                        predicted_before_key = next((key for key in sp.annotation_layers.keys() if key.startswith("predicted:")), None)
                        if predicted_before_key:
                            if sp.get_label(label_name).score > sp.get_label(predicted_before_key).score:
                                sorted_spans_only_better_score.append(sp)
                        else:
                            sorted_spans_only_better_score.append(sp)

                    most_confident = sorted_spans_only_better_score[:ceil(len(sorted_spans_in_sentence)/nr_steps)]
                else:
                    most_confident = sorted_spans_in_sentence[:ceil(len(sorted_spans_in_sentence)/nr_steps)]

                chosen.extend(most_confident)

        #alternative: chose N most distinct (i.e. largest gap to the next probable label) labels
        # import heapq
        #
        # def select_most_distinct_predictions(spans, n):
        #     most_distinct_spans = heapq.nlargest(n, spans, key=lambda sp: abs(sp.get_label("top_0").score - sp.get_label("top_1").score))
        #     return most_distinct_spans
        #
        # chosen = []
        # for s in sentences:
        #     s_predictions = [sp for sp in s.get_spans(label_name) if sp.has_label(self.label_type)]
        #     chosen.extend(select_most_distinct_predictions(s_predictions, ceil(len(s_predictions)/nr_steps)))

        ## try some sort of "relevance/importance to the sencence" score?
        # def compute_proximity_scores(sentence, label_name):
        #     """
        #     Given a sentence and a list of predictions, compute the semantic proximity
        #     scores between the sentence and each predicted label verbalization.
        #     """
        #     # Get some general sentence embedding
        #     self.token_encoder.embed(sentence)
        #     sentence_embedding = torch.mean(torch.stack([token.get_embedding() for token in sentence], 0), 0).unsqueeze(0)
        #
        #     predictions = [sp for sp in s.get_spans(label_name) if sp.has_label(self.label_type)]
        #     predictions_labels = [p.get_label(label_name).value for p in predictions]
        #     predictions_embeddings = self._embed_labels_batchwise_return_stacked_embeddings(predictions_labels, use_tqdm = False)
        #
        #     # compute similarity to sentence
        #     similarity_scores = self.similarity_metric.similarity(sentence_embedding, predictions_embeddings)
        #     paired_prediction_scores = list(zip(predictions, similarity_scores.squeeze(0)))
        #
        #     return paired_prediction_scores
        #
        # # Calculate proximity scores
        # chosen = []
        # for s in sentences:
        #     predictions = [sp for sp in s.get_spans(label_name) if sp.has_label(self.label_type)]
        #     if len(predictions) >0:
        #         paired_prediction_scores = compute_proximity_scores(s, label_name=label_name)
        #
        #         # Sort verbalizations by their proximity score in descending order
        #         paired_prediction_scores_sorted = sorted(paired_prediction_scores, key=lambda x: x[1], reverse=True)
        #         top_predictions, top_scores = zip(*paired_prediction_scores_sorted[:ceil(len(paired_prediction_scores_sorted)/nr_steps)])
        #         chosen.extend(top_predictions)

        return chosen

    def _insert_verbalizations_into_sentence(self, sentence: Sentence, label_type: str, label_map: dict,
                                             string_before_span: str = "", string_before_verbalization: str = " (", string_after_verbalization: str = ")",
                                             #string_before_span: str = "[", string_before_verbalization: str = "] (", string_after_verbalization: str = ")",
                                             #string_before_span: str = '"', string_before_verbalization: str = '" (', string_after_verbalization: str = ')',
                                             #string_before_span: str = "[S_MENTION]", string_before_verbalization: str = "[S_DESC]", string_after_verbalization: str = "[E_DESC]",
                                             #string_before_span: str = "[S_MENTION]", string_before_verbalization: str = "(", string_after_verbalization: str = ")",
                                             verbalize_previous: int = 0, verbalize_next: int = 0,
                                             negative_percentage = 0.0, drop_percentage = 0.0):
        """
        Insert label verbalizations into sentence.
        :param sentence: Flair sentence object to apply label verbalizations to.
        :param label_type: Label type whose value gets verbalized.
        :param label_map: A mapping of label values to more descriptive label verbalization to use for the insertions.
        :param verbalize_previous: Number of context sentences before that also get label insertion applied to. Set to 0 if no insertions into context sentences wanted.
        :param verbalize_next: Number of context sentences after that also get label insertion applied to. Set to 0 if no insertions into context sentences wanted.
        :param negative_percentage: Rate of how many of the verbalizations are corrupted (i.e. (hard) negative labels are used for verbalization), for robustness.
        :return: New Flair sentence object, now with verbalizations. Labels and context get copied from the original input sentence.
        """
        spans = sentence.get_spans()

        tokens_text = [t.text for t in sentence.tokens]
        added_tokens = 0
        token_indices = [[sp.tokens[0].idx, sp.tokens[-1].idx] for sp in spans]

        if negative_percentage > 0.0 and len(sentence.get_spans(label_type)) > 0:
            spans, span_embeddings = self._embed_spans([sentence])
            span_indexes_to_verbalize = [i for i,sp in enumerate(spans) if sp.get_label(label_type).value != "O"]
            span_embeddings_to_verbalize = span_embeddings[span_indexes_to_verbalize]
            true_labels = [sp.get_label(label_type).value for sp in sentence.get_spans(label_type)]
            negative_sampling_factor_before = self._negative_sampling_factor
            if negative_sampling_factor_before == "dyn":
                self._negative_sampling_factor = 1
            negatives, _ = self._negative_sampling_hard(span_embeddings_to_verbalize, true_labels)
            self._negative_sampling_factor = negative_sampling_factor_before

        for i, sp in enumerate(sentence.get_spans(label_type)):
            if negative_percentage == 0.0:
                label_to_insert = sp.get_label(label_type).value
            else:
                if random.random() < negative_percentage:
                    label_to_insert = negatives[i]
                else:
                    label_to_insert = sp.get_label(label_type).value

            #add_at_position_in_tokens = sp.tokens[-1].idx + added_tokens # when only AFTER span
            add_at_position_in_tokens = sp.tokens[0].idx -1 + added_tokens # when also sth before

            verbalization_string = label_map.get(label_to_insert, label_to_insert.replace('_', ' '))

            # cutting off the label name
            if ";" in verbalization_string:
                verbalization_string = verbalization_string.split(";", 1)[1].strip()

                # ONLY taking the label name
                # verbalization_string = verbalization_string.split(";", 1)[0].strip()

            if drop_percentage > 0.0:
                num_tokens = len(Sentence(verbalization_string))
                num_to_drop = int(num_tokens * drop_percentage)
                drop_indices = set(random.sample(range(num_tokens), num_to_drop))
                # create a new list excluding the tokens at drop_indices
                verbalization_string = " ".join([token.text for i, token in enumerate(Sentence(verbalization_string).tokens) if i not in drop_indices])

            if len(string_before_span) >0:
                verbalization_before_span = Sentence(f"{string_before_span}")
                verbalization_before_span_token_text = [t.text for t in verbalization_before_span.tokens]
            else:
                verbalization_before_span_token_text = []

            verbalization_after_span = Sentence(f"{string_before_verbalization} {verbalization_string} {string_after_verbalization}")
            verbalization_after_span_token_text = [t.text for t in verbalization_after_span.tokens]


            len_verbalization_before_tokens = len(verbalization_before_span_token_text)
            len_verbalization_after_tokens = len(verbalization_after_span_token_text)

            len_verbalization_tokens = len_verbalization_before_tokens + len_verbalization_after_tokens

            span_token_text = [t.text for t in sp.tokens]

            tokens_text = tokens_text[:add_at_position_in_tokens] + verbalization_before_span_token_text + span_token_text + verbalization_after_span_token_text + tokens_text[add_at_position_in_tokens+ len(span_token_text):]

            added_tokens += len_verbalization_tokens

            for j, d in enumerate(spans):
                s_start_idx = token_indices[j][0]
                s_end_idx = token_indices[j][1]

                if s_start_idx == add_at_position_in_tokens+1:
                    s_start_idx += len_verbalization_before_tokens
                    s_end_idx += len_verbalization_before_tokens
                    token_indices[j][0] = s_start_idx
                    token_indices[j][1] = s_end_idx

                elif s_start_idx > add_at_position_in_tokens+1:
                    s_start_idx += len_verbalization_tokens
                    s_end_idx += len_verbalization_tokens
                    token_indices[j][0] = s_start_idx
                    token_indices[j][1] = s_end_idx

        # Cannot use new_sentence = Sentence(text), we needed to use tokens instead of working with the text directly because of the weird problem with the space in unicode, e.g. 0x200c
        new_sentence = Sentence(tokens_text)

        for i, sp in enumerate(spans):
            start_token_index, end_token_index = token_indices[i]
            new_sp = Span(new_sentence.tokens[start_token_index - 1:(end_token_index)])

            for k, labels in sp.annotation_layers.items():
                for l in labels:
                    new_sp.set_label(typename=k, value=l.value, score=l.score)

        if verbalize_previous > 0 and sentence._previous_sentence:
            new_sentence._previous_sentence = self._insert_verbalizations_into_sentence(sentence._previous_sentence,
                                                                                  label_type, label_map,
                                                                                  verbalize_previous=verbalize_previous - 1,
                                                                                  verbalize_next=0)
        else:
            new_sentence._previous_sentence = sentence._previous_sentence

        if verbalize_next > 0 and sentence._next_sentence:
            new_sentence._next_sentence = self._insert_verbalizations_into_sentence(sentence._next_sentence, label_type,
                                                                              label_map, verbalize_previous=0,
                                                                              verbalize_next=verbalize_next - 1)
        else:
            new_sentence._next_sentence = sentence._next_sentence

        return new_sentence

    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:
        """
        Forward pass through the (greedy) model. Same as the DualEncoderEntityDisambiguation model class, but adding some gold verbalizations beforehand.
        :param sentences: Sentences in batch.
        :return: Tuple(loss, number of spans)
        """

        marker_name = "to_verbalize"
        # sample some spans that will get verbalized (and not taken into consideration for loss)

        insert_which_labels_before = self.insert_which_labels

        if self.insert_which_labels == "gold_pred":
            # first use gold, then switch to pred
            if self._seen_spans <= 30000:
                self.insert_which_labels = "gold"
            else:
                self.insert_which_labels = "pred"

        if self.insert_which_labels == "gold":
            # use gold labels (and some negatives)
            sampled_spans = self.sample_spans_to_use_for_gold_label_verbalization(sentences, label_name = self.label_type, marker_name = marker_name, search_context_window=self.insert_in_context)
            negative_percentage = 0.1

        elif self.insert_which_labels == "pred":
            # mirror real prediction:
            sampled_spans = self.sample_predicted_spans_for_label_verbalization_in_training(sentences, label_name="predicted_for_forward", marker_name = marker_name)
            negative_percentage = 0.0

        # insert verbalizations (from the sampled_spans) into the sentences, using the verbalization marker:
        verbalized_sentences = [
            self._insert_verbalizations_into_sentence(s, marker_name, label_map = self.label_map,
                                                      verbalize_previous=self.insert_in_context, verbalize_next=self.insert_in_context,
                                                      negative_percentage=negative_percentage) for s in
            sentences]

        self.insert_which_labels = insert_which_labels_before

        # remove the verbalization marker from the ORIGINAL spans so that they remain unmodified:
        for sp in sampled_spans:
            sp.remove_labels(marker_name)

        # delete the label_type for the spans that were used for verbalization from the verbalized_sentences,
        # so that those do not get used in the forward pass afterwards:
        for s in verbalized_sentences:
            for sp in s.get_spans(marker_name):
                sp.remove_labels(self.label_type)

        # do the normal forward pass, now with the modified sentences (with less datapoints):
        return super(GreedyDualEncoderEntityDisambiguation, self).forward_loss(verbalized_sentences)


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
        **kwargs
    ):
        """
        Predict labels for sentences. Uses the predict method from DualEncoderEntityDisambiguation, but in an iterative fashion.
        """
        # remove all annotations from possible previous evaluations:
        for s in sentences:
            label_names = list(s.annotation_layers.keys())
            for l in label_names:
                if l != self.label_type:
                    s.remove_labels(l)

        original_nr_spans = sum([len(s.get_spans(self.label_type)) for s in sentences])
        nr_steps = 3
        level = 0

        # iterate until all spans are predicted
        sentences_to_use = sentences
        while True:
            for s in sentences:
                s.remove_labels("predicted")

            super(GreedyDualEncoderEntityDisambiguation, self).predict(sentences_to_use,
                                  mini_batch_size=mini_batch_size,
                                  return_probabilities_for_all_classes=return_probabilities_for_all_classes,
                                  verbose=verbose,
                                  label_name=label_name,
                                  return_loss=return_loss,
                                  embedding_storage_mode=embedding_storage_mode,
                                  save_top_k=save_top_k,
                                  )

            chosen_spans = self.select_predicted_spans_to_use_for_label_verbalization(sentences_to_use,
                                                                                      label_name=label_name,
                                                                                      nr_steps = nr_steps)
            if level >= 2: # take prediction for all remaining, don't verbalize further
                chosen_spans = []

            # verbalization markers for the current level
            verbalized_label_type = f"verbalized:{level}"
            input_sentence_label_type = f"input_sentence:{level}"

            predicted_spans = []
            for s in sentences_to_use:
                predicted_spans.extend([sp for sp in s.get_spans(label_name) if sp.has_label(self.label_type)])
            # mark the chosen spans as well as the other ones accordingly:
            for sp in predicted_spans:
                predicted_label = sp.get_label(label_name)
                sp.set_label(f"predicted:{level}", value= predicted_label.value, score = predicted_label.score)
                span_marked_sentence = sp.sentence.text[
                                       :sp.start_position] + "[SPAN_START] " + sp.text + " [SPAN_END]" + sp.sentence.text[
                                                                                                         sp.end_position:]
                sp.set_label(input_sentence_label_type, value=span_marked_sentence, score=0.0)
                if sp in chosen_spans:
                    sp.set_label(verbalized_label_type, value=predicted_label.value, score=predicted_label.score)

            # if no spans remaining, break
            if len(chosen_spans) == 0:
                break

            # insert the label verbalizations of the chosen spans
            verbalized_sentences = [
                self._insert_verbalizations_into_sentence(s, verbalized_label_type, label_map = self.label_map, verbalize_previous=self.insert_in_context, verbalize_next=self.insert_in_context, negative_percentage=0.0)
                for s in sentences_to_use]
            # keep the spans in the original sentences unmodified
            for sp in chosen_spans:
                sp.remove_labels(verbalized_label_type)
                sp.remove_labels(input_sentence_label_type)
            # remove the label_type marker from the spans that were used for label verbalization insertion, so they will not be predicted again
            for s in verbalized_sentences:
                for sp in s.get_spans(verbalized_label_type):
                    sp.remove_labels(self.label_type)

            # prepare for the next iteration
            sentences_to_use = verbalized_sentences
            del verbalized_sentences
            level +=1
            nr_steps -=1


        original_spans = []
        for s in sentences:
            original_spans.extend(s.get_spans(self.label_type))

        nr_spans = len(original_spans)

        predicted_spans = []
        for s in sentences_to_use:
            predicted_spans.extend(s.get_spans(label_name))

        assert len(predicted_spans) == len(original_spans), \
            f"Not all spans could be verbalized: original: {len(original_spans)}, predicted: {len(predicted_spans)}"

        # transfer all the predicted labels to the original sentences and their spans
        for (orig, pred) in zip(original_spans, predicted_spans):

            # save the input sentence versions that were used for each span (that include the verbalizations at the time)
            predicted_at = 0
            max_level = 0
            best_score = -np.inf
            for key in pred.annotation_layers.keys():
                if key.startswith("predicted:"):
                    level = int(key.split(":")[1])
                    max_level = max(max_level, level)
                    if pred.get_label(key).score > best_score:
                        predicted_at = level
                        best_score = pred.get_label(key).score
                        label = pred.get_label(key)
                        orig.set_label(label_name, label.value, label.score)
                        orig.set_label("sentence_input", pred.get_label(f"input_sentence:{predicted_at}").value, score=0.0)
                        orig.set_label("predicted_at_step", value=predicted_at, score=0.0)

            # also save the predictions of earlier steps:
            for step in range(max_level +1):
                orig.set_label(f"predicted:{step}", value = pred.get_label(f"predicted:{step}").value, score = pred.get_label(f"predicted:{step}").score)

        del original_spans, predicted_spans, sentences_to_use

        if return_loss:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=False), nr_spans


    def _print_predictions(self, batch, gold_label_type, add_sentence_input: bool = True):
        lines = []
        for datapoint in batch:
            eval_line = f"\n{datapoint.to_original_text()}\n"

            for span in datapoint.get_spans(gold_label_type):
                pred = span.get_label("predicted").value
                score = round(span.get_label("predicted").score, 2)
                symbol = "✓" if span.get_label(gold_label_type).value == pred else "❌"
                verbalization = self.label_map.get(pred,pred.replace("_", " "))
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {score} {pred} ({symbol}) "{verbalization}"\n'
                )
                prediction_steps = []
                step = 0
                while True:
                    label = span.get_label(f"predicted:{step}").value
                    score = round(span.get_label(f"predicted:{step}").score, 2)
                    if label == "O":
                        break
                    prediction_steps.append([score, label])
                    step += 1

                symbols = ["✓" if l[1] == span.get_label(gold_label_type).value else "❌" for l in prediction_steps]
                prediction_steps_string = [f"{str(e[0])},{e[1]}" for e in prediction_steps]
                eval_line += (
                    f'  (steps: {"-->".join(prediction_steps_string)}, so: {"".join(symbols)})\n'
                )

                if add_sentence_input:
                    eval_line += (
                        f'  PREDICTED AT STEP "{span.get_label("predicted_at_step").value}"\n'
                    )
                    eval_line += (
                    f'  <-- "{span.get_label("sentence_input").value}"\n\n'
                    )

            lines.append(eval_line)
        return lines

    def _get_state_dict(self):
        # todo Something missing here?
        model_state = {
            **super()._get_state_dict(),
            "insert_in_context": self.insert_in_context

        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        model = super()._init_model_with_state_dict(
            state,
            **kwargs,
        )

        model.insert_in_context = state.get("insert_in_context", model.insert_in_context)

        return model


class LocalLLM():

    def __init__(self, llm: LLM, max_output_tokens: int = 200, temperature: float = 0.8):
        
        self.llm = llm
        self.max_output_tokens = max_output_tokens
        self.sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=self.max_output_tokens)

    def query(self, prompt: str):
        return self.query_batch([prompt])[0]

    def query_batch(self, prompts: list[str]):
        
        system_msg = "You are a professional entity-disambiguation annotator. For each question in the prompt you must select exactly one answer number.\n\n" \
                     "Hard rules (must follow):" \
                     "1. Read the full text, then answer every question in order (Q1, Q2, ...). \n" \
                     "2. For each question, consider ONLY the mention marked with that question tag (e.g., [Q1]) and ONLY the options listed immediately after that question. Do NOT reuse options from other questions.\n" \
                     "3. Choose the option that best matches the mention given the context. Prefer contextual relevance over surface-form match when they conflict. Prefer the most specific, contextually accurate entry when multiple options fit.\n" \
                     "4. If no option matches, choose '0' (None of the above).\n" \
                     "5. Sometimes, the answer numbering may not be continuous. Carefully look at the answer number of your selected option and provide it.\n" \
                     "6. Output format: exactly N lines for N questions. Each line must contain exactly one digit (the chosen option number). The first line is the answer for Q1, the second for Q2, etc. No other characters, labels, punctuation, or explanation. No blank lines.\n" \
                     "7. Do NOT provide chain-of-thought or any additional text.\n"
        
        message = [
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
            for prompt in prompts
        ]

        outputs = self.llm.chat(message, self.sampling_params)
        parsed_output = []
        input_tokens = []
        output_tokens = []

        for output in outputs:
            generated_text = output.outputs[0].text

            # Check if </think> tags is present. If yes, strip everything before it
            if "</think>" in generated_text:
                generated_text = generated_text.split("</think>")[-1].strip()

            parsed_output.append(generated_text)
            input_tokens.append(len(output.prompt_token_ids))
            output_tokens.append(len(output.outputs[0].token_ids))

        return parsed_output, input_tokens, output_tokens

class GradioLLM():

    def __init__(self, model_name):
        if model_name == "HU-1":
            self.client = GradioClient("https://llm1-compute.cms.hu-berlin.de/")
        elif model_name == "HU-2":
            self.client = GradioClient("https://llm3-compute.cms.hu-berlin.de/")
        else:
            raise NotImplementedError(f"Model {model_name} not supported. Supported options are: HU-1, HU-2.")
        
    def query(self, prompt: str):
        response = self.client.predict(
            param_0=prompt,
            api_name="/chat")
        
        return response, 0, 0


class GoogleLLM():
    """
    Class for querying Google's LLMs, such as Gemini.
    This class requires an API key to access the Google service.
    :param model_name: The name of the Google model to use (default is "gemini-2.5-flash").
    :param api_key: The API key for accessing Google's services.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None, max_output_tokens: int = 200, reasoning: str = "none"):
        self.model_name = model_name
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning

        if not self.api_key:
            raise ValueError("API key is required for Google LLMs. Please provide a valid API key.")

        self.client = genai.Client(
            api_key=self.api_key
        )

    def query(self, prompt: str, temperature: float = 0.7):
        """
        Method for querying the Google LLM.
        :param prompt: The input prompt to send to the LLM.
        :param max_tokens: Maximum number of tokens to generate in the response.
        :param temperature: Sampling temperature for response generation.
        :return: The generated response from the LLM.
        """

        #system_msg = "You are a helpful assistant that disambiguates entities in a text based on a number of options. You will receive a text and a number of multiple choice questions. Only respond with the number of your selected answer for each multiple choice question. One answer per line. Do not add any additional text or explanations.\n" \
        #            "Here are some tips:\n" \
        #            "- Remember that you are doing an Entity Disambiguation tasks. Try to be as good as a professional annotator.\n" \
        #            "- Give extra consideration to answer options whose surface form matches or closely matches the entity in question. These answer options are often correct but not always. If you think another option fits better, pick that one."

        system_msg = "You are a professional entity-disambiguation annotator. For each question in the prompt you must select exactly one answer number.\n\n" \
                     "Hard rules (must follow):" \
                     "1. Read the full text, then answer every question in order (Q1, Q2, ...). \n" \
                     "2. For each question, consider ONLY the mention marked with that question tag (e.g., [Q1]) and ONLY the options listed immediately after that question. Do NOT reuse options from other questions.\n" \
                     "3. Choose the option that best matches the mention given the context. Prefer contextual relevance over surface-form match when they conflict. Prefer the most specific, contextually accurate entry when multiple options fit.\n" \
                     "4. If no option matches, choose '0' (None of the below).\n" \
                     "5. Sometimes, the answer numbering may not be continuous. Carefully look at the answer number of your selected option and provide it.\n" \
                     "6. Output format: exactly N lines for N questions. Each line must contain exactly one digit (the chosen option number). The first line is the answer for Q1, the second for Q2, etc. No other characters, labels, punctuation, or explanation. No blank lines.\n" \
                     "7. Do NOT provide chain-of-thought or any additional text.\n"
    
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_msg,
            ),
            contents=prompt
        )

        input_tokens = int(response.usage_metadata.prompt_token_count)
        output_tokens = int(response.usage_metadata.total_token_count) - input_tokens

        return response.text, input_tokens, output_tokens


class OpenAILLM():
    """
    Class for querying OpenAI's LLMs, such as gpt-4o-mini.
    This class requires an API key to access the OpenAI service.
    :param model_name: The name of the OpenAI model to use (default is "gpt-4o-mini").
    :param api_key: The API key for accessing OpenAI's services.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None, max_output_tokens: int = 200, reasoning: str = "none"):
        self.model_name = model_name
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning

        if not self.api_key:
            raise ValueError("API key is required for OpenAI LLMs. Please provide a valid API key.")
        
        self.client = OpenAI(
            api_key=self.api_key
        )

    def query(self, prompt: str, temperature: float = 0.7):
        """
        Method for querying the OpenAI LLM.
        :param prompt: The input prompt to send to the LLM.
        :param max_tokens: Maximum number of tokens to generate in the response.
        :param temperature: Sampling temperature for response generation.
        :return: The generated response from the LLM.
        """

        #system_msg = "You are a helpful assistant that disambiguates entities in a text based on a number of options. You will receive a text and a number of multiple choice questions. Only respond with the number of your selected answer for each multiple choice question. One answer per line. Do not add any additional text or explanations.\n" \
        #            "Here are some tips:\n" \
        #            "- Remember that you are doing an Entity Disambiguation tasks. Try to be as good as a professional annotator.\n" \
        #            "- Give extra consideration to answer options whose surface form matches or closely matches the entity in question. These answer options are often correct but not always. If you think another option fits better, pick that one."
        
        system_msg = "You are a professional entity-disambiguation annotator. For each question in the prompt you must select exactly one answer number.\n\n" \
                     "Hard rules (must follow):" \
                     "1. Read the full text, then answer every question in order (Q1, Q2, ...). \n" \
                     "2. For each question, consider ONLY the mention marked with that question tag (e.g., [Q1]) and ONLY the options listed immediately after that question. Do NOT reuse options from other questions.\n" \
                     "3. Choose the option that best matches the mention given the context. Prefer contextual relevance over surface-form match when they conflict. Prefer the most specific, contextually accurate entry when multiple options fit.\n" \
                     "4. If no option matches, choose '0' (None of the above).\n" \
                     "5. Sometimes, the answer numbering may not be continuous. Carefully look at the answer number of your selected option and provide it.\n" \
                     "6. Output format: exactly N lines for N questions. Each line must contain exactly one digit (the chosen option number). The first line is the answer for Q1, the second for Q2, etc. No other characters, labels, punctuation, or explanation. No blank lines.\n" \
                     "7. Do NOT provide chain-of-thought or any additional text.\n"

        if self.reasoning == "none":
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "developer", "content": system_msg},
                    {"role": "user", "content": prompt}],
                max_output_tokens=self.max_output_tokens
            )
        else:
            response = self.client.responses.create(
                model=self.model_name,
                reasoning={"effort": self.reasoning},
                input=[
                    {"role": "developer", "content": system_msg},
                    {"role": "user", "content": prompt}],
                max_output_tokens=self.max_output_tokens
            )

        return response.output_text, response.usage.input_tokens, response.usage.output_tokens
    

    def query_batch(self, prompts: str, temperature: float = 0.7):
        """
        Method for querying the OpenAI LLM with a prompt in batch mode.
        :param prompts: List of input prompts to send to the LLM.
        :param max_tokens: Maximum number of tokens to generate in the response.
        :param temperature: Sampling temperature for response generation.
        :return: List of generated responses from the LLM.
        """

        system_msg = "You are a helpful assistant that disambiguates entities in a text based on a number of options. You will receive a text and a number of multiple choice questions. Only respond with the number of your selected answer for each multiple choice question. One answer per line. Do not add any additional text or explanations."
    
        lines_formatted = []
        for idx, prompt in enumerate(prompts):
            line = {"custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body":{"model": self.model_name,
                            "input": [
                                {"role": "developer", "content": system_msg},
                                {"role": "user", "content": prompt}],
                            "max_output_tokens": self.max_output_tokens
                    }
            }
            lines_formatted.append(line)
        
        # Save the formatted lines to a file
        with open("./stored/batch_requests.jsonl", "w") as f:
            for line in lines_formatted:
                f.write(json.dumps(line) + "\n")

        # Upload file to the OpenAI API
        batch_input_file = self.client.files.create(
            file=open("./stored/batch_requests.jsonl", "rb"),
            purpose="batch"
        )

        # Start Batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": "Batch processing for entity disambiguation",
            }
        )
        batch_id = batch.id
        print(f"OpenAI | Batch submitted with ID: {batch_id}, Waiting for completion...")
        print(f"OpenAI | Batch status will be checked every 3 minutes.")

        waited_mins = 0
        while True:
            time.sleep(180)  # Wait for the batch to be processed
            waited_mins += 3
            # Get Batch object
            batch = self.client.batches.retrieve(batch_id)
            # Check status of object
            if batch.status == "completed":
                print(f"OpenAI | {waited_mins} mins | Batch completed successfully.")
                break
            elif batch.status in ["failed", "cancelled", "cancelling", "expired"]:
                print(f"OpenAI | {waited_mins} mins | Batch failed or was cancelled.")
                print("Execution failed. Exiting...")
                sys.exit(1)
            else:
                print(f"OpenAI | {waited_mins} mins | Batch status: {batch.status}. Waiting for completion...")

        file_response = self.client.files.content(batch.output_file_id).text

        # Verify that all prompts have been answered
        assert len(file_response.splitlines()) == len(prompts), \
            f"Number of responses {len(file_response.splitlines())} does not match number of prompts {len(prompts)}."

        # LLM responses may not be in order. Sort using custom_id field of each dictionary in the response
        responses_dict = [json.loads(line) for line in file_response.splitlines()]
        responses_dict.sort(key=lambda x: x["custom_id"])

        responses = []
        for response in responses_dict:
            print(response) # DEBUG
            responses.append(response["response"]["body"]["choices"][0]["message"]["content"].strip())

        return responses



class LLMDualEncoderEntityDisambiguation(DualEncoderEntityDisambiguation):
    """
    This class extends DualEncoderEntityDisambiguation to integrate large language models (LLMs) for entity disambiguation.
    It enables hybrid prediction workflows, where LLMs can be used to refine or override dual encoder predictions,
    supports multiple LLM providers, and offers flexible strategies for span selection and iterative LLM querying.
    """

    def __init__(
            self,
            LLM_model_type: str = ["HU"],
            LLM_model_name: str = ["HU-1"],
            local_llm: LLM = None,
            top_k: int = 5,
            dynamic_top_k: bool = False,
            LLM_strategy: str = "default",
            LLM_selection_strategy: str = "similarity",
            LLM_verbalization_strategy: str = "description",
            LLM_verbalization_candidate_strategy: str = "description",
            threshold: float = 0.7,
            iterations: int = 3,
            api_key: str = None,
            batching: bool = False,
            regenerate_sentences: bool = False,
            reasoning: str = ["none"],
            num_agents: int = 1,
            verbalization_data: dict = None,
            **kwargs
    ):
        super(LLMDualEncoderEntityDisambiguation, self).__init__(**kwargs)

        self.LLM_model_type = LLM_model_type        # Type of the LLM model (e.g., OpenAI, HU)
        self.LLM_model_name = LLM_model_name        # Name of the LLM model to use (e.g., OpenAI-o4, HU-1, HU-2)
        self.local_llm = local_llm                  # Local LLM instance if using a local model
        self.top_k = top_k                          # Number of top predictions to consider
        self.dynamic_top_k = dynamic_top_k          # Whether to use dynamic top-k based on the span score
        self.LLM_selection_strategy = LLM_selection_strategy  # Strategy for selecting difficult cases (e.g., similarity, difference, both). Append _abs for absolute thresholds (e.g., similarity_abs, difference_abs, both_abs)
        self.LLM_strategy = LLM_strategy            # Strategy for using the LLM (e.g., default, all)
        self.threshold = threshold                  # Threshold for filtering predictions if strategy is set to default
        self.iterations = iterations                # Number of LLM prediction rounds
        self.api_key = api_key                      # API key for accessing the LLM service, required for OpenAI-o4
        self.batching = batching                    # Whether to use batching for LLM predictions
        self.regenerate_sentences = regenerate_sentences    # Whether to regenerate sentences or load them from disk
        self.reasoning = reasoning                  # How much reasoning should be used for reasoning models (low, medium, high)
        self.num_agents = num_agents                # Number of LLM agents to use for LLM predictions. Default: 1
        self.LLM_verbalization_strategy = LLM_verbalization_strategy  # Strategy for verbalizing labels (e.g., description, categories, both, none)
        self.LLM_verbalization_candidate_strategy = LLM_verbalization_candidate_strategy  # Strategy for verbalizing candidate labels (e.g., description, categories, both, none)
        self.verbalization_data = verbalization_data  # Additional data for verbalizing labels containing categories

        # Validate Inputs
        if not self.LLM_strategy in ["default", "all"]:
            raise NotImplementedError(
                f"LLM strategy {self.LLM_strategy} is not supported. Supported strategies are: default, all."
            )
        if (self.threshold < 0.0 or self.threshold > 1.0) and not "abs" in self.LLM_selection_strategy:
            raise ValueError(
                f"Threshold {threshold} is not valid. It should be between 0.0 and 1.0 if not using absolute thresholds."
            )
        if self.iterations < 1:
            raise ValueError(
                f"Iterations {self.iterations} must be at least 1."
            )
        if self.LLM_model_type in ["OpenAI", "Google"] and not api_key:
            raise ValueError(
                "API key is required for OpenAI and Google models. Please provide a valid API key."
            )

    def save_sentences(self, sentences, label_name: str = "predicted", LLM_label_name: str = "LLMPrediction"):
        sentence_dict = {}
        for i, sentence in enumerate(sentences):
            sentence_dict[i] = {}
            for j, span in enumerate(sentence.get_spans()):
                labels = {}
                labels["text"] = span.text
                labels["start_position"] = span.start_position
                labels["end_position"] = span.end_position

                # Get Labels for the span
                if span.has_label(label_name):
                    labels[label_name] = (span.get_label(label_name).value, span.get_label(label_name).score)
                
                k = 0
                while(span.has_label(f"top_{k}")):
                    top_label = f"top_{k}"
                    labels[top_label] = (span.get_label(top_label).value, span.get_label(top_label).score)
                    k += 1

                if span.has_label(LLM_label_name):
                    labels[LLM_label_name] = (span.get_label(LLM_label_name).value, span.get_label(LLM_label_name).score)

                # Save Span data to dict
                sentence_dict[i][j] = labels
        return sentence_dict
    
    def load_sentences(self, sentences, sentence_dict):
        for i, sentence in enumerate(sentences):
            for j, span in enumerate(sentence.get_spans()):
                # Check if the spans match
                try:
                    assert span.text == sentence_dict[i][j]["text"] and \
                           span.start_position == sentence_dict[i][j]["start_position"] and \
                           span.end_position == sentence_dict[i][j]["end_position"]
                except AssertionError:
                    print(f"Span mismatch at sentence {i}, span {j}:")
                    print(f"Expected: {sentence_dict[i][j]}")
                    print(f"Got: text={span.text}, start={span.start_position}, end={span.end_position}")
                    return False

                # Set Labels for the Span
                for key in sentence_dict[i][j]:
                    if key in ["text", "start_position", "end_position", "LLMPrediction", "LLM_start_iteration", "LLM_predict_iteration"]:
                        continue
                    span.set_label(typename=key,
                                   value=sentence_dict[i][j][key][0],
                                   score=sentence_dict[i][j][key][1])
        
        return True

    def split_spans_based_on_threshold(self, sentences, label_name: str = "predicted"):
        """
        Splits spans based on a threshold for the predicted labels.
        :param sentences: List of sentences to process.
        :param label_name: The label type to check against the threshold.
        :return: A list of sentences with spans split based on the threshold.
        """


        # Sort spans from all sentences based on the threshold into one big list
        spans = []
        for s in sentences:
            spans.extend([sp for sp in s.get_spans(label_name)])

        # Sort spans based on LLM selection strategy
        if self.LLM_selection_strategy in ["similarity", "similarity_abs"]:
            # Sort spans based on score in descending order
            spans = sorted(spans, key=lambda sp: sp.get_label(label_name).score, reverse=True)
        elif self.LLM_selection_strategy in ["difference", "difference_abs"]:
            # Sort spans based on the difference between top_0 and top_1 in descending order
            spans = sorted(spans, key=lambda sp: sp.get_label(f"top_0").score - sp.get_label(f"top_1").score, reverse=True)
        elif self.LLM_selection_strategy in ["both", "both_abs"]:
            spans1 = sorted(spans, key=lambda sp: sp.get_label(label_name).score, reverse=True)
            spans2 = sorted(spans, key=lambda sp: sp.get_label(f"top_0").score - sp.get_label(f"top_1").score, reverse=True)
        
            # Combine both lists, ensuring no duplicates, in alternating order
            spans = []
            span_list = [spans1, spans2]
            span_list_idx = 0
            i = 0
            while i < max(len(spans1), len(spans2)):
                # Get span from current span_list
                selected_span = span_list[span_list_idx][i] if i < len(span_list[span_list_idx]) else None

                # If there is not span at the current index, switch to the next span_list
                # If we go back to the first span_list, increment the index
                if not selected_span:
                    span_list_idx = (span_list_idx + 1) % 2
                    if span_list_idx == 0:
                        i += 1
                    continue
                
                # Check if the selected span is already in the spans list
                if selected_span not in spans:
                    spans.append(selected_span)

                # Switch to the next span_list for the next iteration
                # If we go back to the first span_list, increment the index
                span_list_idx = (span_list_idx + 1) % 2
                if span_list_idx == 0:
                    i += 1

            # Spans are now sorted in alternating order from both lists
        else:
            raise NotImplementedError(
                f"LLM selection strategy {self.LLM_selection_strategy} is not supported. Supported strategies are: similarity, difference, both."
            )
        
        # If the strategy is "all", we do not filter based on threshold
        if self.LLM_strategy == "all":
            threshold = 0.0
        else:
            threshold = self.threshold

        # Add label to each span indicating which percentage it is
        # This helps for later evaluation
        for idx, span in enumerate(spans):
            span.set_label(typename="span_percentage", value = floor((idx / len(spans)) * 100), score = 0)

        # Get top threshold percent of spans (those with the best scores)
        if "abs" not in self.LLM_selection_strategy:
            cutoff = floor(len(spans) * threshold)
            spans_verbalization = spans[0:cutoff]   # Spans that will be verbalized
            spans_LLM = spans[cutoff:]              # Spans that will be processed by the LLM
        else:
            spans_verbalization, spans_LLM = [], []
            for span in spans:
                if self.LLM_selection_strategy == "similarity_abs":
                    if span.get_label(label_name).score >= threshold:
                        spans_verbalization.append(span)
                    else:
                        spans_LLM.append(span)
                elif self.LLM_selection_strategy == "difference_abs":
                    if abs(span.get_label(f"top_0").score - span.get_label(f"top_1").score) >= threshold:
                        spans_verbalization.append(span)
                    else:
                        spans_LLM.append(span)
                else:
                    raise NotImplementedError(
                        f"LLM selection strategy {self.LLM_selection_strategy} is not supported. Supported strategies are: similarity_abs, difference_abs."
                    )


        # Split spans_LLM into multiple lists based on the number of iterations
        spans_LLM_batches = []
        batch_size = ceil(len(spans_LLM) / self.iterations)
        for i in range(self.iterations):
            start_index = i * batch_size
            end_index = start_index + batch_size
            spans_LLM_batches.append(spans_LLM[start_index:min(end_index, len(spans_LLM))])  # Ensure we don't go out of bounds

        return spans_verbalization, spans_LLM_batches

    def get_label_category_information(self, label: str) -> str:
        """
        Retrieves category information for a given label from the verbalization data.
        :param label: The label to retrieve category information for.
        :return: A string containing the category information.
        """
        if not self.verbalization_data:
            raise ValueError("Verbalization data is not defined. Please set the verbalization_data attribute before calling this method.")

        if label not in self.verbalization_data:
            raise NotImplementedError(f"Label {label} not found in verbalization data.")

        categories = {'instance of': [], 'part of': [], 'country': [], 'occupation': [], 'subclass of': []}
        data = self.verbalization_data[label].get('wikidata_properties', [])

        for key, value in data:
            if key in categories:
                categories[key].append(value)

        category_info = ""
        for key, values in categories.items():
            if values:
                if category_info != "":
                    category_info += " ; "
                category_info += f"{key}: {', '.join(values)}"

        return category_info

    def get_label_excerpt(self, label: str) -> str:
        """
        Retrieves an excerpt for a given label from Wikipedia.
        :param label: The label to retrieve the excerpt for.
        :return: A string containing the excerpt.
        """
        raise NotImplementedError("get_label_excerpt() method is not yet implemented.")

        return excerpt
    
    def label_to_verbalization(self, label: str, parse: bool = True, candidate: bool = False) -> str:
        """
        Converts a label to its verbalization based on the label map.
        :param label: The label to convert.
        :param parse: Whether to parse the verbalization string (default is True). This removes any prefix before a semicolon.
        :param candidate: Whether the label is for a candidate (default is False).
        :return: The verbalization of the label.
        """
        if not self.label_map:
            raise ValueError("Label map is not defined. Please set the label_map attribute before calling this method.")
        
        strategy = self.LLM_verbalization_strategy if not candidate else self.LLM_verbalization_candidate_strategy

        if strategy in ["description", "none"]:
            verbalization = self.label_map.get(label, label.replace("_", " "))
            if parse:
                verbalization = verbalization.split(";", 1)[1].strip() if ";" in verbalization else verbalization
        elif strategy == "categories":
            verbalization = self.get_label_category_information(label)
        elif strategy == "description+categories":
            verbalization = self.label_map.get(label, label.replace("_", " "))
            if parse:
                verbalization = verbalization.split(";", 1)[1].strip() if ";" in verbalization else verbalization
            verbalization += "; " + self.get_label_category_information(label)
        elif strategy == "excerpt":
            verbalization = "Descriptive Excerpt: " + self.get_label_excerpt(label)
        else:
            raise NotImplementedError(
                f"LLM verbalization strategy {strategy} is not supported. Supported strategies are: description, categories, description+categories, excerpt, none."
            )

        return verbalization

    def add_verbalizations_to_sentences(self, sentences, sentences_data, spans_verbalization, label_name: str = "predicted", iter_count: int = 0):
        """
        Adds verbalizations of spans to the sentences.
        :param sentences: List of sentences to modify.
        :param spans_verbalization: List of spans to verbalize.
        :param label_name: The label type to use for verbalization.
        """

        # Iterate over all spans in all sentences
        for idx, sentence in enumerate(sentences):
            for sp in sentence.get_spans():
                # If the span is in the spans_verbalization, add the verbalization
                if sp in spans_verbalization and self.LLM_verbalization_strategy != "none":
                    if not sp.has_label(label_name):
                        continue
                    verbalization = self.label_to_verbalization(sp.get_label(label_name).value)
                    
                    # Add the verbalization to the sentences_data dictionary (stored by insert position of the verbalization)
                    sentences_data[idx]["verbalizations"][sp.end_position] = {"verbalization": verbalization, "iteration": iter_count}
            
            # Sort the verbalizations in place by their end position in the sentence in descending order
            sentences_data[idx]["verbalizations"] = dict(sorted(sentences_data[idx]["verbalizations"].items(), key=lambda x: x[0], reverse=True))

    def parse_LLM_response(self, response, expected_length: int = None):
        """
        Parses the response from the LLM.
        :param response: The response from the LLM.
        :param expected_length: The expected number of answers in the response (optional).
        :return: Parsed response as a list of integers.
        """
        parsed = []
        for line in response.split("\n"):
            # Identify first whole number in the line
            match = re.findall(r'\d+', line.strip())
            if match:
                parsed.append(int(match[-1]))  # Take the last matched number in the line

        # Attempt more advanced parsing if the length does not match:
        if expected_length and len(parsed) != expected_length:
            # Go up from bottom of response and collect numbers until expected length is reached
            # If at one point more than one number is found in a line, stop parsing
            adv_parsed = []
            for line in list(reversed(response.split("\n"))):
                # Identify all numbers in line
                match = re.findall(r'\d+', line.strip())
                # Check if only one match exists
                # If yes, add it to the parsed list
                if match:
                    if len(match) == 1:
                        adv_parsed.append(int(match[0]))
                    else:
                        break
                        
                if len(adv_parsed) == expected_length:
                    parsed = list(reversed(adv_parsed))
                    break

        return parsed

            


    def generate_prompt(self, sentence_data, spans, label_name, top_k, iteration_count, mappings, allow_extra_iteration):
        """
        Generates a prompt for the LLM based on the sentence and its spans.
        :param sentence: The sentence to process.
        :param sentence_data: Data containing verbalizations and other information.
        :param spans: List of spans to include in the prompt.
        :param label_name: The label type to use for generating the prompt.
        :return: A formatted string prompt for the LLM.
        """
        final_iteration = True if iteration_count == self.iterations + allow_extra_iteration - 1 else False
        prompt = f"Consider the following text:\n\n"

        sentence_text = sentence_data["text"]

        # Add verbalizations to the sentence text from back to front
        # This is done to avoid messing up the positions of the verbalizations when inserting them
        to_insert = {}
        for end_position, verbalization in sentence_data["verbalizations"].items():
            to_insert[end_position] = {"insert_type": "Verbalization", "text": verbalization["verbalization"]}
        for idx, span in enumerate(spans):
            to_insert[span.end_position] = {"insert_type": "Span", "text": f"**Q{idx+1}**"}

        # Sort to_insert by end position in descending order
        to_insert = dict(sorted(to_insert.items(), key=lambda x: x[0], reverse=True))
        # Insert verbalizations and spans into the sentence text
        for end_position, insert_data in to_insert.items():
            if insert_data["insert_type"] == "Verbalization":
                sentence_text = sentence_text[:end_position] + " (" + insert_data["text"] + ")" + sentence_text[end_position:]
            elif insert_data["insert_type"] == "Span":
                sentence_text = sentence_text[:end_position] + " [" + insert_data["text"] + "]" + sentence_text[end_position:]
        
        # Add the modified sentence text to the prompt
        prompt += f"{sentence_text}\n\n----------\n\n"

        for idx, span in enumerate(spans):
            if not final_iteration:
                span_text = f"Which entry does the mention '{span.text}' at [**Q{idx+1}**] refer to? Evaluate all options carefully.\n\n"
            else:
                span_text = f"Which entry does the mention '{span.text}' at [**Q{idx+1}**] refer to? Evaluate all options carefully. If you are unsure, select the answer you think fits the most.\n\n"

            span_text += f"(0) None of the answers below or Unsure\n" if not final_iteration else f"(0) None of the answers below\n"

            for i in range(len(mappings[idx])):
                # Check if top_k label exists
                if not span.has_label(f"top_{mappings[idx][i]}"):
                    print(f"Warning | Span {span.text} does not have label 'top_{mappings[idx][i]}'. Please check that at least k candidates are available for each span.")
                
                label_candidate = span.get_label(f"top_{mappings[idx][i]}").value

                # Check if this candidate is restricted. Only occurs when multiple agents are used and there were ties in the previous iteration
                if span.has_label("restrict_choices"):
                    if not label_candidate in span.get_label("restrict_choices").value:
                        # Only choices that are white-listed in the restrict_choices label are allowed
                        continue

                label_verbalization = self.label_to_verbalization(label_candidate, candidate=True)
                if self.LLM_verbalization_candidate_strategy != "none":
                    span_text += f"({i+1}) {label_candidate.replace('_', ' ')} - ({label_verbalization})\n"
                else:
                    span_text += f"({i+1}) {label_candidate.replace('_', ' ')}\n"

            prompt += f"{span_text}\n"

        return prompt

    def prompt_LLM(self, sentence, sentence_data, spans, label_name, LLM_label_name, client, top_k, iteration_count, random_candidate_order, allow_extra_iteration):
        """
        Prompts the LLM with the generated prompt and returns the response.
        :param sentence: The sentence to process.
        :param sentence_data: Data containing verbalizations and other information.
        :param spans: List of spans to include in the prompt.
        :param label_name: The label type to use for generating the prompt.
        :param client: The LLM client to use for querying.
        :param top_k: Number of top predictions to consider.
        :return: The response from the LLM.
        """
        # If top_k i set to -1 the number of answer options is dynamic based on the following dict:
        dyn_top_k = {-30: 20,
                        -29: 20,
                        -28: 20,
                        -27: 20,
                        -26: 15,
                        -25: 15,
                        -24: 10,
                        -23: 10,
                        -22: 5,
                        -21: 5,
                        -20: 5,
                        "default": 5}
        
        mappings = []
        for span in spans:
            # Determine the appropriate top_k for this span
            if self.dynamic_top_k:
                this_top_k = min(dyn_top_k.get(int(round(span.get_label("top_0").score, 0)), dyn_top_k["default"]), top_k)
            else:
                this_top_k = top_k
            span.set_label(typename="top_k_used", value=this_top_k)

            # Set the order of the candidates in the prompt
            this_mapping = list(range(this_top_k))
            if random_candidate_order:
                random.shuffle(this_mapping)
            mappings.append(this_mapping)

        # Generate the prompt
        prompt = self.generate_prompt(sentence_data, spans, label_name, top_k, iteration_count, mappings, allow_extra_iteration)
        # Query the LLM
        attempts = 1
        while True:
            try:
                if self.batching:
                    response = client.query_batch(prompt)
                    input_tokens, output_tokens = 0, 0
                else:
                    response, input_tokens, output_tokens = client.query(prompt)
                    
                # Evaluate Response
                parsed_response = self.parse_LLM_response(response, expected_length=len(spans))

            except Exception as e:
                print(f"Error occurred while querying LLM: {e} | Number of spans affected: {len(spans)}\n")
                input_tokens, output_tokens = 0, 0
                parsed_response = []


            if len(parsed_response) == len(spans):
                break
            elif attempts >= 10:
                print(f"Error: LLM response could not be parsed correctly after 10 attempts. Setting all spans to 0 (Not sure / Neither).")
                # Set parsed_response to all 0 (no prediction)
                parsed_response = [0] * len(spans)
                break
            attempts += 1
            time.sleep(0.5 * attempts)  # Increasing backoff


        # Add the LLM predictions to the spans in the sentence
        for i, span in enumerate(spans):
            # Check if LLM has made a prediction for this span
            if parsed_response[i] != 0:
                # Add label to span for predicted response
                # Carefully map the parsed response to the correct label based on the mapping
                try:
                    span.set_label(typename=LLM_label_name, value=span.get_label(f"top_{mappings[i][int(parsed_response[i]) - 1]}").value, score=span.get_label(f"top_{mappings[i][int(parsed_response[i]) - 1]}").score)
                except Exception as e:
                    print(f"Error occurred while setting label for span {i}.\nPrompt: {prompt}\nResponse: {response}, Parsed response: {parsed_response}, Mappings: {mappings[i]}. Error: {e}")

            # Add required tokens to span
            if not span.has_label("_input_tokens"):
                span.set_label(typename="_input_tokens", value=0, score=0.0)
            if not span.has_label("_output_tokens"):
                span.set_label(typename="_output_tokens", value=0, score=0.0)
            span.set_label(typename="_input_tokens", value=span.get_label("_input_tokens").value + int(input_tokens / len(spans)) , score=0.0)
            span.set_label(typename="_output_tokens", value=span.get_label("_output_tokens").value + int(output_tokens / len(spans)) , score=0.0)

    def perform_LLM_iteration(
            self,
            sentences,
            sentences_data,
            spans_LLM_this_batch,
            label_name,
            LLM_label_name,
            client,
            top_k,
            spans_per_prompt: int = 5,
            iteration_count: int = 0,
            random_candidate_order: bool = False,
            allow_extra_iteration: bool = False
    ):
        print(f"Iteration {iteration_count + 1} / {self.iterations} | Starting LLM predictions for {len(spans_LLM_this_batch)} spans...")

        self.timer.resume()
        threads_created = []
        # Iterate over all sentences
        for idx, sentence in enumerate(sentences):
            # Get spans from sentence which are also in spans_LLM_this_batch
            spans_in_sentence = [sp for sp in sentence.get_spans(label_name) if sp in spans_LLM_this_batch]

            # Select up to spans_per_sentence spans per prompt
            for i in range(0, ceil(len(spans_in_sentence) / spans_per_prompt)):
                selected_spans = spans_in_sentence[i * spans_per_prompt:min((i + 1) * spans_per_prompt, len(spans_in_sentence))]

                # Create a thread to process the spans with the LLM
                thread = threading.Thread(target=self.prompt_LLM, args=(sentence, sentences_data[idx], selected_spans, label_name, LLM_label_name, client, top_k, iteration_count, random_candidate_order, allow_extra_iteration))
                thread.start()
                threads_created.append(thread)

        # Wait for all threads to complete
        print(f"Iteration {iteration_count + 1} / {self.iterations} | LLM Predictions for {len(spans_LLM_this_batch)} spans in {len(threads_created)} prompts underway...")
        # Create a tqdm progress bar here with length equal to the number of created threads. The progress bar should update as each thread completes.
        # We check the status pf each thread in a loop until all threads are done.
        # The threads will be checked by using thread.is_alive() method.
        progress_bar = tqdm(total=len(threads_created), desc=f"LLM Predictions", unit="prompts", leave=False)
        while True:
            threads_complete = 0
            for thread in threads_created:
                if not thread.is_alive():
                    threads_complete += 1

            progress_bar.update(threads_complete - progress_bar.n)
            time.sleep(0.5)
            if threads_complete == len(threads_created):
                self.timer.pause()
                time.sleep(1)
                break



    def perform_LLM_iteration_batched(
            self,
            sentences,
            sentences_data,
            spans_LLM_this_batch,
            label_name,
            LLM_label_name,
            client,
            top_k,
            spans_per_prompt: int = 5,
            iteration_count: int = 0,
            random_candidate_order: bool = False,
            allow_extra_iteration: bool = False
    ):

        prompts = []
        spans_of_prompts = []
        all_mappings = []
        # Iterate over all sentences
        for idx, sentence in enumerate(sentences):
            # Get spans from sentence which are also in spans_LLM_this_batch
            spans_in_sentence = [sp for sp in sentence.get_spans(label_name) if sp in spans_LLM_this_batch]

            # Select up to spans_per_sentence spans per prompt
            for i in range(0, ceil(len(spans_in_sentence) / spans_per_prompt)):
                selected_spans = spans_in_sentence[i * spans_per_prompt:min((i + 1) * spans_per_prompt, len(spans_in_sentence))]

                mappings = [list(range(top_k)) for _ in range(len(selected_spans))]
                if random_candidate_order:
                    for mapping in mappings:
                        random.shuffle(mapping)

                # Create a prompt for the LLM
                prompts.append(self.generate_prompt(sentences_data[idx], selected_spans, label_name, top_k, iteration_count, mappings, allow_extra_iteration))
                spans_of_prompts.append(selected_spans)
                all_mappings.append(mappings)

        print(f"Iteration {iteration_count + 1} / {self.iterations} | Starting LLM predictions in batched mode for {len(spans_LLM_this_batch)} spans in {len(prompts)} prompts...")

        response_mappings = [i for i in range(len(prompts))]
        attempts = 1
        while attempts <= 5:
            # Give prompts to LLM
            self.timer.resume()
            responses, input_tokens, output_tokens = client.query_batch(prompts)
            self.timer.pause()

            # Initialize avriables for possible re-try
            all_valid = True
            new_prompts = []
            new_response_mappings = []

            # Parse and evaluate responses
            for i in range(len(responses)):
                parsed_response = self.parse_LLM_response(responses[i], expected_length=len(spans_of_prompts[response_mappings[i]]))

                # Check if parsed response is of the correct length. If not, set all to 0
                # Also add prompt to retry list
                if len(parsed_response) != len(spans_of_prompts[response_mappings[i]]):
                    parsed_response = [0] * len(spans_of_prompts[response_mappings[i]])

                    all_valid = False
                    new_prompts.append(prompts[i])
                    new_response_mappings.append(response_mappings[i])
                
                # Go through every span in prompt
                for j, span in enumerate(spans_of_prompts[response_mappings[i]]):
                    # Check if LLM has made a prediction for this span
                    if parsed_response[j] != 0:
                        # Add label to span for predicted response
                        try:
                            span.set_label(typename=LLM_label_name, value=span.get_label(f"top_{all_mappings[response_mappings[i]][j][int(parsed_response[j]) - 1]}").value, score=span.get_label(f"top_{all_mappings[response_mappings[i]][j][int(parsed_response[j]) - 1]}").score)
                        except Exception as e:
                            print(f"Error occurred while setting label for span {j}.\nPrompt: {prompts[i]}\nResponse: {responses[i]}, Parsed response: {parsed_response}, Mappings: {all_mappings[response_mappings[i]][j]}. Error: {e}")

                    # Add required tokens to span
                    if not span.has_label("_input_tokens"):
                        span.set_label(typename="_input_tokens", value=0, score=0.0)
                    if not span.has_label("_output_tokens"):
                        span.set_label(typename="_output_tokens", value=0, score=0.0)
                    span.set_label(typename="_input_tokens", value=span.get_label("_input_tokens").value + int(input_tokens[i] / len(spans_of_prompts[response_mappings[i]])) , score=0.0)
                    span.set_label(typename="_output_tokens", value=span.get_label("_output_tokens").value + int(output_tokens[i] / len(spans_of_prompts[response_mappings[i]])) , score=0.0)


            if all_valid:
                break
            elif attempts >= 5:
                break
            # Retry failed prompts after a short wait
            else:
                attempts += 1
                print(f"Iteration {iteration_count + 1} / {self.iterations} | Retrying {len(new_prompts)} (down from {len(prompts)}) prompts due to invalid LLM responses. Attempt {attempts} / 5.")
                if not self.local_llm:
                    time.sleep(0.5 * attempts)  # Increasing backoff for non-local LLMs
                prompts = new_prompts
                response_mappings = new_response_mappings
                



    def predict(
            self,
            sentences,
            mini_batch_size = 32,
            return_probabilities_for_all_classes = False,
            verbose = False,
            label_name = "predicted", # Label name which stores the predictions of the ML model.
            LLM_label_name = "LLMPrediction", # Label name which stores the predictions of the LLM.
            final_prediction_label_name = "final_prediction", # Label name which stores the final predictions
            return_loss=False,
            top_k: int = None, # Number of top predictions to save by the ML model and to consider by the LLM.
            embedding_storage_mode="none",
            return_span_and_label_hidden_states = True,
            spans_per_prompt: int = 5, # Maximum number of spans to include in a single prompt to the LLM.
            load_file: str = None, # File path where saved sentences are stored. If None, the default is "stored/sentences.pkl".
            allow_pred_skip: bool = True, # Allows the ML model to be skipped if the sentences are already predicted and saved to disk.
            allow_LLM_skip: bool = False,
            disable_LLM: bool = False, # Disables the LLM step if set to true. Can be used to predict sentences and save them to a disk without calling the LLM
            rate_limit_timeout: int = 5, # Time in seconds to wait before retrying the LLM query if rate limit is reached. Set to 0 to disable rate limiting.
            max_output_tokens: int = 200, # Maximum number of tokens to generate in the response from the LLM.
            random_candidate_order: bool = False, # If set to True, the candidates in the prompt will be shuffled before sending to the LLM.
            allow_extra_iteration: bool = False, # If set to True, the LLM will be allowed to make extra iteration for entities it was unsure about in the final iteration.
            **kwargs
        ):
        """
        Predict labels for sentences using the LLM strategy.
        :param sentences: List of sentences to process.
        :param mini_batch_size: Size of the mini-batch for processing.
        :param return_probabilities_for_all_classes: Whether to return probabilities for all classes.
        :param verbose: Whether to print verbose output.
        :param label_name: The label type to use for predictions.
        :param return_loss: Whether to return the loss.
        :param top_k: Number of top predictions to consider.
        :param embedding_storage_mode: Mode for storing embeddings.
        :param return_span_and_label_hidden_states: Whether to return hidden states for spans and labels
        :return: None, but modifies the sentences in place with predictions.
        """

        # Use self.top_k if top_k is not provided
        if top_k is None or top_k < self.top_k:
            top_k = self.top_k

        if load_file is None:
            load_file = f"sentence_cache/sentences.pkl"

        # Check if pickle has saved the sentences object
        skip_preds = False
        skip_LLM = False
        if os.path.exists(load_file) and not self.regenerate_sentences:
            print(f"Loading sentences from {load_file}...")
            with open(load_file, "rb") as f:
                try:
                    sentence_dict = pickle.load(f)
                    loss = pickle.load(f)
                    if not self.load_sentences(sentences, sentence_dict):
                        raise AssertionError("Failed to load sentences from disk. Continuing with predictions...")
                    skip_preds = True
                    if "Data" in sentence_dict:
                        if "LLM" in sentence_dict["Data"]:
                            skip_LLM = True
                    print(f"Loaded {len(sentences)} sentences from disk.")
                except Exception as e:
                    pass


        # Get predictions from the Dual Encoder
        if not skip_preds or not allow_pred_skip:
            print("Predicting with LLMDualEncoderEntityDisambiguation...")
            loss = super(LLMDualEncoderEntityDisambiguation, self).predict(
                sentences,
                mini_batch_size=mini_batch_size,
                return_probabilities_for_all_classes=return_probabilities_for_all_classes,
                verbose=verbose,
                label_name=label_name,
                return_loss=return_loss,
                top_k=20,
                embedding_storage_mode=embedding_storage_mode,
                return_span_and_label_hidden_states=return_span_and_label_hidden_states
            )

        # Save sentences object to disk so we don't have to re-predict them on subsequent runs
        # This is useful for debugging and testing purposes
        if not skip_preds:
            with open(load_file, "wb") as f:
                print("Saving sentences to disk...")
                sentence_dict = self.save_sentences(sentences, label_name, LLM_label_name)
                sentence_dict["Data"] = ["Dual_Encoder"]
                pickle.dump(sentence_dict, f)
                pickle.dump(loss, f)
                print(f"Sentences have been saved to disk as {load_file}. You can load them later to skip predictions.")


        # Select spans for LLM processing based on LLM selection straegy and split them into verbalization and LLM processing spans
        spans_verbalization, spans_LLM_batches = self.split_spans_based_on_threshold(
            sentences,
            label_name
        )

        # If we allow an extra iteration, add empty list for extra iteration to spans_LLM_batches
        if allow_extra_iteration:
            spans_LLM_batches.append([])

        # Add verbalization of spans in spans_verbalization to sentences
        sentences_data = {}
        for idx, sentence in enumerate(sentences):
            sentences_data[idx] = {"text" : sentence.text, "verbalizations" : {}}
        self.add_verbalizations_to_sentences(sentences, sentences_data, spans_verbalization, label_name, iter_count = 0)

        if (skip_LLM and allow_LLM_skip):
            print("Skipping LLM predictions because they were already performed in a previous run.")
            return loss if return_loss else None
        elif disable_LLM:
            print("Skipping LLM Predictions because they are disabled.")
            return loss if return_loss else None
        
        print("Starting LLM predictions...")
        print(f"Using model {self.LLM_model_name} by {self.LLM_model_type} with strategy '{self.LLM_strategy}'.")

        # Get LLM model instance
        clients = []
        for i in range(self.num_agents):
            if self.LLM_model_type[i] == "OpenAI":
                clients.append(OpenAILLM(model_name=self.LLM_model_name[i], api_key=self.api_key["OpenAI"], max_output_tokens=max_output_tokens, reasoning=self.reasoning[i]))
            elif self.LLM_model_type[i] == "Google":
                clients.append(GoogleLLM(model_name=self.LLM_model_name[i], api_key=self.api_key["Google"], max_output_tokens=max_output_tokens, reasoning=self.reasoning[i]))
            elif self.LLM_model_type[i] == "HU":
                clients.append(GradioLLM(model_name=self.LLM_model_name[i]))
            elif self.LLM_model_type[i] == "LOCAL":
                clients.append(LocalLLM(llm=self.local_llm, max_output_tokens=max_output_tokens))
            else:
                raise NotImplementedError(f"LLM model type {self.LLM_model_type[i]} is not supported. Supported options are: OpenAI, Google, HU.")
            
        # ----------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------ BEGIN LLM ITERATIONS ------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------

        # Perform iterations of LLM predictions
        for iteration_count in range(self.iterations + allow_extra_iteration):
            # Select the spans for the current iteration
            spans_LLM_this_batch = spans_LLM_batches[iteration_count]

            # Add label 'LLM_start_iteration' to all spans in this batch with the current iteration number
            for span in spans_LLM_this_batch:
                if not span.has_label("LLM_start_iteration"):
                    span.set_label(typename="LLM_start_iteration", value=iteration_count+1, score=0.0)


            # ----------------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------- GENERATE PROMPT AND PROMPT LLM -----------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------------------------------

            # Iterate over each agent. By default, there is only one agent
            for i in range(self.num_agents):
                print(f"Iteration {iteration_count + 1} / {self.iterations} | Starting Agent {i+1}/{self.num_agents}") if self.num_agents > 1 else None

                # Perform the LLM iteration
                if (not self.batching) and (self.LLM_model_type[i] != "LOCAL"):
                    self.perform_LLM_iteration(sentences, sentences_data, spans_LLM_this_batch, label_name, f"{LLM_label_name}_{i}", clients[i], top_k, spans_per_prompt, iteration_count, random_candidate_order, allow_extra_iteration)
                else:
                    self.perform_LLM_iteration_batched(sentences, sentences_data, spans_LLM_this_batch, label_name, f"{LLM_label_name}_{i}", clients[i], top_k, spans_per_prompt, iteration_count, random_candidate_order, allow_extra_iteration)

                print(f"Iteration {iteration_count + 1} / {self.iterations} | Agent {i+1}/{self.num_agents} Finished") if self.num_agents > 1 else None

            # ----------------------------------------------------------------------------------------------------------------------------------
            # -------------------------------------- MULTI-AGENT MAJORITY VOTE LOGIC -----------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------------------------------

            # Select the final LLM Prediction label based on the agents predictions
            for span in spans_LLM_this_batch:
                # If there is only one agent, use its prediction. If the agent voted 'None of the above', the label will not be set
                if self.num_agents == 1:
                    if span.has_label(f"{LLM_label_name}_0"):
                        span.set_label(typename=LLM_label_name, value=span.get_label(f"{LLM_label_name}_0").value, score=span.get_label(f"{LLM_label_name}_0").score)
                        span.remove_labels(f"{LLM_label_name}_0")
                    continue

                # In case we have multiple agents, we perform a majority vote:
                # 1. Gather labels of all agents and the amount of votes for each label
                labels = defaultdict(list)
                for i in range(self.num_agents):
                    label_key = f"{LLM_label_name}_{i}"
                    if span.has_label(label_key):
                        labels[span.get_label(label_key).value].append(i)
                    else:
                        labels["N/A - None of the above"].append(i)

                # 2. Select the label(s) with the most votes
                max_votes = max(len(votes) for votes in labels.values())
                most_voted_label = [(label, votes) for label, votes in labels.items() if len(votes) == max_votes] # structure: [(value, [idx's of agents who voted for it])]

                # 3. Remove choice restrictions from the previous iteration
                span.remove_labels(typename="restrict_choices") # Remove any previous restrictions

                # 4. Pick the winner / Restrict LLM choices in the next iteration
                if len(most_voted_label) == 1:
                    # If there is a clear winner (only one label with the most votes), set the LLM_label to it (unless it is 'None of the above')
                    if most_voted_label[0][0] != "N/A - None of the above":
                        span.set_label(typename=LLM_label_name, value=span.get_label(f"{LLM_label_name}_{most_voted_label[0][1][0]}").value, score=span.get_label(f"{LLM_label_name}_{most_voted_label[0][1][0]}").score)
                else:
                    # If there are multiple labels with the same number of votes
                    if iteration_count == (self.iterations + allow_extra_iteration) - 1:
                        # If this is the final iteration, randomly pick amongst the most voted labels
                        # We do not select 'None of the above' as this is guaranteed wrong
                        while True:
                            random_label = random.choice(most_voted_label) # structure: (value, [idx's of agents who voted for it])
                            if random_label[0] != "N/A - None of the above":
                                span.set_label(typename=LLM_label_name, value=span.get_label(f"{LLM_label_name}_{random_label[1][0]}").value, score=span.get_label(f"{LLM_label_name}_{random_label[1][0]}").score)
                                break
                    else:
                        # If this is not the final iteration, the span will be predicted again in the next iteration as we don't set a definitive label
                        # We make a note of the most voted labels and restrict the LLM to only choose between these labels in the next iteration
                        # If 'None of the above' is in most_voted_labels, the LLM will not be restricted
                        if "N/A - None of the above" not in [label[0] for label in most_voted_label]:
                            span.set_label(typename="restrict_choices", value=[label[0] for label in most_voted_label], score=0.0)

                # 5. Finally, remove temporary agent prediction labels from the span
                for i in range(self.num_agents):
                    # Note: It is not necessary to check if the label exists, as the remove_labels method will not raise an error if the label does not exist
                    span.remove_labels(f"{LLM_label_name}_{i}")

            # ----------------------------------------------------------------------------------------------------------------------------------
            # ---------------------------------- ADD VERBALIZATIONS OF PREDICTED SPANS ---------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------------------------------

            # Add verbalizations to sentences based on the LLM predictions
            print(f"Iteration {iteration_count + 1} / {self.iterations} | Adding verbalizations to sentences...")
            self.add_verbalizations_to_sentences(sentences, sentences_data, spans_LLM_this_batch, label_name=LLM_label_name, iter_count=iteration_count + 1)

            # ----------------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------ ADD FAILED SPANS TO NEXT ITERATION ----------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------------------------------

            # Add spans where the LLM has predicted "None of the above or Unsure" to the next iteration again (if there is one)
            # Also set the label 'LLM_predict_iteration' for all spans that have been predicted by the LLM in this iteration
            spans_failed = 0
            for span in spans_LLM_this_batch:
                if not span.has_label(LLM_label_name):
                    spans_failed += 1
                    if iteration_count < (self.iterations + allow_extra_iteration) - 1:
                        # If this is not the final iteration, retry the span in the next iteration
                        spans_LLM_batches[iteration_count + 1].append(span)
                    else:
                        # If this is the final iteration, set the LLM prediction to "None of the above"
                        # Also note, that the LLM has failed to make a prediction in any iteration
                        span.set_label(LLM_label_name, value="None of the above", score=0.0)
                        span.set_label(typename="LLM_predict_iteration", value=-1, score=0.0)
                else:
                    # If LLM has made a prediction in this iteration, add label 'LLM_predict_iteration' with current iteration for the span
                    span.set_label(typename="LLM_predict_iteration", value=iteration_count+1, score=0.0)
                
            if spans_failed > 0:
                print(f"Notice | LLM Prediction was 'None of the above' or unsuccessful for {spans_failed} mentions.{' Retrying in the next iteration.' if iteration_count < (self.iterations + allow_extra_iteration) - 1 else ''}")

        
            print(f"Iteration {iteration_count + 1} / {self.iterations} | Finished")

            if rate_limit_timeout > 0:
                for i in tqdm(range(rate_limit_timeout), leave=False, desc=f"Waiting for {rate_limit_timeout} seconds to avoid rate limiting"):
                    time.sleep(1)


        print("LLM Predictions completed!\n")
        with open(load_file, "wb") as f:
             print("Saving sentences to disk...")
             sentence_dict = self.save_sentences(sentences, label_name, LLM_label_name)
             sentence_dict["Data"] = ["Dual_Encoder", "LLM"]
             pickle.dump(sentence_dict, f)
             pickle.dump(loss, f)
             print(f"Sentences have been saved to disk as {load_file}. You can load them later to skip predictions.")

        # DEBUG
        changes_made_by_lmm = 0
        for iteration in range(self.iterations):
            for span in spans_LLM_batches[iteration]:
                if span.get_label(LLM_label_name).value != span.get_label(label_name).value:
                    changes_made_by_lmm += 1
        print(f"INFO | LLM made {changes_made_by_lmm} changes to the predictions of the Dual Encoder model.")
        # DEBUG

        # Add final prediction label to all spans
        # Note: This is currently unused
        for sentence in sentences:
            for span in sentence.get_spans():
                if span.has_label(LLM_label_name) and span.get_label(LLM_label_name).value != "None of the above":
                    # Set the final prediction label to the LLM prediction
                    span.set_label(typename=final_prediction_label_name, value=span.get_label(LLM_label_name).value, score=span.get_label(LLM_label_name).score)
                else:
                    # If no LLM prediction was made or the LLM predicted None of the above, use the Dual Encoder prediction instead
                    span.set_label(typename=final_prediction_label_name, value=span.get_label(label_name).value, score=span.get_label(label_name).score)

        return loss if return_loss else None
    
    def _print_predictions(self, batch, gold_label_type, label_name: str = "predicted", LLM_label_name: str = "LLMPrediction"):
        lines = []
        for datapoint in batch:
            eval_line = f"\n{datapoint.to_original_text()}\n"

            for span in datapoint.get_spans(gold_label_type):
                pred = span.get_label(label_name).value
                predLLM = span.get_label(LLM_label_name).value if span.has_label(LLM_label_name) else "N/A"
                symbol = "✓" if span.get_label(gold_label_type).value == pred else "❌"
                symbol_LLM = "✓" if span.get_label(gold_label_type).value == predLLM else "❌"
                eval_line += (
                    f'"{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --- {pred} --- {predLLM}'
                    f' --- {[(span.get_label(f"top_{i}").value, span.get_label(f"top_{i}").score) for i in range(span.get_label("top_k_used").value if span.has_label("top_k_used") else self.top_k)]}'
                    f' --- {span.get_label("LLM_start_iteration").value if span.has_label("LLM_start_iteration") else "N/A"}, {span.get_label("LLM_predict_iteration").value if span.has_label("LLM_predict_iteration") else "N/A"}'
                    f' --- {span.get_label("_input_tokens").value if span.has_label("_input_tokens") else 0}, {span.get_label("_output_tokens").value if span.has_label("_output_tokens") else 0}'
                    f' --- {span.get_label("span_percentage").value if span.has_label("span_percentage") else 0}'
                    f' --- {span.get_label("top_k_used").value if span.has_label("top_k_used") else 0}\n'
                )

            lines.append(eval_line)

        return lines
