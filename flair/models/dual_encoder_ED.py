import sys
import logging
import random
from math import ceil, floor

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

log = logging.getLogger("flair")


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
            embedding_storage_mode="none",
            return_span_and_label_hidden_states: bool = True,
            save_top_k: Optional[int] = None,
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

                # Optionally save the top K predictions:
                if save_top_k:
                    top_k_similarity, top_k_index = torch.topk(similarity_span_all_labels, k=save_top_k, dim=1)

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
                    f' --> {pred} ({symbol}) "{verbalization}"\n'
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
        save_top_k: Optional[int] = None,
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







