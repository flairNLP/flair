import sys
import logging
import random
from tqdm import tqdm
from typing import Tuple, Dict, List, Callable, Literal

import torch
import torch.nn.functional as F

import flair
from flair.data import DT, Dictionary, Optional, Sentence, Span, Union
from flair.embeddings import DocumentEmbeddings, TokenEmbeddings

log = logging.getLogger("flair")



def insert_verbalizations_into_sentence(sentence: Sentence, label_type: str, label_map: dict, verbalize_previous: int = 0, verbalize_next: int = 0):
    """
    Insert label verbalizations into sentence.
    :param sentence: Flair sentence object to apply label verbalizations to.
    :param label_type: Label type whose value gets verbalized.
    :param label_map: A mapping of label values to more descriptive label verbalization to use for the insertions.
    :param verbalize_previous: Number of context sentences before that also get label insertion applied to. Set to 0 if no insertions into context sentences wanted.
    :param verbalize_next: Number of context sentences after that also get label insertion applied to. Set to 0 if no insertions into context sentences wanted.
    :return: New Flair sentence object, now with verbalizations. Labels and context get copied from the original input sentence.
    """
    spans = sentence.get_spans()

    tokens_text = [t.text for t in sentence.tokens]
    added_tokens = 0
    token_indices = [[sp.tokens[0].idx, sp.tokens[-1].idx] for sp in spans]

    for i, sp in enumerate(sentence.get_spans(label_type)):
        label = sp.get_label(label_type).value
        add_at_position_in_tokens = sp.tokens[-1].idx + added_tokens

        verbalization = Sentence(f" ({label_map.get(label, label)})")

        verbalization_token_texts = [t.text for t in verbalization.tokens]
        len_verbalization_tokens = len(verbalization_token_texts)
        tokens_text = tokens_text[:add_at_position_in_tokens] + verbalization_token_texts + tokens_text[add_at_position_in_tokens:]

        added_tokens += len_verbalization_tokens

        for j, d in enumerate(spans):
            s_start_idx= token_indices[j][0]
            s_end_idx = token_indices[j][1]

            if s_start_idx > add_at_position_in_tokens:
                s_start_idx += len(verbalization_token_texts)
                s_end_idx += len(verbalization_token_texts)
                token_indices[j][0] = s_start_idx
                token_indices[j][1] = s_end_idx

    # Cannot use new_sentence = Sentence(text), we needed to use tokens instead of working with the text directly because of the weird problem with the space in unicode, e.g. 0x200c
    new_sentence = Sentence(tokens_text)

    for i, sp in enumerate(spans):
        start_token_index, end_token_index = token_indices[i]
        new_sp = Span(new_sentence.tokens[start_token_index-1:(end_token_index)])

        for k, labels in sp.annotation_layers.items():
            for l in labels:
                new_sp.set_label(typename=k, value=l.value, score=l.score)

    if verbalize_previous > 0 and sentence._previous_sentence:
        new_sentence._previous_sentence = insert_verbalizations_into_sentence(sentence._previous_sentence, label_type, label_map, verbalize_previous = verbalize_previous-1, verbalize_next = 0)
    else:
        new_sentence._previous_sentence = sentence._previous_sentence

    if verbalize_next > 0 and sentence._next_sentence:
        new_sentence._next_sentence = insert_verbalizations_into_sentence(sentence._next_sentence, label_type, label_map, verbalize_previous = 0, verbalize_next = verbalize_next-1)
    else:
        new_sentence._next_sentence = sentence._next_sentence

    return new_sentence






class DEEDTripletMarginLoss(torch.nn.TripletMarginLoss):

    def forward(self, anchor, positive, negative):
        factor = negative.shape[0]
        losses = [super(DEEDTripletMarginLoss, self).forward(anchor, positive, negative[i]) for i in range(factor)]
        losses = torch.stack(losses, dim=0)
        return torch.mean(losses)


class DEEDEuclideanEmbeddingLoss(torch.nn.Module):
    def __init__(self, margin=10.0):
        """
        Similar to pytorch's CosineEmbeddingLoss (https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html) with negatives, but using euclidean distance.
        :param margin: Margin to push the negatives away from the anchor.
        """
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive,  negative):
        # handle positives
        # euclidean distance between anchor and positive embeddings
        positive_loss = torch.sqrt(torch.sum(torch.square(anchor - positive), dim=-1))

        # handle negatives
        # calculate the euclidean distance between the anchor and each batch of negatives
        dist = torch.nn.functional.pairwise_distance(anchor, negative)
        # loss is distance after margin is applied
        negative_loss = torch.max(torch.tensor(0.0), self.margin - dist)

        # take mean over both losses
        # todo If negatives factor > 1, we weigh the negative losses more. Do we want that?
        # could instead do for example:
        # positive_loss = positive_loss.expand(negative_loss.shape)
        losses = torch.cat([positive_loss.unsqueeze(0), negative_loss])
        return torch.mean(losses)


class DualEncoderEntityDisambiguation(flair.nn.Classifier[Sentence]):

    def __init__(self, token_encoder: TokenEmbeddings, label_encoder: DocumentEmbeddings, known_labels: List[str], gold_labels: List[str] = [], label_sample_negative_size: Union[int, None] = None, label_type: str = "nel", label_map: dict = {},
                 negative_sampling_strategy: Literal["shift", "random", "hard"] = "hard", negative_sampling_factor: int = 1,
                 loss_function_name: Literal ["triplet", "binary_embedding"] = "triplet", label_embedding_batch_size: int = 128, sampled_label_embeddings_storage_device: torch.device = None, *args, **kwargs):
        """
        This model uses a dual encoder architecture where both inputs and labels (verbalized) are encoded with separate
        Transformers. It uses some kind of similarity loss to push datapoints and true labels nearer together while pushing negatives away
        and performs KNN like inference.
        More descriptive label verbalizations can be plugged in.
        :param token_encoder: Token embeddings to embed the spans in a sentence.
        :param label_encoder: Document embeddings to embed the label verbalizations.
        :param known_labels: List of all labels that the model can use, in addition to the gold labels.
        :param gold_labels: List of corpus specific gold labels that should be used during predictions.
        :param label_sample_negative_size: The number of the sampled labels from known_labels that is used as a source for negative sampling and in prediction.
                Ideally this would use all known_labels but this can be too expensive for memory (all need to be embedded).
        :param label_type: Label type to predict (e.g. "nel").
        :param label_map: Mapping of label values to more descriptive verbalizations, used for embedding the labels.
        :param negative_sampling_strategy: Strategy to search for negative samples. Must be one of "hard", "shift", "random".
        :param negative_sampling_factor: Number of negatives per positive, e.g. 1 (one negative sample per positive), 2 (two negative samples per positive).
        :param loss_function_name: Loss funtion to use, must be one of "triplet", "binary_embedding".
        :param label_embedding_batch_size: Batch size to use for embedding labels to avoid memory overflow.
        :param sampled_label_embeddings_storage_device: Device to store the sampled label embeddings on. If None, uses flair.device
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
        self._label_sample_negative_size = label_sample_negative_size
        self._sampled_labels = None
        self._sampled_label_embeddings = None
        self._next_prediction_needs_updated_label_embeddings = False
        self._label_embedding_batch_size = label_embedding_batch_size
        if not sampled_label_embeddings_storage_device:
            sampled_label_embeddings_storage_device = flair.device
        self._sampled_label_embeddings_storage_device = sampled_label_embeddings_storage_device
        if loss_function_name == "triplet":
            self.loss_function = DEEDTripletMarginLoss(margin=1.0)
        elif loss_function_name == "binary_embedding":
            self.loss_function = DEEDEuclideanEmbeddingLoss()
        else:
            raise ValueError(f"Loss {loss_function_name} not recognized.")
        self.negative_sampling_strategy = negative_sampling_strategy
        if negative_sampling_strategy == "shift":
            self._negative_sampling_fn = self._negative_sampling_shift
        elif negative_sampling_strategy == "random":
            self._negative_sampling_fn = self._negative_sampling_random_over_all
        elif negative_sampling_strategy == "hard":
            self._negative_sampling_fn = self._negative_sampling_hard
        else:
            raise ValueError(f"Negative Sampling Strategy {negative_sampling_strategy} not supported.")
        self._negative_sampling_factor = negative_sampling_factor

        self.to(flair.device)

        # Checking how many labels in known_labels are NOT found in the label_map
        if label_map:
            print("Checking verbalizations...")
            non_verbalized_labels = []
            for label in tqdm(known_labels):
                if label not in label_map:
                    non_verbalized_labels.append(label)
            print(f"Found {len(non_verbalized_labels)} non verbalized labels:")
            for l in non_verbalized_labels:
                is_gold = " (gold)" if l in gold_labels else ""
                print(f"\t{l}{is_gold}")



    @property
    def label_type(self):
        return self._label_type

    def update_labels(self, known: List[str], gold: List[str]):
        """
        Giving the model a new set on known or gold labels. E.g. when predicting on a new corpus.
        :param known: List of all labels the model should be aware of. Same as known_labels in init.
        :param gold: List of gold labels. Same as gold_labels in init.
        :param sample_negative_size:
        """
        self.known_labels = known
        self.gold_labels = gold
        self._resample_labels()

    def _resample_labels(self):
        """
        Resampling of sampled labels. E.g. necessary for a new epoch or after adding new labels (update_labels()).
        The model uses only the samples labels for looking for negatives and for prediction.
        """
        print("Resampling labels from known_labels.")
        if not self._label_sample_negative_size:
            print("Using ALL known labels because no label sample size given.")
            labels = self.known_labels
        else:
            print(f"Sampling {self._label_sample_negative_size} from known labels.")
            labels = self.gold_labels.copy() # make sure the gold labels are included (keep in mind that for ZELDA gold_labels are {}, because would be too many)
            if len(self.known_labels) > self._label_sample_negative_size:
                labels += random.sample(self.known_labels, self._label_sample_negative_size)
            else:
                print(f"Not enough labels found in known labels, taking them all.")
                labels += self.known_labels
            labels = list(set(labels))
        if self._sampled_labels != labels:
            print("Need label embedding update")
            self._sampled_labels = labels
            self._sampled_label_embeddings = None
        else:
            print("No change in labels.")

    def _embed_spans(self, sentences: List[Sentence]):
        """
        Embed sentences and get embeddings for their spans. Currently, we use mean pooling.
        :param sentences:
        :return:
        """
        spans = []
        sentences_to_embed = []
        for s in sentences:
            sentence_spans = s.get_spans(self.label_type)
            if sentence_spans:
                spans.extend(sentence_spans)
                sentences_to_embed.append(s)
        if not spans:
            return None, None

        self.token_encoder.embed(sentences_to_embed)
        embeddings = [torch.mean(torch.stack([token.get_embedding() for token in span], 0), 0) for span in spans]
        for s in sentences_to_embed:
            s.clear_embeddings()
        return spans, torch.stack(embeddings, dim=0)


    def _embed_labels_batchwise_return_stacked_embeddings(self, labels_sentence_objects: List[flair.data.Sentence], clear_embeddings: bool = True, use_tqdm: bool = True, device: torch.device = None):
        final_embeddings = []
        batch_size = self._label_embedding_batch_size
        batch_iterator = range(0, len(labels_sentence_objects), batch_size)

        if use_tqdm:
            batch_iterator = tqdm(batch_iterator)

        for i in batch_iterator:
            batch = labels_sentence_objects[i:i + batch_size]
            self.label_encoder.embed(batch)
            embeddings = torch.stack([l.get_embedding() for l in batch])
            if device:
                embeddings = embeddings.to(device)
            final_embeddings.append(embeddings)
            if clear_embeddings:
                for l in batch:
                    l.clear_embeddings()

        final_embeddings = torch.cat(final_embeddings)

        return final_embeddings

    def _negative_sampling_shift(self, span_embeddings: torch.Tensor, label_embeddings: torch.Tensor, batch_gold_labels: List[str]):
        """
        Shifting the label embeddings to make them negatives for each other.
        :param span_embeddings: Not used in this strategy.
        :param label_embeddings: Gold label embeddings of the spans.
        :param batch_gold_labels: Not used in this strategy.
        :return: Negative label embeddings. BxNxE (B: self._negative_sampling_factor, N: Number of spans, E: embedding dimension)
        """
        negative_samples_embeddings = []
        for i in range(self._negative_sampling_factor):
            negative_samples_embeddings.append(torch.roll(label_embeddings, shifts=1+i, dims=0))
        return torch.stack(negative_samples_embeddings, dim = 0)


    # def _negative_sampling_random_over_all(self, span_embeddings: torch.Tensor, label_embeddings: torch.Tensor, batch_gold_labels: List[str]):
    #     # todo: gradient of negative samples?
    #     negative_samples_indices = []
    #     for i in range(self._negative_sampling_factor):
    #         negative_samples_indices.append(random.sample(range(len(self.known_labels)), len(batch_gold_labels)))
    #     return self.known_label_embeddings[negative_samples_indices, :]

    def _negative_sampling_hard(self, span_embeddings: torch.Tensor, label_embeddings: torch.Tensor, batch_gold_labels: List[str]):
        """
        Look for difficult (i.e. similarity to span) labels as negatives.
        :param span_embeddings: Embeddings of the spans in this batch.
        :param label_embeddings: Not used in this strategy.
        :param batch_gold_labels: Gold labels of the spans in this batch.
        :return:
        """
        with torch.no_grad():
            span_embeddings = span_embeddings.to(self._sampled_label_embeddings_storage_device)
            similarity_spans_sampled_labels = -torch.cdist(span_embeddings, self.get_sampled_label_embeddings())
            for nr, label in enumerate(batch_gold_labels):
                try:
                    idx = self._sampled_labels.index(label)
                except Exception as _:
                    continue
                similarity_spans_sampled_labels[nr,idx] = -torch.inf
            _, most_similar_label_index = torch.topk(similarity_spans_sampled_labels, self._negative_sampling_factor, dim=1)

        # not used, but nice for debugging
        gettrace = getattr(sys, "gettrace", None)
        if gettrace is not None and gettrace():
            most_similar_labels_per_span = [[self._sampled_labels[i] for i in most_similar_label_index[j,:]] for j in range(most_similar_label_index.shape[0])]

        # flatten to a list
        # the order is (n + n ... +n) = factor*n where n is the number of span embeddings
        most_similar_label_index = most_similar_label_index.T.flatten()

        # reembed the labels (necessary for tracking gradients! Otherwise model cannot learn from the negatives.)
        most_similar_labels = [self._sampled_labels[i] for i in most_similar_label_index]
        most_similar_labels_sentence_objects = [Sentence(self.label_map.get(l,l)) for l in most_similar_labels]

        stacked = self._embed_labels_batchwise_return_stacked_embeddings(most_similar_labels_sentence_objects, use_tqdm = False)

        # return tensor of dimension factor x num_spans x embedding_size
        return torch.reshape(stacked, (self._negative_sampling_factor, *span_embeddings.shape))

    def get_sampled_label_embeddings(self):
        if self._sampled_labels is None:
            self._resample_labels()
        if self._sampled_label_embeddings is None:
            with torch.no_grad():
                print("Updating label embeddings...")
                print(" - Creating label objects...")
                all_labels_sentence_objects = [Sentence(self.label_map.get(l, l)) for l in tqdm(self._sampled_labels)]
                print(" - Embedding label objects...")
                self._sampled_label_embeddings = self._embed_labels_batchwise_return_stacked_embeddings(
                    all_labels_sentence_objects, device=self._sampled_label_embeddings_storage_device)
        return self._sampled_label_embeddings

    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:
        """
        One forward pass through the model. Embed sentences, get span representations, get label representations, sample negative labels, compute loss.
        :param sentences: Sentences in batch.
        :return: Tuple(loss, number of spans)
        """
        (spans, span_embeddings) = self._embed_spans(sentences)
        if spans is None:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        if len(spans) >20: # todo anything better?
            spans = spans[:20]
            span_embeddings = span_embeddings[:20]
            #return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        # get one embedding vector for each label
        labels = [sp.get_label(self.label_type).value for sp in spans]
        labels_sentence_objects = [Sentence(self.label_map.get(l,l)) for l in labels]
        #self.label_encoder.embed(labels_sentence_objects)
        #label_embeddings = torch.stack([l.get_embedding() for l in labels_sentence_objects], dim = 0)
        label_embeddings = self._embed_labels_batchwise_return_stacked_embeddings(labels_sentence_objects, use_tqdm= False)

        # sample negative labels
        negative_samples = self._negative_sampling_fn(span_embeddings, label_embeddings, labels)

        # calculate loss
        loss = self.loss_function(span_embeddings, label_embeddings, negative_samples)

        # label samples will need updated embeddings in the prediction
        self._next_prediction_needs_updated_label_embeddings = True

        return loss, len(spans)

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
        """
        Predicts labels for the spans in sentences. Adds them to the spans under label_name.
        :return:
        """
        with torch.no_grad():

            (spans, span_embeddings) = self._embed_spans(sentences)
            if spans is None:
                if return_loss:
                    return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0
                return

            # After a forward loss the embeddings of the sampled labels might be outdated because the weights of the label encoder might have changed.
            # Also, after resampling of the labels the label embeddings might not yet exist
            # To avoid unnecessary work the labels only get embedded here (important for a large label set, might take very long)
            if self._next_prediction_needs_updated_label_embeddings:
                self._resample_labels()
                self._sampled_label_embeddings = None
                self._next_prediction_needs_updated_label_embeddings = False
            sampled_label_embeddings = self.get_sampled_label_embeddings().to(flair.device)
            # Choosing the most similar label from the set of sampled_labels (might not include the true gold label)
            similarity_span_all_labels = -torch.cdist(span_embeddings, sampled_label_embeddings)
            most_similar_label_similarity, most_similar_label_index = torch.max(similarity_span_all_labels, dim=1)

            for i, sp in enumerate(spans):
                label_value = self._sampled_labels[most_similar_label_index[i]]
                label_score = most_similar_label_similarity[i].item()
                sp.set_label(label_name, label_value, score = label_score)

        if return_loss:
            # todo not yet implemented
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=False), len(spans)

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

            **kwargs,
        )



class GreedyDualEncoderEntityDisambiguation(DualEncoderEntityDisambiguation):
    """
    This is the greedy version of the DualEncoderEntityDisambiguation Class.
    During training, some of the gold labels get used for label verbalization insertion.
    During prediction, the most confident predicted labels get used for insertions, while the new sentences get
    re-embedded and predicted. This process is iterative until all spans have predicted labels.
    """

    def sample_spans_to_use_for_gold_label_verbalization(self, sentences, search_context_window: int = 2):
        """
        Samples random spans with a label_type annotation that will be used for gold label verbalization insertion during training.
        :param sentences: Sentences too search for spans.
        :param search_context_window: Number of context sentences before and after to include in search. Set to 0 if only the current sentence should be used.
        :return: Spans that were chosen for label verbalization.
        """
        spans = []
        for s in sentences:
            spans.extend(s.get_spans(self.label_type))
            (previous, next) = (s._previous_sentence, s._next_sentence)
            for i in range(search_context_window):
                if previous:
                    spans.extend(previous.get_spans(self.label_type))
                    previous = previous._previous_sentence
                if next:
                    spans.extend(next.get_spans(self.label_type))
                    next = next._next_sentence
        number_of_spans_to_verbalize = random.randint(0, len(spans))
        return random.sample(spans, number_of_spans_to_verbalize)

    def select_predicted_spans_to_use_for_label_verbalization(self, sentences, label_name, n: int):
        """
        From all spans with label_tape (e.g. "nel") and label_name (e.g. "predicted") annotation, take the n spans with highest score.
        :param sentences: Sentences to select spans from.
        :param label_name: Label type that is storing the scores of predictions (e.g. "predicted").
        :param n: Number of chosen highest scored spans.
        :return: n or less spans.
        """
        spans = []
        for s in sentences:
            spans.extend([sp for sp in s.get_spans(label_name) if sp.has_label(self.label_type)])
        sorted_spans = sorted(spans, key = lambda sp: sp.get_label(label_name).score)
        return sorted_spans[:n]


    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:
        """
        Forward pass through the (greedy) model. Same as the DualEncoderEntityDisambiguation model class, but adding some gold verbalizations beforehand.
        :param sentences: Sentences in batch.
        :return: Tuple(loss, number of spans)
        """
        sampled_spans = self.sample_spans_to_use_for_gold_label_verbalization(sentences, search_context_window=0)
        # add a verbalization marker to the chosen spans:
        for sp in sampled_spans:
            label = sp.get_label(self.label_type)
            sp.set_label("to_verbalized", value=label.value, score=label.score)
        # insert verbalizations (from the sampled_spans) into the sentences, using the verbalization marker:
        verbalized_sentences = [
            insert_verbalizations_into_sentence(s, "to_verbalized", label_map = self.label_map, verbalize_previous=0, verbalize_next=0) for s in
            sentences]
        # remove the verbalization marker from the ORIGINAL spans so that they remain unmodified:
        for sp in sampled_spans:
            sp.remove_labels("to_verbalized")
        # delete the label_type for the spans that were used for verbalization from the verbalized_sentences,
        # so that those do not get used in the forward pass afterwards:
        for s in verbalized_sentences:
            for sp in s.get_spans("to_verbalized"):
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
        return_span_and_label_hidden_states: bool = True
    ):
        """
        Predict labels for sentences. Uses the predict method from DualEncoderEntityDisambiguation, but in an iterative fashion.
        """
        original_sentences = sentences
        step_size = 5
        level = 0
        # iterate until all spans are predicted
        while True:
            # do the prediction of all remaining spans
            super(GreedyDualEncoderEntityDisambiguation, self).predict(sentences,
                                  mini_batch_size=mini_batch_size,
                                  return_probabilities_for_all_classes=return_probabilities_for_all_classes,
                                  verbose=verbose,
                                  label_name=label_name,
                                  return_loss=return_loss,
                                  embedding_storage_mode=embedding_storage_mode
                                  )

            # select the step_size highest confidence spans
            chosen_spans = self.select_predicted_spans_to_use_for_label_verbalization(sentences, label_name, step_size)

            # if no spans remaining, break
            if len(chosen_spans) == 0:
                break

            # verbalization markers for the current level
            verbalized_label_type = f"verbalized:{level}"
            input_sentence_label_type = f"input_sentence:{level}"

            # add markers to the chosen spans
            for sp in chosen_spans:
                label = sp.get_label(label_name)
                sp.set_label(verbalized_label_type, value=label.value, score=label.score)
                sp.set_label(input_sentence_label_type, value=sp.sentence.text, score = 0.0)

            # insert the label verbalizations of the chosen spans
            verbalized_sentences = [
                insert_verbalizations_into_sentence(s, verbalized_label_type, label_map = self.label_map, verbalize_previous=2, verbalize_next=2)
                for s in sentences]
            # keep the spans in the original sentences unmodified
            for sp in chosen_spans:
                sp.remove_labels(verbalized_label_type)
            # remove the label_type marker from the spans that were used for label verbalization insertion, so they will not be predicted again
            for s in verbalized_sentences:
                for sp in s.get_spans(verbalized_label_type):
                    sp.remove_labels(self.label_type)

            # prepare for the next iteration
            sentences = verbalized_sentences
            level +=1


        original_spans = []
        for s in original_sentences:
            original_spans.extend(s.get_spans(self.label_type))

        predicted_spans = []
        for s in sentences:
            predicted_spans.extend(s.get_spans(label_name))

        assert len(predicted_spans) == len(original_spans), \
            f"Not all spans could be verbalized: original: {len(original_spans)}, predicted: {len(predicted_spans)}"

        # transfer all the predicted labels to the original sentences and their spans
        for (orig, pred) in zip(original_spans, predicted_spans):
            label = pred.get_label(label_name)
            orig.set_label(label_name, label.value, label.score)

            # save the input sentence versions that were used for each span (that include the verbalizations at the time)
            input_sentence_key = next((key for key in pred.annotation_layers.keys() if key.startswith("input_sentence:")), None)
            if input_sentence_key:
                orig.set_label("sentence_input", pred.get_label(input_sentence_key).value, score = 0.0)
            else:
                orig.set_label("sentence_input", orig.sentence.text, score = 0.0) # this should not happen but to be sure

        if return_loss:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=False), len(original_spans)


    def _print_predictions(self, batch, gold_label_type, add_sentence_input: bool = True):
        lines = []
        for datapoint in batch:
            eval_line = f"\n{datapoint.to_original_text()}\n"

            for span in datapoint.get_spans(gold_label_type):
                symbol = "✓" if span.get_label(gold_label_type).value == span.get_label("predicted").value else "❌"
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {span.get_label("predicted").value} ({symbol})\n'
                )
                if add_sentence_input:
                    eval_line += (
                    f'  <-- "{span.get_label("sentence_input").value}"\n\n'
                    )

            lines.append(eval_line)
        return lines

