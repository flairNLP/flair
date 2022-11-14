import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union


import torch

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence, Span, DT
from flair.embeddings import Embeddings
from flair.training_utils import store_embeddings


import os
import json
from pathlib import Path

log = logging.getLogger("flair")


class SpanTaggerWithGazetteerMining(flair.nn.DefaultClassifier[Sentence, Span]):
    """
    Span Tagger
    The model expects text/sentences with annotated tags on token level (e.g. NER), and learns to predict spans.
    All possible combinations of spans (up to a defined max length) are considered, represented e.g. via concatenation
    of the word embeddings of the first and last token in the span.
    An optional span based gazetteer embedding look up is concatenated.
    Then fed through a linear layer to get the actual span label. Overlapping spans can be resolved or kept.
    """

    def __init__(
            self,
            label_dictionary: Dictionary,
            word_embeddings: flair.embeddings.TokenEmbeddings,
            dynamic_gazetteer_embeddings = None,
            update_during_eval: bool = True,
            update_during_train: bool = True,
            gazetteer_embeddings = None,
            mask_gazetteer: bool = False,
            pooling_operation: str = "first_last",
            label_type: str = "ner",
            max_span_length: int = 5,
            delete_goldsubspans_in_training: bool = False,
            concat_span_length_to_embedding: bool = False,
            resolve_overlaps: str = "by_token",
            decoder=None,
            **classifierargs,
    ):
        """
        Initializes a SpanTagger
        :param word_embeddings: embeddings used to embed the words/sentences
        :param gazetteer_embeddings: span embeddings, i. e. coming from a gazetteer
        :param label_dictionary: dictionary that gives ids to all classes. Should contain <unk>
        :param pooling_operation: either 'average', 'first', 'last' or 'first_last'. Specifies the way of how text representations of spans are handled.
        E.g. 'average' means that as text representation we take the average of the embeddings of the words in the span. 'first&last' concatenates
        the embedding of the first and the embedding of the last word.
        :param label_type: name of the label you use.
        :param max_span_length: maximum length of spans (in tokens) that are considered
        :param delete_goldsubspans_in_training: During training delete all spans that are subspans of gold labeled spans. Helpful for making gazetteer signal more clear.
        :param concat_span_length_to_embedding: if set to True normalized span length is concatenated to span embeddings
        :param resolve_overlaps: one of
            'keep_overlaps' : overlapping predictions stay as they are (i.e. not using _post_process_predictions())
            'by_token' : only allow one prediction per token/span (prefer spans with higher confidence)
            'no_boundary_clashes' : predictions cannot overlap boundaries, but can include other predictions (nested NER)
            'prefer_longer' : prefers longer spans over shorter ones # TODO: names are confusing, this is also "by token" but with length instead of score
        """

        final_embedding_size = 0
        if word_embeddings:
            final_embedding_size += word_embeddings.embedding_length * 2 \
                if pooling_operation == "first_last" else word_embeddings.embedding_length

        if gazetteer_embeddings:
            final_embedding_size += gazetteer_embeddings.embedding_length
        elif dynamic_gazetteer_embeddings:
            final_embedding_size += dynamic_gazetteer_embeddings.embedding_length

        if concat_span_length_to_embedding:
            final_embedding_size += 1

        # make sure the label dictionary has an "O" entry for "no tagged span"
        label_dictionary.add_item("O")

        super(SpanTaggerWithGazetteerMining, self).__init__(
            label_dictionary=label_dictionary,
            final_embedding_size=final_embedding_size,
            decoder=decoder,
            **classifierargs,
        )

        self.word_embeddings = word_embeddings
        self.dynamic_gazetteer_embeddings = dynamic_gazetteer_embeddings
        if self.dynamic_gazetteer_embeddings:
            self.dynamic_gazetteer_mining = True
            self.update_during_train = update_during_train
            self.update_during_eval = update_during_eval

        else:
            self.dynamic_gazetteer_mining = False

        if self.dynamic_gazetteer_mining:
            self.gazetteer_embeddings = self.dynamic_gazetteer_embeddings
            # TODO: klappt das so? Werden die benutzt?
        else:
            self.gazetteer_embeddings = gazetteer_embeddings

        self.mask_gazetteer = mask_gazetteer

        self.delete_goldsubspans_in_training = delete_goldsubspans_in_training
        self.concat_span_length_to_embedding = concat_span_length_to_embedding
        if self.concat_span_length_to_embedding:
            final_embedding_size += 1

        self.pooling_operation = pooling_operation
        self._label_type = label_type
        self.max_span_length = max_span_length
        self.resolve_overlaps = resolve_overlaps

        cases = {
            "average": self.emb_mean,
            "first": self.emb_first,
            "last": self.emb_last,
            "first_last": self.emb_firstAndLast
        }

        if pooling_operation not in cases:
            raise KeyError('pooling_operation has to be one of "average", "first", "last" or "first_last"')

        if resolve_overlaps not in ["keep_overlaps", "no_boundary_clashes", "by_token", "prefer_longer"]:
            raise KeyError(
                'resolve_overlaps has to be one of "keep_overlaps", "no_boundary_clashes", "by_token", "prefer_longer"')

        self.aggregated_embedding = cases[pooling_operation]

        self.to(flair.device)

    def emb_first(self, span: Span, embedding_names):
        return span.tokens[0].get_embedding(embedding_names)

    def emb_last(self, span: Span, embedding_names):
        return span.tokens[-1].get_embedding(embedding_names)

    def emb_firstAndLast(self, span: Span, embedding_names):
        return torch.cat(
            (span.tokens[0].get_embedding(embedding_names), span.tokens[-1].get_embedding(embedding_names)), 0
        )

    def emb_mean(self, span, embedding_names):
        return torch.mean(torch.cat([token.get_embedding(embedding_names) for token in span], 0), 0)

    @property
    def _inner_embeddings(self) -> Embeddings[Sentence]:
        return self.word_embeddings

    def _get_prediction_data_points(self, sentences: List[Sentence]) -> List[Span]:
        if not isinstance(sentences, list):
            sentences = [sentences]
        spans: List[Span] = []
        for sentence in sentences:
            # create list of all possible spans (consider given max_span_length)
            spans_sentence = []
            tokens = [token for token in sentence]
            for span_len in range(1, self.max_span_length + 1):
                spans_sentence.extend([Span(tokens[n:n + span_len]) for n in range(len(tokens) - span_len + 1)])

            # delete spans that are subspans of labeled spans (to help make gazetteer training signal more clear)
            if self.delete_goldsubspans_in_training and self.training:
                goldspans = sentence.get_spans(self.label_type)

                # make list of all subspans of goldspans
                gold_subspans = []
                for goldspan in goldspans:
                    goldspan_tokens = [token for token in goldspan.tokens]
                    for span_len in range(1, self.max_span_length + 1):
                        gold_subspans.extend(
                            [Span(goldspan_tokens[n:n + span_len]) for n in
                             range(len(goldspan_tokens) - span_len + 1)])

                gold_subspans = [span for span in gold_subspans if
                                 not span.has_label(self.label_type)]  # FULL goldspans should be kept!

                # finally: remove the gold_subspans from spans_sentence
                spans_sentence = [span for span in spans_sentence if span.unlabeled_identifier not in [s.unlabeled_identifier for s in gold_subspans]]

            spans.extend(spans_sentence)

        return spans

    def _embed_prediction_data_point(self, prediction_data_point: Span) -> torch.Tensor:

        span_embedding_parts = []
        if self.word_embeddings:
            span_embedding_parts.append(self.aggregated_embedding(prediction_data_point, self.word_embeddings.get_names()))

        if self.concat_span_length_to_embedding:
            span_embedding_parts.append(torch.tensor([len(prediction_data_point) / self.max_span_length]).to(flair.device))

        if self.gazetteer_embeddings:
            if not self.mask_gazetteer:
                self.gazetteer_embeddings.embed(prediction_data_point)
                gazetteer_embedding = prediction_data_point.get_embedding(self.gazetteer_embeddings.get_names())
            else:
                gazetteer_embedding = torch.zeros(self.gazetteer_embeddings.embedding_length, device=flair.device)

            span_embedding_parts.append(gazetteer_embedding)

        span_embedding = torch.cat(span_embedding_parts, 0)

        return span_embedding


    def _post_process_predictions(self, batch, label_type):
        """
        Post-processing the span predictions to avoid overlapping predictions, using condition as set by resolve_overlaps parameter.
        When in doubt use the most confident one, i.e. sort the span predictions by confidence, go through them and
        decide via the given criterion.
        :param batch: batch of sentences with already predicted span labels to be "cleaned"
        :param label_type: name of the label that is given to the span
        """

        if not self.resolve_overlaps == "keep_overlaps":
            import operator

            for sentence in batch:
                all_predicted_spans = []

                # get all predicted spans and their confidence score, sort them afterwards
                for span in sentence.get_spans(label_type):
                    span_tokens = span.tokens
                    span_score = span.get_label(label_type).score
                    span_length = len(span_tokens)
                    span_prediction = span.get_label(label_type).value
                    all_predicted_spans.append((span_tokens, span_prediction, span_score, span_length))

                sentence.remove_labels(label_type)  # first remove the predicted labels

                if self.resolve_overlaps in ["by_token", "no_boundary_clashes"]:
                    # sort by confidence score
                    sorted_predicted_spans = sorted(all_predicted_spans, key=operator.itemgetter(2))  # by score

                elif self.resolve_overlaps == "prefer_longer":
                    # sort by length
                    sorted_predicted_spans = sorted(all_predicted_spans, key=operator.itemgetter(3))  # by length

                sorted_predicted_spans.reverse()

                if self.resolve_overlaps in ["by_token", "prefer_longer"]:
                    # in short: if a token already was part of a higher ranked span, break
                    already_seen_token_indices: List[int] = []

                    # starting with highest scored span prediction
                    for predicted_span in sorted_predicted_spans:
                        span_tokens, span_prediction, span_score, span_length = predicted_span

                        # check whether any token in this span already has been labeled
                        tag_span = True
                        for token in span_tokens:
                            if token is None or token.idx in already_seen_token_indices:
                                tag_span = False
                                break

                        # only add if none of the token is part of an already (so "higher") labeled span
                        if tag_span:
                            already_seen_token_indices.extend(token.idx for token in span_tokens)
                            predicted_span = Span(
                                [sentence.get_token(token.idx) for token in span_tokens]
                            )
                            predicted_span.add_label(label_type, value=span_prediction, score=span_score)

                if self.resolve_overlaps == "no_boundary_clashes":
                    # in short: don't allow clashing start/end positions, nesting is okay
                    already_predicted_spans = []  # list of tuples with start/end positions of already predicted spans

                    for predicted_span in sorted_predicted_spans:
                        tag_span = True
                        span_tokens, span_prediction, span_score, span_length = predicted_span
                        start_candidate = span_tokens[0].idx
                        end_candidate = span_tokens[-1].idx

                        # check if boundaries clash, if yes set tag_span = False
                        for (start_already, end_already) in already_predicted_spans:
                            if (start_candidate < start_already <= end_candidate < end_already or
                                    start_already < start_candidate <= end_already < end_candidate):
                                tag_span = False
                                break

                        if tag_span:
                            already_predicted_spans.append((start_candidate, end_candidate))
                            predicted_span = Span(
                                [sentence.get_token(token.idx) for token in span_tokens]
                            )
                            predicted_span.add_label(label_type, value=span_prediction, score=span_score)


    def _update_dynamic_gazetteer(self, batch, label_type, use_predictions=True):
        # TODO: right now this is done both in evaluating on dev AND test.
        # Not as problematic given that I use the predictions, not gold data directly, but should the latter change, we need to change the former!

        if self.dynamic_gazetteer_embeddings.skip_first_epoch:
            raise NotImplementedError
            # TODO how to get nr of epochs we're in?
        else:
            if use_predictions:
                label_identifier = "predicted" # using only the spans predicted by the model
            elif use_predictions == "both":
                raise NotImplementedError # TODO using both the predicted and the gold annotations?
            else:
                label_identifier = label_type # using only the gold annotated spans
                # TODO have not tested this so far
            for sentence in batch:
                for span in sentence.get_spans(label_identifier):
                    span_text = span.text
                    confidence = span.get_label(label_identifier).score
                    span_prediction = span.get_label(label_identifier).value
                    if confidence > self.dynamic_gazetteer_embeddings.confidence_threshold:
                        self.dynamic_gazetteer_embeddings.update_gazetteer_embeddings(span_text, span_prediction, method = "replace")

    # DYNAMIC EMBEDDING UPDATE
    # overwriting forward_loss and predict as to include the dynamic gazetteer updating
    # so
    # 1) do the prediction (we need the labels, not just the scores) and
    # 2) update the embedding

    def forward_loss(self, sentences) -> Tuple[torch.Tensor, int]:
        self.predict(sentences, label_name="predicted")

        return super().forward_loss(sentences)

    def predict(self, sentences, **kwargs):
        rt = super().predict(sentences, **kwargs)

        if self.dynamic_gazetteer_mining:
            if self.training and self.update_during_train:
                self._update_dynamic_gazetteer(sentences, label_type=self.label_type, use_predictions=True)
            elif self.update_during_eval:
                self._update_dynamic_gazetteer(sentences, label_type=self.label_type, use_predictions=True)

            return rt

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Module):
            for layer in m.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                    log.info(f"----- Resetting parameters for: {layer.__class__.__name__}")
                else:
                    log.info(f"----- Not Resetting for: {layer.__class__.__name__}")

    def reset_decoder_parameters(self):
        self.decoder.apply(self._init_weights)

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "word_embeddings": self.word_embeddings,
            "label_type": self.label_type,
            "label_dictionary": self.label_dictionary,
            "pooling_operation": self.pooling_operation,
            "loss_weights": self.weight_dict,
            "resolve_overlaps": self.resolve_overlaps,
            "delete_goldsubspans_in_training": self.delete_goldsubspans_in_training,
            "gazetteer_embeddings": self.gazetteer_embeddings,
            "dynamic_gazetteer_embeddings": self.dynamic_gazetteer_embeddings,
            "max_span_length": self.max_span_length,
            "concat_span_length_to_embedding": self.concat_span_length_to_embedding,
        }
        return model_state

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            printed = []

            eval_line = f"\n{datapoint.to_original_text()}\n"
            for span in datapoint.get_spans(gold_label_type):
                symbol = "✓" if span.get_label(gold_label_type).value == span.get_label("predicted").value else "❌"
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {span.get_label("predicted").value} ({symbol})'
                    f' \tscore:\t{round(span.get_label("predicted").score, 2)}\n'
                )
                printed.append(span)

            # print out also the wrongly predicted (no gold label)
            for span in datapoint.get_spans("predicted"):
                if span.get_label("predicted").value != span.get_label(gold_label_type).value and span not in printed:
                    eval_line += (
                        f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                        f' --> {span.get_label("predicted").value} ("❌")'
                        f' \tscore:\t{round(span.get_label("predicted").score, 2)}\n'

                    )
                    printed.append(span)

            lines.append(eval_line)

        return lines

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            word_embeddings=state.get("word_embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            pooling_operation=state.get("pooling_operation"),
            loss_weights=state.get("loss_weights", {"<unk>": 0.3}),
            resolve_overlaps=state["resolve_overlaps"],
            gazetteer_embeddings=state["gazetteer_embeddings"],
            dynamic_gazetteer_embeddings=state["dynamic_gazetteer_embeddings"],
            concat_span_length_to_embedding=state["concat_span_length_to_embedding"],

            **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type

    def error_analysis(self, batch, gold_label_type, out_directory, save_gazetteer_to_file = True, save_updated_gazetteer_to_file = True):
        lines = []
        lines_just_errors = []

        count_true = 0
        count_error = 0
        count_ground_truth_in_gazetteer = 0
        count_ground_truth_not_in_gazetteer = 0

        self.evaluate(batch, gold_label_type)

        for datapoint in batch:

            eval_line = f"\n{datapoint.to_original_text()}\n"

            # first iterate over gold spans and see if matches predictions
            printed = []
            contains_error = False  # gets set to True if one or more incorrect predictions in datapoint

            for span in datapoint.get_spans(gold_label_type):
                if self.gazetteer_embeddings:
                    if not self.mask_gazetteer:
                        self.gazetteer_embeddings.embed(span)
                        gazetteer_embedding = span.get_embedding()
                        original_gaz_entry = self.gazetteer_embeddings.gazetteer.get(span.text, torch.Tensor([]))

                    else:
                        gazetteer_embedding = torch.zeros(self.gazetteer_embeddings.embedding_length, device=flair.device)
                        original_gaz_entry = gazetteer_embedding

                    if sum(original_gaz_entry) > 0.0:
                        count_ground_truth_in_gazetteer +=1
                    else:
                        count_ground_truth_not_in_gazetteer +=1

                else:
                    gazetteer_embedding = torch.Tensor([])  # dummy

                symbol = "✓" if span.get_label(gold_label_type).value == span.get_label("predicted").value else "❌"
                if span.get_label(gold_label_type).value == span.get_label("predicted").value:
                    count_true += 1
                else:
                    count_error += 1
                    contains_error = True

                printed.append(span)
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {span.get_label("predicted").value} ({symbol})'
                    f' \t score:\t{round(span.get_label("predicted").score, 2)}'
                    f' \t gazetteer entry \t {[round(e, 2) for e in original_gaz_entry.tolist()]}'
                    f' \t gazetteer embedding \t {[round(e, 2) for e in gazetteer_embedding.tolist()]}\n'

                )
            # now add also the predicted spans that have *no* gold span equivalent
            for span in datapoint.get_spans("predicted"):
                if self.gazetteer_embeddings:
                    if not self.mask_gazetteer:
                        self.gazetteer_embeddings.embed(span) #TODO would be nice to have already stored span embeddings
                        gazetteer_embedding = span.get_embedding()
                        original_gaz_entry = self.gazetteer_embeddings.gazetteer.get(span.text, torch.Tensor([]))
                    else:
                        gazetteer_embedding = torch.zeros(self.gazetteer_embeddings.embedding_length, device=flair.device)
                        original_gaz_entry = gazetteer_embedding


                else:
                    gazetteer_embedding = torch.Tensor([])  # dummy

                if span.get_label("predicted").value != span.get_label(gold_label_type).value and span not in printed:
                    count_error += 1
                    contains_error = True

                    printed.append(span)
                    eval_line += (
                        f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                        f' --> {span.get_label("predicted").value} ("❌")'
                        f' \t score:\t{round(span.get_label("predicted").score, 2)}'
                        f' \t gazetteer entry \t {[round(e, 2) for e in original_gaz_entry.tolist()]}'
                        f' \t gazetteer embedding \t {[round(e, 2) for e in gazetteer_embedding.tolist()]}\n'

                    )

            lines.append(eval_line)
            if contains_error:
                lines_just_errors.append(eval_line)

        percentage_true_all = round(count_true / (count_true + count_error) * 100, 2)
        if self.gazetteer_embeddings:
            percentage_found_all = round(count_ground_truth_in_gazetteer / (count_ground_truth_in_gazetteer + count_ground_truth_not_in_gazetteer) * 100, 2)
        else:
            percentage_found_all = 0.0

        error_counts_dict = {"count_true": count_true,
                             "count_error": count_error,
                             "percentage_true_all": percentage_true_all,
                             "count_ground_truth_in_gazetteer": count_ground_truth_in_gazetteer,
                             "count_ground_truth_not_in_gazetteer": count_ground_truth_not_in_gazetteer,
                             "percentage_found_all": percentage_found_all,
                             }

        if self.dynamic_gazetteer_mining:
            if len(self.gazetteer_embeddings.gazetteer) > 0:
                percentage_updated_full = round(len(self.gazetteer_embeddings._get_updated_partition_of_gazetteer())/len(self.gazetteer_embeddings.gazetteer)*100,2)
            else:
                percentage_updated_full = 0.0

            add_dict = {"size_initial_gazetteer": len(self.gazetteer_embeddings.initial_gazetteer) if self.gazetteer_embeddings.initial_gazetteer else 0,
                        "size_full_gazetteer": len(self.gazetteer_embeddings.gazetteer),
                        "size_updated_partition_gazetteer": len(self.gazetteer_embeddings._get_updated_partition_of_gazetteer()),
                        "percentage_updated_full": percentage_updated_full}
            error_counts_dict.update(add_dict)


        if out_directory:
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)
            with open(Path(out_directory / "predictions.txt"), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))
            with open(Path(out_directory / "predictions_just_errors.txt"), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines_just_errors))
            with open(Path(out_directory / "counts_and_percentages.json"), "w", encoding="utf-8") as fp:
                json.dump(error_counts_dict, fp)

            if save_gazetteer_to_file and self.gazetteer_embeddings:
                save_updated_gazetteer_to_file = True

                with open(Path(out_directory / "used_gazetteer_full.txt"), "w", encoding="utf-8") as fp:
                    for key, value in self.gazetteer_embeddings.gazetteer.items():
                        fp.write(f"{str(key)}\t{[round(e, 2) for e in value.tolist()]}\n")

                if self.dynamic_gazetteer_embeddings.starting_with_gazetteer:
                    with open(Path(out_directory / "used_gazetteer_initial.txt"), "w", encoding="utf-8") as fp:
                        for key, value in self.gazetteer_embeddings.initial_gazetteer.items():
                            fp.write(f"{str(key)}\t{[round(e, 2) for e in value.tolist()]}\n")

            if save_updated_gazetteer_to_file and self.gazetteer_embeddings:
                if self.dynamic_gazetteer_mining:
                    with open(Path(out_directory / "used_gazetteer_updated_parts.txt"), "w", encoding="utf-8") as fp:
                        for key, value in self.gazetteer_embeddings._get_updated_partition_of_gazetteer().items():
                            fp.write(f"{str(key)}\t{[round(e, 2) for e in value.tolist()]}\n")



        return error_counts_dict

