import logging
from typing import List, Union

import csv
import torch
import time
import json
import numpy as np

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence, Span

log = logging.getLogger("flair")


class SpanTagger(flair.nn.DefaultClassifier[Sentence]):
    """
    Span Tagger
    The model expects text/sentences with annotated tags on token level (e.g. NER), and learns to predict spans.
    All possible combinations of spans (up to a defined max length) are considered, represented e.g. via concatenation
    of the word embeddings of the first and last token in the span.
    An optional gazetteer look up is added.
    Then fed through a linear layer to get the actual span label. Overlapping spans can be resolved or kept.
    """

    def __init__(
        self,
        word_embeddings: flair.embeddings.TokenEmbeddings,
        label_dictionary: Dictionary,
        pooling_operation: str = "first_last",
        label_type: str = "ner",
        max_span_length: int = 5,
        delete_goldsubspans_in_training: bool = True,
        concat_span_length_to_embedding: bool = False,
        resolve_overlaps: str = "by_token",
        gazetteer_file: str = None,
        gazetteer_untagged_label_name: str = "untagged",
        gazetteer_include_untagged: bool = False,
        gazetteer_count_type: str = "norm_conf_ratio",
        use_precomputed_gazetteer: str = None,
        save_computed_gazetteer: str = None,
        add_lower_case_lookup: bool = False,
        add_substring_gazetteer_lookup: bool = False,
        ignore_embeddings: bool = False,
        use_mlp: bool = False,
        mlp_hidden_dim: int = 1024, #TODO make flexible
        decoder=None,
        **classifierargs,
    ):
        """
        Initializes an SpanTagger
        :param word_embeddings: embeddings used to embed the words/sentences
        :param label_dictionary: dictionary that gives ids to all classes. Should contain <unk>
        :param pooling_operation: either 'average', 'first', 'last' or 'first_last'. Specifies the way of how text representations of spans are handled.
        E.g. 'average' means that as text representation we take the average of the embeddings of the words in the span. 'first&last' concatenates
        the embedding of the first and the embedding of the last word.
        :param label_type: name of the label you use.
        :param max_span_length: maximum length of spans (in tokens) that are considered
        :param delete_goldsubspans_in_training: During training delete all spans that are subspans of gold labeled spans. Helpful for making gazetteer signal more clear.
        :param concat_span_length_to_embedding: if set to True span length (in tokens) is concatenated to span embeddings
        :param resolve_overlaps: one of
            'keep_overlaps' : overlapping predictions stay as they are (i.e. not using _post_process_predictions())
            'by_token' : only allow one prediction per token/span (prefer spans with higher confidence)
            'no_boundary_clashes' : predictions cannot overlap boundaries, but can include other predictions (nested NER)
            'prefer_longer' : prefers longer spans over shorter ones # TODO: names are confusing, this is also "by token" but with length instead of score
        :param gazetteer_count_type: in use only if gazetteer_file is given:
            "abs": absolute counts
                note: look out for effects based on frequency differences between the entries!
            "one_hot": converted into one hot vector (using argmax)
            "multi_hot": converted into multi-hot-vector (using fixed threshold over relative frequency)
            "norm_by_sum": normalized by dividing by sum of vector (whether that means including or excluding O counts)
            "norm_by_observations": normalized by dividing by total number of observations (requires O counts to be in gazetteer)
            "norm_conf": normalize (by sum) and concatenate confidence score based on frequency
            "norm_conf_ratio: normalize (by sum), concatenate confidence score and concatenate tagged/untagged ratio (requires O counts to be in gazetteer)
        :param gazetteer_file: path to a csv file containing a gazetteer list with span strings in rows, label names in columns, counts in cells
        :param gazetteer_untagged_label_name: give column name in gazetteer for "O" (untagged) counts (defaults to None, eg. not existing)
        :param gazetteer_include_untagged: set to True if you want to include untagged in gazetteer feature value (if existing)
        :param use_precomputed_gazetteer #TODO use an already precomputed gazetteer dictionary (e.g. with rel_conf vector), so give its json file path
        :param save_computed_gazetteer # TODO describe
        :param add_lower_case_lookup # TODO describe
        :param add_substring_gazetteer_lookup # TODO describe
        :param ignore_embeddings: simple baseline: just use gazetteer embedding, so ignore embeddings
        :param use_mlp: use a MLP as output layer (instead of default linear layer + xavier transformation), may be helpful for interactions with gazetteer
        :param mlp_hidden_dim: size of the hidden layer in output MLP
        """

        # make sure the label dictionary has an "O" entry for "no tagged span"
        label_dictionary.add_item("O")

        final_embedding_size = word_embeddings.embedding_length * 2 \
            if pooling_operation == "first_last" else word_embeddings.embedding_length

        self.delete_goldsubspans_in_training = delete_goldsubspans_in_training
        self.gazetteer_file = gazetteer_file
        self.gazetteer_untagged_label_name = gazetteer_untagged_label_name
        self.gazetteer_count_type = gazetteer_count_type
        self.add_lower_case_lookup = add_lower_case_lookup
        self.ignore_embeddings = ignore_embeddings
        self.use_precomputed_gazetteer = use_precomputed_gazetteer
        self.save_computed_gazetteer = save_computed_gazetteer
        self.gazetteer_untagged_label_name = gazetteer_untagged_label_name
        self.gazetteer_include_untagged = gazetteer_include_untagged
        self.add_substring_gazetteer_lookup = add_substring_gazetteer_lookup


        if self.ignore_embeddings:
            final_embedding_size = 0

        self.concat_span_length_to_embedding = concat_span_length_to_embedding
        if self.concat_span_length_to_embedding:
            final_embedding_size += 1

        if self.gazetteer_file:
            if self.use_precomputed_gazetteer:
                print("---- Reading already precomputed gazetteer file:", self.use_precomputed_gazetteer)
                with open(self.use_precomputed_gazetteer, 'r', encoding='utf8') as fp:
                    self.gazetteer = json.load(fp)
                print("---- Length of precomputed gazetteer:", len(self.gazetteer))

            if not self.use_precomputed_gazetteer:
                print("---- Reading raw gazetteer file:", self.gazetteer_file)
                self.gazetteer = {}
                with open(self.gazetteer_file, mode='r') as inp:
                    print(f"---- Gazetteer file contains {sum(1 for line in inp)} lines...")
                    inp.seek(0) # to start at beginning again
                    reader = csv.reader(inp)
                    header = next(reader)  # header line
                    print("---- Header is:", header)
                    self.gazetteer_entry_names = header[1:]
                    # get the id in gaz vector that corresponds to "O" counts (if exists)
                    try:
                        self.O_id = self.gazetteer_entry_names.index(self.gazetteer_untagged_label_name)
                    except:
                        self.O_id = -1
                    print("---- Entry names are:", self.gazetteer_entry_names)
                    print(f"---- Looked for O label (column named {self.gazetteer_untagged_label_name}), position in vector is:", self.O_id)

                    self.gazetteer = {row[0]: list(map(float, row[1:])) for row in reader}  # read rest in dict

                print(f"---- Length of used gazetteer:\t", len(self.gazetteer))

                if not self.gazetteer_count_type == "abs":
                    print("---- Converting the gazetteer counts into requested format:", self.gazetteer_count_type, "...")
                    global_start = time.time()
                    start = time.time()

                    for nr, (key, vector) in enumerate(self.gazetteer.items()):
                        now = time.time()

                        if (not self.gazetteer_include_untagged) and (self.O_id != -1):
                            cleaned_vector = vector[:self.O_id] + vector[self.O_id+1:]

                        else:
                            cleaned_vector = vector  # keep O counts in

                        if now-start >=30: # print progress every 30 seconds
                            print("done with \t", round(nr / len(self.gazetteer)*100, 2), " % of gazetteer", end = "\n")
                            start = time.time()

                        if self.gazetteer_count_type == "one_hot":
                            one_hot = np.zeros(len(cleaned_vector))
                            if torch.sum(cleaned_vector) > 0:  # necessary to avoid torch.argmax returning id 0
                                one_hot[np.argmax(cleaned_vector)] = 1
                            rt_vector = one_hot

                        if self.gazetteer_count_type == "multi_hot":
                            if torch.sum(cleaned_vector) > 0:  # avoid zero division
                                rel_vector = cleaned_vector / np.sum(cleaned_vector)
                            else:
                                rel_vector = cleaned_vector
                            THRESHOLD = 0.3  # TODO: what to choose?
                            multi_hot = (rel_vector >= THRESHOLD).type(np.int8)
                            rt_vector = multi_hot

                        if self.gazetteer_count_type == "norm_by_observations": # normalize by observation (spancount) (O label is needed), but dont use O label
                            if self.O_id == -1:
                                raise KeyError(
                                    'Gazetteer does not include counts for O (untagged), so cannot be computed. \n Maybe column name was not given correctly?')

                            observation_count = np.sum(vector)
                            sum_tagged = np.sum(cleaned_vector) # sum without counting the "O" counts

                            if observation_count > 0:
                                rt_vector = (cleaned_vector / observation_count)
                            else:
                                rt_vector = cleaned_vector

                        if self.gazetteer_count_type == "norm_by_sum": # divide by sum (whether O is included or not)
                            if np.sum(cleaned_vector) > 0:
                                rt_vector = (cleaned_vector / np.sum(cleaned_vector))
                            else:
                                rt_vector = cleaned_vector

                        if self.gazetteer_count_type == "norm_conf": # normalize and add confidence score
                            observation_count = np.sum(vector)
                            sum_tagged = np.sum(cleaned_vector) # sum without counting the "O" counts

                            DEFINED_MAX = 50  # TODO: what to choose here? make parameter
                            #confidence = np.array([min(observation_count / DEFINED_MAX, 1)])
                            confidence = np.array([min(sum_tagged / DEFINED_MAX, 1)])

                            if sum_tagged > 0:
                                rt_vector = (np.concatenate((cleaned_vector / sum_tagged, confidence), 0))
                            else:
                                rt_vector = (np.concatenate((cleaned_vector, confidence), 0))

                        if self.gazetteer_count_type == "norm_conf_ratio": # use relative counts, confidence score, add ratio tagged / abs_freq

                            if self.O_id == -1:
                                raise KeyError(
                                    'Gazetteer does not include counts for O (untagged), so ratio cannot be computed. \n Maybe column name was not given correctly?')

                            observation_count = np.sum(vector)
                            sum_tagged = np.sum(cleaned_vector) # sum without counting the "O" counts
                            untagged = observation_count - sum_tagged # nr of untagged
                            if observation_count > 0:
                                tagged_ratio = np.array([sum_tagged / observation_count])
                            #if untagged > 0:
                            #    tagged_ratio = np.array([sum_tagged / untagged])

                            else:
                                tagged_ratio = np.zeros(1)

                            DEFINED_MAX = 50  # TODO: what to choose here? make parameter
                            #confidence = np.array([min(observation_count / DEFINED_MAX, 1)])
                            confidence = np.array([min(sum_tagged / DEFINED_MAX, 1)])

                            if sum_tagged > 0:
                                rt_vector = (np.concatenate((cleaned_vector / sum_tagged, confidence, tagged_ratio), 0))
                            else:
                                rt_vector = (np.concatenate((cleaned_vector, confidence, tagged_ratio), 0))

                        #print(key, vector, "\t", cleaned_vector, "\t", rt_vector)

                        self.gazetteer[key] = np.around(rt_vector, decimals = 5).tolist()
                        #print(self.gazetteer[key])


                    global_end = time.time()
                    print(f"---- Converting took {round(global_end - global_start, 2)} seconds")

                    # saved computed gazetteer, to be able to load later...
                    if self.save_computed_gazetteer:
                        with open(f"{self.save_computed_gazetteer}", 'w', encoding='utf8') as fp:
                            json.dump(self.gazetteer, fp)
                        print("f---- Done saving precomputed dict as: ", f"{self.save_computed_gazetteer}")

            self.size_gazetteer_vector = len(
                next(iter(self.gazetteer.values())))  # one entry in gaz to get its size

            final_embedding_size += self.size_gazetteer_vector

            if self.add_lower_case_lookup:  # one more time
                final_embedding_size += self.size_gazetteer_vector

            if self.add_substring_gazetteer_lookup:
                final_embedding_size += self.size_gazetteer_vector

        self.use_mlp = use_mlp
        self.mlp_hidden_dim = mlp_hidden_dim

        if self.use_mlp:
            #TODO: maybe make more flexible: multiple layers, other nonlinearity, dropout?
            decoder = torch.nn.Sequential(
                torch.nn.Linear(final_embedding_size, self.mlp_hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mlp_hidden_dim, len(label_dictionary))
                )
            for n, p in decoder.named_parameters():
                if '.weight' in n:
                    torch.nn.init.xavier_uniform_(p)

        super(SpanTagger, self).__init__(
            label_dictionary=label_dictionary,
            final_embedding_size=final_embedding_size,
            decoder=decoder,
            **classifierargs,
        )

        self.word_embeddings = word_embeddings
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
            raise KeyError('resolve_overlaps has to be one of "keep_overlaps", "no_boundary_clashes", "by_token", "prefer_longer"')

        self.aggregated_embedding = cases[pooling_operation]

        self.to(flair.device)

    def emb_first(self, arg):
        return arg[0]

    def emb_last(self, arg):
        return arg[-1]

    def emb_firstAndLast(self, arg):
        return torch.cat((arg[0], arg[-1]), 0)

    def emb_mean(self, arg):
        return torch.mean(arg, 0)

    def get_gazetteer_embedding(self, span_string: str) -> torch.Tensor:

        if span_string in self.gazetteer:
            gaz_vector = torch.Tensor(self.gazetteer[span_string])
        else:
            vector_length = self.size_gazetteer_vector
            if self.gazetteer_include_untagged and self.O_id != -1: # if we have O in gazetteer, default O vector should also have 1
                gaz_vector = torch.zeros(vector_length)
                gaz_vector[self.O_id] = 1
            else:
                gaz_vector = torch.zeros(vector_length)  # get same length zero vector

        if self.add_lower_case_lookup:
            if span_string.title() in self.gazetteer: # "BARACK OBAMA" --> "Barack Obama"
                gaz_vector_lower = torch.Tensor(self.gazetteer[span_string.title()])
            else:
                vector_length = self.size_gazetteer_vector
                if self.gazetteer_include_untagged and self.O_id != -1:  # if we have O in gazetteer, default O vector should also have 1
                    gaz_vector_lower = torch.zeros(vector_length)
                    gaz_vector_lower[self.O_id] = 1
                else:
                    gaz_vector_lower = torch.zeros(vector_length)  # get same length zero vector

            gaz_vector = torch.concat((gaz_vector, gaz_vector_lower), 0)

        if self.add_substring_gazetteer_lookup:
            vector_length = self.size_gazetteer_vector
            tokens = span_string.split()
            subspans = []

            for span_len in range(1, len(tokens) +1):
                subspans.extend([" ".join(tokens[n:n + span_len]) for n in range(len(tokens) - span_len + 1)])

            sub_mean_vector = torch.zeros(vector_length)
            for sub in subspans:
                if sub in self.gazetteer:
                    sub_mean_vector += torch.Tensor(self.gazetteer[sub])
            if len(subspans) >0:
                sub_mean_vector = sub_mean_vector/len(subspans)

            gaz_vector = torch.concat((gaz_vector, sub_mean_vector))

        return gaz_vector.to(flair.device)

    def forward_pass(
        self,
        sentences: Union[List[Sentence], Sentence],
        for_prediction: bool = False,
    ):

        if not isinstance(sentences, list):
            sentences = [sentences]

        if not self.ignore_embeddings:
            self.word_embeddings.embed(sentences)
            names = self.word_embeddings.get_names()

        spans_embedded = None
        spans_labels = []
        data_points = []

        embedding_list = []
        for sentence in sentences:

            # create list of all possible spans (consider given max_span_length)
            spans_sentence = []
            tokens = [token for token in sentence]
            for span_len in range(1, self.max_span_length+1):
                spans_sentence.extend([Span(tokens[n:n + span_len]) for n in range(len(tokens) - span_len + 1)])

            # delete spans that are subspans of labeled spans (to help make gazetteer training signal more clear)
            if self.delete_goldsubspans_in_training:
                goldspans = sentence.get_spans(self.label_type)

                # make list of all subspans of goldspans
                gold_subspans = []
                for goldspan in goldspans:
                    goldspan_tokens = [token for token in goldspan.tokens]
                    for span_len in range(1, self.max_span_length+1):
                        gold_subspans.extend([Span(goldspan_tokens[n:n + span_len]) for n in range(len(goldspan_tokens) - span_len + 1)])

                gold_subspans = [span for span in gold_subspans if not span.has_label(self.label_type)] # FULL goldspans should be kept!

                # finally: remove the gold_subspans from spans_sentence
                spans_sentence = [span for span in spans_sentence
                                  if span.unlabeled_identifier not in [s.unlabeled_identifier for s in gold_subspans]]

            # embed each span (concatenate embeddings of first and last token)
            for span in spans_sentence:
                if not self.ignore_embeddings:

                    if self.pooling_operation == "first_last":
                        span_embedding = torch.cat(
                            (span[0].get_embedding(names),
                             span[-1].get_embedding(names)), 0)

                    if self.pooling_operation == "average":
                        span_embedding = torch.mean(
                            torch.stack([span[i].get_embedding(names) for i in range(len(span.tokens))]), 0)

                    if self.pooling_operation == "first":
                        span_embedding = span[0].get_embedding(names)

                    if self.pooling_operation == "last":
                        span_embedding = span[-1].get_embedding(names)

                # if ignore_embeddings == True use dummy "embedding" to have a baseline with just gazetteer
                else:
                    span_embedding = torch.zeros(0).to(flair.device)

                # concat the span length (scalar tensor) to span_embedding
                if self.concat_span_length_to_embedding:
                    length_as_tensor = torch.tensor([len(span)]).to(flair.device)
                    span_embedding = torch.cat((span_embedding, length_as_tensor), 0)

                # if a gazetteer was given, concat the gazetteer embedding to the span_embedding
                if self.gazetteer_file:
                    gazetteer_vector = self.get_gazetteer_embedding(span.text)
                    span_embedding = torch.cat((span_embedding, gazetteer_vector), 0)

                embedding_list.append(span_embedding.unsqueeze(0))

                # use the span gold labels
                spans_labels.append([span.get_label(self.label_type).value])

                # check if everything looks as it should (for spans with gold label other than "O")
                #if span.get_label(self.label_type).value != "O":
                #    print(span, span_embedding, span.get_label(self.label_type).value)

            if for_prediction:
                data_points.extend(spans_sentence)

        if len(embedding_list) > 0:
            spans_embedded = torch.cat(embedding_list, 0)

        if for_prediction:
            #for (span, label, data_point) in zip(spans_embedded, spans_labels, data_points):
            #    print(span, label, data_point)
            return spans_embedded, spans_labels, data_points

        #for (span, label) in zip(spans_embedded, spans_labels):
        #    print(span, label)

        return spans_embedded, spans_labels

    def _post_process_predictions(self, batch, label_type):
        """
        Post-processing the span predictions to avoid overlapping predictions.
        When in doubt use the most confident one, i.e. sort the span predictions by confidence, go through them and
        decide via the given criterion.

        :param batch: batch of sentences with already predicted span labels to be "cleaned"
        :param label_type: name of the label that is given to the span
        """

        # keep every span prediction, overlapping spans remain:
        if self.resolve_overlaps == "keep_overlaps":
            return batch

        # only keep most confident span in case of overlaps:
        else:
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
                                #print("skipping", predicted_span)
                                break

                        # only add if none of the token is part of an already (so "higher") labeled span
                        if tag_span:
                            #print("using", predicted_span)
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
                                #print("found clash for", span_tokens)
                                break

                        if tag_span:
                            already_predicted_spans.append((start_candidate, end_candidate))
                            #print("using ", span_tokens)
                            predicted_span = Span(
                                [sentence.get_token(token.idx) for token in span_tokens]
                            )
                            predicted_span.add_label(label_type, value=span_prediction, score=span_score)


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
            "gazetteer_file": self.gazetteer_file if self.gazetteer_file else None,
            "gazetteer_count_type": self.gazetteer_count_type,
            "add_lower_case_lookup": self.add_lower_case_lookup,
            "add_substring_gazetteer_lookup": self.add_substring_gazetteer_lookup,
            "max_span_length": self.max_span_length,
            "ignore_embeddings": self.ignore_embeddings,
            "use_mlp": self.use_mlp,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "concat_span_length_to_embedding": self.concat_span_length_to_embedding,
            "use_precomputed_gazetteer": self.use_precomputed_gazetteer,
            "gazetteer_untagged_label_name": self.gazetteer_untagged_label_name,
            "gazetteer_include_untagged": self.gazetteer_include_untagged,
            #"decoder": self.decoder,

        }
        return model_state

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:

            eval_line = f"\n{datapoint.to_original_text()}\n"

            # first iterate over gold spans and see if matches predictions
            printed = []
            for span in datapoint.get_spans(gold_label_type):
                symbol = "✓" if span.get_label(gold_label_type).value == span.get_label("predicted").value else "❌"
                printed.append(span)
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {span.get_label("predicted").value} ({symbol})\n'
                )
            # now add also the predicted spans that have *no* gold span equivalent
            for span in datapoint.get_spans("predicted"):
                if span.get_label("predicted").value != span.get_label(gold_label_type).value and span not in printed:
                    printed.append(span)
                    eval_line += (
                        f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                        f' --> {span.get_label("predicted").value} ("❌")\n'
                    )

            lines.append(eval_line)
        return lines

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            word_embeddings=state["word_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            pooling_operation=state["pooling_operation"],
            loss_weights=state["loss_weights"] if "loss_weights" in state else {"<unk>": 0.3},
            resolve_overlaps=state["resolve_overlaps"],
            delete_goldsubspans_in_training=state["delete_goldsubspans_in_training"],
            gazetteer_file=state["gazetteer_file"] if "gazetteer_file" in state else None,
            gazetteer_count_type=state["gazetteer_count_type"],
            add_lower_case_lookup=state["add_lower_case_lookup"],
            add_substring_gazetteer_lookup=state["add_substring_gazetteer_lookup"],
            ignore_embeddings=state["ignore_embeddings"],
            max_span_length=state["max_span_length"],
            use_mlp=state["use_mlp"],
            mlp_hidden_dim=state["mlp_hidden_dim"],
            concat_span_length_to_embedding=state["concat_span_length_to_embedding"],
            use_precomputed_gazetteer=state["use_precomputed_gazetteer"],
            gazetteer_untagged_label_name=state["gazetteer_untagged_label_name"],
            gazetteer_include_untagged=state["gazetteer_include_untagged"],
            #decoder=state["decoder"],

            **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type
