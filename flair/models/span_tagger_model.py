import logging
from typing import List, Union

import csv
import torch

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
        concat_span_length_to_embedding: bool = False,
        resolve_overlaps: str = "keep_overlaps",
        gazetteer_file: str = None,
        gazetteer_count_type: str = "rel",
        gazetteer_concat_confidence_score: bool = True,
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
        :param pooling_operation: either 'average', 'first', 'last' or 'first&last'. Specifies the way of how text representations of entity spans (with more than one word) are handled.
        E.g. 'average' means that as text representation we take the average of the embeddings of the words in the span. 'first&last' concatenates
        the embedding of the first and the embedding of the last word.
        :param label_type: name of the label you use.
        :param max_span_length: maximum length of spans (in tokens) that are considered
        :param concat_span_length_to_embedding: if set to True span length is concatenated to span embeddings
        :param resolve_overlaps: one of
            'keep_overlaps' : overlapping predictions stay as they are (i.e. not using _post_process_predictions())
            'by_token' : only allow one prediction per token/span
            'no_boundary_clashes' : predictions cannot overlap boundaries, but can include other predictions (nested NER)
            'prefer_longer' : #TODO implement this. Somehow favour longer span predictions over shorter ones?
        :param gazetteer_count_type: in use only if gazetteer_file is given:
            "abs": absolute counts,
            "rel": normalized counts (sum to 1) (note: frequency info gets lost!)
            #TODO: more possibilities?
        :param gazetteer_concat_confidence_score: set to True to concatenate a confidence score to gazetteer features,
            based on absolute frequency in gazetteer, especially when using "rel", "one_hot" or "multi_hot"
        :param gazetteer_file: path to a csv file containing a gazetteer list with span strings in rows, label names in columns, counts in cells
        :param ignore_embeddings: simple baseline: just use gazetteer embedding, so ignore embeddings
        :param use_mlp: use a MLP as output layer (instead of default linear layer + xavier transformation), may be helpful for interactions with gazetteer
        :param mlp_hidden_dim: size of the hidden layer in output MLP
        """

        # make sure the label dictionary has an "O" entry for "no tagged span"
        label_dictionary.add_item("O")

        final_embedding_size = word_embeddings.embedding_length * 2 \
            if pooling_operation == "first_last" else word_embeddings.embedding_length

        self.gazetteer_file = gazetteer_file
        self.gazetteer_count_type = gazetteer_count_type
        self.ignore_embeddings = ignore_embeddings

        if self.ignore_embeddings:
            final_embedding_size = 0

        self.concat_span_length_to_embedding = concat_span_length_to_embedding
        if self.concat_span_length_to_embedding:
            final_embedding_size += 1

        if self.gazetteer_file:
            print("Reading gazetteer file:", self.gazetteer_file)
            self.gazetteer = {}
            with open(self.gazetteer_file, mode='r') as inp:
                reader = csv.reader(inp)
                header = next(reader)  # header line
                self.nr_gazetteer_tags = len(header) - 1  # nr of tags (exclude first column, i.e. span_string)
                self.gazetteer = {row[0]: list(map(float, row[1:])) for row in reader}  # read rest in dict
            final_embedding_size += self.nr_gazetteer_tags

        self.gazetteer_concat_confidence_score = gazetteer_concat_confidence_score
        if self.gazetteer_concat_confidence_score:
            final_embedding_size += 1

        #if self.gazetteer_count_type == "rel_add_freq_info":
        #    final_embedding_size += 1

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

    def get_gazetteer_embedding(self, span_string: str,
                                count_type="abs") -> torch.Tensor:

        if span_string in self.gazetteer:
            vector = torch.Tensor(self.gazetteer[span_string]).to(flair.device)
        else:
            vector = torch.zeros(self.nr_gazetteer_tags).to(flair.device)  # get same length zero vector

        if count_type == "abs":
            gaz_vector = vector

        if count_type == "rel":
            if torch.sum(vector) > 0:  # avoid zero division
                gaz_vector = vector / torch.sum(vector)
            else:
                gaz_vector = vector

        if count_type == "one_hot":
            one_hot = torch.zeros(len(vector)).to(flair.device)
            if torch.sum(vector) > 0:  # necessary to avoid torch.argmax returning id 0
                one_hot[torch.argmax(vector)] = 1
            gaz_vector = one_hot

        if count_type == "multi_hot":
            if torch.sum(vector) > 0:  # avoid zero division
                rel_vector = vector / torch.sum(vector)
            else:
                rel_vector = vector
            THRESHOLD = 0.3
            multi_hot = (rel_vector >= THRESHOLD).type(torch.int8)
            gaz_vector = multi_hot

        if self.gazetteer_concat_confidence_score:
            sum_count = torch.sum(vector)
            DEFINED_MAX = 50  # TODO: what to choose here? make parameter
            confidence = torch.tensor([min(sum_count / DEFINED_MAX, 1)]).to(flair.device)
            gaz_vector = torch.cat((gaz_vector, confidence), 0)

        return gaz_vector

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
                    gazetteer_vector = self.get_gazetteer_embedding(span.text, count_type=self.gazetteer_count_type)
                    span_embedding = torch.cat((span_embedding, gazetteer_vector), 0)

                embedding_list.append(span_embedding.unsqueeze(0))

                # use the span gold labels
                spans_labels.append([span.get_label(self.label_type).value])

                # check if everything looks as it should
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
                    span_prediction = span.get_label(label_type).value
                    all_predicted_spans.append((span_tokens, span_prediction, span_score))

                sentence.remove_labels(label_type)  # first remove the predicted labels

                # sort by confidence score
                sorted_predicted_spans = sorted(all_predicted_spans, key=operator.itemgetter(2))
                sorted_predicted_spans.reverse()
                #for s in sorted_predicted_spans:
                #    print(s)

                if self.resolve_overlaps == "by_token":
                    # in short: if a token already was part of a higher ranked span, break
                    already_seen_token_indices: List[int] = []

                    # starting with highest scored span prediction
                    for predicted_span in sorted_predicted_spans:
                        span_tokens, span_prediction, span_score = predicted_span

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
                        span_tokens, span_prediction, span_score = predicted_span
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

                if self.resolve_overlaps == "prefer_longer":
                    # TODO: somehow favour longer spans over shorter ones...? how to combine that with ranking by score
                    raise NotImplementedError


    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "word_embeddings": self.word_embeddings,
            "label_type": self.label_type,
            "label_dictionary": self.label_dictionary,
            "pooling_operation": self.pooling_operation,
            "loss_weights": self.weight_dict,
            "resolve_overlaps": self.resolve_overlaps,
            "gazetteer_file": self.gazetteer_file if self.gazetteer_file else None,
            "gazetteer_count_type": self.gazetteer_count_type,
            "gazetteer_concat_confidence_score": self.gazetteer_concat_confidence_score,
            "max_span_length": self.max_span_length,
            "ignore_embeddings": self.ignore_embeddings,
            "use_mlp": self.use_mlp,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "concat_span_length_to_embedding": self.concat_span_length_to_embedding,
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
            gazetteer_file=state["gazetteer_file"] if "gazetteer_file" in state else None,
            gazetteer_count_type=state["gazetteer_count_type"],
            gazetteer_concat_confidence_score=state["gazetteer_concat_confidence_score"],
            ignore_embeddings=state["ignore_embeddings"],
            max_span_length=state["max_span_length"],
            use_mlp=state["use_mlp"],
            mlp_hidden_dim=state["mlp_hidden_dim"],
            concat_span_length_to_embedding=state["concat_span_length_to_embedding"],
            #decoder=state["decoder"],

        **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type
