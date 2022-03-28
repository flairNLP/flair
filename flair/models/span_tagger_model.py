import logging
from typing import List, Union

import torch

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence, Span

log = logging.getLogger("flair")


class SpanTagger(flair.nn.DefaultClassifier[Sentence]):
    """
    Span Tagger
    The model expects text/sentences with annotated tags on token level (e.g. NER), and learns to predict spans.
    All possible combinations of spans (up to a defined max length) are considered, represented via concatenation
    of the word embeddings of the first and last token in the span. Then fed through a linear layer to get the actual class label.
    """

    def __init__(
        self,
        word_embeddings: flair.embeddings.TokenEmbeddings,
        label_dictionary: Dictionary,
        pooling_operation: str = "first_last",
        label_type: str = "ner",
        max_span_length: int = 5,
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
        """

        # make sure the label dictionary has an "O" entry for "no tagged span"
        label_dictionary.add_item("O")

        super(SpanTagger, self).__init__(
            label_dictionary=label_dictionary,
            final_embedding_size=word_embeddings.embedding_length * 2
            if pooling_operation == "first_last"
            else word_embeddings.embedding_length,
            **classifierargs,
        )

        self.word_embeddings = word_embeddings
        self.pooling_operation = pooling_operation
        self._label_type = label_type
        self.max_span_length = max_span_length

        cases = {
            "average": self.emb_mean,
            "first": self.emb_first,
            "last": self.emb_last,
            "first_last": self.emb_firstAndLast,
        }

        if pooling_operation not in cases:
            raise KeyError('pooling_operation has to be one of "average", "first", "last" or "first_last"')

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

    def forward_pass(
        self,
        sentences: Union[List[Sentence], Sentence],
        for_prediction: bool = False,
    ):

        if not isinstance(sentences, list):
            sentences = [sentences]

        self.word_embeddings.embed(sentences)
        names = self.word_embeddings.get_names()

        # fields to return
        spans_embedded = None
        spans_labels = []
        data_points = []
        #spans = [] #TODO not really used anywhere... nice for debugging

        embedding_list = []
        for sentence in sentences:

            # create list of all possible spans (consider given max_span_length)
            spans_sentence = []
            tokens = [token for token in sentence]
            for span_len in range(1, self.max_span_length+1):
                spans_sentence.extend([Span(tokens[n:n + span_len])
                                       for n in range(len(tokens) - span_len)])

            # embed each span (concatenate embeddings of first and last token)
            for span in spans_sentence:
                if self.pooling_operation == "first_last":
                    span_embedding = torch.cat(
                        (span[0].get_embedding(names),
                         span[-1].get_embedding(names)),
                        0,)

                if self.pooling_operation == "average":
                    span_embedding = torch.mean(
                        torch.stack([span[i].get_embedding(names) for i in range(len(span.tokens))])
                        ,0)

                if self.pooling_operation == "first":
                    span_embedding = span[0].get_embedding(names)

                if self.pooling_operation == "last":
                    span_embedding = span[-1].get_embedding(names)

                embedding_list.append(span_embedding.unsqueeze(0))

                # use the span gold labels
                spans_labels.append([span.get_label(self.label_type).value])

            #spans.extend(spans_sentence)

            if for_prediction:
                data_points.extend(spans_sentence)

        if len(embedding_list) > 0:
            spans_embedded = torch.cat(embedding_list, 0)

        if for_prediction:
            return spans_embedded, spans_labels, data_points

        return spans_embedded, spans_labels

    def _post_process_predictions(self, batch):
        """
        Post processing the span predictions to avoid overlapping predictions.
        Only use the most confident one, i.e. sort the span predictions by confidence, go through them, for each token
        check if it was already part of a used span.
        #TODO at the moment this makes nested NER impossible, maybe change skipping criterion?

        :param batch: batch of sentences with already predicted span labels to be "cleaned"
        """

        import operator

        for sentence in batch:
            all_predicted_spans = []

            # get all predicted spans and their confidence score, sort them afterwards

            # TODO: which of the following is the right choice? this works for training/evaluating but not during model.predict()
            for span in sentence.get_spans("predicted"):
                span_tokens = span.tokens
                span_score = span.get_label("predicted").score
                span_prediction = span.get_label("predicted").value
                all_predicted_spans.append((span_tokens, span_prediction, span_score))

            # this would work in predict()
            # BUT: in training this would use only the gold labels (they have score 1.0)...
            #for span in sentence.get_spans(self.label_type):
            #    span_tokens = span.tokens
            #    span_score = span.get_label(self.label_type).score
            #    span_prediction = span.get_label(self.label_type).value
            #    all_predicted_spans.append((span_tokens, span_prediction, span_score))

            sentence.remove_labels("predicted") #TODO good? or need to do self.label_type instead?
            #sentence.remove_labels(self.label_type) # TODO: but then, would the gold labels be removed as well...?

            already_seen_token_indices: List[int] = []

            # sort by confidence score
            sorted_predicted_spans = sorted(all_predicted_spans, key=operator.itemgetter(2))
            sorted_predicted_spans.reverse()
            #print(sorted_predicted_spans)

            # starting with highest scored span prediction
            for predicted_span in sorted_predicted_spans:
                # print(predicted_span)
                span_tokens, span_prediction, span_score = predicted_span

                # check whether any token in this span already has been labeled
                # TODO: maybe think about better idea so that nested NER would be allowed?
                tag_span = True
                for token in span_tokens:
                    if token is None or token.idx in already_seen_token_indices:
                        tag_span = False
                        #print("skipping stuff...")
                        continue

                # only add if none of the token is part of an already (so "higher") labeled span
                if tag_span:
                    already_seen_token_indices.extend(token.idx for token in span_tokens)
                    predicted_span = Span(
                        [sentence.get_token(token.idx) for token in span_tokens]
                    )
                    predicted_span.add_label("predicted", value=span_prediction, score=span_score) #TODO same question
                    #predicted_span.add_label(self.label_type, value=span_prediction, score=span_score)

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "word_embeddings": self.word_embeddings,
            "label_type": self.label_type,
            "label_dictionary": self.label_dictionary,
            "pooling_operation": self.pooling_operation,
            "loss_weights": self.weight_dict,
        }
        return model_state

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

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            word_embeddings=state["word_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            pooling_operation=state["pooling_operation"],
            loss_weights=state["loss_weights"] if "loss_weights" in state else {"<unk>": 0.3},
            **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type
