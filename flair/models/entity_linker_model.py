import logging
from typing import List, Union

import torch

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence

log = logging.getLogger("flair")


class EntityLinker(flair.nn.DefaultClassifier[Sentence]):
    """
    Entity Linking Model
    The model expects text/sentences with annotated entity mentions and predicts entities to these mentions.
    To this end a word embedding is used to embed the sentences and the embedding of the entity mention goes through a linear layer to get the actual class label.
    The model is able to predict '<unk>' for entity mentions that the model can not confidently match to any of the known labels.
    """

    def __init__(
        self,
        word_embeddings: flair.embeddings.TokenEmbeddings,
        label_dictionary: Dictionary,
        pooling_operation: str = "first_last",
        label_type: str = "nel",
        **classifierargs,
    ):
        """
        Initializes an EntityLinker
        :param word_embeddings: embeddings used to embed the words/sentences
        :param label_dictionary: dictionary that gives ids to all classes. Should contain <unk>
        :param pooling_operation: either 'average', 'first', 'last' or 'first&last'. Specifies the way of how text representations of entity mentions (with more than one word) are handled.
        E.g. 'average' means that as text representation we take the average of the embeddings of the words in the mention. 'first&last' concatenates
        the embedding of the first and the embedding of the last word.
        :param label_type: name of the label you use.
        """

        super(EntityLinker, self).__init__(
            label_dictionary=label_dictionary,
            final_embedding_size=word_embeddings.embedding_length * 2
            if pooling_operation == "first_last"
            else word_embeddings.embedding_length,
            **classifierargs,
        )

        self.word_embeddings = word_embeddings
        self.pooling_operation = pooling_operation
        self._label_type = label_type

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

        # filter sentences with no candidates (no candidates means nothing can be linked anyway)
        filtered_sentences = []
        for sentence in sentences:
            if sentence.get_labels(self.label_type):
                filtered_sentences.append(sentence)

        # fields to return
        data_points = []
        span_labels = []
        embedded_entity_pairs = None

        # embed sentences and send through prediction head
        if len(filtered_sentences) > 0:
            # embed all tokens
            self.word_embeddings.embed(filtered_sentences)

            embedding_names = self.word_embeddings.get_names()

            embedding_list = []
            # get the embeddings of the entity mentions
            for sentence in filtered_sentences:
                entities = sentence.get_spans(self.label_type)

                for entity in entities:

                    # get the label of the entity
                    span_labels.append([entity.get_label(self.label_type).value])

                    if self.pooling_operation == "first_last":
                        mention_emb = torch.cat(
                            (
                                entity.tokens[0].get_embedding(embedding_names),
                                entity.tokens[-1].get_embedding(embedding_names),
                            ),
                            0,
                        )
                    embedding_list.append(mention_emb.unsqueeze(0))

                if for_prediction:
                    data_points.extend(entities)

            if len(embedding_list) > 0:
                embedded_entity_pairs = torch.cat(embedding_list, 0)

        if for_prediction:
            return embedded_entity_pairs, span_labels, data_points

        return embedded_entity_pairs, span_labels

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
