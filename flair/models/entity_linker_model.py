import logging
import random
from typing import List, Optional, Union

import torch
import torch.nn as nn

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence, SpanLabel

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
        pooling_operation: str = "first&last",
        label_type: str = "nel",
        dropout: float = 0.5,
        skip_unk_probability: Optional[float] = None,
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

        super(EntityLinker, self).__init__(label_dictionary, **classifierargs)

        self.word_embeddings = word_embeddings
        self.pooling_operation = pooling_operation
        self._label_type = label_type
        self.skip_unk_probability = skip_unk_probability
        if self.skip_unk_probability:
            self.known_entities = label_dictionary.get_items()

        # ----- Dropout parameters -----
        # dropouts
        self.use_dropout: float = dropout
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        # if we concatenate the embeddings we need double input size in our linear layer
        if self.pooling_operation == "first&last":
            self.decoder = nn.Linear(2 * self.word_embeddings.embedding_length, len(self.label_dictionary)).to(
                flair.device
            )
        else:
            self.decoder = nn.Linear(self.word_embeddings.embedding_length, len(self.label_dictionary)).to(flair.device)

        nn.init.xavier_uniform_(self.decoder.weight)

        cases = {
            "average": self.emb_mean,
            "first": self.emb_first,
            "last": self.emb_last,
            "first&last": self.emb_firstAndLast,
        }

        if pooling_operation not in cases:
            raise KeyError('pooling_operation has to be one of "average", "first", "last" or "first&last"')

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
        return_label_candidates: bool = False,
    ):

        if not isinstance(sentences, list):
            sentences = [sentences]

        # filter sentences with no candidates (no candidates means nothing can be linked anyway)
        filtered_sentences = []
        for sentence in sentences:
            if sentence.get_labels(self.label_type):
                filtered_sentences.append(sentence)

        # fields to return
        span_labels = []
        sentences_to_spans = []
        empty_label_candidates = []

        # if the entire batch has no sentence with candidates, return empty
        if len(filtered_sentences) == 0:
            scores = None

        # otherwise, embed sentence and send through prediction head
        else:
            # embed all tokens
            self.word_embeddings.embed(filtered_sentences)

            embedding_names = self.word_embeddings.get_names()

            embedding_list = []
            # get the embeddings of the entity mentions
            for sentence in filtered_sentences:
                entities = sentence.get_labels(self.label_type)

                for entity in entities:

                    if self.skip_unk_probability and self.training and entity.value not in self.known_entities:
                        sample = random.uniform(0, 1)
                        if sample < self.skip_unk_probability:
                            continue

                    span_labels.append([entity.value])

                    if self.pooling_operation == "first&last":
                        mention_emb = torch.cat(
                            (
                                entity.span.tokens[0].get_embedding(embedding_names),
                                entity.span.tokens[-1].get_embedding(embedding_names),
                            ),
                            0,
                        )
                    embedding_list.append(mention_emb.unsqueeze(0))

                    if return_label_candidates:
                        sentences_to_spans.append(sentence)
                        candidate = SpanLabel(span=entity.span, value=None, score=0.0)
                        empty_label_candidates.append(candidate)

            if len(embedding_list) > 0:
                embedding_tensor = torch.cat(embedding_list, 0).to(flair.device)

                if self.use_dropout:
                    embedding_tensor = self.dropout(embedding_tensor)

                scores = self.decoder(embedding_tensor)
            else:
                scores = None

        if return_label_candidates:
            return scores, span_labels, sentences_to_spans, empty_label_candidates

        return scores, span_labels

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "word_embeddings": self.word_embeddings,
            "label_type": self.label_type,
            "label_dictionary": self.label_dictionary,
            "pooling_operation": self.pooling_operation,
            "loss_weights": self.weight_dict,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = EntityLinker(
            word_embeddings=state["word_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            pooling_operation=state["pooling_operation"],
            loss_weights=state["loss_weights"] if "loss_weights" in state else {"<unk>": 0.3},
        )

        model.load_state_dict(state["state_dict"])
        return model

    @property
    def label_type(self):
        return self._label_type
