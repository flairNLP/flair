import logging
from typing import List, Union

import torch
import torch.nn as nn

import flair.embeddings
import flair.nn
from flair.data import DataPoint, Dictionary, SpanLabel

log = logging.getLogger("flair")


class EntityLinker(flair.nn.DefaultClassifier):
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
            pooling_operation: str = 'average',
            label_type: str = 'nel',
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

        # if we concatenate the embeddings we need double input size in our linear layer
        if self.pooling_operation == 'first&last':
            self.decoder = nn.Linear(
                2 * self.word_embeddings.embedding_length, len(self.label_dictionary)
            ).to(flair.device)
        else:
            self.decoder = nn.Linear(
                self.word_embeddings.embedding_length, len(self.label_dictionary)
            ).to(flair.device)

        nn.init.xavier_uniform_(self.decoder.weight)

        cases = {
            'average': self.emb_mean,
            'first': self.emb_first,
            'last': self.emb_last,
            'first&last': self.emb_firstAndLast
        }

        if pooling_operation not in cases:
            raise KeyError('pooling_operation has to be one of "average", "first", "last" or "first&last"')

        self.aggregated_embedding = cases.get(pooling_operation)

        self.to(flair.device)

    def emb_first(self, arg):
        return arg[0]

    def emb_last(self, arg):
        return arg[-1]

    def emb_firstAndLast(self, arg):
        return torch.cat((arg[0], arg[-1]), 0)

    def emb_mean(self, arg):
        return torch.mean(arg, 0)

    def forward_pass(self,
                     sentences: Union[List[DataPoint], DataPoint],
                     return_label_candidates: bool = False,
                     ):

        if isinstance(sentences, DataPoint):
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
                spans = sentence.get_spans(self.label_type)

                for span in spans:
                    mention_emb = torch.Tensor(0, self.word_embeddings.embedding_length).to(flair.device)

                    for token in span.tokens:
                        mention_emb = torch.cat((mention_emb, token.get_embedding(embedding_names).unsqueeze(0)), 0)

                    embedding_list.append(self.aggregated_embedding(mention_emb).unsqueeze(0))

                    span_labels.append([label.value for label in span.get_labels(typename=self.label_type)])

                    if return_label_candidates:
                        sentences_to_spans.append(sentence)
                        candidate = SpanLabel(span=span, value=None, score=None)
                        empty_label_candidates.append(candidate)

            embedding_tensor = torch.cat(embedding_list, 0).to(flair.device)
            scores = self.decoder(embedding_tensor)

        # minimal return is scores and labels
        return_tuple = (scores, span_labels)

        if return_label_candidates:
            return_tuple += (sentences_to_spans, empty_label_candidates)

        return return_tuple

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "word_embeddings": self.word_embeddings,
            "label_type": self.label_type,
            "label_dictionary": self.label_dictionary,
            "pooling_operation": self.pooling_operation,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = EntityLinker(
            word_embeddings=state["word_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            pooling_operation=state["pooling_operation"],
        )

        model.load_state_dict(state["state_dict"])
        return model

    @property
    def label_type(self):
        return self._label_type
