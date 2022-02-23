import logging
import random
from typing import List, Optional, Union

import torch

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence, SpanLabel

log = logging.getLogger("flair")

class candidate_decoder(torch.nn.Module):

    def __init__(self,
             entity_embedding_size: int,
             mention_embedding_size: int,
             entity_dictionary: Dictionary,
             candidate_generation_method: dict # for now I use a dictionary, one could possibly extend this to a general method
    ):
        # TODO: It would be nice if candidate generation method would return indices directly
        super().__init__()

        self.entity_dictionary = entity_dictionary

        if entity_embedding_size!=mention_embedding_size:
            self.mention_to_entity = torch.nn.Linear(mention_embedding_size, entity_embedding_size)
        else:
            self.mention_to_entity=None

        self.entity_embeddings = torch.randn( len(entity_dictionary), entity_embedding_size, requires_grad=True, device=flair.device) # TODO: more sophisticated way than random for initialization??
        self.bias = torch.randn(len(entity_dictionary), requires_grad=True, device=flair.device)

        self.candidate_generation_method = candidate_generation_method

    def forward(self,mention_embeddings, labels, mentions):

        # get the set of all label candidates for all mentions
        restricted_label_set = set(labels)


        for mention in mentions:
            restricted_label_set.update(self.candidate_generation_method[mention]) # TODO: Make sure all the generated candidates are actually in the label set??!!

        indices_of_labels = [self.entity_dictionary.get_idx_for_item(entity) for entity in restricted_label_set]

        #project mentions in entity representation space if necessary
        if self.mention_to_entity:
            mention_embeddings = self.mention_to_entity(mention_embeddings)

        # compute scores of mentions w.r.t. restricted set of entities
        scores = torch.matmul(self.entity_embeddings[indices_of_labels,:],torch.transpose(mention_embeddings,1,0)) + self.bias[indices_of_labels].unsqueeze(1)
        print(scores.size())
        return scores

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
        candidate_generation_method: dict = None, # Idea: If one provides a candidate generation method, then a costum decoder is handed to DefaultClassifier
        entitiy_embedding_size: int = None,
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

        # ------------------------------------------NEWNEWNENWENWENWENWEWNEWN--------------------------------
        if candidate_generation_method:

            mention_embedding_size = 2 * word_embeddings.embedding_length if pooling_operation == 'first&last' else word_embeddings.embedding_length

            if not entitiy_embedding_size:
                entity_embedding_size = mention_embedding_size

            # update the label_dictionary with the candidate_generation method
            # previously I had the problem that when using only the entities from AIDA-train in the label_dictionary
            # and not adding the ones of the candidate generation method, that almost none of the proposed candidates was conatined in the label_dictionary
            # this way the generated lists deteriorate to one candidate (the gold entity) in the worst case, and the model would not learn to distinguish in that case

            # I assume that the generation method is a dictionary again for now
            for key in candidate_generation_method:
                for candidate in candidate_generation_method[key]:
                    label_dictionary.add_item(candidate)

            decoder = candidate_decoder(entity_embedding_size=entity_embedding_size,
                                        mention_embedding_size=mention_embedding_size,
                                        entity_dictionary=label_dictionary,  # TODO: Do I need to remove <unk>??
                                        candidate_generation_method=candidate_generation_method)
        else:
            decoder = None

        # ---------------------------------------------------------------------------------------------------

        super(EntityLinker, self).__init__(
            label_dictionary=label_dictionary,
            final_embedding_size=word_embeddings.embedding_length * 2
            if pooling_operation == "first&last"
            else word_embeddings.embedding_length,
            decoder=decoder,
            ** classifierargs,
        )

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

    def decode(self, ):
        pass

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
        spans = []
        sentences_to_spans = []
        empty_label_candidates = []
        embedded_entity_pairs = None

        # embed sentences and send through prediction head
        if len(filtered_sentences) > 0:
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
                    spans.append(entity.span.text)

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
                embedded_entity_pairs = torch.cat(embedding_list, 0)

                if self.use_dropout:
                    embedded_entity_pairs = self.dropout(embedded_entity_pairs)

        if return_label_candidates:
            return embedded_entity_pairs, span_labels, sentences_to_spans, empty_label_candidates

        return embedded_entity_pairs, span_labels, spans

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
