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
            word_embeddings=state.get("word_embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            pooling_operation=state.get("pooling_operation"),
            loss_weights=state.get("loss_weights", {"<unk>": 0.3}),
            **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type


class CandidateGenerationMethod():

    def __init__(self, candidate_dict):
        self.candidate_dict = candidate_dict

        # create a dictionary where blanks are removed
        # this leads to some distinct mentions being the same afterwards
        # thus we unify their lists
        simpler_mentions_candidate_dict = {}
        for mention in candidate_dict:
            # create mention without blanks
            simplified_mention=mention.replace(' ','').lower()
            # the simplified mention already occurred from another mention
            if simplified_mention in simpler_mentions_candidate_dict:
                candidates = set(simpler_mentions_candidate_dict[simplified_mention])
                candidates.update(candidate_dict[mention])
                simpler_mentions_candidate_dict[simplified_mention] = list(candidates)
            # its the first occurrence of the simplified mention
            else:
                simpler_mentions_candidate_dict[simplified_mention] = candidate_dict[mention]

        self.simpler_mentions_candidate_dict = simpler_mentions_candidate_dict

        self.punc_remover = re.compile(r"[\W]+")


        even_more_simpler_mentions_candidate_dict = {}
        for mention in candidate_dict:
            # create mention without blanks
            simplified_mention=self.punc_remover.sub("", mention.lower())
            # the simplified mention already occurred from another mention
            if simplified_mention in even_more_simpler_mentions_candidate_dict:
                candidates = set(even_more_simpler_mentions_candidate_dict[simplified_mention])
                candidates.update(candidate_dict[mention])
                even_more_simpler_mentions_candidate_dict[simplified_mention] = list(candidates)
            # its the first occurrence of the simplified mention
            else:
                even_more_simpler_mentions_candidate_dict[simplified_mention] = candidate_dict[mention]

        self.even_more_simpler_mentions_candidate_dict = even_more_simpler_mentions_candidate_dict

    def get_candidates(self, mentions):
        candidate_set = set()
        for mention in mentions:
            try:
               candidate_set.update(self.candidate_dict[mention])
            except KeyError:
                # check if mention without blanks exists
                try:
                    mention_without_blanks = mention.replace(' ','').lower()
                    candidate_set.update(self.simpler_mentions_candidate_dict[mention_without_blanks])
                except KeyError:
                    # check if even more simplified mention exists
                    even_more_simplified_mention = self.punc_remover.sub("", mention.lower())
                    try:
                        candidate_set.update(self.even_more_simpler_mentions_candidate_dict[even_more_simplified_mention])
                    except KeyError:
                        #possible 'fixes': sample 15 random entities or sample candidates with string similarity (like 15 most similar)
                        pass
                        #print('Unknown mention: '+ mention)
        return candidate_set


class CandidateDecoder(torch.nn.Module):

    def __init__(self,
             entity_embedding_size: int,
             mention_embedding_size: int,
             entity_dictionary,
             candidate_generation_method
    ):
        # TODO: It would be nice if the candidate generation method would return indices directly
        super().__init__()

        self.entity_dictionary = entity_dictionary # index for each entity (title or id)

        self.mention_to_entity = torch.nn.Linear(mention_embedding_size, entity_embedding_size) # project mention embedding to entity embedding space

        self.entity_embeddings = torch.nn.Linear(entity_embedding_size ,len(entity_dictionary), bias=False) # each entity is represented by a vector
        # TODO: more sophisticated way than random for initialization for entity vectors??

        self.candidate_generation_method = candidate_generation_method

        #self.to(flair.device)

    def forward(self,mention_embeddings, mentions):

        # create the restricted set of entities for which we compute the scores batch-wise
        scoring_entity_set = self.candidate_generation_method.get_candidates(mentions)

        #print(scoring_entity_set)

        # if no entity set is given, score over all entities
        if scoring_entity_set:
            indices_of_scoring_entities = [self.entity_dictionary.get_idx_for_item(entity) for entity in scoring_entity_set]

            #print(indices_of_scoring_entities)

        #project mentions in entity representation
        #print(mentions)
        #print(mention_embeddings.size())
        projected_mention_embeddings = self.mention_to_entity(mention_embeddings)
        #print(projected_mention_embeddings.size())
        # compute scores
        logits = self.entity_embeddings(projected_mention_embeddings)
        #print(logits.size())
        # if not scoring over all entities we return the corresponding indices
        if scoring_entity_set:
            return logits, indices_of_scoring_entities
        else:
            return logits, None