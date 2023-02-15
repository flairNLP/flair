import logging
from typing import Callable, Dict, List, Optional, Union, Set

import torch

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence, Span

log = logging.getLogger("flair")


class EntityLinker(flair.nn.DefaultClassifier[Sentence, Span]):
    """
    Entity Linking Model
    The model expects text/sentences with annotated entity mentions and predicts entities to these mentions.
    To this end a word embedding is used to embed the sentences and the embedding of the entity mention goes through a linear layer to get the actual class label.
    The model is able to predict '<unk>' for entity mentions that the model can not confidently match to any of the known labels.
    """

    def __init__(
            self,
            embeddings: flair.embeddings.TokenEmbeddings,
            label_dictionary: Dictionary,
            pooling_operation: str = "first_last",
            label_type: str = "nel",
            candidates: Optional[Union[str, dict]] = None,
            **classifierargs,
    ):
        """
        Initializes an EntityLinker
        :param embeddings: embeddings used to embed the words/sentences
        :param label_dictionary: dictionary that gives ids to all classes. Should contain <unk>
        :param pooling_operation: either 'average', 'first', 'last' or 'first&last'. Specifies the way of how text representations of entity mentions (with more than one word) are handled.
        E.g. 'average' means that as text representation we take the average of the embeddings of the words in the mention. 'first&last' concatenates
        the embedding of the first and the embedding of the last word.
        :param label_type: name of the label you use.
        """

        super(EntityLinker, self).__init__(
            embeddings=embeddings,
            label_dictionary=label_dictionary,
            final_embedding_size=embeddings.embedding_length * 2 if pooling_operation == "first_last" \
                else embeddings.embedding_length,
            **classifierargs,
        )

        self.pooling_operation = pooling_operation
        self._label_type = label_type

        cases: Dict[str, Callable[[Span, List[str]], torch.Tensor]] = {
            "average": self.emb_mean,
            "first": self.emb_first,
            "last": self.emb_last,
            "first_last": self.emb_firstAndLast,
        }

        if pooling_operation not in cases:
            raise KeyError('pooling_operation has to be one of "average", "first", "last" or "first_last"')

        self.aggregated_embedding = cases[pooling_operation]

        self.candidates = candidates

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

    def _get_data_points_from_sentence(self, sentence: Sentence) -> List[Span]:
        return sentence.get_spans(self.label_type)

    def _filter_data_point(self, data_point: Sentence) -> bool:
        return bool(data_point.get_labels(self.label_type))

    def _get_embedding_for_data_point(self, prediction_data_point: Span) -> torch.Tensor:
        return self.aggregated_embedding(prediction_data_point, self.embeddings.get_names())

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "word_embeddings": self.embeddings.save_embeddings(use_state_dict=False),
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
        # remap state dict for models serialized with Flair <= 0.11.3
        import re

        state_dict = state["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[re.sub("^word_embeddings\\.", "embeddings.", key)] = state_dict.pop(key)

        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("word_embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            pooling_operation=state.get("pooling_operation"),
            loss_weights=state.get("loss_weights", {"<unk>": 0.3}),
            **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type

    def _mask_scores(self, scores: torch.Tensor, data_points: List[Span]):

        if not self.candidates:
            return scores

        masked_scores = -torch.inf * torch.ones(scores.size(), requires_grad=True, device=flair.device)

        for idx, span in enumerate(data_points):
            candidate_set = self.candidates[span.text] if span.text in self.candidates else []
            print(span, candidate_set)
            indices_of_candidates = [self.label_dictionary.get_idx_for_item(candidate) for candidate in candidate_set]
            masked_scores[idx, indices_of_candidates] = scores[idx, indices_of_candidates]

        return masked_scores


class CandidateGenerator:
    """Abstract base class for methods that, given a mention,
    generate a set of candidates, so that the EntityLinker only
    scores among these candidates and not all entities
    """

    def __init__(self, candidates: Union[str, Dict], lower_case: bool = True):

        self.candidate_lists = candidates
        self.lower_case = lower_case

        if self.lower_case:
            for mention in candidates:
                if mention.lower() in self.candidate_lists:
                    known_mentions = self.candidate_lists[mention.lower()]
                    self.candidate_lists[mention.lower()] = list(set(known_mentions + candidates[mention]))
                else:
                    self.candidate_lists[mention.lower()] = candidates[mention]

    def get_candidates(self, mention: str) -> Set[str]:
        """Given a list of entity mentions this methods returns a constrained set of entity
        candidates for the mentions"""
        if self.lower_case:
            mention = mention.lower()

        return self.candidate_lists[mention]