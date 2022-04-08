import itertools
from typing import Tuple, List, Set, Dict, Iterator, Sequence, NamedTuple, Union, Optional

import torch

from flair.data import Sentence, DataPoint, Span, Dictionary, Label
from flair.embeddings import DocumentEmbeddings
from flair.models import TextClassifier


class _RelationArgument(NamedTuple):
    """A `_RelationArgument` encapsulates either a relation's head or a tail span, including its label."""
    entity: Span
    label: Label


# TODO: This closely shadows the RelationExtractor name. Maybe we need a better name here.
#  - EntityPairRelationClassifier ?
#  - MaskedRelationClassifier ?
class RelationClassifier(TextClassifier):

    def __init__(self,
                 document_embeddings: DocumentEmbeddings,
                 label_dictionary: Dictionary,
                 label_type: str,
                 entity_label_types: Union[str, Sequence[str], Dict[str, Optional[Set[str]]]],
                 relations: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
                 **classifierargs) -> None:
        """
        TODO: Add docstring
        # This does not yet support entities with two labels at the same span.
        # Supports only directional relations, self referencing relations are not supported
        :param document_embeddings:
        :param label_dictionary:
        :param label_type:
        :param entity_label_types:
        :param relations:
        :param classifierargs:
        """
        super().__init__(document_embeddings, label_type, label_dictionary=label_dictionary, **classifierargs)

        if isinstance(entity_label_types, str):
            self.entity_label_types: Dict[str, Optional[Set[str]]] = {entity_label_types: None}
        elif isinstance(entity_label_types, Sequence):
            self.entity_label_types: Dict[str, Optional[Set[str]]] = {entity_label_type: None
                                                                      for entity_label_type in entity_label_types}
        else:
            self.entity_label_types: Dict[str, Optional[Set[str]]] = entity_label_types

        self.relations = relations

        # Control mask templates
        self._head_mask: str = '[H-ENTITY]'
        self._tail_mask: str = '[T-ENTITY]'

    def _entity_pair_permutations(self, sentence: Sentence) -> Iterator[Tuple[_RelationArgument, _RelationArgument]]:
        """
        Yields all valid entity pair permutations.
        The permutations are constructed by a filtered cross-product
        under the specifications of `self.entity_label_types` and `self.relations`.
        :param sentence: A flair `Sentence` object with entity labels
        :return: Tuples of (<HEAD>, <TAIL>) `_RelationArguments`
        """
        entities: Iterator[_RelationArgument] = itertools.chain.from_iterable([  # Flatten nested 2D list
            (
                _RelationArgument(entity=entity, label=entity.get_label(label_type=label_type))
                for entity in sentence.get_spans(type=label_type)
                # Only use entities labelled with the specified labels for each label type
                if labels is None or entity.get_label(label_type=label_type).value in labels
            )
            for label_type, labels in self.entity_label_types.items()
        ])

        # Yield head and tail entity pairs from the cross product of all entities
        for head, tail in itertools.product(entities, repeat=2):

            # Remove identity relation entity pairs
            if head.entity is tail.entity:
                continue

            # Remove entity pairs with labels that do not match any of the specified relations in `self.relations`
            if self.relations is not None and all((head.label.value, tail.label.value) not in pairs
                                                  for pairs in self.relations.values()):
                continue

            yield head, tail

    def _label_aware_head_mask(self, label: str) -> str:
        return self._head_mask.replace('ENTITY', label)

    def _label_aware_tail_mask(self, label: str) -> str:
        return self._tail_mask.replace('ENTITY', label)

    def _create_sentence_with_masked_spans(self, head: _RelationArgument, tail: _RelationArgument) -> Sentence:
        """
        Returns a new `Sentence` object with masked head and tail spans.
        The mask is constructed from the labels of the head and tail span.

        Example:
            For the `head=Larry Page` and `tail=Sergey Brin`
            and the sentence "Larry Page and Sergey Brin founded Google .",
            the masked sentence is "[H-PER] and Sergey Brin founded [T-ORG]"

        :param head: The head `_RelationArgument`
        :param tail: The tail `_RelationArgument`
        :return: The masked sentence
        """
        original_sentence: Sentence = head.entity.sentence
        assert original_sentence is tail.entity.sentence, 'The head and tail need to come from the same sentence.'

        # We can not use the plaintext of the head/tail span in the sentence as the mask
        # since there may be multiple occurrences of the same entity mentioned in the sentence.
        # Therefore, we use the span's position in the sentence.
        masked_sentence_tokens: List[str] = []
        for token in original_sentence:

            if token is head.entity[0]:
                masked_sentence_tokens.append(self._label_aware_head_mask(head.label.value))

            elif token is tail.entity[0]:
                masked_sentence_tokens.append(self._label_aware_tail_mask(tail.label.value))

            elif (all(token is not non_leading_head_token for non_leading_head_token in head.entity.tokens[1:]) and
                  all(token is not non_leading_tail_token for non_leading_tail_token in tail.entity.tokens[1:])):
                masked_sentence_tokens.append(token.text)

        # TODO: Question: When I check the sentence with sentence.to_original_text(), the text is not consistently separated with whitespaces.
        #   Does the TextClassifier use the original text in any way?
        #   If not, I guess that only the tokens matter but not the whitespaces in between.
        return Sentence(masked_sentence_tokens)

    def encode_sentence(self, sentence: Sentence) -> Iterator[Sentence]:
        for head, tail in self._entity_pair_permutations(sentence):
            yield self._create_sentence_with_masked_spans(head, tail)

    def decode_sentence(self, sentence: Sentence):
        pass

    def forward_pass(self,
                     sentences: Union[List[Sentence], Sentence],
                     for_prediction: bool = False) -> Union[Tuple[torch.Tensor, List[List[str]]],
                                                            Tuple[torch.Tensor, List[List[str]], List[DataPoint]]]:
        pass
