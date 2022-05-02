import itertools
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union
)

import torch

import flair
from flair.data import Dictionary, Label, Relation, Sentence, Span, Token
from flair.embeddings import DocumentEmbeddings


class _RelationArgument(NamedTuple):
    """
    A `_RelationArgument` encapsulates either a relation's head or a tail span, including its label.
    This class servers as an internal helper class.
    """
    span: Span
    label: Label


# TODO: This closely shadows the RelationExtractor name. Maybe we need a better name here.
#  - MaskedRelationClassifier ?
class RelationClassifier(flair.nn.DefaultClassifier[Sentence]):
    """
    ---- Task ----
    Relation Classification (RC) is the task of identifying the semantic relation between two entities in a text.
    In contrast to (end-to-end) Relation Extraction (RE), RC requires pre-labelled entities.

    Example:

    For the `founded_by` relation from `ORG` (head) to `PER` (tail) and the sentence
    "Larry Page and Sergey Brin founded Google .", we extract the relations
    - founded_by(head='Google', tail='Larry Page') and
    - founded_by(head='Google', tail='Sergey Brin').

    ---- Architecture ----
    The Relation Classifier Model builds upon a text classifier.
    For a given data point with annotated entities, the model generates an encoded data point
    for each entity pair in the cross product of all entities in the data point.
    In the encoded representation, the entities in the current entity pair are masked with control tokens.
    (For an example, see the docstring of the `_encode_sentence` function.)
    For each encoded data point, the model takes its document embedding and puts the resulting text representation(s)
    through a linear layer to get the class relation label.
    """

    def __init__(self,
                 document_embeddings: DocumentEmbeddings,
                 label_dictionary: Dictionary,
                 label_type: str,
                 entity_label_types: Union[str, Sequence[str], Dict[str, Optional[Set[str]]]],
                 relations: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
                 zero_tag_value: str = 'O',
                 allow_unk_tag: bool = True,
                 train_on_gold_pairs_only: bool = False,
                 mask_remainder: bool = True,
                 **classifierargs) -> None:
        """
        Initializes a RelationClassifier.

        :param document_embeddings: The embeddings used to embed each sentence
        :param label_dictionary: A Dictionary containing all predictable labels from the corpus
        :param label_type: The label type which is going to be predicted, in case a corpus has multiple annotations
        :param entity_label_types: A label type or sequence of label types of the required relation entities.
                                   You can also specify a label filter in a dictionary with the label type as key and
                                   the valid entity labels as values in a set.
                                   E.g. to use only 'PER' and 'ORG' labels from a NER-tagger: `{'ner': {'PER', 'ORG'}}`
                                        to use all labels from 'ner', pass 'ner'.
        :param relations: A dictionary filter of valid relation entity pair combinations
                          to be used as relation candidates. Specify the relation as key and
                          the valid entity pair labels in a set of tuples (<HEAD>, <TAIL>) as value.
                          E.g. for the `born_in` relation, only relations from 'PER' to 'LOC' make sense.
                          Relations from 'PER' to 'PER' are not meaningful.
                          Therefore, it is advised to specify the valid relations as: `{'born_in': {('PER', 'ORG')}}`.

                          This setting may help to reduce the number of relation candidates to be classified.
                          Leaving this parameter as `None` (default) disables the relation-filter,
                          i.e. the model classifies the relation for each entity pair
                          in the cross product of all valid entity pairs.

        :param zero_tag_value: The label to use for out-of-class relations
        :param allow_unk_tag: If `False`, removes `<unk>` from the passed label dictionary, otherwise do nothing
        :param train_on_gold_pairs_only: If `True`, skip out-of-class relations in training.
                                         If `False`, out-of-class relations are used in training as well.
        :param mask_remainder: If `True`, also mask entities which are not part of the current entity pair,
                               otherwise such entities will not be masked.
                               (Setting this parameter to `True` may help to reduce the sentence's sequence length for long entities.)
        :param classifierargs: The remaining parameters passed to the underlying `DefaultClassifier`
        """
        # Set lable type and modify label dictionary
        self._label_type = label_type
        self.zero_tag_value = zero_tag_value
        label_dictionary.add_item(self.zero_tag_value)
        if not allow_unk_tag:
            label_dictionary.remove_item('<unk>')

        # Initialize super default classifier
        super().__init__(label_dictionary=label_dictionary,
                         final_embedding_size=document_embeddings.embedding_length,
                         **classifierargs)

        self.document_embeddings = document_embeddings

        if isinstance(entity_label_types, str):
            self.entity_label_types: Dict[str, Optional[Set[str]]] = {entity_label_types: None}
        elif isinstance(entity_label_types, Sequence):
            self.entity_label_types = {entity_label_type: None for entity_label_type in entity_label_types}
        else:
            self.entity_label_types = entity_label_types

        self.relations = relations

        self.train_on_gold_pairs_only = train_on_gold_pairs_only
        self.mask_remainder = mask_remainder

        # Control mask templates
        self._entity_mask: str = 'ENTITY'
        self._head_mask: str = f'[H-{self._entity_mask}]'
        self._tail_mask: str = f'[T-{self._entity_mask}]'
        self._remainder_mask: str = f'[R-{self._entity_mask}]'

        # Auto-spawn on GPU, if available
        self.to(flair.device)

    def _valid_entities(self, sentence: Sentence) -> Iterator[_RelationArgument]:
        """
        Yields all valid entities, filtered under the specification of `self.entity_label_types`.
        :param sentence: A flair `Sentence` object with entity annotations
        :return: Valid entities as `_RelationArgument`
        """
        for label_type, valid_labels in self.entity_label_types.items():
            for entity_span in sentence.get_spans(type=label_type):

                entity_label: Label = entity_span.get_label(label_type=label_type)

                # Only use entities labelled with the specified labels for each label type
                if valid_labels is not None and entity_label.value not in valid_labels:
                    continue

                yield _RelationArgument(span=entity_span, label=entity_label)

    def _entity_pair_permutations(self, sentence: Sentence) -> Iterator[Tuple[_RelationArgument,
                                                                              _RelationArgument,
                                                                              List[_RelationArgument]]]:
        """
        Yields all valid entity pair permutations (relation candidates) and
        the set difference of all valid entities and the entity pair (the remainder).
        The permutations are constructed by a filtered cross-product
        under the specification of `self.entity_label_types` and `self.relations`.
        :param sentence: A flair `Sentence` object with entity annotations
        :return: Tuples of (<HEAD>, <TAIL>, List[<REMAINDER>]) `_RelationArguments`
        """
        valid_entities: List[_RelationArgument] = list(self._valid_entities(sentence))

        # Yield head and tail entity pairs from the cross product of all entities
        for head, tail in itertools.product(valid_entities, repeat=2):

            # Remove identity relation entity pairs
            if head.span is tail.span:
                continue

            # Remove entity pairs with labels that do not match any of the specified relations in `self.relations`
            if self.relations is not None and all((head.label.value, tail.label.value) not in pairs
                                                  for pairs in self.relations.values()):
                continue

            remainder: List[_RelationArgument] = [
                entity
                for entity in valid_entities
                if entity is not head and entity is not tail
            ]

            yield head, tail, remainder

    def _label_aware_head_mask(self, label: str) -> str:
        return self._head_mask.replace(self._entity_mask, label)

    def _label_aware_tail_mask(self, label: str) -> str:
        return self._tail_mask.replace(self._entity_mask, label)

    def _label_aware_remainder_mask(self, label: str) -> str:
        return self._remainder_mask.replace(self._entity_mask, label)

    def _create_sentence_with_masked_spans(self,
                                           head: _RelationArgument,
                                           tail: _RelationArgument,
                                           remainder: List[_RelationArgument]) -> Sentence:
        """
        Returns a new `Sentence` object with masked head, tail and remainder spans.
        The mask is constructed from the labels of the head/tail/remainder span.

        Example:
            For the `head=Google`, `tail=Larry Page` and `remainder=[Sergey Brin]`
            the sentence "Larry Page and Sergey Brin founded Google .",
            the masked sentence is "[T-PER] and [R-PER] founded [H-ORG]"

        :param head: The head `_RelationArgument`
        :param tail: The tail `_RelationArgument`
        :return: The masked sentence
        """
        # Some sanity checks
        original_sentence: Sentence = head.span.sentence
        assert original_sentence is tail.span.sentence, 'The head and tail need to come from the same sentence.'
        assert all(original_sentence is entity.span.sentence for entity in remainder), \
            'The remainder entities need to come from the same sentence as the head and tail.'

        # Pre-compute non-leading head, tail and remainder tokens for entity masking
        non_leading_head_tokens: List[Token] = head.span.tokens[1:]
        non_leading_tail_tokens: List[Token] = tail.span.tokens[1:]
        non_leading_remainder_tokens: List[Token] = [
            token
            for remainder_entity in remainder
            for token in remainder_entity.span.tokens[1:]
        ]

        # Use a dictionary to find label annotations for a given leading remainder token.
        leading_remainder_token_to_label: Dict[str, str] = {
            remainder_entity.span[0].unlabeled_identifier: remainder_entity.label.value
            for remainder_entity in remainder
        }

        # We can not use the plaintext of the head/tail span in the sentence as the mask
        # since there may be multiple occurrences of the same entity mentioned in the sentence.
        # Therefore, we use the span's position in the sentence.
        masked_sentence_tokens: List[str] = []
        for token in original_sentence:

            remainder_label: Optional[str] = leading_remainder_token_to_label.get(token.unlabeled_identifier)
            if remainder_label is not None:
                masked_sentence_tokens.append(self._label_aware_remainder_mask(remainder_label))

            elif token is head.span[0]:
                masked_sentence_tokens.append(self._label_aware_head_mask(head.label.value))

            elif token is tail.span[0]:
                masked_sentence_tokens.append(self._label_aware_tail_mask(tail.label.value))

            elif all(token is not non_leading_entity_token
                     for non_leading_entity_token in itertools.chain(non_leading_head_tokens,
                                                                     non_leading_tail_tokens,
                                                                     non_leading_remainder_tokens)):
                masked_sentence_tokens.append(token.text)

        # TODO: Question: When I check the sentence with sentence.to_original_text(), the text is not consistently separated with whitespaces.
        #   Does the TextClassifier use the original text in any way?
        #   If not, I guess that only the tokens matter but not the whitespaces in between.
        return Sentence(masked_sentence_tokens)

    def _encode_sentence(self, sentence: Sentence) -> List[Tuple[Sentence, Relation]]:
        """
        Returns masked entity pair sentences and their relation for all valid entity pair permutations.
        The created masked sentences are newly created sentences with no reference to the passed sentence.
        The created relations have head and tail spans from the original passed sentence.

        Example:
            For the `founded_by` relation from `ORG` to `PER` and
            the sentence "Larry Page and Sergey Brin founded Google .",
            the masked sentences and relations are
            - "[T-PER] and Sergey Brin founded [H-ORG]" -> Relation(head='Google', tail='Larry Page')  and
            - "Larry Page and [T-PER] founded [H-ORG]"  -> Relation(head='Google', tail='Sergey Brin').

        :param sentence: A flair `Sentence` object with entity annotations
        :return: Encoded sentences and the corresponding relation in the original sentence
        """
        return [
            (self._create_sentence_with_masked_spans(head, tail, remainder if self.mask_remainder else []),
             Relation(first=head.span, second=tail.span))
            for head, tail, remainder in self._entity_pair_permutations(sentence)
        ]

    def forward_pass(self,
                     sentences: Union[List[Sentence], Sentence],
                     for_prediction: bool = False) -> Union[Tuple[torch.Tensor, List[List[str]]],
                                                            Tuple[torch.Tensor, List[List[str]], List[Relation]]]:
        if not isinstance(sentences, list):
            sentences = [sentences]

        masked_sentence_embeddings: List[torch.Tensor] = []
        masked_sentence_batch_relations: List[Relation] = []
        gold_labels: List[List[str]] = []

        for sentence in sentences:
            # Encode the original sentence into a list of masked sentences and
            # the corresponding relations in the original sentence for all valid entity pair permutations.
            # Each masked sentence is one relation candidate.
            encoded: List[Tuple[Sentence, Relation]] = self._encode_sentence(sentence)

            # Process the encoded sentences, if there's at least one entity pair in the original sentence.
            if encoded:
                masked_sentences, relations = zip(*encoded)

                masked_sentence_batch_relations.extend(relations)

                # Embed masked sentences
                self.document_embeddings.embed(list(masked_sentences))
                encoded_sentence_embedding: torch.Tensor = torch.stack(
                    [masked_sentence.get_embedding(self.document_embeddings.get_names())
                     for masked_sentence in masked_sentences],
                    dim=0,
                )
                masked_sentence_embeddings.append(encoded_sentence_embedding)

                # Add gold labels for each masked sentence, if available.
                # Use a dictionary to find relation annotations for a given entity pair relation.
                relation_to_gold_label: Dict[str, str] = {
                    relation.unlabeled_identifier: relation.get_label(self.label_type,
                                                                      zero_tag_value=self.zero_tag_value).value
                    for relation in sentence.get_relations(self.label_type)
                }
                for relation in relations:
                    gold_label: str = relation_to_gold_label.get(relation.unlabeled_identifier, self.zero_tag_value)
                    if gold_label == self.zero_tag_value and self.train_on_gold_pairs_only:
                        continue  # Skip zero tag value labels, if training on gold pairs only
                    gold_labels.append([gold_label])

        masked_sentence_batch_embeddings: torch.Tensor = (
            torch.cat(masked_sentence_embeddings, dim=0) if masked_sentence_embeddings
            else torch.empty(0, self.document_embeddings.embedding_length)
        )
        if for_prediction:
            return masked_sentence_batch_embeddings, gold_labels, masked_sentence_batch_relations
        return masked_sentence_batch_embeddings, gold_labels

    def _get_state_dict(self) -> Dict[str, Any]:
        model_state: Dict[str, Any] = {
            **super()._get_state_dict(),
            'document_embeddings': self.document_embeddings,
            'label_dictionary': self.label_dictionary,
            'label_type': self.label_type,
            'entity_label_types': self.entity_label_types,
            'relations': self.relations,
            'zero_tag_value': self.zero_tag_value,
            'train_on_gold_pairs_only': self.train_on_gold_pairs_only,
            'mask_remainder': self.mask_remainder
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state: Dict[str, Any], **kwargs):
        return super()._init_model_with_state_dict(
            state,
            document_embeddings=state['document_embeddings'],
            label_dictionary=state['label_dictionary'],
            label_type=state['label_type'],
            entity_label_types=state['entity_label_types'],
            relations=state['relations'],
            zero_tag_value=state['zero_tag_value'],
            train_on_gold_pairs_only=state['train_on_gold_pairs_only'],
            mask_remainder=state['mask_remainder'],
            **kwargs
        )

    @property
    def label_type(self) -> str:
        return self._label_type
