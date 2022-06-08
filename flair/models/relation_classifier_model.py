import itertools
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import torch
from torch.utils.data.dataset import Dataset

import flair
from flair.data import Dictionary, Label, Relation, Sentence, Span, Token, Corpus
from flair.datasets import FlairDatapointDataset
from flair.embeddings import DocumentEmbeddings
from flair.tokenization import SpaceTokenizer


class _EncodedSentence(Sentence):
    """A wrapper of `Sentence` to type-check whether a sentence has been encoded for the relation classifier."""

    pass


class _Entity(NamedTuple):
    """
    A `_Entity` encapsulates either a relation's head or a tail span, including its label.
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

    def __init__(
        self,
        document_embeddings: DocumentEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        entity_label_types: Union[str, Sequence[str], Dict[str, Optional[Set[str]]]],
        entity_pair_labels: Optional[Set[Tuple[str, str]]] = None,
        entity_threshold: Optional[float] = None,
        zero_tag_value: str = "O",
        allow_unk_tag: bool = True,
        cross_augmentation: bool = True,
        mask_remainder: bool = True,
        **classifierargs,
    ) -> None:
        """
        Initializes a `RelationClassifier`.

        :param document_embeddings: The document embeddings used to embed each sentence
        :param label_dictionary: A Dictionary containing all predictable labels from the corpus
        :param label_type: The label type which is going to be predicted, in case a corpus has multiple annotations
        :param entity_label_types: A label type or sequence of label types of the required relation entities.
                                   You can also specify a label filter in a dictionary with the label type as key and
                                   the valid entity labels as values in a set.
                                   E.g. to use only 'PER' and 'ORG' labels from a NER-tagger: `{'ner': {'PER', 'ORG'}}`.
                                   To use all labels from 'ner', pass 'ner'.
        :param entity_pair_labels: A set of valid relation entity pair combinations, used as relation candidates.
                                   Specify valid entity pairs in a set of tuples of labels (<HEAD>, <TAIL>).
                                   E.g. for the `born_in` relation, only relations from 'PER' to 'LOC' make sense.
                                   Here, relations from 'PER' to 'PER' are not meaningful, so
                                   it is advised to specify the `entity_pair_labels` as `{('PER', 'ORG')}`.
                                   This setting may help to reduce the number of relation candidates.
                                   Leaving this parameter as `None` (default) disables the relation-candidate-filter,
                                   i.e. the model classifies the relation for each entity pair
                                   in the cross product of *all* entity pairs (inefficient).
        :param entity_threshold: Only pre-labelled entities above this threshold are taken into account by the model.
        :param zero_tag_value: The label to use for out-of-class relations
        :param allow_unk_tag: If `False`, removes `<unk>` from the passed label dictionary, otherwise do nothing.
        :param cross_augmentation: TODO: Add docstring
        :param mask_remainder: If `True`, also mask entities which are not part of the current entity pair,
                               otherwise such entities will not be masked.
                               (Setting this parameter to `True` may help to
                               reduce the sentence's sequence length for long entities.)
        :param classifierargs: The remaining parameters passed to the underlying `DefaultClassifier`
        """
        # Set lable type and prepare label dictionary
        self._label_type = label_type
        self._zero_tag_value = zero_tag_value
        self._allow_unk_tag = allow_unk_tag

        modified_label_dictionary: Dictionary = Dictionary(add_unk=self._allow_unk_tag)
        modified_label_dictionary.add_item(self._zero_tag_value)
        for label in label_dictionary.get_items():
            modified_label_dictionary.add_item(label)

        # Initialize super default classifier
        super().__init__(
            label_dictionary=modified_label_dictionary,
            final_embedding_size=document_embeddings.embedding_length,
            **classifierargs,
        )

        self.document_embeddings = document_embeddings

        if isinstance(entity_label_types, str):
            self.entity_label_types: Dict[str, Optional[Set[str]]] = {entity_label_types: None}
        elif isinstance(entity_label_types, Sequence):
            self.entity_label_types = {entity_label_type: None for entity_label_type in entity_label_types}
        else:
            self.entity_label_types = entity_label_types

        self.entity_pair_labels = entity_pair_labels

        self.cross_augmentation = cross_augmentation
        self.mask_remainder = mask_remainder
        self.entity_threshold = entity_threshold

        # Control mask templates
        self._entity_mask: str = "ENTITY"
        self._head_mask: str = f"[H-{self._entity_mask}]"
        self._tail_mask: str = f"[T-{self._entity_mask}]"
        self._remainder_mask: str = f"[R-{self._entity_mask}]"

        # Auto-spawn on GPU, if available
        self.to(flair.device)

    def _valid_entities(self, sentence: Sentence) -> Iterator[_Entity]:
        """
        Yields all valid entities, filtered under the specification of `self.entity_label_types`.
        :param sentence: A flair `Sentence` object with entity annotations
        :return: Valid entities as `_Entity`
        """
        for label_type, valid_labels in self.entity_label_types.items():
            for entity_span in sentence.get_spans(type=label_type):

                entity_label: Label = entity_span.get_label(label_type=label_type)

                # Only use entities labelled with the specified labels for each label type
                if valid_labels is not None and entity_label.value not in valid_labels:
                    continue

                # Only use entities above the specified threshold
                if self.entity_threshold is not None and entity_label.score <= self.entity_threshold:
                    continue

                yield _Entity(span=entity_span, label=entity_label)

    def _entity_pair_permutations(
        self,
        sentence: Sentence,
    ) -> Iterator[Tuple[_Entity, _Entity, List[_Entity], Optional[str]]]:
        """
        Yields all valid entity pair permutations (relation candidates) and
        the set difference of all valid entities and the entity pair (the remainder).
        If the passed sentence contains relation annotations,
        the relation gold label will be yielded along with the participating entities.
        The permutations are constructed by a filtered cross-product
        under the specification of `self.entity_label_types` and `self.entity_pair_labels`.
        :param sentence: A flair `Sentence` object with entity annotations
        :return: Tuples of (HEAD, TAIL, List[REMAINDER], gold_label).
                 The head, tail and remainder `_Entity`s have span references to the passed sentence.
        """
        valid_entities: List[_Entity] = list(self._valid_entities(sentence))

        # Use a dictionary to find gold relation annotations for a given entity pair
        relation_to_gold_label: Dict[str, str] = {
            relation.unlabeled_identifier: relation.get_label(self.label_type, zero_tag_value=self.zero_tag_value).value
            for relation in sentence.get_relations(self.label_type)
        }

        # Yield head and tail entity pairs from the cross product of all entities
        for head, tail in itertools.product(valid_entities, repeat=2):

            # Remove identity relation entity pairs
            if head.span is tail.span:
                continue

            # Remove entity pairs with labels that do not match any
            # of the specified relations in `self.entity_pair_labels`
            if (
                self.entity_pair_labels is not None
                and (head.label.value, tail.label.value) not in self.entity_pair_labels
            ):
                continue

            # Obtain remainder entities
            remainder: List[_Entity] = [
                entity for entity in valid_entities if entity is not head and entity is not tail
            ]

            # Obtain gold label, if existing
            original_relation: Relation = Relation(first=head.span, second=tail.span)
            gold_label: Optional[str] = relation_to_gold_label.get(original_relation.unlabeled_identifier)

            yield head, tail, remainder, gold_label

    def _label_aware_head_mask(self, label: str) -> str:
        return self._head_mask.replace(self._entity_mask, label)

    def _label_aware_tail_mask(self, label: str) -> str:
        return self._tail_mask.replace(self._entity_mask, label)

    def _label_aware_remainder_mask(self, label: str) -> str:
        return self._remainder_mask.replace(self._entity_mask, label)

    def _create_masked_sentence(
        self,
        head: _Entity,
        tail: _Entity,
        remainder: List[_Entity],
        gold_label: Optional[str] = None,
    ) -> _EncodedSentence:
        """
        Returns a new `Sentence` object with masked head, tail and remainder spans.
        The label-aware mask is constructed from the head/tail/remainder span labels.
        If provided, the masked sentence also has the corresponding gold label annotation in `self.label_type`.

        Example:
            For the `head=Google`, `tail=Larry Page` and `remainder=[Sergey Brin]`
            the sentence "Larry Page and Sergey Brin founded Google .",
            the masked sentence is "[T-PER] and [R-PER] founded [H-ORG]".

        :param head: The head `_Entity`
        :param tail: The tail `_Entity`
        :param remainder: The list of remainder `_Entity`s
        :param gold_label: An optional gold label of the induced relation by the head and tail entity
        :return: The masked sentence (with gold annotations)
        """
        # Some sanity checks
        original_sentence: Sentence = head.span.sentence
        assert original_sentence is tail.span.sentence, "The head and tail need to come from the same sentence."
        assert all(
            original_sentence is entity.span.sentence for entity in remainder
        ), "The remainder entities need to come from the same sentence as the head and tail."

        # Pre-compute non-leading head, tail and remainder tokens for entity masking
        non_leading_head_tokens: List[Token] = head.span.tokens[1:]
        non_leading_tail_tokens: List[Token] = tail.span.tokens[1:]
        non_leading_remainder_tokens: List[Token] = [
            token for remainder_entity in remainder for token in remainder_entity.span.tokens[1:]
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

            elif all(
                token is not non_leading_entity_token
                for non_leading_entity_token in itertools.chain(
                    non_leading_head_tokens, non_leading_tail_tokens, non_leading_remainder_tokens
                )
            ):
                masked_sentence_tokens.append(token.text)

        # Create masked sentence
        masked_sentence: _EncodedSentence = _EncodedSentence(
            " ".join(masked_sentence_tokens), use_tokenizer=SpaceTokenizer()
        )

        if gold_label is not None:
            # Add gold relation annotation as sentence label
            # Using the sentence label instead of annotating a separate `Relation` object is easier to manage since,
            # during prediction, the forward pass does not need any knowledge about the entities in the sentence.
            masked_sentence.add_label(typename=self.label_type, value=gold_label, score=1.0)

        return masked_sentence

    def _encode_sentence_for_inference(self, sentence: Sentence) -> Iterator[Tuple[_EncodedSentence, Relation]]:
        """
        Yields masked entity pair sentences annotated with their gold relation for all valid entity pair permutations.
        The created masked sentences are newly created sentences with no reference to the passed sentence.
        Important properties:
            - Every sentence has exactly one masked head and tail entity token. Therefore, every encoded sentence has
              **exactly** one induced relation annotation, the gold annotation or `self.zero_tag_value`.
            - The created relations have head and tail spans from the original passed sentence.

        Example:
            For the `founded_by` relation from `ORG` to `PER` and
            the sentence "Larry Page and Sergey Brin founded Google .",
            the masked sentences and relations are
            - "[T-PER] and Sergey Brin founded [H-ORG]" -> Relation(head='Google', tail='Larry Page')  and
            - "Larry Page and [T-PER] founded [H-ORG]"  -> Relation(head='Google', tail='Sergey Brin').

        :param sentence: A flair `Sentence` object with entity annotations
        :return: Encoded sentences annotated with their gold relation and
                 the corresponding relation in the original sentence
        """
        for head, tail, remainder, gold_label in self._entity_pair_permutations(sentence):
            masked_sentence: _EncodedSentence = self._create_masked_sentence(
                head=head,
                tail=tail,
                remainder=remainder if self.mask_remainder else [],
                gold_label=gold_label if gold_label is not None else self.zero_tag_value,
            )
            original_relation: Relation = Relation(first=head.span, second=tail.span)
            yield masked_sentence, original_relation

    def _encode_sentence_for_training(self, sentence: Sentence) -> Iterator[_EncodedSentence]:
        """
        Same as `self._encode_sentence_for_inference`,
        with the option of disabling cross augmentation via `self.cross_augmentation`
        (and that the relation with reference to the original sentence is not returned).
        """
        for head, tail, remainder, gold_label in self._entity_pair_permutations(sentence):

            if gold_label is None:
                if self.cross_augmentation:
                    gold_label = self.zero_tag_value
                else:
                    continue  # Skip generated data points that do not express an originally annotated relation

            masked_sentence: _EncodedSentence = self._create_masked_sentence(
                head=head,
                tail=tail,
                remainder=remainder if self.mask_remainder else [],
                gold_label=gold_label,
            )

            yield masked_sentence

    def transform_sentence(self, sentences: Union[Sentence, List[Sentence]]) -> List[_EncodedSentence]:
        """
        :param sentences:
        :return:
        """

        if not isinstance(sentences, list):
            sentences = [sentences]

        return [
            encoded_sentence
            for sentence in sentences
            for encoded_sentence in self._encode_sentence_for_training(sentence)
        ]

    def transform_dataset(self, dataset: Dataset[Sentence]) -> FlairDatapointDataset[_EncodedSentence]:
        """

        :param dataset:
        :return:
        """
        return FlairDatapointDataset(self.transform_sentence(list(dataset)))

    def transform_corpus(self, corpus: Corpus, transform_test: bool = False) -> Corpus:
        """

        :param corpus:
        :param transform_test:
        :return:
        """
        return Corpus(
            train=self.transform_dataset(corpus.train),
            dev=self.transform_dataset(corpus.dev),
            test=self.transform_dataset(corpus.test) if transform_test else corpus.test,
            name=corpus.name,
        )

    def forward_pass(
        self,
        sentences: Union[List[_EncodedSentence], _EncodedSentence],
        for_prediction: bool = False,
    ) -> Union[Tuple[torch.Tensor, List[List[str]]], Tuple[torch.Tensor, List[List[str]], List[_EncodedSentence]]]:
        """
        This method does a forward pass through the model given a list of **encoded** sentences as input.
        To encode sentences, use the `transform` function of the `RelationClassifier`.
        For more information on the forward pass, see the `forward_pass` method of the `DefaultClassifier`.
        """
        if not isinstance(sentences, list):
            sentences = [sentences]

        # Embed encoded sentences: The input sentences already have been encoded/masked beforehand.
        self.document_embeddings.embed(sentences)
        embedding_names: List[str] = self.document_embeddings.get_names()

        sentence_embeddings: List[torch.Tensor] = []
        gold_labels: List[List[str]] = []

        for sentence in sentences:
            gold_label: List[str] = [label.value for label in sentence.get_labels(self.label_type)]
            gold_labels.append(gold_label)
            sentence_embeddings.append(sentence.get_embedding(embedding_names))

        # Shape: [len(sentences), embedding_size]
        sentence_embeddings_tensor: torch.Tensor = (
            torch.stack(sentence_embeddings, dim=0)
            if sentence_embeddings
            else torch.empty(0, self.document_embeddings.embedding_length, device=flair.device)
        )

        if for_prediction:
            # Since the relation is encoded inside the sentence (as a simple label),
            # the sentence itself is the data point open for prediction.
            return sentence_embeddings_tensor, gold_labels, sentences

        return sentence_embeddings_tensor, gold_labels

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: str = "none",
        entity_threshold: Optional[float] = None,  # TODO: we could add the entity threshold here as well
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """
        # TODO: Add docstring
        """
        prediction_label_type: str = self.label_type if label_name is None else label_name

        if not isinstance(sentences, list):
            sentences = [sentences]

        loss: Optional[Tuple[torch.Tensor, int]]
        if all(isinstance(sentence, _EncodedSentence) for sentence in sentences):
            loss = super().predict(
                sentences,
                mini_batch_size=mini_batch_size,
                return_probabilities_for_all_classes=return_probabilities_for_all_classes,
                verbose=verbose,
                label_name=prediction_label_type,
                return_loss=return_loss,
                embedding_storage_mode=embedding_storage_mode,
            )

        elif all(not isinstance(sentence, _EncodedSentence) for sentence in sentences):

            sentences_with_relation_reference: List[Tuple[_EncodedSentence, Relation]] = list(
                itertools.chain.from_iterable(self._encode_sentence_for_inference(sentence) for sentence in sentences)
            )

            encoded_sentences: List[_EncodedSentence] = [x[0] for x in sentences_with_relation_reference]
            loss = super().predict(
                encoded_sentences,
                mini_batch_size=mini_batch_size,
                return_probabilities_for_all_classes=return_probabilities_for_all_classes,
                verbose=verbose,
                label_name=prediction_label_type,
                return_loss=return_loss,
                embedding_storage_mode=embedding_storage_mode,
            )

            for encoded_sentence, original_relation in sentences_with_relation_reference:
                for label in encoded_sentence.get_labels(prediction_label_type):
                    original_relation.add_label(prediction_label_type, value=label.value, score=label.value)

        else:
            raise ValueError("All passed sentences must be either uniformly encoded or not.")

        return loss if return_loss else None

    def _get_state_dict(self) -> Dict[str, Any]:
        model_state: Dict[str, Any] = {
            **super()._get_state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "entity_label_types": self.entity_label_types,
            "entity_pair_labels": self.entity_pair_labels,
            "entity_threshold": self.entity_threshold,
            "zero_tag_value": self.zero_tag_value,
            "allow_unk_tag": self.allow_unk_tag,
            "cross_augmentation": self.cross_augmentation,
            "mask_remainder": self.mask_remainder,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state: Dict[str, Any], **kwargs):
        return super()._init_model_with_state_dict(
            state,
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            entity_label_types=state["entity_label_types"],
            entity_pair_labels=state["entity_pair_labels"],
            entity_threshold=state["entity_threshold"],
            zero_tag_value=state["zero_tag_value"],
            allow_unk_tag=state["allow_unk_tag"],
            cross_augmentation=state["cross_augmentation"],
            mask_remainder=state["mask_remainder"],
            **kwargs,
        )

    @property
    def label_type(self) -> str:
        return self._label_type

    @property
    def zero_tag_value(self) -> str:
        return self._zero_tag_value

    @property
    def allow_unk_tag(self) -> bool:
        return self._allow_unk_tag
