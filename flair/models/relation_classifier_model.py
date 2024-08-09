import itertools
import logging
import typing
from abc import ABC, abstractmethod
from pathlib import Path
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
    Union,
    cast,
)

import torch
from torch.utils.data.dataset import Dataset

import flair
from flair.data import (
    Corpus,
    Dictionary,
    Label,
    Relation,
    Sentence,
    Span,
    Token,
    _iter_dataset,
)
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import DocumentEmbeddings, TransformerDocumentEmbeddings
from flair.tokenization import SpaceTokenizer
from flair.training_utils import EmbeddingStorageMode

logger: logging.Logger = logging.getLogger("flair")


class EncodedSentence(Sentence):
    """A Sentence that expresses that a sentence is encoded and compatible with the relation classifier.

    For inference, i.e. `predict` and `evaluate`, the relation classifier internally encodes the sentences.
    Therefore, these functions work with the regular flair sentence objects.
    """


class EncodingStrategy(ABC):
    """The encoding of the head and tail entities in a sentence with a relation annotation."""

    special_tokens: Set[str] = set()

    def __init__(self, add_special_tokens: bool = False) -> None:
        self.add_special_tokens = add_special_tokens

    @abstractmethod
    def encode_head(self, head_span: Span, label: Label) -> str:
        """Returns the encoded string representation of the head span.

        Multi-token head encodings tokens are separated by a space.
        """
        ...

    @abstractmethod
    def encode_tail(self, tail_span: Span, label: Label) -> str:
        """Returns the encoded string representation of the tail span.

        Multi-token tail encodings tokens are separated by a space.
        """
        ...


class EntityMask(EncodingStrategy):
    """An `class`:EncodingStrategy: that masks the head and tail relation entities.

    Example:
    -------
        For the `founded_by` relation from `ORG` to `PER` and
        the sentence "Larry Page and Sergey Brin founded Google .",
        the encoded sentences and relations are
        - "[TAIL] and Sergey Brin founded [HEAD]" -> Relation(head='Google', tail='Larry Page')  and
        - "Larry Page and [TAIL] founded [HEAD]"  -> Relation(head='Google', tail='Sergey Brin').
    """

    special_tokens: Set[str] = {"[HEAD]", "[TAIL]"}

    def encode_head(self, head_span: Span, label: Label) -> str:
        return "[HEAD]"

    def encode_tail(self, tail_span: Span, label: Label) -> str:
        return "[TAIL]"


class TypedEntityMask(EncodingStrategy):
    """An `class`:EncodingStrategy: that masks the head and tail relation entities with their label.

    Example:
    -------
        For the `founded_by` relation from `ORG` to `PER` and
        the sentence "Larry Page and Sergey Brin founded Google .",
        the encoded sentences and relations are
        - "[TAIL-PER] and Sergey Brin founded [HEAD-ORG]" -> Relation(head='Google', tail='Larry Page')  and
        - "Larry Page and [TAIL-PER] founded [HEAD-ORG]"  -> Relation(head='Google', tail='Sergey Brin').
    """

    def encode_head(self, head: Span, label: Label) -> str:
        return f"[HEAD-{label.value}]"

    def encode_tail(self, tail: Span, label: Label) -> str:
        return f"[TAIL-{label.value}]"


class EntityMarker(EncodingStrategy):
    """An `class`:EncodingStrategy: that marks the head and tail relation entities.

    Example:
    -------
        For the `founded_by` relation from `ORG` to `PER` and
        the sentence "Larry Page and Sergey Brin founded Google .",
        the encoded sentences and relations are
        - "[HEAD] Larry Page [/HEAD] and Sergey Brin founded [TAIL] Google [/TAIL]"
            -> Relation(head='Google', tail='Larry Page')  and
        - "Larry Page and [HEAD] Sergey Brin [/HEAD] founded [TAIL] Google [/TAIL]"
            -> Relation(head='Google', tail='Sergey Brin').
    """

    special_tokens: Set[str] = {"[HEAD]", "[/HEAD]", "[TAIL]", "[/TAIL]"}

    def encode_head(self, head: Span, label: Label) -> str:
        space_tokenized_text: str = " ".join(token.text for token in head)
        return f"[HEAD] {space_tokenized_text} [/HEAD]"

    def encode_tail(self, tail: Span, label: Label) -> str:
        space_tokenized_text: str = " ".join(token.text for token in tail)
        return f"[TAIL] {space_tokenized_text} [/TAIL]"


class TypedEntityMarker(EncodingStrategy):
    """An `class`:EncodingStrategy: that marks the head and tail relation entities with their label.

    Example:
    -------
        For the `founded_by` relation from `ORG` to `PER` and
        the sentence "Larry Page and Sergey Brin founded Google .",
        the encoded sentences and relations are
        - "[HEAD-PER] Larry Page [/HEAD-PER] and Sergey Brin founded [TAIL-ORG] Google [/TAIL-ORG]"
            -> Relation(head='Google', tail='Larry Page')  and
        - "Larry Page and [HEAD-PER] Sergey Brin [/HEAD-PER] founded [TAIL-ORG] Google [/TAIL-ORG]"
            -> Relation(head='Google', tail='Sergey Brin').
    """

    def encode_head(self, head: Span, label: Label) -> str:
        space_tokenized_text: str = " ".join(token.text for token in head)
        return f"[HEAD-{label.value}] {space_tokenized_text} [/HEAD-{label.value}]"

    def encode_tail(self, tail: Span, label: Label) -> str:
        space_tokenized_text: str = " ".join(token.text for token in tail)
        return f"[TAIL-{label.value}] {space_tokenized_text} [/TAIL-{label.value}]"


class EntityMarkerPunct(EncodingStrategy):
    """An alternate version of `class`:EntityMarker: with punctuations as control tokens.

    Example:
    -------
        For the `founded_by` relation from `ORG` to `PER` and
        the sentence "Larry Page and Sergey Brin founded Google .",
        the encoded sentences and relations are
        - "@ Larry Page @ and Sergey Brin founded # Google #" -> Relation(head='Google', tail='Larry Page')  and
        - "Larry Page and @ Sergey Brin @ founded # Google #" -> Relation(head='Google', tail='Sergey Brin').
    """

    def encode_head(self, head: Span, label: Label) -> str:
        space_tokenized_text: str = " ".join(token.text for token in head)
        return f"@ {space_tokenized_text} @"

    def encode_tail(self, tail: Span, label: Label) -> str:
        space_tokenized_text: str = " ".join(token.text for token in tail)
        return f"# {space_tokenized_text} #"


class TypedEntityMarkerPunct(EncodingStrategy):
    """An alternate version of `class`:TypedEntityMarker: with punctuations as control tokens.

    Example:
    -------
        For the `founded_by` relation from `ORG` to `PER` and
        the sentence "Larry Page and Sergey Brin founded Google .",
        the encoded sentences and relations are
        - "@ * PER * Larry Page @ and Sergey Brin founded # * ORG * Google #"
            -> Relation(head='Google', tail='Larry Page')  and
        - "Larry Page and @ * PER * Sergey Brin @ founded # * ORG * Google #"
            -> Relation(head='Google', tail='Sergey Brin').
    """

    def encode_head(self, head: Span, label: Label) -> str:
        space_tokenized_text: str = " ".join(token.text for token in head)
        return f"@ * {label.value} * {space_tokenized_text} @"

    def encode_tail(self, tail: Span, label: Label) -> str:
        space_tokenized_text: str = " ".join(token.text for token in tail)
        return f"# ^ {label.value} ^ {space_tokenized_text} #"


class _Entity(NamedTuple):
    """A `_Entity` encapsulates either a relation's head or a tail span, including its label.

    This class servers as an internal helper class.
    """

    span: Span
    label: Label


# TODO: This closely shadows the RelationExtractor name. Maybe we need a better name here.
#  - MaskedRelationClassifier ?
#   This depends if this relation classification architecture should replace or offer as an alternative.
class RelationClassifier(flair.nn.DefaultClassifier[EncodedSentence, EncodedSentence]):
    """Relation Classifier to predict the relation between two entities.

    Task
    ----
    Relation Classification (RC) is the task of identifying the semantic relation between two entities in a text.
    In contrast to (end-to-end) Relation Extraction (RE), RC requires pre-labelled entities.

    Example:
    --------
    For the `founded_by` relation from `ORG` (head) to `PER` (tail) and the sentence
    "Larry Page and Sergey Brin founded Google .", we extract the relations
    - founded_by(head='Google', tail='Larry Page') and
    - founded_by(head='Google', tail='Sergey Brin').

    Architecture
    ------------
    The Relation Classifier Model builds upon a text classifier.
    The model generates an encoded sentence for each entity pair
    in the cross product of all entities in the original sentence.
    In the encoded representation, the entities in the current entity pair are masked/marked with control tokens.
    (For an example, see the docstrings of different encoding strategies, e.g. :class:`TypedEntityMarker`.)
    Then, for each encoded sentence, the model takes its document embedding and puts the resulting
    text representation(s) through a linear layer to get the class relation label.

    The implemented encoding strategies are taken from this paper by Zhou et al.: https://arxiv.org/abs/2102.01373

    .. warning::
        Currently, the model has no multi-label support.

    """

    def __init__(
        self,
        embeddings: DocumentEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        entity_label_types: Union[str, Sequence[str], Dict[str, Optional[Set[str]]]],
        entity_pair_labels: Optional[Set[Tuple[str, str]]] = None,
        entity_threshold: Optional[float] = None,
        cross_augmentation: bool = True,
        encoding_strategy: EncodingStrategy = TypedEntityMarker(),
        zero_tag_value: str = "O",
        allow_unk_tag: bool = True,
        **classifierargs,
    ) -> None:
        """Initializes a `RelationClassifier`.

        Args:
            embeddings: The document embeddings used to embed each sentence
            label_dictionary: A Dictionary containing all predictable labels from the corpus
            label_type: The label type which is going to be predicted, in case a corpus has multiple annotations
            entity_label_types: A label type or sequence of label types of the required relation entities. You can also specify a label filter in a dictionary with the label type as key and the valid entity labels as values in a set. E.g. to use only 'PER' and 'ORG' labels from a NER-tagger: `{'ner': {'PER', 'ORG'}}`. To use all labels from 'ner', pass 'ner'.
            entity_pair_labels: A set of valid relation entity pair combinations, used as relation candidates. Specify valid entity pairs in a set of tuples of labels (<HEAD>, <TAIL>). E.g. for the `born_in` relation, only relations from 'PER' to 'LOC' make sense. Here, relations from 'PER' to 'PER' are not meaningful, so it is advised to specify the `entity_pair_labels` as `{('PER', 'ORG')}`. This setting may help to reduce the number of relation candidates. Leaving this parameter as `None` (default) disables the relation-candidate-filter, i.e. the model classifies the relation for each entity pair in the cross product of *all* entity pairs (inefficient).
            entity_threshold: Only pre-labelled entities above this threshold are taken into account by the model.
            cross_augmentation: If `True`, use cross augmentation to transform `Sentence`s into `EncodedSentenece`s. When cross augmentation is enabled, the transformation functions, e.g. `transform_corpus`, generate an encoded sentence for each entity pair in the cross product of all entities in the original sentence. When disabling cross augmentation, the transform functions only generate  encoded sentences for each gold relation annotation in the original sentence.
            encoding_strategy: An instance of a class conforming the :class:`EncodingStrategy` protocol
            zero_tag_value: The label to use for out-of-class relations
            allow_unk_tag: If `False`, removes `<unk>` from the passed label dictionary, otherwise do nothing.
            classifierargs: The remaining parameters passed to the underlying :class:`flair.models.DefaultClassifier`
        """
        # Set label type and prepare label dictionary
        self._label_type = label_type
        self._zero_tag_value = zero_tag_value
        self._allow_unk_tag = allow_unk_tag

        modified_label_dictionary: Dictionary = Dictionary(add_unk=self._allow_unk_tag)
        modified_label_dictionary.add_item(self._zero_tag_value)
        for label in label_dictionary.get_items():
            if label != "<unk>":
                modified_label_dictionary.add_item(label)

        # Initialize super default classifier
        super().__init__(
            embeddings=embeddings,
            label_dictionary=modified_label_dictionary,
            final_embedding_size=embeddings.embedding_length,
            **classifierargs,
        )

        if isinstance(entity_label_types, str):
            self.entity_label_types: Dict[str, Optional[Set[str]]] = {entity_label_types: None}
        elif isinstance(entity_label_types, Sequence):
            self.entity_label_types = {entity_label_type: None for entity_label_type in entity_label_types}
        else:
            self.entity_label_types = entity_label_types

        self.entity_pair_labels = entity_pair_labels

        self.entity_threshold = entity_threshold
        self.cross_augmentation = cross_augmentation
        self.encoding_strategy = encoding_strategy

        # Add the special tokens from the encoding strategy
        if (
            self.encoding_strategy.add_special_tokens
            and self.encoding_strategy.special_tokens
            and isinstance(self.embeddings, TransformerDocumentEmbeddings)
        ):
            special_tokens: List[str] = list(self.encoding_strategy.special_tokens)
            tokenizer = self.embeddings.tokenizer
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self.embeddings.model.resize_token_embeddings(len(tokenizer))

            logger.info(
                f"{self.__class__.__name__}: "
                f"Added {', '.join(special_tokens)} as additional special tokens to {self.embeddings.name}"
            )

        # Auto-spawn on GPU, if available
        self.to(flair.device)

    def _valid_entities(self, sentence: Sentence) -> Iterator[_Entity]:
        """Yields all valid entities, filtered under the specification of :attr:`~entity_label_types`.

        Args:
            sentence: A Sentence object with entity annotations

        Yields:
            Valid entities as `_Entity`
        """
        for label_type, valid_labels in self.entity_label_types.items():
            for entity_span in sentence.get_spans(label_type=label_type):
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
    ) -> Iterator[Tuple[_Entity, _Entity, Optional[str]]]:
        """Yields all valid entity pair permutations (relation candidates).

        If the passed sentence contains relation annotations,
        the relation gold label will be yielded along with the participating entities.
        The permutations are constructed by a filtered cross-product
        under the specification of :py:meth:~`flair.models.RelationClassifier.entity_label_types`
        and :py:meth:~`flair.models.RelationClassifier.entity_pair_labels`.

        Args:
            sentence: A Sentence with entity annotations

        Yields:
            Tuples of (HEAD, TAIL, gold_label): The head and tail `_Entity`s` have span references to the passed sentence.
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

            # Obtain gold label, if existing
            original_relation: Relation = Relation(first=head.span, second=tail.span)
            gold_label: Optional[str] = relation_to_gold_label.get(original_relation.unlabeled_identifier)

            yield head, tail, gold_label

    def _encode_sentence(
        self,
        head: _Entity,
        tail: _Entity,
        gold_label: Optional[str] = None,
    ) -> EncodedSentence:
        """Returns a new Sentence object with masked/marked head and tail spans according to the encoding strategy.

        If provided, the encoded sentence also has the corresponding gold label annotation from :attr:`~label_type`.

        Args:
            head: The head Entity
            tail: The tail Entity
            gold_label: An optional gold label of the induced relation by the head and tail entity

        Returns: The EncodedSentence with Gold Annotations
        """
        # Some sanity checks
        original_sentence: Sentence = head.span.sentence
        assert original_sentence is tail.span.sentence, "The head and tail need to come from the same sentence."

        # Pre-compute non-leading head and tail tokens for entity masking
        non_leading_head_tokens: List[Token] = head.span.tokens[1:]
        non_leading_tail_tokens: List[Token] = tail.span.tokens[1:]

        # We can not use the plaintext of the head/tail span in the sentence as the mask/marker
        # since there may be multiple occurrences of the same entity mentioned in the sentence.
        # Therefore, we use the span's position in the sentence.
        encoded_sentence_tokens: List[str] = []
        for token in original_sentence:
            if token is head.span[0]:
                encoded_sentence_tokens.append(self.encoding_strategy.encode_head(head.span, head.label))

            elif token is tail.span[0]:
                encoded_sentence_tokens.append(self.encoding_strategy.encode_tail(tail.span, tail.label))

            elif all(
                token is not non_leading_entity_token
                for non_leading_entity_token in itertools.chain(non_leading_head_tokens, non_leading_tail_tokens)
            ):
                encoded_sentence_tokens.append(token.text)

        # Create masked sentence
        encoded_sentence: EncodedSentence = EncodedSentence(
            " ".join(encoded_sentence_tokens), use_tokenizer=SpaceTokenizer()
        )

        if gold_label is not None:
            # Add gold relation annotation as sentence label
            # Using the sentence label instead of annotating a separate `Relation` object is easier to manage since,
            # during prediction, the forward pass does not need any knowledge about the entities in the sentence.
            encoded_sentence.add_label(typename=self.label_type, value=gold_label, score=1.0)
        encoded_sentence.copy_context_from_sentence(original_sentence)
        return encoded_sentence

    def _encode_sentence_for_inference(
        self,
        sentence: Sentence,
    ) -> Iterator[Tuple[EncodedSentence, Relation]]:
        """Create Encoded Sentences and Relation pairs for Inference.

        Yields encoded sentences annotated with their gold relation and
        the corresponding relation object in the original sentence for all valid entity pair permutations.
        The created encoded sentences are newly created sentences with no reference to the passed sentence.

        Important properties:
            - Every sentence has exactly one encoded head and tail entity token. Therefore, every encoded sentence has
              **exactly** one induced relation annotation, the gold annotation or `self.zero_tag_value`.
            - The created relations have head and tail spans from the original passed sentence.

        Args:
            sentence: A flair `Sentence` object with entity annotations

        Returns: Encoded sentences annotated with their gold relation and the corresponding relation in the original sentence
        """
        for head, tail, gold_label in self._entity_pair_permutations(sentence):
            masked_sentence: EncodedSentence = self._encode_sentence(
                head=head,
                tail=tail,
                gold_label=gold_label if gold_label is not None else self.zero_tag_value,
            )
            original_relation: Relation = Relation(first=head.span, second=tail.span)
            yield masked_sentence, original_relation

    def _encode_sentence_for_training(self, sentence: Sentence) -> Iterator[EncodedSentence]:
        """Create Encoded Sentences and Relation pairs for Training.

        Same as `self._encode_sentence_for_inference`.

        with the option of disabling cross augmentation via `self.cross_augmentation`
        (and that the relation with reference to the original sentence is not returned).
        """
        for head, tail, gold_label in self._entity_pair_permutations(sentence):
            if gold_label is None:
                if self.cross_augmentation:
                    gold_label = self.zero_tag_value
                else:
                    continue  # Skip generated data points that do not express an originally annotated relation

            masked_sentence: EncodedSentence = self._encode_sentence(
                head=head,
                tail=tail,
                gold_label=gold_label,
            )

            yield masked_sentence

    def transform_sentence(self, sentences: Union[Sentence, List[Sentence]]) -> List[EncodedSentence]:
        """Transforms sentences into encoded sentences specific to the `RelationClassifier`.

        For more information on the internal sentence transformation procedure,
        see the :class:`flair.models.RelationClassifier` architecture and
        the different :class:`flair.models.relation_classifier_model.EncodingStrategy` variants docstrings.

        Args:
            sentences: sentences to transform

        Returns:
            A list of encoded sentences specific to the `RelationClassifier`
        """
        if not isinstance(sentences, list):
            sentences = [sentences]

        return [
            encoded_sentence
            for sentence in sentences
            for encoded_sentence in self._encode_sentence_for_training(sentence)
        ]

    def transform_dataset(self, dataset: Dataset[Sentence]) -> FlairDatapointDataset[EncodedSentence]:
        """Transforms a dataset into a dataset containing encoded sentences specific to the `RelationClassifier`.

        The returned dataset is stored in memory.
        For more information on the internal sentence transformation procedure,
        see the :class:`RelationClassifier` architecture and
        the different :class:`EncodingStrategy` variants docstrings.

        Args:
            dataset: A dataset of sentences to transform

        Returns: A dataset of encoded sentences specific to the `RelationClassifier`
        """
        data_loader: DataLoader = DataLoader(dataset, batch_size=1)
        original_sentences: List[Sentence] = [batch[0] for batch in iter(data_loader)]
        return FlairDatapointDataset(self.transform_sentence(original_sentences))

    def transform_corpus(self, corpus: Corpus[Sentence]) -> Corpus[EncodedSentence]:
        """Transforms a corpus into a corpus containing encoded sentences specific to the `RelationClassifier`.

        The splits of the returned corpus are stored in memory.
        For more information on the internal sentence transformation procedure,
        see the :class:`RelationClassifier` architecture and
        the different :class:`EncodingStrategy` variants docstrings.

        Args:
            corpus: A corpus of sentences to transform

        Returns: A corpus of encoded sentences specific to the `RelationClassifier`
        """
        return Corpus(
            train=self.transform_dataset(corpus.train) if corpus.train is not None else None,
            dev=self.transform_dataset(corpus.dev) if corpus.dev is not None else None,
            test=self.transform_dataset(corpus.test) if corpus.test is not None else None,
            name=corpus.name,
            # If we sample missing splits, the encoded sentences that correspond to the same original sentences
            # may get distributed into different splits. For training purposes, this is always undesired.
            sample_missing_splits=False,
        )

    def _get_embedding_for_data_point(self, prediction_data_point: EncodedSentence) -> torch.Tensor:
        embedding_names: List[str] = self.embeddings.get_names()
        return prediction_data_point.get_embedding(embedding_names)

    def _get_data_points_from_sentence(self, sentence: EncodedSentence) -> List[EncodedSentence]:
        """Returns the encoded sentences to which labels are added.

        To encode sentences, use the `transform` function of the `RelationClassifier`.
        """
        # Ensure that all sentences are encoded properly
        if not isinstance(sentence, EncodedSentence):
            raise ValueError(
                "Some of the passed sentences are not encoded "
                "to be compatible with the relation classifier's forward pass.\n"
                "Did you transform your raw sentences into encoded sentences? "
                "Use the\n"
                "\t- transform_sentence\n"
                "\t- transform_dataset\n"
                "\t- transform_corpus\n"
                "functions to transform you data first. "
                "When using the ModelTrainer to train a relation classification model, "
                "be sure to pass a transformed corpus:\n"
                "WRONG:   trainer: ModelTrainer = ModelTrainer(model=model, corpus=corpus)\n"
                "CORRECT: trainer: ModelTrainer = ModelTrainer(model=model, corpus=model.transform_corpus(corpus))"
            )

        return [sentence]

    def predict(
        self,
        sentences: Union[List[Sentence], List[EncodedSentence], Sentence, EncodedSentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: EmbeddingStorageMode = "none",
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """Predicts the class labels for the given sentence(s).

        Standard `Sentence` objects and `EncodedSentences` specific to the `RelationClassifier` are allowed as input.
        The (relation) labels are directly added to the sentences.

        Args:
            sentences: A list of (encoded) sentences.
            mini_batch_size: The mini batch size to use
            return_probabilities_for_all_classes: Return probabilities for all classes instead of only best predicted
            verbose: Set to display a progress bar
            return_loss: Set to return loss
            label_name: Set to change the predicted label type name
            embedding_storage_mode: The default is 'none', which is always best. Only set to 'cpu' or 'gpu' if you wish to predict and keep the generated embeddings in CPU or GPU memory, respectively.

        Returns: The loss and the total number of classes, if `return_loss` is set
        """
        prediction_label_type: str = self.label_type if label_name is None else label_name

        if not isinstance(sentences, list):
            sentences = [sentences]

        loss: Optional[Tuple[torch.Tensor, int]]
        encoded_sentences: List[EncodedSentence]

        if all(isinstance(sentence, EncodedSentence) for sentence in sentences):
            # Deal with the case where all sentences are encoded sentences

            # mypy does not infer the type of "sentences" restricted by the if statement
            encoded_sentences = cast(List[EncodedSentence], sentences)
            loss = super().predict(
                encoded_sentences,
                mini_batch_size=mini_batch_size,
                return_probabilities_for_all_classes=return_probabilities_for_all_classes,
                verbose=verbose,
                label_name=prediction_label_type,
                return_loss=return_loss,
                embedding_storage_mode=embedding_storage_mode,
            )

        elif all(not isinstance(sentence, EncodedSentence) for sentence in sentences):
            # Deal with the case where all sentences are standard (non-encoded) sentences
            Sentence.set_context_for_sentences(cast(List[Sentence], sentences))
            sentences_with_relation_reference: List[Tuple[EncodedSentence, Relation]] = list(
                itertools.chain.from_iterable(self._encode_sentence_for_inference(sentence) for sentence in sentences)
            )

            encoded_sentences = [x[0] for x in sentences_with_relation_reference]
            loss = super().predict(
                encoded_sentences,
                mini_batch_size=mini_batch_size,
                return_probabilities_for_all_classes=return_probabilities_for_all_classes,
                verbose=verbose,
                label_name=prediction_label_type,
                return_loss=return_loss,
                embedding_storage_mode=embedding_storage_mode,
            )

            # For each encoded sentence, transfer its prediction onto the original relation
            for encoded_sentence, original_relation in sentences_with_relation_reference:
                for label in encoded_sentence.get_labels(prediction_label_type):
                    original_relation.add_label(prediction_label_type, value=label.value, score=label.score)

        else:
            raise ValueError("All passed sentences must be either uniformly encoded or not.")

        return loss if return_loss else None

    def _get_state_dict(self) -> Dict[str, Any]:
        model_state: Dict[str, Any] = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "entity_label_types": self.entity_label_types,
            "entity_pair_labels": self.entity_pair_labels,
            "entity_threshold": self.entity_threshold,
            "cross_augmentation": self.cross_augmentation,
            "encoding_strategy": self.encoding_strategy,
            "zero_tag_value": self.zero_tag_value,
            "allow_unk_tag": self.allow_unk_tag,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state: Dict[str, Any], **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state["embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            entity_label_types=state["entity_label_types"],
            entity_pair_labels=state["entity_pair_labels"],
            entity_threshold=state["entity_threshold"],
            cross_augmentation=state["cross_augmentation"],
            encoding_strategy=state["encoding_strategy"],
            zero_tag_value=state["zero_tag_value"],
            allow_unk_tag=state["allow_unk_tag"],
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

    def get_used_tokens(
        self, corpus: Corpus, context_length: int = 0, respect_document_boundaries: bool = True
    ) -> typing.Iterable[List[str]]:
        yield from super().get_used_tokens(corpus, context_length, respect_document_boundaries)
        for sentence in _iter_dataset(corpus.get_all_sentences()):
            for span in sentence.get_spans(self.label_type):
                yield self.encoding_strategy.encode_head(span, span.get_label(self.label_type)).split(" ")
                yield self.encoding_strategy.encode_tail(span, span.get_label(self.label_type)).split(" ")

    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "RelationClassifier":
        from typing import cast

        return cast("RelationClassifier", super().load(model_path=model_path))
