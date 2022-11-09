import collections as co
import itertools
from operator import itemgetter
from typing import (
    Any,
    Counter,
    DefaultDict,
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
from torch.utils.data.dataset import ConcatDataset, Dataset

import flair
from flair.data import Corpus, Dictionary, Label, Relation, Sentence, Span, Token
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import DocumentEmbeddings
from flair.tokenization import SpaceTokenizer


class EncodedSentence(Sentence):
    """
    This class is a wrapper of the regular `Sentence` object
    that expresses that a sentence is encoded and compatible with the relation classifier.
    For inference, i.e. `predict` and `evaluate`, the relation classifier internally encodes the sentences.
    Therefore, these functions work with the regular flair sentence objects.
    """

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
#   This depends if this relation classification architecture should replace or offer as an alternative.
class RelationClassifier(flair.nn.DefaultClassifier[EncodedSentence, EncodedSentence]):
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
    The model generates an encoded sentence for each entity pair
    in the cross product of all entities in the original sentence.
    In the encoded representation, the entities in the current entity pair are masked with special control tokens.
    (For an example, see the docstring of the `_encode_sentence_for_training` function.)
    Then, for each encoded sentence, the model takes its document embedding and puts the resulting
    text representation(s) through a linear layer to get the class relation label.

    Note: Currently, the model has no multi-label support.
    """

    def __init__(
        self,
        embeddings: DocumentEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        entity_label_types: Union[str, Sequence[str], Dict[str, Optional[Set[str]]]],
        entity_pair_labels: Optional[Set[Tuple[str, str]]] = None,
        entity_threshold: Optional[float] = None,
        zero_tag_value: str = "O",
        allow_unk_tag: bool = True,
        cross_augmentation: bool = True,
        mask_type: str = "mark",
        **classifierargs,
    ) -> None:
        """
        Initializes a `RelationClassifier`.

        :param embeddings: The document embeddings used to embed each sentence
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
        :param cross_augmentation: If `True`, use cross augmentation to transform `Sentence`s into `EncodedSentenece`s.
                                   When cross augmentation is enabled, the transformation functions,
                                   e.g. `transform_corpus`, generate an encoded sentence for each entity pair
                                   in the cross product of all entities in the original sentence.
                                   When disabling cross augmentation, the transform functions only generate
                                   encoded sentences for each gold relation annotation in the original sentence.
        :param classifierargs: The remaining parameters passed to the underlying `DefaultClassifier`
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

        self.cross_augmentation = cross_augmentation
        self.mask_type = mask_type
        self.entity_threshold = entity_threshold

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

    def _mask(self, entity: _Entity, role: str) -> str:
        if self.mask_type == "label-aware":
            return f"[{role}-{entity.label.value}]"
        if self.mask_type == "entity":
            return f"[{role}-ENTITY]"
        if self.mask_type == "mark":
            return f"[[{role}-{entity.span.text}]]"

        # by default, use "mark" masking
        return f"[[{role}-{entity.span.text}]]"

    def _create_masked_sentence(
        self,
        head: _Entity,
        tail: _Entity,
        gold_label: Optional[str] = None,
    ) -> EncodedSentence:
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
        :param gold_label: An optional gold label of the induced relation by the head and tail entity
        :return: The masked sentence (with gold annotations)
        """
        # Some sanity checks
        original_sentence: Sentence = head.span.sentence
        assert original_sentence is tail.span.sentence, "The head and tail need to come from the same sentence."

        # Pre-compute non-leading head, tail and remainder tokens for entity masking
        non_leading_head_tokens: List[Token] = head.span.tokens[1:]
        non_leading_tail_tokens: List[Token] = tail.span.tokens[1:]

        # We can not use the plaintext of the head/tail span in the sentence as the mask
        # since there may be multiple occurrences of the same entity mentioned in the sentence.
        # Therefore, we use the span's position in the sentence.
        masked_sentence_tokens: List[str] = []
        for token in original_sentence:

            if token is head.span[0]:
                masked_sentence_tokens.append(self._mask(entity=head, role="H"))

            elif token is tail.span[0]:
                masked_sentence_tokens.append(self._mask(entity=tail, role="T"))

            elif all(
                token is not non_leading_entity_token
                for non_leading_entity_token in itertools.chain(non_leading_head_tokens, non_leading_tail_tokens)
            ):
                masked_sentence_tokens.append(token.text)

        # Create masked sentence
        masked_sentence: EncodedSentence = EncodedSentence(
            " ".join(masked_sentence_tokens), use_tokenizer=SpaceTokenizer()
        )

        if gold_label is not None:
            # Add gold relation annotation as sentence label
            # Using the sentence label instead of annotating a separate `Relation` object is easier to manage since,
            # during prediction, the forward pass does not need any knowledge about the entities in the sentence.
            masked_sentence.add_label(typename=self.label_type, value=gold_label, score=1.0)

        return masked_sentence

    def _encode_sentence_for_inference(self, sentence: Sentence) -> Iterator[Tuple[EncodedSentence, Relation]]:
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
            masked_sentence: EncodedSentence = self._create_masked_sentence(
                head=head,
                tail=tail,
                gold_label=gold_label if gold_label is not None else self.zero_tag_value,
            )
            original_relation: Relation = Relation(first=head.span, second=tail.span)
            yield masked_sentence, original_relation

    def _encode_sentence_for_training(self, sentence: Sentence) -> Iterator[EncodedSentence]:
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

            masked_sentence: EncodedSentence = self._create_masked_sentence(
                head=head,
                tail=tail,
                gold_label=gold_label,
            )

            yield masked_sentence

    def transform_sentence(self, sentences: Union[Sentence, List[Sentence]]) -> List[EncodedSentence]:
        """
        Transforms sentences into encoded sentences specific to the `RelationClassifier`.
        For more information on the internal sentence transformation procedure,
        see the `RelationClassifier` architecture docstring and
        the `_encode_sentence_for_training` and `_encode_sentence_for_inference` docstrings.

        :param sentences: A (list) of sentence(s) to transform
        :return: A list of encoded sentences specific to the `RelationClassifier`
        """
        if not isinstance(sentences, list):
            sentences = [sentences]

        return [
            encoded_sentence
            for sentence in sentences
            for encoded_sentence in self._encode_sentence_for_training(sentence)
        ]

    def transform_dataset(self, dataset: Dataset[Sentence]) -> FlairDatapointDataset[EncodedSentence]:
        """
        Transforms a dataset into a dataset containing encoded sentences specific to the `RelationClassifier`.
        The returned dataset is stored in memory.
        For more information on the internal sentence transformation procedure,
        see the `RelationClassifier` architecture docstring and
        the `_encode_sentence_for_training` and `_encode_sentence_for_inference` docstrings.

        :param dataset: A dataset of sentences to transform
        :return: A dataset of encoded sentences specific to the `RelationClassifier`
        """
        data_loader: DataLoader = DataLoader(dataset, batch_size=1, num_workers=0)
        original_sentences: List[Sentence] = [batch[0] for batch in iter(data_loader)]
        return FlairDatapointDataset(self.transform_sentence(original_sentences))

    def transform_corpus(self, corpus: Corpus[Sentence]) -> Corpus[EncodedSentence]:
        """
        Transforms a corpus into a corpus containing encoded sentences specific to the `RelationClassifier`.
        The splits of the returned corpus are stored in memory.
        For more information on the internal sentence transformation procedure,
        see the `RelationClassifier` architecture docstring and
        the `_encode_sentence_for_training` and `_encode_sentence_for_inference` docstrings.

        :param corpus: A corpus of sentences to transform
        :return: A corpus of encoded sentences specific to the `RelationClassifier`
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
        """
        Returns the encoded sentences to which labels are added.
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
        embedding_storage_mode: str = "none",
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Predicts the class labels for the given sentence(s).
        Standard `Sentence` objects and `EncodedSentences` specific to the `RelationClassifier` are allowed as input.
        The (relation) labels are directly added to the sentences.

        :param sentences: A list of (encoded) sentences.
        :param mini_batch_size: The mini batch size to use
        :param return_probabilities_for_all_classes: Return probabilities for all classes instead of only best predicted
        :param verbose: Set to display a progress bar
        :param return_loss: Set to return loss
        :param label_name: Set to change the predicted label type name
        :param embedding_storage_mode: The default is 'none', which is always best.
                                       Only set to 'cpu' or 'gpu' if you wish to predict
                                       and keep the generated embeddings in CPU or GPU memory, respectively.
        :return: The loss and the total number of classes, if `return_loss` is set
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
            "embeddings": self.embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "entity_label_types": self.entity_label_types,
            "entity_pair_labels": self.entity_pair_labels,
            "entity_threshold": self.entity_threshold,
            "zero_tag_value": self.zero_tag_value,
            "allow_unk_tag": self.allow_unk_tag,
            "cross_augmentation": self.cross_augmentation,
            "mask_type": self.mask_type,
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
            zero_tag_value=state["zero_tag_value"],
            allow_unk_tag=state["allow_unk_tag"],
            cross_augmentation=state["cross_augmentation"],
            mask_type=state["mask_type"],
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


def inspect_relations(
    corpus: Corpus[Sentence],
    relation_label_type: str,
    entity_label_types: Optional[Union[Sequence[str], str]] = None,
) -> DefaultDict[str, Counter[Tuple[str, str]]]:
    if entity_label_types is not None and not isinstance(entity_label_types, Sequence):
        entity_label_types = [entity_label_types]

    # Dictionary of [<relation label>, <counter of relation entity labels (HEAD, TAIL)>]
    relations: DefaultDict[str, Counter[Tuple[str, str]]] = co.defaultdict(co.Counter)

    data_loader: DataLoader = DataLoader(
        ConcatDataset(split for split in [corpus.train, corpus.dev, corpus.test] if split is not None),
        batch_size=1,
        num_workers=0,
    )
    for sentence in map(itemgetter(0), data_loader):
        for relation in sentence.get_relations(relation_label_type):
            entity_counter = relations[relation.get_label(relation_label_type).value]

            head_relation_label: str
            tail_relation_label: str
            if entity_label_types is None:
                head_relation_label = relation.first.get_label().value
                tail_relation_label = relation.second.get_label().value
            else:
                head_relation_label = next(
                    relation.first.get_label(label_type).value for label_type in entity_label_types
                )
                tail_relation_label = next(
                    relation.second.get_label(label_type).value for label_type in entity_label_types
                )

            entity_counter.update([(head_relation_label, tail_relation_label)])

    return relations
