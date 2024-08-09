import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch.utils.data import Dataset

import flair.data
from flair.data import Corpus, Sentence, Token
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.file_utils import hf_download
from flair.models.sequence_tagger_model import SequenceTagger


class PrefixedSentence(Sentence):
    """An PrefixedSentence expresses that a sentence is augmented and compatible with the PrefixedSequenceTagger.

    For inference, i.e. `predict` and `evaluate`, the PrefixedSequenceTagger internally encodes the sentences.
    Therefore, these functions work with the regular flair sentence objects.
    """


class SentenceAugmentationStrategy(ABC):
    """Strategy to augment a sentence with additional information or instructions."""

    @abstractmethod
    def augment_sentence(
        self, sentence: Sentence, annotation_layers: Optional[Union[str, List[str]]] = None
    ) -> PrefixedSentence:
        """Augments the given sentence text with additional instructions for working / predicting the task on the given annotations.

        Args:
            sentence: The sentence to be augmented
            annotation_layers: Annotations which should be predicted.
        """
        ...

    @abstractmethod
    def apply_predictions(
        self,
        augmented_sentence: Sentence,
        original_sentence: Sentence,
        source_annotation_layer: str,
        target_annotation_layer: str,
    ):
        """Transfers the predictions made on the augmented sentence to the original one.

        Args:
              augmented_sentence: The augmented sentence instance
              original_sentence: The original sentence before the augmentation was applied
              source_annotation_layer: Annotation layer of the augmented sentence in which the predictions are stored.
              target_annotation_layer: Annotation layer in which the predictions should be stored in the original sentence.
        """
        ...

    @abstractmethod
    def _get_state_dict(self):
        """Returns the state dict for the given augmentation strategy."""
        ...

    @classmethod
    def _init_strategy_with_state_dict(cls, state, **kwargs):
        """Initializes the strategy from the given state."""

    def augment_dataset(
        self, dataset: Dataset[Sentence], annotation_layers: Optional[Union[str, List[str]]] = None
    ) -> FlairDatapointDataset[PrefixedSentence]:
        """Transforms a dataset into a dataset containing augmented sentences specific to the `PrefixedSequenceTagger`.

        The returned dataset is stored in memory. For more information on the internal sentence transformation
        procedure, see the :class:`PrefixedSequenceTagger` architecture.

        Args:
            dataset: A dataset of sentences to augment
            annotation_layers: Annotations which should be predicted.

        Returns: A dataset of augmented sentences specific to the `PrefixedSequenceTagger`
        """
        data_loader: DataLoader = DataLoader(dataset, batch_size=1)
        original_sentences: List[Sentence] = [batch[0] for batch in iter(data_loader)]

        augmented_sentences = [self.augment_sentence(sentence, annotation_layers) for sentence in original_sentences]

        return FlairDatapointDataset(augmented_sentences)

    def augment_corpus(
        self, corpus: Corpus[Sentence], annotation_layers: Optional[Union[str, List[str]]] = None
    ) -> Corpus[PrefixedSentence]:
        """Transforms a corpus into a corpus containing augmented sentences specific to the `PrefixedSequenceTagger`.

        The splits of the returned corpus are stored in memory. For more information on the internal
        sentence augmentation procedure, see the :class:`PrefixedSequenceTagger`.

        Args:
            corpus: A corpus of sentences to augment
            annotation_layers: Annotations which should be predicted.

        Returns: A corpus of encoded sentences specific to the `PrefixedSequenceTagger`
        """
        return Corpus(
            train=self.augment_dataset(corpus.train, annotation_layers) if corpus.train is not None else None,
            dev=self.augment_dataset(corpus.dev, annotation_layers) if corpus.dev is not None else None,
            test=self.augment_dataset(corpus.test, annotation_layers) if corpus.test is not None else None,
            name=corpus.name,
            # If we sample missing splits, the encoded sentences that correspond to the same original sentences
            # may get distributed into different splits. For training purposes, this is always undesired.
            sample_missing_splits=False,
        )


class EntityTypeTaskPromptAugmentationStrategy(SentenceAugmentationStrategy):
    """Augmentation strategy that augments a sentence with a task description which specifies which entity types should be tagged.

    This approach is inspired by the paper from Luo et al.:
    AIONER: All-in-one scheme-based biomedical named entity recognition using deep learning
    https://arxiv.org/abs/2211.16944

    Example:
        "[Tag gene and disease] Mutations in the TP53 tumour suppressor gene are found in ~50% of human cancers"
    """

    def __init__(self, entity_types: List[str]):
        if len(entity_types) <= 0:
            raise AssertionError

        self.entity_types = entity_types
        self.task_prompt = self._build_tag_prompt_prefix(entity_types)

    def augment_sentence(
        self, sentence: Sentence, annotation_layers: Optional[Union[str, List[str]]] = None
    ) -> PrefixedSentence:
        # Prepend the task description prompt to the sentence text
        augmented_sentence = PrefixedSentence(
            text=self.task_prompt + [t.text for t in sentence.tokens],
            use_tokenizer=False,
            language_code=sentence.language_code,
            start_position=sentence.start_position,
        )

        # Make sure it's a list
        if annotation_layers and isinstance(annotation_layers, str):
            annotation_layers = [annotation_layers]

        # Reconstruct all annotations from the original sentence (necessary for learning classifiers)
        layers = annotation_layers if annotation_layers else sentence.annotation_layers.keys()
        len_task_prompt = len(self.task_prompt)

        for layer in layers:
            for label in sentence.get_labels(layer):
                if isinstance(label.data_point, Token):
                    label_span = augmented_sentence[
                        len_task_prompt + label.data_point.idx - 1 : len_task_prompt + label.data_point.idx
                    ]
                else:
                    label_span = augmented_sentence[
                        len_task_prompt
                        + label.data_point.tokens[0].idx
                        - 1 : len_task_prompt
                        + label.data_point.tokens[-1].idx
                    ]

                label_span.add_label(layer, label.value, label.score)

        return augmented_sentence

    def apply_predictions(
        self,
        augmented_sentence: Sentence,
        original_sentence: Sentence,
        source_annotation_layer: str,
        target_annotation_layer: str,
    ):
        new_labels = augmented_sentence.get_labels(source_annotation_layer)
        len_task_prompt = len(self.task_prompt)

        for label in new_labels:
            if label.data_point.tokens[0].idx - len_task_prompt - 1 < 0:
                continue
            orig_span = original_sentence[
                label.data_point.tokens[0].idx - len_task_prompt - 1 : label.data_point.tokens[-1].idx - len_task_prompt
            ]
            orig_span.add_label(target_annotation_layer, label.value, label.score)

    def _build_tag_prompt_prefix(self, entity_types: List[str]) -> List[str]:
        if len(self.entity_types) == 1:
            prompt = f"[ Tag {entity_types[0]} ]"
        else:
            prompt = "[ Tag " + ", ".join(entity_types[:-1]) + " and " + entity_types[-1] + " ]"

        return prompt.split()

    def _get_state_dict(self):
        return {"entity_types": self.entity_types}

    @classmethod
    def _init_strategy_with_state_dict(cls, state, **kwargs):
        return cls(state["entity_types"])


class PrefixedSequenceTagger(SequenceTagger):
    def __init__(self, *args, augmentation_strategy: SentenceAugmentationStrategy, **kwargs):
        super().__init__(*args, **kwargs)

        if augmentation_strategy is None:
            logging.warning("No augmentation strategy provided. Make sure that the strategy is set.")

        self.augmentation_strategy = augmentation_strategy

    def _get_state_dict(self):
        state = super()._get_state_dict()
        state["augmentation_strategy"] = self.augmentation_strategy

        return state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        strategy = state["augmentation_strategy"]
        return super()._init_model_with_state_dict(state, augmentation_strategy=strategy, **kwargs)

    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "PrefixedSequenceTagger":
        from typing import cast

        return cast("PrefixedSequenceTagger", super().load(model_path=model_path))

    def forward_loss(self, sentences: Union[List[Sentence], List[PrefixedSentence]]) -> Tuple[torch.Tensor, int]:
        # If all sentences are not augmented -> augment them
        if all(isinstance(sentence, Sentence) for sentence in sentences):
            # mypy does not infer the type of "sentences" restricted by the if statement
            sentences = cast(List[Sentence], sentences)

            sentences = self.augment_sentences(sentences=sentences, annotation_layers=self.tag_type)
        elif not all(isinstance(sentence, PrefixedSentence) for sentence in sentences):
            raise ValueError("All passed sentences must be either uniformly augmented or not.")

        # mypy does not infer the type of "sentences" restricted by code above
        sentences = cast(List[Sentence], sentences)

        return super().forward_loss(sentences)

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence, List[PrefixedSentence], PrefixedSentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
        force_token_predictions: bool = False,
    ):
        # Compute prediction label type
        prediction_label_type: str = self.label_type if label_name is None else label_name

        # make sure it's a list
        if not isinstance(sentences, list) and not isinstance(sentences, flair.data.Dataset):
            sentences = [sentences]

        # If all sentences are already augmented (i.e. compatible with this class), just forward the sentences
        if all(isinstance(sentence, PrefixedSentence) for sentence in sentences):
            # mypy does not infer the type of "sentences" restricted by the if statement
            sentences = cast(List[Sentence], sentences)

            return super().predict(
                sentences,
                mini_batch_size=mini_batch_size,
                return_probabilities_for_all_classes=return_probabilities_for_all_classes,
                verbose=verbose,
                label_name=prediction_label_type,
                return_loss=return_loss,
                embedding_storage_mode=embedding_storage_mode,
            )

        elif not all(isinstance(sentence, Sentence) for sentence in sentences):
            raise ValueError("All passed sentences must be either uniformly augmented or not.")

        # Remove existing labels
        if label_name is not None:
            for sentence in sentences:
                sentence.remove_labels(prediction_label_type)

        sentences = cast(List[Sentence], sentences)

        # Augment sentences - copy all annotation of the given tag type
        augmented_sentences = self.augment_sentences(sentences, self.tag_type)

        mypy_safe_augmented_sentences = cast(List[Sentence], augmented_sentences)

        # Predict on augmented sentence and store it in an internal annotation layer / label
        loss_and_count = super().predict(
            sentences=mypy_safe_augmented_sentences,
            mini_batch_size=mini_batch_size,
            return_probabilities_for_all_classes=return_probabilities_for_all_classes,
            verbose=verbose,
            label_name=prediction_label_type,
            return_loss=return_loss,
            embedding_storage_mode=embedding_storage_mode,
        )

        # Append predicted labels to the original sentences
        for orig_sent, aug_sent in zip(sentences, augmented_sentences):
            self.augmentation_strategy.apply_predictions(
                aug_sent, orig_sent, prediction_label_type, prediction_label_type
            )

            if prediction_label_type == "predicted":
                orig_sent.remove_labels("predicted_bio")
                orig_sent.remove_labels("gold_bio")

        if loss_and_count is not None:
            return loss_and_count

    def augment_sentences(
        self, sentences: Union[Sentence, List[Sentence]], annotation_layers: Optional[Union[str, List[str]]] = None
    ) -> List[PrefixedSentence]:
        if not isinstance(sentences, list) and not isinstance(sentences, flair.data.Dataset):
            sentences = [sentences]

        return [self.augmentation_strategy.augment_sentence(sentence, annotation_layers) for sentence in sentences]

    @staticmethod
    def _fetch_model(model_name) -> str:
        huggingface_model_map = {"hunflair2": "hunflair/hunflair2-ner", "bioner": "hunflair/hunflair2-ner"}

        # check if model name is a valid local file
        if Path(model_name).exists():
            model_path = model_name

        # check if model name is a pre-configured hf model
        elif model_name in huggingface_model_map:
            hf_model_name = huggingface_model_map[model_name]
            return hf_download(hf_model_name)

        else:
            model_path = hf_download(model_name)

        return model_path
