import bisect
import copy
import logging
import re
import typing
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable
from operator import itemgetter
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union, cast

import torch
from deprecated.sphinx import deprecated
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataset import ConcatDataset, Subset
from torch import device as torch_device  # Import torch.device for type hint

import flair
from flair.file_utils import Tqdm
from flair.tokenization import SegtokTokenizer, SpaceTokenizer, Tokenizer, NoTokenizer

T_co = typing.TypeVar("T_co", covariant=True)

log = logging.getLogger("flair")


def _iter_dataset(dataset: Optional[Dataset]) -> typing.Iterable:
    """Iterates over a Dataset yielding single data points.

    Args:
        dataset (Optional[Dataset]): The dataset to iterate over.

    Returns:
        typing.Iterable: An iterable yielding individual data points.
    """
    if dataset is None:
        return []
    from flair.datasets import DataLoader

    return (x[0] for x in DataLoader(dataset, batch_size=1))


def _len_dataset(dataset: Optional[Dataset]) -> int:
    """Calculates the length (number of data points) in a Dataset.

    Args:
        dataset (Optional[Dataset]): The dataset whose length is required.

    Returns:
        int: The number of data points in the dataset, or 0 if the dataset is None.
    """
    if dataset is None:
        return 0
    from flair.datasets import DataLoader

    loader = DataLoader(dataset, batch_size=1)
    return len(loader)


class BoundingBox(NamedTuple):
    """Represents a bounding box with left, top, right, and bottom coordinates."""

    left: str
    top: int
    right: int
    bottom: int


class Dictionary:
    """This class holds a dictionary that maps strings to unique integer IDs.

    Used throughout Flair for representing words, tags, characters, etc.
    Handles unknown items (<unk>) and flags for multi-label or span tasks.
    Items are stored internally as bytes for efficiency.
    """

    def __init__(self, add_unk: bool = True) -> None:
        """Initializes a Dictionary.

        Args:
            add_unk (bool, optional): If True, adds a special '<unk>' item.
                Defaults to True.
        """
        # init dictionaries
        self.item2idx: dict[bytes, int] = {}
        self.idx2item: list[bytes] = []
        self.add_unk = add_unk
        self.multi_label = False
        self.span_labels = False
        # in order to deal with unknown tokens, add <unk>
        if add_unk:
            self.add_item("<unk>")

    def remove_item(self, item: str):
        """Removes an item from the dictionary.

        Note: This operation might be slow for large dictionaries as it involves
        list removal. It currently doesn't re-index subsequent items.

        Args:
            item (str): The string item to remove.
        """
        bytes_item = item.encode("utf-8")
        if bytes_item in self.item2idx:
            self.idx2item.remove(bytes_item)
            del self.item2idx[bytes_item]

    def add_item(self, item: str) -> int:
        """Adds a string item to the dictionary.

        If the item exists, returns its ID. Otherwise, adds it and returns the new ID.

        Args:
            item (str): The string item to add.

        Returns:
            int: The integer ID of the item.
        """
        bytes_item = item.encode("utf-8")
        if bytes_item not in self.item2idx:
            self.idx2item.append(bytes_item)
            self.item2idx[bytes_item] = len(self.idx2item) - 1
        return self.item2idx[bytes_item]

    def get_idx_for_item(self, item: str) -> int:
        """Retrieves the integer ID for a given string item.

        Args:
            item (str): The string item.

        Returns:
            int: The integer ID. Returns 0 if item is not found and `add_unk` is True.

        Raises:
            IndexError: If the item is not found and `add_unk` is False.
        """
        item_encoded = item.encode("utf-8")
        if item_encoded in self.item2idx:
            return self.item2idx[item_encoded]
        elif self.add_unk:
            return 0
        else:
            log.error(f"The string '{item}' is not in dictionary! Dictionary contains only: {self.get_items()}")
            log.error(
                "You can create a Dictionary that handles unknown items with an <unk>-key by setting add_unk = True in the construction."
            )
            raise IndexError

    def get_idx_for_items(self, items: list[str]) -> list[int]:
        """Retrieves the integer IDs for a list of string items. (No cache version)"""
        if not items:
            return []

        indices: list[int] = []
        unk_idx = 0  # Assuming <unk> is index 0 if add_unk is True

        for item in items:
            item_bytes = item.encode("utf-8")
            idx = self.item2idx.get(item_bytes)  # Look up bytes directly

            if idx is not None:
                indices.append(idx)
            elif self.add_unk:
                indices.append(unk_idx)  # Append 0 for <unk>
            else:
                # Raise error if not found and add_unk is False
                log.error(f"Item '{item}' not found in dictionary (add_unk=False).")
                # ... (error logging) ...
                raise IndexError(f"Item '{item}' not found in dictionary.")

        return indices

    def get_items(self) -> list[str]:
        """Returns a list of all items in the dictionary in order of their IDs."""
        return [item.decode("UTF-8") for item in self.idx2item]

    def __len__(self) -> int:
        """Returns the total number of items in the dictionary."""
        return len(self.idx2item)

    def get_item_for_index(self, idx: int) -> str:
        """Retrieves the string item corresponding to a given integer ID.

        Args:
            idx (int): The integer ID.

        Returns:
            str: The string item.

        Raises:
            IndexError: If the index is out of bounds.
        """
        return self.idx2item[idx].decode("UTF-8")

    def has_item(self, item: str) -> bool:
        """Checks if a given string item exists in the dictionary."""
        return item.encode("utf-8") in self.item2idx

    def set_start_stop_tags(self) -> None:
        """Adds special <START> and <STOP> tags to the dictionary (often used for CRFs)."""
        self.add_item("<START>")
        self.add_item("<STOP>")

    def is_span_prediction_problem(self) -> bool:
        """Checks if the dictionary likely represents BIOES/BIO span labels.

        Returns True if `span_labels` flag is set or any item starts with 'B-', 'I-', 'S-'.

        Returns:
            bool: True if likely span labels, False otherwise.
        """
        if self.span_labels:
            return True
        return any(item.startswith(("B-", "S-", "I-")) for item in self.get_items())

    def start_stop_tags_are_set(self) -> bool:
        """Checks if <START> and <STOP> tags have been added."""
        return {b"<START>", b"<STOP>"}.issubset(self.item2idx.keys())

    def save(self, savefile: PathLike):
        """Saves the dictionary mapping to a file using pickle.

        Args:
            savefile (PathLike): The path to the output file.
        """
        import pickle

        with open(savefile, "wb") as f:
            mappings = {"idx2item": self.idx2item, "item2idx": self.item2idx}
            pickle.dump(mappings, f)

    def __setstate__(self, d: dict) -> None:
        self.__dict__ = d
        # set 'add_unk' if the dictionary was created with a version of Flair older than 0.9
        if "add_unk" not in self.__dict__:
            self.__dict__["add_unk"] = b"<unk>" in self.__dict__["idx2item"]

    @classmethod
    def load_from_file(cls, filename: Union[str, Path]) -> "Dictionary":
        """Loads a Dictionary previously saved using the `.save()` method.

        Args:
            filename (Union[str, Path]): Path to the saved dictionary file.

        Returns:
            Dictionary: The loaded Dictionary object.
        """
        import pickle

        with Path(filename).open("rb") as f:
            mappings = pickle.load(f, encoding="latin1")
            idx2item = mappings["idx2item"]
            item2idx = mappings["item2idx"]

        # set 'add_unk' depending on whether <unk> is a key
        add_unk = b"<unk>" in idx2item

        dictionary: Dictionary = Dictionary(add_unk=add_unk)
        dictionary.item2idx = item2idx
        dictionary.idx2item = idx2item
        return dictionary

    @classmethod
    def load(cls, name: str) -> "Dictionary":
        """Loads a pre-built character dictionary or a dictionary from a file path.

        Args:
            name (str): The name of the pre-built dictionary (e.g., 'chars')
                        or a path to a dictionary file.

        Returns:
            Dictionary: The loaded Dictionary object.

        Raises:
            ValueError: If the name is not recognized or the path is invalid.
        """
        from flair.file_utils import cached_path

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/characters"
        if name == "chars" or name == "common-chars":
            char_dict = cached_path(f"{hu_path}/common_characters", cache_dir="datasets")
            return Dictionary.load_from_file(char_dict)

        if name == "chars-large" or name == "common-chars-large":
            char_dict = cached_path(f"{hu_path}/common_characters_large", cache_dir="datasets")
            return Dictionary.load_from_file(char_dict)

        if name == "chars-xl" or name == "common-chars-xl":
            char_dict = cached_path(f"{hu_path}/common_characters_xl", cache_dir="datasets")
            return Dictionary.load_from_file(char_dict)

        if name == "chars-lemmatizer" or name == "common-chars-lemmatizer":
            char_dict = cached_path(f"{hu_path}/common_characters_lemmatizer", cache_dir="datasets")
            return Dictionary.load_from_file(char_dict)

        return Dictionary.load_from_file(name)

    def __eq__(self, o: object) -> bool:
        """Checks if two Dictionary objects are equal based on content and flags."""
        if not isinstance(o, Dictionary):
            return False
        return self.item2idx == o.item2idx and self.idx2item == o.idx2item and self.add_unk == o.add_unk

    def __str__(self) -> str:
        tags = ", ".join(self.get_item_for_index(i) for i in range(min(len(self), 50)))
        return f"Dictionary with {len(self)} tags: {tags}"


class Label:
    """Represents a label assigned to a DataPoint (e.g., Token, Span, Sentence).

    Attributes:
        data_point (DataPoint): The data point this label is attached to.
        value (str): The string value of the label (e.g., "PERSON", "POSITIVE").
        score (float): The confidence score of the label (0.0 to 1.0).
        metadata (dict): A dictionary for storing arbitrary additional metadata.
        typename (Optional[str]): The name of the annotation layer (set via `DataPoint.add_label`).
    """

    def __init__(self, data_point: "DataPoint", value: str, score: float = 1.0, **metadata) -> None:
        """Initializes a Label.

        Args:
            data_point (DataPoint): The data point this label describes.
            value (str): The label's string value.
            score (float, optional): The confidence score (0.0-1.0). Defaults to 1.0.
            **metadata: Arbitrary keyword arguments stored as metadata.
        """
        self.data_point = data_point
        self._value = value
        self._score = score
        self.metadata = metadata
        # Add a new attribute to store the typename
        self._typename: Optional[str] = None
        super().__init__()

    def set_value(self, value: str, score: float = 1.0):
        """Updates the value and score of the label."""
        self._value = value
        self._score = score

    @property
    def value(self) -> str:
        """The string value of the label."""
        return self._value

    @property
    def score(self) -> float:
        """The confidence score of the label (between 0.0 and 1.0)."""
        return self._score

    def to_dict(self):
        return {"value": self.value, "confidence": self.score}

    def __str__(self) -> str:
        return f"{self.data_point.unlabeled_identifier}{flair._arrow}{self._value}{self.metadata_str} ({round(self._score, 4)})"

    @property
    def shortstring(self):
        return f'"{self.data_point.text}"/{self._value}'

    def __repr__(self) -> str:
        return f"'{self.data_point.unlabeled_identifier}'/'{self._value}'{self.metadata_str} ({round(self._score, 4)})"

    def __eq__(self, other):
        return self.value == other.value and self.score == other.score and self.data_point == other.data_point

    def __hash__(self):
        return hash(self.__repr__())

    def __lt__(self, other):
        return self.data_point < other.data_point

    @property
    def metadata_str(self) -> str:
        if not self.metadata:
            return ""
        rep = "/".join(f"{k}={v}" for k, v in self.metadata.items())
        return f"/{rep}"

    @property
    def labeled_identifier(self):
        return f"{self.data_point.unlabeled_identifier}/{self.value}"

    @property
    def unlabeled_identifier(self):
        return f"{self.data_point.unlabeled_identifier}"

    @property
    def typename(self) -> Optional[str]:
        """The name of the annotation layer this label belongs to (e.g., "ner")."""
        if self._typename is not None:
            return self._typename

        # Find the typename by checking which label type this label belongs to
        # Note: this should rarely if ever be triggered, as labels are usually added via the DataPoint.add_label() method
        if self.data_point is not None:
            for type_name, labels in self.data_point.annotation_layers.items():
                if self in labels:
                    self._typename = type_name
                    return type_name

        return None

    # Add a setter for typename to be used when creating the label
    @typename.setter
    def typename(self, value: str) -> None:
        """Sets the annotation layer name (typename) for this label."""
        self._typename = value


class DataPoint(ABC):
    """Abstract base class for all data points in Flair (e.g., Token, Sentence, Image).

    Defines core functionalities like holding embeddings, managing labels across
    different annotation layers, and providing basic positional/textual info.
    """

    def __init__(self) -> None:
        """Initializes a DataPoint with empty annotation/embedding/metadata storage."""
        self.annotation_layers: dict[str, list[Label]] = {}
        self._embeddings: dict[str, torch.Tensor] = {}
        self._metadata: dict[str, Any] = {}

    @property
    @abstractmethod
    def embedding(self) -> torch.Tensor:
        """Provides the primary embedding representation of the data point."""
        pass

    def set_embedding(self, name: str, vector: torch.Tensor):
        """Stores an embedding tensor under a given name.

        Args:
            name (str): The name to identify this embedding (e.g., "word", "flair").
            vector (torch.Tensor): The embedding tensor.
        """
        self._embeddings[name] = vector

    def get_embedding(self, names: Optional[list[str]] = None) -> torch.Tensor:
        """Retrieves embeddings, concatenating if multiple names are given or if names is None.

        Args:
            names (Optional[list[str]], optional): Specific embedding names to retrieve.
                If None, concatenates all stored embeddings sorted by name. Defaults to None.

        Returns:
            torch.Tensor: A single tensor representing the requested embedding(s).
                          Returns an empty tensor if no relevant embeddings are found.
        """
        # if one embedding name, directly return it
        if names and len(names) == 1:
            if names[0] in self._embeddings:
                return self._embeddings[names[0]].to(flair.device)
            else:
                return torch.tensor([], device=flair.device)

        # if multiple embedding names, concatenate them
        embeddings = self.get_each_embedding(names)
        if embeddings:
            return torch.cat(embeddings, dim=0)
        else:
            return torch.tensor([], device=flair.device)

    def get_each_embedding(self, embedding_names: Optional[list[str]] = None) -> list[torch.Tensor]:
        """Retrieves a list of individual embedding tensors.

        Args:
            embedding_names (Optional[list[str]], optional): If provided, filters by these names.
                Otherwise, returns all stored embeddings. Defaults to None.

        Returns:
            list[torch.Tensor]: List of embedding tensors, sorted by name.
        """
        embeddings = []
        for embed_name in sorted(self._embeddings.keys()):
            if embedding_names and embed_name not in embedding_names:
                continue
            embed = self._embeddings[embed_name].to(flair.device)
            embeddings.append(embed)
        return embeddings

    def to(self, device: Union[str, torch.device], pin_memory: bool = False) -> None:
        """Moves all stored embedding tensors to the specified device.

        Args:
            device (Union[str, torch.device]): Target device (e.g., 'cpu', 'cuda:0').
            pin_memory (bool, optional): If True and moving to CUDA, attempts to pin memory.
                Defaults to False.
        """
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(device, non_blocking=True).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking=True)

    def clear_embeddings(self, embedding_names: Optional[list[str]] = None) -> None:
        """Removes stored embeddings to free memory.

        Args:
            embedding_names (Optional[list[str]], optional): Specific names to remove.
                If None, removes all embeddings. Defaults to None.
        """
        if embedding_names is None:
            self._embeddings = {}
        else:
            for name in embedding_names:
                if name in self._embeddings:
                    del self._embeddings[name]

    def has_label(self, typename: str) -> bool:
        """Checks if the data point has at least one label for the given annotation type."""
        return typename in self.annotation_layers

    def add_metadata(self, key: str, value: Any) -> None:
        """Adds a key-value pair to the data point's metadata."""
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Retrieves metadata associated with the given key.

        Args:
            key (str): The metadata key.

        Returns:
            Any: The metadata value.

        Raises:
            KeyError: If the key is not found.
        """
        return self._metadata[key]

    def has_metadata(self, key: str) -> bool:
        """Checks if the data point has metadata for the given key."""
        return key in self._metadata

    def add_label(self, typename: str, value: str, score: float = 1.0, **metadata) -> "DataPoint":
        """Adds a new label to a specific annotation layer.

        Args:
            typename (str): Name of the annotation layer (e.g., "ner", "sentiment").
            value (str): String value of the label (e.g., "PERSON", "POSITIVE").
            score (float, optional): Confidence score (0.0-1.0). Defaults to 1.0.
            **metadata: Additional keyword arguments stored as metadata on the Label.

        Returns:
            DataPoint: Returns self for chaining.
        """
        label = Label(self, value, score, **metadata)
        label.typename = typename

        if typename not in self.annotation_layers:
            self.annotation_layers[typename] = [label]
        else:
            self.annotation_layers[typename].append(label)

        return self

    def set_label(self, typename: str, value: str, score: float = 1.0, **metadata) -> "DataPoint":
        """Sets the label(s) for an annotation layer, overwriting any existing ones.

        Args:
            typename (str): The name of the annotation layer.
            value (str): The string value of the new label.
            score (float, optional): Confidence score (0.0-1.0). Defaults to 1.0.
            **metadata: Additional keyword arguments for the new Label's metadata.

        Returns:
            DataPoint: Returns self for chaining.
        """
        label = Label(self, value, score, **metadata)
        label.typename = typename
        self.annotation_layers[typename] = [label]
        return self

    def remove_labels(self, typename: str) -> None:
        """Removes all labels associated with a specific annotation layer.

        Args:
            typename (str): The name of the annotation layer to clear.
        """
        if typename in self.annotation_layers:
            del self.annotation_layers[typename]

    def get_label(self, label_type: Optional[str] = None, zero_tag_value: str = "O") -> Label:
        """Retrieves the primary label for a given type, or a default 'O' label.

        Args:
            label_type (Optional[str], optional): The annotation layer name. Defaults to None (uses first overall label).
            zero_tag_value (str, optional): Value for the default label if none found. Defaults to "O".

        Returns:
            Label: The primary label, or a default label with score 0.0.
        """
        if len(self.get_labels(label_type)) == 0:
            return Label(self, zero_tag_value)
        return self.get_labels(label_type)[0]

    def get_labels(self, typename: Optional[str] = None) -> list[Label]:
        """Retrieves all labels for a specific annotation layer.

        Args:
            typename (Optional[str], optional): The layer name. If None, returns all labels
                from all layers. Defaults to None.

        Returns:
            list[Label]: List of Label objects, or empty list if none found.
        """
        if typename is None:
            return self.labels

        return self.annotation_layers.get(typename, [])

    @property
    def labels(self) -> list[Label]:
        """Returns a list of all labels from all annotation layers."""
        all_labels = []
        for key in self.annotation_layers:
            all_labels.extend(self.annotation_layers[key])
        return all_labels

    @property
    @abstractmethod
    def unlabeled_identifier(self) -> str:
        """A string identifier for the data point itself, without label info."""
        raise NotImplementedError

    def _printout_labels(self, main_label=None, add_score: bool = True, add_metadata: bool = True) -> str:
        """Internal helper to format labels attached *directly* to this DataPoint for string representation."""
        all_labels = []
        keys = [main_label] if main_label is not None else self.annotation_layers.keys()

        sep = "; " if add_score else "/"
        sent_sep = flair._arrow if add_score else "/"
        for key in keys:
            for label in self.get_labels(key):
                if label.data_point is not self:
                    continue
                value = label.value
                if add_metadata:
                    value = f"{value}{label.metadata_str}"
                if add_score:
                    value = f"{value} ({label.score:.04f})"
                all_labels.append(value)
        if not all_labels:
            return ""
        return sent_sep + sep.join(all_labels)

    def __str__(self) -> str:
        return self.unlabeled_identifier + self._printout_labels()

    @property
    @abstractmethod
    def start_position(self) -> int:
        """The starting character offset within the original text."""
        raise NotImplementedError

    @property
    @abstractmethod
    def end_position(self) -> int:
        """The ending character offset (exclusive) within the original text."""
        raise NotImplementedError

    @property
    @abstractmethod
    def text(self) -> str:
        """The textual representation of this data point."""
        raise NotImplementedError

    @property
    def tag(self) -> str:
        """Shortcut property for the value of the *first* label added."""
        return self.labels[0].value

    @property
    def score(self) -> float:
        """Shortcut property for the score of the *first* label added."""
        return self.labels[0].score

    def __lt__(self, other: "DataPoint") -> bool:
        """Compares data points based on start position for sorting."""
        return self.start_position < other.start_position

    def __len__(self) -> int:
        """Length of the data point (e.g., number of tokens for Sentence)."""
        raise NotImplementedError

    # Default implementation for simpler classes
    def _get_dynamic_embedding_names(self) -> set[str]:
        """
        Internal helper to find names of embeddings with requires_grad=True.
        Default implementation checks only direct embeddings. Subclasses override
        for recursive checks if needed.
        """
        return {name for name, vec in self._embeddings.items() if vec.requires_grad}

    # Default implementation for simpler classes
    def _get_all_embedding_names(self) -> set[str]:
        """
        Internal helper to find names of all embeddings.
        Default implementation checks only direct embeddings. Subclasses override
        for recursive checks if needed.
        """
        return set(self._embeddings.keys())


class EntityCandidate:
    """Represents a potential candidate entity from a knowledge base for entity linking."""

    def __init__(
        self,
        concept_id: str,
        concept_name: str,
        database_name: str,
        additional_ids: Optional[list[str]] = None,
        synonyms: Optional[list[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initializes an EntityCandidate.

        Args:
            concept_id (str): Primary identifier (e.g., "Q5").
            concept_name (str): Canonical name (e.g., "human").
            database_name (str): Source KB name (e.g., "Wikidata").
            additional_ids (Optional[list[str]], optional): Alternative IDs. Defaults to None.
            synonyms (Optional[list[str]], optional): List of synonyms. Defaults to None.
            description (Optional[str], optional): Textual description. Defaults to None.
        """
        self.concept_id = concept_id
        self.concept_name = concept_name
        self.database_name = database_name
        self.description = description
        if additional_ids is None:
            self.additional_ids = []
        else:
            self.additional_ids = additional_ids
        if synonyms is None:
            self.synonyms = []
        else:
            self.synonyms = synonyms

    def __str__(self) -> str:
        string = f"EntityLinkingCandidate: {self.database_name}:{self.concept_id} - {self.concept_name}"
        if self.additional_ids:
            string += f" - {'|'.join(self.additional_ids)}"
        return string

    def __repr__(self) -> str:
        return str(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "database_name": self.database_name,
            "additional_ids": self.additional_ids,
            "synonyms": self.synonyms,
            "description": self.description,
        }


DT = typing.TypeVar("DT", bound=DataPoint)
DT2 = typing.TypeVar("DT2", bound=DataPoint)
DT3 = typing.TypeVar("DT3", bound=DataPoint)


class _PartOfSentence(DataPoint, ABC):
    """Abstract base for data points within a Sentence (Token, Span, Relation).

    Ensures labels added to these parts are also registered in the parent Sentence.

    Attributes:
        sentence (Sentence): The parent Sentence object.
    """

    def __init__(self, sentence) -> None:
        """Initializes _PartOfSentence.

        Args:
            sentence: The parent sentence, can be None initially for Token.
        """
        super().__init__()
        self.sentence = sentence

    def add_label(self, typename: str, value: str, score: float = 1.0, **metadata) -> "_PartOfSentence":
        """Adds a label, propagating it to the parent Sentence's layer."""
        super().add_label(typename, value, score, **metadata)
        self.sentence.annotation_layers.setdefault(typename, []).append(Label(self, value, score, **metadata))
        return self

    def set_label(self, typename: str, value: str, score: float = 1.0, **metadata) -> "_PartOfSentence":
        """Sets a label (overwriting), propagating the change to the parent Sentence."""
        if len(self.annotation_layers.get(typename, [])) > 0:
            # First we remove any existing labels for this PartOfSentence in self.sentence
            self.sentence.annotation_layers[typename] = [
                label for label in self.sentence.annotation_layers.get(typename, []) if label.data_point != self
            ]
        self.sentence.annotation_layers.setdefault(typename, []).append(Label(self, value, score, **metadata))
        super().set_label(typename, value, score, **metadata)
        return self

    def remove_labels(self, typename: str) -> None:
        """Removes labels of a type, also removing them from the parent Sentence layer."""
        # labels also need to be deleted at Sentence object
        for label in self.get_labels(typename):
            self.sentence.annotation_layers[typename].remove(label)

        # delete labels at object itself
        super().remove_labels(typename)


class Token(_PartOfSentence):
    """Represents a single token (word, punctuation) within a Sentence.

    Attributes:
        form (str): The textual content of the token.
        idx (int): The 1-based index within the sentence (-1 if not attached).
        head_id (Optional[int]): 1-based index of the dependency head.
        whitespace_after (int): Number of spaces following this token.
        start_position (int): Character offset where this token begins.
        tags_proba_dist (dict[str, list[Label]]): Stores full probability distributions over tags.
    """

    def __init__(
        self,
        text: str,
        head_id: Optional[int] = None,
        whitespace_after: int = 1,
        start_position: int = 0,
        sentence: Optional["Sentence"] = None,
    ) -> None:
        """Initializes a Token.

        Args:
            text (str): The token text.
            head_id (Optional[int], optional): 1-based index of dependency head. Defaults to None.
            whitespace_after (int, optional): Spaces after token. Defaults to 1.
            start_position (int, optional): Character start offset. Defaults to 0.
            sentence (Optional[Sentence], optional): Parent sentence. Defaults to None.
        """
        super().__init__(sentence=sentence)

        self.form: str = text
        self._internal_index: Optional[int] = None
        self.head_id: Optional[int] = head_id
        self.whitespace_after: int = whitespace_after

        self._start_position = start_position

        self._embeddings: dict[str, torch.Tensor] = {}
        self.tags_proba_dist: dict[str, list[Label]] = {}

    @property
    def idx(self) -> int:
        """The 1-based index within the sentence (-1 if not attached)."""
        if self._internal_index is not None:
            return self._internal_index
        else:
            return -1

    @property
    def text(self) -> str:
        """The text content of the token."""
        return self.form

    @property
    def unlabeled_identifier(self) -> str:
        """String identifier: 'Token[<idx>]: "<text>"'."""
        return f'Token[{self.idx - 1}]: "{self.text}"'

    def add_tags_proba_dist(self, tag_type: str, tags: list[Label]) -> None:
        """Stores a list of Labels representing a probability distribution for a tag type.

        Args:
            tag_type (str): The annotation layer name (e.g., "pos").
            tags (list[Label]): List of Labels, each with a tag value and probability score.
        """
        self.tags_proba_dist[tag_type] = tags

    def get_tags_proba_dist(self, tag_type: str) -> list[Label]:
        """Retrieves the stored probability distribution for a given tag type.

        Args:
            tag_type (str): The annotation layer name.

        Returns:
            list[Label]: List of Labels representing the distribution,
                         or empty list if none stored.
        """
        if tag_type in self.tags_proba_dist:
            return self.tags_proba_dist[tag_type]
        return []

    def get_head(self) -> Optional["Token"]:
        """Returns the head Token in the dependency parse, if available."""
        if self.head_id is not None and self.sentence is not None:
            # Add assertion to satisfy mypy about head_id being an int here
            assert isinstance(self.head_id, int)
            return self.sentence.get_token(self.head_id)
        return None

    @property
    def start_position(self) -> int:
        """Character offset where the token begins in the Sentence text."""
        return self._start_position

    @start_position.setter
    def start_position(self, value: int) -> None:
        """Sets the character start offset."""
        self._start_position = value

    @property
    def end_position(self) -> int:
        """Character offset where the token ends (exclusive)."""
        return self.start_position + len(self.text)

    @property
    def embedding(self) -> torch.Tensor:
        """Returns the concatenated embeddings stored for this token."""
        return self.get_embedding()

    def __len__(self) -> int:
        """Length of a token is always 1."""
        return 1

    def __repr__(self) -> str:
        return self.__str__()

    def add_label(self, typename: str, value: str, score: float = 1.0, **metadata):
        # The Token is a special _PartOfSentence in that it may be initialized without a Sentence.
        # therefore, labels get added only to the Sentence if it exists
        if self.sentence:
            super().add_label(typename=typename, value=value, score=score, **metadata)
        else:
            DataPoint.add_label(self, typename=typename, value=value, score=score, **metadata)

    def set_label(self, typename: str, value: str, score: float = 1.0, **metadata):
        # The Token is a special _PartOfSentence in that it may be initialized without a Sentence.
        # Therefore, labels get set only to the Sentence if it exists
        if self.sentence:
            super().set_label(typename=typename, value=value, score=score, **metadata)
        else:
            DataPoint.set_label(self, typename=typename, value=value, score=score, **metadata)

    def to_dict(self, tag_type: Optional[str] = None) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_pos": self.start_position,
            "end_pos": self.end_position,
            "labels": [label.to_dict() for label in self.get_labels(tag_type)],
        }


class Span(_PartOfSentence):
    """Represents a contiguous sequence of Tokens within a Sentence.

    Used for entities, phrases, etc. Implements caching via __new__ within Sentence.

    Attributes:
        tokens (list[Token]): The list of tokens constituting the span.
    """

    def __new__(cls, tokens: list[Token]):
        """Creates a new Span or returns a cached instance from the Sentence.

        Args:
            tokens (list[Token]): Non-empty list of tokens from the *same* sentence.

        Returns:
            Span: The Span object.

        Raises:
            ValueError: If token list is empty or tokens are from different sentences/no sentence.
        """
        # check if the span already exists. If so, return it
        unlabeled_identifier = cls._make_unlabeled_identifier(tokens)
        if unlabeled_identifier in tokens[0].sentence._known_spans:
            span = tokens[0].sentence._known_spans[unlabeled_identifier]
            return span

        # else make a new span
        else:
            span = super().__new__(cls)
            span.initialized = False
            tokens[0].sentence._known_spans[unlabeled_identifier] = span
            return span

    def __init__(self, tokens: list[Token]) -> None:
        """Initializes the Span (called only once per unique span via __new__).

        Args:
            tokens (list[Token]): The list of tokens forming the span.
        """
        if not self.initialized:
            super().__init__(tokens[0].sentence)
            self.tokens = tokens
            self.initialized: bool = True

    @property
    def start_position(self) -> int:
        """Character offset where the span begins (start of the first token)."""
        return self.tokens[0].start_position

    @property
    def end_position(self) -> int:
        """Character offset where the span ends (end of the last token, exclusive)."""
        return self.tokens[-1].end_position

    @property
    def text(self) -> str:
        """The combined text of tokens in the span, respecting whitespace offsets."""
        return "".join([t.text + t.whitespace_after * " " for t in self.tokens]).strip()

    @staticmethod
    def _make_unlabeled_identifier(tokens: list[Token]):
        """Creates a unique identifier string based on token indices and text preview."""
        text = "".join([t.text + t.whitespace_after * " " for t in tokens]).strip()
        return f'Span[{tokens[0].idx - 1}:{tokens[-1].idx}]: "{text}"'

    @property
    def unlabeled_identifier(self) -> str:
        """String identifier: 'Span[<start_idx>:<end_idx>]: "<text_preview>"'."""
        return self._make_unlabeled_identifier(self.tokens)

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, idx: int) -> Token:
        """Accesses a token within the span by its relative index."""
        return self.tokens[idx]

    def __iter__(self):
        """Allows iteration over the tokens within the span."""
        return iter(self.tokens)

    def __len__(self) -> int:
        """Returns the number of tokens in the span."""
        return len(self.tokens)

    @property
    def embedding(self) -> torch.Tensor:
        """Returns embeddings stored *directly* on the Span object (if any)."""
        return self.get_embedding()

    def to_dict(self, tag_type: Optional[str] = None):
        return {
            "text": self.text,
            "start_pos": self.start_position,
            "end_pos": self.end_position,
            "labels": [label.to_dict() for label in self.get_labels(tag_type)],
        }

    # Keep overridden implementation for Span
    def _get_dynamic_embedding_names(self) -> set[str]:
        """Finds dynamic embedding names from the span itself and its tokens."""
        # Start with default implementation for self._embeddings
        names = super()._get_dynamic_embedding_names()
        for token in self.tokens:
            names.update(token._get_dynamic_embedding_names())  # Token uses default impl.
        return names

    # Keep overridden implementation for Span
    def _get_all_embedding_names(self) -> set[str]:
        """Finds all embedding names from the span itself and its tokens."""
        # Start with default implementation for self._embeddings
        names = super()._get_all_embedding_names()
        for token in self.tokens:
            names.update(token._get_all_embedding_names())  # Token uses default impl.
        return names


class Relation(_PartOfSentence):
    """Represents a directed relationship between two Spans in the same Sentence.

    Used for Relation Extraction. Caching via __new__ ensures uniqueness.

    Attributes:
        first (Span): The head Span of the relation.
        second (Span): The tail Span of the relation.
    """

    def __new__(cls, first: Span, second: Span) -> "Relation":
        """Creates a new Relation or returns a cached instance from the Sentence.

        Args:
            first (Span): The head span.
            second (Span): The tail span.

        Returns:
            Relation: The Relation object.

        Raises:
            ValueError: If the spans belong to different sentences.
        """
        # check if the relation already exists. If so, return it
        unlabeled_identifier = cls._make_unlabeled_identifier(first, second)
        if unlabeled_identifier in first.sentence._known_spans:
            span = first.sentence._known_spans[unlabeled_identifier]
            return span

        # else make a new relation
        else:
            span = super().__new__(cls)
            span.initialized = False
            first.sentence._known_spans[unlabeled_identifier] = span
            return span

    def __init__(self, first: Span, second: Span) -> None:
        """Initializes the Relation (called only once per unique relation via __new__).

        Args:
            first (Span): The head span.
            second (Span): The tail span.
        """
        if not self.initialized:
            super().__init__(sentence=first.sentence)
            self.first: Span = first
            self.second: Span = second
            self.initialized: bool = True

    def __repr__(self) -> str:
        return str(self)

    @property
    def tag(self):
        return self.labels[0].value

    @property
    def text(self) -> str:
        """A simple textual representation: '<head_text_preview> -> <tail_text_preview>'."""

        return f"{self.first.text} -> {self.second.text}"

    @staticmethod
    def _make_unlabeled_identifier(first: Span, second: Span) -> str:
        """Creates unique identifier based on span indices and text previews."""
        text = f"{first.text} -> {second.text}"
        return (
            f"Relation"
            f"[{first.tokens[0].idx - 1}:{first.tokens[-1].idx}]"
            f"[{second.tokens[0].idx - 1}:{second.tokens[-1].idx}]"
            f': "{text}"'
        )

    @property
    def unlabeled_identifier(self) -> str:
        """String identifier including span indices and text previews."""
        return self._make_unlabeled_identifier(self.first, self.second)

    @property
    def start_position(self) -> int:
        """Character offset of the earliest start position of the two spans."""
        return min(self.first.start_position, self.second.start_position)

    @property
    def end_position(self) -> int:
        """Character offset of the latest end position of the two spans."""
        return max(self.first.end_position, self.second.end_position)

    @property
    def embedding(self) -> torch.Tensor:
        """Placeholder for relation embedding (usually computed on the fly)."""
        return self.get_embedding()

    def to_dict(self, tag_type: Optional[str] = None):
        return {
            "from_text": self.first.text,
            "to_text": self.second.text,
            "from_idx": self.first.tokens[0].idx - 1,
            "to_idx": self.second.tokens[0].idx - 1,
            "labels": [label.to_dict() for label in self.get_labels(tag_type)],
        }


class Sentence(DataPoint):
    """A central data structure representing a sentence or text passage as Tokens.

    Holds text, tokens, labels (sentence/token/span/relation levels), embeddings,
    and document context information.

    Attributes:
        tokens (list[Token]): List of tokens (lazy tokenization if initialized with str).
        text (str): Original, untokenized text.
        language_code (Optional[str]): ISO 639-1 language code.
        start_position (int): Character offset in a larger document.
    """

    def __init__(
        self,
        text: Union[str, list[str], list[Token]],
        use_tokenizer: Union[bool, Tokenizer] = True,
        language_code: Optional[str] = None,
        start_position: int = 0,
    ) -> None:
        """Initializes a Sentence.

        Args:
            text: Either pass the text as a string, or provide an already tokenized text as either a list of strings or a list of :class:`Token` objects.
            use_tokenizer: You can optionally specify a custom tokenizer to split the text into tokens. By default we use
                :class:`flair.tokenization.SegtokTokenizer`. If `use_tokenizer` is set to False,
                :class:`flair.tokenization.SpaceTokenizer` will be used instead. The tokenizer will be ignored,
                if `text` refers to pretokenized tokens.
            language_code: Language of the sentence. If not provided, `langdetect <https://pypi.org/project/langdetect/>`_
                will be called when the language_code is accessed for the first time.
            start_position: Start char offset of the sentence in the superordinate document.
        """
        super().__init__()

        self._tokens: Optional[list[Token]] = None
        self._text: str = ""  # Change from Optional[str] to str with empty string default

        # track which tokenizer created the current tokens
        self._tokenizer_that_created_tokens: Optional[Tokenizer] = None

        # private field for all known spans with explicit typing
        self._known_spans: dict[str, Union[Span, Relation]] = {}

        self.language_code: Optional[str] = language_code

        self._start_position = start_position

        # the tokenizer used for this sentence
        if isinstance(use_tokenizer, Tokenizer):
            self._tokenizer = use_tokenizer
        elif isinstance(use_tokenizer, bool):
            self._tokenizer = SegtokTokenizer() if use_tokenizer else SpaceTokenizer()
        else:
            raise AssertionError("Unexpected type of parameter 'use_tokenizer'. Parameter should be bool or Tokenizer")

        self.tokenized: Optional[str] = None

        # some sentences represent a document boundary (but most do not)
        self.is_document_boundary: bool = False

        # internal variables to denote position inside dataset
        self._previous_sentence: Optional[Sentence] = None
        self._has_context: bool = False
        self._next_sentence: Optional[Sentence] = None
        self._position_in_dataset: Optional[tuple[Dataset, int]] = None

        # if list of strings or tokens is passed, create tokens directly
        if not isinstance(text, str):
            self._tokens = []

            # if list of tokens is passed, use them directly
            if all(isinstance(t, Token) for t in text):
                self._tokenizer_that_created_tokens = self._tokenizer
                for token in text:
                    self._add_token(token)

                # The last token of a sentence should not have trailing whitespace.
                if len(self._tokens) > 0:
                    self._tokens[-1].whitespace_after = 0

                # reconstruct text from tokens
                if len(self._tokens) > 0:
                    self._text = "".join([t.text + t.whitespace_after * " " for t in self._tokens]).strip()
                else:
                    self._text = ""

            # otherwise, create tokens from strings
            elif all(isinstance(t, str) for t in text):
                # if list of strings are passed, we use no tokenizer
                self._tokenizer = NoTokenizer()
                self._tokenizer_that_created_tokens = self._tokenizer

                str_list = cast(list[str], text)
                self._text = " ".join(str_list)
                current_position = 0
                for i, item in enumerate(str_list):
                    token = Token(text=item, start_position=current_position)
                    token.whitespace_after = 0 if i == len(str_list) - 1 else 1
                    self._add_token(token)
                    current_position += len(token.text) + token.whitespace_after

                if len(str_list) > 0:
                    self._tokens[-1].whitespace_after = 0

            elif len(text) > 0:
                raise TypeError("All elements must be either Token or str")

            else:
                self._text = ""

        else:
            self._text = Sentence._handle_problem_characters(text)

        # log a warning if the dataset is empty
        if self._text == "":
            log.warning("Warning: An empty Sentence was created! Are there empty strings in your dataset?")

    @property
    def tokens(self) -> list[Token]:
        """The list of Token objects. Triggers tokenization if needed (lazy evaluation)."""
        # If the sentence was never tokenized, just tokenize
        if self._tokens is None:
            self._tokenize()
            # After tokenizing, we need to set the tokenizer that created the tokens
            self._tokenizer_that_created_tokens = self._tokenizer

        # If the tokenizer was changed since last tokenization, perform re-tokenization
        elif self._tokenizer != self._tokenizer_that_created_tokens:
            log.debug(
                f"Retokenizing sentence with '{self._tokenizer.__class__.__name__}' (was '{self._tokenizer_that_created_tokens.__class__.__name__}'). "
                f"Attempting to preserve span/relation/sentence annotations. "
                f"Token-level annotations will be lost."
            )
            self._perform_retokenization_with_annotation_preservation(new_tokenizer=self._tokenizer)

        if self._tokens is None:
            raise ValueError("Tokens are None after tokenization - this indicates a bug in the tokenization process")
        return self._tokens

    def _perform_retokenization_with_annotation_preservation(self, new_tokenizer: Tokenizer) -> None:
        """
        Internal method to retokenize the sentence, attempting to preserve annotations.
        This method directly manipulates self._tokens and updates self._tokenizer_that_created_tokens.
        """
        # 1. Capture all annotations from the current tokenization
        annotation_data = self._capture_annotations()

        # 2. Clear all internal state of the sentence
        self._clear_internal_state()
        self._tokenizer_that_created_tokens = new_tokenizer

        # 3. Retokenize the sentence text using the new tokenizer
        self._tokenize()

        # 4. Re-apply the captured annotations to the new tokenization
        self._reapply_annotations(annotation_data)

    @property
    def tokenizer(self) -> Tokenizer:
        """Gets the tokenizer currently intended for this sentence."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, new_tokenizer: Tokenizer) -> None:
        """
        Sets the new intended tokenizer for this sentence.
        Retokenization is lazy and will occur the next time .tokens is accessed if the
        new_tokenizer is different from the one that last created the tokens.
        """
        if self._tokenizer is not new_tokenizer:
            log.debug(
                f"Sentence tokenizer intent changed from {self._tokenizer.__class__.__name__} to {new_tokenizer.__class__.__name__}. "
                f"Retokenization will be lazy."
            )
            self._tokenizer = new_tokenizer
            # Note: We DO NOT change self._tokenizer_that_created_tokens here.
            # That state reflects the past creation of self._tokens.
            # The .tokens property will compare self._tokenizer and self._tokenizer_that_created_tokens.

    def _tokenize(self) -> None:
        """Internal method to perform tokenization based on `self.text` and `self._tokenizer`."""

        # tokenize the text
        words = self._tokenizer.tokenize(self._text)

        # determine token positions and whitespace_after flag
        current_offset: int = 0
        previous_token: Optional[Token] = None
        self._tokens = []

        for word in words:
            word_start_position: int = self._text.index(word, current_offset)
            delta_offset: int = word_start_position - current_offset

            token: Token = Token(text=word, start_position=word_start_position)
            self._add_token(token)

            if previous_token is not None:
                previous_token.whitespace_after = delta_offset

            current_offset = token.end_position
            previous_token = token

        # the last token has no whitespace after
        if len(self._tokens) > 0:
            self._tokens[-1].whitespace_after = 0

    def __iter__(self):
        """Allows iteration over tokens. Triggers tokenization if not yet tokenized."""
        return iter(self.tokens)

    def __len__(self) -> int:
        """Returns the number of tokens in this sentence. Triggers tokenization if not yet tokenized."""
        return len(self.tokens)

    @property
    def unlabeled_identifier(self):
        # Try to get token count without forcing tokenization if _tokens exist
        num_tokens = len(self._tokens) if self._tokens is not None else "?"
        return f'Sentence[{num_tokens}]: "{self._text}"'

    @property
    def text(self) -> str:
        """Returns the original text of this sentence. Does not trigger tokenization."""
        return self._text

    def to_original_text(self) -> str:
        """Returns the original text of this sentence."""
        return self._text

    def to_tagged_string(self, main_label: Optional[str] = None) -> str:
        # For sentence-level labels, we don't need tokenization
        if not self._tokens:
            output = f'Sentence: "{self.text}"'
            if self.labels:
                output += self._printout_labels(main_label)
            return output

        # Only tokenize if we have token-level labels or spans to print
        already_printed = [self]
        output = super().__str__()

        label_append = []
        for label in self.get_labels(main_label):
            if label.data_point in already_printed:
                continue
            label_append.append(
                f'"{label.data_point.text}"{label.data_point._printout_labels(main_label=main_label, add_score=False)}'
            )
            already_printed.append(label.data_point)

        if len(label_append) > 0:
            output += f"{flair._arrow}[" + ", ".join(label_append) + "]"

        return output

    def get_relations(self, label_type: Optional[str] = None) -> list[Relation]:
        """Retrieves all Relation objects associated with this sentence."""
        relations: list[Relation] = []  # Explicitly type the list
        for label in self.get_labels(label_type):
            if isinstance(label.data_point, Relation):
                relations.append(label.data_point)
        return sorted(relations, key=lambda r: r.first.start_position)  # Ensure return matches hint

    def get_spans(self, label_type: Optional[str] = None) -> list[Span]:
        spans: list[Span] = []
        for potential_span in self._known_spans.values():
            if isinstance(potential_span, Span) and (label_type is None or potential_span.has_label(label_type)):
                spans.append(potential_span)
        return sorted(spans)

    def get_token(self, token_id: int) -> Optional[Token]:
        for token in self.tokens:
            if token.idx == token_id:
                return token
        return None

    def _add_token(self, token: Union[Token, str]):
        if isinstance(token, Token):
            # when initializing a sentence with tokens, they might be already part of a sentence
            if token.sentence is not None and token.sentence != self:
                raise ValueError("Cannot add a token that is already part of another sentence.")
            # setting sentence to None is required for the logic below
            token.sentence = None

        if isinstance(token, str):
            token = Token(token)
        token = cast(Token, token)

        # data with zero-width characters cannot be handled
        if token.text == "":
            return

        if self._tokens is None:
            self._tokens = []

        # set token idx and sentence
        token.sentence = self
        token._internal_index = len(self._tokens) + 1

        # if token has no start position, try to determine it from last token in sentence
        if token.start_position == 0 and len(self._tokens) > 0:
            last_token = self._tokens[-1]
            token.start_position = last_token.end_position + last_token.whitespace_after

        # append token to sentence
        self._tokens.append(token)

        # register token annotations on sentence
        for typename in token.annotation_layers:
            for label in token.get_labels(typename):
                if typename not in self.annotation_layers:
                    self.annotation_layers[typename] = [Label(token, label.value, label.score)]
                else:
                    self.annotation_layers[typename].append(Label(token, label.value, label.score))

    @property
    def embedding(self):
        return self.get_embedding()

    def to(self, device: Union[str, torch_device], pin_memory: bool = False):
        # move sentence embeddings to device
        super().to(device=device, pin_memory=pin_memory)

        # also move token embeddings to device
        for token in self:
            token.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: Optional[list[str]] = None):
        # clear sentence embeddings
        super().clear_embeddings(embedding_names)

        # clear token embeddings if sentence is tokenized
        if self._is_tokenized():
            if self._tokens is not None:
                for token in self._tokens:
                    token.clear_embeddings(embedding_names)

    def left_context(self, context_length: int, respect_document_boundaries: bool = True) -> list[Token]:
        sentence = self
        left_context: list[Token] = []
        while len(left_context) < context_length:
            sentence = sentence.previous_sentence()
            if sentence is None:
                break

            if respect_document_boundaries and sentence.is_document_boundary:
                break

            left_context = sentence.tokens + left_context
        return left_context[-context_length:]

    def right_context(self, context_length: int, respect_document_boundaries: bool = True) -> list[Token]:
        sentence = self
        right_context: list[Token] = []
        while len(right_context) < context_length:
            sentence = sentence.next_sentence()
            if sentence is None:
                break
            if respect_document_boundaries and sentence.is_document_boundary:
                break

            right_context += sentence.tokens
        return right_context[:context_length]

    def __str__(self) -> str:
        return self.to_tagged_string()

    def to_tokenized_string(self) -> str:
        if self.tokenized is None:
            self.tokenized = " ".join([t.text for t in self.tokens])

        return self.tokenized

    def to_plain_string(self) -> str:
        plain = ""
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after > 0:
                plain += token.whitespace_after * " "
        return plain.rstrip()

    def infer_space_after(self):
        """Heuristics in case you wish to infer whitespace_after values for tokenized text.

        This is useful for some old NLP tasks (such as CoNLL-03 and CoNLL-2000) that provide only tokenized data with
        no info of original whitespacing.
        :return:
        """
        last_token = None
        quote_count: int = 0
        # infer whitespace after field

        for token in self.tokens:
            if token.text == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token.whitespace_after = 0
                elif last_token is not None:
                    last_token.whitespace_after = 0

            if last_token is not None:
                if token.text in [".", ":", ",", ";", ")", "n't", "!", "?"]:
                    last_token.whitespace_after = 0

                if token.text.startswith("'"):
                    last_token.whitespace_after = 0

            if token.text in ["("]:
                token.whitespace_after = 0

            last_token = token
        return self

    def to_dict(self) -> dict[str, Any]:
        """
        Creates a dictionary representation of the Sentence.
        This dictionary can be used to recreate the sentence with from_dict().
        Returns:
            A dictionary containing the sentence's data and annotations.
        """
        # capture all annotations
        annotation_data = self._capture_annotations()

        # get token data
        tokens_data = []
        for token in self.tokens:
            tokens_data.append(
                {
                    "text": token.text,
                    "start_pos": token.start_position,
                    "whitespace_after": token.whitespace_after,
                    "head_id": token.head_id,
                }
            )

        return {
            "text": self.text,
            "tokens": tokens_data,
            "labels": self.get_labels(),  # legacy sentence-level labels
            "annotations": annotation_data,
            "language_code": self.language_code,
            "start_position": self.start_position,
            "tokenizer": self.tokenizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, sentence_dict: dict[str, Any]) -> "Sentence":
        """
        Creates a Sentence from a dictionary.
        Args:
            sentence_dict: A dictionary in the format produced by to_dict().
        Returns:
            The reconstructed Sentence object.
        """
        # Create tokenizer from its dictionary representation
        tokenizer_dict = sentence_dict.get("tokenizer")
        if tokenizer_dict:
            # Find the tokenizer class dynamically
            tokenizer_module = __import__(tokenizer_dict["class_module"], fromlist=[tokenizer_dict["class_name"]])
            tokenizer_class = getattr(tokenizer_module, tokenizer_dict["class_name"])
            tokenizer = tokenizer_class.from_dict(tokenizer_dict)
        else:
            # Fallback to default tokenizer if none is specified
            tokenizer = SegtokTokenizer()

        # Reconstruct tokens from token data
        tokens = []
        for token_data in sentence_dict.get("tokens", []):
            token = Token(
                text=token_data["text"],
                start_position=token_data["start_pos"],
                whitespace_after=token_data.get("whitespace_after", 1),
                head_id=token_data.get("head_id"),
            )
            tokens.append(token)

        # Initialize sentence with the list of reconstructed Token objects
        sentence = Sentence(
            tokens,  # Pass the list of tokens, not the text string
            use_tokenizer=tokenizer,
            language_code=sentence_dict.get("language_code"),
            start_position=sentence_dict.get("start_position", 0),
        )

        # Re-apply annotations
        if "annotations" in sentence_dict:
            sentence._reapply_annotations(sentence_dict["annotations"])
        # for legacy reasons, also support old "labels" field
        elif "labels" in sentence_dict:
            for label_data in sentence_dict.get("labels", []):
                # This part of the code is simplified since to_dict() now saves all labels under 'annotations'
                if isinstance(label_data, Label):
                    # The label's typename should be preserved if it exists.
                    # We default to "label" for legacy sentence-level labels.
                    typename = label_data.typename if label_data.typename else "label"
                    sentence.add_label(typename=typename, value=label_data.value, score=label_data.score)
                else:
                    # Handle dict-based label representation from very old formats
                    value = label_data.get("value")
                    # older formats used 'confidence' key for score
                    score = label_data.get("score", label_data.get("confidence", 1.0))
                    if value:
                        sentence.add_label(typename="label", value=value, score=score)

        return sentence

    def __deepcopy__(self, memo):
        """
        Custom deepcopy implementation to handle complex object graph with Spans and Relations.
        """
        # --- 1. Create the basic copy of the Sentence ---
        # First, create a new sentence with the same text and tokenizer.
        # This will create a fresh set of tokens.
        new_sentence = Sentence(
            self.text,
            use_tokenizer=self.tokenizer,
            language_code=self.language_code,
            start_position=self.start_position,
        )

        # The `memo` dict is crucial for deepcopy to handle cycles and avoid re-copying objects.
        # We must add our new sentence to it immediately.
        memo[id(self)] = new_sentence

        # --- 2. Copy all sentence-level labels ---
        for label in self.get_labels():
            if label.data_point is self:
                new_sentence.add_label(typename=label.typename, value=label.value, score=label.score, **label.metadata)

        # --- 3. Manually copy spans and build a mapping from old spans to new spans ---
        span_map = {}
        for original_span in self.get_spans():
            # Get the token indices from the original span
            token_indices = [t.idx - 1 for t in original_span.tokens]
            start, end = token_indices[0], token_indices[-1] + 1

            # Create the new span using the tokens from the new_sentence
            new_span = new_sentence[start:end]
            span_map[original_span] = new_span

            # Copy all labels from the original span to the new span
            for label in original_span.get_labels():
                new_span.add_label(typename=label.typename, value=label.value, score=label.score, **label.metadata)

        # --- 4. Manually copy relations using the span_map ---
        for original_relation in self.get_relations():
            # Find the new spans corresponding to the original relation's spans
            new_first = span_map.get(original_relation.first)
            new_second = span_map.get(original_relation.second)

            if new_first and new_second:
                # Create the new relation using the new spans
                new_relation = Relation(new_first, new_second)

                # Copy all labels
                for label in original_relation.get_labels():
                    new_relation.add_label(
                        typename=label.typename, value=label.value, score=label.score, **label.metadata
                    )

        # --- 5. Deepcopy all embeddings ---
        new_sentence._embeddings = copy.deepcopy(self._embeddings, memo)
        for original_token, new_token in zip(self.tokens, new_sentence.tokens):
            new_token._embeddings = copy.deepcopy(original_token._embeddings, memo)

        return new_sentence

    def get_span(self, start: int, stop: int) -> Span:
        span_slice = slice(start, stop)
        return self[span_slice]

    @typing.overload
    def __getitem__(self, idx: int) -> Token: ...

    @typing.overload
    def __getitem__(self, s: slice) -> Span: ...

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return Span(self.tokens[subscript])
        else:
            return self.tokens[subscript]

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def start_position(self) -> int:
        return self._start_position

    @start_position.setter
    def start_position(self, value: int) -> None:
        self._start_position = value

    @property
    def end_position(self) -> int:
        # The sentence's start position is not propagated to its tokens.
        # Therefore, we need to add the sentence's start position to its last token's end position, including whitespaces.
        return self.start_position + self[-1].end_position + self[-1].whitespace_after

    def get_language_code(self) -> str:
        if self.language_code is None:
            import langdetect

            try:
                self.language_code = langdetect.detect(self.to_plain_string())
            except Exception as e:
                log.debug(e)
                self.language_code = "en"

        return self.language_code

    @staticmethod
    def _handle_problem_characters(text: str) -> str:
        text = Sentence.__restore_windows_1252_characters(text)
        return text

    @staticmethod
    def __remove_zero_width_characters(text: str) -> str:
        text = text.replace("\u200c", "")
        text = text.replace("\u200b", "")
        text = text.replace("\ufe0f", "")
        text = text.replace("\ufeff", "")

        text = text.replace(
            "\u2028", ""
        )  # LINE SEPARATOR & PARAGRAPH SEPARATOR are usually used for wrapping & displaying texts,
        text = text.replace("\u2029", "")  # but not for semantic meaning -> ignore them.
        return text

    @staticmethod
    def __restore_windows_1252_characters(text: str) -> str:
        def to_windows_1252(match):
            try:
                return bytes([ord(match.group(0))]).decode("windows-1252")
            except UnicodeDecodeError:
                # No character at the corresponding code point: remove it
                return ""

        return re.sub(r"[\u0080-\u0099]", to_windows_1252, text)

    def next_sentence(self):
        """Get the next sentence in the document.

        This only works if context is set through dataloader or elsewhere
        :return: next Sentence in document if set, otherwise None
        """
        if self._next_sentence is not None:
            return self._next_sentence

        if self._position_in_dataset is not None:
            dataset = self._position_in_dataset[0]
            index = self._position_in_dataset[1] + 1
            if index < len(dataset):
                return dataset[index]

        return None

    def previous_sentence(self):
        """Get the previous sentence in the document.

        works only if context is set through dataloader or elsewhere
        :return: previous Sentence in document if set, otherwise None
        """
        if self._previous_sentence is not None:
            return self._previous_sentence

        if self._position_in_dataset is not None:
            dataset = self._position_in_dataset[0]
            index = self._position_in_dataset[1] - 1
            if index >= 0:
                return dataset[index]

        return None

    def is_context_set(self) -> bool:
        """Determines if this sentence has a context of sentences before or after set.

        Return True or False depending on whether context is set (for instance in dataloader or elsewhere)
        :return: True if context is set, else False
        """
        return (
            self._has_context
            or self._previous_sentence is not None
            or self._next_sentence is not None
            or self._position_in_dataset is not None
        )

    def copy_context_from_sentence(self, sentence: "Sentence") -> None:
        self._previous_sentence = sentence._previous_sentence
        self._next_sentence = sentence._next_sentence
        self._position_in_dataset = sentence._position_in_dataset

    @classmethod
    def set_context_for_sentences(cls, sentences: list["Sentence"]) -> None:
        previous_sentence = None
        for sentence in sentences:
            if sentence.is_context_set():
                continue
            sentence._previous_sentence = previous_sentence
            sentence._next_sentence = None
            sentence._has_context = True
            if previous_sentence is not None:
                previous_sentence._next_sentence = sentence
            previous_sentence = sentence

    def get_labels(self, label_type: Optional[str] = None):
        # if no label if specified, return all labels
        if label_type is None:
            return sorted(self.labels)

        # if the label type exists in the Sentence, return it
        if label_type in self.annotation_layers:
            return sorted(self.annotation_layers[label_type])

        # return empty list if none of the above
        return []

    def remove_labels(self, typename: str):
        # only access tokens if already tokenized
        if self._is_tokenized():
            # labels also need to be deleted at all tokens
            if self._tokens is not None:
                for token in self._tokens:
                    token.remove_labels(typename)

            # labels also need to be deleted at all known spans
            for span in self._known_spans.values():
                span.remove_labels(typename)

        # delete labels at object itself first
        super().remove_labels(typename)

    def _is_tokenized(self) -> bool:
        return self._tokens is not None

    def truncate(self, max_tokens: int) -> None:
        """Truncates the sentence to `max_tokens`, cleaning up associated annotations."""
        if len(self.tokens) <= max_tokens:
            return

        # Truncate tokens
        self._tokens = self.tokens[:max_tokens]

        # Remove spans that reference removed tokens
        self._known_spans = {
            identifier: span
            for identifier, span in self._known_spans.items()
            if isinstance(span, Span) and all(token.idx <= max_tokens for token in span.tokens)
        }

        # Remove relations that reference removed spans
        self._known_spans = {
            identifier: relation
            for identifier, relation in self._known_spans.items()
            if not isinstance(relation, Relation)
            or (
                all(token.idx <= max_tokens for token in relation.first.tokens)
                and all(token.idx <= max_tokens for token in relation.second.tokens)
            )
        }

        # Clean up any labels that reference removed spans/relations
        for typename in list(self.annotation_layers.keys()):
            self.annotation_layers[typename] = [
                label
                for label in self.annotation_layers[typename]
                if (
                    not isinstance(label.data_point, (Span, Relation))
                    or label.data_point.unlabeled_identifier in self._known_spans
                )
            ]

    def _clear_internal_state(self) -> None:
        """
        Resets the internal tokenization and annotation state of the sentence.
        Used before operations like retokenization that rebuild the sentence structure.
        """
        # Clear the central annotation registry
        self.annotation_layers: dict[str, list[Label]] = {}
        # Clear token list
        self._tokens = []
        # Clear known spans/relations cache
        self._known_spans = {}
        # Reset cached tokenized string representation
        self.tokenized = None

    def _capture_annotations(self) -> dict[str, Any]:
        """
        Captures all annotations (sentence, span, relation labels) in a serializable format.
        This is a non-destructive, read-only operation.
        Returns:
            A dictionary containing the structured annotation data.
        """
        # 1. Capture sentence-level labels
        sentence_labels = []
        for typename, label_list in self.annotation_layers.items():
            for label in label_list:
                if label.data_point is self:
                    sentence_labels.append(
                        {
                            "typename": typename,
                            "value": label.value,
                            "score": label.score,
                            "metadata": label.metadata,
                        }
                    )

        # 2. Capture spans and relations, using character offsets for robustness
        spans = {}
        relations = []
        if self._known_spans:
            for dp in self._known_spans.values():
                if isinstance(dp, Span):
                    span_id = dp.unlabeled_identifier
                    span_labels = []
                    for label in dp.get_labels():
                        if label.data_point is dp:
                            span_labels.append(
                                {
                                    "typename": label.typename,
                                    "value": label.value,
                                    "score": label.score,
                                    "metadata": label.metadata,
                                }
                            )

                    spans[span_id] = {
                        "start_char": dp.start_position - self.start_position,
                        "end_char": dp.end_position - self.start_position,
                        "text": dp.text,
                        "labels": span_labels,
                    }

                elif isinstance(dp, Relation):
                    relation_labels = []
                    for label in dp.get_labels():
                        if label.data_point is dp:
                            relation_labels.append(
                                {
                                    "typename": label.typename,
                                    "value": label.value,
                                    "score": label.score,
                                    "metadata": label.metadata,
                                }
                            )
                    relations.append(
                        {
                            "first": dp.first.unlabeled_identifier,
                            "second": dp.second.unlabeled_identifier,
                            "labels": relation_labels,
                        }
                    )

        return {
            "sentence_labels": sentence_labels,
            "spans": spans,
            "relations": relations,
        }

    def _reapply_annotations(self, annotation_data: dict[str, Any]) -> None:
        """
        Applies a dictionary of annotations to the current sentence.
        This is used by both retokenization and deserialization.
        Args:
            annotation_data: A dictionary in the format produced by _capture_annotations.
        """
        # 1. Re-apply sentence-level labels
        for label_info in annotation_data.get("sentence_labels", []):
            self.add_label(
                typename=label_info["typename"],
                value=label_info["value"],
                score=label_info["score"],
                **label_info.get("metadata", {}),
            )

        # 2. Re-create spans and their labels
        reconstructed_span_map: dict[str, Span] = {}
        for span_id, span_info in annotation_data.get("spans", {}).items():
            start_char = span_info["start_char"]
            end_char = span_info["end_char"]

            target_tokens: list[Token] = []
            if self._tokens is not None:
                for token in self._tokens:
                    # FIX: Make token positions relative to sentence for comparison
                    token_start_in_sentence = token.start_position - self.start_position
                    token_end_in_sentence = token.end_position - self.start_position

                    # Check for overlap
                    if max(start_char, token_start_in_sentence) < min(end_char, token_end_in_sentence):
                        target_tokens.append(token)

            if target_tokens:
                new_span = Span(target_tokens)
                for label_info in span_info.get("labels", []):
                    new_span.add_label(
                        typename=label_info["typename"],
                        value=label_info["value"],
                        score=label_info["score"],
                        **label_info.get("metadata", {}),
                    )
                reconstructed_span_map[span_id] = new_span
            else:
                log.warning(
                    f"Could not map original span '{span_info.get('text', span_id)}' "
                    f"(orig chars {start_char}-{end_char}) to tokens."
                )

        # 3. Re-create relations and their labels
        for rel_info in annotation_data.get("relations", []):
            first_span = reconstructed_span_map.get(rel_info["first"])
            second_span = reconstructed_span_map.get(rel_info["second"])

            if first_span and second_span:
                new_relation = Relation(first_span, second_span)
                for label_info in rel_info.get("labels", []):
                    new_relation.add_label(
                        typename=label_info["typename"],
                        value=label_info["value"],
                        score=label_info["score"],
                        **label_info.get("metadata", {}),
                    )
            else:
                log.warning(
                    f"Could not reconstruct relation between original spans '{rel_info['first']}' and "
                    f"'{rel_info['second']}' due to missing span(s)."
                )

    def retokenize(self, new_tokenizer: Tokenizer) -> None:
        """
        Eagerly retokenizes the sentence using the provided tokenizer.
        This attempts to preserve span, relation, and sentence labels.
        Token-level labels are generally discarded as their basis (the tokens themselves) changes.

        Args:
            new_tokenizer: The tokenizer to use for retokenization.
        """
        log.debug(f"Eager retokenize called with '{new_tokenizer.__class__.__name__}'.")
        self._tokenizer = new_tokenizer  # Set the new intent
        # Immediately perform the retokenization.
        # The .tokens property would also do this on next access, but this forces it now.
        self._perform_retokenization_with_annotation_preservation(new_tokenizer)

    # Keep overridden implementation for Sentence
    def _get_dynamic_embedding_names(self) -> set[str]:
        """Finds dynamic embedding names from the sentence itself and its tokens."""
        # Start with default implementation for self._embeddings
        names = super()._get_dynamic_embedding_names()
        # Check embeddings of constituent tokens as well
        if self._is_tokenized():  # Avoid tokenizing if not already done
            for token in self.tokens:
                names.update(token._get_dynamic_embedding_names())  # Token uses default impl.
        return names

    # Keep overridden implementation for Sentence
    def _get_all_embedding_names(self) -> set[str]:
        """Finds all embedding names from the sentence itself and its tokens."""
        # Start with default implementation for self._embeddings
        names = super()._get_all_embedding_names()
        if self._is_tokenized():  # Avoid tokenizing if not already done
            for token in self.tokens:
                names.update(token._get_all_embedding_names())  # Token uses default impl.
        return names


class DataPair(DataPoint, typing.Generic[DT, DT2]):
    """Represents a pair of DataPoints, often used for sentence-pair tasks."""

    def __init__(self, first: DT, second: DT2) -> None:
        """Initializes a DataPair.

        Args:
            first (DT): The first data point.
            second (DT2): The second data point.
        """
        super().__init__()
        self.first = first
        self.second = second
        self.concatenated_data: Optional[Union[DT, DT2]] = None

    def to(self, device: Union[str, torch_device], pin_memory: bool = False):
        self.first.to(device, pin_memory)
        self.second.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: Optional[list[str]] = None):
        self.first.clear_embeddings(embedding_names)
        self.second.clear_embeddings(embedding_names)
        if self.concatenated_data is not None:
            self.concatenated_data.clear_embeddings(embedding_names)

    @property
    def embedding(self):
        return torch.cat([self.first.embedding, self.second.embedding])

    def __len__(self) -> int:
        return len(self.first) + len(self.second)

    @property
    def unlabeled_identifier(self):
        return f"DataPair: '{self.first.unlabeled_identifier}' + '{self.second.unlabeled_identifier}'"

    @property
    def start_position(self) -> int:
        return self.first.start_position

    @property
    def end_position(self) -> int:
        return self.first.end_position

    @property
    def text(self):
        return self.first.text + " || " + self.second.text

    # Keep overridden implementation for DataPair
    def _get_dynamic_embedding_names(self) -> set[str]:
        """Finds dynamic embedding names from the pair itself and its components."""
        # Start with default implementation for self._embeddings
        names = super()._get_dynamic_embedding_names()
        names.update(self.first._get_dynamic_embedding_names())  # Recursive call
        names.update(self.second._get_dynamic_embedding_names())  # Recursive call
        return names

    # Keep overridden implementation for DataPair
    def _get_all_embedding_names(self) -> set[str]:
        """Finds all embedding names from the pair itself and its components."""
        # Start with default implementation for self._embeddings
        names = super()._get_all_embedding_names()
        names.update(self.first._get_all_embedding_names())  # Recursive call
        names.update(self.second._get_all_embedding_names())  # Recursive call
        return names


TextPair = DataPair[Sentence, Sentence]
"""Type alias for a DataPair consisting of two Sentences."""


class DataTriple(DataPoint, typing.Generic[DT, DT2, DT3]):
    """Represents a triplet of DataPoints."""

    def __init__(self, first: DT, second: DT2, third: DT3) -> None:
        """Initializes a DataTriple.

        Args:
            first (DT): The first data point.
            second (DT2): The second data point.
            third (DT3): The third data point.
        """
        super().__init__()
        self.first = first
        self.second = second
        self.third = third

    def to(self, device: Union[str, torch_device], pin_memory: bool = False):
        self.first.to(device, pin_memory)
        self.second.to(device, pin_memory)
        self.third.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: Optional[list[str]] = None):
        self.first.clear_embeddings(embedding_names)
        self.second.clear_embeddings(embedding_names)
        self.third.clear_embeddings(embedding_names)

    @property
    def embedding(self):
        return torch.cat([self.first.embedding, self.second.embedding, self.third.embedding])

    def __len__(self):
        return len(self.first) + len(self.second) + len(self.third)

    @property
    def unlabeled_identifier(self):
        return f"DataTriple: '{self.first.unlabeled_identifier}' + '{self.second.unlabeled_identifier}' + '{self.third.unlabeled_identifier}'"

    @property
    def start_position(self) -> int:
        return self.first.start_position

    @property
    def end_position(self) -> int:
        return self.first.end_position

    @property
    def text(self):
        return self.first.text + " || " + self.second.text + "||" + self.third.text

    # Keep overridden implementation for DataTriple
    def _get_dynamic_embedding_names(self) -> set[str]:
        """Finds dynamic embedding names from the triple itself and its components."""
        # Start with default implementation for self._embeddings
        names = super()._get_dynamic_embedding_names()
        names.update(self.first._get_dynamic_embedding_names())  # Recursive call
        names.update(self.second._get_dynamic_embedding_names())  # Recursive call
        names.update(self.third._get_dynamic_embedding_names())  # Recursive call
        return names

    # Keep overridden implementation for DataTriple
    def _get_all_embedding_names(self) -> set[str]:
        """Finds all embedding names from the triple itself and its components."""
        # Start with default implementation for self._embeddings
        names = super()._get_all_embedding_names()
        names.update(self.first._get_all_embedding_names())  # Recursive call
        names.update(self.second._get_all_embedding_names())  # Recursive call
        names.update(self.third._get_all_embedding_names())  # Recursive call
        return names


TextTriple = DataTriple[Sentence, Sentence, Sentence]
"""Type alias for a DataTriple consisting of three Sentences."""


class Image(DataPoint):
    """Represents an image as a data point, holding image data or a URL."""

    def __init__(self, data: Any = None, imageURL: Optional[str] = None) -> None:
        """Initializes an Image data point.

        Args:
            data (Any, optional): Raw image data (e.g., torch.Tensor, PIL Image). Defaults to None.
            imageURL (Optional[str], optional): URL of the image. Defaults to None.
        """
        super().__init__()

        self.data = data
        self._embeddings: dict[str, torch.Tensor] = {}
        self.imageURL = imageURL

    @property
    def embedding(self) -> torch.Tensor:
        """Returns the concatenated embeddings stored for this image."""
        return self.get_embedding()

    def __str__(self) -> str:
        image_repr = self.data.size() if self.data else ""
        image_url = self.imageURL if self.imageURL else ""

        return f"Image: {image_repr} {image_url}"

    @property
    def start_position(self) -> int:
        raise NotImplementedError

    @property
    def end_position(self) -> int:
        raise NotImplementedError

    @property
    def text(self) -> str:
        raise NotImplementedError

    @property
    def unlabeled_identifier(self) -> str:
        raise NotImplementedError


class Corpus(typing.Generic[T_co]):
    """The main container for holding train, dev, and test datasets for a task.

    A corpus consists of three splits: A `train` split used for training, a `dev` split used for model selection
    or early stopping and a `test` split used for testing. All three splits are optional, so it is possible
    to create a corpus only using one or two splits. If the option `sample_missing_splits` is set to True,
    missing splits will be randomly sampled from the training split.
    Provides methods for sampling, filtering, and creating dictionaries.

    Generics:
        T_co: The covariant type of DataPoint in the datasets (e.g., Sentence).

    Attributes:
        train (Optional[Dataset[T_co]]): Training data split.
        dev (Optional[Dataset[T_co]]): Development (validation) data split.
        test (Optional[Dataset[T_co]]): Testing data split.
        name (str): Name of the corpus.
    """

    def __init__(
        self,
        train: Optional[Dataset[T_co]] = None,
        dev: Optional[Dataset[T_co]] = None,
        test: Optional[Dataset[T_co]] = None,
        name: str = "corpus",
        sample_missing_splits: Union[bool, str] = True,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initializes a Corpus, potentially sampling missing dev/test splits from train.

        You can define the train, dev and test split
        by passing the corresponding Dataset object to the constructor. At least one split should be defined.
        If the option `sample_missing_splits` is set to True, missing splits will be randomly sampled from the
        train split.
        In most cases, you will not use the constructor yourself. Rather, you will create a corpus using one of our
        helper methods that read common NLP filetypes. For instance, you can use
        :class:`flair.datasets.sequence_labeling.ColumnCorpus` to read CoNLL-formatted files directly into
        a :class:`Corpus`.

        Args:
            train (Optional[Dataset[T_co]], optional): Training data. Defaults to None.
            dev (Optional[Dataset[T_co]], optional): Development data. Defaults to None.
            test (Optional[Dataset[T_co]], optional): Testing data. Defaults to None.
            name (str, optional): Corpus name. Defaults to "corpus".
            sample_missing_splits (Union[bool, str], optional): Policy for handling missing splits.
                True (default): sample dev(10%)/test(10%) from train. False: keep None.
                "only_dev": sample only dev. "only_test": sample only test.
            random_seed (Optional[int], optional): Seed for reproducible sampling. Defaults to None.
        """
        # set name
        self.name: str = name

        # abort if no data is provided
        if not train and not dev and not test:
            raise RuntimeError("No data provided when initializing corpus object.")

        # sample test data from train if none is provided
        if test is None and sample_missing_splits and train and sample_missing_splits != "only_dev":
            test_portion = 0.1
            train_length = _len_dataset(train)
            test_size: int = round(train_length * test_portion)
            test, train = randomly_split_into_two_datasets(train, test_size, random_seed)
            log.warning(
                "No test split found. Using %.0f%% (i.e. %d samples) of the train split as test data",
                test_portion * 100,
                test_size,
            )

        # sample dev data from train if none is provided
        if dev is None and sample_missing_splits and train and sample_missing_splits != "only_test":
            dev_portion = 0.1
            train_length = _len_dataset(train)
            dev_size: int = round(train_length * dev_portion)
            dev, train = randomly_split_into_two_datasets(train, dev_size, random_seed)
            log.warning(
                "No dev split found. Using %.0f%% (i.e. %d samples) of the train split as dev data",
                dev_portion * 100,
                dev_size,
            )

        # set train dev and test data
        self._train: Optional[Dataset[T_co]] = train
        self._test: Optional[Dataset[T_co]] = test
        self._dev: Optional[Dataset[T_co]] = dev

        # --- Add attribute to store tokenizer (will be set by subclasses) ---
        self.tokenizer: Optional[Tokenizer] = None

    @property
    def train(self) -> Optional[Dataset[T_co]]:
        """The training split as a :class:`torch.utils.data.Dataset` object."""
        return self._train

    @property
    def dev(self) -> Optional[Dataset[T_co]]:
        """The dev split as a :class:`torch.utils.data.Dataset` object."""
        return self._dev

    @property
    def test(self) -> Optional[Dataset[T_co]]:
        """The test split as a :class:`torch.utils.data.Dataset` object."""
        return self._test

    @property
    def corpus_tokenizer(self) -> Optional[Tokenizer]:
        """
        Returns the custom tokenizer provided during corpus initialization for retokenization, if any.
        Returns None if no custom retokenizer was specified.
        """
        # The tokenizer attribute is set by subclasses like ColumnCorpus during their init
        return self.tokenizer

    def downsample(
        self,
        percentage: float = 0.1,
        downsample_train: bool = True,
        downsample_dev: bool = True,
        downsample_test: bool = True,
        random_seed: Optional[int] = None,
    ) -> "Corpus":
        """Randomly downsample the corpus to the given percentage (by removing data points).

        This method is an in-place operation, meaning that the Corpus object itself is modified by removing
        data points. It additionally returns a pointer to itself for use in method chaining.

        Args:
            percentage: A float value between 0. and 1. that indicates to which percentage the corpus
                should be downsampled. Default value is 0.1, meaning it gets downsampled to 10%.
            downsample_train: Whether or not to include the training split in downsampling. Default is True.
            downsample_dev: Whether or not to include the dev split in downsampling. Default is True.
            downsample_test: Whether or not to include the test split in downsampling. Default is True.
            random_seed: An optional random seed to make downsampling reproducible.

        Returns:
            Corpus: Returns self for chaining.
        """
        if downsample_train and self._train is not None:
            self._train = self._downsample_to_proportion(self._train, percentage, random_seed)

        if downsample_dev and self._dev is not None:
            self._dev = self._downsample_to_proportion(self._dev, percentage, random_seed)

        if downsample_test and self._test is not None:
            self._test = self._downsample_to_proportion(self._test, percentage, random_seed)

        return self

    def filter_empty_sentences(self):
        """A method that filters all sentences consisting of 0 tokens.

        This is an in-place operation that directly modifies the Corpus object itself by removing these sentences.
        """
        log.info("Filtering empty sentences")
        if self._train is not None:
            self._train = Corpus._filter_empty_sentences(self._train)
        if self._test is not None:
            self._test = Corpus._filter_empty_sentences(self._test)
        if self._dev is not None:
            self._dev = Corpus._filter_empty_sentences(self._dev)
        log.info(self)

    def filter_long_sentences(self, max_charlength: int):
        """A method that filters all sentences for which the plain text is longer than a specified number of characters.

        This is an in-place operation that directly modifies the Corpus object itself by removing these sentences.

        Args:
            max_charlength (int): Maximum allowed character length.
        """
        log.info("Filtering long sentences")
        if self._train is not None:
            self._train = Corpus._filter_long_sentences(self._train, max_charlength)
        if self._test is not None:
            self._test = Corpus._filter_long_sentences(self._test, max_charlength)
        if self._dev is not None:
            self._dev = Corpus._filter_long_sentences(self._dev, max_charlength)
        log.info(self)

    @staticmethod
    def _filter_long_sentences(dataset: Dataset, max_charlength: int) -> Union[Dataset, Subset]:
        """Internal helper to filter sentences exceeding a character length."""
        # find out empty sentence indices
        empty_sentence_indices = []
        non_empty_sentence_indices = []

        for index, sentence in Tqdm.tqdm(enumerate(_iter_dataset(dataset))):
            if len(sentence.to_plain_string()) > max_charlength:
                empty_sentence_indices.append(index)
            else:
                non_empty_sentence_indices.append(index)

        # create subset of non-empty sentence indices
        subset = Subset(dataset, non_empty_sentence_indices)

        return subset

    @staticmethod
    def _filter_empty_sentences(dataset) -> Dataset:
        # find out empty sentence indices
        empty_sentence_indices = []
        non_empty_sentence_indices = []

        for index, sentence in enumerate(_iter_dataset(dataset)):
            if len(sentence) == 0:
                empty_sentence_indices.append(index)
            else:
                non_empty_sentence_indices.append(index)

        # create subset of non-empty sentence indices
        subset = Subset(dataset, non_empty_sentence_indices)

        return subset

    def make_vocab_dictionary(self, max_tokens: int = -1, min_freq: int = 1) -> Dictionary:
        """Creates a :class:`Dictionary` of all tokens contained in the corpus.

        By defining `max_tokens` you can set the maximum number of tokens that should be contained in the dictionary.
        If there are more than `max_tokens` tokens in the corpus, the most frequent tokens are added first.
        If `min_freq` is set to a value greater than 1 only tokens occurring more than `min_freq` times are considered
        to be added to the dictionary.

        Args:
            max_tokens: The maximum number of tokens that should be added to the dictionary (providing a value of "-1"
                means that there is no maximum in this regard).
            min_freq: A token needs to occur at least `min_freq` times to be added to the dictionary (providing a value
                of "-1" means that there is no limitation in this regard).

        Returns:
            Dictionary: Vocabulary Dictionary mapping tokens to IDs (includes <unk>).
        """
        tokens = self._get_most_common_tokens(max_tokens, min_freq)

        vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            vocab_dictionary.add_item(token)

        return vocab_dictionary

    def _get_most_common_tokens(self, max_tokens: int, min_freq: int) -> list[str]:
        """Helper to get frequent tokens from the training split."""
        tokens_and_frequencies = Counter(self._get_all_tokens())

        tokens: list[str] = []
        for token, freq in tokens_and_frequencies.most_common():
            if (min_freq != -1 and freq < min_freq) or (max_tokens != -1 and len(tokens) == max_tokens):
                break
            tokens.append(token)
        return tokens

    def _get_all_tokens(self) -> list[str]:
        """Helper to extract all token texts from the training split."""
        assert self.train
        tokens = [s.tokens for s in _iter_dataset(self.train)]
        tokens = [token for sublist in tokens for token in sublist]
        return [t.text for t in tokens]

    @staticmethod
    def _downsample_to_proportion(dataset: Dataset, proportion: float, random_seed: Optional[int] = None) -> Subset:
        """Internal helper to create a Subset representing a proportion."""
        sampled_size: int = round(_len_dataset(dataset) * proportion)
        splits = randomly_split_into_two_datasets(dataset, sampled_size, random_seed=random_seed)
        return splits[0]

    def obtain_statistics(self, label_type: Optional[str] = None, pretty_print: bool = True) -> Union[dict, str]:
        """Print statistics about the corpus, including the length of the sentences and the labels in the corpus.

        Args:
            label_type: Optionally set this value to obtain statistics only for one specific type of label (such
                as "ner" or "pos"). If not set, statistics for all labels will be returned.
            pretty_print: If set to True, returns pretty json (indented for readabilty). If not, the json is
                returned as a single line. Default: True.

        Returns:
            If pretty_print is True, returns a pretty print formatted string in json format. Otherwise, returns a
                dictionary holding a json.
        """
        json_data = {
            "TRAIN": self._obtain_statistics_for(self.train, "TRAIN", label_type),
            "TEST": self._obtain_statistics_for(self.test, "TEST", label_type),
            "DEV": self._obtain_statistics_for(self.dev, "DEV", label_type),
        }
        if pretty_print:
            import json

            return json.dumps(json_data, indent=4)
        return json_data

    @staticmethod
    def _obtain_statistics_for(dataset, name: str, tag_type: Optional[str]) -> dict:
        """Helper to compute statistics for a single dataset split."""
        if len(dataset) == 0:
            return {}

        classes_to_count = Corpus._count_sentence_labels(dataset)
        tags_to_count = Corpus._count_token_labels(dataset, tag_type)
        tokens_per_sentence = Corpus._get_tokens_per_sentence(dataset)

        label_size_dict = dict(classes_to_count)
        tag_size_dict = dict(tags_to_count)

        return {
            "dataset": name,
            "total_number_of_documents": len(dataset),
            "number_of_documents_per_class": label_size_dict,
            "number_of_tokens_per_tag": tag_size_dict,
            "number_of_tokens": {
                "total": sum(tokens_per_sentence),
                "min": min(tokens_per_sentence),
                "max": max(tokens_per_sentence),
                "avg": sum(tokens_per_sentence) / len(dataset),
            },
        }

    @staticmethod
    def _get_tokens_per_sentence(sentences: Iterable[Sentence]) -> list[int]:
        """Helper to get list of token counts per sentence."""
        return [len(x.tokens) for x in sentences]

    @staticmethod
    def _count_sentence_labels(sentences: Iterable[Sentence]) -> defaultdict[str, int]:
        """Helper to count sentence-level labels."""
        label_count: defaultdict[str, int] = defaultdict(lambda: 0)
        for sent in sentences:
            for label in sent.labels:
                label_count[label.value] += 1
        return label_count

    @staticmethod
    def _count_token_labels(sentences: Iterable[Sentence], label_type: Optional[str]) -> defaultdict[str, int]:
        """Helper to count token-level labels for a specific type.

        Args:
            sentences: The sentences to count labels from
            label_type: The type of label to count, if None returns empty defaultdict

        Returns:
            defaultdict[str, int]: Counts of each label value
        """
        label_count: defaultdict[str, int] = defaultdict(lambda: 0)
        for sent in sentences:
            for token in sent.tokens:
                if label_type in token.annotation_layers:
                    label = token.get_label(label_type)
                    label_count[label.value] += 1
        return label_count

    def __str__(self) -> str:
        # Use helper function _len_dataset which handles None
        return (
            f"Corpus: {_len_dataset(self.train)} train + "
            f"{_len_dataset(self.dev)} dev + "
            f"{_len_dataset(self.test)} test sentences"
        )

    def make_label_dictionary(
        self, label_type: str, min_count: int = 1, add_unk: bool = True, add_dev_test: bool = False
    ) -> Dictionary:
        """Creates a Dictionary for a specific label type from the corpus.

        Args:
            label_type: The name of the label type for which the dictionary should be created. Some corpora have
                multiple layers of annotation, such as "pos" and "ner". In this case, you should choose the label type
                you are interested in.
            min_count: Optionally set this to exclude rare labels from the dictionary (i.e., labels seen fewer
                than the provided integer value).
            add_unk: Optionally set this to True to include a "UNK" value in the dictionary. In most cases, this
                is not needed since the label dictionary is well-defined, but some use cases might have open classes
                and require this.
            add_dev_test: Optionally set this to True to construct the label dictionary not only from the train
                split, but also from dev and test. This is only necessary if some labels never appear in train but do
                appear in one of the other splits.

        Returns:
            Dictionary: Dictionary mapping label values to IDs.

        Raises:
            ValueError: If `label_type` is not found.
            AssertionError: If no data splits are available to scan.
        """
        if min_count > 0 and not add_unk:
            add_unk = True
            log.info("Adding <unk>-token to dictionary since min_count is set.")

        label_dictionary: Dictionary = Dictionary(add_unk=add_unk)
        label_dictionary.span_labels = False

        assert self.train
        datasets = [self.train]

        if add_dev_test and self.dev is not None:
            datasets.append(self.dev)

        if add_dev_test and self.test is not None:
            datasets.append(self.test)

        data: ConcatDataset = ConcatDataset(datasets)

        log.info("Computing label dictionary. Progress:")

        sentence_label_type_counter: typing.Counter[str] = Counter()
        label_value_counter: typing.Counter[str] = Counter()
        all_sentence_labels: list[str] = []

        # first, determine the datapoint type by going through dataset until first label is found
        datapoint_type = None
        for sentence in Tqdm.tqdm(_iter_dataset(data)):
            labels = sentence.get_labels(label_type)
            for label in labels:
                datapoint_type = type(label.data_point)
            if datapoint_type:
                break

        if datapoint_type == Span:
            label_dictionary.span_labels = True

        for sentence in Tqdm.tqdm(_iter_dataset(data)):
            # count all label types per sentence
            sentence_label_type_counter.update(sentence.annotation_layers.keys())

            # go through all labels of label_type and count values
            labels = sentence.get_labels(label_type)
            label_value_counter.update(label.value for label in labels if label.value not in all_sentence_labels)

            # special handling for Token-level annotations. Add all untagged as 'O' label
            if datapoint_type == Token and len(sentence) > len(labels):
                label_value_counter["O"] += len(sentence) - len(labels)

            if not label_dictionary.multi_label and len(labels) > 1:
                label_dictionary.multi_label = True

        # if an unk threshold is set, UNK all label values below this threshold
        total_count = 0
        unked_count = 0
        for label, count in label_value_counter.most_common():
            if count >= min_count:
                label_dictionary.add_item(label)
                total_count += count
            else:
                unked_count += count

        if len(label_dictionary.idx2item) == 0 or (
            len(label_dictionary.idx2item) == 1 and label_dictionary.has_item("<unk>")
        ):
            log.error(f"ERROR: You specified label_type='{label_type}' which is not in this dataset!")
            contained_labels = ", ".join(
                [f"'{label[0]}' (in {label[1]} sentences)" for label in sentence_label_type_counter.most_common()]
            )
            log.error(f"ERROR: The corpus contains the following label types: {contained_labels}")
            raise ValueError(
                f"You specified a label type ({label_type}) that is not contained in the corpus:\n{contained_labels}"
            )

        log.info(
            f"Dictionary created for label '{label_type}' with {len(label_dictionary)} "
            f"values: {', '.join([label[0] + f' (seen {label[1]} times)' for label in label_value_counter.most_common(20)])}"
        )

        if unked_count > 0:
            log.info(f" - at UNK threshold {min_count}, {unked_count} instances are UNK'ed and {total_count} remain")

        return label_dictionary

    def add_label_noise(
        self,
        label_type: str,
        labels: list[str],
        noise_share: float = 0.2,
        split: str = "train",
        noise_transition_matrix: Optional[dict[str, list[float]]] = None,
    ) -> None:
        """Adds artificial label noise to a specified split (in-place).

        Stores original labels under `{label_type}_clean`.

        Args:
            label_type (str): Target label type.
            labels (list[str]): List of all possible valid labels for the type.
            noise_share (float, optional): Target proportion for uniform noise (0.0-1.0).
                                          Ignored if matrix is given. Defaults to 0.2.
            split (str, optional): Split to modify ('train', 'dev', 'test'). Defaults to "train".
            noise_transition_matrix (Optional[dict[str, list[float]]], optional):
                Matrix for class-dependent noise. Defaults to None (use uniform noise).
        """
        import numpy as np

        if split == "train":
            assert self.train
            datasets = [self.train]
        elif split == "dev":
            assert self.dev
            datasets = [self.dev]
        elif split == "test":
            assert self.test
            datasets = [self.test]
        else:
            raise ValueError("split must be either train, dev or test.")

        data: ConcatDataset = ConcatDataset(datasets)

        corrupted_count = 0
        total_label_count = 0

        if noise_transition_matrix:
            ntm_labels = noise_transition_matrix.keys()

            if set(ntm_labels) != set(labels):
                raise AssertionError(
                    "Label values in the noise transition matrix have to coincide with label values in the dataset"
                )

            log.info("Generating noisy labels. Progress:")

            for data_point in Tqdm.tqdm(_iter_dataset(data)):
                for label in data_point.get_labels(label_type):
                    total_label_count += 1
                    orig_label = label.value
                    # sample randomly from a label distribution according to the probabilities defined by the noise transition matrix
                    new_label = np.random.default_rng().choice(
                        a=list(ntm_labels),
                        p=noise_transition_matrix[orig_label],
                    )
                    # replace the old label with the new one
                    label.data_point.set_label(label_type, new_label)
                    # keep track of the old (clean) label using another label type category
                    label.data_point.add_label(label_type + "_clean", orig_label)
                    # keep track of how many labels in total are flipped
                    if new_label != orig_label:
                        corrupted_count += 1

        else:
            if noise_share < 0 or noise_share > 1:
                raise ValueError("noise_share must be between 0 and 1.")

            orig_label_p = 1 - noise_share
            other_label_p = noise_share / (len(labels) - 1)

            log.info("Generating noisy labels. Progress:")

            for data_point in Tqdm.tqdm(_iter_dataset(data)):
                for label in data_point.get_labels(label_type):
                    total_label_count += 1
                    orig_label = label.value
                    prob_dist = [other_label_p] * len(labels)
                    prob_dist[labels.index(orig_label)] = orig_label_p
                    # sample randomly from a label distribution according to the probabilities defined by the desired noise share
                    new_label = np.random.default_rng().choice(a=labels, p=prob_dist)
                    # replace the old label with the new one
                    label.data_point.set_label(label_type, new_label)
                    # keep track of the old (clean) label using another label type category
                    label.data_point.add_label(label_type + "_clean", orig_label)
                    # keep track of how many labels in total are flipped
                    if new_label != orig_label:
                        corrupted_count += 1

        log.info(
            f"Total labels corrupted: {corrupted_count}. Resulting noise share: {round((corrupted_count / total_label_count) * 100, 2)}%."
        )

    def get_label_distribution(self):
        """Counts occurrences of each label in the corpus and returns them as a dictionary object.

        This allows you to get an idea of which label appears how often in the Corpus.

        Returns:
            Dictionary with labels as keys and their occurrences as values.
        """
        class_to_count = defaultdict(lambda: 0)
        for sent in self.train:
            for label in sent.labels:
                class_to_count[label.value] += 1
        return class_to_count

    def get_all_sentences(self) -> ConcatDataset:
        """Returns all sentences (spanning all three splits) in the :class:`Corpus`.

        Returns:
            A :class:`torch.utils.data.Dataset` object that includes all sentences of this corpus.
        """
        parts = []
        if self.train:
            parts.append(self.train)
        if self.dev:
            parts.append(self.dev)
        if self.test:
            parts.append(self.test)
        return ConcatDataset(parts)

    @deprecated(version="0.8", reason="Use 'make_label_dictionary(add_unk=False)' instead.")
    def make_tag_dictionary(self, tag_type: str) -> Dictionary:
        """DEPRECATED: Creates tag dictionary ensuring 'O', '<START>', '<STOP>'."""
        tag_dictionary: Dictionary = Dictionary(add_unk=False)
        tag_dictionary.add_item("O")
        for sentence in _iter_dataset(self.get_all_sentences()):
            for token in sentence.tokens:
                tag_dictionary.add_item(token.get_label(tag_type).value)
        tag_dictionary.add_item("<START>")
        tag_dictionary.add_item("<STOP>")
        return tag_dictionary


class MultiCorpus(Corpus):
    """A Corpus composed of multiple individual Corpus objects, often for multi-task learning."""

    def __init__(
        self,
        corpora: list[Corpus],
        task_ids: Optional[list[str]] = None,
        name: str = "multicorpus",
        **corpusargs,
    ) -> None:
        """Initializes a MultiCorpus by concatenating splits from individual corpora.

        Args:
            corpora (list[Corpus]): List of Corpus objects to combine.
            task_ids (Optional[list[str]], optional): List of string IDs for each corpus/task.
                If None, generates default IDs like "Task_0", "Task_1". Defaults to None.
            name (str, optional): Name for the combined corpus. Defaults to "multicorpus".
            **corpusargs: Additional arguments passed to the parent Corpus constructor
                          (e.g., `sample_missing_splits`).
        """
        self.corpora: list[Corpus] = corpora

        ids = task_ids if task_ids else [f"Task_{i}" for i in range(len(corpora))]

        train_parts = []
        dev_parts = []
        test_parts = []
        for corpus in self.corpora:
            if corpus.train:
                train_parts.append(corpus.train)
            if corpus.dev:
                dev_parts.append(corpus.dev)
            if corpus.test:
                test_parts.append(corpus.test)

        super().__init__(
            ConcatFlairDataset(train_parts, ids) if len(train_parts) > 0 else None,
            ConcatFlairDataset(dev_parts, ids) if len(dev_parts) > 0 else None,
            ConcatFlairDataset(test_parts, ids) if len(test_parts) > 0 else None,
            name=name,
            **corpusargs,
        )

    def __str__(self) -> str:
        output = (
            f"MultiCorpus: "
            f"{_len_dataset(self.train) if self.train else 0} train + "
            f"{_len_dataset(self.dev) if self.dev else 0} dev + "
            f"{_len_dataset(self.test) if self.test else 0} test sentences\n - "
        )
        output += "\n - ".join([f"{type(corpus).__name__} {corpus!s} - {corpus.name}" for corpus in self.corpora])
        return output


class FlairDataset(Dataset):
    """Abstract base class for Flair datasets, adding an in-memory check."""

    @abstractmethod
    def is_in_memory(self) -> bool:
        """Returns True if the entire dataset is currently loaded in memory, False otherwise."""
        pass


class ConcatFlairDataset(Dataset):
    """Concatenates multiple datasets, adding a multitask_id label to each sentence.

    Args:
        datasets (Iterable[Dataset]): List of datasets to concatenate.
        ids (Iterable[str]): List of task IDs corresponding to each dataset.
    """

    datasets: list[Dataset]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            length_of_e = len(e)
            r.append(length_of_e + s)
            s += length_of_e
        return r

    def __init__(self, datasets: Iterable[Dataset], ids: Iterable[str]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        self.ids = list(ids)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatSentenceDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Sentence:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        sentence = self.datasets[dataset_idx][sample_idx]
        sentence.set_label("multitask_id", self.ids[dataset_idx])
        return sentence

    @property
    def cummulative_sizes(self) -> list[int]:
        return self.cumulative_sizes


def randomly_split_into_two_datasets(
    dataset: Dataset, length_of_first: int, random_seed: Optional[int] = None
) -> tuple[Subset, Subset]:
    """Shuffles and splits a dataset into two Subsets.

    Args:
        dataset (Dataset): Input dataset.
        length_of_first (int): Desired number of samples in the first subset.
        random_seed (Optional[int], optional): Seed for reproducible shuffle. Defaults to None.

    Returns:
        tuple[Subset, Subset]: The two dataset subsets.

    Raises:
        ValueError: If `length_of_first` is invalid.
    """
    import random

    indices = list(range(_len_dataset(dataset)))
    if random_seed is None:
        random.shuffle(indices)
    else:
        random_generator = random.Random(random_seed)
        random_generator.shuffle(indices)

    first_dataset = indices[:length_of_first]
    second_dataset = indices[length_of_first:]
    first_dataset.sort()
    second_dataset.sort()

    return Subset(dataset, first_dataset), Subset(dataset, second_dataset)


def get_spans_from_bio(
    bioes_tags: list[str], bioes_scores: Optional[list[float]] = None
) -> list[tuple[list[int], float, str]]:
    """Decodes a sequence of BIOES/BIO tags into labeled spans with scores.

    Args:
        bioes_tags (list[str]): List of predicted tags (e.g., "B-PER", "I-PER").
        bioes_scores (Optional[list[float]], optional): Confidence scores for each tag.
            Defaults to 1.0 if None.

    Returns:
        list[tuple[list[int], float, str]]: List of found spans:
            (token_indices, avg_score, label_type).
    """
    # add a dummy "O" to close final prediction
    bioes_tags.append("O")
    # return complex list
    found_spans = []
    # internal variables
    current_tag_weights: dict[str, float] = {}
    previous_tag = "O-"
    current_span: list[int] = []
    current_span_scores: list[float] = []
    for idx, bioes_tag in enumerate(bioes_tags):
        # non-set tags are OUT tags
        if bioes_tag == "" or bioes_tag == "O" or bioes_tag == "_":
            bioes_tag = "O-"

        # anything that is not OUT is IN
        in_span = bioes_tag != "O-"

        # does this prediction start a new span?
        starts_new_span = False

        if bioes_tag[:2] in {"B-", "S-"} or (
            in_span and previous_tag[2:] != bioes_tag[2:] and (bioes_tag[:2] == "I-" or previous_tag[2:] == "S-")
        ):
            # B- and S- always start new spans
            # if the predicted class changes, I- starts a new span
            # if the predicted class changes and S- was previous tag, start a new span
            starts_new_span = True

        # if an existing span is ended (either by reaching O or starting a new span)
        if (starts_new_span or not in_span) and len(current_span) > 0:
            # determine score and value
            span_score = sum(current_span_scores) / len(current_span_scores)
            span_value = max(current_tag_weights.keys(), key=current_tag_weights.__getitem__)

            # append to result list
            found_spans.append((current_span, span_score, span_value))

            # reset for-loop variables for new span
            current_span = []
            current_span_scores = []
            current_tag_weights = {}

        if in_span:
            current_span.append(idx)
            current_span_scores.append(bioes_scores[idx] if bioes_scores else 1.0)
            weight = 1.1 if starts_new_span else 1.0
            current_tag_weights[bioes_tag[2:]] = current_tag_weights.setdefault(bioes_tag[2:], 0.0) + weight

        # remember previous tag
        previous_tag = bioes_tag

    return found_spans
