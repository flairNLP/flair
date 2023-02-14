import bisect
import logging
import re
import typing
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, namedtuple
from functools import lru_cache
from operator import itemgetter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, cast

import torch
from deprecated import deprecated
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataset import ConcatDataset, Subset

import flair
from flair.file_utils import Tqdm
from flair.tokenization import SegtokTokenizer, SpaceTokenizer, Tokenizer

T_co = typing.TypeVar("T_co", covariant=True)

log = logging.getLogger("flair")


def _iter_dataset(dataset: Optional[Dataset]) -> typing.Iterable:
    if dataset is None:
        return []
    from flair.datasets import DataLoader

    return map(lambda x: x[0], DataLoader(dataset, batch_size=1, num_workers=0))


def _len_dataset(dataset: Optional[Dataset]) -> int:
    if dataset is None:
        return 0
    from flair.datasets import DataLoader

    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    return len(loader)


BoundingBox = namedtuple("BoundingBox", ["left", "top", "right", "bottom"])


class Dictionary:
    """
    This class holds a dictionary that maps strings to IDs, used to generate one-hot encodings of strings.
    """

    def __init__(self, add_unk=True):
        # init dictionaries
        self.item2idx: Dict[bytes, int] = {}
        self.idx2item: List[bytes] = []
        self.add_unk = add_unk
        self.multi_label = False
        self.span_labels = False
        # in order to deal with unknown tokens, add <unk>
        if add_unk:
            self.add_item("<unk>")

    def remove_item(self, item: str):
        bytes_item = item.encode("utf-8")
        if bytes_item in self.item2idx:
            self.idx2item.remove(bytes_item)
            del self.item2idx[bytes_item]

    def add_item(self, item: str) -> int:
        """
        add string - if already in dictionary returns its ID. if not in dictionary, it will get a new ID.
        :param item: a string for which to assign an id.
        :return: ID of string
        """
        bytes_item = item.encode("utf-8")
        if bytes_item not in self.item2idx:
            self.idx2item.append(bytes_item)
            self.item2idx[bytes_item] = len(self.idx2item) - 1
        return self.item2idx[bytes_item]

    def get_idx_for_item(self, item: str) -> int:
        """
        returns the ID of the string, otherwise 0
        :param item: string for which ID is requested
        :return: ID of string, otherwise 0
        """
        item_encoded = item.encode("utf-8")
        if item_encoded in self.item2idx.keys():
            return self.item2idx[item_encoded]
        elif self.add_unk:
            return 0
        else:
            log.error(f"The string '{item}' is not in dictionary! Dictionary contains only: {self.get_items()}")
            log.error(
                "You can create a Dictionary that handles unknown items with an <unk>-key by setting add_unk = True in the construction."
            )
            raise IndexError

    def get_idx_for_items(self, items: List[str]) -> List[int]:
        """
        returns the IDs for each item of the list of string, otherwise 0 if not found
        :param items: List of string for which IDs are requested
        :return: List of ID of strings
        """
        if not hasattr(self, "item2idx_not_encoded"):
            d = dict([(key.decode("UTF-8"), value) for key, value in self.item2idx.items()])
            self.item2idx_not_encoded = defaultdict(int, d)

        if not items:
            return []
        results = itemgetter(*items)(self.item2idx_not_encoded)
        if isinstance(results, int):
            return [results]
        return list(results)

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2item:
            items.append(item.decode("UTF-8"))
        return items

    def __len__(self) -> int:
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode("UTF-8")

    def set_start_stop_tags(self):
        self.add_item("<START>")
        self.add_item("<STOP>")

    def start_stop_tags_are_set(self):
        if {"<START>".encode(), "<STOP>".encode()}.issubset(self.item2idx.keys()):
            return True
        else:
            return False

    def save(self, savefile):
        import pickle

        with open(savefile, "wb") as f:
            mappings = {"idx2item": self.idx2item, "item2idx": self.item2idx}
            pickle.dump(mappings, f)

    def __setstate__(self, d):
        self.__dict__ = d
        # set 'add_unk' if the dictionary was created with a version of Flair older than 0.9
        if "add_unk" not in self.__dict__.keys():
            self.__dict__["add_unk"] = True if b"<unk>" in self.__dict__["idx2item"] else False

    @classmethod
    def load_from_file(cls, filename: Union[str, Path]):
        import pickle

        f = open(filename, "rb")
        mappings = pickle.load(f, encoding="latin1")
        idx2item = mappings["idx2item"]
        item2idx = mappings["item2idx"]
        f.close()

        # set 'add_unk' depending on whether <unk> is a key
        add_unk = True if b"<unk>" in idx2item else False

        dictionary: Dictionary = Dictionary(add_unk=add_unk)
        dictionary.item2idx = item2idx
        dictionary.idx2item = idx2item
        return dictionary

    @classmethod
    def load(cls, name: str):
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
        if not isinstance(o, Dictionary):
            return False
        return self.item2idx == o.item2idx and self.idx2item == o.idx2item and self.add_unk == o.add_unk

    def __str__(self):
        tags = ", ".join(self.get_item_for_index(i) for i in range(min(len(self), 50)))
        return f"Dictionary with {len(self)} tags: {tags}"


class Label:
    """
    This class represents a label. Each label has a value and optionally a confidence score. The
    score needs to be between 0.0 and 1.0. Default value for the score is 1.0.
    """

    def __init__(self, data_point: "DataPoint", value: str, score: float = 1.0):
        self._value = value
        self._score = score
        self.data_point: DataPoint = data_point
        super().__init__()

    def set_value(self, value: str, score: float = 1.0):
        self.value = value
        self.score = score

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != "":
            raise ValueError("Incorrect label value provided. Label value needs to be set.")
        else:
            self._value = value

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    def to_dict(self):
        return {"value": self.value, "confidence": self.score}

    def __str__(self):
        return f"{self.data_point.unlabeled_identifier}{flair._arrow}{self._value} ({round(self._score, 4)})"

    @property
    def shortstring(self):
        return f'"{self.data_point.text}"/{self._value}'

    def __repr__(self):
        return f"'{self.data_point.unlabeled_identifier}'/'{self._value}' ({round(self._score, 4)})"

    def __eq__(self, other):
        return self.value == other.value and self.score == other.score and self.data_point == other.data_point

    def __hash__(self):
        return hash(self.__repr__())

    def __lt__(self, other):
        return self.data_point < other.data_point

    @property
    def labeled_identifier(self):
        return f"{self.data_point.unlabeled_identifier}/{self.value}"

    @property
    def unlabeled_identifier(self):
        return f"{self.data_point.unlabeled_identifier}"


class DataPoint:
    """
    This is the parent class of all data points in Flair (including Token, Sentence, Image, etc.). Each DataPoint
    must be embeddable (hence the abstract property embedding() and methods to() and clear_embeddings()). Also,
    each DataPoint may have Labels in several layers of annotation (hence the functions add_label(), get_labels()
    and the property 'label')
    """

    def __init__(self):
        self.annotation_layers = {}
        self._embeddings: Dict[str, torch.Tensor] = {}
        self._metadata: Dict[str, typing.Any] = {}

    @property
    @abstractmethod
    def embedding(self):
        pass

    def set_embedding(self, name: str, vector: torch.Tensor):
        self._embeddings[name] = vector

    def get_embedding(self, names: Optional[List[str]] = None) -> torch.Tensor:
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

    def get_each_embedding(self, embedding_names: Optional[List[str]] = None) -> List[torch.Tensor]:
        embeddings = []
        for embed_name in sorted(self._embeddings.keys()):
            if embedding_names and embed_name not in embedding_names:
                continue
            embed = self._embeddings[embed_name].to(flair.device)
            embeddings.append(embed)
        return embeddings

    def to(self, device: str, pin_memory: bool = False):
        for name, vector in self._embeddings.items():
            if str(vector.device) != str(device):
                if pin_memory:
                    self._embeddings[name] = vector.to(device, non_blocking=True).pin_memory()
                else:
                    self._embeddings[name] = vector.to(device, non_blocking=True)

    def clear_embeddings(self, embedding_names: List[str] = None):
        if embedding_names is None:
            self._embeddings = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]

    def has_label(self, type) -> bool:
        if type in self.annotation_layers:
            return True
        else:
            return False

    def add_metadata(self, key: str, value: typing.Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str) -> typing.Any:
        return self._metadata[key]

    def has_metadata(self, key: str) -> bool:
        return key in self._metadata

    def add_label(self, typename: str, value: str, score: float = 1.0):
        if typename not in self.annotation_layers:
            self.annotation_layers[typename] = [Label(self, value, score)]
        else:
            self.annotation_layers[typename].append(Label(self, value, score))

        return self

    def set_label(self, typename: str, value: str, score: float = 1.0):
        self.annotation_layers[typename] = [Label(self, value, score)]
        return self

    def remove_labels(self, typename: str):
        if typename in self.annotation_layers.keys():
            del self.annotation_layers[typename]

    def get_label(self, label_type: str = None, zero_tag_value="O"):
        if len(self.get_labels(label_type)) == 0:
            return Label(self, zero_tag_value)
        return self.get_labels(label_type)[0]

    def get_labels(self, typename: str = None):
        if typename is None:
            return self.labels

        return self.annotation_layers[typename] if typename in self.annotation_layers else []

    @property
    def labels(self) -> List[Label]:
        all_labels = []
        for key in self.annotation_layers.keys():
            all_labels.extend(self.annotation_layers[key])
        return all_labels

    @property
    @abstractmethod
    def unlabeled_identifier(self):
        raise NotImplementedError

    def _printout_labels(self, main_label=None, add_score: bool = True):
        all_labels = []
        keys = [main_label] if main_label is not None else self.annotation_layers.keys()
        if add_score:
            for key in keys:
                all_labels.extend(
                    [
                        f"{label.value} ({round(label.score, 4)})"
                        for label in self.get_labels(key)
                        if label.data_point == self
                    ]
                )
            labels = "; ".join(all_labels)
            if labels != "":
                labels = flair._arrow + labels
        else:
            for key in keys:
                all_labels.extend([f"{label.value}" for label in self.get_labels(key) if label.data_point == self])
            labels = "/".join(all_labels)
            if labels != "":
                labels = "/" + labels
        return labels

    def __str__(self) -> str:
        return self.unlabeled_identifier + self._printout_labels()

    @property
    @abstractmethod
    def start_position(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def end_position(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def text(self):
        raise NotImplementedError

    @property
    def tag(self):
        return self.labels[0].value

    @property
    def score(self):
        return self.labels[0].score

    def __lt__(self, other):
        return self.start_position < other.start_position

    def __eq__(self, other):
        # TODO: does it make sense to exclude labels? Two data points of identical text (but different labels)
        #  would be equal now.
        return self.unlabeled_identifier == other.unlabeled_identifier

    def __hash__(self):
        return hash(self.unlabeled_identifier)

    def __len__(self):
        raise NotImplementedError


DT = typing.TypeVar("DT", bound=DataPoint)
DT2 = typing.TypeVar("DT2", bound=DataPoint)


class _PartOfSentence(DataPoint, ABC):
    def __init__(self, sentence):
        super().__init__()
        self.sentence: Sentence = sentence

    def _init_labels(self):
        if self.unlabeled_identifier in self.sentence._known_spans:
            self.annotation_layers = self.sentence._known_spans[self.unlabeled_identifier].annotation_layers
            del self.sentence._known_spans[self.unlabeled_identifier]
        self.sentence._known_spans[self.unlabeled_identifier] = self

    def add_label(self, typename: str, value: str, score: float = 1.0):
        super().add_label(typename, value, score)
        self.sentence.annotation_layers.setdefault(typename, []).append(Label(self, value, score))

    def set_label(self, typename: str, value: str, score: float = 1.0):
        if len(self.annotation_layers.get(typename, [])) > 0:
            # First we remove any existing labels for this PartOfSentence in self.sentence
            self.sentence.annotation_layers[typename] = [
                label for label in self.sentence.annotation_layers.get(typename, []) if label.data_point != self
            ]
        self.sentence.annotation_layers.setdefault(typename, []).append(Label(self, value, score))
        super().set_label(typename, value, score)
        return self

    def remove_labels(self, typename: str):
        # labels also need to be deleted at Sentence object
        for label in self.get_labels(typename):
            self.sentence.annotation_layers[typename].remove(label)

        # delete labels at object itself
        super(_PartOfSentence, self).remove_labels(typename)


class Token(_PartOfSentence):
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.
    """

    def __init__(
        self,
        text: str,
        head_id: int = None,
        whitespace_after: int = 1,
        start_position: int = 0,
        sentence=None,
    ):
        super().__init__(sentence=sentence)

        self.form: str = text
        self._internal_index: Optional[int] = None
        self.head_id: Optional[int] = head_id
        self.whitespace_after: int = whitespace_after

        self._start_position = start_position

        self._embeddings: Dict = {}
        self.tags_proba_dist: Dict[str, List[Label]] = {}

    @property
    def idx(self) -> int:
        if isinstance(self._internal_index, int):
            return self._internal_index
        else:
            return -1

    @property
    def text(self) -> str:
        return self.form

    @property
    def unlabeled_identifier(self) -> str:
        return f'Token[{self.idx - 1}]: "{self.text}"'

    def add_tags_proba_dist(self, tag_type: str, tags: List[Label]):
        self.tags_proba_dist[tag_type] = tags

    def get_tags_proba_dist(self, tag_type: str) -> List[Label]:
        if tag_type in self.tags_proba_dist:
            return self.tags_proba_dist[tag_type]
        return []

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    @property
    def start_position(self) -> int:
        return self._start_position

    @start_position.setter
    def start_position(self, value: int) -> None:
        self._start_position = value

    @property
    def end_position(self) -> int:
        return self.start_position + len(self.text)

    @property
    def embedding(self):
        return self.get_embedding()

    def __len__(self):
        return 1

    def __repr__(self):
        return self.__str__()

    def add_label(self, typename: str, value: str, score: float = 1.0):
        """
        The Token is a special _PartOfSentence in that it may be initialized without a Sentence.
        Therefore, labels get added only to the Sentence if it exists
        """
        if self.sentence:
            super().add_label(typename=typename, value=value, score=score)
        else:
            DataPoint.add_label(self, typename=typename, value=value, score=score)

    def set_label(self, typename: str, value: str, score: float = 1.0):
        """
        The Token is a special _PartOfSentence in that it may be initialized without a Sentence.
        Therefore, labels get set only to the Sentence if it exists
        """
        if self.sentence:
            super().set_label(typename=typename, value=value, score=score)
        else:
            DataPoint.set_label(self, typename=typename, value=value, score=score)


class Span(_PartOfSentence):
    """
    This class represents one textual span consisting of Tokens.
    """

    def __init__(self, tokens: List[Token]):
        super().__init__(tokens[0].sentence)
        self.tokens = tokens
        super()._init_labels()

    @property
    def start_position(self) -> int:
        return self.tokens[0].start_position

    @property
    def end_position(self) -> int:
        return self.tokens[-1].end_position

    @property
    def text(self) -> str:
        return "".join([t.text + t.whitespace_after * " " for t in self.tokens]).strip()

    @property
    def unlabeled_identifier(self) -> str:
        return f'Span[{self.tokens[0].idx - 1}:{self.tokens[-1].idx}]: "{self.text}"'

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def embedding(self):
        return self.get_embedding()


class Relation(_PartOfSentence):
    def __init__(self, first: Span, second: Span):
        super().__init__(sentence=first.sentence)
        self.first: Span = first
        self.second: Span = second
        super()._init_labels()

    def __repr__(self):
        return str(self)

    @property
    def tag(self):
        return self.labels[0].value

    @property
    def text(self):
        return f"{self.first.text} -> {self.second.text}"

    @property
    def unlabeled_identifier(self) -> str:
        return (
            f"Relation"
            f"[{self.first.tokens[0].idx - 1}:{self.first.tokens[-1].idx}]"
            f"[{self.second.tokens[0].idx - 1}:{self.second.tokens[-1].idx}]"
            f': "{self.text}"'
        )

    @property
    def start_position(self) -> int:
        return min(self.first.start_position, self.second.start_position)

    @property
    def end_position(self) -> int:
        return max(self.first.end_position, self.second.end_position)

    @property
    def embedding(self):
        pass


class Sentence(DataPoint):
    """
    A Sentence is a list of tokens and is used to represent a sentence or text fragment.
    """

    def __init__(
        self,
        text: Union[str, List[str], List[Token]],
        use_tokenizer: Union[bool, Tokenizer] = True,
        language_code: str = None,
        start_position: int = 0,
    ):
        """
        Class to hold all meta related to a text (tokens, predictions, language code, ...)
        :param text: original string (sentence), or a list of string tokens (words)
        :param use_tokenizer: a custom tokenizer (default is :class:`SpaceTokenizer`)
            more advanced options are :class:`SegTokTokenizer` to use segtok or :class:`SpacyTokenizer`
            to use Spacy library if available). Check the implementations of abstract class Tokenizer or
            implement your own subclass (if you need it). If instead of providing a Tokenizer, this parameter
            is just set to True (deprecated), :class:`SegtokTokenizer` will be used.
        :param language_code: Language of the sentence
        :param start_position: Start char offset of the sentence in the superordinate document
        """
        super().__init__()

        self.tokens: List[Token] = []

        # private field for all known spans
        self._known_spans: Dict[str, _PartOfSentence] = {}

        self.language_code: Optional[str] = language_code

        self._start_position = start_position

        # the tokenizer used for this sentence
        if isinstance(use_tokenizer, Tokenizer):
            tokenizer = use_tokenizer

        elif type(use_tokenizer) == bool:
            tokenizer = SegtokTokenizer() if use_tokenizer else SpaceTokenizer()

        else:
            raise AssertionError("Unexpected type of parameter 'use_tokenizer'. Parameter should be bool or Tokenizer")

        self.tokenized: Optional[str] = None

        # some sentences represent a document boundary (but most do not)
        self.is_document_boundary: bool = False

        # internal variables to denote position inside dataset
        self._previous_sentence: Optional[Sentence] = None
        self._has_context: bool = False
        self._next_sentence: Optional[Sentence] = None
        self._position_in_dataset: Optional[typing.Tuple[Dataset, int]] = None

        # if text is passed, instantiate sentence with tokens (words)
        if isinstance(text, str):
            text = Sentence._handle_problem_characters(text)
            words = tokenizer.tokenize(text)
        elif text and isinstance(text[0], Token):
            for t in text:
                self._add_token(t)
            self.tokens[-1].whitespace_after = 0
            return
        else:
            words = cast(List[str], text)
            text = " ".join(words)

        # determine token positions and whitespace_after flag
        current_offset: int = 0
        previous_token: Optional[Token] = None
        for word in words:
            word_start_position: int = text.index(word, current_offset)
            delta_offset: int = word_start_position - current_offset

            token: Token = Token(text=word, start_position=word_start_position)
            self._add_token(token)

            if previous_token is not None:
                previous_token.whitespace_after = delta_offset

            current_offset = token.end_position
            previous_token = token

        # the last token has no whitespace after
        if len(self) > 0:
            self.tokens[-1].whitespace_after = 0

        # log a warning if the dataset is empty
        if text == "":
            log.warning("Warning: An empty Sentence was created! Are there empty strings in your dataset?")

    @property
    def unlabeled_identifier(self):
        return f'Sentence[{len(self)}]: "{self.text}"'

    def get_relations(self, type: str) -> List[Relation]:
        relations: List[Relation] = []
        for label in self.get_labels(type):
            if isinstance(label.data_point, Relation):
                relations.append(label.data_point)
        return relations

    def get_spans(self, type: str) -> List[Span]:
        spans: List[Span] = []
        for potential_span in self._known_spans.values():
            if isinstance(potential_span, Span) and potential_span.has_label(type):
                spans.append(potential_span)
        return sorted(spans)

    def get_token(self, token_id: int) -> Optional[Token]:
        for token in self.tokens:
            if token.idx == token_id:
                return token
        return None

    def _add_token(self, token: Union[Token, str]):
        if isinstance(token, Token):
            assert token.sentence is None

        if type(token) is str:
            token = Token(token)
        token = cast(Token, token)

        # data with zero-width characters cannot be handled
        if token.text == "":
            return

        # set token idx and sentence
        token.sentence = self
        token._internal_index = len(self.tokens) + 1
        if token.start_position == 0 and len(self) > 0:
            token.start_position = len(self.to_original_text()) + self[-1].whitespace_after

        # append token to sentence
        self.tokens.append(token)

        # register token annotations on sentence
        for typename in token.annotation_layers.keys():
            for label in token.get_labels(typename):
                if typename not in token.sentence.annotation_layers:
                    token.sentence.annotation_layers[typename] = [Label(token, label.value, label.score)]
                else:
                    token.sentence.annotation_layers[typename].append(Label(token, label.value, label.score))

    @property
    def embedding(self):
        return self.get_embedding()

    def to(self, device: str, pin_memory: bool = False):
        # move sentence embeddings to device
        super().to(device=device, pin_memory=pin_memory)

        # also move token embeddings to device
        for token in self:
            token.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: List[str] = None):
        super().clear_embeddings(embedding_names)

        # clear token embeddings
        for token in self:
            token.clear_embeddings(embedding_names)

    @lru_cache(maxsize=1)  # cache last context, as training repeats calls
    def left_context(self, context_length: int, respect_document_boundaries: bool = True) -> List[Token]:
        sentence = self
        left_context: List[Token] = []
        while len(left_context) < context_length:
            sentence = sentence.previous_sentence()
            if sentence is None:
                break

            if respect_document_boundaries and sentence.is_document_boundary:
                break

            left_context = sentence.tokens + left_context
        return left_context[-context_length:]

    @lru_cache(maxsize=1)  # cache last context, as training repeats calls
    def right_context(self, context_length: int, respect_document_boundaries: bool = True) -> List[Token]:
        sentence = self
        right_context: List[Token] = []
        while len(right_context) < context_length:
            sentence = sentence.next_sentence()
            if sentence is None:
                break
            if respect_document_boundaries and sentence.is_document_boundary:
                break

            right_context += sentence.tokens
        return right_context[:context_length]

    def __str__(self):
        return self.to_tagged_string()

    def to_tagged_string(self, main_label=None) -> str:
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

    @property
    def text(self):
        return self.to_original_text()

    def to_tokenized_string(self) -> str:
        if self.tokenized is None:
            self.tokenized = " ".join([t.text for t in self.tokens])

        return self.tokenized

    def to_plain_string(self):
        plain = ""
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after > 0:
                plain += token.whitespace_after * " "
        return plain.rstrip()

    def infer_space_after(self):
        """
        Heuristics in case you wish to infer whitespace_after values for tokenized text. This is useful for some old NLP
        tasks (such as CoNLL-03 and CoNLL-2000) that provide only tokenized data with no info of original whitespacing.
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

    def to_original_text(self) -> str:
        # if sentence has no tokens, return empty string
        if len(self) == 0:
            return ""
        # otherwise, return concatenation of tokens with the correct offsets
        return self[0].start_position * " " + "".join([t.text + t.whitespace_after * " " for t in self.tokens]).strip()

    def to_dict(self, tag_type: str = None):
        labels = []

        if tag_type:
            labels = [label.to_dict() for label in self.get_labels(tag_type)]
            return {"text": self.to_original_text(), tag_type: labels}

        if self.labels:
            labels = [label.to_dict() for label in self.labels]

        return {"text": self.to_original_text(), "all labels": labels}

    def get_span(self, start: int, stop: int):
        span_slice = slice(start, stop)
        return self[span_slice]

    @typing.overload
    def __getitem__(self, idx: int) -> Token:
        ...

    @typing.overload
    def __getitem__(self, s: slice) -> Span:
        ...

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return Span(self.tokens[subscript])
        else:
            return self.tokens[subscript]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self):
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
            except Exception:
                self.language_code = "en"

        return self.language_code

    @staticmethod
    def _handle_problem_characters(text: str) -> str:
        text = Sentence.__remove_zero_width_characters(text)
        text = Sentence.__restore_windows_1252_characters(text)
        return text

    @staticmethod
    def __remove_zero_width_characters(text: str) -> str:
        text = text.replace("\u200c", "")
        text = text.replace("\u200b", "")
        text = text.replace("\ufe0f", "")
        text = text.replace("\ufeff", "")
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
        """
        Get the next sentence in the document (works only if context is set through dataloader or elsewhere)
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
        """
        Get the previous sentence in the document (works only if context is set through dataloader or elsewhere)
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
        """
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
    def set_context_for_sentences(cls, sentences: List["Sentence"]) -> None:
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

    def get_labels(self, label_type: str = None):
        # if no label if specified, return all labels
        if label_type is None:
            return sorted(self.labels)

        # if the label type exists in the Sentence, return it
        if label_type in self.annotation_layers:
            return sorted(self.annotation_layers[label_type])

        # return empty list if none of the above
        return []

    def remove_labels(self, typename: str):
        # labels also need to be deleted at all tokens
        for token in self:
            token.remove_labels(typename)

        # labels also need to be deleted at all known spans
        for span in self._known_spans.values():
            span.remove_labels(typename)

        # remove spans without labels
        self._known_spans = {k: v for k, v in self._known_spans.items() if len(v.labels) > 0}

        # delete labels at object itself
        super().remove_labels(typename)


class DataPair(DataPoint, typing.Generic[DT, DT2]):
    def __init__(self, first: DT, second: DT2):
        super().__init__()
        self.first = first
        self.second = second

    def to(self, device: str, pin_memory: bool = False):
        self.first.to(device, pin_memory)
        self.second.to(device, pin_memory)

    def clear_embeddings(self, embedding_names: List[str] = None):
        self.first.clear_embeddings(embedding_names)
        self.second.clear_embeddings(embedding_names)

    @property
    def embedding(self):
        return torch.cat([self.first.embedding, self.second.embedding])

    def __len__(self):
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


TextPair = DataPair[Sentence, Sentence]


class Image(DataPoint):
    def __init__(self, data=None, imageURL=None):
        super().__init__()

        self.data = data
        self._embeddings: Dict = {}
        self.imageURL = imageURL

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self):
        image_repr = self.data.size() if self.data else ""
        image_url = self.imageURL if self.imageURL else ""

        return f"Image: {image_repr} {image_url}"

    @property
    def start_position(self) -> int:
        pass

    @property
    def end_position(self) -> int:
        pass

    @property
    def text(self):
        pass

    @property
    def unlabeled_identifier(self):
        pass


class Corpus(typing.Generic[T_co]):
    def __init__(
        self,
        train: Optional[Dataset[T_co]] = None,
        dev: Optional[Dataset[T_co]] = None,
        test: Optional[Dataset[T_co]] = None,
        name: str = "corpus",
        sample_missing_splits: Union[bool, str] = True,
    ):
        # set name
        self.name: str = name

        # abort if no data is provided
        if not train and not dev and not test:
            raise RuntimeError("No data provided when initializing corpus object.")

        # sample test data from train if none is provided
        if test is None and sample_missing_splits and train and not sample_missing_splits == "only_dev":
            train_length = _len_dataset(train)
            test_size: int = round(train_length / 10)
            test, train = randomly_split_into_two_datasets(train, test_size)

        # sample dev data from train if none is provided
        if dev is None and sample_missing_splits and train and not sample_missing_splits == "only_test":
            train_length = _len_dataset(train)
            dev_size: int = round(train_length / 10)
            dev, train = randomly_split_into_two_datasets(train, dev_size)

        # set train dev and test data
        self._train: Optional[Dataset[T_co]] = train
        self._test: Optional[Dataset[T_co]] = test
        self._dev: Optional[Dataset[T_co]] = dev

    @property
    def train(self) -> Optional[Dataset[T_co]]:
        return self._train

    @property
    def dev(self) -> Optional[Dataset[T_co]]:
        return self._dev

    @property
    def test(self) -> Optional[Dataset[T_co]]:
        return self._test

    def downsample(
        self,
        percentage: float = 0.1,
        downsample_train=True,
        downsample_dev=True,
        downsample_test=True,
    ):
        if downsample_train and self._train is not None:
            self._train = self._downsample_to_proportion(self._train, percentage)

        if downsample_dev and self._dev is not None:
            self._dev = self._downsample_to_proportion(self._dev, percentage)

        if downsample_test and self._test is not None:
            self._test = self._downsample_to_proportion(self._test, percentage)

        return self

    def filter_empty_sentences(self):
        log.info("Filtering empty sentences")
        if self._train is not None:
            self._train = Corpus._filter_empty_sentences(self._train)
        if self._test is not None:
            self._test = Corpus._filter_empty_sentences(self._test)
        if self._dev is not None:
            self._dev = Corpus._filter_empty_sentences(self._dev)
        log.info(self)

    def filter_long_sentences(self, max_charlength: int):
        log.info("Filtering long sentences")
        if self._train is not None:
            self._train = Corpus._filter_long_sentences(self._train, max_charlength)
        if self._test is not None:
            self._test = Corpus._filter_long_sentences(self._test, max_charlength)
        if self._dev is not None:
            self._dev = Corpus._filter_long_sentences(self._dev, max_charlength)
        log.info(self)

    @staticmethod
    def _filter_long_sentences(dataset, max_charlength: int) -> Dataset:
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

    def make_vocab_dictionary(self, max_tokens=-1, min_freq=1) -> Dictionary:
        """
        Creates a dictionary of all tokens contained in the corpus.
        By defining `max_tokens` you can set the maximum number of tokens that should be contained in the dictionary.
        If there are more than `max_tokens` tokens in the corpus, the most frequent tokens are added first.
        If `min_freq` is set the a value greater than 1 only tokens occurring more than `min_freq` times are considered
        to be added to the dictionary.
        :param max_tokens: the maximum number of tokens that should be added to the dictionary (-1 = take all tokens)
        :param min_freq: a token needs to occur at least `min_freq` times to be added to the dictionary (-1 = there is no limitation)
        :return: dictionary of tokens
        """
        tokens = self._get_most_common_tokens(max_tokens, min_freq)

        vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            vocab_dictionary.add_item(token)

        return vocab_dictionary

    def _get_most_common_tokens(self, max_tokens, min_freq) -> List[str]:
        tokens_and_frequencies = Counter(self._get_all_tokens())

        tokens: List[str] = []
        for token, freq in tokens_and_frequencies.most_common():
            if (min_freq != -1 and freq < min_freq) or (max_tokens != -1 and len(tokens) == max_tokens):
                break
            tokens.append(token)
        return tokens

    def _get_all_tokens(self) -> List[str]:
        assert self.train
        tokens = list(map((lambda s: s.tokens), _iter_dataset(self.train)))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))

    @staticmethod
    def _downsample_to_proportion(dataset: Dataset, proportion: float):
        sampled_size: int = round(_len_dataset(dataset) * proportion)
        splits = randomly_split_into_two_datasets(dataset, sampled_size)
        return splits[0]

    def obtain_statistics(self, label_type: str = None, pretty_print: bool = True) -> Union[dict, str]:
        """
        Print statistics about the class distribution (only labels of sentences are taken into account) and sentence
        sizes.
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
    def _obtain_statistics_for(sentences, name, tag_type) -> dict:
        if len(sentences) == 0:
            return {}

        classes_to_count = Corpus._count_sentence_labels(sentences)
        tags_to_count = Corpus._count_token_labels(sentences, tag_type)
        tokens_per_sentence = Corpus._get_tokens_per_sentence(sentences)

        label_size_dict = {}
        for label, c in classes_to_count.items():
            label_size_dict[label] = c

        tag_size_dict = {}
        for tag, c in tags_to_count.items():
            tag_size_dict[tag] = c

        return {
            "dataset": name,
            "total_number_of_documents": len(sentences),
            "number_of_documents_per_class": label_size_dict,
            "number_of_tokens_per_tag": tag_size_dict,
            "number_of_tokens": {
                "total": sum(tokens_per_sentence),
                "min": min(tokens_per_sentence),
                "max": max(tokens_per_sentence),
                "avg": sum(tokens_per_sentence) / len(sentences),
            },
        }

    @staticmethod
    def _get_tokens_per_sentence(sentences):
        return list(map(lambda x: len(x.tokens), sentences))

    @staticmethod
    def _count_sentence_labels(sentences):
        label_count = defaultdict(lambda: 0)
        for sent in sentences:
            for label in sent.labels:
                label_count[label.value] += 1
        return label_count

    @staticmethod
    def _count_token_labels(sentences, label_type):
        label_count = defaultdict(lambda: 0)
        for sent in sentences:
            for token in sent.tokens:
                if label_type in token.annotation_layers.keys():
                    label = token.get_label(label_type)
                    label_count[label.value] += 1
        return label_count

    def __str__(self) -> str:
        return "Corpus: %d train + %d dev + %d test sentences" % (
            _len_dataset(self.train) if self.train else 0,
            _len_dataset(self.dev) if self.dev else 0,
            _len_dataset(self.test) if self.test else 0,
        )

    def make_label_dictionary(self, label_type: str, min_count: int = -1, add_unk: bool = True) -> Dictionary:
        """
        Creates a dictionary of all labels assigned to the sentences in the corpus.
        :return: dictionary of labels
        """
        if min_count > 0 and not add_unk:
            raise ValueError("Cannot require a minimum count if no unk-token is created.")

        label_dictionary: Dictionary = Dictionary(add_unk=add_unk)
        label_dictionary.span_labels = False

        assert self.train
        datasets = [self.train]

        data: ConcatDataset = ConcatDataset(datasets)

        log.info("Computing label dictionary. Progress:")

        sentence_label_type_counter: typing.Counter[str] = Counter()
        label_value_counter: typing.Counter[str] = Counter()
        all_sentence_labels: List[str] = []
        for sentence in Tqdm.tqdm(_iter_dataset(data)):
            # count all label types per sentence
            sentence_label_type_counter.update(sentence.annotation_layers.keys())

            # go through all labels of label_type and count values
            labels = sentence.get_labels(label_type)
            for label in labels:
                if label.value not in all_sentence_labels:
                    label_value_counter[label.value] += 1

                # check if there are any span labels
                if type(label.data_point) == Span and len(label.data_point) > 1:
                    label_dictionary.span_labels = True

            if not label_dictionary.multi_label:
                if len(labels) > 1:
                    label_dictionary.multi_label = True

        # if an unk threshold is set, UNK all label values below this threshold
        erfasst_count = 0
        unked_count = 0
        for label, count in label_value_counter.most_common():
            if count >= min_count:
                label_dictionary.add_item(label)
                erfasst_count += count
            else:
                unked_count += count

        if len(label_dictionary.idx2item) == 0 or (
            len(label_dictionary.idx2item) == 1 and "<unk>" in label_dictionary.get_items()
        ):
            log.error(f"ERROR: You specified label_type='{label_type}' which is not in this dataset!")
            contained_labels = ", ".join(
                [f"'{label[0]}' (in {label[1]} sentences)" for label in sentence_label_type_counter.most_common()]
            )
            log.error(f"ERROR: The corpus contains the following label types: {contained_labels}")
            raise Exception

        log.info(
            f"Dictionary created for label '{label_type}' with {len(label_dictionary)} "
            f"values: {', '.join([label[0] + f' (seen {label[1]} times)' for label in label_value_counter.most_common(20)])}"
        )

        if unked_count > 0:
            log.info(f" - at UNK threshold {min_count}, {unked_count} instances are UNK'ed and {erfasst_count} remain")

        return label_dictionary

    def _corrupt_labels(self, noise_share: float, label_type: str, labels: List[str], split: str = "train"):
        """
        Generates uniform label noise distribution the chosen dataset split.
        :noise_share: the desired share of noise in the train split.
        :label_type: the type of labels for which the noise should be simulated.
        :labels: an array with unique labels of said type (retrievable from label dictionary).
        :split: in which dataset split the noise is to be simulated.
        """
        import numpy as np

        if noise_share < 0 or noise_share > 1:
            raise ValueError("noise_share must be between 0 and 1.")

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

        orig_label_p = 1 - noise_share
        other_label_p = noise_share / (len(labels) - 1)

        corrupted_count = 0
        total_label_count = 0

        log.info("Generating noisy labels. Progress:")

        for data_point in Tqdm.tqdm(_iter_dataset(data)):
            for label in data_point.get_labels(label_type):
                total_label_count += 1
                orig_label = label.value
                prob_dist = [other_label_p] * len(labels)
                prob_dist[labels.index(orig_label)] = orig_label_p
                # sample randomly from a label distribution according to the probabilities defined by the desired noise share
                new_label = np.random.choice(a=labels, p=prob_dist)
                # replace the old label with the new one
                label.value = new_label
                # keep track of the old (clean) label using another label type category
                label.data_point.add_label(label_type + "_clean", orig_label)
                # keep track of how many labels in total are flipped
                if new_label != orig_label:
                    corrupted_count += 1

        log.info(
            f"Total labels corrupted: {corrupted_count}. Resulting noise share: {round((corrupted_count/total_label_count)*100, 2)}%."
        )

    def get_label_distribution(self):
        class_to_count = defaultdict(lambda: 0)
        for sent in self.train:
            for label in sent.labels:
                class_to_count[label.value] += 1
        return class_to_count

    def get_all_sentences(self) -> ConcatDataset:
        parts = []
        if self.train:
            parts.append(self.train)
        if self.dev:
            parts.append(self.dev)
        if self.test:
            parts.append(self.test)
        return ConcatDataset(parts)

    @deprecated(version="0.8", reason="Use 'make_label_dictionary' instead.")
    def make_tag_dictionary(self, tag_type: str) -> Dictionary:
        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary(add_unk=False)
        tag_dictionary.add_item("O")
        for sentence in _iter_dataset(self.get_all_sentences()):
            for token in sentence.tokens:
                tag_dictionary.add_item(token.get_label(tag_type).value)
        tag_dictionary.add_item("<START>")
        tag_dictionary.add_item("<STOP>")
        return tag_dictionary


class MultiCorpus(Corpus):
    def __init__(
        self,
        corpora: List[Corpus],
        task_ids: Optional[List[str]] = None,
        name: str = "multicorpus",
        **corpusargs,
    ):
        self.corpora: List[Corpus] = corpora

        ids = task_ids if task_ids else [f"Task_{i}" for i in range(len(corpora))]

        train_parts = []
        dev_parts = []
        test_parts = []
        print(self.corpora)
        for corpus in self.corpora:
            if corpus.train:
                train_parts.append(corpus.train)
            if corpus.dev:
                dev_parts.append(corpus.dev)
            if corpus.test:
                test_parts.append(corpus.test)

        super(MultiCorpus, self).__init__(
            ConcatFlairDataset(train_parts, ids) if len(train_parts) > 0 else None,
            ConcatFlairDataset(dev_parts, ids) if len(dev_parts) > 0 else None,
            ConcatFlairDataset(test_parts, ids) if len(test_parts) > 0 else None,
            name=name,
            **corpusargs,
        )

    def __str__(self):
        output = (
            f"MultiCorpus: "
            f"{len(self.train) if self.train else 0} train + "
            f"{len(self.dev) if self.dev else 0} dev + "
            f"{len(self.test) if self.test else 0} test sentences\n - "
        )
        output += "\n - ".join([f"{type(corpus).__name__} {str(corpus)} - {corpus.name}" for corpus in self.corpora])
        return output


class FlairDataset(Dataset):
    @abstractmethod
    def is_in_memory(self) -> bool:
        pass


class ConcatFlairDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            length_of_e = len(e)
            r.append(length_of_e + s)
            s += length_of_e
        return r

    def __init__(self, datasets: Iterable[Dataset], ids: Iterable[str]) -> None:
        super(ConcatFlairDataset, self).__init__()
        self.datasets = list(datasets)
        self.ids = list(ids)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatSentenceDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sentence = self.datasets[dataset_idx][sample_idx]
        sentence.set_label("multitask_id", self.ids[dataset_idx])
        return sentence

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag.value == "O":
            continue
        split = tag.value.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            return False
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1].value == "O":  # conversion IOB1 to IOB2
            tags[i].value = "B" + tag.value[1:]
        elif tags[i - 1].value[1:] == tag.value[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i].value = "B" + tag.value[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    for i, tag in enumerate(tags):
        if tag.value == "O" or tag.value == "":
            tag.value = "O"
            continue
        t, label = tag.value.split("-", 1)
        if len(tags) == i + 1 or tags[i + 1].value == "O":
            next_same = False
        else:
            nt, next_label = tags[i + 1].value.split("-", 1)
            next_same = nt == "I" and next_label == label
        if t == "B":
            if not next_same:
                tag.value = tag.value.replace("B-", "S-")
        elif t == "I":
            if not next_same:
                tag.value = tag.value.replace("I-", "E-")
        else:
            raise Exception("Invalid IOB format!")


def randomly_split_into_two_datasets(dataset, length_of_first):
    import random

    indices = [i for i in range(len(dataset))]
    random.shuffle(indices)

    first_dataset = indices[:length_of_first]
    second_dataset = indices[length_of_first:]
    first_dataset.sort()
    second_dataset.sort()

    return Subset(dataset, first_dataset), Subset(dataset, second_dataset)


def get_spans_from_bio(bioes_tags: List[str], bioes_scores=None) -> List[typing.Tuple[List[int], float, str]]:
    # add a dummy "O" to close final prediction
    bioes_tags.append("O")
    # return complex list
    found_spans = []
    # internal variables
    current_tag_weights: Dict[str, float] = defaultdict(lambda: 0.0)
    previous_tag = "O-"
    current_span: List[int] = []
    current_span_scores: List[float] = []
    for idx, bioes_tag in enumerate(bioes_tags):
        # non-set tags are OUT tags
        if bioes_tag == "" or bioes_tag == "O" or bioes_tag == "_":
            bioes_tag = "O-"

        # anything that is not OUT is IN
        in_span = False if bioes_tag == "O-" else True

        # does this prediction start a new span?
        starts_new_span = False

        # begin and single tags start new spans
        if bioes_tag[0:2] in ["B-", "S-"]:
            starts_new_span = True

        # in IOB format, an I tag starts a span if it follows an O or is a different span
        if bioes_tag[0:2] == "I-" and previous_tag[2:] != bioes_tag[2:]:
            starts_new_span = True

        # single tags that change prediction start new spans
        if bioes_tag[0:2] in ["S-"] and previous_tag[2:] != bioes_tag[2:]:
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
            current_tag_weights = defaultdict(lambda: 0.0)

        if in_span:
            current_span.append(idx)
            current_span_scores.append(bioes_scores[idx] if bioes_scores else 1.0)
            weight = 1.1 if starts_new_span else 1.0
            current_tag_weights[bioes_tag[2:]] += weight

        # remember previous tag
        previous_tag = bioes_tag

    return found_spans
