import re
import typing
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from flair.data import Sentence, Span, Token


@dataclass
class TokenCollection:
    """
    A utility class for RegexpTagger to hold all tokens for a given Sentence and define some functionality
    :param sentence: A Sentence object
    """

    sentence: Sentence
    __tokens_start_pos: List[int] = field(init=False, default_factory=list)
    __tokens_end_pos: List[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        for token in self.tokens:
            self.__tokens_start_pos.append(token.start_pos)
            self.__tokens_end_pos.append(token.end_pos)

    @property
    def tokens(self) -> List[Token]:
        return list(self.sentence)

    def get_token_span(self, span: Tuple[int, int]) -> Span:
        """
        Given an interval specified with start and end pos as tuple, this function returns a Span object
        spanning the tokens included in the interval. If the interval is overlapping with a token span, a
        ValueError is raised

        :param span: Start and end pos of the requested span as tuple
        :return: A span object spanning the requested token interval
        """
        span_start: int = self.__tokens_start_pos.index(span[0])
        span_end: int = self.__tokens_end_pos.index(span[1])
        return Span(self.tokens[span_start : span_end + 1])


class RegexpTagger:
    def __init__(self, mapping: Union[List[Tuple[str, str]], Tuple[str, str]]):
        """
        This tagger is capable of tagging sentence objects with given regexp -> label mappings.

        I.e: The tuple (r'(["\'])(?:(?=(\\?))\2.)*?\1', 'QUOTE') maps every match of the regexp to
        a <QUOTE> labeled span and therefore labels the given sentence object with RegexpTagger.predict().
        This tagger supports multilabeling so tokens can be included in multiple labeled spans.
        The regexp are compiled internally and an re.error will be raised if the compilation of a given regexp fails.

        If a match violates (in this case overlaps) a token span, an exception is raised.

        :param mapping: A list of tuples or a single tuple representing a mapping as regexp -> label
        """
        self._regexp_mapping: Dict[str, typing.Pattern] = {}
        self.register_labels(mapping=mapping)

    @property
    def registered_labels(self):
        return self._regexp_mapping

    def register_labels(self, mapping: Union[List[Tuple[str, str]], Tuple[str, str]]):
        """
        Register a regexp -> label mapping.
        :param mapping: A list of tuples or a single tuple representing a mapping as regexp -> label
        """
        mapping = self._listify(mapping)

        for regexp, label in mapping:
            try:
                self._regexp_mapping[label] = re.compile(regexp)
            except re.error as err:
                raise re.error(
                    f"Couldn't compile regexp '{regexp}' for label '{label}'. Aborted with error: '{err.msg}'"
                )

    def remove_labels(self, labels: Union[List[str], str]):
        """
        Remove a registered regexp -> label mapping given by label.
        :param labels: A list of labels or a single label as strings.
        """
        labels = self._listify(labels)

        for label in labels:
            if not self._regexp_mapping.get(label):
                continue
            self._regexp_mapping.pop(label)

    @staticmethod
    def _listify(element: object) -> list:
        if not isinstance(element, list):
            return [element]
        else:
            return element

    def predict(self, sentences: Union[List[Sentence], Sentence]) -> List[Sentence]:
        """
        Predict the given sentences according to the registered mappings.
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        if not sentences:
            return sentences

        sentences = self._listify(sentences)
        for sentence in sentences:
            self._label(sentence)
        return sentences

    def _label(self, sentence: Sentence):
        """
        This will add a complex_label to the given sentence for every match.span() for every registered_mapping.
        If a match span overlaps with a token span an exception is raised.
        """
        collection = TokenCollection(sentence)

        for label, pattern in self._regexp_mapping.items():
            for match in pattern.finditer(sentence.to_original_text()):
                span: Tuple[int, int] = match.span()
                try:
                    token_span = collection.get_token_span(span)
                except ValueError:
                    raise Exception(f"The match span {span} for label '{label}' is overlapping with a token!")
                token_span.add_label(label, label)
