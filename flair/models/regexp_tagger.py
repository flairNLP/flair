import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union

from flair.data import Sentence


class RegexpTagger:
    @dataclass
    class TokenCollection:
        sentence: Sentence
        __tokens_start_pos: List[int] = field(init=False, default_factory=list)
        __tokens_end_pos: List[int] = field(init=False, default_factory=list)

        def __post_init__(self):
            for token in self.tokens:
                self.__tokens_start_pos.append(token.start_pos)
                self.__tokens_end_pos.append(token.end_pos)

        @property
        def tokens(self):
            return list(self.sentence)

        def get_token_indexes_for_span(self, span: Tuple[int, int]) -> Tuple[int, int]:
            span_start: int = self.__tokens_start_pos.index(span[0])
            span_end: int = self.__tokens_end_pos.index(span[1])
            return span_start, span_end

    def __init__(self, regexps: Union[List[Tuple[str, str]], Tuple[str, str]]):
        self._regexp_mapping: Dict[str, re.Pattern] = {}

        self.register_tags(regexps=regexps)

    def remove_tags(self, tags: Union[List[str], str]):
        tags = self._listify(tags)

        for tag in tags:
            self._regexp_mapping.pop(key=tag)

    def register_tags(self, regexps: List[Tuple[str, str]]):
        regexps = self._listify(regexps)

        for regexp, tag in regexps:
            try:
                self._regexp_mapping[tag] = re.compile(regexp)
            except re.error as err:
                raise re.error(f"Couldn't compile regexp '{regexp}' for tag '{tag}'. Aborted with error: '{err.msg}'")

    @staticmethod
    def _listify(element: object) -> list:
        if not isinstance(element, list):
            return [element]
        else:
            return element

    def predict(self, sentences: Union[List[Sentence], Sentence]):
        if not sentences:
            return sentences

        sentences = self._listify(sentences)
        for sentence in sentences:
            self._tag(sentence)
        return sentences

    def _tag(self, sentence: Sentence):
        collection = RegexpTagger.TokenCollection(sentence)

        for tag, pattern in self._regexp_mapping.items():
            for match in pattern.finditer(sentence.to_original_text()):
                span: Tuple[int, int] = match.span()
                try:
                    token_span = collection.get_token_indexes_for_span(span)
                except ValueError:
                    raise Exception(f"The match span {span} for tag '{tag}' is overlapping with a token!")
                if token_span[1]-token_span[0] > 0:
                    sentence.tokens[token_span[0]].add_tag('regexp', 'B-' + tag)
                    for i in range(token_span[0]+1, token_span[1]+1):
                        sentence.tokens[i].add_tag('regexp', 'I-' + tag)
                    sentence.tokens[token_span[1]].add_tag('regexp', 'E-' + tag)
                else:
                    sentence.tokens[token_span[0]].add_tag('regexp', 'S-' + tag)


if __name__ == '__main__':
    sentence = Sentence('"This talk" is interesting.')
    tagger = RegexpTagger([('This talk', 'TEST'), (r'(?<=\s)is(?=\s)', 'TEST_2'), ('toll', 'TEST_3')])
    tagger.predict(sentence)
    for entity in sentence.get_spans('regexp'):
        print(entity)
