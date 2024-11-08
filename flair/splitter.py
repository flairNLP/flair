from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from segtok.segmenter import split_multi

from flair.data import Sentence
from flair.tokenization import (
    SciSpacyTokenizer,
    SegtokTokenizer,
    SpacyTokenizer,
    Tokenizer,
)


class SentenceSplitter(ABC):
    r"""An abstract class representing a :class:`SentenceSplitter`.

    Sentence splitters are used to represent algorithms and models to split plain text into
    sentences and individual tokens / words. All subclasses should overwrite :meth:`splits`,
    which splits the given plain text into a sequence of sentences (:class:`Sentence`). The
    individual sentences are in turn subdivided into tokens / words. In most cases, this can
    be controlled by passing custom implementation of :class:`Tokenizer`.

    Moreover, subclasses may overwrite :meth:`name`, returning a unique identifier representing
    the sentence splitter's configuration.
    """

    def split(self, text: str, link_sentences: Optional[bool] = True) -> list[Sentence]:
        sentences = self._perform_split(text)
        if not link_sentences:
            return sentences

        Sentence.set_context_for_sentences(sentences)
        return sentences

    @abstractmethod
    def _perform_split(self, text: str) -> list[Sentence]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        raise NotImplementedError


class SegtokSentenceSplitter(SentenceSplitter):
    """Sentence Splitter using SegTok.

    Implementation of :class:`SentenceSplitter` using the SegTok library.

    For further details see: https://github.com/fnl/segtok
    """

    def __init__(self, tokenizer: Tokenizer = SegtokTokenizer()) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    def _perform_split(self, text: str) -> list[Sentence]:
        plain_sentences: list[str] = split_multi(text)
        sentence_offset = 0

        sentences: list[Sentence] = []
        for sentence in plain_sentences:
            try:
                sentence_offset = text.index(sentence, sentence_offset)
            except ValueError as error:
                raise AssertionError(
                    f"Can't find the sentence offset for sentence {sentence} "
                    f"starting from position {sentence_offset}"
                ) from error
            sentences.append(
                Sentence(
                    text=sentence,
                    use_tokenizer=self._tokenizer,
                    start_position=sentence_offset,
                )
            )

            sentence_offset += len(sentence)

        return sentences

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        self._tokenizer = value


class SpacySentenceSplitter(SentenceSplitter):
    """Sentence Splitter using Spacy.

    Implementation of :class:`SentenceSplitter`, using models from Spacy.

    Args:
        model: Spacy V2 model or the name of the model to load.
        tokenizer: Custom tokenizer to use (default :class:`SpacyTokenizer`)
    """

    def __init__(self, model: Union[Any, str], tokenizer: Optional[Tokenizer] = None) -> None:
        super().__init__()

        try:
            import spacy
            from spacy.language import Language
        except ImportError:
            raise ImportError(
                "Please install spacy v3.4.4 or higher before using the SpacySentenceSplitter, "
                "otherwise you can use SegtokSentenceSplitter as alternative implementation."
            )

        if isinstance(model, Language):
            self.model: Language = model
        else:
            assert isinstance(model, str)
            self.model = spacy.load(model)

        if tokenizer is None:
            self._tokenizer: Tokenizer = SpacyTokenizer("en_core_sci_sm")
        else:
            self._tokenizer = tokenizer

    def _perform_split(self, text: str) -> list[Sentence]:
        document = self.model(text)

        sentences = [
            Sentence(
                text=str(spacy_sent),
                use_tokenizer=self._tokenizer,
                start_position=spacy_sent.start_char,
            )
            for spacy_sent in document.sents
            if len(str(spacy_sent)) > 0
        ]

        return sentences

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        self._tokenizer = value

    @property
    def name(self) -> str:
        return (
            self.__class__.__name__
            + "_"
            + self.model.meta["name"]
            + "_"
            + self.model.meta["version"]
            + "_"
            + self._tokenizer.name
        )


class SciSpacySentenceSplitter(SpacySentenceSplitter):
    """Sentence splitter using the spacy model `en_core_sci_sm`.

    Convenience class to instantiate :class:`SpacySentenceSplitter` with Spacy model `en_core_sci_sm`
    for sentence splitting and :class:`SciSpacyTokenizer` as tokenizer.
    """

    def __init__(self) -> None:
        super().__init__("en_core_sci_sm", SciSpacyTokenizer())


class TagSentenceSplitter(SentenceSplitter):
    """SentenceSplitter which assumes that there is a tag within the text that is used to mark sentence boundaries.

    Implementation of :class:`SentenceSplitter` which assumes that there is a special tag within
    the text that is used to mark sentence boundaries.
    """

    def __init__(self, tag: str, tokenizer: Tokenizer = SegtokTokenizer()) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self.tag = tag

    def _perform_split(self, text: str) -> list[Sentence]:
        plain_sentences = text.split(self.tag)

        sentences = []
        last_offset = 0

        for sentence in plain_sentences:
            if len(sentence.strip()) == 0:
                continue

            sentences += [
                Sentence(
                    text=sentence,
                    use_tokenizer=self._tokenizer,
                    start_position=last_offset,
                )
            ]

            last_offset += len(sentence) + len(self.tag)

        return sentences

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        self._tokenizer = value

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.tag + "_" + self._tokenizer.name


class NewlineSentenceSplitter(TagSentenceSplitter):
    r"""Sentence Splitter using newline as boundary marker.

    Convenience class to instantiate :class:`SentenceTagSplitter` with newline ("\n") as
    sentence boundary marker.
    """

    def __init__(self, tokenizer: Tokenizer = SegtokTokenizer()) -> None:
        super().__init__(tag="\n", tokenizer=tokenizer)

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self._tokenizer.name


class NoSentenceSplitter(SentenceSplitter):
    """Sentence Splitter which treats the full text as a single Sentence.

    Implementation of :class:`SentenceSplitter` which treats the complete text as one sentence.
    """

    def __init__(self, tokenizer: Tokenizer = SegtokTokenizer()) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    def _perform_split(self, text: str) -> list[Sentence]:
        return [Sentence(text=text, use_tokenizer=self._tokenizer, start_position=0)]

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        self._tokenizer = value

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self._tokenizer.name
