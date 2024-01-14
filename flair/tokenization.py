import logging
import sys
from abc import ABC, abstractmethod
from typing import Callable, List

from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions, word_tokenizer

log = logging.getLogger("flair")


class Tokenizer(ABC):
    r"""An abstract class representing a :class:`Tokenizer`.

    Tokenizers are used to represent algorithms and models to split plain text into
    individual tokens / words. All subclasses should overwrite :meth:`tokenize`, which
    splits the given plain text into tokens. Moreover, subclasses may overwrite
    :meth:`name`, returning a unique identifier representing the tokenizer's
    configuration.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SpacyTokenizer(Tokenizer):
    """Tokenizer using spacy under the hood.

    Args:
        model: a Spacy V2 model or the name of the model to load.
    """

    def __init__(self, model) -> None:
        super().__init__()

        try:
            import spacy
            from spacy.language import Language
        except ImportError:
            raise ImportError(
                "Please install Spacy v3.4.4 or better before using the Spacy tokenizer, "
                "otherwise you can use SegtokTokenizer as advanced tokenizer."
            )

        if isinstance(model, Language):
            self.model = model
        elif isinstance(model, str):
            self.model = spacy.load(model)
        else:
            raise AssertionError(
                "Unexpected type of parameter model. Please provide a loaded "
                "spacy model or the name of the model to load."
            )

    def tokenize(self, text: str) -> List[str]:
        from spacy.tokens.doc import Doc

        doc: Doc = self.model.make_doc(text)
        words: List[str] = []
        for word in doc:
            if len(word.text.strip()) == 0:
                continue
            words.append(word.text)
        return words

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.model.meta["name"] + "_" + self.model.meta["version"]


class SegtokTokenizer(Tokenizer):
    """Tokenizer using segtok, a third party library dedicated to rules-based Indo-European languages.

    For further details see: https://github.com/fnl/segtok
    """

    def __init__(self) -> None:
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        return SegtokTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[str]:
        words: List[str] = []

        sentences = split_single(text)
        for sentence in sentences:
            contractions = split_contractions(word_tokenizer(sentence))
            words.extend(contractions)

        words = list(filter(None, words))

        return words


class SpaceTokenizer(Tokenizer):
    """Tokenizer based on space character only."""

    def __init__(self) -> None:
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        return SpaceTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        word = ""
        index = -1
        for index, char in enumerate(text):
            if char == " ":
                if len(word) > 0:
                    tokens.append(word)

                word = ""
            else:
                word += char
        # increment for last token in sentence if not followed by whitespace
        index += 1
        if len(word) > 0:
            tokens.append(word)

        return tokens


class JapaneseTokenizer(Tokenizer):
    """Tokenizer using konoha to support popular japanese tokenizers.

    Tokenizer using konoha, a third party library which supports
    multiple Japanese tokenizer such as MeCab, Janome and SudachiPy.

    For further details see:
        https://github.com/himkt/konoha
    """

    def __init__(self, tokenizer: str, sudachi_mode: str = "A") -> None:
        super().__init__()

        available_tokenizers = ["mecab", "janome", "sudachi"]

        if tokenizer.lower() not in available_tokenizers:
            raise NotImplementedError(
                f"Currently, {tokenizer} is only supported. Supported tokenizers: {available_tokenizers}."
            )

        try:
            import konoha
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "konoha" is not installed!')
            log.warning(
                '- If you want to use MeCab, install mecab with "sudo apt install mecab libmecab-dev mecab-ipadic".'
            )
            log.warning('- Install konoha with "pip install konoha[{tokenizer_name}]"')
            log.warning('  - You can choose tokenizer from ["mecab", "janome", "sudachi"].')
            log.warning("-" * 100)
            sys.exit()

        self.tokenizer = tokenizer
        self.sentence_tokenizer = konoha.SentenceTokenizer()
        self.word_tokenizer = konoha.WordTokenizer(tokenizer, mode=sudachi_mode)

    def tokenize(self, text: str) -> List[str]:
        words: List[str] = []

        sentences = self.sentence_tokenizer.tokenize(text)
        for sentence in sentences:
            konoha_tokens = self.word_tokenizer.tokenize(sentence)
            words.extend(list(map(str, konoha_tokens)))

        return words

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.tokenizer


class TokenizerWrapper(Tokenizer):
    """Helper class to wrap tokenizer functions to the class-based tokenizer interface."""

    def __init__(self, tokenizer_func: Callable[[str], List[str]]) -> None:
        super().__init__()
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer_func(text)

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.tokenizer_func.__name__


class SciSpacyTokenizer(Tokenizer):
    """Tokenizer that uses the en_core_sci_sm Spacy model and some special heuristics.

    Implementation of :class:`Tokenizer` which uses the en_core_sci_sm Spacy model
    extended by special heuristics to consider characters such as "(", ")" "-" as
    additional token separators. The latter distinguishes this implementation from
    :class:`SpacyTokenizer`.

    Note, you if you want to use the "normal" SciSpacy tokenization just use
    :class:`SpacyTokenizer`.
    """

    def __init__(self) -> None:
        super().__init__()

        try:
            import spacy
            from spacy.lang import char_classes
        except ImportError:
            raise ImportError(
                "  Please install scispacy version 0.5.1 (recommended) or higher before using the SciSpacy tokenizer, "
                "otherwise you can use SegtokTokenizer as alternative implementation.\n"
                "  You can install scispacy (version 0.5.1) by running:\n\n"
                "     pip install scispacy==0.5.1\n\n"
                "  By default HunFlair uses the `en_core_sci_sm` model. You can install the model by running:\n\n"
                "     pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz\n\n"
                "  Note that the scispacy version and the version of the model must match to work properly!"
            )

        def combined_rule_prefixes() -> List[str]:
            """Helper function that returns the prefix pattern for the tokenizer.

            It is a helper function to accommodate spacy tests that only test prefixes.
            """
            prefix_punct = char_classes.PUNCT.replace("|", " ")

            prefixes = [
                "§",
                "%",
                "=",
                "\\+",
                *char_classes.split_chars(prefix_punct),
                *char_classes.LIST_ELLIPSES,
                *char_classes.LIST_QUOTES,
                *char_classes.LIST_CURRENCY,
                *char_classes.LIST_ICONS,
            ]
            return prefixes

        infixes = (
            char_classes.LIST_ELLIPSES
            + char_classes.LIST_ICONS
            + [
                r"x",  # added this special x character to tokenize it separately
                r"[\(\)\[\]\{\}]",  # want to split at every bracket
                r"/",  # want to split at every slash
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                rf"(?<=[{char_classes.ALPHA_LOWER}])\.(?=[{char_classes.ALPHA_UPPER}])",
                rf"(?<=[{char_classes.ALPHA}]),(?=[{char_classes.ALPHA}])",
                rf'(?<=[{char_classes.ALPHA}])[?";:=,.]*(?:{char_classes.HYPHENS})(?=[{char_classes.ALPHA}])',
                rf"(?<=[{char_classes.ALPHA}0-9])[:<>=/](?=[{char_classes.ALPHA}])",
            ]
        )

        prefix_re = spacy.util.compile_prefix_regex(combined_rule_prefixes())
        infix_re = spacy.util.compile_infix_regex(infixes)

        self.model = spacy.load(
            "en_core_sci_sm",
            disable=["tagger", "ner", "parser", "textcat", "lemmatizer"],
        )
        self.model.tokenizer.prefix_search = prefix_re.search
        self.model.tokenizer.infix_finditer = infix_re.finditer

    def tokenize(self, text: str) -> List[str]:
        sentence = self.model(text)
        words: List[str] = []
        for word in sentence:
            words.append(word.text)
        return words

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.model.meta["name"] + "_" + self.model.meta["version"]
