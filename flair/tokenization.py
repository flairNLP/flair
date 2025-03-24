import logging
import re
import sys
from abc import ABC, abstractmethod
from typing import Callable, Optional

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
    def tokenize(self, text: str) -> list[str]:
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

    def tokenize(self, text: str) -> list[str]:
        from spacy.tokens.doc import Doc

        doc: Doc = self.model.make_doc(text)
        words: list[str] = []
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

    def __init__(self, additional_split_characters: Optional[list[str]] = None) -> None:
        """Initializes the SegtokTokenizer.

        The default behavior uses simple rules to split text into tokens. If you want to ensure that certain characters
        always become their own token, you can change default behavior by setting the ``additional_split_characters``
        parameter.

        Args:
            additional_split_characters: An optional list of characters that should always be split. For instance, if
                you want to make sure that paragraph symbols always become their own token, instantiate with
                additional_split_characters = ['§']
        """
        self.additional_split_characters = additional_split_characters
        super().__init__()

    def _add_whitespace_around_symbols(self, text, symbols):
        # Build the regular expression pattern dynamically based on the provided symbols
        # This will match any character from the symbols list that doesn't have spaces around it
        symbol_pattern = f"[{re.escape(''.join(symbols))}]"

        # Add space before and after symbols, where necessary
        # Ensure that we are adding a space only if there isn't one already
        text = re.sub(r"(\S)(" + symbol_pattern + r")", r"\1 \2", text)  # Space before symbol
        text = re.sub(r"(" + symbol_pattern + r")(\S)", r"\1 \2", text)  # Space after symbol

        return text

    def tokenize(self, text: str) -> list[str]:
        if self.additional_split_characters:
            text = self._add_whitespace_around_symbols(text, self.additional_split_characters)
        return SegtokTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> list[str]:
        words: list[str] = []

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

    def tokenize(self, text: str) -> list[str]:
        return SpaceTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> list[str]:
        tokens: list[str] = []
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

    def tokenize(self, text: str) -> list[str]:
        words: list[str] = []

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

    def __init__(self, tokenizer_func: Callable[[str], list[str]]) -> None:
        super().__init__()
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text: str) -> list[str]:
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

        def combined_rule_prefixes() -> list[str]:
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

    def tokenize(self, text: str) -> list[str]:
        sentence = self.model(text)
        words: list[str] = []
        for word in sentence:
            words.append(word.text)
        return words

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.model.meta["name"] + "_" + self.model.meta["version"]


class StaccatoTokenizer(Tokenizer):
    """
    A string-based tokenizer that splits text into tokens based on the following rules:
    - Punctuation characters are split into individual tokens
    - Sequences of numbers are kept together as single tokens
    - Kanji characters are split into individual tokens
    - Uninterrupted sequences of letters (Latin, Cyrillic, etc.) and kana are preserved as single tokens
    """

    def __init__(self):
        super().__init__()
        # Define patterns for different character types
        self.punctuation = r"[^\w\s]"  # Non-alphanumeric, non-whitespace
        self.digits = r"\d+"  # One or more digits
        self.kanji = r"[\u4e00-\u9fff]"  # Kanji characters

        # Unicode ranges for various alphabets and scripts
        # This includes Latin, Cyrillic, Greek, Hebrew, Arabic, etc.
        self.alphabets = [
            r"[a-zA-Z]+",  # Latin
            r"[\u0400-\u04FF\u0500-\u052F]+",  # Cyrillic and Cyrillic Supplement
            r"[\u0370-\u03FF\u1F00-\u1FFF]+",  # Greek and Coptic
            r"[\u0590-\u05FF]+",  # Hebrew
            r"[\u0600-\u06FF\u0750-\u077F]+",  # Arabic
            r"[\u0E00-\u0E7F]+",  # Thai
            r"[\u3040-\u309F]+",  # Hiragana
            r"[\u30A0-\u30FF]+",  # Katakana
            r"[\uAC00-\uD7AF]+",  # Hangul (Korean)
            # Add more scripts as needed
        ]

        # Combined pattern for tokenization
        self.alphabet_pattern = "|".join(self.alphabets)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the input text according to the defined rules.

        Args:
            text: The input text to tokenize

        Returns:
            A list of tokens
        """
        # Create a pattern that matches:
        # 1. Punctuation characters
        # 2. Number sequences
        # 3. Kanji characters individually
        # 4. Letter sequences from various scripts
        pattern = f"({self.punctuation}|{self.digits}|{self.kanji})"

        # First split by punctuation, numbers, and kanji
        raw_tokens = []
        parts = re.split(pattern, text)

        # Filter out empty strings
        for part in parts:
            if part:
                # If part is punctuation, number, or kanji, add it directly
                if re.fullmatch(pattern, part):
                    raw_tokens.append(part)
                else:
                    # For other text, split by whitespace
                    subparts = part.split()
                    raw_tokens.extend(subparts)

        return raw_tokens
