import logging
import re
import sys
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Any

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

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serializes the tokenizer's configuration to a dictionary."""
        # Base implementation should include class identity
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict[str, Any]) -> "Tokenizer":
        """Instantiates the tokenizer from a configuration dictionary."""
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tokenizer):
            return NotImplemented
        # Compare based on the dictionary representation.
        # This relies on to_dict() accurately reflecting the tokenizer's state.
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        # Hash based on an immutable representation of the dictionary.
        # Convert dict items to a sorted tuple of tuples to ensure consistent hash value.
        d = self.to_dict()
        # Ensure all values in the dict are hashable. Convert lists to tuples.
        hashable_items = []
        for k, v in sorted(d.items()):
            if isinstance(v, list):
                hashable_items.append((k, tuple(v)))
            else:
                hashable_items.append((k, v))
        return hash(tuple(hashable_items))


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

    def to_dict(self) -> dict:
        """Serialize the tokenizer's configuration to a dictionary."""
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
            "model_name": self.model.meta["name"],
            # Optionally, include model_version if crucial for functional difference
            # "model_version": self.model.meta["version"],
        }

    @classmethod
    def from_dict(cls, config: dict) -> "SpacyTokenizer":
        """Instantiate the tokenizer from a configuration dictionary."""
        try:
            import spacy
        except ImportError:
            raise ImportError("Spacy not installed. Please install spacy v3.4.4 or better to load this tokenizer.")

        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("Config dictionary for SpacyTokenizer must contain 'model_name'.")

        # Try loading the spacy model
        try:
            spacy_model = spacy.load(model_name)
        except OSError:
            log.error(
                f"Could not load spacy model '{model_name}'. "
                f"Please make sure you have downloaded it (e.g. python -m spacy download {model_name})"
            )
            raise
        return cls(spacy_model)


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

    def to_dict(self) -> dict:
        """Serialize the tokenizer's configuration to a dictionary."""
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
            "additional_split_characters": self.additional_split_characters,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "SegtokTokenizer":
        """Instantiate the tokenizer from a configuration dictionary."""
        additional_split_characters = config.get("additional_split_characters")
        return cls(additional_split_characters=additional_split_characters)


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

    def to_dict(self) -> dict:
        """Serialize the tokenizer's configuration to a dictionary."""
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "SpaceTokenizer":
        """Instantiate the tokenizer from a configuration dictionary."""
        # No specific configuration needed for instantiation
        return cls()


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
        self.sudachi_mode = sudachi_mode  # Store sudachi_mode

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

    def to_dict(self) -> dict:
        """Serialize the tokenizer's configuration to a dictionary."""
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
            "tokenizer_name": self.tokenizer,
            "sudachi_mode": self.sudachi_mode,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "JapaneseTokenizer":
        """Instantiate the tokenizer from a configuration dictionary."""
        try:
            import konoha  # Check if konoha is installed during load
        except ModuleNotFoundError:
            raise ImportError('The library "konoha" is not installed! Please install it to load this tokenizer.')

        tokenizer_name = config.get("tokenizer_name")
        sudachi_mode = config.get("sudachi_mode", "A")  # Default if missing
        if not tokenizer_name:
            raise ValueError("Config dictionary for JapaneseTokenizer must contain 'tokenizer_name'.")

        return cls(tokenizer=tokenizer_name, sudachi_mode=sudachi_mode)


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

    def to_dict(self) -> dict:
        """Serialize the tokenizer's configuration to a dictionary."""
        # For TokenizerWrapper, equality based on function identity might be more robust
        # than relying on a potentially non-serializable/reconstructable state.
        # However, to fit the pattern, we ensure its to_dict reflects this.
        # If two wrappers wrap functions with the same name and module, their dicts will be equal.
        try:
            func_name = self.tokenizer_func.__name__
            func_module = self.tokenizer_func.__module__
        except AttributeError:
            func_name = f"<unnamed_func_{id(self.tokenizer_func)}>"
            func_module = "<unknown_module>"

        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
            "function_name": func_name,
            "function_module": func_module,
            "serializable": False,  # Still mark as not truly serializable/reconstructable
        }

    @classmethod
    def from_dict(cls, config: dict) -> "TokenizerWrapper":
        """Instantiate the tokenizer from a configuration dictionary."""
        # Cannot reliably reconstruct the function from saved state.
        raise NotImplementedError(
            f"Cannot automatically reconstruct TokenizerWrapper for function '{config.get('function_name', '<unknown>')}'."
            " Please re-wrap the function manually after loading the model."
        )


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

    def to_dict(self) -> dict:
        """Serialize the tokenizer's configuration to a dictionary."""
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
            # Model is fixed, so no extra params needed for functional identity
        }

    @classmethod
    def from_dict(cls, config: dict) -> "SciSpacyTokenizer":
        """Instantiate the tokenizer from a configuration dictionary."""
        try:
            import spacy  # Check imports during load
        except ImportError:
            raise ImportError("Spacy or SciSpacy not installed. Please install them to load this tokenizer.")
        # No specific configuration needed for instantiation
        return cls()


class StaccatoTokenizer(Tokenizer):
    """
    A string-based tokenizer that splits text into tokens based on the following rules:
    - Punctuation characters are split into individual tokens
    - Sequences of numbers are kept together as single tokens
    - Kanji characters are split into individual tokens
    - Uninterrupted sequences of letters (Latin, Cyrillic, etc.) and kana are preserved as single tokens
    - Whitespace and common zero-width characters are ignored.
    """

    def __init__(self):
        super().__init__()
        # Define patterns for different character types
        # Punctuation/Symbols: Non-alphanumeric, non-whitespace, excluding common zero-width characters and BOM
        self.punctuation = r"[^\w\s\uFE00-\uFE0F\u200B-\u200D\u2060-\u206F\uFEFF]"
        self.digits = r"\d+"  # One or more digits
        self.kanji = r"[\u4e00-\u9fff]"  # Kanji characters

        # Base pattern for letters in Latin-based scripts, including diacritics
        latin_chars = r"[a-zA-Z\u00C0-\u02AF\u1E00-\u1EFF]"

        # Pattern to capture abbreviations with at least two periods (e.g., "U.S.", "e.g.")
        # This prevents matching single words at the end of a sentence (e.g., "X.").
        abbrev_segment = f"{latin_chars}{{1,3}}"
        self.abbreviations = rf"\b(?:{abbrev_segment}\.){{2,}}"

        # Unicode ranges for various alphabets and scripts
        # This includes Latin, Cyrillic, Greek, Hebrew, Arabic, Japanese Kana, Korean Hangul, etc.
        alphabets_list = [
            rf"{latin_chars}+",  # Latin
            r"[\u0400-\u04FF\u0500-\u052F]+",  # Cyrillic and Cyrillic Supplement
            r"[\u0370-\u03FF\u1F00-\u1FFF]+",  # Greek and Coptic
            r"[\u0590-\u05FF]+",  # Hebrew
            r"[\u0600-\u06FF\u0750-\u077F]+",  # Arabic
            r"[\u0E00-\u0E7F]+",  # Thai
            r"[\u3040-\u309F]+",  # Hiragana
            r"[\u30A0-\u30FF]+",  # Katakana
            r"[\uAC00-\uD7AF]+",  # Hangul (Korean)
            # Add more script ranges here if needed
        ]
        self.alphabet_pattern = "|".join(alphabets_list)

        # Combined pattern for re.findall:
        # Captures abbreviations OR letter sequences OR digit sequences OR Kanji OR punctuation/symbols
        combined_pattern = (
            f"({self.abbreviations})|({self.alphabet_pattern})|({self.digits})|({self.kanji})|({self.punctuation})"
        )
        # Pre-compile the regex for efficiency
        self.token_pattern = re.compile(combined_pattern)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the input text using re.findall to extract valid tokens.

        Args:
            text: The input text to tokenize

        Returns:
            A list of tokens (strings)
        """
        # Find all matches for the defined token patterns
        matches = self.token_pattern.findall(text)

        # re.findall returns a list of tuples, where each tuple corresponds to the capturing groups.
        # For a match, only one group will be non-empty. We extract that non-empty group.
        # Example match: ('word', '', '', '') or ('', '123', '', '') or ('', '', '好', '') or ('', '', '', '.')
        tokens: list[str] = [next(filter(None, match_tuple)) for match_tuple in matches]

        return tokens

    def to_dict(self) -> dict:
        """Serialize the tokenizer's configuration to a dictionary."""
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "StaccatoTokenizer":
        """Instantiate the tokenizer from a configuration dictionary."""
        # No specific configuration needed for instantiation
        return cls()


class NoTokenizer(Tokenizer):
    """
    A dummy tokenizer that performs no tokenization.
    It returns the original text as a single token in a list,
    or an empty list if the text is empty or whitespace.
    Useful when text is pre-tokenized or to disable tokenization.
    """

    def __init__(self) -> None:
        super().__init__()

    def tokenize(self, text: str) -> list[str]:
        """
        Returns the text as a single token if not empty/whitespace,
        otherwise returns an empty list.
        """
        stripped_text = text.strip()
        if not stripped_text:
            return []
        return [text]  # Return the original text, not the stripped version

    def to_dict(self) -> dict:
        """Serialize the tokenizer's configuration to a dictionary."""
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "NoTokenizer":
        """Instantiate the tokenizer from a configuration dictionary."""
        return cls()
