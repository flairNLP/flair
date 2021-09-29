import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from more_itertools import stagger
from segtok.segmenter import split_single, split_multi
from segtok.tokenizer import split_contractions, word_tokenizer

from flair.data import Sentence, Token, Tokenizer

log = logging.getLogger("flair")


class SpacyTokenizer(Tokenizer):
    """
    Implementation of :class:`Tokenizer`, using models from Spacy.

    :param model a Spacy V2 model or the name of the model to load.
    """

    def __init__(self, model):
        super(SpacyTokenizer, self).__init__()

        try:
            import spacy
            from spacy.language import Language
        except ImportError:
            raise ImportError(
                "Please install Spacy v2.0 or better before using the Spacy tokenizer, "
                "otherwise you can use SegtokTokenizer as advanced tokenizer."
            )

        if isinstance(model, Language):
            self.model: Language = model
        elif isinstance(model, str):
            self.model: Language = spacy.load(model)
        else:
            raise AssertionError(f"Unexpected type of parameter model. Please provide a loaded "
                                 f"spacy model or the name of the model to load.")

    def tokenize(self, text: str) -> List[Token]:
        from spacy.tokens.doc import Doc
        from spacy.tokens.token import Token as SpacyToken

        doc: Doc = self.model.make_doc(text)
        previous_token = None
        tokens: List[Token] = []
        for word in doc:
            word: SpacyToken = word
            if len(word.text.strip()) == 0:
                continue

            token = Token(
                text=word.text, start_position=word.idx, whitespace_after=True
            )
            tokens.append(token)

            if (previous_token is not None) and (
                    token.start_pos == previous_token.start_pos + len(previous_token.text)
            ):
                previous_token.whitespace_after = False

            previous_token = token

        return tokens

    @property
    def name(self) -> str:
        return (
                self.__class__.__name__
                + "_"
                + self.model.meta["name"]
                + "_"
                + self.model.meta["version"]
        )


class SegtokTokenizer(Tokenizer):
    """
        Tokenizer using segtok, a third party library dedicated to rules-based Indo-European languages.

        For further details see: https://github.com/fnl/segtok
    """

    def __init__(self):
        super(SegtokTokenizer, self).__init__()

    def tokenize(self, text: str) -> List[Token]:
        return SegtokTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[Token]:
        tokens: List[Token] = []
        words: List[str] = []

        sentences = split_single(text)
        for sentence in sentences:
            contractions = split_contractions(word_tokenizer(sentence))
            words.extend(contractions)

        words = list(filter(None, words))

        # determine offsets for whitespace_after field
        index = text.index
        current_offset = 0
        previous_word_offset = -1
        previous_token = None
        for word in words:
            try:
                word_offset = index(word, current_offset)
                start_position = word_offset
            except:
                word_offset = previous_word_offset + 1
                start_position = (
                    current_offset + 1 if current_offset > 0 else current_offset
                )

            if word:
                token = Token(
                    text=word, start_position=start_position, whitespace_after=True
                )
                tokens.append(token)

            if (previous_token is not None) and word_offset - 1 == previous_word_offset:
                previous_token.whitespace_after = False

            current_offset = word_offset + len(word)
            previous_word_offset = current_offset - 1
            previous_token = token

        return tokens


class SpaceTokenizer(Tokenizer):
    """
        Tokenizer based on space character only.
    """

    def __init__(self):
        super(SpaceTokenizer, self).__init__()

    def tokenize(self, text: str) -> List[Token]:
        return SpaceTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[Token]:
        tokens: List[Token] = []
        word = ""
        index = -1
        for index, char in enumerate(text):
            if char == " ":
                if len(word) > 0:
                    start_position = index - len(word)
                    tokens.append(
                        Token(
                            text=word, start_position=start_position, whitespace_after=True
                        )
                    )

                word = ""
            else:
                word += char
        # increment for last token in sentence if not followed by whitespace
        index += 1
        if len(word) > 0:
            start_position = index - len(word)
            tokens.append(
                Token(text=word, start_position=start_position, whitespace_after=False)
            )

        return tokens


class JapaneseTokenizer(Tokenizer):
    """
        Tokenizer using konoha, a third party library which supports
        multiple Japanese tokenizer such as MeCab, Janome and SudachiPy.

        For further details see:
            https://github.com/himkt/konoha
    """

    def __init__(self, tokenizer: str, sudachi_mode: str = "A"):
        super(JapaneseTokenizer, self).__init__()

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
            exit()

        self.tokenizer = tokenizer
        self.sentence_tokenizer = konoha.SentenceTokenizer()
        self.word_tokenizer = konoha.WordTokenizer(tokenizer, mode=sudachi_mode)

    def tokenize(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        words: List[str] = []

        sentences = self.sentence_tokenizer.tokenize(text)
        for sentence in sentences:
            konoha_tokens = self.word_tokenizer.tokenize(sentence)
            words.extend(list(map(str, konoha_tokens)))

        # determine offsets for whitespace_after field
        index = text.index
        current_offset = 0
        previous_word_offset = -1
        previous_token = None
        for word in words:
            try:
                word_offset = index(word, current_offset)
                start_position = word_offset
            except:
                word_offset = previous_word_offset + 1
                start_position = (
                    current_offset + 1 if current_offset > 0 else current_offset
                )

            token = Token(
                text=word, start_position=start_position, whitespace_after=True
            )
            tokens.append(token)

            if (previous_token is not None) and word_offset - 1 == previous_word_offset:
                previous_token.whitespace_after = False

            current_offset = word_offset + len(word)
            previous_word_offset = current_offset - 1
            previous_token = token

        return tokens

    @property
    def name(self) -> str:
        return (
                self.__class__.__name__
                + "_"
                + self.tokenizer
        )


class TokenizerWrapper(Tokenizer):
    """
        Helper class to wrap tokenizer functions to the class-based tokenizer interface.
    """

    def __init__(self, tokenizer_func: Callable[[str], List[Token]]):
        super(TokenizerWrapper, self).__init__()
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text: str) -> List[Token]:
        return self.tokenizer_func(text)

    @property
    def name(self) -> str:
        return self.__class__.__name__ + "_" + self.tokenizer_func.__name__


class SciSpacyTokenizer(Tokenizer):
    """
        Implementation of :class:`Tokenizer` which uses the en_core_sci_sm Spacy model
        extended by special heuristics to consider characters such as "(", ")" "-" as
        additional token separators. The latter distinguishs this implementation from
        :class:`SpacyTokenizer`.

        Note, you if you want to use the "normal" SciSpacy tokenization just use
        :class:`SpacyTokenizer`.
    """

    def __init__(self):
        super(SciSpacyTokenizer, self).__init__()

        try:
            import spacy
            from spacy.lang import char_classes
        except ImportError:
            raise ImportError(
                "  Please install scispacy version 0.2.5 (recommended) or higher before using the SciSpacy tokenizer, "
                "otherwise you can use SegtokTokenizer as alternative implementation.\n"
                "  You can install scispacy (version 0.2.5) by running:\n\n"
                "     pip install scispacy==0.2.5\n\n"
                "  By default HunFlair uses the `en_core_sci_sm` model. You can install the model by running:\n\n"
                "     pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz\n\n"
                "  Note that the scispacy version and the version of the model must match to work properly!"
            )

        def combined_rule_prefixes() -> List[str]:
            """Helper function that returns the prefix pattern for the tokenizer.
               It is a helper function to accommodate spacy tests that only test
               prefixes.
            """
            prefix_punct = char_classes.PUNCT.replace("|", " ")

            prefixes = (
                    ["§", "%", "=", r"\+"]
                    + char_classes.split_chars(prefix_punct)
                    + char_classes.LIST_ELLIPSES
                    + char_classes.LIST_QUOTES
                    + char_classes.LIST_CURRENCY
                    + char_classes.LIST_ICONS
            )
            return prefixes

        infixes = (
                char_classes.LIST_ELLIPSES
                + char_classes.LIST_ICONS
                + [
                    r"×",  # added this special x character to tokenize it separately
                    r"[\(\)\[\]\{\}]",  # want to split at every bracket
                    r"/",  # want to split at every slash
                    r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                    r"(?<=[{al}])\.(?=[{au}])".format(
                        al=char_classes.ALPHA_LOWER, au=char_classes.ALPHA_UPPER
                    ),
                    r"(?<=[{a}]),(?=[{a}])".format(a=char_classes.ALPHA),
                    r'(?<=[{a}])[?";:=,.]*(?:{h})(?=[{a}])'.format(
                        a=char_classes.ALPHA, h=char_classes.HYPHENS
                    ),
                    r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=char_classes.ALPHA),
                ]
        )

        prefix_re = spacy.util.compile_prefix_regex(combined_rule_prefixes())
        infix_re = spacy.util.compile_infix_regex(infixes)

        self.model = spacy.load(
            "en_core_sci_sm", disable=["tagger", "ner", "parser", "textcat", "lemmatizer"]
        )
        self.model.tokenizer.prefix_search = prefix_re.search
        self.model.tokenizer.infix_finditer = infix_re.finditer

    def tokenize(self, text: str) -> List[Token]:
        from spacy.tokens.token import Token as SpacyToken

        sentence = self.model(text)

        previous_token = None
        tokens: List[Token] = []
        for word in sentence:
            word: SpacyToken = word
            token = Token(
                text=word.text, start_position=word.idx, whitespace_after=True
            )
            tokens.append(token)

            if (previous_token is not None) and (
                    token.start_pos == previous_token.start_pos + len(previous_token.text)
            ) and (not word.text[0].isspace()):
                previous_token.whitespace_after = False

            previous_token = token

        return tokens

    @property
    def name(self) -> str:
        return (
                self.__class__.__name__
                + "_"
                + self.model.meta["name"]
                + "_"
                + self.model.meta["version"]
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

    @abstractmethod
    def split(self, text: str) -> List[Sentence]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError()

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        raise NotImplementedError()


class SegtokSentenceSplitter(SentenceSplitter):
    """
        Implementation of :class:`SentenceSplitter` using the SegTok library.

        For further details see: https://github.com/fnl/segtok
    """

    def __init__(self, tokenizer: Tokenizer = SegtokTokenizer()):
        super(SegtokSentenceSplitter, self).__init__()
        self._tokenizer = tokenizer

    def split(self, text: str) -> List[Sentence]:
        plain_sentences: List[str] = list(split_multi(text))

        try:
            sentence_offset: Optional[int] = text.index(plain_sentences[0])
        except ValueError as error:
            raise AssertionError(f"Can't find the sentence offset for sentence {repr(plain_sentences[0])} "
                                 f"from the text's starting position") from error

        sentences: List[Sentence] = []
        for sentence, next_sentence in stagger(plain_sentences, offsets=(0, 1), longest=True):

            sentences.append(
                Sentence(
                    text=sentence,
                    use_tokenizer=self._tokenizer,
                    start_position=sentence_offset
                )
            )

            offset: int = sentence_offset + len(sentence)
            try:
                sentence_offset = text.index(next_sentence, offset) if next_sentence is not None else None
            except ValueError as error:
                raise AssertionError(f"Can't find the sentence offset for sentence {repr(sentence)} "
                                     f"starting from position {repr(offset)}") from error

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
    """
    Implementation of :class:`SentenceSplitter`, using models from Spacy.

    :param model Spacy V2 model or the name of the model to load.
    :param tokenizer Custom tokenizer to use (default :class:`SpacyTokenizer`)
    """

    def __init__(self, model: str, tokenizer: Tokenizer = None):
        super(SpacySentenceSplitter, self).__init__()

        try:
            import spacy
            from spacy.language import Language
        except ImportError:
            raise ImportError(
                "Please install spacy v2.3.2 or higher before using the SpacySentenceSplitter, "
                "otherwise you can use SegtokSentenceSplitter as alternative implementation."
            )

        if isinstance(model, Language):
            self.model: Language = model
        elif isinstance(model, str):
            self.model: Language = spacy.load(model)

        if tokenizer is None:
            self._tokenizer = SpacyTokenizer("en_core_sci_sm")
        else:
            self._tokenizer = tokenizer

    def split(self, text: str) -> List[Sentence]:
        document = self.model(text)

        sentences = [
            Sentence(
                text=str(spacy_sent),
                use_tokenizer=self._tokenizer,
                start_position=spacy_sent.start_char
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
    """
    Convenience class to instantiate :class:`SpacySentenceSplitter` with Spacy model `en_core_sci_sm`
    for sentence splitting and :class:`SciSpacyTokenizer` as tokenizer.
    """

    def __init__(self):
        super(SciSpacySentenceSplitter, self).__init__("en_core_sci_sm", SciSpacyTokenizer())


class TagSentenceSplitter(SentenceSplitter):
    """
    Implementation of :class:`SentenceSplitter` which assumes that there is a special tag within
    the text that is used to mark sentence boundaries.
    """

    def __init__(self, tag: str, tokenizer: Tokenizer = SegtokTokenizer()):
        super(TagSentenceSplitter, self).__init__()
        self._tokenizer = tokenizer
        self.tag = tag

    def split(self, text: str) -> List[Sentence]:
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
                    start_position=last_offset
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
        return (
                self.__class__.__name__
                + "_"
                + self.tag
                + "_"
                + self._tokenizer.name
        )


class NewlineSentenceSplitter(TagSentenceSplitter):
    """
    Convenience class to instantiate :class:`SentenceTagSplitter` with newline ("\n") as
    sentence boundary marker.
    """

    def __init__(self, tokenizer: Tokenizer = SegtokTokenizer()):
        super(NewlineSentenceSplitter, self).__init__(tag="\n", tokenizer=tokenizer)

    @property
    def name(self) -> str:
        return (
                self.__class__.__name__
                + "_"
                + self._tokenizer.name
        )


class NoSentenceSplitter(SentenceSplitter):
    """
    Implementation of :class:`SentenceSplitter` which treats the complete text as one sentence.
    """

    def __init__(self, tokenizer: Tokenizer = SegtokTokenizer()):
        super(NoSentenceSplitter, self).__init__()
        self._tokenizer = tokenizer

    def split(self, text: str) -> List[Sentence]:
        return [
            Sentence(
                text=text,
                use_tokenizer=self._tokenizer,
                start_position=0
            )
        ]

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
                + self._tokenizer.name
        )
