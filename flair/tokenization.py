import logging

from typing import List, Callable

from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions, word_tokenizer

from flair.data import Tokenizer, Token


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
                "Please install Spacy v2.0 or better before using the Spacy tokenizer, otherwise you can use segtok_tokenizer as advanced tokenizer."
            )

        if isinstance(model, Language):
            self.model: Language = model
        elif isinstance(model, str):
            self.model: Language = spacy.load(model)
        else:
            raise AssertionError(f"Unexpected type of parameter model. Please provide a loaded spacy model or the name of the model to load.")

    def tokenize(self, text: str) -> List[Token]:
        from spacy.tokens.doc import Doc
        from spacy.tokens.token import Token as SpacyToken

        doc: Doc = self.model.make_doc(text)
        previous_token = None
        tokens: List[Token] = []
        for word in doc:
            word: SpacyToken = word
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
        multiple Japanese tokenizer such as MeCab, KyTea and SudachiPy.

        For further details see:
            https://github.com/himkt/konoha
    """

    def __init__(self, tokenizer: str):
        super(JapaneseTokenizer, self).__init__()

        if tokenizer.lower() != "mecab":
            raise NotImplementedError("Currently, MeCab is only supported.")

        try:
            import konoha
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "konoha" is not installed!')
            log.warning(
                'To use Japanese tokenizer, please first install with the following steps:'
            )
            log.warning(
                '- Install mecab with "sudo apt install mecab libmecab-dev mecab-ipadic"'
            )
            log.warning('- Install konoha with "pip install konoha[mecab]"')
            log.warning("-" * 100)
            pass

        self.tokenizer = tokenizer
        self.sentence_tokenizer = konoha.SentenceTokenizer()
        self.word_tokenizer = konoha.WordTokenizer(tokenizer)

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
