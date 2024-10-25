import copy
import json
import logging
import os
import re
import shutil
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
    cast,
)

from torch.utils.data import ConcatDataset, Dataset

import flair
from flair.data import (
    Corpus,
    FlairDataset,
    MultiCorpus,
    Relation,
    Sentence,
    Token,
    get_spans_from_bio,
)
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path, unpack_file
from flair.tokenization import Tokenizer

log = logging.getLogger("flair")


class MultiFileJsonlCorpus(Corpus):
    """This class represents a generic Jsonl corpus with multiple train, dev, and test files."""

    def __init__(
        self,
        train_files=None,
        test_files=None,
        dev_files=None,
        encoding: str = "utf-8",
        text_column_name: str = "data",
        label_column_name: str = "label",
        metadata_column_name: str = "metadata",
        label_type: str = "ner",
        use_tokenizer: Union[bool, Tokenizer] = True,
        **corpusargs,
    ) -> None:
        """Instantiates a MuliFileJsonlCorpus as, e.g., created with doccanos JSONL export.

        Note that at least one of train_files, test_files, and dev_files must contain one path.
        Otherwise, the initialization will fail.

        :param corpusargs: Additional arguments for Corpus initialization
        :param train_files: the name of the train files
        :param test_files: the name of the test files
        :param dev_files: the name of the dev files, if empty, dev data is sampled from train
        :param encoding: file encoding (default "utf-8")
        :param text_column_name: Name of the text column inside the jsonl files.
        :param label_column_name: Name of the label column inside the jsonl files.
        :param metadata_column_name: Name of the metadata column inside the jsonl files.
        :param label_type: he type of label to predict (default "ner")
        :param use_tokenizer: Specify a custom tokenizer to split the text into tokens.

        :raises RuntimeError: If no paths are given
        """
        train: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(
                        train_file,
                        text_column_name=text_column_name,
                        label_column_name=label_column_name,
                        metadata_column_name=metadata_column_name,
                        label_type=label_type,
                        encoding=encoding,
                        use_tokenizer=use_tokenizer,
                    )
                    for train_file in train_files
                ]
            )
            if train_files and train_files[0]
            else None
        )

        # read in test file if exists
        test: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(
                        test_file,
                        text_column_name=text_column_name,
                        label_column_name=label_column_name,
                        metadata_column_name=metadata_column_name,
                        label_type=label_type,
                        encoding=encoding,
                        use_tokenizer=use_tokenizer,
                    )
                    for test_file in test_files
                ]
            )
            if test_files and test_files[0]
            else None
        )

        # read in dev file if exists
        dev: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(
                        dev_file,
                        text_column_name=text_column_name,
                        label_column_name=label_column_name,
                        metadata_column_name=metadata_column_name,
                        label_type=label_type,
                        encoding=encoding,
                        use_tokenizer=use_tokenizer,
                    )
                    for dev_file in dev_files
                ]
            )
            if dev_files and dev_files[0]
            else None
        )
        super().__init__(train, dev, test, **corpusargs)


class JsonlCorpus(MultiFileJsonlCorpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        train_file: Optional[Union[str, Path]] = None,
        test_file: Optional[Union[str, Path]] = None,
        dev_file: Optional[Union[str, Path]] = None,
        encoding: str = "utf-8",
        text_column_name: str = "data",
        label_column_name: str = "label",
        metadata_column_name: str = "metadata",
        label_type: str = "ner",
        autofind_splits: bool = True,
        name: Optional[str] = None,
        use_tokenizer: Union[bool, Tokenizer] = True,
        **corpusargs,
    ) -> None:
        """Instantiates a JsonlCorpus with one file per Dataset (train, dev, and test).

        :param data_folder: Path to the folder containing the JSONL corpus
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param encoding: file encoding (default "utf-8")
        :param text_column_name: Name of the text column inside the JSONL file.
        :param label_column_name: Name of the label column inside the JSONL file.
        :param metadata_column_name: Name of the metadata column inside the JSONL file.
        :param label_type: The type of label to predict (default "ner")
        :param autofind_splits: Whether train, test and dev file should be determined automatically
        :param name: name of the Corpus see flair.data.Corpus
        :param use_tokenizer: Specify a custom tokenizer to split the text into tokens.
        """
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(
            data_folder, dev_file, test_file, train_file, autofind_splits
        )
        super().__init__(
            dev_files=[dev_file] if dev_file else [],
            train_files=[train_file] if train_file else [],
            test_files=[test_file] if test_file else [],
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            metadata_column_name=metadata_column_name,
            label_type=label_type,
            name=name if data_folder is None else str(data_folder),
            encoding=encoding,
            use_tokenizer=use_tokenizer,
            **corpusargs,
        )


class JsonlDataset(FlairDataset):
    def __init__(
        self,
        path_to_jsonl_file: Union[str, Path],
        encoding: str = "utf-8",
        text_column_name: str = "data",
        label_column_name: str = "label",
        metadata_column_name: str = "metadata",
        label_type: str = "ner",
        use_tokenizer: Union[bool, Tokenizer] = True,
    ) -> None:
        """Instantiates a JsonlDataset and converts all annotated char spans to token tags using the IOB scheme.

        The expected file format is:

        .. code-block:: json

            {
                "<text_column_name>": "<text>",
                "<label_column_name>": [[<start_char_index>, <end_char_index>, <label>],...],
                "<metadata_column_name>": [[<metadata_key>, <metadata_value>],...]
            }

        Args:
            path_to_jsonl_file: File to read
            encoding: file encoding (default "utf-8")
            text_column_name: Name of the text column
            label_column_name: Name of the label column
            metadata_column_name: Name of the metadata column
            label_type: The type of label to predict (default "ner")
            use_tokenizer: Specify a custom tokenizer to split the text into tokens.
        """
        path_to_json_file = Path(path_to_jsonl_file)

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.metadata_column_name = metadata_column_name
        self.label_type = label_type
        self.path_to_json_file = path_to_json_file

        self.sentences: list[Sentence] = []
        with path_to_json_file.open(encoding=encoding) as jsonl_fp:
            for line in jsonl_fp:
                current_line = json.loads(line)
                raw_text = current_line[text_column_name]
                current_labels = current_line[label_column_name]
                current_metadatas = current_line.get(self.metadata_column_name, [])
                current_sentence = Sentence(raw_text, use_tokenizer=use_tokenizer)

                self._add_labels_to_sentence(raw_text, current_sentence, current_labels)
                self._add_metadatas_to_sentence(current_sentence, current_metadatas)

                self.sentences.append(current_sentence)

    def _add_labels_to_sentence(self, raw_text: str, sentence: Sentence, labels: list[list[Any]]):
        # Add tags for each annotated span
        for label in labels:
            self._add_label_to_sentence(raw_text, sentence, label[0], label[1], label[2])

    def _add_label_to_sentence(self, text: str, sentence: Sentence, start: int, end: int, label: str):
        """Adds a NE label to a given sentence.

        :param text: raw sentence (with all whitespaces etc.). Is used to determine the token indices.
        :param sentence: Tokenized flair Sentence.
        :param start: Start character index of the label.
        :param end: End character index of the label.
        :param label: Label to assign to the given range.
        :return: Nothing. Changes sentence as INOUT-param
        """
        annotated_part = text[start:end]

        # Remove leading and trailing whitespaces from annotated spans
        while re.search(r"^\s", annotated_part):
            start += 1
            annotated_part = text[start:end]

        while re.search(r"\s$", annotated_part):
            end -= 1
            annotated_part = text[start:end]

        # Search start and end token index for current span
        start_idx = -1
        end_idx = -1
        for token in sentence:
            if token.start_position <= start <= token.end_position and start_idx == -1:
                start_idx = token.idx - 1

            if token.start_position <= end <= token.end_position and end_idx == -1:
                end_idx = token.idx - 1

        # If end index is not found set to last token
        if end_idx == -1:
            end_idx = sentence[-1].idx - 1

        # Throw error if indices are not valid
        if start_idx == -1 or start_idx > end_idx:
            raise ValueError(
                f"Could not create token span from char span.\n\
                    Sen: {sentence}\nStart: {start}, End: {end}, Label: {label}\n\
                        Ann: {annotated_part}\nRaw: {text}\nCo: {start_idx}, {end_idx}"
            )

        sentence[start_idx : end_idx + 1].add_label(self.label_type, label)

    def _add_metadatas_to_sentence(self, sentence: Sentence, metadatas: list[tuple[str, str]]):
        # Add metadatas for sentence
        for metadata in metadatas:
            self._add_metadata_to_sentence(sentence, metadata[0], metadata[1])

    @staticmethod
    def _add_metadata_to_sentence(sentence: Sentence, metadata_key: str, metadata_value: str):
        sentence.add_metadata(metadata_key, metadata_value)

    def is_in_memory(self) -> bool:
        # Currently all Jsonl Datasets are stored in Memory
        return True

    def __len__(self) -> int:
        """Number of sentences in the Dataset."""
        return len(self.sentences)

    def __getitem__(self, index: int) -> Sentence:
        """Returns the sentence at a given index."""
        return self.sentences[index]


class MultiFileColumnCorpus(Corpus):
    def __init__(
        self,
        column_format: dict[int, str],
        train_files=None,
        test_files=None,
        dev_files=None,
        column_delimiter: str = r"\s+",
        comment_symbol: Optional[str] = None,
        encoding: str = "utf-8",
        document_separator_token: Optional[str] = None,
        skip_first_line: bool = False,
        in_memory: bool = True,
        label_name_map: Optional[dict[str, str]] = None,
        banned_sentences: Optional[list[str]] = None,
        default_whitespace_after: int = 1,
        **corpusargs,
    ) -> None:
        r"""Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        Args:
            data_folder: base folder with the task data
            column_format: a map specifying the column format
            train_files: the name of the train files
            test_files: the name of the test files
            dev_files: the name of the dev files, if empty, dev data is sampled from train
            column_delimiter: default is to split on any separatator, but you can overwrite for instance with "\t" to split only on tabs
            comment_symbol: if set, lines that begin with this symbol are treated as comments
            encoding: file encoding (default "utf-8")
            document_separator_token: If provided, sentences that function as document boundaries are so marked
            skip_first_line: set to True if your dataset has a header line
            in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
            label_name_map: Optionally map tag names to different schema.
            banned_sentences: Optionally remove sentences from the corpus. Works only if `in_memory` is true
        """
        # get train data
        train: Optional[Dataset] = (
            ConcatDataset(
                [
                    ColumnDataset(
                        train_file,
                        column_format,
                        encoding=encoding,
                        comment_symbol=comment_symbol,
                        column_delimiter=column_delimiter,
                        banned_sentences=banned_sentences,
                        in_memory=in_memory,
                        document_separator_token=document_separator_token,
                        skip_first_line=skip_first_line,
                        label_name_map=label_name_map,
                        default_whitespace_after=default_whitespace_after,
                    )
                    for train_file in train_files
                ]
            )
            if train_files and train_files[0]
            else None
        )

        # read in test file if exists
        test: Optional[Dataset] = (
            ConcatDataset(
                [
                    ColumnDataset(
                        test_file,
                        column_format,
                        encoding=encoding,
                        comment_symbol=comment_symbol,
                        column_delimiter=column_delimiter,
                        banned_sentences=banned_sentences,
                        in_memory=in_memory,
                        document_separator_token=document_separator_token,
                        skip_first_line=skip_first_line,
                        label_name_map=label_name_map,
                        default_whitespace_after=default_whitespace_after,
                    )
                    for test_file in test_files
                ]
            )
            if test_files and test_files[0]
            else None
        )

        # read in dev file if exists
        dev: Optional[Dataset] = (
            ConcatDataset(
                [
                    ColumnDataset(
                        dev_file,
                        column_format,
                        encoding=encoding,
                        comment_symbol=comment_symbol,
                        column_delimiter=column_delimiter,
                        banned_sentences=banned_sentences,
                        in_memory=in_memory,
                        document_separator_token=document_separator_token,
                        skip_first_line=skip_first_line,
                        label_name_map=label_name_map,
                        default_whitespace_after=default_whitespace_after,
                    )
                    for dev_file in dev_files
                ]
            )
            if dev_files and dev_files[0]
            else None
        )

        super().__init__(train, dev, test, **corpusargs)


class ColumnCorpus(MultiFileColumnCorpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        column_format: dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        autofind_splits: bool = True,
        name: Optional[str] = None,
        comment_symbol="# ",
        **corpusargs,
    ) -> None:
        r"""Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        Args:
            data_folder: base folder with the task data
            column_format: a map specifying the column format
            train_file: the name of the train file
            test_file: the name of the test file
            dev_file: the name of the dev file, if None, dev data is sampled from train
            column_delimiter: default is to split on any separatator, but you can overwrite for instance with "\t" to split only on tabs
            comment_symbol: if set, lines that begin with this symbol are treated as comments
            document_separator_token: If provided, sentences that function as document boundaries are so marked
            skip_first_line: set to True if your dataset has a header line
            in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
            label_name_map: Optionally map tag names to different schema.
            banned_sentences: Optionally remove sentences from the corpus. Works only if `in_memory` is true
        """
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(
            data_folder, dev_file, test_file, train_file, autofind_splits
        )
        super().__init__(
            column_format,
            dev_files=[dev_file] if dev_file else [],
            train_files=[train_file] if train_file else [],
            test_files=[test_file] if test_file else [],
            name=name if data_folder is None else str(data_folder),
            comment_symbol=comment_symbol,
            **corpusargs,
        )


class ColumnDataset(FlairDataset):
    # special key for space after
    SPACE_AFTER_KEY = "space-after"
    # special key for feature columns
    FEATS = ["feats", "misc"]
    # special key for dependency head id
    HEAD = ["head", "head_id"]

    def __init__(
        self,
        path_to_column_file: Union[str, Path],
        column_name_map: dict[int, str],
        column_delimiter: str = r"\s+",
        comment_symbol: Optional[str] = None,
        banned_sentences: Optional[list[str]] = None,
        in_memory: bool = True,
        document_separator_token: Optional[str] = None,
        encoding: str = "utf-8",
        skip_first_line: bool = False,
        label_name_map: Optional[dict[str, str]] = None,
        default_whitespace_after: int = 1,
    ) -> None:
        r"""Instantiates a column dataset.

        Args:
            path_to_column_file: path to the file with the column-formatted data
            column_name_map: a map specifying the column format
            column_delimiter: default is to split on any separator, but you can overwrite for instance with "\t" to split only on tabs
            comment_symbol: if set, lines that begin with this symbol are treated as comments
            in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
            document_separator_token: If provided, sentences that function as document boundaries are so marked
            skip_first_line: set to True if your dataset has a header line
            label_name_map: Optionally map tag names to different schema.
            banned_sentences: Optionally remove sentences from the corpus. Works only if `in_memory` is true
        """
        path_to_column_file = Path(path_to_column_file)
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.column_delimiter = re.compile(column_delimiter)
        self.comment_symbol = comment_symbol
        self.document_separator_token = document_separator_token
        self.label_name_map = label_name_map
        self.banned_sentences = banned_sentences
        self.default_whitespace_after = default_whitespace_after

        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_column: int = 0
        self.head_id_column: Optional[int] = None
        for column in column_name_map:
            if column_name_map[column] == "text":
                self.text_column = column
            if column_name_map[column] in self.HEAD:
                self.head_id_column = column

        # determine encoding of text file
        self.encoding = encoding

        # identify which columns are spans and which are word-level
        self._identify_span_columns(column_name_map, skip_first_line)

        # now load all sentences
        with open(str(self.path_to_column_file), encoding=self.encoding) as file:
            # skip first line if to selected
            if skip_first_line:
                file.readline()

            # option 1: keep Sentence objects in memory
            if self.in_memory:
                self.sentences: list[Sentence] = []

                # pointer to previous
                previous_sentence = None
                while True:
                    # parse next sentence
                    next_sentence = self._read_next_sentence(file)

                    # quit if last sentence reached
                    if len(next_sentence) == 0:
                        break

                    sentence = self._convert_lines_to_sentence(
                        next_sentence,
                        word_level_tag_columns=self.word_level_tag_columns,
                        span_level_tag_columns=self.span_level_tag_columns,
                    )

                    if not sentence:
                        continue

                    # skip banned sentences
                    if self.banned_sentences is not None and any(
                        d in sentence.to_plain_string() for d in self.banned_sentences
                    ):
                        continue

                    # set previous and next sentence for context
                    sentence._previous_sentence = previous_sentence
                    sentence._next_sentence = None
                    if previous_sentence:
                        previous_sentence._next_sentence = sentence

                    # append parsed sentence to list in memory
                    self.sentences.append(sentence)

                    previous_sentence = sentence

                self.total_sentence_count = len(self.sentences)

            # option 2: keep source data in memory
            if not self.in_memory:
                self.sentences_raw: list[list[str]] = []

                while True:
                    # read lines for next sentence, but don't parse
                    sentence_raw = self._read_next_sentence(file)

                    # quit if last sentence reached
                    if len(sentence_raw) == 0:
                        break

                    # append raw lines for each sentence
                    self.sentences_raw.append(sentence_raw)

                self.total_sentence_count = len(self.sentences_raw)

    def _identify_span_columns(self, column_name_map, skip_first_line):
        # we make a distinction between word-level tags and span-level tags
        self.span_level_tag_columns = {}
        self.word_level_tag_columns = {self.text_column: "text"}
        # read first sentence to determine which columns are span-labels
        with open(str(self.path_to_column_file), encoding=self.encoding) as file:
            # skip first line if to selected
            if skip_first_line:
                file.readline()

            # check the first 5 sentences
            probe = []
            for _i in range(5):
                next_sentence = self._read_next_sentence(file)
                if len(next_sentence) == 0:
                    break

                sentence = self._convert_lines_to_sentence(next_sentence, word_level_tag_columns=column_name_map)
                if sentence:
                    probe.append(sentence)
                else:
                    break

            # go through all annotations and identify word- and span-level annotations
            # - if a column has at least one BIES we know it's a Span label
            # - if a column has at least one tag that is not BIOES, we know it's a Token label
            # - problem cases are columns for which we see only O - in this case we default to Span
            for sentence in probe:
                for column in column_name_map:
                    # skip assigned columns
                    if (
                        column in self.word_level_tag_columns
                        or column in self.span_level_tag_columns
                        or column == self.head_id_column
                    ):
                        continue

                    layer = column_name_map[column]

                    # the space after key is always word-levels
                    if column_name_map[column] == self.SPACE_AFTER_KEY:
                        self.word_level_tag_columns[column] = layer
                        continue

                    if layer in self.FEATS:
                        self.word_level_tag_columns[column] = layer
                        continue

                    for token in sentence:
                        # if at least one token has a BIES, we know it's a span label
                        if token.get_label(layer).value[0:2] in ["B-", "I-", "E-", "S-"]:
                            self.span_level_tag_columns[column] = layer
                            break

                        # if at least one token has a label other than BIOES, we know it's a token label
                        elif token.get_label(layer, "O").value != "O":
                            self.word_level_tag_columns[column] = layer
                            break

            # all remaining columns that are not word-level are span-level
            for column in column_name_map:
                if column not in self.word_level_tag_columns:
                    self.span_level_tag_columns[column] = column_name_map[column]

            for column in self.span_level_tag_columns:
                log.debug(f"Column {column} ({self.span_level_tag_columns[column]}) is a span-level column.")

            # for column in self.word_level_tag_columns:
            #     log.info(f"Column {column} ({self.word_level_tag_columns[column]}) is a word-level column.")

    def _read_next_sentence(self, file):
        lines = []
        line = file.readline()
        while line:
            if not line.isspace():
                lines.append(line)

            # if sentence ends, break
            if len(lines) > 0 and self.__line_completes_sentence(line):
                break

            line = file.readline()
        return lines

    def _convert_lines_to_sentence(
        self, lines, word_level_tag_columns: dict[int, str], span_level_tag_columns: Optional[dict[int, str]] = None
    ):
        token: Optional[Token] = None
        tokens: list[Token] = []
        filtered_lines = []
        comments = []
        for line in lines:
            # parse comments if possible
            if self.comment_symbol is not None and line.startswith(self.comment_symbol):
                comments.append(line)
                continue

            filtered_lines.append(line)

            # otherwise, this line is a token. parse and add to sentence
            token = self._parse_token(line, word_level_tag_columns, token)
            tokens.append(token)
        sentence: Sentence = Sentence(text=tokens)

        # check if this sentence is a document boundary
        if sentence.to_original_text() == self.document_separator_token:
            sentence.is_document_boundary = True

        # add span labels
        if span_level_tag_columns:
            for span_column in span_level_tag_columns:
                try:
                    bioes_tags = [self.column_delimiter.split(line.rstrip())[span_column] for line in filtered_lines]

                    # discard tags from tokens that are not added to the sentence
                    bioes_tags = [tag for tag, token in zip(bioes_tags, tokens) if token._internal_index is not None]
                    predicted_spans = get_spans_from_bio(bioes_tags)
                    for span_indices, score, label in predicted_spans:
                        span = sentence[span_indices[0] : span_indices[-1] + 1]
                        value = self._remap_label(label)
                        if value != "O":
                            span.add_label(span_level_tag_columns[span_column], value=value, score=score)
                except Exception:
                    pass

        for comment in comments:
            # parse relations if they are set
            if comment.startswith("# relations = "):
                relations_string = comment.strip().split("# relations = ")[1]
                for relation in relations_string.split("|"):
                    indices = relation.split(";")
                    head_start = int(indices[0])
                    head_end = int(indices[1])
                    tail_start = int(indices[2])
                    tail_end = int(indices[3])
                    label = indices[4]
                    # head and tail span indices are 1-indexed and end index is inclusive
                    relation = Relation(
                        first=sentence[head_start - 1 : head_end], second=sentence[tail_start - 1 : tail_end]
                    )
                    remapped = self._remap_label(label)
                    if remapped != "O":
                        relation.add_label(typename="relation", value=remapped)

            # parse comments such as '# id cd27886d-6895-4d02-a8df-e5fa763fa88f	domain=de-orcas'
            # to set the metadata "domain" to "de-orcas"
            for comment_row in comment.split("\t"):
                if "=" in comment_row:
                    key, value = comment_row.split("=", 1)
                    sentence.add_metadata(key, value)

        if len(sentence) > 0:
            return sentence
        return None

    def _parse_token(self, line: str, column_name_map: dict[int, str], last_token: Optional[Token] = None) -> Token:
        # get fields from line
        fields: list[str] = self.column_delimiter.split(line.rstrip())
        field_count = len(fields)
        # get head_id if exists (only in dependency parses)
        head_id = int(fields[self.head_id_column]) if self.head_id_column else None

        if last_token is None:
            start = 0
        else:
            assert last_token.end_position is not None
            start = last_token.end_position + last_token.whitespace_after

        # initialize token
        token = Token(
            fields[self.text_column],
            head_id=head_id,
            whitespace_after=self.default_whitespace_after,
            start_position=start,
        )

        # go through all columns
        for column, column_type in column_name_map.items():
            if field_count <= column:
                continue

            if column == self.text_column:
                continue

            if column == self.head_id_column:
                continue

            if column_type == self.SPACE_AFTER_KEY:
                if fields[column] == "-":
                    token.whitespace_after = 0
                continue

            # 'feats' and 'misc' column should be split into different fields
            if column_type in self.FEATS:
                for feature in fields[column].split("|"):
                    # special handling for whitespace after
                    if feature == "SpaceAfter=No":
                        token.whitespace_after = 0
                        continue

                    if "=" in feature:
                        # add each other feature as label-value pair
                        label_name, original_label_value = feature.split("=", 1)
                        label_value = self._remap_label(original_label_value)
                        if label_value != "O":
                            token.add_label(label_name, label_value)
            else:
                # get the task name (e.g. 'ner')
                label_name = column_type
                # get the label value
                label_value = self._remap_label(fields[column])
                # add label
                if label_value != "O":
                    token.add_label(label_name, label_value)

        return token

    def _remap_label(self, tag):
        # remap regular tag names
        if self.label_name_map and tag in self.label_name_map:
            tag = self.label_name_map[tag]  # for example, transforming 'PER' to 'person'
        return tag

    def __line_completes_sentence(self, line: str) -> bool:
        sentence_completed = line.isspace() or line == ""
        return sentence_completed

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self) -> int:
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        # if in memory, retrieve parsed sentence
        if self.in_memory:
            sentence = self.sentences[index]

        # else skip to position in file where sentence begins
        else:
            sentence = self._convert_lines_to_sentence(
                self.sentences_raw[index],
                word_level_tag_columns=self.word_level_tag_columns,
                span_level_tag_columns=self.span_level_tag_columns,
            )

            # set sentence context using partials TODO: pointer to dataset is really inefficient
            sentence._has_context = True
            sentence._position_in_dataset = (self, index)

        return sentence


class ONTONOTES(MultiFileColumnCorpus):
    archive_url = "https://data.mendeley.com/public-files/datasets/zmycy7t9h9/files/b078e1c4-f7a4-4427-be7f-9389967831ef/file_downloaded"

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        version: str = "v4",
        language: str = "english",
        domain: Union[None, str, list[str], dict[str, Union[None, str, list[str]]]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        assert version in ["v4", "v12"]
        if version == "v12":
            assert language == "english"
        else:
            assert language in ["english", "chinese", "arabic"]

        column_format = {0: "text", 1: "pos", 2: "ner"}

        processed_data_path = self._ensure_data_processed(base_path, language, version)

        kw = {"version": version, "language": language, "domain": domain, "processed_data_path": processed_data_path}

        dev_files = list(self._get_processed_file_paths(split="development", **kw))
        train_files = list(self._get_processed_file_paths(split="train", **kw))
        test_files = list(self._get_processed_file_paths(split="test", **kw))

        super().__init__(
            dev_files=dev_files,
            train_files=train_files,
            test_files=test_files,
            name="/".join((self.__class__.__name__, language, version)),
            column_format=column_format,
            in_memory=in_memory,
            column_delimiter="\t",
            **corpusargs,
        )

    @classmethod
    def get_available_domains(
        cls,
        base_path: Optional[Union[str, Path]] = None,
        version: str = "v4",
        language: str = "english",
        split: str = "train",
    ) -> list[str]:
        processed_data_path = cls._ensure_data_processed(base_path=base_path, language=language, version=version)

        processed_split_path = processed_data_path / "splits" / version / language / split

        return [domain_path.name for domain_path in processed_split_path.iterdir()]

    @classmethod
    def _get_processed_file_paths(
        cls,
        processed_data_path: Path,
        split: str = "train",
        version: str = "v4",
        language: str = "english",
        domain: Optional[Union[str, list[str], dict[str, Union[None, str, list[str]]]]] = None,
    ) -> Iterable[Path]:
        processed_split_path = processed_data_path / "splits" / version / language / split

        if domain is None:
            # use all domains
            assert processed_split_path.exists(), f"Processed data not found (expected at: {processed_split_path})"
            yield from sorted(filter(os.path.isfile, processed_split_path.rglob("*")))

        elif isinstance(domain, str):
            domain_path = processed_split_path / domain
            assert domain_path.exists(), f"Processed data not found (expected at: {domain_path})"
            yield from sorted(filter(os.path.isfile, domain_path.rglob("*")))

        elif isinstance(domain, list):
            for d in domain:
                domain_path = processed_split_path / d
                assert domain_path.exists(), f"Processed data not found (expected at: {domain_path})"
                yield from sorted(filter(os.path.isfile, domain_path.rglob("*")))

        else:
            assert isinstance(domain, dict)

            for d, sources in domain.items():
                domain_path = processed_split_path / d

                assert domain_path.exists(), f"Processed data not found (expected at: {domain_path})"

                if sources is None:
                    yield from sorted(domain_path.rglob("*"))

                elif isinstance(sources, str):
                    source_path = domain_path / sources
                    assert source_path.exists(), f"Processed data not found (expected at: {source_path})"
                    yield source_path

                else:
                    assert isinstance(sources, list)

                    for s in sources:
                        source_path = domain_path / s
                        assert source_path.exists(), f"Processed data not found (expected at: {source_path})"
                        yield source_path

    @classmethod
    def _ensure_data_processed(cls, base_path, language: str, version: str):
        raw_data_path = cls._ensure_data_downloaded(base_path)

        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        dataset_name = cls.__name__.lower()

        processed_data_path = base_path / dataset_name

        processed_split_path = processed_data_path / "splits" / version / language

        if not processed_split_path.exists():
            log.info(f"OntoNotes splits for {version}/{language} have not been generated yet, generating it now.")

            for split in ["train", "development", "test"]:
                log.info(f"Generating {split} split for {version}/{language}")

                raw_split_path = raw_data_path / version / "data" / split / "data" / language / "annotations"

                # iter over all domains / sources and create target files

                for raw_domain_path in raw_split_path.iterdir():
                    for raw_source_path in raw_domain_path.iterdir():
                        conll_files = sorted(raw_source_path.rglob("*gold_conll"))

                        processed_source_path = (
                            processed_split_path / split / raw_domain_path.name / raw_source_path.name
                        )
                        processed_source_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(processed_source_path, "w") as f:
                            for conll_file in conll_files:
                                for sent in cls.sentence_iterator(conll_file):
                                    if language == "arabic":
                                        trimmed_sentence = [_sent.split("#")[0] for _sent in sent["sentence"]]
                                        sent["sentence"] = trimmed_sentence
                                    for row in zip(sent["sentence"], sent["pos_tags"], sent["named_entities"]):
                                        f.write("\t".join(row) + "\n")
                                    f.write("\n")
        return processed_data_path

    @classmethod
    def _ensure_data_downloaded(cls, base_path: Optional[Union[str, Path]] = None) -> Path:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        data_folder = base_path / "conll-2012"

        if not data_folder.exists():
            unpack_file(cached_path(cls.archive_url, data_folder), data_folder.parent, "zip", False)

        return data_folder

    @classmethod
    def _process_coref_span_annotations_for_word(
        cls,
        label: str,
        word_index: int,
        clusters: defaultdict[int, list[tuple[int, int]]],
        coref_stacks: defaultdict[int, list[int]],
    ) -> None:
        """For a given coref label, add it to a currently open span(s), complete a span(s) or ignore it, if it is outside of all spans.

        This method mutates the clusters and coref_stacks dictionaries.

        Args:
            label: The coref label for this word.
            word_index : The word index into the sentence.
            clusters : A dictionary mapping cluster ids to lists of inclusive spans into the sentence.
            coref_stacks : Stacks for each cluster id to hold the start indices of open spans. Spans with the same id can be nested, which is why we collect these opening spans on a stack, e.g: [Greg, the baker who referred to [himself]_ID1 as 'the bread man']_ID1
        """
        if label != "-":
            for segment in label.split("|"):
                # The conll representation of coref spans allows spans to
                # overlap. If spans end or begin at the same word, they are
                # separated by a "|".
                if segment[0] == "(":
                    # The span begins at this word.
                    if segment[-1] == ")":
                        # The span begins and ends at this word (single word span).
                        cluster_id = int(segment[1:-1])
                        clusters[cluster_id].append((word_index, word_index))
                    else:
                        # The span is starting, so we record the index of the word.
                        cluster_id = int(segment[1:])
                        coref_stacks[cluster_id].append(word_index)
                else:
                    # The span for this id is ending, but didn't start at this word.
                    # Retrieve the start index from the document state and
                    # add the span to the clusters for this id.
                    cluster_id = int(segment[:-1])
                    start = coref_stacks[cluster_id].pop()
                    clusters[cluster_id].append((start, word_index))

    @classmethod
    def _process_span_annotations_for_word(
        cls,
        annotations: list[str],
        span_labels: list[list[str]],
        current_span_labels: list[Optional[str]],
    ) -> None:
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")

            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + cast(str, current_span_labels[annotation_index])
                span_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                span_labels[annotation_index].append("O")
            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None

    @classmethod
    def _conll_rows_to_sentence(cls, conll_rows: list[str]) -> dict:
        document_id: str
        sentence_id: int
        # The words in the sentence.
        sentence: list[str] = []
        # The pos tags of the words in the sentence.
        pos_tags: list[str] = []
        # the pieces of the parse tree.
        parse_pieces: list[Optional[str]] = []
        # The lemmatised form of the words in the sentence which
        # have SRL or word sense information.
        predicate_lemmas: list[Optional[str]] = []
        # The FrameNet ID of the predicate.
        predicate_framenet_ids: list[Optional[str]] = []
        # The sense of the word, if available.
        word_senses: list[Optional[float]] = []
        # The current speaker, if available.
        speakers: list[Optional[str]] = []

        verbal_predicates: list[str] = []
        span_labels: list[list[str]] = []
        current_span_labels: list[Optional[str]] = []

        # Cluster id -> List of (start_index, end_index) spans.
        clusters: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        # Cluster id -> List of start_indices which are open for this id.
        coref_stacks: defaultdict[int, list[int]] = defaultdict(list)

        for index, row in enumerate(conll_rows):
            conll_components = row.split()

            document_id = conll_components[0]
            sentence_id = int(conll_components[1])
            word = conll_components[3]
            pos_tag = conll_components[4]

            parse_piece: Optional[str]

            # Replace brackets in text and pos tags
            # with a different token for parse trees.
            if pos_tag != "XX" and word != "XX":
                if word == "(":
                    parse_word = "-LRB-"
                elif word == ")":
                    parse_word = "-RRB-"
                else:
                    parse_word = word
                if pos_tag == "(":
                    pos_tag = "-LRB-"
                if pos_tag == ")":
                    pos_tag = "-RRB-"
                (left_brackets, right_hand_side) = conll_components[5].split("*")
                # only keep ')' if there are nested brackets with nothing in them.
                right_brackets = right_hand_side.count(")") * ")"
                parse_piece = f"{left_brackets} ({pos_tag} {parse_word}) {right_brackets}"
            else:
                # There are some bad annotations in the CONLL data.
                # They contain no information, so to make this explicit,
                # we just set the parse piece to be None which will result
                # in the overall parse tree being None.
                parse_piece = None

            lemmatised_word = conll_components[6]
            framenet_id = conll_components[7]
            word_sense = conll_components[8]
            speaker = conll_components[9]

            if not span_labels:
                # If this is the first word in the sentence, create
                # empty lists to collect the NER and SRL BIO labels.
                # We can't do this upfront, because we don't know how many
                # components we are collecting, as a sentence can have
                # variable numbers of SRL frames.
                span_labels = [[] for _ in conll_components[10:-1]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[10:-1]]

            cls._process_span_annotations_for_word(conll_components[10:-1], span_labels, current_span_labels)

            # If any annotation marks this word as a verb predicate,
            # we need to record its index. This also has the side effect
            # of ordering the verbal predicates by their location in the
            # sentence, automatically aligning them with the annotations.
            word_is_verbal_predicate = any("(V" in x for x in conll_components[11:-1])
            if word_is_verbal_predicate:
                verbal_predicates.append(word)

            cls._process_coref_span_annotations_for_word(conll_components[-1], index, clusters, coref_stacks)

            sentence.append(word)
            pos_tags.append(pos_tag)
            parse_pieces.append(parse_piece)
            predicate_lemmas.append(lemmatised_word if lemmatised_word != "-" else None)
            predicate_framenet_ids.append(framenet_id if framenet_id != "-" else None)
            word_senses.append(float(word_sense) if word_sense != "-" else None)
            speakers.append(speaker if speaker != "-" else None)

        named_entities = span_labels[0]
        srl_frames = list(zip(verbal_predicates, span_labels[1:]))

        # this would not be reached if parse_pieces contained None, hence the cast
        parse_tree = "".join(cast(list[str], parse_pieces)) if all(parse_pieces) else None

        coref_span_tuples = {(cluster_id, span) for cluster_id, span_list in clusters.items() for span in span_list}
        return {
            "document_id": document_id,
            "sentence_id": sentence_id,
            "sentence": sentence,
            "pos_tags": pos_tags,
            "parse_tree": parse_tree,
            "predicate_lemmas": predicate_lemmas,
            "predicate_framenet_ids": predicate_framenet_ids,
            "word_senses": word_senses,
            "speakers": speakers,
            "named_entities": named_entities,
            "srl_frames": srl_frames,
            "coref_span_tuples": coref_span_tuples,
        }

    @classmethod
    def dataset_document_iterator(cls, file_path: Union[Path, str]) -> Iterator[list[dict]]:
        """An iterator over CONLL formatted files which yields documents, regardless of the number of document annotations in a particular file.

        This is useful for conll data which has been preprocessed, such
        as the preprocessing which takes place for the 2012 CONLL
        Coreference Resolution task.
        """
        with open(file_path, encoding="utf8") as open_file:
            conll_rows = []
            document: list[dict] = []
            for line in open_file:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        document.append(cls._conll_rows_to_sentence(conll_rows))
                        conll_rows = []
                if line.startswith("#end document"):
                    yield document
                    document = []
            if document:
                # Collect any stragglers or files which might not
                # have the '#end document' format for the end of the file.
                yield document

    @classmethod
    def sentence_iterator(cls, file_path: Union[Path, str]) -> Iterator:
        """An iterator over the sentences in an individual CONLL formatted file."""
        for document in cls.dataset_document_iterator(file_path):
            yield from document


class CONLL_03(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        column_format={0: "text", 1: "pos", 3: "ner"},
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the CoNLL-03 corpus.

        This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put the eng.testa, .testb, .train
        files in a folder called 'conll_03'. Then set the base_path parameter in the constructor to the path to the
        parent directory where the conll_03 folder resides.
        If using entity linking, the conll03 dateset is reduced by about 20 Documents, which are not part of the yago dataset.
        :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03' folder) on your machine
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'WARNING: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super().__init__(
            data_folder,
            column_format=column_format,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )


class CONLL_03_GERMAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the CoNLL-03 corpus for German.

        This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put the respective files in a folder called
        'conll_03_german'. Then set the base_path parameter in the constructor to the path to the parent directory where
        the conll_03_german folder resides.
        :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03_german' folder) on your machine
        word lemmas, POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "lemma", 2: "pos", 3: "np", 4: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'WARNING: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )


class CONLL_03_DUTCH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the CoNLL-03 corpus for Dutch.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        conll_02_path = "https://www.clips.uantwerpen.be/conll2002/ner/data/"

        # download files if not present locally
        cached_path(f"{conll_02_path}ned.testa", data_folder / "raw")
        cached_path(f"{conll_02_path}ned.testb", data_folder / "raw")
        cached_path(f"{conll_02_path}ned.train", data_folder / "raw")

        # we need to slightly modify the original files by adding some new lines after document separators
        train_data_file = data_folder / "train.txt"
        if not train_data_file.is_file():
            self.__offset_docstarts(data_folder / "raw" / "ned.train", data_folder / "train.txt")
            self.__offset_docstarts(data_folder / "raw" / "ned.testa", data_folder / "dev.txt")
            self.__offset_docstarts(data_folder / "raw" / "ned.testb", data_folder / "test.txt")

        super().__init__(
            data_folder,
            columns,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="test.txt",
            encoding="latin-1",
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )

    @staticmethod
    def __offset_docstarts(file_in: Union[str, Path], file_out: Union[str, Path]):
        with open(file_in, encoding="latin-1") as f:
            lines = f.readlines()
        with open(file_out, "w", encoding="latin-1") as f:
            for line in lines:
                f.write(line)
                if line.startswith("-DOCSTART-"):
                    f.write("\n")


class CONLL_03_SPANISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the CoNLL-03 corpus for Spanish.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        conll_02_path = "https://www.clips.uantwerpen.be/conll2002/ner/data/"
        cached_path(f"{conll_02_path}esp.testa", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}esp.testb", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}esp.train", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="latin-1",
            in_memory=in_memory,
            **corpusargs,
        )


class CONLL_2000(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the CoNLL-2000 corpus for English chunking.

        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "np"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        conll_2000_path = "https://www.clips.uantwerpen.be/conll2000/chunking/"
        data_file = flair.cache_root / "datasets" / dataset_name / "train.txt"
        if not data_file.is_file():
            cached_path(f"{conll_2000_path}train.txt.gz", Path("datasets") / dataset_name)
            cached_path(f"{conll_2000_path}test.txt.gz", Path("datasets") / dataset_name)
            import gzip

            with (
                gzip.open(flair.cache_root / "datasets" / dataset_name / "train.txt.gz", "rb") as f_in,
                open(
                    flair.cache_root / "datasets" / dataset_name / "train.txt",
                    "wb",
                ) as f_out,
            ):
                shutil.copyfileobj(f_in, f_out)
            with (
                gzip.open(flair.cache_root / "datasets" / dataset_name / "test.txt.gz", "rb") as f_in,
                open(
                    flair.cache_root / "datasets" / dataset_name / "test.txt",
                    "wb",
                ) as f_out,
            ):
                shutil.copyfileobj(f_in, f_out)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            **corpusargs,
        )


class WNUT_17(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        wnut_path = "https://noisy-text.github.io/2017/files/"
        cached_path(f"{wnut_path}wnut17train.conll", Path("datasets") / dataset_name)
        cached_path(f"{wnut_path}emerging.dev.conll", Path("datasets") / dataset_name)
        cached_path(f"{wnut_path}emerging.test.annotated", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            **corpusargs,
        )


class FEWNERD(ColumnCorpus):
    def __init__(
        self,
        setting: str = "supervised",
        **corpusargs,
    ) -> None:
        assert setting in ["supervised", "inter", "intra"]

        base_path = flair.cache_root / "datasets"
        self.dataset_name = self.__class__.__name__.lower()
        self.data_folder = base_path / self.dataset_name / setting
        self.bio_format_data = base_path / self.dataset_name / setting / "bio_format"

        if not self.data_folder.exists():
            self._download(setting=setting)

        if not self.bio_format_data.exists():
            self._generate_splits(setting)

        super().__init__(
            self.bio_format_data,
            column_format={0: "text", 1: "ner"},
            **corpusargs,
        )

    def _download(self, setting):
        _URLs = {
            "supervised": "https://cloud.tsinghua.edu.cn/f/09265750ae6340429827/?dl=1",
            "intra": "https://cloud.tsinghua.edu.cn/f/a0d3efdebddd4412b07c/?dl=1",
            "inter": "https://cloud.tsinghua.edu.cn/f/165693d5e68b43558f9b/?dl=1",
        }

        log.info(f"FewNERD ({setting}) dataset not found, downloading.")
        dl_path = _URLs[setting]
        dl_dir = cached_path(dl_path, Path("datasets") / self.dataset_name / setting)

        if setting not in os.listdir(self.data_folder):
            import zipfile

            from tqdm import tqdm

            log.info("FewNERD dataset has not been extracted yet, extracting it now. This might take a while.")
            with zipfile.ZipFile(dl_dir, "r") as zip_ref:
                for f in tqdm(zip_ref.namelist()):
                    if f.endswith("/"):
                        os.makedirs(self.data_folder / f)
                    else:
                        zip_ref.extract(f, path=self.data_folder)

    def _generate_splits(self, setting):
        log.info(
            f"FewNERD splits for {setting} have not been parsed into BIO format, parsing it now. This might take a while."
        )
        os.mkdir(self.bio_format_data)
        for split in os.listdir(self.data_folder / setting):
            with open(self.data_folder / setting / split) as source, open(self.bio_format_data / split, "w") as target:
                previous_tag = None
                for line in source:
                    if line == "" or line == "\n":
                        target.write("\n")
                    else:
                        token, tag = line.split("\t")
                        tag = tag.replace("\n", "")
                        if tag == "O":
                            target.write(token + "\t" + tag + "\n")
                        elif previous_tag != tag and tag != "O":
                            target.write(token + "\t" + "B-" + tag + "\n")
                        elif previous_tag == tag and tag != "O":
                            target.write(token + "\t" + "I-" + tag + "\n")
                        previous_tag = tag


class BIOSCOPE(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "tag"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        bioscope_path = (
            "https://raw.githubusercontent.com/whoisjones/BioScopeSequenceLabelingData/master/sequence_labeled/"
        )
        cached_path(f"{bioscope_path}output.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            train_file="output.txt",
            **corpusargs,
        )


class NER_ARABIC_ANER(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize a preprocessed version of the Arabic Named Entity Recognition Corpus (ANERCorp).

        The dataset is downloaded from http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp
        Column order is swapped
        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        anercorp_path = "https://megantosh.s3.eu-central-1.amazonaws.com/ANERcorp/"
        # cached_path(f"{anercorp_path}test.txt", Path("datasets") / dataset_name)
        cached_path(f"{anercorp_path}train.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_ARABIC_AQMAR(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize a preprocessed and modified version of the American and Qatari Modeling of Arabic (AQMAR) dataset.

        The dataset is downloaded from  http://www.cs.cmu.edu/~ark/AQMAR/

        - Modifications from original dataset: Miscellaneous tags (MIS0, MIS1, MIS2, MIS3) are merged to one tag "MISC" as these categories deviate across the original dataset
        - The 28 original Wikipedia articles are merged into a single file containing the articles in alphabetical order

        The first time you call this constructor it will automatically download the dataset.

        This dataset is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License.
        please cite: "Behrang Mohit, Nathan Schneider, Rishav Bhowmick, Kemal Oflazer, and Noah A. Smith (2012),
        Recall-Oriented Learning of Named Entities in Arabic Wikipedia. Proceedings of EACL."

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        aqmar_path = "https://megantosh.s3.eu-central-1.amazonaws.com/AQMAR/"
        # cached_path(f"{anercorp_path}test.txt", Path("datasets") / dataset_name)
        cached_path(f"{aqmar_path}train.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_BASQUE(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ner_basque_path = "http://ixa2.si.ehu.eus/eiec/"
        data_path = flair.cache_root / "datasets" / dataset_name
        data_file = data_path / "named_ent_eu.train"
        if not data_file.is_file():
            cached_path(f"{ner_basque_path}/eiec_v1.0.tgz", Path("datasets") / dataset_name)

            with tarfile.open(
                flair.cache_root / "datasets" / dataset_name / "eiec_v1.0.tgz",
                "r:gz",
            ) as f_in:
                corpus_files = (
                    "eiec_v1.0/named_ent_eu.train",
                    "eiec_v1.0/named_ent_eu.test",
                )
                for corpus_file in corpus_files:
                    f_in.extract(corpus_file, data_path)
                    shutil.move(f"{data_path}/{corpus_file}", data_path)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_CHINESE_WEIBO(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the WEIBO_NER corpus.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        weiboNER_conll_path = "https://raw.githubusercontent.com/87302380/WEIBO_NER/main/data/"
        cached_path(
            f"{weiboNER_conll_path}weiboNER_2nd_conll_format.train",
            Path("datasets") / dataset_name,
        )
        cached_path(
            f"{weiboNER_conll_path}weiboNER_2nd_conll_format.test",
            Path("datasets") / dataset_name,
        )
        cached_path(
            f"{weiboNER_conll_path}weiboNER_2nd_conll_format.dev",
            Path("datasets") / dataset_name,
        )

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            in_memory=in_memory,
            train_file="weiboNER_2nd_conll_format.train",
            test_file="weiboNER_2nd_conll_format.test",
            dev_file="weiboNER_2nd_conll_format.dev",
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_DANISH_DANE(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 3: "pos", 9: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        data_path = flair.cache_root / "datasets" / dataset_name
        train_data_file = data_path / "ddt.train.conllu"
        if not train_data_file.is_file():
            temp_file = cached_path(
                "https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip",
                Path("datasets") / dataset_name,
            )
            from zipfile import ZipFile

            with ZipFile(temp_file, "r") as zip_file:
                zip_file.extractall(path=data_path)

            # Remove CoNLL-U meta information in the last column
            for part in ["train", "dev", "test"]:
                lines = []
                data_file = f"ddt.{part}.conllu"
                with open(data_path / data_file) as file:
                    for line in file:
                        if line.startswith("#") or line == "\n":
                            lines.append(line)
                        lines.append(line.replace("name=", "").replace("|SpaceAfter=No", ""))

                with open(data_path / data_file, "w") as file:
                    file.writelines(lines)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            comment_symbol="#",
            **corpusargs,
        )


class NER_ENGLISH_MOVIE_SIMPLE(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the eng corpus of the MIT Movie Corpus.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        # column format
        columns = {0: "ner", 1: "text"}

        # dataset name
        dataset_name = self.__class__.__name__.lower()

        # data folder: default dataset folder is the cache root
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        data_folder = base_path / dataset_name

        # download data if necessary
        mit_movie_path = "https://groups.csail.mit.edu/sls/downloads/movie/"
        train_file = "engtrain.bio"
        test_file = "engtest.bio"
        cached_path(f"{mit_movie_path}{train_file}", Path("datasets") / dataset_name)
        cached_path(f"{mit_movie_path}{test_file}", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            train_file=train_file,
            test_file=test_file,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_MOVIE_COMPLEX(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the trivia10k13 corpus of the MIT Movie Corpus.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        # column format
        columns = {0: "ner", 1: "text"}

        # dataset name
        dataset_name = self.__class__.__name__.lower()

        # data folder: default dataset folder is the cache root
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        data_folder = base_path / dataset_name

        # download data if necessary
        mit_movie_path = "https://groups.csail.mit.edu/sls/downloads/movie/"
        train_file = "trivia10k13train.bio"
        test_file = "trivia10k13test.bio"
        cached_path(f"{mit_movie_path}{train_file}", Path("datasets") / dataset_name)
        cached_path(f"{mit_movie_path}{test_file}", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            train_file=train_file,
            test_file=test_file,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_SEC_FILLINGS(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize corpus of SEC-fillings annotated with English NER tags.

        See paper "Domain Adaption of Named Entity Recognition to Support Credit Risk Assessment" by Alvarado et al, 2015: https://aclanthology.org/U15-1010/

        Args:
            base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03' folder) on your machine
            in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        SEC_FILLINGS_Path = "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/SEC-filings/CONLL-format/data/"
        cached_path(f"{SEC_FILLINGS_Path}test/FIN3.txt", Path("datasets") / dataset_name)
        cached_path(f"{SEC_FILLINGS_Path}train/FIN5.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            in_memory=in_memory,
            train_file="FIN5.txt",
            test_file="FIN3.txt",
            skip_first_line=True,
            **corpusargs,
        )


class NER_ENGLISH_RESTAURANT(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the MIT Restaurant corpus.

        The corpus will be downloaded from https://groups.csail.mit.edu/sls/downloads/restaurant/.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        mit_restaurants_path = "https://megantosh.s3.eu-central-1.amazonaws.com/MITRestoCorpus/"
        cached_path(f"{mit_restaurants_path}test.txt", Path("datasets") / dataset_name)
        cached_path(f"{mit_restaurants_path}train.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="latin-1",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_STACKOVERFLOW(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the STACKOVERFLOW_NER corpus.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        """
        The Datasets are represented in the Conll format.
           In this format each line of the Dataset is in the following format:
           <word>+"\t"+<NE>"\t"+<word>+"\t"<markdown>
           The end of sentence is marked with an empty line.
           In each line NE represented the human annotated named entity
           and <markdown> represented the code tags provided by the users who wrote the posts.
           """
        # column format
        columns = {0: "word", 1: "ner", 3: "markdown"}

        # entity_mapping
        entity_mapping = {
            "Library_Function": "Function",
            "Function_Name": "Function",
            "Class_Name": "Class",
            "Library_Class": "Class",
            "Organization": "Website",
            "Library_Variable": "Variable",
            "Variable_Name": "Variable",
            "Error_Name": "O",
            "Keyboard_IP": "O",
            "Value": "O",
            "Output_Block": "O",
        }

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        STACKOVERFLOW_NER_path = "https://raw.githubusercontent.com/jeniyat/StackOverflowNER/master/resources/annotated_ner_data/StackOverflow/"

        # data validation
        banned_sentences = [
            "code omitted for annotation",
            "omitted for annotation",
            "CODE_BLOCK :",
            "OP_BLOCK :",
            "Question_URL :",
            "Question_ID :",
        ]

        files = ["train", "test", "dev"]

        for file in files:
            questions = 0
            answers = 0

            cached_path(f"{STACKOVERFLOW_NER_path}{file}.txt", Path("datasets") / dataset_name)
            with (data_folder / (file + ".txt")).open(encoding="utf-8") as fin:
                for line in fin:
                    if line.startswith("Question_ID"):
                        questions += 1

                    if line.startswith("Answer_to_Question_ID"):
                        answers += 1
            log.info(f"File {file} has {questions} questions and {answers} answers.")

        super().__init__(
            data_folder,
            columns,
            train_file="train.txt",
            test_file="test.txt",
            dev_file="dev.txt",
            encoding="utf-8",
            banned_sentences=banned_sentences,
            in_memory=in_memory,
            label_name_map=entity_mapping,
            **corpusargs,
        )


class NER_ENGLISH_TWITTER(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the twitter_ner corpus.

        The corpus will be downoaded from https://raw.githubusercontent.com/aritter/twitter_nlp/master/data/annotated/ner.txt.
        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        twitter_ner_path = "https://raw.githubusercontent.com/aritter/twitter_nlp/master/data/annotated/"
        cached_path(f"{twitter_ner_path}ner.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="latin-1",
            train_file="ner.txt",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_PERSON(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
    ) -> None:
        """Initialize the PERSON_NER corpus for person names.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        conll_path = "https://raw.githubusercontent.com/das-sudeshna/genid/master/"

        # download files if not present locallys
        cached_path(f"{conll_path}conll-g.conll", data_folder / "raw")
        cached_path(f"{conll_path}ieer-g.conll", data_folder / "raw")
        cached_path(f"{conll_path}textbook-g.conll", data_folder / "raw")
        cached_path(f"{conll_path}wiki-g.conll", data_folder / "raw")

        self.__concatAllFiles(data_folder)

        super().__init__(data_folder, columns, in_memory=in_memory, train_file="bigFile.conll")

    @staticmethod
    def __concatAllFiles(data_folder):
        arr = os.listdir(data_folder / "raw")

        with open(data_folder / "bigFile.conll", "w") as outfile:
            for fname in arr:
                with open(data_folder / "raw" / fname) as infile:
                    outfile.write(infile.read())


class NER_ENGLISH_WEBPAGES(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the WEBPAGES_NER corpus.

        The corpus was introduced in the paper "Design Challenges and Misconceptions in Named Entity Recognition" by Ratinov and Roth (2009): https://aclanthology.org/W09-1119/.
        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "ner", 5: "text"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        if not os.path.isfile(data_folder / "webpages_ner.txt"):
            #     # download zip
            tar_file = "https://cogcomp.seas.upenn.edu/Data/NERWebpagesColumns.tgz"
            webpages_ner_path = cached_path(tar_file, Path("datasets") / dataset_name)
            with tarfile.open(webpages_ner_path) as tf:
                tf.extractall(data_folder)
        outputfile = os.path.abspath(data_folder)

        # merge the files in one as the zip is containing multiples files

        with open(outputfile / data_folder / "webpages_ner.txt", "w+") as outfile:
            for files in os.walk(outputfile):
                f = files[1]
                ff = os.listdir(outputfile / data_folder / f[-1])
                for _i, file in enumerate(ff):
                    if file.endswith(".gold"):
                        with open(
                            outputfile / data_folder / f[-1] / file,
                            "r+",
                            errors="replace",
                        ) as infile:
                            content = infile.read()
                        outfile.write(content)
                break

        super().__init__(
            data_folder,
            columns,
            train_file="webpages_ner.txt",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_WNUT_2020(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the WNUT_2020_NER corpus.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        github_url = "https://github.com/jeniyat/WNUT_2020_NER/archive/master.zip"

        for sample in ["train", "test", "dev"]:
            sample_file = data_folder / (sample + ".txt")
            if not sample_file.is_file():
                zip_path = cached_path(f"{github_url}", Path("datasets") / dataset_name)

                # unzip the downloaded repo and merge the train, dev and test datasets
                unpack_file(zip_path, data_folder, "zip", False)  # unzipped folder name: WNUT_2020_NER-master

                if sample == "test":
                    file_path = data_folder / Path("WNUT_2020_NER-master/data/" + sample + "_data_2020/Conll_Format/")
                else:
                    file_path = data_folder / Path("WNUT_2020_NER-master/data/" + sample + "_data/Conll_Format/")
                filenames = os.listdir(file_path)
                with open(data_folder / (sample + ".txt"), "w") as outfile:
                    for fname in filenames:
                        with open(file_path / fname) as infile:
                            lines = infile.read()
                            outfile.write(lines)

                shutil.rmtree(str(data_folder / "WNUT_2020_NER-master"))  # clean up when done

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_ENGLISH_WIKIGOLD(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the wikigold corpus.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        wikigold_ner_path = "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/wikigold/CONLL-format/data/"
        cached_path(f"{wikigold_ner_path}wikigold.conll.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            in_memory=in_memory,
            train_file="wikigold.conll.txt",
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_FINNISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ner_finnish_path = "https://raw.githubusercontent.com/mpsilfve/finer-data/master/data/digitoday."
        cached_path(f"{ner_finnish_path}2014.train.csv", Path("datasets") / dataset_name)
        cached_path(f"{ner_finnish_path}2014.dev.csv", Path("datasets") / dataset_name)
        cached_path(f"{ner_finnish_path}2015.test.csv", Path("datasets") / dataset_name)

        self._remove_lines_without_annotations(data_file=Path(data_folder / "digitoday.2015.test.csv"))

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            skip_first_line=True,
            **corpusargs,
        )

    def _remove_lines_without_annotations(self, data_file: Union[str, Path]):
        with open(data_file) as f:
            lines = f.readlines()
        with open(data_file, "w") as f:
            for line in lines:
                if len(line.split()) != 1:
                    f.write(line)


class NER_GERMAN_BIOFID(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "lemma", 2: "pos", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        biofid_path = "https://raw.githubusercontent.com/texttechnologylab/BIOfid/master/BIOfid-Dataset-NER/"
        cached_path(f"{biofid_path}train.conll", Path("datasets") / dataset_name)
        cached_path(f"{biofid_path}dev.conll", Path("datasets") / dataset_name)
        cached_path(f"{biofid_path}test.conll", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_GERMAN_EUROPARL(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the EUROPARL_NER_GERMAN corpus.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training. Not recommended due to heavy RAM usage.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "lemma", 2: "pos", 3: "np", 4: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        europarl_ner_german_path = "https://nlpado.de/~sebastian/software/ner/"
        cached_path(
            f"{europarl_ner_german_path}ep-96-04-15.conll",
            Path("datasets") / dataset_name,
        )
        cached_path(
            f"{europarl_ner_german_path}ep-96-04-16.conll",
            Path("datasets") / dataset_name,
        )

        self._add_IOB_tags(
            data_file=Path(data_folder / "ep-96-04-15.conll"),
            encoding="latin-1",
            ner_column=4,
        )
        self._add_IOB_tags(
            data_file=Path(data_folder / "ep-96-04-16.conll"),
            encoding="latin-1",
            ner_column=4,
        )

        super().__init__(
            data_folder,
            columns,
            encoding="latin-1",
            in_memory=in_memory,
            train_file="ep-96-04-16.conll",
            test_file="ep-96-04-15.conll",
            **corpusargs,
        )

    def _add_IOB_tags(self, data_file: Union[str, Path], encoding: str = "utf8", ner_column: int = 1):
        """Function that adds IOB tags if only chunk names are provided.

        e.g. words are tagged PER instead of B-PER or I-PER. Replaces '0' with 'O' as the no-chunk tag since ColumnCorpus expects
        the letter 'O'. Additionally it removes lines with no tags in the data file and can also
        be used if the data is only partially IOB tagged.

        Parameters
        ----------
        data_file : Union[str, Path]
            Path to the data file.
        encoding : str, optional
            Encoding used in open function. The default is "utf8".
        ner_column : int, optional
            Specifies the ner-tagged column. The default is 1 (the second column).
        """

        def add_I_prefix(current_line: list[str], ner: int, tag: str):
            for i in range(len(current_line)):
                if i == 0:
                    f.write(line_list[i])
                elif i == ner:
                    f.write(" I-" + tag)
                else:
                    f.write(" " + current_line[i])
            f.write("\n")

        with open(file=data_file, encoding=encoding) as f:
            lines = f.readlines()
        with open(file=data_file, mode="w", encoding=encoding) as f:
            pred = "O"  # remembers ner tag of predecessing line
            for line in lines:
                line_list = line.split()
                if len(line_list) > 2:  # word with tags
                    ner_tag = line_list[ner_column]
                    if ner_tag in ["0", "O"]:  # no chunk
                        for i in range(len(line_list)):
                            if i == 0:
                                f.write(line_list[i])
                            elif i == ner_column:
                                f.write(" O")
                            else:
                                f.write(" " + line_list[i])
                        f.write("\n")
                        pred = "O"
                    elif "-" not in ner_tag:  # no IOB tags
                        if pred == "O":  # found a new chunk
                            add_I_prefix(line_list, ner_column, ner_tag)
                            pred = ner_tag
                        else:  # found further part of chunk or new chunk directly after old chunk
                            add_I_prefix(line_list, ner_column, ner_tag)
                            pred = ner_tag
                    else:  # line already has IOB tag (tag contains '-')
                        f.write(line)
                        pred = ner_tag.split("-")[1]
                elif len(line_list) == 0:  # empty line
                    f.write("\n")
                    pred = "O"


class NER_GERMAN_LEGAL(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the LER_GERMAN (Legal Entity Recognition) corpus.

        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training. Not recommended due to heavy RAM usage.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ler_path = "https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/"

        for split in ["train", "dev", "test"]:
            cached_path(f"{ler_path}ler_{split}.conll", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            train_file="ler_train.conll",
            dev_file="ler_dev.conll",
            test_file="ler_test.conll",
            **corpusargs,
        )


class NER_GERMAN_GERMEVAL(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the GermEval NER corpus for German.

        This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://sites.google.com/site/germeval2014ner/data and put it into some folder.
        Then point the base_path parameter in the constructor to this folder
        :param base_path: Path to the GermEval corpus on your machine
        :param in_memory:If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            # create folder
            os.makedirs(data_folder)

            # download dataset
            import gdown

            gdown.download(
                url="https://drive.google.com/uc?id={}".format("1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P"),
                output=str(data_folder / "train.tsv"),
            )
            gdown.download(
                url="https://drive.google.com/uc?id={}".format("1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH"),
                output=str(data_folder / "test.tsv"),
            )
            gdown.download(
                url="https://drive.google.com/uc?id={}".format("1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm"),
                output=str(data_folder / "dev.tsv"),
            )

        super().__init__(
            data_folder,
            columns,
            comment_symbol="#",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_GERMAN_POLITICS(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        column_delimiter: str = r"\s+",
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize corpus with Named Entity Model for German Politics (NEMGP).

        data from https://www.thomas-zastrow.de/nlp/.

        The first time you call this constructor it will automatically download the
        dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download and parse data if necessary
        german_politics_path = "https://www.thomas-zastrow.de/nlp/nemgp_trainingdata_01.txt.zip"
        corpus_file_name = "nemgp_trainingdata_01.txt"
        parsed_dataset = data_folder / "raw" / corpus_file_name

        if not parsed_dataset.exists():
            german_politics_zip = cached_path(f"{german_politics_path}", Path("datasets") / dataset_name / "raw")
            unpack_file(german_politics_zip, data_folder / "raw", "zip", False)
            self._convert_to_column_corpus(parsed_dataset)

        # create train test dev if not exist
        train_dataset = data_folder / "train.txt"
        if not train_dataset.exists():
            self._create_datasets(parsed_dataset, data_folder)

        super().__init__(
            data_folder,
            columns,
            column_delimiter=column_delimiter,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="test.txt",
            encoding="utf-8",
            in_memory=in_memory,
            **corpusargs,
        )

    def _convert_to_column_corpus(self, data_file: Union[str, Path]):
        with open(data_file, encoding="utf-8") as f:
            lines = f.readlines()
        with open(data_file, "w", encoding="utf-8") as f:
            tag_bool = False
            new_sentence = True
            for line in lines:
                line_splits = re.sub(r"\s{2,}", " ", line).strip().split(" ")
                for substr in line_splits:
                    if substr == ".":
                        f.write("\n")
                        new_sentence = True
                    elif "<START:" in substr:
                        tag_bool = True
                        tag = substr.strip("<START:").strip(">")
                        if "loc" in tag:
                            tag_IOB = "-LOC"
                        elif "per" in tag:
                            tag_IOB = "-PER"
                        elif "org" in tag:
                            tag_IOB = "-ORG"
                        elif "misc" in tag:
                            tag_IOB = "-MISC"
                    elif "<END>" in substr:
                        tag_bool = False
                        new_sentence = True
                    else:
                        if tag_bool:
                            if new_sentence is True:
                                start = "B"
                                new_sentence = False
                            else:
                                start = "I"
                            f.write(substr.strip(" ") + " " + start + tag_IOB + "\n")
                        else:
                            f.write(substr.strip(" ") + " " + "O" + "\n")

    def _create_datasets(self, data_file: Union[str, Path], data_folder: Path):
        with open(data_file) as file:
            num_lines = len(file.readlines())
            file.seek(0)

            train_len = round(num_lines * 0.8)
            test_len = round(num_lines * 0.1)

            with (
                (data_folder / "train.txt").open("w", encoding="utf-8") as train,
                (data_folder / "test.txt").open("w", encoding="utf-8") as test,
                (data_folder / "dev.txt").open("w", encoding="utf-8") as dev,
            ):
                for k, line in enumerate(file.readlines(), start=1):
                    if k <= train_len:
                        train.write(line)
                    elif train_len < k <= (train_len + test_len):
                        test.write(line)
                    elif (train_len + test_len) < k <= num_lines:
                        dev.write(line)


class NER_HUNGARIAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the NER Business corpus for Hungarian.

        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # If the extracted corpus file is not yet present in dir
        if not os.path.isfile(data_folder / "hun_ner_corpus.txt"):
            # download zip if necessary
            hun_ner_path = "https://rgai.sed.hu/sites/rgai.sed.hu/files/business_NER.zip"
            path_to_zipped_corpus = cached_path(hun_ner_path, Path("datasets") / dataset_name)
            # extracted corpus is not present , so unpacking it.
            unpack_file(path_to_zipped_corpus, data_folder, mode="zip", keep=True)

        super().__init__(
            data_folder,
            columns,
            train_file="hun_ner_corpus.txt",
            column_delimiter="\t",
            encoding="latin-1",
            in_memory=in_memory,
            label_name_map={"0": "O"},
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_ICELANDIC(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the ICELANDIC_NER corpus.

        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if not os.path.isfile(data_folder / "icelandic_ner.txt"):
            # download zip
            icelandic_ner = "https://repository.clarin.is/repository/xmlui/handle/20.500.12537/42/allzip"
            icelandic_ner_path = cached_path(icelandic_ner, Path("datasets") / dataset_name)

            # unpacking the zip
            unpack_file(icelandic_ner_path, data_folder, mode="zip", keep=True)
        outputfile = os.path.abspath(data_folder)

        # merge the files in one as the zip is containing multiples files

        with open(outputfile / data_folder / "icelandic_ner.txt", "wb") as outfile:
            for files in os.walk(outputfile / data_folder):
                f = files[2]

                for i in range(len(f)):
                    if f[i].endswith(".txt"):
                        with open(outputfile / data_folder / f[i], "rb") as infile:
                            contents = infile.read()
                        outfile.write(contents)

        super().__init__(
            data_folder,
            columns,
            train_file="icelandic_ner.txt",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_JAPANESE(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the Hironsan/IOB2 corpus for Japanese.

        The first time you call this constructor it will automatically download the dataset.

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data from github if necessary (hironsan.txt, ja.wikipedia.conll)
        IOB2_path = "https://raw.githubusercontent.com/Hironsan/IOB2Corpus/master/"

        # download files if not present locally
        cached_path(f"{IOB2_path}hironsan.txt", data_folder / "raw")
        cached_path(f"{IOB2_path}ja.wikipedia.conll", data_folder / "raw")

        # we need to modify the original files by adding new lines after after the end of each sentence
        train_data_file = data_folder / "train.txt"
        if not train_data_file.is_file():
            self.__prepare_jap_wikinews_corpus(data_folder / "raw" / "hironsan.txt", data_folder / "train.txt")
            self.__prepare_jap_wikipedia_corpus(data_folder / "raw" / "ja.wikipedia.conll", data_folder / "train.txt")

        super().__init__(
            data_folder,
            columns,
            train_file="train.txt",
            in_memory=in_memory,
            default_whitespace_after=0,
            **corpusargs,
        )

    @staticmethod
    def __prepare_jap_wikipedia_corpus(file_in: Union[str, Path], file_out: Union[str, Path]):
        with open(file_in) as f:
            lines = f.readlines()
        with open(file_out, "a") as f:
            for line in lines:
                if line[0] == "。":
                    f.write(line)
                    f.write("\n")
                elif line[0] == "\n":
                    continue
                else:
                    f.write(line)

    @staticmethod
    def __prepare_jap_wikinews_corpus(file_in: Union[str, Path], file_out: Union[str, Path]):
        with open(file_in) as f:
            lines = f.readlines()
        with open(file_out, "a") as f:
            for line in lines:
                sp_line = line.split("\t")
                if sp_line[0] == "\n":
                    f.write("\n")
                else:
                    f.write(sp_line[0] + "\t" + sp_line[-1])


class NER_MASAKHANE(MultiCorpus):
    def __init__(
        self,
        languages: Union[str, list[str]] = "luo",
        version: str = "v2",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the Masakhane corpus available on https://github.com/masakhane-io/masakhane-ner/tree/main/data.

        It consists of ten African languages. Pass a language code or a list of language codes to initialize the corpus
        with the languages you require. If you pass "all", all languages will be initialized.
        :version: Specifies version of the dataset. Currently, only "v1" and "v2" are supported, using "v2" as default.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # if only one language is given
        if isinstance(languages, str):
            languages = [languages]

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        supported_versions = ["v1", "v2"]

        if version not in supported_versions:
            log.error(f"The specified version '{version}' is not in the list of supported version!")
            log.error(f"Supported versions are '{supported_versions}'!")
            raise Exception

        data_folder = base_path / dataset_name / version

        languages_to_code = {
            "v1": {
                "amharic": "amh",
                "hausa": "hau",
                "igbo": "ibo",
                "kinyarwanda": "kin",
                "luganda": "lug",
                "luo": "luo",
                "naija": "pcm",
                "swahili": "swa",
                "yoruba": "yor",
                "wolof": "wol",
            },
            "v2": {
                "bambara": "bam",
                "ghomala": "bbj",
                "ewe": "ewe",
                "fon": "fon",
                "hausa": "hau",
                "igbo": "ibo",
                "kinyarwanda": "kin",
                "luganda": "lug",
                "mossi": "mos",
                "naija": "pcm",
                "chichewa": "nya",
                "chishona": "sna",
                "kiswahili": "swa",
                "setswana": "tsn",
                "akan_twi": "twi",
                "wolof": "wol",
                "isixhosa": "xho",
                "yoruba": "yor",
                "isizulu": "zul",
            },
        }

        language_to_code = languages_to_code[version]

        data_paths = {
            "v1": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data",
            "v2": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/MasakhaNER2.0/data",
        }

        # use all languages if explicitly set to "all"
        if languages == ["all"]:
            languages = list(language_to_code.values())

        corpora: list[Corpus] = []
        for language in languages:
            if language in language_to_code:
                language = language_to_code[language]

            if language not in language_to_code.values():
                log.error(f"Language '{language}' is not in list of supported languages!")
                log.error(f"Supported are '{language_to_code.values()}'!")
                log.error("Instantiate this Corpus for instance like so 'corpus = NER_MASAKHANE(languages='luo')'")
                raise Exception

            language_folder = data_folder / language

            # download data if necessary
            data_path = f"{data_paths[version]}/{language}/"
            cached_path(f"{data_path}dev.txt", language_folder)
            cached_path(f"{data_path}test.txt", language_folder)
            cached_path(f"{data_path}train.txt", language_folder)

            # initialize comlumncorpus and add it to list
            log.info(f"Reading data for language {language}@{version}")
            corp = ColumnCorpus(
                data_folder=language_folder,
                column_format=columns,
                encoding="utf-8",
                in_memory=in_memory,
                name=language,
                **corpusargs,
            )
            corpora.append(corp)

        super().__init__(
            corpora,
            name="masakhane-" + "-".join(languages),
        )


class NER_MULTI_CONER(MultiFileColumnCorpus):
    def __init__(
        self,
        task: str = "multi",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Download and Initialize the MultiCoNer corpus.

        Args:
            task: either 'multi', 'code-switch', or the language code for one of the mono tasks.
            base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03' folder) on your machine POS tags or chunks respectively
            in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        folders = {
            "bn": "BN-Bangla",
            "de": "DE-German",
            "en": "EN-English",
            "es": "ES-Espanish",
            "fa": "FA-Farsi",
            "hi": "HI-Hindi",
            "ko": "KO-Korean",
            "nl": "NL-Dutch",
            "ru": "RU-Russian",
            "tr": "TR-Turkish",
            "zh": "ZH-Chinese",
            "mix": "MIX_Code_mixed",
            "multi": "MULTI_Multilingual",
        }

        possible_tasks = list(folders.keys())
        task = task.lower()

        if task not in possible_tasks:
            raise ValueError(f"task has to be one of {possible_tasks}, but is '{task}'")

        # column format
        columns = {0: "text", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = cached_path("s3://multiconer", base_path / dataset_name) / "multiconer2022"

        train_files = [data_folder / folders[task] / f"{task}_train.conll"]
        dev_files = [data_folder / folders[task] / f"{task}_dev.conll"]
        test_files = [data_folder / folders[task] / f"{task}_test.conll"]

        super().__init__(
            train_files=train_files,
            dev_files=dev_files,
            test_files=test_files,
            column_format=columns,
            comment_symbol="# id ",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_MULTI_CONER_V2(MultiFileColumnCorpus):
    def __init__(
        self,
        task: str = "multi",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        use_dev_as_test: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the MultiCoNer V2 corpus for the Semeval2023 workshop.

        This is only possible if you've applied and downloaded it to your machine.
        Apply for the corpus from here https://multiconer.github.io/dataset and unpack the .zip file's content into
        a folder called 'ner_multi_coner_v2'. Then set the base_path parameter in the constructor to the path to the
        parent directory where the ner_multi_coner_v2 folder resides. You can also create the multiconer in
        the {FLAIR_CACHE_ROOT}/datasets folder to leave the path empty.
        :param base_path: Path to the ner_multi_coner_v2 corpus (i.e. 'ner_multi_coner_v2' folder) on your machine
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param use_dev_as_test: If True, it uses the dev set as test set and samples random training data for a dev split.
        :param task: either 'multi', 'code-switch', or the language code for one of the mono tasks.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        folders = {
            "bn": "BN-Bangla",
            "de": "DE-German",
            "en": "EN-English",
            "es": "ES-Espanish",
            "fa": "FA-Farsi",
            "fr": "FR-French",
            "hi": "HI-Hindi",
            "it": "IT-Italian",
            "pt": "PT-Portuguese",
            "sv": "SV-Swedish",
            "uk": "UK-Ukrainian",
            "zh": "ZH-Chinese",
        }

        possible_tasks = [*list(folders.keys()), "multi"]
        task = task.lower()

        if task not in possible_tasks:
            raise ValueError(f"task has to be one of {possible_tasks}, but is '{task}'")

        # column format
        columns = {0: "text", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name / "train_dev"

        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'WARNING: MultiCoNerV2 dataset not found at "{data_folder}".')
            log.warning('Instructions for obtaining the data can be found here: https://multiconer.github.io/dataset"')
            log.warning("-" * 100)

        if task == "multi":
            train_files = list(data_folder.glob("*-train.conll"))
            dev_files = list(data_folder.glob("*-dev.conll"))
        else:
            train_files = [data_folder / f"{task}-train.conll"]
            dev_files = [data_folder / f"{task}-dev.conll"]
        test_files = []

        if use_dev_as_test:
            test_files = dev_files
            dev_files = []
        super().__init__(
            train_files=train_files,
            dev_files=dev_files,
            test_files=test_files,
            column_format=columns,
            comment_symbol="# id ",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_MULTI_WIKIANN(MultiCorpus):
    def __init__(
        self,
        languages: Union[str, list[str]] = "en",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the WkiAnn corpus for cross-lingual NER consisting of datasets from 282 languages that exist in Wikipedia.

        See https://elisa-ie.github.io/wikiann/ for details and for the languages and their
        respective abbreveations, i.e. "en" for english. (license: https://opendatacommons.org/licenses/by/)

        Parameters
        ----------
        languages : Union[str, list[str]]
            Should be an abbreviation of a language ("en", "de",..) or a list of abbreviations.
            The datasets of all passed languages will be saved in one MultiCorpus.
            (Note that, even though listed on https://elisa-ie.github.io/wikiann/ some datasets are empty.
            This includes "aa", "cho", "ho", "hz", "ii", "jam", "kj", "kr", "mus", "olo" and "tcy".)
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
            The data is in bio-format. It will by default (with the string "ner" as value) be transformed
            into the bioes format. If you dont want that set it to None.
        in_memory : bool, optional
            Specify that the dataset should be loaded in memory, which speeds up the training process but takes increases the RAM usage significantly.
        """
        if isinstance(languages, str):
            languages = [languages]

        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # For each language in languages, the file is downloaded if not existent
        # Then a comlumncorpus of that data is created and saved in a list
        # this list is handed to the multicorpus

        # list that contains the columncopora
        corpora: list[Corpus] = []

        google_drive_path = "https://drive.google.com/uc?id="
        # download data if necessary
        first = True
        for language in languages:
            language_folder = data_folder / language
            file_name = "wikiann-" + language + ".bio"

            # if language not downloaded yet, download it
            if not language_folder.exists():
                if first:
                    import gdown

                    first = False
                # create folder
                os.makedirs(language_folder)
                # get google drive id from list
                google_id = self._google_drive_id_from_language_name(language)
                url = google_drive_path + google_id

                # download from google drive
                gdown.download(url, str(language_folder / language) + ".tar.gz")

                # unzip
                log.info("Extracting data...")
                with tarfile.open(str(language_folder / language) + ".tar.gz", "r:gz") as tar:
                    tar.extract(file_name, str(language_folder))
                log.info("...done.")

                # transform data into required format
                # the processed dataset has the additional ending "_new"
                log.info("Processing dataset...")
                self._silver_standard_to_simple_ner_annotation(str(language_folder / file_name))
                # remove the unprocessed dataset
                os.remove(str(language_folder / file_name))
                log.info("...done.")

            # initialize comlumncorpus and add it to list
            log.info(f"Reading data for language {language}")
            corp = ColumnCorpus(
                data_folder=language_folder,
                column_format=columns,
                train_file=file_name + "_new",
                in_memory=in_memory,
                **corpusargs,
            )
            corpora.append(corp)
            log.info("...done.")

        super().__init__(
            corpora,
            name="wikiann",
        )

    def _silver_standard_to_simple_ner_annotation(self, data_file: Union[str, Path]):
        with (
            open(data_file, encoding="utf-8") as f_read,
            open(str(data_file) + "_new", "w+", encoding="utf-8") as f_write,
        ):
            while True:
                line = f_read.readline()
                if line:
                    if line == "\n":
                        f_write.write(line)
                    else:
                        liste = line.split()
                        f_write.write(liste[0] + " " + liste[-1] + "\n")
                else:
                    break

    def _google_drive_id_from_language_name(self, language):
        languages_ids = {
            "aa": "1tDDlydKq7KQQ3_23Ysbtke4HJOe4snIk",  # leer
            "ab": "1hB8REj2XA_0DjI9hdQvNvSDpuBIb8qRf",
            "ace": "1WENJS2ppHcZqaBEXRZyk2zY-PqXkTkgG",
            "ady": "1n6On8WWDHxEoybj7F9K15d_fkGPy6KgO",
            "af": "1CPB-0BD2tg3zIT60D3hmJT0i5O_SKja0",
            "ak": "1l2vlGHnQwvm9XhW5S-403fetwUXhBlZm",
            "als": "196xyYjhbie7sYLHLZHWkkurOwQLi8wK-",
            "am": "1ug1IEoExKD3xWpvfZprAPSQi82YF9Cet",
            "an": "1DNLgPOAOsGZBYd6rC5ddhzvc9_DtWnk2",
            "ang": "1W_0ti7Tl8AkqM91lRCMPWEuUnPOAZroV",
            "ar": "1tyvd32udEQG_cNeVpaD5I2fxvCc6XKIS",
            "arc": "1hSOByStqPmP3b9HfQ39EclUZGo8IKCMb",
            "arz": "1CKW5ZhxTpIHmc8Jt5JLz_5O6Cr8Icsan",
            "as": "12opBoIweBLM8XciMHT4B6-MAaKdYdvpE",
            "ast": "1rp64PxGZBDfcw-tpFBjLg_ddLDElG1II",
            "av": "1hncGUrkG1vwAAQgLtwOf41BWkHkEvdss",
            "ay": "1VmIsWpMTz442b4Mx798ZOgtB9vquKQtf",
            "az": "1FXDXsvBSdqc7GGIDZv0hqBOaaw12Ip2-",
            "azb": "1amVqOuHLEkhjn8rkGUl-mXdZlaACWyNT",
            "ba": "1aLx1d8GagI11VZVYOGQy0BEePeqoT0x3",
            "bar": "1JZ8-k8ZmnpWYI_Yl_cBBgjVdxoM9Daci",
            "bat-smg": "1trxKXDFSeKsygTMKi-ZqXSJs7F90k5a8",
            "bcl": "1Hs0k7KVZ2DPsqroZ4cUKcwZG4HdPV794",
            "be-x-old": "1gaK-spj1m6eGYQ-SsngLxxLUvP1VRk08",
            "be": "1_ttfOSy9BzCRkIT_p3mImT82XRPpEiuH",
            "bg": "1Iug6gYKemb0OrLTUrKDc_c66YGypTfCF",
            "bh": "12OcSFLu940A8tVQLxI8pnxKBpTeZHmrh",
            "bi": "1rftVziS_pqARx4mvLJC0sKLY-OL5ZIjE",
            "bjn": "1n17mkRjPUAOWQk5LQs2C3Tz3ShxK0enZ",
            "bm": "1284dwO_sfdsWE7FR06HhfBRUb8ePesKR",
            "bn": "1K2DM1mT4hkr6NlAIBTj95BeVXcgvpgDm",
            "bo": "1SzGHDVK-OguKdjZ4DXWiOJVrie1iHeWm",
            "bpy": "1m-e5EoruJufvwBEgJLmJtx6jzx64pYN2",
            "br": "1xdaBoJ1DnwI0iEq7gQN1dWcABAs_bM9H",
            "bs": "167dsB01trMYFQl8FshtIdfhjw7IfVKbk",
            "bug": "1yCnevM9_KJzFk27Vxsva_20OacLo4Uam",
            "bxr": "1DlByAX3zB-9UyEAVD4wtX-R7mXC-8xum",
            "ca": "1LuUgbd9sGa-5Ahcsy31EK89a3WOowftY",
            "cbk-zam": "1kgF8xoD-kIOWZET_9kp_4yNX6AAXn6PI",
            "cdo": "14x1y6611G-UAEGq92QEHRpreVkYnoUCw",
            "ce": "1QUUCVKA-fkiCHd3KT3zUWefaWnxzlZLu",
            "ceb": "1DJZE9RfaMoPNXHI73KBXAm4YSe-_YCUk",
            "ch": "1YzAfhmatkmTpkZbAcD6X83epCgzD5S2_",
            "cho": "1ciY0vF3c5a2mTOo_k32A2wMs0klK98Kb",  # leer
            "chr": "1EHaxz1UZHn7v2bbRzCLAhPsNtRzrG3Ae",
            "chy": "1nNWwMAJr1KNdz3bHf6uIn-thZCknlTeB",
            "ckb": "1llpaftcUSiXCZQZMdAqaJSrhwMdcf9IV",
            "co": "1ZP-8oWgMYfW7a6w6ygEFkKDGbN39QnDn",
            "cr": "1ST0xRicLAG4JdCZwGdaY-0pEXooQh7e6",
            "crh": "1Jmpq2XVYUR_XaXU5XNhtOMnz-qkpsgpE",
            "cs": "1Vydyze-jBkK_S1uV5ewV_Y6dbwhXr7lk",
            "csb": "1naUyF74lZPnnopXdOqf5Xor2kT4WoHfS",
            "cu": "1EN5dVTU6jc7YOYPCHq8EYUF31HlMUKs7",
            "cv": "1gEUAlqYSSDI4TrWCqP1LUq2n0X1XEjN3",
            "cy": "1q5g6NJE5GXf65Vc_P4BnUMHQ49Prz-J1",
            "da": "11onAGOLkkqrIwM784siWlg-cewa5WKm8",
            "de": "1f9nWvNkCCy6XWhd9uf4Dq-2--GzSaYAb",
            "diq": "1IkpJaVbEOuOs9qay_KG9rkxRghWZhWPm",
            "dsb": "1hlExWaMth-2eVIQ3i3siJSG-MN_7Z6MY",
            "dv": "1WpCrslO4I7TMb2uaKVQw4U2U8qMs5szi",
            "dz": "10WX52ePq2KfyGliwPvY_54hIjpzW6klV",
            "ee": "1tYEt3oN2KPzBSWrk9jpCqnW3J1KXdhjz",
            "el": "1cxq4NUYmHwWsEn5waYXfFSanlINXWLfM",
            "eml": "17FgGhPZqZNtzbxpTJOf-6nxEuI5oU4Vd",
            "en": "1mqxeCPjxqmO7e8utj1MQv1CICLFVvKa-",
            "eo": "1YeknLymGcqj44ug2yd4P7xQVpSK27HkK",
            "es": "1Dnx3MVR9r5cuoOgeew2gT8bDvWpOKxkU",
            "et": "1Qhb3kYlQnLefWmNimdN_Vykm4mWzbcWy",
            "eu": "1f613wH88UeITYyBSEMZByK-nRNMwLHTs",
            "ext": "1D0nLOZ3aolCM8TShIRyCgF3-_MhWXccN",
            "fa": "1QOG15HU8VfZvJUNKos024xI-OGm0zhEX",
            "ff": "1h5pVjxDYcq70bSus30oqi9KzDmezVNry",
            "fi": "1y3Kf6qYsSvL8_nSEwE1Y6Bf6ninaPvqa",
            "fiu-vro": "1oKUiqG19WgPd3CCl4FGudk5ATmtNfToR",
            "fj": "10xDMuqtoTJlJFp5ghbhKfNWRpLDK3W4d",
            "fo": "1RhjYqgtri1276Be1N9RrNitdBNkpzh0J",
            "fr": "1sK_T_-wzVPJYrnziNqWTriU52rEsXGjn",
            "frp": "1NUm8B2zClBcEa8dHLBb-ZgzEr8phcQyZ",
            "frr": "1FjNqbIUlOW1deJdB8WCuWjaZfUzKqujV",
            "fur": "1oqHZMK7WAV8oHoZLjGR0PfmO38wmR6XY",
            "fy": "1DvnU6iaTJc9bWedmDklHyx8nzKD1s3Ge",
            "ga": "1Ql6rh7absdYQ8l-3hj_MVKcEC3tHKeFB",
            "gag": "1zli-hOl2abuQ2wsDJU45qbb0xuvYwA3a",
            "gan": "1u2dOwy58y-GaS-tCPJS_i9VRDQIPXwCr",
            "gd": "1umsUpngJiwkLdGQbRqYpkgxZju9dWlRz",
            "gl": "141K2IbLjJfXwFTIf-kthmmG0YWdi8liE",
            "glk": "1ZDaxQ6ilXaoivo4_KllagabbvfOuiZ0c",
            "gn": "1hM4MuCaVnZqnL-w-0N-WcWag22ikVLtZ",
            "gom": "1BNOSw75tzPC0wEgLOCKbwu9wg9gcLOzs",
            "got": "1YSHYBtXc1WvUvMIHPz6HHgJvaXKulJUj",
            "gu": "1VdK-B2drqFwKg8KD23c3dKXY-cZgCMgd",
            "gv": "1XZFohYNbKszEFR-V-yDXxx40V41PV9Zm",
            "ha": "18ZG4tUU0owRtQA8Ey3Dl72ALjryEJWMC",
            "hak": "1QQe3WgrCWbvnVH42QXD7KX4kihHURB0Z",
            "haw": "1FLqlK-wpz4jy768XbQAtxd9PhC-9ciP7",
            "he": "18K-Erc2VOgtIdskaQq4D5A3XkVstDmfX",
            "hi": "1lBRapb5tjBqT176gD36K5yb_qsaFeu-k",
            "hif": "153MQ9Ga4NQ-CkK8UiJM3DjKOk09fhCOV",
            "ho": "1c1AoS7yq15iVkTEE-0f3x25NT4F202B8",  # leer
            "hr": "1wS-UtB3sGHuXJQQGR0F5lDegogsgoyif",
            "hsb": "1_3mMLzAE5OmXn2z64rW3OwWbo85Mirbd",
            "ht": "1BwCaF0nfdgkM7Yt7A7d7KyVk0BcuwPGk",
            "hu": "10AkDmTxUWNbOXuYLYZ-ZPbLAdGAGZZ8J",
            "hy": "1Mi2k2alJJquT1ybd3GC3QYDstSagaWdo",
            "hz": "1c1m_-Q92v0Di7Nez6VuaccrN19i8icKV",  # leer
            "ia": "1jPyqTmDuVhEhj89N606Cja5heJEbcMoM",
            "id": "1JWIvIh8fQoMQqk1rPvUThaskxnTs8tsf",
            "ie": "1TaKRlTtB8-Wqu4sfvx6JQKIugAlg0pV-",
            "ig": "15NFAf2Qx6BXSjv_Oun9_3QRBWNn49g86",
            "ii": "1qldGJkMOMKwY13DpcgbxQCbff0K982f9",  # leer
            "ik": "1VoSTou2ZlwVhply26ujowDz6gjwtxmny",
            "ilo": "1-xMuIT6GaM_YeHqgm1OamGkxYfBREiv3",
            "io": "19Zla0wsAcrZm2c0Pw5ghpp4rHjYs26Pp",
            "is": "11i-NCyqS6HbldIbYulsCgQGZFXR8hwoB",
            "it": "1HmjlOaQunHqL2Te7pIkuBWrnjlmdfYo_",
            "iu": "18jKm1S7Ls3l0_pHqQH8MycG3LhoC2pdX",
            "ja": "10dz8UxyK4RIacXE2HcGdrharmp5rwc3r",
            "jam": "1v99CXf9RnbF6aJo669YeTR6mQRTOLZ74",  # leer
            "jbo": "1_LmH9hc6FDGE3F7pyGB1fUEbSwuTYQdD",
            "jv": "1qiSu1uECCLl4IBZS27FBdJIBivkJ7GwE",
            "ka": "172UFuFRBX2V1aWeXlPSpu9TjS-3cxNaD",
            "kaa": "1kh6hMPUdqO-FIxRY6qaIBZothBURXxbY",
            "kab": "1oKjbZI6ZrrALCqnPCYgIjKNrKDA7ehcs",
            "kbd": "1jNbfrboPOwJmlXQBIv053d7n5WXpMRv7",
            "kg": "1iiu5z-sdJ2JLC4Ja9IgDxpRZklIb6nDx",
            "ki": "1GUtt0QI84c5McyLGGxoi5uwjHOq1d6G8",
            "kj": "1nSxXUSGDlXVCIPGlVpcakRc537MwuKZR",  # leer
            "kk": "1ryC3UN0myckc1awrWhhb6RIi17C0LCuS",
            "kl": "1gXtGtX9gcTXms1IExICnqZUHefrlcIFf",
            "km": "1DS5ATxvxyfn1iWvq2G6qmjZv9pv0T6hD",
            "kn": "1ZGLYMxbb5-29MNmuUfg2xFhYUbkJFMJJ",
            "ko": "12r8tIkTnwKhLJxy71qpIcoLrT6NNhQYm",
            "koi": "1EdG_wZ_Qk124EPAZw-w6rdEhYLsgcvIj",
            "kr": "19VNQtnBA-YL_avWuVeHQHxJZ9MZ04WPF",  # leer
            "krc": "1nReV4Mb7Wdj96czpO5regFbdBPu0zZ_y",
            "ks": "1kzh0Pgrv27WRMstR9MpU8mu7p60TcT-X",
            "ksh": "1iHJvrl2HeRaCumlrx3N7CPrHQ2KuLUkt",
            "ku": "1YqJog7Bkk0fHBCSTxJ9heeE-bfbkbkye",
            "kv": "1s91HI4eq8lQYlZwfrJAgaGlCyAtIhvIJ",
            "kw": "16TaIX2nRfqDp8n7zudd4bqf5abN49dvW",
            "ky": "17HPUKFdKWhUjuR1NOp5f3PQYfMlMCxCT",
            "la": "1NiQuBaUIFEERvVXo6CQLwosPraGyiRYw",
            "lad": "1PEmXCWLCqnjLBomMAYHeObM1AmVHtD08",
            "lb": "1nE4g10xoTU23idmDtOQ0w2QCuizZ6QH_",
            "lbe": "1KOm-AdRcCHfSc1-uYBxBA4GjxXjnIlE-",
            "lez": "1cJAXshrLlF1TZlPHJTpDwEvurIOsz4yR",
            "lg": "1Ur0y7iiEpWBgHECrIrT1OyIC8um_y4th",
            "li": "1TikIqfqcZlSDWhOae1JnjJiDko4nj4Dj",
            "lij": "1ro5ItUcF49iP3JdV82lhCQ07MtZn_VjW",
            "lmo": "1W4rhBy2Pi5SuYWyWbNotOVkVY3kYWS_O",
            "ln": "1bLSV6bWx0CgFm7ByKppZLpYCFL8EIAoD",
            "lo": "1C6SSLeKF3QirjZbAZAcpVX_AXYg_TJG3",
            "lrc": "1GUcS28MlJe_OjeQfS2AJ8uczpD8ut60e",
            "lt": "1gAG6TcMTmC128wWK0rCXRlCTsJY9wFQY",
            "ltg": "12ziP8t_fAAS9JqOCEC0kuJObEyuoiOjD",
            "lv": "1MPuAM04u-AtfybXdpHwCqUpFWbe-zD0_",
            "mai": "1d_nUewBkka2QGEmxCc9v3dTfvo7lPATH",
            "map-bms": "1wrNIE-mqp2xb3lrNdwADe6pb7f35NP6V",
            "mdf": "1BmMGUJy7afuKfhfTBMiKxM3D7FY-JrQ2",
            "mg": "105WaMhcWa-46tCztoj8npUyg0aH18nFL",
            "mh": "1Ej7n6yA1cF1cpD5XneftHtL33iHJwntT",
            "mhr": "1CCPIUaFkEYXiHO0HF8_w07UzVyWchrjS",
            "mi": "1F6au9xQjnF-aNBupGJ1PwaMMM6T_PgdQ",
            "min": "1tVK5SHiCy_DaZSDm3nZBgT5bgWThbJt_",
            "mk": "18NpudytGhSWq_LbmycTDw10cSftlSBGS",
            "ml": "1V73UE-EvcE-vV3V1RTvU4sak6QFcP91y",
            "mn": "14jRXicA87oXZOZllWqUjKBMetNpQEUUp",
            "mo": "1YsLGNMsJ7VsekhdcITQeolzOSK4NzE6U",
            "mr": "1vOr1AIHbgkhTO9Ol9Jx5Wh98Qdyh1QKI",
            "mrj": "1dW-YmEW8a9D5KyXz8ojSdIXWGekNzGzN",
            "ms": "1bs-_5WNRiZBjO-DtcNtkcIle-98homf_",
            "mt": "1L7aU3iGjm6SmPIU74k990qRgHFV9hrL0",
            "mus": "1_b7DcRqiKJFEFwp87cUecqf8A5BDbTIJ",  # leer
            "mwl": "1MfP0jba2jQfGVeJOLq26MjI6fYY7xTPu",
            "my": "16wsIGBhNVd2lC2p6n1X8rdMbiaemeiUM",
            "myv": "1KEqHmfx2pfU-a1tdI_7ZxMQAk5NJzJjB",
            "mzn": "1CflvmYEXZnWwpsBmIs2OvG-zDDvLEMDJ",
            "na": "1r0AVjee5wNnrcgJxQmVGPVKg5YWz1irz",
            "nah": "1fx6eu91NegyueZ1i0XaB07CKjUwjHN7H",
            "nap": "1bhT4sXCJvaTchCIV9mwLBtf3a7OprbVB",
            "nds-nl": "1UIFi8eOCuFYJXSAXZ9pCWwkQMlHaY4ye",
            "nds": "1FLgZIXUWa_vekDt4ndY0B5XL7FNLiulr",
            "ne": "1gEoCjSJmzjIH4kdHsbDZzD6ID4_78ekS",
            "new": "1_-p45Ny4w9UvGuhD8uRNSPPeaARYvESH",
            "ng": "11yxPdkmpmnijQUcnFHZ3xcOmLTYJmN_R",
            "nl": "1dqYXg3ilzVOSQ_tz_dF47elSIvSIhgqd",
            "nn": "1pDrtRhQ001z2WUNMWCZQU3RV_M0BqOmv",
            "no": "1zuT8MI96Ivpiu9mEVFNjwbiM8gJlSzY2",
            "nov": "1l38388Rln0NXsSARMZHmTmyfo5C0wYTd",
            "nrm": "10vxPq1Nci7Wpq4XOvx3dtqODskzjdxJQ",
            "nso": "1iaIV8qlT0RDnbeQlnxJ3RehsG3gU5ePK",
            "nv": "1oN31jT0w3wP9aGwAPz91pSdUytnd9B0g",
            "ny": "1eEKH_rUPC560bfEg11kp3kbe8qWm35IG",
            "oc": "1C01cW8G_j8US-DTrsmeal_ENHTtNWn-H",
            "olo": "1vbDwKZKqFq84dusr1SvDx5JbBcPanx9L",  # leer
            "om": "1q3h22VMbWg2kgVFm-OArR-E4y1yBQ1JX",
            "or": "1k8LwCE8nC7lq6neXDaS3zRn0KOrd9RnS",
            "os": "1u81KAB34aEQfet00dLMRIBJsfRwbDTij",
            "pa": "1JDEHL1VcLHBamgTPBom_Ryi8hk6PBpsu",
            "pag": "1k905VUWnRgY8kFb2P2431Kr4dZuolYGF",
            "pam": "1ssugGyJb8ipispC60B3I6kzMsri1WcvC",
            "pap": "1Za0wfwatxYoD7jGclmTtRoBP0uV_qImQ",
            "pcd": "1csJlKgtG04pdIYCUWhsCCZARKIGlEYPx",
            "pdc": "1Xnms4RXZKZ1BBQmQJEPokmkiweTpouUw",
            "pfl": "1tPQfHX7E0uKMdDSlwNw5aGmaS5bUK0rn",
            "pi": "16b-KxNxzbEuyoNSlI3bfe2YXmdSEsPFu",
            "pih": "1vwyihTnS8_PE5BNK7cTISmIBqGWvsVnF",
            "pl": "1fijjS0LbfpKcoPB5V8c8fH08T8AkXRp9",
            "pms": "12ySc7X9ajWWqMlBjyrPiEdc-qVBuIkbA",
            "pnb": "1RB3-wjluhTKbdTGCsk3nag1bM3m4wENb",
            "pnt": "1ZCUzms6fY4on_fW8uVgO7cEs9KHydHY_",
            "ps": "1WKl9Av6Sqz6aHKyUM5kIh90mzFzyVWH9",
            "pt": "13BX-_4_hcTUp59HDyczFDI32qUB94vUY",
            "qu": "1CB_C4ygtRoegkqgcqfXNHr8oQd-UcvDE",
            "rm": "1YRSGgWoxEqSojHXuBHJnY8vAHr1VgLu-",
            "rmy": "1uFcCyvOWBJWKFQxbkYSp373xUXVl4IgF",
            "rn": "1ekyyb2MvupYGY_E8_BhKvV664sLvW4aE",
            "ro": "1YfeNTSoxU-zJMnyQotLk5X8B_6nHryBu",
            "roa-rup": "150s4H4TdQ5nNYVC6j0E416TUAjBE85yy",
            "roa-tara": "1H6emfQsD_a5yohK4RMPQ-GrnHXqqVgr3",
            "ru": "11gP2s-SYcfS3j9MjPp5C3_nFeQB-8x86",
            "rue": "1OuSglZAndja1J5D5IUmdbt_niTTyEgYK",
            "rw": "1NuhHfi0-B-Xlr_BApijnxCw0WMEltttP",
            "sa": "1P2S3gL_zvKgXLKJJxg-Fb4z8XdlVpQik",
            "sah": "1qz0MpKckzUref2FX_FYiNzI2p4BDc5oR",
            "sc": "1oAYj_Fty4FUwjAOBEBaiZt_cY8dtpDfA",
            "scn": "1sDN9zHkXWYoHYx-DUu-GPvsUgB_IRa8S",
            "sco": "1i8W7KQPj6YZQLop89vZBSybJNgNsvXWR",
            "sd": "1vaNqfv3S8Gl5pQmig3vwWQ3cqRTsXmMR",
            "se": "1RT9xhn0Vl90zjWYDTw5V1L_u1Oh16tpP",
            "sg": "1iIh2oXD2Szz_AygUvTt3_ZK8a3RYEGZ_",
            "sh": "1qPwLiAm6t4__G-zVEOrBgYx6VRmgDgiS",
            "si": "1G5ryceID0TP6SAO42e-HAbIlCvYmnUN7",
            "simple": "1FVV49o_RlK6M5Iw_7zeJOEDQoTa5zSbq",
            "sk": "11mkYvbmAWKTInj6t4Ma8BUPxoR5o6irL",
            "sl": "1fsIZS5LgMzMzZ6T7ogStyj-ILEZIBRvO",
            "sm": "1yefECpKX_Y4R7G2tggIxvc_BvJfOAz-t",
            "sn": "1fYeCjMPvRAv94kvZjiKI-ktIDLkbv0Ve",
            "so": "1Uc-eSZnJb36SgeTvRU3GirXZOlGD_NB6",
            "sq": "11u-53n71O_yjpwRiCQSwgL7N2w72ZptX",
            "sr": "1PGLGlQi8Q0Eac6dib-uuCJAAHK6SF5Pz",
            "srn": "1JKiL3TSXqK1-KhPfAwMK0uqw90WEzg7M",
            "ss": "1e0quNEsA1dn57-IbincF4D82dRWgzQlp",
            "st": "1ny-FBzpBqIDgv6jMcsoFev3Ih65FNZFO",
            "stq": "15Fx32ROy2IM6lSqAPUykkr3CITR6Xd7v",
            "su": "1C0FJum7bYZpnyptBvfAgwJb0TX2hggtO",
            "sv": "1YyqzOSXzK5yrAou9zeTDWH_7s569mDcz",
            "sw": "1_bNTj6T8eXlNAIuHaveleWlHB_22alJs",
            "szl": "1_dXEip1snK4CPVGqH8x7lF5O-6FdCNFW",
            "ta": "1ZFTONsxGtSnC9QB6RpWSvgD_MbZwIhHH",
            "tcy": "15R6u7KQs1vmDSm_aSDrQMJ3Q6q3Be0r7",  # leer
            "te": "11Sx-pBAPeZOXGyv48UNSVMD0AH7uf4YN",
            "tet": "11mr2MYLcv9pz7mHhGGNi5iNCOVErYeOt",
            "tg": "16ttF7HWqM9Cnj4qmgf3ZfNniiOJfZ52w",
            "th": "14xhIt-xr5n9nMuvcwayCGM1-zBCFZquW",
            "ti": "123q5e9MStMShp8eESGtHdSBGLDrCKfJU",
            "tk": "1X-JNInt34BNGhg8A8Peyjw2WjsALdXsD",
            "tl": "1WkQHbWd9cqtTnSHAv0DpUThaBnzeSPTJ",
            "tn": "1fHfQHetZn8-fLuRZEu-cvs-kQYwPvjyL",
            "to": "1cHOLaczYJ8h-OqQgxeoH9vMG3izg6muT",
            "tpi": "1YsRjxVu6NYOrXRb8oqMO9FPaicelFEcu",
            "tr": "1J1Zy02IxvtCK0d1Ba2h_Ulit1mVb9UIX",
            "ts": "1pIcfAt3KmtmDkyhOl-SMSeoM8aP8bOpl",
            "tt": "1vsfzCjj-_bMOn5jBai41TF5GjKJM_Ius",
            "tum": "1NWcg65daI2Bt0awyEgU6apUDbBmiqCus",
            "tw": "1WCYKZIqS7AagS76QFSfbteiOgFNBvNne",
            "ty": "1DIqaP1l-N9VXTNokrlr6EuPMGE765o4h",
            "tyv": "1F3qa05OYLBcjT1lXMurAJFDXP_EesCvM",
            "udm": "1T0YMTAPLOk768sstnewy5Jxgx2RPu3Rb",
            "ug": "1fjezvqlysyZhiQMZdazqLGgk72PqtXAw",
            "uk": "1UMJCHtzxkfLDBJE7NtfN5FeMrnnUVwoh",
            "ur": "1WNaD2TuHvdsF-z0k_emQYchwoQQDFmRk",
            "uz": "11wrG2FSTpRJc2jb5MhgvxjkVDYhT8M-l",
            "ve": "1PucJ7pJ4CXGEXZ5p_WleZDs2usNz74to",
            "vec": "1cAVjm_y3ehNteDQIYz9yyoq1EKkqOXZ0",
            "vep": "1K_eqV7O6C7KPJWZtmIuzFMKAagj-0O85",
            "vi": "1yQ6nhm1BmG9lD4_NaG1hE5VV6biEaV5f",
            "vls": "1bpQQW6pKHruKJJaKtuggH5rReMXyeVXp",
            "vo": "1D80QRdTpe7H4mHFKpfugscsjX71kiMJN",
            "wa": "1m4B81QYbf74htpInDU5p7d0n0ot8WLPZ",
            "war": "1EC3jsHtu22tHBv6jX_I4rupC5RwV3OYd",
            "wo": "1vChyqNNLu5xYHdyHpACwwpw4l3ptiKlo",
            "wuu": "1_EIn02xCUBcwLOwYnA-lScjS2Lh2ECw6",
            "xal": "19bKXsL1D2UesbB50JPyc9TpG1lNc2POt",
            "xh": "1pPVcxBG3xsCzEnUzlohc_p89gQ9dSJB3",
            "xmf": "1SM9llku6I_ZuZz05mOBuL2lx-KQXvehr",
            "yi": "1WNWr1oV-Nl7c1Jv8x_MiAj2vxRtyQawu",
            "yo": "1yNVOwMOWeglbOcRoZzgd4uwlN5JMynnY",
            "za": "1i7pg162cD_iU9h8dgtI2An8QCcbzUAjB",
            "zea": "1EWSkiSkPBfbyjWjZK0VuKdpqFnFOpXXQ",
            "zh-classical": "1uUKZamNp08KA7s7794sKPOqPALvo_btl",
            "zh-min-nan": "1oSgz3YBXLGUgI7kl-uMOC_ww6L0FNFmp",
            "zh-yue": "1zhwlUeeiyOAU1QqwqZ8n91yXIRPFA7UE",
            "zh": "1LZ96GUhkVHQU-aj2C3WOrtffOp0U3Z7f",
            "zu": "1FyXl_UK1737XB3drqQFhGXiJrJckiB1W",
        }
        return languages_ids[language]


class NER_MULTI_XTREME(MultiCorpus):
    def __init__(
        self,
        languages: Union[str, list[str]] = "en",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
        **corpusargs,
    ) -> None:
        """Xtreme corpus for cross-lingual NER consisting of datasets of a total of 40 languages.

        The data comes from the google research work XTREME https://github.com/google-research/xtreme.
        The data is derived from the wikiann dataset https://elisa-ie.github.io/wikiann/ (license: https://opendatacommons.org/licenses/by/)

        Parameters
        ----------
        languages : Union[str, list[str]], optional
            Specify the languages you want to load. Provide an empty list or string to select all languages.
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
        in_memory : bool, optional
            Specify that the dataset should be loaded in memory, which speeds up the training process but takes increases the RAM usage significantly.
        """
        # if no languages are given as argument all languages used in XTREME will be loaded
        if not languages:
            languages = [
                "af",
                "ar",
                "bg",
                "bn",
                "de",
                "el",
                "en",
                "es",
                "et",
                "eu",
                "fa",
                "fi",
                "fr",
                "he",
                "hi",
                "hu",
                "id",
                "it",
                "ja",
                "jv",
                "ka",
                "kk",
                "ko",
                "ml",
                "mr",
                "ms",
                "my",
                "nl",
                "pt",
                "ru",
                "sw",
                "ta",
                "te",
                "th",
                "tl",
                "tr",
                "ur",
                "vi",
                "yo",
                "zh",
            ]

        # if only one language is given
        if isinstance(languages, str):
            languages = [languages]

        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # For each language in languages, the file is downloaded if not existent
        # Then a comlumncorpus of that data is created and saved in a list
        # This list is handed to the multicorpus

        # list that contains the columncopora
        corpora: list[Corpus] = []

        hu_path = "https://nlp.informatik.hu-berlin.de/resources/datasets/panx_dataset"

        # download data if necessary
        for language in languages:
            language_folder = data_folder / language

            # if language not downloaded yet, download it
            if not language_folder.exists():
                file_name = language + ".tar.gz"
                # create folder
                os.makedirs(language_folder)

                # download from HU Server
                temp_file = cached_path(
                    hu_path + "/" + file_name,
                    Path("datasets") / dataset_name / language,
                )

                # unzip
                log.info("Extracting data...")

                with tarfile.open(str(temp_file), "r:gz") as tar:
                    for part in ["train", "test", "dev"]:
                        tar.extract(part, str(language_folder))
                log.info("...done.")

                # transform data into required format
                log.info("Processing dataset...")
                for part in ["train", "test", "dev"]:
                    self._xtreme_to_simple_ner_annotation(str(language_folder / part))
                log.info("...done.")

            # initialize comlumncorpus and add it to list
            log.info(f"Reading data for language {language}")
            corp = ColumnCorpus(
                data_folder=language_folder,
                column_format=columns,
                in_memory=in_memory,
                **corpusargs,
            )
            corpora.append(corp)

        super().__init__(
            corpora,
            name="xtreme",
        )

    def _xtreme_to_simple_ner_annotation(self, data_file: Union[str, Path]):
        with open(data_file, encoding="utf-8") as f:
            lines = f.readlines()
        with open(data_file, "w", encoding="utf-8") as f:
            for line in lines:
                if line == "\n":
                    f.write(line)
                else:
                    liste = line.split()
                    f.write(liste[0].split(":", 1)[1] + " " + liste[1] + "\n")


class NER_MULTI_WIKINER(MultiCorpus):
    def __init__(
        self,
        languages: Union[str, list[str]] = "en",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # if only one language is given
        if isinstance(languages, str):
            languages = [languages]

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        corpora: list[Corpus] = []
        for language in languages:
            language_folder = data_folder / language

            # download data if necessary
            self._download_wikiner(language, str(language_folder))

            # initialize comlumncorpus and add it to list
            log.info(f"Read data for language {language}")
            corp = ColumnCorpus(
                data_folder=language_folder,
                column_format=columns,
                in_memory=in_memory,
                **corpusargs,
            )
            corpora.append(corp)

        super().__init__(
            corpora,
            name="wikiner",
        )

    def _download_wikiner(self, language_code: str, dataset_name: str):
        # download data if necessary
        wikiner_path = "https://raw.githubusercontent.com/dice-group/FOX/master/input/Wikiner/"
        lc = language_code

        data_file = flair.cache_root / "datasets" / dataset_name / f"aij-wikiner-{lc}-wp3.train"
        if not data_file.is_file():
            cached_path(
                f"{wikiner_path}aij-wikiner-{lc}-wp3.bz2",
                Path("datasets") / dataset_name,
            )
            import bz2

            # unpack and write out in CoNLL column-like format
            bz_file = bz2.BZ2File(
                flair.cache_root / "datasets" / dataset_name / f"aij-wikiner-{lc}-wp3.bz2",
                "rb",
            )
            with (
                bz_file as f,
                open(
                    flair.cache_root / "datasets" / dataset_name / f"aij-wikiner-{lc}-wp3.train",
                    "w",
                    encoding="utf-8",
                ) as out,
            ):
                for lineb in f:
                    line = lineb.decode("utf-8")
                    words = line.split(" ")
                    for word in words:
                        out.write("\t".join(word.split("|")) + "\n")


class NER_SWEDISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the NER_SWEDISH corpus for Swedish.

        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ner_spraakbanken_path = "https://raw.githubusercontent.com/klintan/swedish-ner-corpus/master/"
        cached_path(f"{ner_spraakbanken_path}test_corpus.txt", Path("datasets") / dataset_name)
        cached_path(f"{ner_spraakbanken_path}train_corpus.txt", Path("datasets") / dataset_name)

        # data is not in IOB2 format. Thus we transform it to IOB2
        self._add_IOB2_tags(data_file=Path(data_folder / "test_corpus.txt"))
        self._add_IOB2_tags(data_file=Path(data_folder / "train_corpus.txt"))

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            **corpusargs,
        )

    def _add_IOB2_tags(self, data_file: Union[str, Path], encoding: str = "utf8"):
        """Function that adds IOB2 tags if only chunk names are provided.

        e.g. words are tagged PER instead of B-PER or I-PER. Replaces '0' with 'O' as the no-chunk tag since ColumnCorpus expects
        the letter 'O'. Additionally it removes lines with no tags in the data file and can also
        be used if the data is only partially IOB tagged.

        Parameters
        ----------
        data_file : Union[str, Path]
            Path to the data file.
        encoding : str, optional
            Encoding used in open function. The default is "utf8".
        """
        with open(file=data_file, encoding=encoding) as f:
            lines = f.readlines()
        with open(file=data_file, mode="w", encoding=encoding) as f:
            pred = "O"  # remembers tag of predecessing line
            for line in lines:
                line_list = line.split()
                if len(line_list) == 2:  # word with tag
                    word = line_list[0]
                    tag = line_list[1]
                    if tag in ["0", "O"]:  # no chunk
                        f.write(word + " O\n")
                        pred = "O"
                    elif "-" not in tag:  # no IOB tags
                        if pred == "O":  # found a new chunk
                            f.write(word + " B-" + tag + "\n")
                            pred = tag
                        else:  # found further part of chunk or new chunk directly after old chunk
                            if pred == tag:
                                f.write(word + " I-" + tag + "\n")
                            else:
                                f.write(word + " B-" + tag + "\n")
                                pred = tag
                    else:  # line already has IOB tag (tag contains '-')
                        f.write(line)
                        pred = tag.split("-")[1]
                elif len(line_list) == 0:  # empty line
                    f.write("\n")
                    pred = "O"


class NER_TURKU(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the Finnish TurkuNER corpus.

        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        conll_path = "https://raw.githubusercontent.com/TurkuNLP/turku-ner-corpus/master/data/conll"
        dev_file = "dev.tsv"
        test_file = "test.tsv"
        train_file = "train.tsv"
        cached_path(f"{conll_path}/{dev_file}", Path("datasets") / dataset_name)
        cached_path(f"{conll_path}/{test_file}", Path("datasets") / dataset_name)
        cached_path(f"{conll_path}/{train_file}", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            dev_file=dev_file,
            test_file=test_file,
            train_file=train_file,
            column_delimiter="\t",
            encoding="latin-1",
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )


class NER_UKRAINIAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the Ukrainian NER corpus from lang-uk project.

        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        conll_path = "https://raw.githubusercontent.com/lang-uk/flair-ner/master/fixed-split"
        test_file = "test.iob"
        train_file = "train.iob"
        cached_path(f"{conll_path}/{test_file}", Path("datasets") / dataset_name)
        cached_path(f"{conll_path}/{train_file}", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            test_file=test_file,
            train_file=train_file,
            column_delimiter=" ",
            encoding="utf-8",
            in_memory=in_memory,
            **corpusargs,
        )


class KEYPHRASE_SEMEVAL2017(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "keyword"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        semeval2017_path = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/SemEval-2017"
        cached_path(f"{semeval2017_path}/train.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2017_path}/test.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2017_path}/dev.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            **corpusargs,
        )


class KEYPHRASE_INSPEC(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "keyword"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        inspec_path = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/Inspec"
        cached_path(f"{inspec_path}/train.txt", Path("datasets") / dataset_name)
        cached_path(f"{inspec_path}/test.txt", Path("datasets") / dataset_name)
        if "dev.txt" not in os.listdir(data_folder):
            cached_path(f"{inspec_path}/valid.txt", Path("datasets") / dataset_name)
            # rename according to train - test - dev - convention
            os.rename(data_folder / "valid.txt", data_folder / "dev.txt")

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            **corpusargs,
        )


class KEYPHRASE_SEMEVAL2010(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "keyword"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        semeval2010_path = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/processed_semeval-2010"
        cached_path(f"{semeval2010_path}/train.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2010_path}/test.txt", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            **corpusargs,
        )


class UP_CHINESE(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the Chinese dataset from the Universal Propositions Bank.

        The dataset is downloaded from  https://github.com/System-T/UniversalPropositions

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        up_zh_path = "https://raw.githubusercontent.com/System-T/UniversalPropositions/master/UP_Chinese/"
        cached_path(f"{up_zh_path}zh-up-train.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_zh_path}zh-up-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_zh_path}zh-up-test.conllu", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            train_file="zh-up-train.conllu",
            test_file="zh-up-test.conllu",
            dev_file="zh-up-dev.conllu",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            comment_symbol="#",
            **corpusargs,
        )


class UP_ENGLISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the English dataset from the Universal Propositions Bank.

        The dataset is downloaded from  https://github.com/System-T/UniversalPropositions

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 10: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        up_en_path = "https://raw.githubusercontent.com/System-T/UniversalPropositions/master/UP_English-EWT/"
        cached_path(f"{up_en_path}en_ewt-up-train.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_en_path}en_ewt-up-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_en_path}en_ewt-up-test.conllu", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            train_file="en_ewt-up-train.conllu",
            test_file="en_ewt-up-test.conllu",
            dev_file="en_ewt-up-dev.conllu",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            comment_symbol="#",
            label_name_map={"_": "O"},
            **corpusargs,
        )


class UP_FRENCH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the French dataset from the Universal Propositions Bank.

        The dataset is downloaded from  https://github.com/System-T/UniversalPropositions

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        up_fr_path = "https://raw.githubusercontent.com/System-T/UniversalPropositions/master/UP_French/"
        cached_path(f"{up_fr_path}fr-up-train.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_fr_path}fr-up-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_fr_path}fr-up-test.conllu", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            train_file="fr-up-train.conllu",
            test_file="fr-up-test.conllu",
            dev_file="fr-up-dev.conllu",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            comment_symbol="#",
            **corpusargs,
        )


class UP_FINNISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the Finnish dataset from the Universal Propositions Bank.

        The dataset is downloaded from  https://github.com/System-T/UniversalPropositions

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        up_fi_path = "https://raw.githubusercontent.com/System-T/UniversalPropositions/master/UP_Finnish/"
        cached_path(f"{up_fi_path}fi-up-train.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_fi_path}fi-up-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_fi_path}fi-up-test.conllu", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            train_file="fi-up-train.conllu",
            test_file="fi-up-test.conllu",
            dev_file="fi-up-dev.conllu",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            comment_symbol="#",
            **corpusargs,
        )


class UP_GERMAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the German dataset from the Universal Propositions Bank.

        The dataset is downloaded from  https://github.com/System-T/UniversalPropositions

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        up_de_path = "https://raw.githubusercontent.com/System-T/UniversalPropositions/master/UP_German/"
        cached_path(f"{up_de_path}de-up-train.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_de_path}de-up-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_de_path}de-up-test.conllu", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            train_file="de-up-train.conllu",
            test_file="de-up-test.conllu",
            dev_file="de-up-dev.conllu",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            comment_symbol="#",
            **corpusargs,
        )


class UP_ITALIAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the Italian dataset from the Universal Propositions Bank.

        The dataset is downloaded from  https://github.com/System-T/UniversalPropositions

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        up_it_path = "https://raw.githubusercontent.com/System-T/UniversalPropositions/master/UP_Italian/"
        cached_path(f"{up_it_path}it-up-train.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_it_path}it-up-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_it_path}it-up-test.conllu", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            train_file="it-up-train.conllu",
            test_file="it-up-test.conllu",
            dev_file="it-up-dev.conllu",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            comment_symbol="#",
            **corpusargs,
        )


class UP_SPANISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the Spanish dataset from the Universal Propositions Bank.

        The dataset is downloaded from  https://github.com/System-T/UniversalPropositions

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        up_es_path = "https://raw.githubusercontent.com/System-T/UniversalPropositions/master/UP_Spanish/"
        cached_path(f"{up_es_path}es-up-train.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_es_path}es-up-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_es_path}es-up-test.conllu", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            train_file="es-up-train.conllu",
            test_file="es-up-test.conllu",
            dev_file="es-up-dev.conllu",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            comment_symbol="#",
            **corpusargs,
        )


class UP_SPANISH_ANCORA(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ) -> None:
        """Initialize the Spanish AnCora dataset from the Universal Propositions Bank.

        The dataset is downloaded from https://github.com/System-T/UniversalPropositions

        Args:
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training.
            document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        up_es_path = "https://raw.githubusercontent.com/System-T/UniversalPropositions/master/UP_Spanish-AnCora/"
        cached_path(f"{up_es_path}es_ancora-up-train.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_es_path}es_ancora-up-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{up_es_path}es_ancora-up-test.conllu", Path("datasets") / dataset_name)

        super().__init__(
            data_folder,
            columns,
            encoding="utf-8",
            train_file="es_ancora-up-train.conllu",
            test_file="es_ancora-up-test.conllu",
            dev_file="es_ancora-up-dev.conllu",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            comment_symbol="#",
            **corpusargs,
        )


class NER_HIPE_2022(ColumnCorpus):
    @staticmethod
    def _prepare_corpus(
        file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
    ):
        with open(file_in, encoding="utf-8") as f_p:
            lines = f_p.readlines()

        with open(file_out, "w", encoding="utf-8") as f_out:
            # Add missing newline after header
            f_out.write(lines[0] + "\n")

            for line in lines[1:]:
                if line.startswith(" \t"):
                    # Workaround for empty tokens
                    continue

                line = line.strip()

                # Add "real" document marker
                if add_document_separator and line.startswith(document_separator):
                    f_out.write("-DOCSTART- O\n\n")

                f_out.write(line + "\n")

                if eos_marker in line:
                    f_out.write("\n")

    def __init__(
        self,
        dataset_name: str,
        language: str,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        version: str = "v2.1",
        branch_name: str = "main",
        dev_split_name="dev",
        add_document_separator=False,
        sample_missing_splits=False,
        preproc_fn=None,
        **corpusargs,
    ) -> None:
        """Initialize the CLEF-HIPE 2022 NER dataset.

        The first time you call this constructor it will automatically
        download the specified dataset (by given a language).
        :dataset_name: Supported datasets are: ajmc, hipe2020, letemps, newseye, sonar and topres19th.
        :language: Language for a supported dataset.
        :base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :in_memory: If True, keeps dataset in memory giving speedups in training.
        :version: Version of CLEF-HIPE dataset. Currently only v1.0 is supported and available.
        :branch_name: Defines git branch name of HIPE data repository (main by default).
        :dev_split_name: Defines default name of development split (dev by default). Only the NewsEye dataset has
        currently two development splits: dev and dev2.
        :add_document_separator: If True, a special document seperator will be introduced. This is highly
        recommended when using our FLERT approach.
        :sample_missing_splits: If True, data is automatically sampled when certain data splits are None.
        :preproc_fn: Function that is used for dataset preprocessing. If None, default preprocessing will be performed.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # Dataset split mapping
        hipe_available_splits = {
            "v1.0": {
                "ajmc": {"de": ["sample"], "en": ["sample"]},
                "hipe2020": {"de": ["train", "dev"], "en": ["dev"], "fr": ["train", "dev"]},
                "letemps": {"fr": ["train", "dev"]},
                "newseye": {
                    "de": ["train", "dev", "dev2"],
                    "fi": ["train", "dev", "dev2"],
                    "fr": ["train", "dev", "dev2"],
                    "sv": ["train", "dev", "dev2"],
                },
                "sonar": {"de": ["dev"]},
                "topres19th": {"en": ["train", "dev"]},
            }
        }

        # v2.0 only adds new language and splits for AJMC dataset
        hipe_available_splits["v2.0"] = copy.deepcopy(hipe_available_splits["v1.0"])
        hipe_available_splits["v2.0"]["ajmc"] = {"de": ["train", "dev"], "en": ["train", "dev"], "fr": ["train", "dev"]}

        hipe_available_splits["v2.1"] = copy.deepcopy(hipe_available_splits["v2.0"])
        for dataset_name_values in hipe_available_splits["v2.1"].values():
            for splits in dataset_name_values.values():
                splits.append("test")  # test datasets are only available for >= v2.1

        eos_marker = "EndOfSentence"
        document_separator = "# hipe2022:document_id"

        # Special document marker for sample splits in AJMC dataset
        if f"{dataset_name}" == "ajmc":
            document_separator = "# hipe2022:original_source"

        columns = {0: "text", 1: "ner"}

        dataset_base = self.__class__.__name__.lower()
        data_folder = base_path / dataset_base / version / dataset_name / language

        data_url = (
            f"https://github.com/hipe-eval/HIPE-2022-data/raw/{branch_name}/data/{version}/{dataset_name}/{language}"
        )

        dataset_splits = hipe_available_splits[version][dataset_name][language]

        for split in dataset_splits:
            cached_path(
                f"{data_url}/HIPE-2022-{version}-{dataset_name}-{split}-{language}.tsv", data_folder / "original"
            )

        train_file = "train.txt" if "train" in dataset_splits else None
        dev_file = f"{dev_split_name}.txt" if "sample" not in dataset_splits else "sample.txt"
        test_file = "test.txt" if "test" in dataset_splits else None

        new_data_folder = data_folder

        if add_document_separator:
            new_data_folder = new_data_folder / "with_doc_seperator"
            new_data_folder.mkdir(parents=True, exist_ok=True)

        self.preproc_fn = preproc_fn if preproc_fn else self._prepare_corpus

        if not all(  # Only reprocess if some files are not there yet
            split_path.exists()
            for split_path in [new_data_folder / f"{split_file}.txt" for split_file in dataset_splits]
        ):
            for split in dataset_splits:
                original_filename = f"HIPE-2022-{version}-{dataset_name}-{split}-{language}.tsv"
                self.preproc_fn(
                    data_folder / "original" / original_filename,
                    new_data_folder / f"{split}.txt",
                    eos_marker,
                    document_separator,
                    add_document_separator,
                )

        super().__init__(
            new_data_folder,
            columns,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            skip_first_line=True,
            column_delimiter="\t",
            comment_symbol="# ",
            sample_missing_splits=sample_missing_splits,
            **corpusargs,
        )


class NER_ICDAR_EUROPEANA(ColumnCorpus):
    def __init__(
        self,
        language: str,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the ICDAR Europeana NER dataset.

        The dataset is based on the French and Dutch Europeana NER corpora
        from the Europeana Newspapers NER dataset (https://lab.kb.nl/dataset/europeana-newspapers-ner), with additional
        preprocessing steps being performed (sentence splitting, punctuation normalizing, training/development/test splits).
        The resulting dataset is released in the "Data Centric Domain Adaptation for Historical Text with OCR Errors" ICDAR paper
        by Luisa März, Stefan Schweter, Nina Poerner, Benjamin Roth and Hinrich Schütze.
        :param language: Language for a supported dataset. Supported languages are "fr" (French) and "nl" (Dutch).
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training. Not recommended due to heavy RAM usage.
        """
        supported_languages = ["fr", "nl"]

        if language not in supported_languages:
            log.error(f"Language '{language}' is not in list of supported languages!")
            log.error(f"Supported are '{supported_languages}'!")
            raise Exception

        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name / language

        # download data if necessary
        github_path = "https://raw.githubusercontent.com/stefan-it/historic-domain-adaptation-icdar/main/data"

        for split in ["train", "dev", "test"]:
            cached_path(f"{github_path}/{language}/{split}.txt", data_folder)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="test.txt",
            comment_symbol="# ",
            column_delimiter="\t",
            **corpusargs,
        )


class NER_NERMUD(MultiCorpus):
    def __init__(
        self,
        domains: Union[str, list[str]] = "all",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
        **corpusargs,
    ) -> None:
        """Initilize the NERMuD 2023 dataset.

        NERMuD is a task presented at EVALITA 2023 consisting in the extraction and classification
        of named-entities in a document, such as persons, organizations, and locations. NERMuD 2023 will include two different sub-tasks:

        - Domain-agnostic classification (DAC). Participants will be asked to select and classify entities among three categories
          (person, organization, location) in different types of texts (news, fiction, political speeches) using one single general model.

        - Domain-specific classification (DSC). Participants will be asked to deploy a different model for each of the above types,
          trying to increase the accuracy for each considered type.

        Args:
            domains: Domains to be used. Supported are "WN" (Wikinews), "FIC" (fiction), "ADG" (De Gasperi subset) and "all".
            base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
            in_memory: If True, keeps dataset in memory giving speedups in training. Not recommended due to heavy RAM usage.
        """
        supported_domains = ["WN", "FIC", "ADG"]

        if isinstance(domains, str) and domains == "all":
            domains = supported_domains

        if isinstance(domains, str):
            domains = [domains]

        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        corpora: list[Corpus] = []

        github_path = "https://raw.githubusercontent.com/dhfbk/KIND/main/evalita-2023"

        for domain in domains:
            if domain not in supported_domains:
                log.error(f"Domain '{domain}' is not in list of supported domains!")
                log.error(f"Supported are '{supported_domains}'!")
                raise Exception

            domain_folder = data_folder / domain.lower()

            for split in ["train", "dev"]:
                cached_path(f"{github_path}/{domain}_{split}.tsv", domain_folder)

            corpus = ColumnCorpus(
                data_folder=domain_folder,
                train_file=f"{domain}_train.tsv",
                dev_file=f"{domain}_dev.tsv",
                test_file=None,
                column_format=columns,
                in_memory=in_memory,
                sample_missing_splits=False,  # No test data is available, so do not shrink dev data for shared task preparation!
                **corpusargs,
            )
            corpora.append(corpus)
        super().__init__(
            corpora,
            sample_missing_splits=False,
            name="nermud",
        )


class NER_GERMAN_MOBIE(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the German MobIE NER dataset.

        The German MobIE Dataset was introduced in the MobIE paper (https://aclanthology.org/2021.konvens-1.22/).

        This is a German-language dataset that has been human-annotated with 20 coarse- and fine-grained entity types,
        and it includes entity linking information for geographically linkable entities. The dataset comprises 3,232
        social media texts and traffic reports, totaling 91K tokens, with 20.5K annotated entities, of which 13.1K are
        linked to a knowledge base. In total, 20 different named entities are annotated.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training. Not recommended due to heavy RAM usage.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        dataset_name = self.__class__.__name__.lower()
        data_folder = base_path / dataset_name
        data_path = flair.cache_root / "datasets" / dataset_name

        columns = {0: "text", 3: "ner"}

        train_data_file = data_path / "train.conll2003"
        if not train_data_file.is_file():
            temp_file = cached_path(
                "https://github.com/DFKI-NLP/MobIE/raw/master/v1_20210811/ner_conll03_formatted.zip",
                Path("datasets") / dataset_name,
            )
            from zipfile import ZipFile

            with ZipFile(temp_file, "r") as zip_file:
                zip_file.extractall(path=data_path)

        super().__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            comment_symbol=None,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )


class NER_ESTONIAN_NOISY(ColumnCorpus):
    data_url = "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/patnlp/estner.cnll.zip"
    label_url = "https://raw.githubusercontent.com/uds-lsv/NoisyNER/master/data/only_labels"

    def __init__(
        self,
        version: int = 0,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the NoisyNER corpus.

        Args:
            version (int): Chooses the labelset for the data.
                v0 (default): Clean labels
                v1 to v7: Different kinds of noisy labelsets (details: https://ojs.aaai.org/index.php/AAAI/article/view/16938)
            base_path (Optional[Union[str, Path]]): Path to the data.
                Default is None, meaning the corpus gets automatically downloaded and saved.
                You can override this by passing a path to a directory containing the unprocessed files but typically this
                should not be necessary.
            in_memory (bool): If True the dataset is kept in memory achieving speedups in training.
            **corpusargs: The arguments propagated to :meth:'flair.datasets.ColumnCorpus.__init__'.
        """
        if version not in range(8):
            raise Exception(
                "Please choose a version (int) from 0 to 7. With v0 (default) you get the clean labelset for the data, while v1 to v7 provide different kinds of noisy labelsets. For details see https://ojs.aaai.org/index.php/AAAI/article/view/16938."
            )

        base_path = self._set_path(base_path)
        features = self._load_features(base_path)

        if version == 0:
            preinstances = self._process_clean_labels(features)
        else:
            rdcd_features = self._rmv_clean_labels(features)
            labels = self._load_noisy_labels(version, base_path)
            preinstances = self._process_noisy_labels(rdcd_features, labels)

        instances = self._delete_empty_labels(version, preinstances)

        train, dev, test = self._split_data(instances)

        self._write_instances(version, base_path, "train", train)
        self._write_instances(version, base_path, "dev", dev)
        self._write_instances(version, base_path, "test", test)

        super().__init__(
            data_folder=base_path,
            train_file=f"estner_noisy_labelset{version}_train.tsv",
            dev_file=f"estner_noisy_labelset{version}_dev.tsv",
            test_file=f"estner_noisy_labelset{version}_test.tsv",
            column_format={0: "text", 1: "ner"},
            in_memory=in_memory,
            column_delimiter="\t",
            **corpusargs,
        )

    @classmethod
    def _set_path(cls, base_path) -> Path:
        base_path = flair.cache_root / "datasets" / "estner" if not base_path else Path(base_path)
        return base_path

    @classmethod
    def _load_features(cls, base_path) -> list[list[str]]:
        print(base_path)
        unpack_file(cached_path(cls.data_url, base_path), base_path, "zip", False)
        with open(f"{base_path}/estner.cnll", encoding="utf-8") as in_file:
            prefeatures = in_file.readlines()
        features = [feature.strip().split("\t") for feature in prefeatures]
        return features

    @classmethod
    def _process_clean_labels(cls, features) -> list[list[str]]:
        preinstances = [[instance[0], instance[len(instance) - 1]] for instance in features]
        return preinstances

    @classmethod
    def _rmv_clean_labels(cls, features) -> list[str]:
        rdcd_features = [feature[:-1] for feature in features]
        return rdcd_features

    @classmethod
    def _load_noisy_labels(cls, version, base_path) -> list[str]:
        file_name = f"NoisyNER_labelset{version}.labels"
        cached_path(f"{cls.label_url}/{file_name}", base_path)
        with open(f"{base_path}/{file_name}", encoding="utf-8") as in_file:
            labels = in_file.read().splitlines()
        return labels

    @classmethod
    def _process_noisy_labels(cls, rdcd_features, labels) -> list[list[str]]:
        instances = []
        label_idx = 0
        for feature in rdcd_features:
            if len(feature) == 0:
                instances.append([""])
            else:
                assert label_idx < len(labels)
                instance = [feature[0], labels[label_idx]]
                instances.append(instance)
                label_idx += 1
        assert label_idx == len(labels), ""
        return instances

    @classmethod
    def _delete_empty_labels(cls, version, preinstances) -> list[str]:
        instances = []
        if version == 0:
            for instance in preinstances:
                if instance[0] != "--":
                    instances.append(instance)
        else:
            for instance in preinstances:
                if instance != "--":
                    instances.append(instance)
        return instances

    @classmethod
    def _split_data(cls, instances) -> tuple[list[str], list[str], list[str]]:
        train = instances[:185708]
        dev = instances[185708:208922]
        test = instances[208922:]
        return train, dev, test

    @classmethod
    def _write_instances(cls, version, base_path, split, data):
        column_separator = "\t"  # CoNLL format
        with open(f"{base_path}/estner_noisy_labelset{version}_{split}.tsv", "w", encoding="utf-8") as out_file:
            for instance in data:
                out_file.write(column_separator.join(instance))
                out_file.write("\n")


class MASAKHA_POS(MultiCorpus):
    def __init__(
        self,
        languages: Union[str, list[str]] = "bam",
        version: str = "v1",
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        """Initialize the MasakhaPOS corpus available on https://github.com/masakhane-io/masakhane-pos.

        It consists of 20 African languages. Pass a language code or a list of language codes to initialize the corpus
        with the languages you require. If you pass "all", all languages will be initialized.
        :version: Specifies version of the dataset. Currently, only "v1" is supported.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # if only one language is given
        if isinstance(languages, str):
            languages = [languages]

        # column format
        columns = {0: "text", 1: "pos"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        supported_versions = ["v1"]

        if version not in supported_versions:
            log.error(f"The specified version '{version}' is not in the list of supported version!")
            log.error(f"Supported versions are '{supported_versions}'!")
            raise Exception

        data_folder = base_path / dataset_name / version

        supported_languages = [
            "bam",
            "bbj",
            "ewe",
            "fon",
            "hau",
            "ibo",
            "kin",
            "lug",
            "luo",
            "mos",
            "pcm",
            "nya",
            "sna",
            "swa",
            "tsn",
            "twi",
            "wol",
            "xho",
            "yor",
            "zul",
        ]

        data_paths = {
            "v1": "https://raw.githubusercontent.com/masakhane-io/masakhane-pos/main/data",
        }

        # use all languages if explicitly set to "all"
        if languages == ["all"]:
            languages = supported_languages

        corpora: list[Corpus] = []
        for language in languages:
            if language not in supported_languages:
                log.error(f"Language '{language}' is not in list of supported languages!")
                log.error(f"Supported are '{supported_languages}'!")
                log.error("Instantiate this Corpus for instance like so 'corpus = MASAKHA_POS(languages='bam')'")
                raise Exception

            language_folder = data_folder / language

            # download data if necessary
            data_path = f"{data_paths[version]}/{language}"
            cached_path(f"{data_path}/dev.txt", language_folder)
            cached_path(f"{data_path}/test.txt", language_folder)
            cached_path(f"{data_path}/train.txt", language_folder)

            # initialize comlumncorpus and add it to list
            log.info(f"Reading data for language {language}@{version}")
            corp = ColumnCorpus(
                data_folder=language_folder,
                column_format=columns,
                encoding="utf-8",
                in_memory=in_memory,
                name=language,
                **corpusargs,
            )
            corpora.append(corp)
        super().__init__(
            corpora,
            name="masakha-pos-" + "-".join(languages),
        )
