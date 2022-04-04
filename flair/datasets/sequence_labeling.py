import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import ConcatDataset, Dataset

import flair
from flair.data import Corpus, FlairDataset, MultiCorpus, Relation, Sentence, Token
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path, unpack_file
from flair.models.sequence_tagger_utils.bioes import get_spans_from_bio

log = logging.getLogger("flair")


class MultiFileJsonlCorpus(Corpus):
    """
    This class represents a generic Jsonl corpus with multiple train, dev, and test files.
    """

    def __init__(
        self,
        train_files=None,
        test_files=None,
        dev_files=None,
        encoding: str = "utf-8",
        text_column_name: str = "data",
        label_column_name: str = "label",
        label_type: str = "ner",
        **corpusargs,
    ):
        """
        Instantiates a MuliFileJsonlCorpus as, e.g., created with doccanos JSONL export.
        Note that at least one of train_files, test_files, and dev_files must contain one path.
        Otherwise, the initialization will fail.

        :param corpusargs: Additional arguments for Corpus initialization
        :param train_files: the name of the train files
        :param test_files: the name of the test files
        :param dev_files: the name of the dev files, if empty, dev data is sampled from train
        :param text_column_name: Name of the text column inside the jsonl files.
        :param label_column_name: Name of the label column inside the jsonl files.

        :raises RuntimeError: If no paths are given
        """
        train: Optional[Dataset] = (
            ConcatDataset(
                [
                    JsonlDataset(
                        train_file,
                        text_column_name=text_column_name,
                        label_column_name=label_column_name,
                        label_type=label_type,
                        encoding=encoding,
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
                        label_type=label_type,
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
                        label_type=label_type,
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
        label_type: str = "ner",
        autofind_splits: bool = True,
        name: Optional[str] = None,
        **corpusargs,
    ):
        """
        Instantiates a JsonlCorpus with one file per Dataset (train, dev, and test).

        :param data_folder: Path to the folder containing the JSONL corpus
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param text_column_name: Name of the text column inside the JSONL file.
        :param label_column_name: Name of the label column inside the JSONL file.
        :param autofind_splits: Whether train, test and dev file should be determined automatically
        :param name: name of the Corpus see flair.data.Corpus
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
            label_type=label_type,
            name=name if data_folder is None else str(data_folder),
            encoding=encoding,
            **corpusargs,
        )


class JsonlDataset(FlairDataset):
    def __init__(
        self,
        path_to_jsonl_file: Union[str, Path],
        encoding: str = "utf-8",
        text_column_name: str = "data",
        label_column_name: str = "label",
        label_type: str = "ner",
    ):
        """
        Instantiates a JsonlDataset and converts all annotated char spans to token tags using the IOB scheme.

        The expected file format is:
        { "<text_column_name>": "<text>", "label_column_name": [[<start_char_index>, <end_char_index>, <label>],...] }

        :param path_to_json._file: File to read
        :param text_column_name: Name of the text column
        :param label_column_name: Name of the label column

        """
        path_to_json_file = Path(path_to_jsonl_file)

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.label_type = label_type
        self.path_to_json_file = path_to_json_file

        self.sentences: List[Sentence] = []
        with path_to_json_file.open(encoding=encoding) as jsonl_fp:
            for line in jsonl_fp:
                current_line = json.loads(line)
                raw_text = current_line[text_column_name]
                current_labels = current_line[label_column_name]
                current_sentence = Sentence(raw_text)

                self._add_labels_to_sentence(raw_text, current_sentence, current_labels)

                self.sentences.append(current_sentence)

    def _add_labels_to_sentence(self, raw_text: str, sentence: Sentence, labels: List[List[Any]]):
        # Add tags for each annotated span
        for label in labels:
            self._add_label_to_sentence(raw_text, sentence, label[0], label[1], label[2])

        # Tag all other token as Outer (O)
        for token in sentence:
            if token.get_label(self.label_type).value == "":
                token.get_label(self.label_type, "O")

    def _add_label_to_sentence(self, text: str, sentence: Sentence, start: int, end: int, label: str):
        """
        Adds a NE label to a given sentence.

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
            if token.start_pos <= start <= token.end_pos and start_idx == -1:
                start_idx = token.idx - 1

            if token.start_pos <= end <= token.end_pos and end_idx == -1:
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

        # Add IOB tags
        prefix = "B"
        for token in sentence[start_idx : end_idx + 1]:
            token.add_label(self.label_type, f"{prefix}-{label}")
            prefix = "I"

    def is_in_memory(self) -> bool:
        """
        Currently all Jsonl Datasets are stored in Memory
        """
        return True

    def __len__(self):
        """
        Number of sentences in the Dataset
        """
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        """
        Returns the sentence at a given index
        """
        return self.sentences[index]


class MultiFileColumnCorpus(Corpus):
    def __init__(
        self,
        column_format: Dict[int, str],
        train_files=None,
        test_files=None,
        dev_files=None,
        tag_to_bioes=None,
        column_delimiter: str = r"\s+",
        comment_symbol: str = None,
        encoding: str = "utf-8",
        document_separator_token: str = None,
        skip_first_line: bool = False,
        in_memory: bool = True,
        label_name_map: Dict[str, str] = None,
        banned_sentences: List[str] = None,
        default_whitespace_after: bool = True,
        **corpusargs,
    ):
        """
        Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.
        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_files: the name of the train files
        :param test_files: the name of the test files
        :param dev_files: the name of the dev files, if empty, dev data is sampled from train
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param column_delimiter: default is to split on any separatator, but you can overwrite for instance with "\t"
        to split only on tabs
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param document_separator_token: If provided, sentences that function as document boundaries are so marked
        :param skip_first_line: set to True if your dataset has a header line
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param label_name_map: Optionally map tag names to different schema.
        :param banned_sentences: Optionally remove sentences from the corpus. Works only if `in_memory` is true
        :return: a Corpus with annotated train, dev and test data
        """
        # get train data
        train: Optional[Dataset] = (
            ConcatDataset(
                [
                    ColumnDataset(
                        train_file,
                        column_format,
                        tag_to_bioes,
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
                        tag_to_bioes,
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
                        tag_to_bioes,
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

        super(MultiFileColumnCorpus, self).__init__(train, dev, test, **corpusargs)


class ColumnCorpus(MultiFileColumnCorpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        autofind_splits: bool = True,
        name: Optional[str] = None,
        comment_symbol="# ",
        **corpusargs,
    ):
        """
        Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.
        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param column_delimiter: default is to split on any separatator, but you can overwrite for instance with "\t"
        to split only on tabs
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param document_separator_token: If provided, sentences that function as document boundaries are so marked
        :param skip_first_line: set to True if your dataset has a header line
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param label_name_map: Optionally map tag names to different schema.
        :param banned_sentences: Optionally remove sentences from the corpus. Works only if `in_memory` is true
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(
            data_folder, dev_file, test_file, train_file, autofind_splits
        )
        super(ColumnCorpus, self).__init__(
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
        column_name_map: Dict[int, str],
        tag_to_bioes: str = None,
        column_delimiter: str = r"\s+",
        comment_symbol: str = None,
        banned_sentences: List[str] = None,
        in_memory: bool = True,
        document_separator_token: str = None,
        encoding: str = "utf-8",
        skip_first_line: bool = False,
        label_name_map: Dict[str, str] = None,
        default_whitespace_after: bool = True,
    ):
        """
        Instantiates a column dataset (typically used for sequence labeling or word-level prediction).
        :param path_to_column_file: path to the file with the column-formatted data
        :param column_name_map: a map specifying the column format
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param column_delimiter: default is to split on any separatator, but you can overwrite for instance with "\t"
        to split only on tabs
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param document_separator_token: If provided, sentences that function as document boundaries are so marked
        :param skip_first_line: set to True if your dataset has a header line
        :param label_name_map: Optionally map tag names to different schema.
        :param banned_sentences: Optionally remove sentences from the corpus. Works only if `in_memory` is true
        :return: a dataset with annotated data
        """
        path_to_column_file = Path(path_to_column_file)
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.tag_to_bioes = tag_to_bioes
        self.column_delimiter = column_delimiter
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
                self.sentences: List[Sentence] = []

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
                        [d in sentence.to_plain_string() for d in self.banned_sentences]
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
                self.sentences_raw: List[List[str]] = []

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

            sentence_1 = self._convert_lines_to_sentence(
                self._read_next_sentence(file), word_level_tag_columns=column_name_map
            )
            # sentence_2 = self._convert_lines_to_sentence(self._read_next_sentence(file),
            #                                              word_level_tag_columns=column_name_map)

            for sentence in [sentence_1]:
                # go through all annotations
                for column in column_name_map:
                    if column == self.text_column or column == self.head_id_column:
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
                        if token.get_label(layer, "O").value != "O" and token.get_label(layer).value[0:2] not in [
                            "B-",
                            "I-",
                            "E-",
                            "S-",
                        ]:
                            self.word_level_tag_columns[column] = layer
                            break
                    if column not in self.word_level_tag_columns:
                        self.span_level_tag_columns[column] = layer

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
        self, lines, word_level_tag_columns: Dict[int, str], span_level_tag_columns: Optional[Dict[int, str]] = None
    ):

        sentence: Sentence = Sentence(text=[])
        token: Optional[Token] = None
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
            sentence.add_token(token)

        # check if this sentence is a document boundary
        if sentence.to_original_text() == self.document_separator_token:
            sentence.is_document_boundary = True

        # add span labels
        if span_level_tag_columns:
            for span_column in span_level_tag_columns:
                try:
                    bioes_tags = [
                        re.split(self.column_delimiter, line.rstrip())[span_column] for line in filtered_lines
                    ]
                    predicted_spans = get_spans_from_bio(bioes_tags)
                    for span_indices, score, label in predicted_spans:
                        span = sentence[span_indices[0] : span_indices[-1] + 1]
                        value = self._remap_label(label)
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
                    relation.add_label(typename="relation", value=self._remap_label(label))

        if len(sentence) > 0:
            return sentence

    def _parse_token(self, line: str, column_name_map: Dict[int, str], last_token: Optional[Token] = None) -> Token:

        # get fields from line
        fields: List[str] = re.split(self.column_delimiter, line.rstrip())

        # get head_id if exists (only in dependency parses)
        head_id = int(fields[self.head_id_column]) if self.head_id_column else None

        # initialize token
        token = Token(fields[self.text_column], head_id=head_id, whitespace_after=self.default_whitespace_after)

        # go through all columns
        for column in column_name_map:
            if len(fields) > column:
                if (
                    column != self.text_column
                    and column != self.head_id_column
                    and column_name_map[column] != self.SPACE_AFTER_KEY
                ):

                    # 'feats' and 'misc' column should be split into different fields
                    if column_name_map[column] in self.FEATS:
                        for feature in fields[column].split("|"):

                            # special handling for whitespace after
                            if feature == "SpaceAfter=No":
                                token.whitespace_after = False
                                continue

                            if "=" in feature:
                                # add each other feature as label-value pair
                                label_name = feature.split("=")[0]
                                label_value = self._remap_label(feature.split("=")[1])
                                token.add_label(label_name, label_value)

                    else:
                        # get the task name (e.g. 'ner')
                        label_name = column_name_map[column]
                        # get the label value
                        label_value = self._remap_label(fields[column])
                        # add label
                        token.add_label(label_name, label_value)

                if column_name_map[column] == self.SPACE_AFTER_KEY and fields[column] == "-":
                    token.whitespace_after = False
        if last_token is None:
            start = 0
        else:
            assert last_token.end_pos is not None
            start = last_token.end_pos
            if last_token.whitespace_after:
                start += 1
        token.start_pos = start
        token.end_pos = token.start_pos + len(token.text)
        return token

    def _remap_label(self, tag):
        # remap regular tag names
        if self.label_name_map and tag in self.label_name_map.keys():
            tag = self.label_name_map[tag]  # for example, transforming 'PER' to 'person'
        return tag

    def __line_completes_sentence(self, line: str) -> bool:
        sentence_completed = line.isspace() or line == ""
        return sentence_completed

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
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
            sentence._position_in_dataset = (self, index)

        return sentence


class CONLL_03(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        column_format={0: "text", 1: "pos", 3: "ner"},
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put the eng.testa, .testb, .train
        files in a folder called 'conll_03'. Then set the base_path parameter in the constructor to the path to the
        parent directory where the conll_03 folder resides.
        If using entity linking, the conll03 dateset is reduced by about 20 Documents, which are not part of the yago dataset.
        :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03' folder) on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(CONLL_03, self).__init__(
            data_folder,
            column_format=column_format,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )


class CONLL_03_GERMAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the CoNLL-03 corpus for German. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put the respective files in a folder called
        'conll_03_german'. Then set the base_path parameter in the constructor to the path to the parent directory where
        the conll_03_german folder resides.
        :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03_german' folder) on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'lemma', 'pos' or 'np' to predict
        word lemmas, POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(CONLL_03_GERMAN, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )


class CONLL_03_DUTCH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the CoNLL-03 corpus for Dutch. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(CONLL_03_DUTCH, self).__init__(
            data_folder,
            columns,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="test.txt",
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )

    @staticmethod
    def __offset_docstarts(file_in: Union[str, Path], file_out: Union[str, Path]):
        with open(file_in, "r", encoding="latin-1") as f:
            lines = f.readlines()
        with open(file_out, "w", encoding="latin-1") as f:
            for line in lines:
                f.write(line)
                if line.startswith("-DOCSTART-"):
                    f.write("\n")


class CONLL_03_SPANISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the CoNLL-03 corpus for Spanish. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, should not be changed
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(CONLL_03_SPANISH, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            **corpusargs,
        )


class CONLL_2000(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "np",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the CoNLL-2000 corpus for English chunking.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: 'np' by default, should not be changed, but you can set 'pos' instead to predict POS tags
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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
            import shutil

            with gzip.open(
                flair.cache_root / "datasets" / dataset_name / "train.txt.gz",
                "rb",
            ) as f_in:
                with open(
                    flair.cache_root / "datasets" / dataset_name / "train.txt",
                    "wb",
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            with gzip.open(flair.cache_root / "datasets" / dataset_name / "test.txt.gz", "rb") as f_in:
                with open(
                    flair.cache_root / "datasets" / dataset_name / "test.txt",
                    "wb",
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)

        super(CONLL_2000, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class WNUT_17(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(WNUT_17, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class BIOSCOPE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        **corpusargs,
    ):
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(BIOSCOPE, self).__init__(
            data_folder,
            columns,
            in_memory=in_memory,
            train_file="output.txt",
            **corpusargs,
        )


class NER_ARABIC_ANER(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize a preprocessed version of the Arabic Named Entity Recognition Corpus (ANERCorp) dataset available
        from https://github.com/EmnamoR/Arabic-named-entity-recognition/blob/master/ANERCorp.rar.
        http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp
        Column order is swapped
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_ARABIC_ANER, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="utf-8",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_ARABIC_AQMAR(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize a preprocessed and modified version of the American and Qatari Modeling of Arabic (AQMAR) dataset available
        from http://www.cs.cmu.edu/~ark/ArabicNER/AQMAR_Arabic_NER_corpus-1.0.zip.
        via http://www.cs.cmu.edu/~ark/AQMAR/

        - Modifications from original dataset: Miscellaneous tags (MIS0, MIS1, MIS2, MIS3) are merged to one tag "MISC" as these categories deviate across the original dataset
        - The 28 original Wikipedia articles are merged into a single file containing the articles in alphabetical order

        The first time you call this constructor it will automatically download the dataset.

        This dataset is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License.
        please cite: "Behrang Mohit, Nathan Schneider, Rishav Bhowmick, Kemal Oflazer, and Noah A. Smith (2012),
        Recall-Oriented Learning of Named Entities in Arabic Wikipedia. Proceedings of EACL."

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_ARABIC_AQMAR, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="utf-8",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_BASQUE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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
            import shutil
            import tarfile

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

        super(NER_BASQUE, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_CHINESE_WEIBO(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the WEIBO_NER corpus . The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_CHINESE_WEIBO, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
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
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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
                data_file = "ddt.{}.conllu".format(part)
                with open(data_path / data_file, "r") as file:
                    for line in file:
                        if line.startswith("#") or line == "\n":
                            lines.append(line)
                        lines.append(line.replace("name=", "").replace("|SpaceAfter=No", ""))

                with open(data_path / data_file, "w") as file:
                    file.writelines(lines)

        super(NER_DANISH_DANE, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            comment_symbol="#",
            **corpusargs,
        )


class NER_ENGLISH_MOVIE_SIMPLE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the eng corpus of the MIT Movie Corpus (it has simpler queries compared to the trivia10k13 corpus)
        in BIO format. The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        # column format
        columns = {0: "ner", 1: "text"}

        # dataset name
        dataset_name = self.__class__.__name__.lower()

        # data folder: default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)
        data_folder = base_path / dataset_name

        # download data if necessary
        mit_movie_path = "https://groups.csail.mit.edu/sls/downloads/movie/"
        train_file = "engtrain.bio"
        test_file = "engtest.bio"
        cached_path(f"{mit_movie_path}{train_file}", Path("datasets") / dataset_name)
        cached_path(f"{mit_movie_path}{test_file}", Path("datasets") / dataset_name)

        super(NER_ENGLISH_MOVIE_SIMPLE, self).__init__(
            data_folder,
            columns,
            train_file=train_file,
            test_file=test_file,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_MOVIE_COMPLEX(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the trivia10k13 corpus of the MIT Movie Corpus (it has more complex queries compared to the eng corpus)
        in BIO format. The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        # column format
        columns = {0: "ner", 1: "text"}

        # dataset name
        dataset_name = self.__class__.__name__.lower()

        # data folder: default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)
        data_folder = base_path / dataset_name

        # download data if necessary
        mit_movie_path = "https://groups.csail.mit.edu/sls/downloads/movie/"
        train_file = "trivia10k13train.bio"
        test_file = "trivia10k13test.bio"
        cached_path(f"{mit_movie_path}{train_file}", Path("datasets") / dataset_name)
        cached_path(f"{mit_movie_path}{test_file}", Path("datasets") / dataset_name)

        super(NER_ENGLISH_MOVIE_COMPLEX, self).__init__(
            data_folder,
            columns,
            train_file=train_file,
            test_file=test_file,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_SEC_FILLINGS(ColumnCorpus):
    """
    Initialize corpus of SEC-fillings annotated with English NER tags. See paper "Domain Adaption of Named Entity
    Recognition to Support Credit Risk Assessment" by Alvarado et al, 2015: https://aclanthology.org/U15-1010/
    :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03' folder) on your machine
    :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
    POS tags or chunks respectively
    :param in_memory: If True, keeps dataset in memory giving speedups in training.
    :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        SEC_FILLINGS_Path = "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/SEC-filings/CONLL-format/data/"
        cached_path(f"{SEC_FILLINGS_Path}test/FIN3.txt", Path("datasets") / dataset_name)
        cached_path(f"{SEC_FILLINGS_Path}train/FIN5.txt", Path("datasets") / dataset_name)

        super(NER_ENGLISH_SEC_FILLINGS, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
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
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the experimental MIT Restaurant corpus available on https://groups.csail.mit.edu/sls/downloads/restaurant/.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        mit_restaurants_path = "https://megantosh.s3.eu-central-1.amazonaws.com/MITRestoCorpus/"
        cached_path(f"{mit_restaurants_path}test.txt", Path("datasets") / dataset_name)
        cached_path(f"{mit_restaurants_path}train.txt", Path("datasets") / dataset_name)

        super(NER_ENGLISH_RESTAURANT, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_STACKOVERFLOW(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the STACKOVERFLOW_NER corpus. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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
            for line in open(data_folder / (file + ".txt"), mode="r", encoding="utf-8"):
                if line.startswith("Question_ID"):
                    questions += 1

                if line.startswith("Answer_to_Question_ID"):
                    answers += 1
            log.info(f"File {file} has {questions} questions and {answers} answers.")

        super(NER_ENGLISH_STACKOVERFLOW, self).__init__(
            data_folder,
            columns,
            train_file="train.txt",
            test_file="test.txt",
            dev_file="dev.txt",
            tag_to_bioes=tag_to_bioes,
            encoding="utf-8",
            banned_sentences=banned_sentences,
            in_memory=in_memory,
            label_name_map=entity_mapping,
            **corpusargs,
        )


class NER_ENGLISH_TWITTER(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize a dataset called twitter_ner which can be found on the following page:
        https://raw.githubusercontent.com/aritter/twitter_nlp/master/data/annotated/ner.txt.

        The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_ENGLISH_TWITTER, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            train_file="ner.txt",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_PERSON(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
    ):
        """
        Initialize the PERSON_NER corpus for person names. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_ENGLISH_PERSON, self).__init__(data_folder, columns, in_memory=in_memory, train_file="bigFile.conll")

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
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the WEBPAGES_NER corpus introduced in the paper "Design Challenges and Misconceptions in Named Entity
        Recognition" by Ratinov and Roth (2009): https://aclanthology.org/W09-1119/.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "ner", 5: "text"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name
        import tarfile

        if not os.path.isfile(data_folder / "webpages_ner.txt"):
            #     # download zip
            tar_file = "https://cogcomp.seas.upenn.edu/Data/NERWebpagesColumns.tgz"
            webpages_ner_path = cached_path(tar_file, Path("datasets") / dataset_name)
            tf = tarfile.open(webpages_ner_path)
            tf.extractall(data_folder)
            tf.close()
        outputfile = os.path.abspath(data_folder)

        # merge the files in one as the zip is containing multiples files

        with open(outputfile / data_folder / "webpages_ner.txt", "w+") as outfile:
            for files in os.walk(outputfile):
                f = files[1]
                ff = os.listdir(outputfile / data_folder / f[-1])
                for i, file in enumerate(ff):
                    if file.endswith(".gold"):
                        with open(
                            outputfile / data_folder / f[-1] / file,
                            "r+",
                            errors="replace",
                        ) as infile:
                            content = infile.read()
                        outfile.write(content)
                break

        super(NER_ENGLISH_WEBPAGES, self).__init__(
            data_folder,
            columns,
            train_file="webpages_ner.txt",
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_ENGLISH_WNUT_2020(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the WNUT_2020_NER corpus. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, since it is the only option of the WNUT corpus.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_ENGLISH_WNUT_2020, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="utf-8",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_ENGLISH_WIKIGOLD(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the wikigold corpus. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, should not be changed
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        wikigold_ner_path = "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/wikigold/CONLL-format/data/"
        cached_path(f"{wikigold_ner_path}wikigold.conll.txt", Path("datasets") / dataset_name)

        super(NER_ENGLISH_WIKIGOLD, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="utf-8",
            in_memory=in_memory,
            train_file="wikigold.conll.txt",
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_FINNISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_FINNISH, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            skip_first_line=True,
            **corpusargs,
        )

    def _remove_lines_without_annotations(self, data_file: Union[str, Path]):
        with open(data_file, "r") as f:
            lines = f.readlines()
        with open(data_file, "w") as f:
            for line in lines:
                if len(line.split()) != 1:
                    f.write(line)


class NER_GERMAN_BIOFID(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_GERMAN_BIOFID, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_GERMAN_EUROPARL(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the EUROPARL_NER_GERMAN corpus. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: 'ner' by default, should not be changed.
        :param in_memory: If True, keeps dataset in memory giving speedups in training. Not recommended due to heavy RAM usage.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_GERMAN_EUROPARL, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            train_file="ep-96-04-16.conll",
            test_file="ep-96-04-15.conll",
            **corpusargs,
        )

    def _add_IOB_tags(self, data_file: Union[str, Path], encoding: str = "utf8", ner_column: int = 1):
        """
        Function that adds IOB tags if only chunk names are provided (e.g. words are tagged PER instead
        of B-PER or I-PER). Replaces '0' with 'O' as the no-chunk tag since ColumnCorpus expects
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

        def add_I_prefix(current_line: List[str], ner: int, tag: str):
            for i in range(0, len(current_line)):
                if i == 0:
                    f.write(line_list[i])
                elif i == ner:
                    f.write(" I-" + tag)
                else:
                    f.write(" " + current_line[i])
            f.write("\n")

        with open(file=data_file, mode="r", encoding=encoding) as f:
            lines = f.readlines()
        with open(file=data_file, mode="w", encoding=encoding) as f:
            pred = "O"  # remembers ner tag of predecessing line
            for line in lines:
                line_list = line.split()
                if len(line_list) > 2:  # word with tags
                    ner_tag = line_list[ner_column]
                    if ner_tag in ["0", "O"]:  # no chunk
                        for i in range(0, len(line_list)):
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
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the LER_GERMAN (Legal Entity Recognition) corpus. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training. Not recommended due to heavy RAM usage.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ler_path = "https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/"
        cached_path(f"{ler_path}ler.conll", Path("datasets") / dataset_name)

        super(NER_GERMAN_LEGAL, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            train_file="ler.conll",
            **corpusargs,
        )


class NER_GERMAN_GERMEVAL(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the GermEval NER corpus for German. This is only possible if you've manually downloaded it to your
        machine. Obtain the corpus from https://sites.google.com/site/germeval2014ner/data and put it into some folder.
        Then point the base_path parameter in the constructor to this folder
        :param base_path: Path to the GermEval corpus on your machine
        :param tag_to_bioes: 'ner' by default, should not be changed.
        :param in_memory:If True, keeps dataset in memory giving speedups in training.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_GERMAN_GERMEVAL, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            comment_symbol="#",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_GERMAN_POLITICS(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        column_delimiter: str = r"\s+",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize corpus with Named Entity Model for German, Politics (NEMGP) data from
        https://www.thomas-zastrow.de/nlp/. The first time you call this constructor it will automatically download the
        dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_GERMAN_POLITICS, self).__init__(
            data_folder,
            columns,
            column_delimiter=column_delimiter,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="test.txt",
            tag_to_bioes=tag_to_bioes,
            encoding="utf-8",
            in_memory=in_memory,
            **corpusargs,
        )

    def _convert_to_column_corpus(self, data_file: Union[str, Path]):
        with open(data_file, "r", encoding="utf-8") as f:
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
        with open(data_file, "r") as file:
            num_lines = len(file.readlines())
            file.seek(0)

            train_len = round(num_lines * 0.8)
            test_len = round(num_lines * 0.1)

            train = open(data_folder / "train.txt", "w")
            test = open(data_folder / "test.txt", "w")
            dev = open(data_folder / "dev.txt", "w")

            k = 0
            for line in file.readlines():
                k += 1
                if k <= train_len:
                    train.write(line)
                elif k > train_len and k <= (train_len + test_len):
                    test.write(line)
                elif k > (train_len + test_len) and k <= num_lines:
                    dev.write(line)


class NER_HUNGARIAN(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the NER Business corpus for Hungarian. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_HUNGARIAN, self).__init__(
            data_folder,
            columns,
            train_file="hun_ner_corpus.txt",
            column_delimiter="\t",
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            label_name_map={"0": "O"},
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
            **corpusargs,
        )


class NER_ICELANDIC(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the ICELANDIC_NER corpus. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_ICELANDIC, self).__init__(
            data_folder,
            columns,
            train_file="icelandic_ner.txt",
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class NER_JAPANESE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the Hironsan/IOB2 corpus for Japanese. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_JAPANESE, self).__init__(
            data_folder,
            columns,
            train_file="train.txt",
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            default_whitespace_after=False,
            **corpusargs,
        )

    @staticmethod
    def __prepare_jap_wikipedia_corpus(file_in: Union[str, Path], file_out: Union[str, Path]):
        with open(file_in, "r") as f:
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
        with open(file_in, "r") as f:
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
        languages: Union[str, List[str]] = "luo",
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the Masakhane corpus available on https://github.com/masakhane-io/masakhane-ner/tree/main/data.
        It consists of ten African languages. Pass a language code or a list of language codes to initialize the corpus
        with the languages you require. If you pass "all", all languages will be initialized.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # if only one language is given
        if type(languages) == str:
            languages = [languages]

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        language_to_code = {
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
        }

        # use all languages if explicitly set to "all"
        if languages == ["all"]:
            languages = list(language_to_code.values())

        corpora: List[Corpus] = []
        for language in languages:

            if language in language_to_code.keys():
                language = language_to_code[language]

            if language not in language_to_code.values():
                log.error(f"Language '{language}' is not in list of supported languages!")
                log.error(f"Supported are '{language_to_code.values()}'!")
                log.error("Instantiate this Corpus for instance like so 'corpus = NER_MASAKHANE(languages='luo')'")
                raise Exception()

            language_folder = data_folder / language

            # download data if necessary
            data_path = f"https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/{language}/"
            cached_path(f"{data_path}dev.txt", language_folder)
            cached_path(f"{data_path}test.txt", language_folder)
            cached_path(f"{data_path}train.txt", language_folder)

            # initialize comlumncorpus and add it to list
            log.info(f"Reading data for language {language}")
            corp = ColumnCorpus(
                data_folder=language_folder,
                column_format=columns,
                tag_to_bioes=tag_to_bioes,
                encoding="utf-8",
                in_memory=in_memory,
                name=language,
                **corpusargs,
            )
            corpora.append(corp)

        super(NER_MASAKHANE, self).__init__(
            corpora,
            name="masakhane-" + "-".join(languages),
        )


class NER_MULTI_CONER(MultiFileColumnCorpus):
    def __init__(
        self,
        task: str = "multi",
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        use_dev_as_test: bool = True,
        **corpusargs,
    ):
        """
        Initialize the MultiCoNer corpus. This is only possible if you've applied and downloaded it to your machine.
        Apply for the corpus from here https://multiconer.github.io/dataset and unpack the .zip file's content into
        a folder called 'multiconer'. Then set the base_path parameter in the constructor to the path to the
        parent directory where the multiconer folder resides. You can also create the multiconer in
        the {FLAIR_CACHE_ROOT}/datasets folder to leave the path empty.
        :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03' folder) on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param use_dev_as_test: If True, it uses the dev set as test set and samples random training data for a dev split.
        :param task: either 'multi', 'code-switch', or the language code for one of the mono tasks.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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
        }

        possible_tasks = ["multi", "code-switch"] + list(folders.keys())
        task = task.lower()

        if task not in possible_tasks:
            raise ValueError(f"task has to be one of {possible_tasks}, but is '{task}'")

        # column format
        columns = {0: "text", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'WARNING: MultiCoNer dataset not found at "{data_folder}".')
            log.warning('Instructions for obtaining the data can be found here: https://multiconer.github.io/dataset"')
            log.warning("-" * 100)

        if task in ["multi", "code-switch"]:
            # code-switch uses the same training data than multi but provides a different test set.
            # as the test set is not published, those two tasks are the same.
            train_files = list(data_folder.rglob("*_train.conll"))
            dev_files = list(data_folder.rglob("*_dev.conll"))
        else:
            train_files = [data_folder / folders[task] / f"{task}_train.conll"]
            dev_files = [data_folder / folders[task] / f"{task}_dev.conll"]

        if use_dev_as_test:
            test_files = dev_files
            dev_files = []
        else:
            test_files = []

        super(NER_MULTI_CONER, self).__init__(
            train_files=train_files,
            dev_files=dev_files,
            test_files=test_files,
            column_format=columns,
            tag_to_bioes=tag_to_bioes,
            comment_symbol="# id ",
            in_memory=in_memory,
            **corpusargs,
        )


class NER_MULTI_WIKIANN(MultiCorpus):
    def __init__(
        self,
        languages: Union[str, List[str]] = "en",
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
        **corpusargs,
    ):
        """
        WkiAnn corpus for cross-lingual NER consisting of datasets from 282 languages that exist
        in Wikipedia. See https://elisa-ie.github.io/wikiann/ for details and for the languages and their
        respective abbreveations, i.e. "en" for english. (license: https://opendatacommons.org/licenses/by/)
        Parameters
        ----------
        languages : Union[str, List[str]]
            Should be an abbreviation of a language ("en", "de",..) or a list of abbreviations.
            The datasets of all passed languages will be saved in one MultiCorpus.
            (Note that, even though listed on https://elisa-ie.github.io/wikiann/ some datasets are empty.
            This includes "aa", "cho", "ho", "hz", "ii", "jam", "kj", "kr", "mus", "olo" and "tcy".)
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        tag_to_bioes : str, optional
            The data is in bio-format. It will by default (with the string "ner" as value) be transformed
            into the bioes format. If you dont want that set it to None.

        """
        if type(languages) == str:
            languages = [languages]

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # For each language in languages, the file is downloaded if not existent
        # Then a comlumncorpus of that data is created and saved in a list
        # this list is handed to the multicorpus

        # list that contains the columncopora
        corpora: List[Corpus] = []

        google_drive_path = "https://drive.google.com/uc?id="
        # download data if necessary
        first = True
        for language in languages:

            language_folder = data_folder / language
            file_name = "wikiann-" + language + ".bio"

            # if language not downloaded yet, download it
            if not language_folder.exists():
                if first:
                    import tarfile

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
                tar = tarfile.open(str(language_folder / language) + ".tar.gz", "r:gz")
                # tar.extractall(language_folder,members=[tar.getmember(file_name)])
                tar.extract(file_name, str(language_folder))
                tar.close()
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
                tag_to_bioes=tag_to_bioes,
                in_memory=in_memory,
                **corpusargs,
            )
            corpora.append(corp)
            log.info("...done.")

        super(NER_MULTI_WIKIANN, self).__init__(
            corpora,
            name="wikiann",
        )

    def _silver_standard_to_simple_ner_annotation(self, data_file: Union[str, Path]):
        with open(data_file, "r", encoding="utf-8") as f_read, open(
            str(data_file) + "_new", "w+", encoding="utf-8"
        ) as f_write:
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
        languages: Union[str, List[str]] = "en",
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
        **corpusargs,
    ):
        """
        Xtreme corpus for cross-lingual NER consisting of datasets of a total of 176 languages. The data comes from the google
        research work XTREME https://github.com/google-research/xtreme. All datasets for NER and respective language abbreviations (e.g.
        "en" for english can be found here https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN/folder/C43gs51bSIaq5sFTQkWNCQ?_encoding=UTF8&*Version*=1&*entries*=0&mgh=1 )
        The data is derived from the wikiann dataset https://elisa-ie.github.io/wikiann/ (license: https://opendatacommons.org/licenses/by/)

        Parameters
        ----------
        languages : Union[str, List[str]], optional
            Default the 40 languages that are used in XTREME are loaded. Otherwise on can hand over a strings or a list of strings
            consisiting of abbreviations for languages. All datasets will be loaded in a MultiCorpus object.
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        tag_to_bioes : str, optional
            The data is in bio-format. It will by default (with the string "ner" as value) be transformed
            into the bioes format. If you dont want that set it to None.

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
        if type(languages) == str:
            languages = [languages]

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # For each language in languages, the file is downloaded if not existent
        # Then a comlumncorpus of that data is created and saved in a list
        # This list is handed to the multicorpus

        # list that contains the columncopora
        corpora: List[Corpus] = []

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
                import tarfile

                tar = tarfile.open(str(temp_file), "r:gz")
                for part in ["train", "test", "dev"]:
                    tar.extract(part, str(language_folder))
                tar.close()
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
                tag_to_bioes=tag_to_bioes,
                in_memory=in_memory,
                **corpusargs,
            )
            corpora.append(corp)

        super(NER_MULTI_XTREME, self).__init__(
            corpora,
            name="xtreme",
        )

    def _xtreme_to_simple_ner_annotation(self, data_file: Union[str, Path]):
        with open(data_file, "r", encoding="utf-8") as f:
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
        languages: Union[str, List[str]] = "en",
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = False,
        **corpusargs,
    ):
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # if only one language is given
        if type(languages) == str:
            languages = [languages]

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        corpora: List[Corpus] = []
        for language in languages:
            language_folder = data_folder / language

            # download data if necessary
            self._download_wikiner(language, str(language_folder))

            # initialize comlumncorpus and add it to list
            log.info(f"Read data for language {language}")
            corp = ColumnCorpus(
                data_folder=language_folder,
                column_format=columns,
                tag_to_bioes=tag_to_bioes,
                in_memory=in_memory,
                **corpusargs,
            )
            corpora.append(corp)

        super(NER_MULTI_WIKINER, self).__init__(
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
            with bz_file as f, open(
                flair.cache_root / "datasets" / dataset_name / f"aij-wikiner-{lc}-wp3.train",
                "w",
                encoding="utf-8",
            ) as out:
                for lineb in f:
                    line = lineb.decode("utf-8")
                    words = line.split(" ")
                    for word in words:
                        out.write("\t".join(word.split("|")) + "\n")


class NER_SWEDISH(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the NER_SWEDISH corpus for Swedish. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_SWEDISH, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )

    def _add_IOB2_tags(self, data_file: Union[str, Path], encoding: str = "utf8"):
        """
        Function that adds IOB2 tags if only chunk names are provided (e.g. words are tagged PER instead
        of B-PER or I-PER). Replaces '0' with 'O' as the no-chunk tag since ColumnCorpus expects
        the letter 'O'. Additionally it removes lines with no tags in the data file and can also
        be used if the data is only partially IOB tagged.
        Parameters
        ----------
        data_file : Union[str, Path]
            Path to the data file.
        encoding : str, optional
            Encoding used in open function. The default is "utf8".

        """
        with open(file=data_file, mode="r", encoding=encoding) as f:
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
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the Finnish TurkuNER corpus. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' to predict
        POS tags instead
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(NER_TURKU, self).__init__(
            data_folder,
            columns,
            dev_file=dev_file,
            test_file=test_file,
            train_file=train_file,
            column_delimiter="\t",
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            **corpusargs,
        )


class KEYPHRASE_SEMEVAL2017(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "keyword",
        in_memory: bool = True,
        **corpusargs,
    ):

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "keyword"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        semeval2017_path = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/SemEval-2017"
        cached_path(f"{semeval2017_path}/train.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2017_path}/test.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2017_path}/dev.txt", Path("datasets") / dataset_name)

        super(KEYPHRASE_SEMEVAL2017, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class KEYPHRASE_INSPEC(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "keyword",
        in_memory: bool = True,
        **corpusargs,
    ):

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(KEYPHRASE_INSPEC, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class KEYPHRASE_SEMEVAL2010(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "keyword",
        in_memory: bool = True,
        **corpusargs,
    ):

        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "keyword"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        semeval2010_path = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/processed_semeval-2010"
        cached_path(f"{semeval2010_path}/train.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2010_path}/test.txt", Path("datasets") / dataset_name)

        super(KEYPHRASE_SEMEVAL2010, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            **corpusargs,
        )


class UP_CHINESE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the Chinese dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions/tree/master/UP_Chinese

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(UP_CHINESE, self).__init__(
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
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the English dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions.

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(UP_ENGLISH, self).__init__(
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
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the French dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions.

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(UP_FRENCH, self).__init__(
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
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the Finnish dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions/tree/master/UP_Finnish

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(UP_FINNISH, self).__init__(
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
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the German dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions.

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(UP_GERMAN, self).__init__(
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
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the Italian dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions/tree/master/UP_Italian

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(UP_ITALIAN, self).__init__(
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
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the Spanish dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(UP_SPANISH, self).__init__(
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
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        document_as_sequence: bool = False,
        **corpusargs,
    ):
        """
        Initialize the Spanish AnCora dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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

        super(UP_SPANISH_ANCORA, self).__init__(
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
    def __init__(
        self,
        dataset_name: str,
        language: str,
        base_path: Union[str, Path] = None,
        tag_to_bioes: str = "ner",
        in_memory: bool = True,
        version: str = "v2.0",
        branch_name: str = "main",
        dev_split_name="dev",
        add_document_separator=False,
        sample_missing_splits=False,
        **corpusargs,
    ):
        """
        Initialize the CLEF-HIPE 2022 NER dataset. The first time you call this constructor it will automatically
        download the specified dataset (by given a language).
        :dataset_name: Supported datasets are: ajmc, hipe2020, letemps, newseye, sonar and topres19th.
        :language: Language for a supported dataset.
        :base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :tag_to_bioes: Dataset will automatically transformed into BIOES format (internally).
        :in_memory: If True, keeps dataset in memory giving speedups in training.
        :version: Version of CLEF-HIPE dataset. Currently only v1.0 is supported and available.
        :branch_name: Defines git branch name of HIPE data repository (main by default).
        :dev_split_name: Defines default name of development split (dev by default). Only the NewsEye dataset has
        currently two development splits: dev and dev2.
        :add_document_separator: If True, a special document seperator will be introduced. This is highly
        recommended when using our FLERT approach.
        :sample_missing_splits: If True, data is automatically sampled when certain data splits are None.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

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
        hipe_available_splits["v2.0"] = hipe_available_splits["v1.0"].copy()
        hipe_available_splits["v2.0"]["ajmc"] = {"de": ["train", "dev"], "en": ["train", "dev"], "fr": ["train", "dev"]}

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

        dev_path = new_data_folder / dev_file

        if not dev_path.exists():
            for split in dataset_splits:
                original_filename = f"HIPE-2022-{version}-{dataset_name}-{split}-{language}.tsv"
                self.__prepare_corpus(
                    data_folder / "original" / original_filename,
                    new_data_folder / f"{split}.txt",
                    eos_marker,
                    document_separator,
                    add_document_separator,
                )

        super(NER_HIPE_2022, self).__init__(
            new_data_folder,
            columns,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            skip_first_line=True,
            column_delimiter="\t",
            comment_symbol="# ",
            sample_missing_splits=sample_missing_splits,
            **corpusargs,
        )

    @staticmethod
    def __prepare_corpus(
        file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
    ):
        with open(file_in, "rt") as f_p:
            lines = f_p.readlines()

        with open(file_out, "wt") as f_out:
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
