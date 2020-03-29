import csv
import logging
import os
import re
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Union, Callable

import numpy as np
import json
import urllib
from tqdm import tqdm

import torch.utils.data.dataloader
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset, ConcatDataset

import flair
from flair.data import (
    Sentence,
    Corpus,
    Token,
    FlairDataset,
    DataPair,
    Image,
    space_tokenizer,
    segtok_tokenizer,
)
from flair.file_utils import cached_path, unzip_file

log = logging.getLogger("flair")


class ColumnCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            column_format: Dict[int, str],
            train_file=None,
            test_file=None,
            dev_file=None,
            tag_to_bioes=None,
            comment_symbol: str = None,
            in_memory: bool = True,
            encoding: str = "utf-8",
            document_separator_token: str = None,
    ):
        """
        Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param document_separator_token: If provided, multiple sentences are read into one object. Provide the string token
        that indicates that a new document begins
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        # get train data
        train = ColumnDataset(
            train_file,
            column_format,
            tag_to_bioes,
            encoding=encoding,
            comment_symbol=comment_symbol,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
        )

        # read in test file if exists
        test = ColumnDataset(
            test_file,
            column_format,
            tag_to_bioes,
            encoding=encoding,
            comment_symbol=comment_symbol,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
        ) if test_file is not None else None

        # read in dev file if exists
        dev = ColumnDataset(
            dev_file,
            column_format,
            tag_to_bioes,
            encoding=encoding,
            comment_symbol=comment_symbol,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
        ) if dev_file is not None else None

        super(ColumnCorpus, self).__init__(train, dev, test, name=str(data_folder))


class UniversalDependenciesCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            train_file=None,
            test_file=None,
            dev_file=None,
            in_memory: bool = True,
    ):
        """
        Instantiates a Corpus from CoNLL-U column-formatted task data such as the UD corpora

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        # get train data
        train = UniversalDependenciesDataset(train_file, in_memory=in_memory)

        # get test data
        test = UniversalDependenciesDataset(test_file, in_memory=in_memory)

        # get dev data
        dev = UniversalDependenciesDataset(dev_file, in_memory=in_memory)

        super(UniversalDependenciesCorpus, self).__init__(
            train, dev, test, name=str(data_folder)
        )


class ClassificationCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            label_type: str = 'class',
            train_file=None,
            test_file=None,
            dev_file=None,
            tokenizer: Callable[[str], List[Token]] = space_tokenizer,
            max_tokens_per_doc: int = -1,
            max_chars_per_doc: int = -1,
            in_memory: bool = False,
            encoding: str = 'utf-8',
    ):
        """
        Instantiates a Corpus from text classification-formatted task data

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        train: FlairDataset = ClassificationDataset(
            train_file,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            encoding=encoding,
        )

        # use test_file to create test split if available
        test: FlairDataset = ClassificationDataset(
            test_file,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            encoding=encoding,
        ) if test_file is not None else None

        # use dev_file to create test split if available
        dev: FlairDataset = ClassificationDataset(
            dev_file,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            encoding=encoding,
        ) if dev_file is not None else None

        super(ClassificationCorpus, self).__init__(
            train, dev, test, name=str(data_folder)
        )


class FeideggerCorpus(Corpus):
    def __init__(self, **kwargs):
        dataset = "feidegger"

        # cache Feidegger config file
        json_link = "https://raw.githubusercontent.com/zalandoresearch/feidegger/master/data/FEIDEGGER_release_1.1.json"
        json_local_path = cached_path(json_link, Path("datasets") / dataset)

        # cache Feidegger images
        dataset_info = json.load(open(json_local_path, "r"))
        images_cache_folder = os.path.join(os.path.dirname(json_local_path), "images")
        if not os.path.isdir(images_cache_folder):
            os.mkdir(images_cache_folder)
        for image_info in tqdm(dataset_info):
            name = os.path.basename(image_info["url"])
            filename = os.path.join(images_cache_folder, name)
            if not os.path.isfile(filename):
                urllib.request.urlretrieve(image_info["url"], filename)
            # replace image URL with local cached file
            image_info["url"] = filename

        feidegger_dataset: Dataset = FeideggerDataset(dataset_info, **kwargs)

        train_indices = list(
            np.where(np.in1d(feidegger_dataset.split, list(range(8))))[0]
        )
        train = torch.utils.data.dataset.Subset(feidegger_dataset, train_indices)

        dev_indices = list(np.where(np.in1d(feidegger_dataset.split, [8]))[0])
        dev = torch.utils.data.dataset.Subset(feidegger_dataset, dev_indices)

        test_indices = list(np.where(np.in1d(feidegger_dataset.split, [9]))[0])
        test = torch.utils.data.dataset.Subset(feidegger_dataset, test_indices)

        super(FeideggerCorpus, self).__init__(train, dev, test, name="feidegger")


class CSVClassificationCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            column_name_map: Dict[int, str],
            label_type: str = 'class',
            train_file=None,
            test_file=None,
            dev_file=None,
            tokenizer: Callable[[str], List[Token]] = segtok_tokenizer,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            in_memory: bool = False,
            skip_header: bool = False,
            encoding: str = 'utf-8',
            **fmtparams,
    ):
        """
        Instantiates a Corpus for text classification from CSV column formatted data

        :param data_folder: base folder with the task data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param fmtparams: additional parameters for the CSV file reader
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        train: FlairDataset = CSVClassificationDataset(
            train_file,
            column_name_map,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_header=skip_header,
            encoding=encoding,
            **fmtparams,
        )

        test: FlairDataset = CSVClassificationDataset(
            test_file,
            column_name_map,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_header=skip_header,
            encoding=encoding,
            **fmtparams,
        ) if test_file is not None else None

        dev: FlairDataset = CSVClassificationDataset(
            dev_file,
            column_name_map,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_header=skip_header,
            encoding=encoding,
            **fmtparams,
        ) if dev_file is not None else None

        super(CSVClassificationCorpus, self).__init__(
            train, dev, test, name=str(data_folder)
        )


class ParallelTextCorpus(Corpus):
    def __init__(
            self,
            source_file: Union[str, Path],
            target_file: Union[str, Path],
            name: str = None,
            use_tokenizer: bool = True,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            in_memory: bool = True,
    ):
        """
        Instantiates a Corpus for text classification from CSV column formatted data

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :return: a Corpus with annotated train, dev and test data
        """

        train: FlairDataset = ParallelTextDataset(
            source_file,
            target_file,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
        )

        super(ParallelTextCorpus, self).__init__(train, name=name)


class OpusParallelCorpus(ParallelTextCorpus):
    def __init__(
            self,
            dataset: str,
            l1: str,
            l2: str,
            use_tokenizer: bool = True,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            in_memory: bool = True,
    ):
        """
        Instantiates a Parallel Corpus from OPUS (http://opus.nlpl.eu/)
        :param dataset: Name of the dataset (one of "tatoeba")
        :param l1: Language code of first language in pair ("en", "de", etc.)
        :param l2: Language code of second language in pair ("en", "de", etc.)
        :param use_tokenizer: Whether or not to use in-built tokenizer
        :param max_tokens_per_doc: If set, shortens sentences to this maximum number of tokens
        :param max_chars_per_doc: If set, shortens sentences to this maximum number of characters
        :param in_memory: If True, keeps dataset fully in memory
        """

        if l1 > l2:
            l1, l2 = l2, l1

        # check if dataset is supported
        supported_datasets = ["tatoeba"]
        if dataset not in supported_datasets:
            log.error(f"Dataset must be one of: {supported_datasets}")

        # set file names
        if dataset == "tatoeba":
            link = f"https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/moses/{l1}-{l2}.txt.zip"

            l1_file = (
                    Path(flair.cache_root)
                    / "datasets"
                    / dataset
                    / f"{l1}-{l2}"
                    / f"Tatoeba.{l1}-{l2}.{l1}"
            )
            l2_file = (
                    Path(flair.cache_root)
                    / "datasets"
                    / dataset
                    / f"{l1}-{l2}"
                    / f"Tatoeba.{l1}-{l2}.{l2}"
            )

        # download and unzip in file structure if necessary
        if not l1_file.exists():
            path = cached_path(link, Path("datasets") / dataset / f"{l1}-{l2}")
            unzip_file(
                path, Path(flair.cache_root) / Path("datasets") / dataset / f"{l1}-{l2}"
            )

        # instantiate corpus
        super(OpusParallelCorpus, self).__init__(
            l1_file,
            l2_file,
            name=f"{dataset}-{l1_file}-{l2_file}",
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
        )


class SentenceDataset(FlairDataset):
    """
    A simple Dataset object to wrap a List of Sentence
    """

    def __init__(self, sentences: Union[Sentence, List[Sentence]]):
        """
        Instantiate SentenceDataset
        :param sentences: Sentence or List of Sentence that make up SentenceDataset
        """
        # cast to list if necessary
        if type(sentences) == Sentence:
            sentences = [sentences]
        self.sentences = sentences

    @abstractmethod
    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        return self.sentences[index]


class StringDataset(FlairDataset):
    """
    A Dataset taking string as input and returning Sentence during iteration
    """

    def __init__(
            self,
            texts: Union[str, List[str]],
            use_tokenizer: Union[bool, Callable[[str], List[Token]]] = space_tokenizer,
    ):
        """
        Instantiate StringDataset
        :param texts: a string or List of string that make up StringDataset
        :param use_tokenizer: a custom tokenizer (default is space based tokenizer,
        more advanced options are segtok_tokenizer to use segtok or build_spacy_tokenizer to use Spacy library
        if available). Check the code of space_tokenizer to implement your own (if you need it).
        If instead of providing a function, this parameter is just set to True, segtok will be used.
        """
        # cast to list if necessary
        if type(texts) == Sentence:
            texts = [texts]
        self.texts = texts
        self.use_tokenizer = use_tokenizer

    @abstractmethod
    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int = 0) -> Sentence:
        text = self.texts[index]
        return Sentence(text, use_tokenizer=self.use_tokenizer)


class ColumnDataset(FlairDataset):
    def __init__(
            self,
            path_to_column_file: Path,
            column_name_map: Dict[int, str],
            tag_to_bioes: str = None,
            comment_symbol: str = None,
            in_memory: bool = True,
            document_separator_token: str = None,
            encoding: str = "utf-8",
    ):
        """
        Instantiates a column dataset (typically used for sequence labeling or word-level prediction).

        :param path_to_column_file: path to the file with the column-formatted data
        :param column_name_map: a map specifying the column format
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param document_separator_token: If provided, multiple sentences are read into one object. Provide the string token
        that indicates that a new document begins
        """
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.tag_to_bioes = tag_to_bioes
        self.column_name_map = column_name_map
        self.comment_symbol = comment_symbol
        self.document_separator_token = document_separator_token

        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory
        if self.in_memory:
            self.sentences: List[Sentence] = []
        else:
            self.indices: List[int] = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_column: int = 0
        for column in self.column_name_map:
            if column_name_map[column] == "text":
                self.text_column = column

        # determine encoding of text file
        self.encoding = encoding

        sentence: Sentence = Sentence()
        sentence_started: bool = False
        with open(str(self.path_to_column_file), encoding=self.encoding) as f:

            line = f.readline()
            position = 0

            while line:

                if self.comment_symbol is not None and line.startswith(comment_symbol):
                    line = f.readline()
                    continue

                if self.__line_completes_sentence(line):

                    if sentence_started:

                        sentence.infer_space_after()
                        if self.in_memory:
                            if self.tag_to_bioes is not None:
                                sentence.convert_tag_scheme(
                                    tag_type=self.tag_to_bioes, target_scheme="iobes"
                                )
                            self.sentences.append(sentence)
                        else:
                            self.indices.append(position)
                            position = f.tell()
                        self.total_sentence_count += 1
                    sentence: Sentence = Sentence()
                    sentence_started = False

                elif self.in_memory:
                    fields: List[str] = re.split("\s+", line)
                    token = Token(fields[self.text_column])
                    for column in column_name_map:
                        if len(fields) > column:
                            if column != self.text_column:
                                token.add_label(
                                    self.column_name_map[column], fields[column]
                                )

                    if not line.isspace():
                        sentence.add_token(token)
                        sentence_started = True
                elif not line.isspace():
                    sentence_started = True

                line = f.readline()

        if sentence_started:
            sentence.infer_space_after()
            if self.in_memory:
                self.sentences.append(sentence)
            else:
                self.indices.append(position)
            self.total_sentence_count += 1

    def __line_completes_sentence(self, line: str) -> bool:
        sentence_completed = line.isspace()
        if self.document_separator_token:
            sentence_completed = False
            fields: List[str] = re.split("\s+", line)
            if len(fields) >= self.text_column:
                if fields[self.text_column] == self.document_separator_token:
                    sentence_completed = True
        return sentence_completed

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        if self.in_memory:
            sentence = self.sentences[index]

        else:
            with open(str(self.path_to_column_file), encoding=self.encoding) as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence: Sentence = Sentence()
                while line:
                    if self.comment_symbol is not None and line.startswith(
                            self.comment_symbol
                    ):
                        line = file.readline()
                        continue

                    if self.__line_completes_sentence(line):
                        if len(sentence) > 0:
                            sentence.infer_space_after()
                            if self.tag_to_bioes is not None:
                                sentence.convert_tag_scheme(
                                    tag_type=self.tag_to_bioes, target_scheme="iobes"
                                )
                            return sentence

                    else:
                        fields: List[str] = re.split("\s+", line)
                        token = Token(fields[self.text_column])
                        for column in self.column_name_map:
                            if len(fields) > column:
                                if column != self.text_column:
                                    token.add_label(
                                        self.column_name_map[column], fields[column]
                                    )

                        if not line.isspace():
                            sentence.add_token(token)

                    line = file.readline()
        return sentence


class UniversalDependenciesDataset(FlairDataset):
    def __init__(self, path_to_conll_file: Path, in_memory: bool = True):
        """
        Instantiates a column dataset in CoNLL-U format.

        :param path_to_conll_file: Path to the CoNLL-U formatted file
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        """
        assert path_to_conll_file.exists()

        self.in_memory = in_memory
        self.path_to_conll_file = path_to_conll_file
        self.total_sentence_count: int = 0

        if self.in_memory:
            self.sentences: List[Sentence] = []
        else:
            self.indices: List[int] = []

        with open(str(self.path_to_conll_file), encoding="utf-8") as file:

            line = file.readline()
            position = 0
            sentence: Sentence = Sentence()
            while line:

                line = line.strip()
                fields: List[str] = re.split("\t+", line)
                if line == "":
                    if len(sentence) > 0:
                        self.total_sentence_count += 1
                        if self.in_memory:
                            self.sentences.append(sentence)
                        else:
                            self.indices.append(position)
                            position = file.tell()
                    sentence: Sentence = Sentence()

                elif line.startswith("#"):
                    line = file.readline()
                    continue
                elif "." in fields[0]:
                    line = file.readline()
                    continue
                elif "-" in fields[0]:
                    line = file.readline()
                    continue
                else:
                    token = Token(fields[1], head_id=int(fields[6]))
                    token.add_label("lemma", str(fields[2]))
                    token.add_label("upos", str(fields[3]))
                    token.add_label("pos", str(fields[4]))
                    token.add_label("dependency", str(fields[7]))

                    if len(fields) > 9 and 'SpaceAfter=No' in fields[9]:
                        token.whitespace_after = False

                    for morph in str(fields[5]).split("|"):
                        if "=" not in morph:
                            continue
                        token.add_label(morph.split("=")[0].lower(), morph.split("=")[1])

                    if len(fields) > 10 and str(fields[10]) == "Y":
                        token.add_label("frame", str(fields[11]))

                    sentence.add_token(token)

                line = file.readline()
            if len(sentence.tokens) > 0:
                self.total_sentence_count += 1
                if self.in_memory:
                    self.sentences.append(sentence)
                else:
                    self.indices.append(position)

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        if self.in_memory:
            sentence = self.sentences[index]
        else:
            with open(str(self.path_to_conll_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence: Sentence = Sentence()
                while line:

                    line = line.strip()
                    fields: List[str] = re.split("\t+", line)
                    if line == "":
                        if len(sentence) > 0:
                            break

                    elif line.startswith("#"):
                        line = file.readline()
                        continue
                    elif "." in fields[0]:
                        line = file.readline()
                        continue
                    elif "-" in fields[0]:
                        line = file.readline()
                        continue
                    else:
                        token = Token(fields[1], head_id=int(fields[6]))
                        token.add_label("lemma", str(fields[2]))
                        token.add_label("upos", str(fields[3]))
                        token.add_label("pos", str(fields[4]))
                        token.add_label("dependency", str(fields[7]))

                        if len(fields) > 9 and 'SpaceAfter=No' in fields[9]:
                            token.whitespace_after = False

                        for morph in str(fields[5]).split("|"):
                            if "=" not in morph:
                                continue
                            token.add_label(
                                morph.split("=")[0].lower(), morph.split("=")[1]
                            )

                        if len(fields) > 10 and str(fields[10]) == "Y":
                            token.add_label("frame", str(fields[11]))

                        sentence.add_token(token)

                    line = file.readline()
        return sentence


class CSVClassificationDataset(FlairDataset):
    def __init__(
            self,
            path_to_file: Union[str, Path],
            column_name_map: Dict[int, str],
            label_type: str = "class",
            max_tokens_per_doc: int = -1,
            max_chars_per_doc: int = -1,
            tokenizer=segtok_tokenizer,
            in_memory: bool = True,
            skip_header: bool = False,
            encoding: str = 'utf-8',
            **fmtparams,
    ):
        """
        Instantiates a Dataset for text classification from CSV column formatted data

        :param path_to_file: path to the file with the CSV data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param skip_header: If True, skips first line because it is header
        :param fmtparams: additional parameters for the CSV file reader
        :return: a Corpus with annotated train, dev and test data
        """

        if type(path_to_file) == str:
            path_to_file: Path = Path(path_to_file)

        assert path_to_file.exists()

        # variables
        self.path_to_file = path_to_file
        self.in_memory = in_memory
        self.tokenizer = tokenizer
        self.column_name_map = column_name_map
        self.max_tokens_per_doc = max_tokens_per_doc
        self.max_chars_per_doc = max_chars_per_doc

        self.label_type = label_type

        # different handling of in_memory data than streaming data
        if self.in_memory:
            self.sentences = []
        else:
            self.raw_data = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_columns: List[int] = []
        for column in column_name_map:
            if column_name_map[column] == "text":
                self.text_columns.append(column)

        with open(self.path_to_file, encoding=encoding) as csv_file:

            csv_reader = csv.reader(csv_file, **fmtparams)

            if skip_header:
                next(csv_reader, None)  # skip the headers

            for row in csv_reader:

                # test if format is OK
                wrong_format = False
                for text_column in self.text_columns:
                    if text_column >= len(row):
                        wrong_format = True

                if wrong_format:
                    continue

                # test if at least one label given
                has_label = False
                for column in self.column_name_map:
                    if self.column_name_map[column].startswith("label") and row[column]:
                        has_label = True
                        break

                if not has_label:
                    continue

                if self.in_memory:

                    text = " ".join(
                        [row[text_column] for text_column in self.text_columns]
                    )

                    if self.max_chars_per_doc > 0:
                        text = text[: self.max_chars_per_doc]

                    sentence = Sentence(text, use_tokenizer=self.tokenizer)

                    for column in self.column_name_map:
                        if (
                                self.column_name_map[column].startswith("label")
                                and row[column]
                        ):
                            sentence.add_label(label_type, row[column])

                    if 0 < self.max_tokens_per_doc < len(sentence):
                        sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]
                    self.sentences.append(sentence)

                else:
                    self.raw_data.append(row)

                self.total_sentence_count += 1

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            return self.sentences[index]
        else:
            row = self.raw_data[index]

            text = " ".join([row[text_column] for text_column in self.text_columns])

            if self.max_chars_per_doc > 0:
                text = text[: self.max_chars_per_doc]

            sentence = Sentence(text, use_tokenizer=self.tokenizer)
            for column in self.column_name_map:
                if self.column_name_map[column].startswith("label") and row[column]:
                    sentence.add_label(self.label_type, row[column])

            if 0 < self.max_tokens_per_doc < len(sentence):
                sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]

            return sentence


class ClassificationDataset(FlairDataset):
    def __init__(
            self,
            path_to_file: Union[str, Path],
            label_type: str = 'class',
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            tokenizer=segtok_tokenizer,
            in_memory: bool = True,
            encoding: str = 'utf-8',
    ):
        """
        Reads a data file for text classification. The file should contain one document/text per line.
        The line should have the following format:
        __label__<class_name> <text>
        If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,
        __label__<class_name_1> __label__<class_name_2> <text>
        :param path_to_file: the path to the data file
        :param max_tokens_per_doc: Takes at most this amount of tokens per document. If set to -1 all documents are taken as is.
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :return: list of sentences
        """
        if type(path_to_file) == str:
            path_to_file: Path = Path(path_to_file)

        assert path_to_file.exists()

        self.label_prefix = "__label__"
        self.label_type = label_type

        self.in_memory = in_memory
        self.tokenizer = tokenizer

        if self.in_memory:
            self.sentences = []
        else:
            self.indices = []

        self.total_sentence_count: int = 0
        self.max_chars_per_doc = max_chars_per_doc
        self.max_tokens_per_doc = max_tokens_per_doc

        self.path_to_file = path_to_file

        with open(str(path_to_file), encoding=encoding) as f:
            line = f.readline()
            position = 0
            while line:
                if "__label__" not in line or " " not in line:
                    position = f.tell()
                    line = f.readline()
                    continue

                if self.in_memory:
                    sentence = self._parse_line_to_sentence(
                        line, self.label_prefix, tokenizer
                    )
                    if sentence is not None and len(sentence.tokens) > 0:
                        self.sentences.append(sentence)
                        self.total_sentence_count += 1
                else:
                    self.indices.append(position)
                    self.total_sentence_count += 1

                position = f.tell()
                line = f.readline()

    def _parse_line_to_sentence(
            self, line: str, label_prefix: str, tokenizer: Callable[[str], List[Token]]
    ):
        words = line.split()

        labels = []
        l_len = 0

        for i in range(len(words)):
            if words[i].startswith(label_prefix):
                l_len += len(words[i]) + 1
                label = words[i].replace(label_prefix, "")
                labels.append(label)
            else:
                break

        text = line[l_len:].strip()

        if self.max_chars_per_doc > 0:
            text = text[: self.max_chars_per_doc]

        if text and labels:
            sentence = Sentence(text, use_tokenizer=tokenizer)

            for label in labels:
                sentence.add_label(self.label_type, label)

            if (
                    sentence is not None
                    and 0 < self.max_tokens_per_doc < len(sentence)
            ):
                sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]

            return sentence
        return None

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            return self.sentences[index]
        else:

            with open(str(self.path_to_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence = self._parse_line_to_sentence(
                    line, self.label_prefix, self.tokenizer
                )
                return sentence


class MongoDataset(FlairDataset):
    def __init__(
            self,
            query: str,
            host: str,
            port: int,
            database: str,
            collection: str,
            text_field: str,
            categories_field: List[str] = None,
            max_tokens_per_doc: int = -1,
            max_chars_per_doc: int = -1,
            tokenizer=segtok_tokenizer,
            in_memory: bool = True,
    ):
        """
        Reads Mongo collections. Each collection should contain one document/text per item.

        Each item should have the following format:
        {
        'Beskrivning': 'Abrahamsby. Gård i Gottröra sn, Långhundra hd, Stockholms län, nära Långsjön.',
        'Län':'Stockholms län',
        'Härad': 'Långhundra',
        'Församling': 'Gottröra',
        'Plats': 'Abrahamsby'
        }

        :param query: Query, e.g. {'Län': 'Stockholms län'}
        :param host: Host, e.g. 'localhost',
        :param port: Port, e.g. 27017
        :param database: Database, e.g. 'rosenberg',
        :param collection: Collection, e.g. 'book',
        :param text_field: Text field, e.g. 'Beskrivning',
        :param categories_field: List of category fields, e.g ['Län', 'Härad', 'Tingslag', 'Församling', 'Plats'],
        :param max_tokens_per_doc: Takes at most this amount of tokens per document. If set to -1 all documents are taken as is.
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :return: list of sentences
        """

        # first, check if pymongo is installed
        try:
            import pymongo
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "pymongo" is not installed!')
            log.warning(
                'To use MongoDataset, please first install with "pip install pymongo"'
            )
            log.warning("-" * 100)
            pass

        self.in_memory = in_memory
        self.tokenizer = tokenizer

        if self.in_memory:
            self.sentences = []
        else:
            self.indices = []

        self.total_sentence_count: int = 0
        self.max_chars_per_doc = max_chars_per_doc
        self.max_tokens_per_doc = max_tokens_per_doc

        self.__connection = pymongo.MongoClient(host, port)
        self.__cursor = self.__connection[database][collection]

        self.text = text_field
        self.categories = categories_field if categories_field is not None else []

        start = 0

        kwargs = lambda start: {"filter": query, "skip": start, "limit": 0}

        if self.in_memory:
            for document in self.__cursor.find(**kwargs(start)):
                sentence = self._parse_document_to_sentence(
                    document[self.text],
                    [document[_] if _ in document else "" for _ in self.categories],
                    tokenizer,
                )
                if sentence is not None and len(sentence.tokens) > 0:
                    self.sentences.append(sentence)
                    self.total_sentence_count += 1
        else:
            self.indices = self.__cursor.find().distinct("_id")
            self.total_sentence_count = self.__cursor.count_documents()

    def _parse_document_to_sentence(
            self, text: str, labels: List[str], tokenizer: Callable[[str], List[Token]]
    ):
        if self.max_chars_per_doc > 0:
            text = text[: self.max_chars_per_doc]

        if text and labels:
            sentence = Sentence(text, labels=labels, use_tokenizer=tokenizer)

            if self.max_tokens_per_doc > 0:
                sentence.tokens = sentence.tokens[
                                  : min(len(sentence), self.max_tokens_per_doc)
                                  ]

            return sentence
        return None

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            return self.sentences[index]
        else:
            document = self.__cursor.find_one({"_id": index})
            sentence = self._parse_document_to_sentence(
                document[self.text],
                [document[_] if _ in document else "" for _ in self.categories],
                self.tokenizer,
            )
            return sentence


class ParallelTextDataset(FlairDataset):
    def __init__(
            self,
            path_to_source: Union[str, Path],
            path_to_target: Union[str, Path],
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            use_tokenizer=True,
            in_memory: bool = True,
    ):
        if type(path_to_source) == str:
            path_to_source: Path = Path(path_to_source)
        if type(path_to_target) == str:
            path_to_target: Path = Path(path_to_target)

        assert path_to_source.exists()
        assert path_to_target.exists()

        self.in_memory = in_memory

        self.use_tokenizer = use_tokenizer
        self.max_tokens_per_doc = max_tokens_per_doc

        self.total_sentence_count: int = 0

        if self.in_memory:
            self.bi_sentences: List[DataPair] = []
        else:
            self.source_lines: List[str] = []
            self.target_lines: List[str] = []

        with open(str(path_to_source), encoding="utf-8") as source_file, open(
                str(path_to_target), encoding="utf-8"
        ) as target_file:

            source_line = source_file.readline()
            target_line = target_file.readline()

            while source_line and target_line:

                source_line = source_file.readline()
                target_line = target_file.readline()

                if source_line.strip() == "":
                    continue
                if target_line.strip() == "":
                    continue

                if max_chars_per_doc > 0:
                    source_line = source_line[:max_chars_per_doc]
                    target_line = target_line[:max_chars_per_doc]

                if self.in_memory:
                    bi_sentence = self._make_bi_sentence(source_line, target_line)
                    self.bi_sentences.append(bi_sentence)
                else:
                    self.source_lines.append(source_line)
                    self.target_lines.append(target_line)

                self.total_sentence_count += 1

    def _make_bi_sentence(self, source_line: str, target_line: str):

        source_sentence = Sentence(source_line, use_tokenizer=self.use_tokenizer)
        target_sentence = Sentence(target_line, use_tokenizer=self.use_tokenizer)

        if self.max_tokens_per_doc > 0:
            source_sentence.tokens = source_sentence.tokens[: self.max_tokens_per_doc]
            target_sentence.tokens = target_sentence.tokens[: self.max_tokens_per_doc]

        return DataPair(source_sentence, target_sentence)

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> DataPair:
        if self.in_memory:
            return self.bi_sentences[index]
        else:
            return self._make_bi_sentence(
                self.source_lines[index], self.target_lines[index]
            )


class FeideggerDataset(FlairDataset):
    def __init__(self, dataset_info, in_memory: bool = True, **kwargs):
        super(FeideggerDataset, self).__init__()

        self.data_points: List[DataPair] = []
        self.split: List[int] = []

        preprocessor = lambda x: x
        if "lowercase" in kwargs and kwargs["lowercase"]:
            preprocessor = lambda x: x.lower()

        for image_info in dataset_info:
            image = Image(imageURL=image_info["url"])
            for caption in image_info["descriptions"]:
                # append Sentence-Image data point
                self.data_points.append(
                    DataPair(Sentence(preprocessor(caption), use_tokenizer=True), image)
                )
                self.split.append(int(image_info["split"]))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index: int = 0) -> DataPair:
        return self.data_points[index]


class CONLL_03(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            document_as_sequence: bool = False,
    ):
        """
        Initialize the CoNLL-03 corpus. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'pos' or 'np' to predict
        POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "np", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
        )


class CONLL_03_GERMAN(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            document_as_sequence: bool = False,
    ):
        """
        Initialize the CoNLL-03 corpus for German. This is only possible if you've manually downloaded it to your machine.
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put it into some folder. Then point
        the base_path parameter in the constructor to this folder
        :param base_path: Path to the CoNLL-03 corpus on your machine
        :param tag_to_bioes: NER by default, need not be changed, but you could also select 'lemma', 'pos' or 'np' to predict
        word lemmas, POS tags or chunks respectively
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "lemma", 2: "pos", 3: "np", 4: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: CoNLL-03 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://www.clips.uantwerpen.be/conll2003/ner/"'
            )
            log.warning("-" * 100)

        super(CONLL_03_GERMAN, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
        )


class CONLL_03_DUTCH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            document_as_sequence: bool = False,
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
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        conll_02_path = "https://www.clips.uantwerpen.be/conll2002/ner/data/"
        cached_path(f"{conll_02_path}ned.testa", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}ned.testb", Path("datasets") / dataset_name)
        cached_path(f"{conll_02_path}ned.train", Path("datasets") / dataset_name)

        super(CONLL_03_DUTCH, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
        )


class CONLL_03_SPANISH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
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
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
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
        )


class CONLL_2000(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "np",
            in_memory: bool = True,
    ):
        """
        Initialize the CoNLL-2000 corpus for English chunking.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param tag_to_bioes: 'np' by default, should not be changed, but you can set 'pos' instead to predict POS tags
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "np"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        conll_2000_path = "https://www.clips.uantwerpen.be/conll2000/chunking/"
        data_file = Path(flair.cache_root) / "datasets" / dataset_name / "train.txt"
        if not data_file.is_file():
            cached_path(
                f"{conll_2000_path}train.txt.gz", Path("datasets") / dataset_name
            )
            cached_path(
                f"{conll_2000_path}test.txt.gz", Path("datasets") / dataset_name
            )
            import gzip, shutil

            with gzip.open(
                    Path(flair.cache_root) / "datasets" / dataset_name / "train.txt.gz",
                    "rb",
            ) as f_in:
                with open(
                        Path(flair.cache_root) / "datasets" / dataset_name / "train.txt",
                        "wb",
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            with gzip.open(
                    Path(flair.cache_root) / "datasets" / dataset_name / "test.txt.gz", "rb"
            ) as f_in:
                with open(
                        Path(flair.cache_root) / "datasets" / dataset_name / "test.txt",
                        "wb",
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)

        super(CONLL_2000, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class SENTEVAL_CR(ClassificationCorpus):
    def __init__(
            self,
            in_memory: bool = True,
    ):
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            senteval_path = "https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip"
            cached_path(senteval_path, Path("datasets") / "senteval")
            senteval_folder = Path(flair.cache_root) / "datasets" / "senteval"
            unzip_file(senteval_folder / "datasmall_NB_ACL12.zip", senteval_folder)

            # create dataset directory if necessary
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            # create train.txt file by iterating over pos and neg file
            with open(data_folder / "train.txt", "a") as train_file:

                with open(senteval_folder / "data" / "customerr" / "custrev.pos", encoding="latin1") as file:
                    for line in file:
                        train_file.write(f"__label__POSITIVE {line}")

                with open(senteval_folder / "data" / "customerr" / "custrev.neg", encoding="latin1") as file:
                    for line in file:
                        train_file.write(f"__label__NEGATIVE {line}")

        super(SENTEVAL_CR, self).__init__(
            data_folder, label_type='sentiment', tokenizer=segtok_tokenizer, in_memory=in_memory
        )


class SENTEVAL_MR(ClassificationCorpus):
    def __init__(
            self,
            in_memory: bool = True,
    ):
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            senteval_path = "https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip"
            cached_path(senteval_path, Path("datasets") / "senteval")
            senteval_folder = Path(flair.cache_root) / "datasets" / "senteval"
            unzip_file(senteval_folder / "datasmall_NB_ACL12.zip", senteval_folder)

            # create dataset directory if necessary
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            # create train.txt file by iterating over pos and neg file
            with open(data_folder / "train.txt", "a") as train_file:

                with open(senteval_folder / "data" / "rt10662" / "rt-polarity.pos", encoding="latin1") as file:
                    for line in file:
                        train_file.write(f"__label__POSITIVE {line}")

                with open(senteval_folder / "data" / "rt10662" / "rt-polarity.neg", encoding="latin1") as file:
                    for line in file:
                        train_file.write(f"__label__NEGATIVE {line}")

        super(SENTEVAL_MR, self).__init__(
            data_folder, label_type='sentiment', tokenizer=segtok_tokenizer, in_memory=in_memory
        )


class SENTEVAL_SUBJ(ClassificationCorpus):
    def __init__(
            self,
            in_memory: bool = True,
    ):
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            senteval_path = "https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip"
            cached_path(senteval_path, Path("datasets") / "senteval")
            senteval_folder = Path(flair.cache_root) / "datasets" / "senteval"
            unzip_file(senteval_folder / "datasmall_NB_ACL12.zip", senteval_folder)

            # create dataset directory if necessary
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            # create train.txt file by iterating over pos and neg file
            with open(data_folder / "train.txt", "a") as train_file:

                with open(senteval_folder / "data" / "subj" / "subj.subjective", encoding="latin1") as file:
                    for line in file:
                        train_file.write(f"__label__SUBJECTIVE {line}")

                with open(senteval_folder / "data" / "subj" / "subj.objective", encoding="latin1") as file:
                    for line in file:
                        train_file.write(f"__label__OBJECTIVE {line}")

        super(SENTEVAL_SUBJ, self).__init__(
            data_folder, label_type='objectivity', tokenizer=segtok_tokenizer, in_memory=in_memory
        )


class SENTEVAL_MPQA(ClassificationCorpus):
    def __init__(
            self,
            in_memory: bool = True,
    ):
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            senteval_path = "https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip"
            cached_path(senteval_path, Path("datasets") / "senteval")
            senteval_folder = Path(flair.cache_root) / "datasets" / "senteval"
            unzip_file(senteval_folder / "datasmall_NB_ACL12.zip", senteval_folder)

            # create dataset directory if necessary
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            # create train.txt file by iterating over pos and neg file
            with open(data_folder / "train.txt", "a") as train_file:

                with open(senteval_folder / "data" / "mpqa" / "mpqa.pos", encoding="latin1") as file:
                    for line in file:
                        train_file.write(f"__label__POSITIVE {line}")

                with open(senteval_folder / "data" / "mpqa" / "mpqa.neg", encoding="latin1") as file:
                    for line in file:
                        train_file.write(f"__label__NEGATIVE {line}")

        super(SENTEVAL_MPQA, self).__init__(
            data_folder, label_type='sentiment', tokenizer=segtok_tokenizer, in_memory=in_memory
        )


class SENTEVAL_SST_BINARY(CSVClassificationCorpus):
    def __init__(
            self,
            in_memory: bool = True,
    ):
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-train', Path("datasets") / dataset_name)
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-test', Path("datasets") / dataset_name)
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-dev', Path("datasets") / dataset_name)

        super(SENTEVAL_SST_BINARY, self).__init__(
            data_folder,
            column_name_map={0: 'text', 1: 'label'},
            tokenizer=segtok_tokenizer,
            in_memory=in_memory,
            delimiter='\t',
            quotechar=None,
        )


class SENTEVAL_SST_GRANULAR(ClassificationCorpus):
    def __init__(
            self,
            in_memory: bool = True,
    ):
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            cached_path('https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/stsa.fine.train', Path("datasets") / dataset_name / 'raw')
            cached_path('https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/stsa.fine.test', Path("datasets") / dataset_name / 'raw')
            cached_path('https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/stsa.fine.dev', Path("datasets") / dataset_name / 'raw')

            # convert to FastText format
            for split in ['train', 'dev', 'test']:
                with open(data_folder / f"{split}.txt", "w") as train_file:

                    with open(data_folder / 'raw' / f'stsa.fine.{split}', encoding="latin1") as file:
                        for line in file:
                            train_file.write(f"__label__{line[0]} {line[2:]}")

        super(SENTEVAL_SST_GRANULAR, self).__init__(
            data_folder,
            tokenizer=segtok_tokenizer,
            in_memory=in_memory,
        )


class GERMEVAL(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
    ):
        """
        Initialize the GermEval NER corpus for German. This is only possible if you've manually downloaded it to your
        machine. Obtain the corpus from https://sites.google.com/site/germeval2014ner/home/ and put it into some folder.
        Then point the base_path parameter in the constructor to this folder
        :param base_path: Path to the GermEval corpus on your machine
        :param tag_to_bioes: 'ner' by default, should not be changed.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {1: "text", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # check if data there
        if not data_folder.exists():
            log.warning("-" * 100)
            log.warning(f'ACHTUNG: GermEval-14 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://sites.google.com/site/germeval2014ner/home/"'
            )
            log.warning("-" * 100)
        super(GERMEVAL, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            comment_symbol="#",
            in_memory=in_memory,
        )


class IMDB(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        imdb_acl_path = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        data_path = Path(flair.cache_root) / "datasets" / dataset_name
        data_file = data_path / "train.txt"
        if not data_file.is_file():
            cached_path(imdb_acl_path, Path("datasets") / dataset_name)
            import tarfile

            with tarfile.open(
                    Path(flair.cache_root)
                    / "datasets"
                    / dataset_name
                    / "aclImdb_v1.tar.gz",
                    "r:gz",
            ) as f_in:
                datasets = ["train", "test"]
                labels = ["pos", "neg"]

                for label in labels:
                    for dataset in datasets:
                        f_in.extractall(
                            data_path,
                            members=[
                                m
                                for m in f_in.getmembers()
                                if f"{dataset}/{label}" in m.name
                            ],
                        )
                        with open(f"{data_path}/{dataset}.txt", "at") as f_p:
                            current_path = data_path / "aclImdb" / dataset / label
                            for file_name in current_path.iterdir():
                                if file_name.is_file() and file_name.name.endswith(
                                        ".txt"
                                ):
                                    f_p.write(
                                        f"__label__{label} "
                                        + file_name.open("rt", encoding="utf-8").read()
                                        + "\n"
                                    )

        super(IMDB, self).__init__(
            data_folder, tokenizer=space_tokenizer, in_memory=in_memory
        )


class NEWSGROUPS(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        twenty_newsgroups_path = (
            "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
        )
        data_path = Path(flair.cache_root) / "datasets" / dataset_name
        data_file = data_path / "20news-bydate-train.txt"
        if not data_file.is_file():
            cached_path(
                twenty_newsgroups_path, Path("datasets") / dataset_name / "original"
            )

            import tarfile

            with tarfile.open(
                    Path(flair.cache_root)
                    / "datasets"
                    / dataset_name
                    / "original"
                    / "20news-bydate.tar.gz",
                    "r:gz",
            ) as f_in:
                datasets = ["20news-bydate-test", "20news-bydate-train"]
                labels = [
                    "alt.atheism",
                    "comp.graphics",
                    "comp.os.ms-windows.misc",
                    "comp.sys.ibm.pc.hardware",
                    "comp.sys.mac.hardware",
                    "comp.windows.x",
                    "misc.forsale",
                    "rec.autos",
                    "rec.motorcycles",
                    "rec.sport.baseball",
                    "rec.sport.hockey",
                    "sci.crypt",
                    "sci.electronics",
                    "sci.med",
                    "sci.space",
                    "soc.religion.christian",
                    "talk.politics.guns",
                    "talk.politics.mideast",
                    "talk.politics.misc",
                    "talk.religion.misc",
                ]

                for label in labels:
                    for dataset in datasets:
                        f_in.extractall(
                            data_path / "original",
                            members=[
                                m
                                for m in f_in.getmembers()
                                if f"{dataset}/{label}" in m.name
                            ],
                        )
                        with open(
                                f"{data_path}/{dataset}.txt", "at", encoding="utf-8"
                        ) as f_p:
                            current_path = data_path / "original" / dataset / label
                            for file_name in current_path.iterdir():
                                if file_name.is_file():
                                    f_p.write(
                                        f"__label__{label} "
                                        + file_name.open("rt", encoding="latin1")
                                        .read()
                                        .replace("\n", " <n> ")
                                        + "\n"
                                    )

        super(NEWSGROUPS, self).__init__(
            data_folder, tokenizer=space_tokenizer, in_memory=in_memory
        )

class DANE(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {1: 'text', 3: 'pos', 9: 'ner'}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        data_path = Path(flair.cache_root) / "datasets" / dataset_name
        train_data_file = data_path / "ddt.train.conllu"
        if not train_data_file.is_file():
            temp_file = cached_path(
                'https://danlp.s3.eu-central-1.amazonaws.com/datasets/ddt.zip',
                Path("datasets") / dataset_name
            )
            from zipfile import ZipFile

            with ZipFile(temp_file, 'r') as zip_file:
                zip_file.extractall(path=data_path)

            # Remove CoNLL-U meta information in the last column
            for part in ['train', 'dev', 'test']:
                lines = []
                data_file = "ddt.{}.conllu".format(part)
                with open(data_path / data_file, 'r') as file:
                    for line in file:
                        if line.startswith("#") or line == "\n":
                            lines.append(line)
                        lines.append(line.replace("name=", "").replace("|SpaceAfter=No", ""))

                with open(data_path / data_file, 'w') as file:
                    file.writelines(lines)

                print(data_path / data_file)

        super(DANE, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes,
            in_memory=in_memory, comment_symbol="#"
        )


class NER_BASQUE(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ner_basque_path = "http://ixa2.si.ehu.eus/eiec/"
        data_path = Path(flair.cache_root) / "datasets" / dataset_name
        data_file = data_path / "named_ent_eu.train"
        if not data_file.is_file():
            cached_path(
                f"{ner_basque_path}/eiec_v1.0.tgz", Path("datasets") / dataset_name
            )
            import tarfile, shutil

            with tarfile.open(
                    Path(flair.cache_root) / "datasets" / dataset_name / "eiec_v1.0.tgz",
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
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class TREC_50(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        trec_path = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"

        original_filenames = ["train_5500.label", "TREC_10.label"]
        new_filenames = ["train.txt", "test.txt"]
        for original_filename in original_filenames:
            cached_path(
                f"{trec_path}{original_filename}",
                Path("datasets") / dataset_name / "original",
            )

        data_file = data_folder / new_filenames[0]

        if not data_file.is_file():
            for original_filename, new_filename in zip(
                    original_filenames, new_filenames
            ):
                with open(
                        data_folder / "original" / original_filename,
                        "rt",
                        encoding="latin1",
                ) as open_fp:
                    with open(
                            data_folder / new_filename, "wt", encoding="utf-8"
                    ) as write_fp:
                        for line in open_fp:
                            line = line.rstrip()
                            fields = line.split()
                            old_label = fields[0]
                            question = " ".join(fields[1:])

                            # Create flair compatible labels
                            # TREC-6 : NUM:dist -> __label__NUM
                            # TREC-50: NUM:dist -> __label__NUM:dist
                            new_label = "__label__"
                            new_label += old_label

                            write_fp.write(f"{new_label} {question}\n")

        super(TREC_50, self).__init__(
            data_folder, tokenizer=space_tokenizer, in_memory=in_memory
        )


class TREC_6(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        trec_path = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"

        original_filenames = ["train_5500.label", "TREC_10.label"]
        new_filenames = ["train.txt", "test.txt"]
        for original_filename in original_filenames:
            cached_path(
                f"{trec_path}{original_filename}",
                Path("datasets") / dataset_name / "original",
            )

        data_file = data_folder / new_filenames[0]

        if not data_file.is_file():
            for original_filename, new_filename in zip(
                    original_filenames, new_filenames
            ):
                with open(
                        data_folder / "original" / original_filename,
                        "rt",
                        encoding="latin1",
                ) as open_fp:
                    with open(
                            data_folder / new_filename, "wt", encoding="utf-8"
                    ) as write_fp:
                        for line in open_fp:
                            line = line.rstrip()
                            fields = line.split()
                            old_label = fields[0]
                            question = " ".join(fields[1:])

                            # Create flair compatible labels
                            # TREC-6 : NUM:dist -> __label__NUM
                            # TREC-50: NUM:dist -> __label__NUM:dist
                            new_label = "__label__"
                            new_label += old_label.split(":")[0]

                            write_fp.write(f"{new_label} {question}\n")

        super(TREC_6, self).__init__(
            data_folder, label_type='question_type', tokenizer=space_tokenizer, in_memory=in_memory
        )


class UD_ENGLISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
        cached_path(f"{web_path}/en_ewt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{web_path}/en_ewt-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{web_path}/en_ewt-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_ENGLISH, self).__init__(data_folder, in_memory=in_memory)


class UD_GERMAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master"
        cached_path(f"{ud_path}/de_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/de_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/de_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_GERMAN, self).__init__(data_folder, in_memory=in_memory)


class UD_GERMAN_HDT(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = (
            "https://raw.githubusercontent.com/UniversalDependencies/UD_German-HDT/dev"
        )
        cached_path(f"{ud_path}/de_hdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/de_hdt-ud-test.conllu", Path("datasets") / dataset_name)

        train_filenames = [
            "de_hdt-ud-train-a-1.conllu",
            "de_hdt-ud-train-a-2.conllu",
            "de_hdt-ud-train-b-1.conllu",
            "de_hdt-ud-train-b-2.conllu",
        ]

        for train_file in train_filenames:
            cached_path(
                f"{ud_path}/{train_file}", Path("datasets") / dataset_name / "original"
            )

        data_path = Path(flair.cache_root) / "datasets" / dataset_name

        new_train_file: Path = data_path / "de_hdt-ud-train-all.conllu"

        if not new_train_file.is_file():
            with open(new_train_file, "wt") as f_out:
                for train_filename in train_filenames:
                    with open(data_path / "original" / train_filename, "rt") as f_in:
                        f_out.write(f_in.read())

        super(UD_GERMAN_HDT, self).__init__(data_folder, in_memory=in_memory)


class UD_DUTCH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/master"
        cached_path(
            f"{ud_path}/nl_alpino-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/nl_alpino-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/nl_alpino-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_DUTCH, self).__init__(data_folder, in_memory=in_memory)


class UD_FRENCH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master"
        cached_path(f"{ud_path}/fr_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fr_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/fr_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_FRENCH, self).__init__(data_folder, in_memory=in_memory)


class UD_ITALIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master"
        cached_path(f"{ud_path}/it_isdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/it_isdt-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/it_isdt-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_ITALIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_SPANISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master"
        cached_path(f"{ud_path}/es_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/es_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/es_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_SPANISH, self).__init__(data_folder, in_memory=in_memory)


class UD_PORTUGUESE(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/master"
        cached_path(
            f"{ud_path}/pt_bosque-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/pt_bosque-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/pt_bosque-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_PORTUGUESE, self).__init__(data_folder, in_memory=in_memory)


class UD_ROMANIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master"
        cached_path(f"{ud_path}/ro_rrt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ro_rrt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/ro_rrt-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_ROMANIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_CATALAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Catalan-AnCora/master"
        cached_path(
            f"{ud_path}/ca_ancora-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ca_ancora-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ca_ancora-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_CATALAN, self).__init__(data_folder, in_memory=in_memory)


class UD_POLISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-LFG/master"
        cached_path(f"{ud_path}/pl_lfg-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/pl_lfg-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/pl_lfg-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_POLISH, self).__init__(data_folder, in_memory=in_memory)


class UD_CZECH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-PDT/master"
        cached_path(f"{ud_path}/cs_pdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/cs_pdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/cs_pdt-ud-train-c.conllu",
            Path("datasets") / dataset_name / "original",
        )
        cached_path(
            f"{ud_path}/cs_pdt-ud-train-l.conllu",
            Path("datasets") / dataset_name / "original",
        )
        cached_path(
            f"{ud_path}/cs_pdt-ud-train-m.conllu",
            Path("datasets") / dataset_name / "original",
        )
        cached_path(
            f"{ud_path}/cs_pdt-ud-train-v.conllu",
            Path("datasets") / dataset_name / "original",
        )
        data_path = Path(flair.cache_root) / "datasets" / dataset_name

        train_filenames = [
            "cs_pdt-ud-train-c.conllu",
            "cs_pdt-ud-train-l.conllu",
            "cs_pdt-ud-train-m.conllu",
            "cs_pdt-ud-train-v.conllu",
        ]

        new_train_file: Path = data_path / "cs_pdt-ud-train-all.conllu"

        if not new_train_file.is_file():
            with open(new_train_file, "wt") as f_out:
                for train_filename in train_filenames:
                    with open(data_path / "original" / train_filename, "rt") as f_in:
                        f_out.write(f_in.read())
        super(UD_CZECH, self).__init__(data_folder, in_memory=in_memory)


class UD_SLOVAK(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Slovak-SNK/master"
        cached_path(f"{ud_path}/sk_snk-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sk_snk-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/sk_snk-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_SLOVAK, self).__init__(data_folder, in_memory=in_memory)


class UD_SWEDISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/master"
        cached_path(
            f"{ud_path}/sv_talbanken-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/sv_talbanken-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/sv_talbanken-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_SWEDISH, self).__init__(data_folder, in_memory=in_memory)


class UD_DANISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Danish-DDT/master"
        cached_path(f"{ud_path}/da_ddt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/da_ddt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/da_ddt-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_DANISH, self).__init__(data_folder, in_memory=in_memory)


class UD_NORWEGIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Norwegian-Bokmaal/master"
        cached_path(
            f"{ud_path}/no_bokmaal-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/no_bokmaal-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/no_bokmaal-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_NORWEGIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_FINNISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-TDT/master"
        cached_path(f"{ud_path}/fi_tdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fi_tdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/fi_tdt-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_FINNISH, self).__init__(data_folder, in_memory=in_memory)


class UD_SLOVENIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Slovenian-SSJ/master"
        cached_path(f"{ud_path}/sl_ssj-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sl_ssj-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/sl_ssj-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_SLOVENIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_CROATIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Croatian-SET/master"
        cached_path(f"{ud_path}/hr_set-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/hr_set-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/hr_set-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_CROATIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_SERBIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Serbian-SET/master"
        cached_path(f"{ud_path}/sr_set-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sr_set-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/sr_set-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_SERBIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_BULGARIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Bulgarian-BTB/master"
        cached_path(f"{ud_path}/bg_btb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/bg_btb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/bg_btb-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_BULGARIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_ARABIC(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master"
        cached_path(f"{ud_path}/ar_padt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/ar_padt-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ar_padt-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_ARABIC, self).__init__(data_folder, in_memory=in_memory)


class UD_HEBREW(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master"
        cached_path(f"{ud_path}/he_htb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/he_htb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/he_htb-ud-train.conllu", Path("datasets") / dataset_name
        )
        super(UD_HEBREW, self).__init__(data_folder, in_memory=in_memory)


class UD_TURKISH(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master"
        cached_path(f"{ud_path}/tr_imst-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/tr_imst-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/tr_imst-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_TURKISH, self).__init__(data_folder, in_memory=in_memory)


class UD_PERSIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Persian-Seraji/master"
        cached_path(
            f"{ud_path}/fa_seraji-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/fa_seraji-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/fa_seraji-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_PERSIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_RUSSIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master"
        cached_path(
            f"{ud_path}/ru_syntagrus-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ru_syntagrus-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ru_syntagrus-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_RUSSIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_HINDI(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/master"
        cached_path(f"{ud_path}/hi_hdtb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/hi_hdtb-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/hi_hdtb-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_HINDI, self).__init__(data_folder, in_memory=in_memory)


class UD_INDONESIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-GSD/master"
        cached_path(f"{ud_path}/id_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/id_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/id_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_INDONESIAN, self).__init__(data_folder, in_memory=in_memory)


class UD_JAPANESE(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master"
        cached_path(f"{ud_path}/ja_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ja_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/ja_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_JAPANESE, self).__init__(data_folder, in_memory=in_memory)


class UD_CHINESE(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master"
        cached_path(f"{ud_path}/zh_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/zh_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/zh_gsd-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_CHINESE, self).__init__(data_folder, in_memory=in_memory)


class UD_KOREAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Korean-Kaist/master"
        cached_path(
            f"{ud_path}/ko_kaist-ud-dev.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ko_kaist-ud-test.conllu", Path("datasets") / dataset_name
        )
        cached_path(
            f"{ud_path}/ko_kaist-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_KOREAN, self).__init__(data_folder, in_memory=in_memory)


class UD_BASQUE(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Basque-BDT/master"
        cached_path(f"{ud_path}/eu_bdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/eu_bdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{ud_path}/eu_bdt-ud-train.conllu", Path("datasets") / dataset_name
        )

        super(UD_BASQUE, self).__init__(data_folder, in_memory=in_memory)


def _download_wassa_if_not_there(emotion, data_folder, dataset_name):
    for split in ["train", "dev", "test"]:

        data_file = data_folder / f"{emotion}-{split}.txt"

        if not data_file.is_file():

            if split == "train":
                url = f"http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/{emotion}-ratings-0to1.train.txt"
            if split == "dev":
                url = f"http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/{emotion}-ratings-0to1.dev.gold.txt"
            if split == "test":
                url = f"http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/{emotion}-ratings-0to1.test.gold.txt"

            path = cached_path(url, Path("datasets") / dataset_name)

            with open(path, "r") as f:
                with open(data_file, "w") as out:
                    next(f)
                    for line in f:
                        fields = line.split("\t")
                        out.write(f"__label__{fields[3].rstrip()} {fields[1]}\n")

            os.remove(path)


class WASSA_ANGER(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("anger", data_folder, dataset_name)

        super(WASSA_ANGER, self).__init__(
            data_folder, tokenizer=space_tokenizer, in_memory=in_memory
        )


class WASSA_FEAR(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("fear", data_folder, dataset_name)

        super(WASSA_FEAR, self).__init__(
            data_folder, tokenizer=space_tokenizer, in_memory=in_memory
        )


class WASSA_JOY(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("joy", data_folder, dataset_name)

        super(WASSA_JOY, self).__init__(
            data_folder, tokenizer=space_tokenizer, in_memory=in_memory
        )


class WASSA_SADNESS(ClassificationCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = False):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("sadness", data_folder, dataset_name)

        super(WASSA_SADNESS, self).__init__(
            data_folder, tokenizer=space_tokenizer, in_memory=in_memory
        )


def _download_wikiner(language_code: str, dataset_name: str):
    # download data if necessary
    wikiner_path = (
        "https://raw.githubusercontent.com/dice-group/FOX/master/input/Wikiner/"
    )
    lc = language_code

    data_file = (
            Path(flair.cache_root)
            / "datasets"
            / dataset_name
            / f"aij-wikiner-{lc}-wp3.train"
    )
    if not data_file.is_file():

        cached_path(
            f"{wikiner_path}aij-wikiner-{lc}-wp3.bz2", Path("datasets") / dataset_name
        )
        import bz2, shutil

        # unpack and write out in CoNLL column-like format
        bz_file = bz2.BZ2File(
            Path(flair.cache_root)
            / "datasets"
            / dataset_name
            / f"aij-wikiner-{lc}-wp3.bz2",
            "rb",
        )
        with bz_file as f, open(
                Path(flair.cache_root)
                / "datasets"
                / dataset_name
                / f"aij-wikiner-{lc}-wp3.train",
                "w",
        ) as out:
            for line in f:
                line = line.decode("utf-8")
                words = line.split(" ")
                for word in words:
                    out.write("\t".join(word.split("|")) + "\n")


class WIKINER_ENGLISH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("en", dataset_name)

        super(WIKINER_ENGLISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_GERMAN(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("de", dataset_name)

        super(WIKINER_GERMAN, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_DUTCH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("nl", dataset_name)

        super(WIKINER_DUTCH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_FRENCH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("fr", dataset_name)

        super(WIKINER_FRENCH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_ITALIAN(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("it", dataset_name)

        super(WIKINER_ITALIAN, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_SPANISH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("es", dataset_name)

        super(WIKINER_SPANISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_PORTUGUESE(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("pt", dataset_name)

        super(WIKINER_PORTUGUESE, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_POLISH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("pl", dataset_name)

        super(WIKINER_POLISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WIKINER_RUSSIAN(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "pos", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wikiner("ru", dataset_name)

        super(WIKINER_RUSSIAN, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class WNUT_17(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        wnut_path = "https://noisy-text.github.io/2017/files/"
        cached_path(f"{wnut_path}wnut17train.conll", Path("datasets") / dataset_name)
        cached_path(f"{wnut_path}emerging.dev.conll", Path("datasets") / dataset_name)
        cached_path(
            f"{wnut_path}emerging.test.annotated", Path("datasets") / dataset_name
        )

        super(WNUT_17, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=8,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
    ):

        # in certain cases, multi-CPU data loading makes no sense and slows
        # everything down. For this reason, we detect if a dataset is in-memory:
        # if so, num_workers is set to 0 for faster processing
        flair_dataset = dataset
        while True:
            if type(flair_dataset) is Subset:
                flair_dataset = flair_dataset.dataset
            elif type(flair_dataset) is ConcatDataset:
                flair_dataset = flair_dataset.datasets[0]
            else:
                break

        if type(flair_dataset) is list:
            num_workers = 0
        elif isinstance(flair_dataset, FlairDataset) and flair_dataset.is_in_memory():
            num_workers = 0

        super(DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=list,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def find_train_dev_test_files(data_folder, dev_file, test_file, train_file):
    if type(data_folder) == str:
        data_folder: Path = Path(data_folder)

    if train_file is not None:
        train_file = data_folder / train_file
    if test_file is not None:
        test_file = data_folder / test_file
    if dev_file is not None:
        dev_file = data_folder / dev_file

    suffixes_to_ignore = {".gz", ".swp"}
    
    # automatically identify train / test / dev files
    if train_file is None:
        for file in data_folder.iterdir():
            file_name = file.name
            if not suffixes_to_ignore.isdisjoint(file.suffixes):
                continue
            if "train" in file_name and not "54019" in file_name:
                train_file = file
            if "dev" in file_name:
                dev_file = file
            if "testa" in file_name:
                dev_file = file
            if "testb" in file_name:
                test_file = file

        # if no test file is found, take any file with 'test' in name
        if test_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if not suffixes_to_ignore.isdisjoint(file.suffixes):
                    continue
                if "test" in file_name:
                    test_file = file

    log.info("Reading data from {}".format(data_folder))
    log.info("Train: {}".format(train_file))
    log.info("Dev: {}".format(dev_file))
    log.info("Test: {}".format(test_file))

    return dev_file, test_file, train_file
