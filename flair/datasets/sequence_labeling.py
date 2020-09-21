import logging
import re
import os
from pathlib import Path
from typing import Union, Dict, List

import flair
from flair.data import Corpus, MultiCorpus, FlairDataset, Sentence, Token
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path

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
            column_delimiter: str = r"\s+",
            comment_symbol: str = None,
            encoding: str = "utf-8",
            document_separator_token: str = None,
            skip_first_line: bool = False,
            in_memory: bool = True,
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
        :param document_separator_token: If provided, multiple sentences are read into one object. Provide the string token
        that indicates that a new document begins
        :param skip_first_line: set to True if your dataset has a header line
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
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
            column_delimiter=column_delimiter,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
            skip_first_line=skip_first_line,
        )

        # read in test file if exists
        test = ColumnDataset(
            test_file,
            column_format,
            tag_to_bioes,
            encoding=encoding,
            comment_symbol=comment_symbol,
            column_delimiter=column_delimiter,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
            skip_first_line=skip_first_line,
        ) if test_file is not None else None

        # read in dev file if exists
        dev = ColumnDataset(
            dev_file,
            column_format,
            tag_to_bioes,
            encoding=encoding,
            comment_symbol=comment_symbol,
            column_delimiter=column_delimiter,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
            skip_first_line=skip_first_line,
        ) if dev_file is not None else None

        super(ColumnCorpus, self).__init__(train, dev, test, name=str(data_folder))


class ColumnDataset(FlairDataset):
    # special key for space after
    SPACE_AFTER_KEY = "space-after"

    def __init__(
            self,
            path_to_column_file: Union[str, Path],
            column_name_map: Dict[int, str],
            tag_to_bioes: str = None,
            column_delimiter: str = r"\s+",
            comment_symbol: str = None,
            in_memory: bool = True,
            document_separator_token: str = None,
            encoding: str = "utf-8",
            skip_first_line: bool = False,
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
        :param document_separator_token: If provided, multiple sentences are read into one object. Provide the string token
        that indicates that a new document begins
        :param skip_first_line: set to True if your dataset has a header line
        """
        if type(path_to_column_file) is str:
            path_to_column_file = Path(path_to_column_file)
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.tag_to_bioes = tag_to_bioes
        self.column_name_map = column_name_map
        self.column_delimiter = column_delimiter
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

            # skip first line if to selected
            if skip_first_line:
                f.readline()

            line = f.readline()
            position = 0

            while line:

                if self.comment_symbol is not None and line.startswith(comment_symbol):
                    line = f.readline()
                    continue

                if self.__line_completes_sentence(line):

                    if sentence_started:

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
                    token = self._parse_token(line)
                    if not line.isspace():
                        sentence.add_token(token)
                        sentence_started = True

                elif not line.isspace():
                    sentence_started = True

                line = f.readline()

        if sentence_started:
            if self.in_memory:
                self.sentences.append(sentence)
            else:
                self.indices.append(position)
            self.total_sentence_count += 1

    def _parse_token(self, line: str) -> Token:
        fields: List[str] = re.split(self.column_delimiter, line.rstrip())
        token = Token(fields[self.text_column])
        for column in self.column_name_map:
            if len(fields) > column:
                if column != self.text_column and self.column_name_map[column] != self.SPACE_AFTER_KEY:
                    token.add_label(
                        self.column_name_map[column], fields[column]
                    )
                if self.column_name_map[column] == self.SPACE_AFTER_KEY and fields[column] == '-':
                    token.whitespace_after = False
        return token

    def __line_completes_sentence(self, line: str) -> bool:
        sentence_completed = line.isspace()
        if self.document_separator_token:
            sentence_completed = False
            fields: List[str] = re.split(self.column_delimiter, line)
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
                            if self.tag_to_bioes is not None:
                                sentence.convert_tag_scheme(
                                    tag_type=self.tag_to_bioes, target_scheme="iobes"
                                )
                            return sentence

                    else:
                        token = self._parse_token(line)
                        if not line.isspace():
                            sentence.add_token(token)

                    line = file.readline()
        return sentence


class BIOFID(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "lemma", 2: "pos", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        biofid_path = "https://raw.githubusercontent.com/texttechnologylab/BIOfid/master/BIOfid-Dataset-NER/"
        cached_path(f"{biofid_path}train.conll", Path("datasets") / dataset_name)
        cached_path(f"{biofid_path}dev.conll", Path("datasets") / dataset_name)
        cached_path(f"{biofid_path}test.conll", Path("datasets") / dataset_name)

        super(BIOFID, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


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
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put the eng.testa, .testb, .train
        files in a folder called 'conll_03'. Then set the base_path parameter in the constructor to the path to the
        parent directory where the conll_03 folder resides.
        :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03' folder) on your machine
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
            log.warning(f'WARNING: CoNLL-03 dataset not found at "{data_folder}".')
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
        Obtain the corpus from https://www.clips.uantwerpen.be/conll2003/ner/ and put the respective files in a folder called
        'conll_03_german'. Then set the base_path parameter in the constructor to the path to the parent directory where
        the conll_03_german folder resides.
        :param base_path: Path to the CoNLL-03 corpus (i.e. 'conll_03_german' folder) on your machine
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


class WIKIGOLD_NER(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            document_as_sequence: bool = False,
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
        wikigold_ner_path = "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/wikigold/CONLL-format/data/"
        cached_path(f"{wikigold_ner_path}wikigold.conll.txt", Path("datasets") / dataset_name)

        super(WIKIGOLD_NER, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="utf-8",
            in_memory=in_memory,
            train_file='wikigold.conll.txt',
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
        )


class TWITTER_NER(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            document_as_sequence: bool = False,
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
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: 'text', 1: 'ner'}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        twitter_ner_path = "https://raw.githubusercontent.com/aritter/twitter_nlp/master/data/annotated/"
        cached_path(f"{twitter_ner_path}ner.txt", Path("datasets") / dataset_name)

        super(TWITTER_NER, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            train_file="ner.txt",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
        )


class MIT_RESTAURANTS(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
            document_as_sequence: bool = False,
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
        mit_restaurants_path = "https://megantosh.s3.eu-central-1.amazonaws.com/MITRestoCorpus/"
        cached_path(f"{mit_restaurants_path}test.txt", Path("datasets") / dataset_name)
        cached_path(f"{mit_restaurants_path}train.txt", Path("datasets") / dataset_name)

        super(MIT_RESTAURANTS, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            document_separator_token=None if not document_as_sequence else "-DOCSTART-",
        )


def add_IOB_tags(data_file: Union[str, Path], encoding: str = "utf8", ner_column: int = 1):
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
                f.write(' I-' + tag)
            else:
                f.write(' ' + current_line[i])
        f.write('\n')

    with open(file=data_file, mode='r', encoding=encoding) as f:
        lines = f.readlines()
    with open(file=data_file, mode='w', encoding=encoding) as f:
        pred = 'O'  # remembers ner tag of predecessing line
        for line in lines:
            line_list = line.split()
            if len(line_list) > 2:  # word with tags
                ner_tag = line_list[ner_column]
                if ner_tag in ['0', 'O']:  # no chunk
                    for i in range(0, len(line_list)):
                        if i == 0:
                            f.write(line_list[i])
                        elif i == ner_column:
                            f.write(' O')
                        else:
                            f.write(' ' + line_list[i])
                    f.write('\n')
                    pred = 'O'
                elif '-' not in ner_tag:  # no IOB tags
                    if pred == 'O':  # found a new chunk
                        add_I_prefix(line_list, ner_column, ner_tag)
                        pred = ner_tag
                    else:  # found further part of chunk or new chunk directly after old chunk
                        add_I_prefix(line_list, ner_column, ner_tag)
                        pred = ner_tag
                else:  # line already has IOB tag (tag contains '-')
                    f.write(line)
                    pred = ner_tag.split('-')[1]
            elif len(line_list) == 0:  # empty line
                f.write('\n')
                pred = 'O'


def add_IOB2_tags(data_file: Union[str, Path], encoding: str = "utf8"):
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
    with open(file=data_file, mode='r', encoding=encoding) as f:
        lines = f.readlines()
    with open(file=data_file, mode='w', encoding=encoding) as f:
        pred = 'O'  # remembers tag of predecessing line
        for line in lines:
            line_list = line.split()
            if len(line_list) == 2:  # word with tag
                word = line_list[0]
                tag = line_list[1]
                if tag in ['0', 'O']:  # no chunk
                    f.write(word + ' O\n')
                    pred = 'O'
                elif '-' not in tag:  # no IOB tags
                    if pred == 'O':  # found a new chunk
                        f.write(word + ' B-' + tag + '\n')
                        pred = tag
                    else:  # found further part of chunk or new chunk directly after old chunk
                        if pred == tag:
                            f.write(word + ' I-' + tag + '\n')
                        else:
                            f.write(word + ' B-' + tag + '\n')
                            pred = tag
                else:  # line already has IOB tag (tag contains '-')
                    f.write(line)
                    pred = tag.split('-')[1]
            elif len(line_list) == 0:  # empty line
                f.write('\n')
                pred = 'O'


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


class XTREME(MultiCorpus):
    def __init__(
            self,
            languages: Union[str, List[str]] = None,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
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
            languages = ["af", "ar", "bg", "bn", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "he", "hi", "hu",
                         "id", "it", "ja", "jv", "ka", "kk", "ko", "ml", "mr", "ms", "my", "nl", "pt", "ru", "sw", "ta",
                         "te", "th", "tl", "tr", "ur", "vi", "yo", "zh"]

        # if only one language is given
        if type(languages) == str:
            languages = [languages]

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = "xtreme"

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # For each language in languages, the file is downloaded if not existent
        # Then a comlumncorpus of that data is created and saved in a list
        # This list is handed to the multicorpus

        # list that contains the columncopora
        corpora = []

        hu_path = "https://nlp.informatik.hu-berlin.de/resources/datasets/panx_dataset"

        # download data if necessary
        for language in languages:

            language_folder = data_folder / language

            # if language not downloaded yet, download it
            if not language_folder.exists():

                file_name = language + '.tar.gz'
                # create folder
                os.makedirs(language_folder)

                # download from HU Server
                temp_file = cached_path(
                    hu_path + "/" + file_name,
                    Path("datasets") / dataset_name / language
                )

                # unzip
                print("Extract data...")
                import tarfile
                tar = tarfile.open(str(temp_file), "r:gz")
                for part in ["train", "test", "dev"]:
                    tar.extract(part, str(language_folder))
                tar.close()
                print('...done.')

                # transform data into required format
                print("Process dataset...")
                for part in ["train", "test", "dev"]:
                    xtreme_to_simple_ner_annotation(str(language_folder / part))
                print('...done.')

            # initialize comlumncorpus and add it to list
            print("Read data into corpus...")
            corp = ColumnCorpus(data_folder=language_folder,
                                column_format=columns,
                                tag_to_bioes=tag_to_bioes,
                                in_memory=in_memory,
                                )
            corpora.append(corp)
            print("...done.")

        super(XTREME, self).__init__(
            corpora, name='xtreme'
        )


def xtreme_to_simple_ner_annotation(data_file: Union[str, Path]):
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(data_file, 'w', encoding='utf-8') as f:
        for line in lines:
            if line == '\n':
                f.write(line)
            else:
                liste = line.split()
                f.write(liste[0].split(':', 1)[1] + ' ' + liste[1] + '\n')


class WIKIANN(MultiCorpus):
    def __init__(
            self,
            languages: Union[str, List[str]],
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
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

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = "wikiann"

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # For each language in languages, the file is downloaded if not existent
        # Then a comlumncorpus of that data is created and saved in a list
        # this list is handed to the multicorpus

        # list that contains the columncopora
        corpora = []

        google_drive_path = 'https://drive.google.com/uc?id='
        # download data if necessary
        first = True
        for language in languages:

            language_folder = data_folder / language
            file_name = 'wikiann-' + language + '.bio'

            # if language not downloaded yet, download it
            if not language_folder.exists():
                if first == True:
                    import gdown
                    import tarfile
                    first = False
                # create folder
                os.makedirs(language_folder)
                # get google drive id from list
                google_id = google_drive_id_from_language_name(language)
                url = google_drive_path + google_id

                # download from google drive
                gdown.download(url, str(language_folder / language) + '.tar.gz')

                # unzip
                print("Extract data...")
                tar = tarfile.open(str(language_folder / language) + '.tar.gz', "r:gz")
                # tar.extractall(language_folder,members=[tar.getmember(file_name)])
                tar.extract(file_name, str(language_folder))
                tar.close()
                print('...done.')

                # transform data into required format
                # the processed dataset has the additional ending "_new"
                print("Process dataset...")
                silver_standard_to_simple_ner_annotation(str(language_folder / file_name))
                # remove the unprocessed dataset
                os.remove(str(language_folder / file_name))
                print('...done.')

            # initialize comlumncorpus and add it to list
            print("Read data into corpus...")
            corp = ColumnCorpus(data_folder=language_folder,
                                column_format=columns,
                                train_file=file_name + '_new',
                                tag_to_bioes=tag_to_bioes,
                                in_memory=in_memory,
                                )
            corpora.append(corp)
            print("...done.")

        super(WIKIANN, self).__init__(
            corpora, name='wikiann'
        )


def silver_standard_to_simple_ner_annotation(data_file: Union[str, Path]):
    f_read = open(data_file, 'r', encoding='utf-8')
    f_write = open(data_file + '_new', 'w+', encoding='utf-8')
    while True:
        line = f_read.readline()
        if line:
            if line == '\n':
                f_write.write(line)
            else:
                liste = line.split()
                f_write.write(liste[0] + ' ' + liste[-1] + '\n')
        else:
            break
    f_read.close()
    f_write.close()


def google_drive_id_from_language_name(language):
    languages_ids = {
        'aa': '1tDDlydKq7KQQ3_23Ysbtke4HJOe4snIk',  # leer
        'ab': '1hB8REj2XA_0DjI9hdQvNvSDpuBIb8qRf',
        'ace': '1WENJS2ppHcZqaBEXRZyk2zY-PqXkTkgG',
        'ady': '1n6On8WWDHxEoybj7F9K15d_fkGPy6KgO',
        'af': '1CPB-0BD2tg3zIT60D3hmJT0i5O_SKja0',
        'ak': '1l2vlGHnQwvm9XhW5S-403fetwUXhBlZm',
        'als': '196xyYjhbie7sYLHLZHWkkurOwQLi8wK-',
        'am': '1ug1IEoExKD3xWpvfZprAPSQi82YF9Cet',
        'an': '1DNLgPOAOsGZBYd6rC5ddhzvc9_DtWnk2',
        'ang': '1W_0ti7Tl8AkqM91lRCMPWEuUnPOAZroV',
        'ar': '1tyvd32udEQG_cNeVpaD5I2fxvCc6XKIS',
        'arc': '1hSOByStqPmP3b9HfQ39EclUZGo8IKCMb',
        'arz': '1CKW5ZhxTpIHmc8Jt5JLz_5O6Cr8Icsan',
        'as': '12opBoIweBLM8XciMHT4B6-MAaKdYdvpE',
        'ast': '1rp64PxGZBDfcw-tpFBjLg_ddLDElG1II',
        'av': '1hncGUrkG1vwAAQgLtwOf41BWkHkEvdss',
        'ay': '1VmIsWpMTz442b4Mx798ZOgtB9vquKQtf',
        'az': '1FXDXsvBSdqc7GGIDZv0hqBOaaw12Ip2-',
        'azb': '1amVqOuHLEkhjn8rkGUl-mXdZlaACWyNT',
        'ba': '1aLx1d8GagI11VZVYOGQy0BEePeqoT0x3',
        'bar': '1JZ8-k8ZmnpWYI_Yl_cBBgjVdxoM9Daci',
        'bat-smg': '1trxKXDFSeKsygTMKi-ZqXSJs7F90k5a8',
        'bcl': '1Hs0k7KVZ2DPsqroZ4cUKcwZG4HdPV794',
        'be-x-old': '1gaK-spj1m6eGYQ-SsngLxxLUvP1VRk08',
        'be': '1_ttfOSy9BzCRkIT_p3mImT82XRPpEiuH',
        'bg': '1Iug6gYKemb0OrLTUrKDc_c66YGypTfCF',
        'bh': '12OcSFLu940A8tVQLxI8pnxKBpTeZHmrh',
        'bi': '1rftVziS_pqARx4mvLJC0sKLY-OL5ZIjE',
        'bjn': '1n17mkRjPUAOWQk5LQs2C3Tz3ShxK0enZ',
        'bm': '1284dwO_sfdsWE7FR06HhfBRUb8ePesKR',
        'bn': '1K2DM1mT4hkr6NlAIBTj95BeVXcgvpgDm',
        'bo': '1SzGHDVK-OguKdjZ4DXWiOJVrie1iHeWm',
        'bpy': '1m-e5EoruJufvwBEgJLmJtx6jzx64pYN2',
        'br': '1xdaBoJ1DnwI0iEq7gQN1dWcABAs_bM9H',
        'bs': '167dsB01trMYFQl8FshtIdfhjw7IfVKbk',
        'bug': '1yCnevM9_KJzFk27Vxsva_20OacLo4Uam',
        'bxr': '1DlByAX3zB-9UyEAVD4wtX-R7mXC-8xum',
        'ca': '1LuUgbd9sGa-5Ahcsy31EK89a3WOowftY',
        'cbk-zam': '1kgF8xoD-kIOWZET_9kp_4yNX6AAXn6PI',
        'cdo': '14x1y6611G-UAEGq92QEHRpreVkYnoUCw',
        'ce': '1QUUCVKA-fkiCHd3KT3zUWefaWnxzlZLu',
        'ceb': '1DJZE9RfaMoPNXHI73KBXAm4YSe-_YCUk',
        'ch': '1YzAfhmatkmTpkZbAcD6X83epCgzD5S2_',
        'cho': '1ciY0vF3c5a2mTOo_k32A2wMs0klK98Kb',  # leer
        'chr': '1EHaxz1UZHn7v2bbRzCLAhPsNtRzrG3Ae',
        'chy': '1nNWwMAJr1KNdz3bHf6uIn-thZCknlTeB',
        'ckb': '1llpaftcUSiXCZQZMdAqaJSrhwMdcf9IV',
        'co': '1ZP-8oWgMYfW7a6w6ygEFkKDGbN39QnDn',
        'cr': '1ST0xRicLAG4JdCZwGdaY-0pEXooQh7e6',
        'crh': '1Jmpq2XVYUR_XaXU5XNhtOMnz-qkpsgpE',
        'cs': '1Vydyze-jBkK_S1uV5ewV_Y6dbwhXr7lk',
        'csb': '1naUyF74lZPnnopXdOqf5Xor2kT4WoHfS',
        'cu': '1EN5dVTU6jc7YOYPCHq8EYUF31HlMUKs7',
        'cv': '1gEUAlqYSSDI4TrWCqP1LUq2n0X1XEjN3',
        'cy': '1q5g6NJE5GXf65Vc_P4BnUMHQ49Prz-J1',
        'da': '11onAGOLkkqrIwM784siWlg-cewa5WKm8',
        'de': '1f9nWvNkCCy6XWhd9uf4Dq-2--GzSaYAb',
        'diq': '1IkpJaVbEOuOs9qay_KG9rkxRghWZhWPm',
        'dsb': '1hlExWaMth-2eVIQ3i3siJSG-MN_7Z6MY',
        'dv': '1WpCrslO4I7TMb2uaKVQw4U2U8qMs5szi',
        'dz': '10WX52ePq2KfyGliwPvY_54hIjpzW6klV',
        'ee': '1tYEt3oN2KPzBSWrk9jpCqnW3J1KXdhjz',
        'el': '1cxq4NUYmHwWsEn5waYXfFSanlINXWLfM',
        'eml': '17FgGhPZqZNtzbxpTJOf-6nxEuI5oU4Vd',
        'en': '1mqxeCPjxqmO7e8utj1MQv1CICLFVvKa-',
        'eo': '1YeknLymGcqj44ug2yd4P7xQVpSK27HkK',
        'es': '1Dnx3MVR9r5cuoOgeew2gT8bDvWpOKxkU',
        'et': '1Qhb3kYlQnLefWmNimdN_Vykm4mWzbcWy',
        'eu': '1f613wH88UeITYyBSEMZByK-nRNMwLHTs',
        'ext': '1D0nLOZ3aolCM8TShIRyCgF3-_MhWXccN',
        'fa': '1QOG15HU8VfZvJUNKos024xI-OGm0zhEX',
        'ff': '1h5pVjxDYcq70bSus30oqi9KzDmezVNry',
        'fi': '1y3Kf6qYsSvL8_nSEwE1Y6Bf6ninaPvqa',
        'fiu-vro': '1oKUiqG19WgPd3CCl4FGudk5ATmtNfToR',
        'fj': '10xDMuqtoTJlJFp5ghbhKfNWRpLDK3W4d',
        'fo': '1RhjYqgtri1276Be1N9RrNitdBNkpzh0J',
        'fr': '1sK_T_-wzVPJYrnziNqWTriU52rEsXGjn',
        'frp': '1NUm8B2zClBcEa8dHLBb-ZgzEr8phcQyZ',
        'frr': '1FjNqbIUlOW1deJdB8WCuWjaZfUzKqujV',
        'fur': '1oqHZMK7WAV8oHoZLjGR0PfmO38wmR6XY',
        'fy': '1DvnU6iaTJc9bWedmDklHyx8nzKD1s3Ge',
        'ga': '1Ql6rh7absdYQ8l-3hj_MVKcEC3tHKeFB',
        'gag': '1zli-hOl2abuQ2wsDJU45qbb0xuvYwA3a',
        'gan': '1u2dOwy58y-GaS-tCPJS_i9VRDQIPXwCr',
        'gd': '1umsUpngJiwkLdGQbRqYpkgxZju9dWlRz',
        'gl': '141K2IbLjJfXwFTIf-kthmmG0YWdi8liE',
        'glk': '1ZDaxQ6ilXaoivo4_KllagabbvfOuiZ0c',
        'gn': '1hM4MuCaVnZqnL-w-0N-WcWag22ikVLtZ',
        'gom': '1BNOSw75tzPC0wEgLOCKbwu9wg9gcLOzs',
        'got': '1YSHYBtXc1WvUvMIHPz6HHgJvaXKulJUj',
        'gu': '1VdK-B2drqFwKg8KD23c3dKXY-cZgCMgd',
        'gv': '1XZFohYNbKszEFR-V-yDXxx40V41PV9Zm',
        'ha': '18ZG4tUU0owRtQA8Ey3Dl72ALjryEJWMC',
        'hak': '1QQe3WgrCWbvnVH42QXD7KX4kihHURB0Z',
        'haw': '1FLqlK-wpz4jy768XbQAtxd9PhC-9ciP7',
        'he': '18K-Erc2VOgtIdskaQq4D5A3XkVstDmfX',
        'hi': '1lBRapb5tjBqT176gD36K5yb_qsaFeu-k',
        'hif': '153MQ9Ga4NQ-CkK8UiJM3DjKOk09fhCOV',
        'ho': '1c1AoS7yq15iVkTEE-0f3x25NT4F202B8',  # leer
        'hr': '1wS-UtB3sGHuXJQQGR0F5lDegogsgoyif',
        'hsb': '1_3mMLzAE5OmXn2z64rW3OwWbo85Mirbd',
        'ht': '1BwCaF0nfdgkM7Yt7A7d7KyVk0BcuwPGk',
        'hu': '10AkDmTxUWNbOXuYLYZ-ZPbLAdGAGZZ8J',
        'hy': '1Mi2k2alJJquT1ybd3GC3QYDstSagaWdo',
        'hz': '1c1m_-Q92v0Di7Nez6VuaccrN19i8icKV',  # leer
        'ia': '1jPyqTmDuVhEhj89N606Cja5heJEbcMoM',
        'id': '1JWIvIh8fQoMQqk1rPvUThaskxnTs8tsf',
        'ie': '1TaKRlTtB8-Wqu4sfvx6JQKIugAlg0pV-',
        'ig': '15NFAf2Qx6BXSjv_Oun9_3QRBWNn49g86',
        'ii': '1qldGJkMOMKwY13DpcgbxQCbff0K982f9',  # leer
        'ik': '1VoSTou2ZlwVhply26ujowDz6gjwtxmny',
        'ilo': '1-xMuIT6GaM_YeHqgm1OamGkxYfBREiv3',
        'io': '19Zla0wsAcrZm2c0Pw5ghpp4rHjYs26Pp',
        'is': '11i-NCyqS6HbldIbYulsCgQGZFXR8hwoB',
        'it': '1HmjlOaQunHqL2Te7pIkuBWrnjlmdfYo_',
        'iu': '18jKm1S7Ls3l0_pHqQH8MycG3LhoC2pdX',
        'ja': '10dz8UxyK4RIacXE2HcGdrharmp5rwc3r',
        'jam': '1v99CXf9RnbF6aJo669YeTR6mQRTOLZ74',  # leer
        'jbo': '1_LmH9hc6FDGE3F7pyGB1fUEbSwuTYQdD',
        'jv': '1qiSu1uECCLl4IBZS27FBdJIBivkJ7GwE',
        'ka': '172UFuFRBX2V1aWeXlPSpu9TjS-3cxNaD',
        'kaa': '1kh6hMPUdqO-FIxRY6qaIBZothBURXxbY',
        'kab': '1oKjbZI6ZrrALCqnPCYgIjKNrKDA7ehcs',
        'kbd': '1jNbfrboPOwJmlXQBIv053d7n5WXpMRv7',
        'kg': '1iiu5z-sdJ2JLC4Ja9IgDxpRZklIb6nDx',
        'ki': '1GUtt0QI84c5McyLGGxoi5uwjHOq1d6G8',
        'kj': '1nSxXUSGDlXVCIPGlVpcakRc537MwuKZR',  # leer
        'kk': '1ryC3UN0myckc1awrWhhb6RIi17C0LCuS',
        'kl': '1gXtGtX9gcTXms1IExICnqZUHefrlcIFf',
        'km': '1DS5ATxvxyfn1iWvq2G6qmjZv9pv0T6hD',
        'kn': '1ZGLYMxbb5-29MNmuUfg2xFhYUbkJFMJJ',
        'ko': '12r8tIkTnwKhLJxy71qpIcoLrT6NNhQYm',
        'koi': '1EdG_wZ_Qk124EPAZw-w6rdEhYLsgcvIj',
        'kr': '19VNQtnBA-YL_avWuVeHQHxJZ9MZ04WPF',  # leer
        'krc': '1nReV4Mb7Wdj96czpO5regFbdBPu0zZ_y',
        'ks': '1kzh0Pgrv27WRMstR9MpU8mu7p60TcT-X',
        'ksh': '1iHJvrl2HeRaCumlrx3N7CPrHQ2KuLUkt',
        'ku': '1YqJog7Bkk0fHBCSTxJ9heeE-bfbkbkye',
        'kv': '1s91HI4eq8lQYlZwfrJAgaGlCyAtIhvIJ',
        'kw': '16TaIX2nRfqDp8n7zudd4bqf5abN49dvW',
        'ky': '17HPUKFdKWhUjuR1NOp5f3PQYfMlMCxCT',
        'la': '1NiQuBaUIFEERvVXo6CQLwosPraGyiRYw',
        'lad': '1PEmXCWLCqnjLBomMAYHeObM1AmVHtD08',
        'lb': '1nE4g10xoTU23idmDtOQ0w2QCuizZ6QH_',
        'lbe': '1KOm-AdRcCHfSc1-uYBxBA4GjxXjnIlE-',
        'lez': '1cJAXshrLlF1TZlPHJTpDwEvurIOsz4yR',
        'lg': '1Ur0y7iiEpWBgHECrIrT1OyIC8um_y4th',
        'li': '1TikIqfqcZlSDWhOae1JnjJiDko4nj4Dj',
        'lij': '1ro5ItUcF49iP3JdV82lhCQ07MtZn_VjW',
        'lmo': '1W4rhBy2Pi5SuYWyWbNotOVkVY3kYWS_O',
        'ln': '1bLSV6bWx0CgFm7ByKppZLpYCFL8EIAoD',
        'lo': '1C6SSLeKF3QirjZbAZAcpVX_AXYg_TJG3',
        'lrc': '1GUcS28MlJe_OjeQfS2AJ8uczpD8ut60e',
        'lt': '1gAG6TcMTmC128wWK0rCXRlCTsJY9wFQY',
        'ltg': '12ziP8t_fAAS9JqOCEC0kuJObEyuoiOjD',
        'lv': '1MPuAM04u-AtfybXdpHwCqUpFWbe-zD0_',
        'mai': '1d_nUewBkka2QGEmxCc9v3dTfvo7lPATH',
        'map-bms': '1wrNIE-mqp2xb3lrNdwADe6pb7f35NP6V',
        'mdf': '1BmMGUJy7afuKfhfTBMiKxM3D7FY-JrQ2',
        'mg': '105WaMhcWa-46tCztoj8npUyg0aH18nFL',
        'mh': '1Ej7n6yA1cF1cpD5XneftHtL33iHJwntT',
        'mhr': '1CCPIUaFkEYXiHO0HF8_w07UzVyWchrjS',
        'mi': '1F6au9xQjnF-aNBupGJ1PwaMMM6T_PgdQ',
        'min': '1tVK5SHiCy_DaZSDm3nZBgT5bgWThbJt_',
        'mk': '18NpudytGhSWq_LbmycTDw10cSftlSBGS',
        'ml': '1V73UE-EvcE-vV3V1RTvU4sak6QFcP91y',
        'mn': '14jRXicA87oXZOZllWqUjKBMetNpQEUUp',
        'mo': '1YsLGNMsJ7VsekhdcITQeolzOSK4NzE6U',
        'mr': '1vOr1AIHbgkhTO9Ol9Jx5Wh98Qdyh1QKI',
        'mrj': '1dW-YmEW8a9D5KyXz8ojSdIXWGekNzGzN',
        'ms': '1bs-_5WNRiZBjO-DtcNtkcIle-98homf_',
        'mt': '1L7aU3iGjm6SmPIU74k990qRgHFV9hrL0',
        'mus': '1_b7DcRqiKJFEFwp87cUecqf8A5BDbTIJ',  # leer
        'mwl': '1MfP0jba2jQfGVeJOLq26MjI6fYY7xTPu',
        'my': '16wsIGBhNVd2lC2p6n1X8rdMbiaemeiUM',
        'myv': '1KEqHmfx2pfU-a1tdI_7ZxMQAk5NJzJjB',
        'mzn': '1CflvmYEXZnWwpsBmIs2OvG-zDDvLEMDJ',
        'na': '1r0AVjee5wNnrcgJxQmVGPVKg5YWz1irz',
        'nah': '1fx6eu91NegyueZ1i0XaB07CKjUwjHN7H',
        'nap': '1bhT4sXCJvaTchCIV9mwLBtf3a7OprbVB',
        'nds-nl': '1UIFi8eOCuFYJXSAXZ9pCWwkQMlHaY4ye',
        'nds': '1FLgZIXUWa_vekDt4ndY0B5XL7FNLiulr',
        'ne': '1gEoCjSJmzjIH4kdHsbDZzD6ID4_78ekS',
        'new': '1_-p45Ny4w9UvGuhD8uRNSPPeaARYvESH',
        'ng': '11yxPdkmpmnijQUcnFHZ3xcOmLTYJmN_R',
        'nl': '1dqYXg3ilzVOSQ_tz_dF47elSIvSIhgqd',
        'nn': '1pDrtRhQ001z2WUNMWCZQU3RV_M0BqOmv',
        'no': '1zuT8MI96Ivpiu9mEVFNjwbiM8gJlSzY2',
        'nov': '1l38388Rln0NXsSARMZHmTmyfo5C0wYTd',
        'nrm': '10vxPq1Nci7Wpq4XOvx3dtqODskzjdxJQ',
        'nso': '1iaIV8qlT0RDnbeQlnxJ3RehsG3gU5ePK',
        'nv': '1oN31jT0w3wP9aGwAPz91pSdUytnd9B0g',
        'ny': '1eEKH_rUPC560bfEg11kp3kbe8qWm35IG',
        'oc': '1C01cW8G_j8US-DTrsmeal_ENHTtNWn-H',
        'olo': '1vbDwKZKqFq84dusr1SvDx5JbBcPanx9L',  # leer
        'om': '1q3h22VMbWg2kgVFm-OArR-E4y1yBQ1JX',
        'or': '1k8LwCE8nC7lq6neXDaS3zRn0KOrd9RnS',
        'os': '1u81KAB34aEQfet00dLMRIBJsfRwbDTij',
        'pa': '1JDEHL1VcLHBamgTPBom_Ryi8hk6PBpsu',
        'pag': '1k905VUWnRgY8kFb2P2431Kr4dZuolYGF',
        'pam': '1ssugGyJb8ipispC60B3I6kzMsri1WcvC',
        'pap': '1Za0wfwatxYoD7jGclmTtRoBP0uV_qImQ',
        'pcd': '1csJlKgtG04pdIYCUWhsCCZARKIGlEYPx',
        'pdc': '1Xnms4RXZKZ1BBQmQJEPokmkiweTpouUw',
        'pfl': '1tPQfHX7E0uKMdDSlwNw5aGmaS5bUK0rn',
        'pi': '16b-KxNxzbEuyoNSlI3bfe2YXmdSEsPFu',
        'pih': '1vwyihTnS8_PE5BNK7cTISmIBqGWvsVnF',
        'pl': '1fijjS0LbfpKcoPB5V8c8fH08T8AkXRp9',
        'pms': '12ySc7X9ajWWqMlBjyrPiEdc-qVBuIkbA',
        'pnb': '1RB3-wjluhTKbdTGCsk3nag1bM3m4wENb',
        'pnt': '1ZCUzms6fY4on_fW8uVgO7cEs9KHydHY_',
        'ps': '1WKl9Av6Sqz6aHKyUM5kIh90mzFzyVWH9',
        'pt': '13BX-_4_hcTUp59HDyczFDI32qUB94vUY',
        'qu': '1CB_C4ygtRoegkqgcqfXNHr8oQd-UcvDE',
        'rm': '1YRSGgWoxEqSojHXuBHJnY8vAHr1VgLu-',
        'rmy': '1uFcCyvOWBJWKFQxbkYSp373xUXVl4IgF',
        'rn': '1ekyyb2MvupYGY_E8_BhKvV664sLvW4aE',
        'ro': '1YfeNTSoxU-zJMnyQotLk5X8B_6nHryBu',
        'roa-rup': '150s4H4TdQ5nNYVC6j0E416TUAjBE85yy',
        'roa-tara': '1H6emfQsD_a5yohK4RMPQ-GrnHXqqVgr3',
        'ru': '11gP2s-SYcfS3j9MjPp5C3_nFeQB-8x86',
        'rue': '1OuSglZAndja1J5D5IUmdbt_niTTyEgYK',
        'rw': '1NuhHfi0-B-Xlr_BApijnxCw0WMEltttP',
        'sa': '1P2S3gL_zvKgXLKJJxg-Fb4z8XdlVpQik',
        'sah': '1qz0MpKckzUref2FX_FYiNzI2p4BDc5oR',
        'sc': '1oAYj_Fty4FUwjAOBEBaiZt_cY8dtpDfA',
        'scn': '1sDN9zHkXWYoHYx-DUu-GPvsUgB_IRa8S',
        'sco': '1i8W7KQPj6YZQLop89vZBSybJNgNsvXWR',
        'sd': '1vaNqfv3S8Gl5pQmig3vwWQ3cqRTsXmMR',
        'se': '1RT9xhn0Vl90zjWYDTw5V1L_u1Oh16tpP',
        'sg': '1iIh2oXD2Szz_AygUvTt3_ZK8a3RYEGZ_',
        'sh': '1qPwLiAm6t4__G-zVEOrBgYx6VRmgDgiS',
        'si': '1G5ryceID0TP6SAO42e-HAbIlCvYmnUN7',
        'simple': '1FVV49o_RlK6M5Iw_7zeJOEDQoTa5zSbq',
        'sk': '11mkYvbmAWKTInj6t4Ma8BUPxoR5o6irL',
        'sl': '1fsIZS5LgMzMzZ6T7ogStyj-ILEZIBRvO',
        'sm': '1yefECpKX_Y4R7G2tggIxvc_BvJfOAz-t',
        'sn': '1fYeCjMPvRAv94kvZjiKI-ktIDLkbv0Ve',
        'so': '1Uc-eSZnJb36SgeTvRU3GirXZOlGD_NB6',
        'sq': '11u-53n71O_yjpwRiCQSwgL7N2w72ZptX',
        'sr': '1PGLGlQi8Q0Eac6dib-uuCJAAHK6SF5Pz',
        'srn': '1JKiL3TSXqK1-KhPfAwMK0uqw90WEzg7M',
        'ss': '1e0quNEsA1dn57-IbincF4D82dRWgzQlp',
        'st': '1ny-FBzpBqIDgv6jMcsoFev3Ih65FNZFO',
        'stq': '15Fx32ROy2IM6lSqAPUykkr3CITR6Xd7v',
        'su': '1C0FJum7bYZpnyptBvfAgwJb0TX2hggtO',
        'sv': '1YyqzOSXzK5yrAou9zeTDWH_7s569mDcz',
        'sw': '1_bNTj6T8eXlNAIuHaveleWlHB_22alJs',
        'szl': '1_dXEip1snK4CPVGqH8x7lF5O-6FdCNFW',
        'ta': '1ZFTONsxGtSnC9QB6RpWSvgD_MbZwIhHH',
        'tcy': '15R6u7KQs1vmDSm_aSDrQMJ3Q6q3Be0r7',  # leer
        'te': '11Sx-pBAPeZOXGyv48UNSVMD0AH7uf4YN',
        'tet': '11mr2MYLcv9pz7mHhGGNi5iNCOVErYeOt',
        'tg': '16ttF7HWqM9Cnj4qmgf3ZfNniiOJfZ52w',
        'th': '14xhIt-xr5n9nMuvcwayCGM1-zBCFZquW',
        'ti': '123q5e9MStMShp8eESGtHdSBGLDrCKfJU',
        'tk': '1X-JNInt34BNGhg8A8Peyjw2WjsALdXsD',
        'tl': '1WkQHbWd9cqtTnSHAv0DpUThaBnzeSPTJ',
        'tn': '1fHfQHetZn8-fLuRZEu-cvs-kQYwPvjyL',
        'to': '1cHOLaczYJ8h-OqQgxeoH9vMG3izg6muT',
        'tpi': '1YsRjxVu6NYOrXRb8oqMO9FPaicelFEcu',
        'tr': '1J1Zy02IxvtCK0d1Ba2h_Ulit1mVb9UIX',
        'ts': '1pIcfAt3KmtmDkyhOl-SMSeoM8aP8bOpl',
        'tt': '1vsfzCjj-_bMOn5jBai41TF5GjKJM_Ius',
        'tum': '1NWcg65daI2Bt0awyEgU6apUDbBmiqCus',
        'tw': '1WCYKZIqS7AagS76QFSfbteiOgFNBvNne',
        'ty': '1DIqaP1l-N9VXTNokrlr6EuPMGE765o4h',
        'tyv': '1F3qa05OYLBcjT1lXMurAJFDXP_EesCvM',
        'udm': '1T0YMTAPLOk768sstnewy5Jxgx2RPu3Rb',
        'ug': '1fjezvqlysyZhiQMZdazqLGgk72PqtXAw',
        'uk': '1UMJCHtzxkfLDBJE7NtfN5FeMrnnUVwoh',
        'ur': '1WNaD2TuHvdsF-z0k_emQYchwoQQDFmRk',
        'uz': '11wrG2FSTpRJc2jb5MhgvxjkVDYhT8M-l',
        've': '1PucJ7pJ4CXGEXZ5p_WleZDs2usNz74to',
        'vec': '1cAVjm_y3ehNteDQIYz9yyoq1EKkqOXZ0',
        'vep': '1K_eqV7O6C7KPJWZtmIuzFMKAagj-0O85',
        'vi': '1yQ6nhm1BmG9lD4_NaG1hE5VV6biEaV5f',
        'vls': '1bpQQW6pKHruKJJaKtuggH5rReMXyeVXp',
        'vo': '1D80QRdTpe7H4mHFKpfugscsjX71kiMJN',
        'wa': '1m4B81QYbf74htpInDU5p7d0n0ot8WLPZ',
        'war': '1EC3jsHtu22tHBv6jX_I4rupC5RwV3OYd',
        'wo': '1vChyqNNLu5xYHdyHpACwwpw4l3ptiKlo',
        'wuu': '1_EIn02xCUBcwLOwYnA-lScjS2Lh2ECw6',
        'xal': '19bKXsL1D2UesbB50JPyc9TpG1lNc2POt',
        'xh': '1pPVcxBG3xsCzEnUzlohc_p89gQ9dSJB3',
        'xmf': '1SM9llku6I_ZuZz05mOBuL2lx-KQXvehr',
        'yi': '1WNWr1oV-Nl7c1Jv8x_MiAj2vxRtyQawu',
        'yo': '1yNVOwMOWeglbOcRoZzgd4uwlN5JMynnY',
        'za': '1i7pg162cD_iU9h8dgtI2An8QCcbzUAjB',
        'zea': '1EWSkiSkPBfbyjWjZK0VuKdpqFnFOpXXQ',
        'zh-classical': '1uUKZamNp08KA7s7794sKPOqPALvo_btl',
        'zh-min-nan': '1oSgz3YBXLGUgI7kl-uMOC_ww6L0FNFmp',
        'zh-yue': '1zhwlUeeiyOAU1QqwqZ8n91yXIRPFA7UE',
        'zh': '1LZ96GUhkVHQU-aj2C3WOrtffOp0U3Z7f',
        'zu': '1FyXl_UK1737XB3drqQFhGXiJrJckiB1W'
    }
    return languages_ids[language]


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
                'https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip',
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


class EUROPARL_NER_GERMAN(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
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

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: 'text', 1: 'lemma', 2: 'pos', 3: 'np', 4: 'ner'}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        europarl_ner_german_path = "https://nlpado.de/~sebastian/software/ner/"
        cached_path(f"{europarl_ner_german_path}ep-96-04-15.conll", Path("datasets") / dataset_name)
        cached_path(f"{europarl_ner_german_path}ep-96-04-16.conll", Path("datasets") / dataset_name)

        add_IOB_tags(data_file=Path(data_folder / "ep-96-04-15.conll"), encoding="latin-1", ner_column=4)
        add_IOB_tags(data_file=Path(data_folder / "ep-96-04-16.conll"), encoding="latin-1", ner_column=4)

        super(EUROPARL_NER_GERMAN, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            encoding="latin-1",
            in_memory=in_memory,
            train_file='ep-96-04-16.conll',
            test_file='ep-96-04-15.conll'
        )


class GERMEVAL_14(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
    ):
        """
        Initialize the GermEval NER corpus for German. This is only possible if you've manually downloaded it to your
        machine. Obtain the corpus from https://sites.google.com/site/germeval2014ner/data and put it into some folder.
        Then point the base_path parameter in the constructor to this folder
        :param base_path: Path to the GermEval corpus on your machine
        :param tag_to_bioes: 'ner' by default, should not be changed.
        :param in_memory:If True, keeps dataset in memory giving speedups in training.
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
            log.warning(f'WARNING: GermEval-14 dataset not found at "{data_folder}".')
            log.warning(
                'Instructions for obtaining the data can be found here: https://sites.google.com/site/germeval2014ner/data"'
            )
            log.warning("-" * 100)
        super(GERMEVAL_14, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            comment_symbol="#",
            in_memory=in_memory,
        )


class INSPEC(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "keyword",
            in_memory: bool = True,
    ):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "keyword"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        inspec_path = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/Inspec"
        cached_path(f"{inspec_path}/train.txt", Path("datasets") / dataset_name)
        cached_path(f"{inspec_path}/test.txt", Path("datasets") / dataset_name)
        if not "dev.txt" in os.listdir(data_folder):
            cached_path(f"{inspec_path}/valid.txt", Path("datasets") / dataset_name)
            # rename according to train - test - dev - convention
            os.rename(data_folder / "valid.txt", data_folder / "dev.txt")

        super(INSPEC, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class LER_GERMAN(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = False,
    ):
        """
        Initialize the LER_GERMAN (Legal Entity Recognition) corpus. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training. Not recommended due to heavy RAM usage.
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
        ler_path = "https://raw.githubusercontent.com/elenanereiss/Legal-Entity-Recognition/master/data/"
        cached_path(f"{ler_path}ler.conll", Path("datasets") / dataset_name)

        super(LER_GERMAN, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
            train_file='ler.conll'
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


class NER_FINNISH(ColumnCorpus):
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
        ner_finnish_path = "https://raw.githubusercontent.com/mpsilfve/finer-data/master/data/digitoday."
        cached_path(f"{ner_finnish_path}2014.train.csv", Path("datasets") / dataset_name)
        cached_path(f"{ner_finnish_path}2014.dev.csv", Path("datasets") / dataset_name)
        cached_path(f"{ner_finnish_path}2015.test.csv", Path("datasets") / dataset_name)

        _remove_lines_without_annotations(data_file=Path(data_folder / "digitoday.2015.test.csv"))

        super(NER_FINNISH, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory, skip_first_line=True
        )


def _remove_lines_without_annotations(data_file: Union[str, Path] = None):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    with open(data_file, 'w') as f:
        for line in lines:
            if len(line.split()) != 1:
                f.write(line)


class NER_SWEDISH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True,
    ):
        """
        Initialize the NER_SWEDISH corpus for Swedish. The first time you call this constructor it will automatically
        download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
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
        ner_spraakbanken_path = "https://raw.githubusercontent.com/klintan/swedish-ner-corpus/master/"
        cached_path(f"{ner_spraakbanken_path}test_corpus.txt", Path("datasets") / dataset_name)
        cached_path(f"{ner_spraakbanken_path}train_corpus.txt", Path("datasets") / dataset_name)

        # data is not in IOB2 format. Thus we transform it to IOB2
        add_IOB2_tags(data_file=Path(data_folder / "test_corpus.txt"))
        add_IOB2_tags(data_file=Path(data_folder / "train_corpus.txt"))

        super(NER_SWEDISH, self).__init__(
            data_folder,
            columns,
            tag_to_bioes=tag_to_bioes,
            in_memory=in_memory,
        )


class SEMEVAL2017(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "keyword",
            in_memory: bool = True,
    ):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "keyword"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        semeval2017_path = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/SemEval-2017"
        cached_path(f"{semeval2017_path}/train.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2017_path}/test.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2017_path}/dev.txt", Path("datasets") / dataset_name)

        super(SEMEVAL2017, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


class SEMEVAL2010(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "keyword",
            in_memory: bool = True,
    ):

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "keyword"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        semeval2010_path = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/processed_semeval-2010"
        cached_path(f"{semeval2010_path}/train.txt", Path("datasets") / dataset_name)
        cached_path(f"{semeval2010_path}/test.txt", Path("datasets") / dataset_name)

        super(SEMEVAL2010, self).__init__(
            data_folder, columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
        )


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


class BIOSCOPE(ColumnCorpus):

    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "tag"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        bioscope_path = "https://raw.githubusercontent.com/whoisjones/BioScopeSequenceLabelingData/master/sequence_labeled/"
        cached_path(f"{bioscope_path}output.txt", Path("datasets") / dataset_name)

        super(BIOSCOPE, self).__init__(
            data_folder, columns, in_memory=in_memory, train_file="output.txt"
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
                encoding="utf-8"
        ) as out:
            for line in f:
                line = line.decode("utf-8")
                words = line.split(" ")
                for word in words:
                    out.write("\t".join(word.split("|")) + "\n")

class UP_GERMAN(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            document_as_sequence: bool = False,
    ):
        """
        Initialize the German dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions.

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
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
        )

class UP_FRENCH(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            document_as_sequence: bool = False,
    ):
        """
        Initialize the French dataset from the Universal Propositions Bank, comming from that webpage:
        https://github.com/System-T/UniversalPropositions.

        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {1: "text", 9: "frame"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
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
        )