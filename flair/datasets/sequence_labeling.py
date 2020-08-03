import logging
import re
import os
from pathlib import Path
from typing import Union, Dict, List

import flair
from flair.data import Corpus, FlairDataset, Sentence, Token
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
        fields: List[str] = re.split(self.column_delimiter, line)
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


def add_IOB2_tags(data_file: Union[str, Path], encoding: str = "utf8"):
    """
Function that adds IOB2 tags if only chunk names are provided (e.g. words are tagged PER instead
of B-PER or I-PER). Replaces '0' with 'O' as the no-chunk tag since ColumnCorpus expects
the letter 'O'. Additionaly it removes lines with no tags in the data file and can also
be used if the data is only partialy IOB tagged.
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


class GERMEVAL_14(ColumnCorpus):
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
                'Instructions for obtaining the data can be found here: https://sites.google.com/site/germeval2014ner/home/"'
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
