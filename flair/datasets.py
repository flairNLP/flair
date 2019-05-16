from torch.utils.data import Dataset, random_split
from typing import List, Dict, Union
import re
import logging
from pathlib import Path

import flair
from flair.data import Sentence, Corpus, Token
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
        tag_to_biloes=None,
    ):
        """
        Helper function to get a TaggedCorpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_biloes: whether to convert to BILOES tagging scheme
        :return: a TaggedCorpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith(".gz"):
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
                    if file_name.endswith(".gz"):
                        continue
                    if "test" in file_name:
                        test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        # get train data
        train = ColumnDataset(train_file, column_format, tag_to_biloes)

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            test = ColumnDataset(test_file, column_format, tag_to_biloes)
        else:
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - test_size, test_size])
            train = splits[0]
            test = splits[1]

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            dev = ColumnDataset(dev_file, column_format, tag_to_biloes)
        else:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]

        super(ColumnCorpus, self).__init__(train, dev, test, name=data_folder.name)


class UniversalDependenciesCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        train_file=None,
        test_file=None,
        dev_file=None,
    ):
        """
        Helper function to get a TaggedCorpus from CoNLL-U column-formatted task data such as the UD corpora

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :return: a TaggedCorpus with annotated train, dev and test data
        """
        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Test: {}".format(test_file))
        log.info("Dev: {}".format(dev_file))

        # get train data
        train = UniversalDependenciesDataset(train_file)

        # get test data
        test = UniversalDependenciesDataset(test_file)

        # get dev data
        dev = UniversalDependenciesDataset(dev_file)

        super(UniversalDependenciesCorpus, self).__init__(
            train, dev, test, name=data_folder.name
        )


class ClassificationCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        train_file=None,
        test_file=None,
        dev_file=None,
        use_tokenizer: bool = True,
        max_tokens_per_doc=-1,
        in_memory: bool = False,
    ):
        """
        Helper function to get a TaggedCorpus from text classification-formatted task data

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :return: a TaggedCorpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if "train" in file_name:
                    train_file = file
                if "test" in file_name:
                    test_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        train: Dataset = ClassificationDataset(
            train_file,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            in_memory=in_memory,
        )
        test: Dataset = ClassificationDataset(
            test_file,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            in_memory=in_memory,
        )

        if dev_file is not None:
            dev: Dataset = ClassificationDataset(
                dev_file,
                use_tokenizer=use_tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                in_memory=in_memory,
            )
        else:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]

        super(ClassificationCorpus, self).__init__(
            train, dev, test, name=data_folder.name
        )


class ColumnDataset(Dataset):
    def __init__(
        self,
        path_to_column_file: Path,
        column_name_map: Dict[int, str],
        tag_to_biloes=None,
    ):
        assert path_to_column_file.exists()

        self.sentences: List[Sentence] = []

        try:
            lines: List[str] = open(
                str(path_to_column_file), encoding="utf-8"
            ).read().strip().split("\n")
        except:
            log.info(
                'UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(
                    path_to_column_file
                )
            )
            lines: List[str] = open(
                str(path_to_column_file), encoding="latin1"
            ).read().strip().split("\n")

        # most data sets have the token text in the first column, if not, pass 'text' as column
        text_column: int = 0
        for column in column_name_map:
            if column_name_map[column] == "text":
                text_column = column

        sentence: Sentence = Sentence()
        for line in lines:

            if line.startswith("#"):
                continue

            if line.strip().replace("﻿", "") == "":
                if len(sentence) > 0:
                    sentence.infer_space_after()
                    self.sentences.append(sentence)
                sentence: Sentence = Sentence()

            else:
                fields: List[str] = re.split("\s+", line)
                token = Token(fields[text_column])
                for column in column_name_map:
                    if len(fields) > column:
                        if column != text_column:
                            token.add_tag(column_name_map[column], fields[column])

                sentence.add_token(token)

        if len(sentence.tokens) > 0:
            sentence.infer_space_after()
            self.sentences.append(sentence)

        if tag_to_biloes is not None:
            # convert tag scheme to iobes
            for sentence in self.sentences:
                sentence.convert_tag_scheme(
                    tag_type=tag_to_biloes, target_scheme="iobes"
                )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        return self.sentences[index]


class UniversalDependenciesDataset(Dataset):
    def __init__(self, path_to_conll_file: Path):
        assert path_to_conll_file.exists()

        self.sentences: List[Sentence] = []

        lines: List[str] = open(
            path_to_conll_file, encoding="utf-8"
        ).read().strip().split("\n")

        sentence: Sentence = Sentence()
        for line in lines:

            fields: List[str] = re.split("\t+", line)
            if line == "":
                if len(sentence) > 0:
                    self.sentences.append(sentence)
                sentence: Sentence = Sentence()

            elif line.startswith("#"):
                continue
            elif "." in fields[0]:
                continue
            elif "-" in fields[0]:
                continue
            else:
                token = Token(fields[1], head_id=int(fields[6]))
                token.add_tag("lemma", str(fields[2]))
                token.add_tag("upos", str(fields[3]))
                token.add_tag("pos", str(fields[4]))
                token.add_tag("dependency", str(fields[7]))

                for morph in str(fields[5]).split("|"):
                    if not "=" in morph:
                        continue
                    token.add_tag(morph.split("=")[0].lower(), morph.split("=")[1])

                if len(fields) > 10 and str(fields[10]) == "Y":
                    token.add_tag("frame", str(fields[11]))

                sentence.add_token(token)

        if len(sentence.tokens) > 0:
            self.sentences.append(sentence)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        return self.sentences[index]


class ClassificationDataset(Dataset):
    def __init__(
        self,
        path_to_file: Union[str, Path],
        max_tokens_per_doc=-1,
        use_tokenizer=True,
        in_memory: bool = True,
    ):
        """
        Reads a data file for text classification. The file should contain one document/text per line.
        The line should have the following format:
        __label__<class_name> <text>
        If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,
        __label__<class_name_1> __label__<class_name_2> <text>
        :param path_to_file: the path to the data file
        :param max_tokens_per_doc: Takes at most this amount of tokens per document. If set to -1 all documents are taken as is.
        :return: list of sentences
        """
        if type(path_to_file) == str:
            path_to_file: Path = Path(path_to_file)

        assert path_to_file.exists()

        self.label_prefix = "__label__"

        self.in_memory = in_memory
        self.use_tokenizer = use_tokenizer

        if self.in_memory:
            self.sentences = []
        else:
            self.indices = []

        self.total_sentence_count: int = 0

        self.path_to_file = path_to_file

        # self.file = open(str(path_to_file), encoding="utf-8")

        with open(str(path_to_file), encoding="utf-8") as f:
            line = f.readline()
            position = 0
            while line:
                sentence = self._parse_line_to_sentence(
                    line, self.label_prefix, use_tokenizer
                )

                if (
                    sentence is not None
                    and len(sentence) > max_tokens_per_doc
                    and max_tokens_per_doc > 0
                ):
                    sentence.tokens = sentence.tokens[:max_tokens_per_doc]
                if sentence is not None and len(sentence.tokens) > 0:
                    if self.in_memory:
                        self.sentences.append(sentence)
                    else:
                        self.indices.append(position)
                    self.total_sentence_count += 1

                position = f.tell()
                line = f.readline()

    def _parse_line_to_sentence(
        self, line: str, label_prefix: str, use_tokenizer: bool = True
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

        if text and labels:
            sentence = Sentence(text, labels=labels, use_tokenizer=use_tokenizer)
            return sentence
        return None

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
                    line, self.label_prefix, self.use_tokenizer
                )
                return sentence


class CONLL_03(ColumnCorpus):
    def __init__(self, base_path=None, tag_to_biloes: str = "ner"):

        # column format
        columns = {0: "text", 1: "pos", 2: "np", 3: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(CONLL_03, self).__init__(
            data_folder, columns, tag_to_biloes=tag_to_biloes
        )


class CONLL_03_DUTCH(ColumnCorpus):
    def __init__(self, base_path=None, tag_to_biloes: str = "ner"):

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
            data_folder, columns, tag_to_biloes=tag_to_biloes
        )


class CONLL_03_SPANISH(ColumnCorpus):
    def __init__(self, base_path=None, tag_to_biloes: str = "ner"):

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
            data_folder, columns, tag_to_biloes=tag_to_biloes
        )


class GERMEVAL(ColumnCorpus):
    def __init__(self, base_path=None, tag_to_biloes: str = "ner"):

        # column format
        columns = {1: "text", 2: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(GERMEVAL, self).__init__(
            data_folder, columns, tag_to_biloes=tag_to_biloes
        )


class CONLL_2000(ColumnCorpus):
    def __init__(self, base_path=None, tag_to_biloes: str = "np"):

        # column format
        columns = {0: "text", 1: "pos", 2: "np"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        super(CONLL_2000, self).__init__(
            data_folder, columns, tag_to_biloes=tag_to_biloes
        )


class UD_ENGLISH(UniversalDependenciesCorpus):
    def __init__(self, base_path=None):

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

        super(UD_ENGLISH, self).__init__(data_folder)


class UD_GERMAN(UniversalDependenciesCorpus):
    def __init__(self, base_path=None):

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

        super(UD_GERMAN, self).__init__(data_folder)
