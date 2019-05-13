from torch.utils.data import Dataset, random_split
from typing import List, Dict, Union
import re
import logging
from pathlib import Path

import flair
from flair.data import Sentence, TaggedCorpus, Token
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class ColumnCorpus(TaggedCorpus):
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


class UniversalDependenciesCorpus(TaggedCorpus):
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

    def __iter__(self):
        return iter(self.sentences)


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

    def __iter__(self):
        return iter(self.sentences)


class CONLL_03(ColumnCorpus):
    def __init__(self, base_path=None, tag_to_biloes: str = "ner"):
        columns = {0: "text", 1: "pos", 2: "np", 3: "ner"}

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / "conll_03"

        super(CONLL_03, self).__init__(
            data_folder, columns, tag_to_biloes=tag_to_biloes
        )


class CONLL_2000(ColumnCorpus):
    def __init__(self, base_path=None, tag_to_biloes: str = "np"):
        columns = {0: "text", 1: "pos", 2: "np"}

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / "conll_2000"

        super(CONLL_2000, self).__init__(
            data_folder, columns, tag_to_biloes=tag_to_biloes
        )


class UD_ENGLISH(UniversalDependenciesCorpus):
    def __init__(self, base_path=None):
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / "ud_english"

        # download data if necessary
        web_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
        cached_path(f"{web_path}/en_ewt-ud-dev.conllu", Path("datasets") / "ud_english")
        cached_path(
            f"{web_path}/en_ewt-ud-test.conllu", Path("datasets") / "ud_english"
        )
        cached_path(
            f"{web_path}/en_ewt-ud-train.conllu", Path("datasets") / "ud_english"
        )

        super(UD_ENGLISH, self).__init__(data_folder)


class UD_GERMAN(UniversalDependenciesCorpus):
    def __init__(self, base_path=None):
        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / "ud_german"

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master"
        cached_path(f"{ud_path}/de_gsd-ud-dev.conllu", Path("datasets") / "ud_german")
        cached_path(f"{ud_path}/de_gsd-ud-test.conllu", Path("datasets") / "ud_german")
        cached_path(f"{ud_path}/de_gsd-ud-train.conllu", Path("datasets") / "ud_german")

        super(UD_GERMAN, self).__init__(data_folder)
