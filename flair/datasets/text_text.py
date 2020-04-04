import logging
from pathlib import Path
from typing import List, Union

import flair
from flair.data import (
    Sentence,
    Corpus,
    FlairDataset,
    DataPair,
)
from flair.file_utils import cached_path, unzip_file

log = logging.getLogger("flair")


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