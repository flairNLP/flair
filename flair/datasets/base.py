import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Callable

import torch.utils.data.dataloader
from torch.utils.data.dataset import Subset, ConcatDataset

from flair.data import (
    Sentence,
    Token,
    FlairDataset,
    space_tokenizer,
    segtok_tokenizer,
)


log = logging.getLogger("flair")


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
