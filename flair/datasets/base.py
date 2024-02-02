import logging
from abc import abstractmethod
from pathlib import Path
from typing import Generic, List, Optional, Union

import torch.utils.data.dataloader
from deprecated.sphinx import deprecated

from flair.data import DT, FlairDataset, Sentence, Tokenizer
from flair.tokenization import SegtokTokenizer, SpaceTokenizer

log = logging.getLogger("flair")


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=list,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlairDatapointDataset(FlairDataset, Generic[DT]):
    """A simple Dataset object to wrap a List of Datapoints, for example Sentences."""

    def __init__(self, datapoints: Union[DT, List[DT]]) -> None:
        """Instantiate FlairDatapointDataset.

        Args:
            datapoints: DT or List of DT that make up FlairDatapointDataset
        """
        # cast to list if necessary
        if not isinstance(datapoints, list):
            datapoints = [datapoints]
        self.datapoints = datapoints

    def is_in_memory(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, index: int = 0) -> DT:
        return self.datapoints[index]


class SentenceDataset(FlairDatapointDataset):
    @deprecated(version="0.11", reason="The 'SentenceDataset' class was renamed to 'FlairDatapointDataset'")
    def __init__(self, sentences: Union[Sentence, List[Sentence]]) -> None:
        super().__init__(sentences)


class StringDataset(FlairDataset):
    """A Dataset taking string as input and returning Sentence during iteration."""

    def __init__(
        self,
        texts: Union[str, List[str]],
        use_tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
    ) -> None:
        """Instantiate StringDataset.

        Args:
            texts: a string or List of string that make up StringDataset
            use_tokenizer:
                Custom tokenizer to use. If instead of providing a function, this parameter is just set to True,
                :class:`flair.tokenization.SegTokTokenizer` will be used.
        """
        # cast to list if necessary
        if isinstance(texts, str):
            texts = [texts]
        self.texts = texts
        self.use_tokenizer = use_tokenizer

    @abstractmethod
    def is_in_memory(self) -> bool:
        return True

    def __len__(self) -> int:
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
        categories_field: Optional[List[str]] = None,
        max_tokens_per_doc: int = -1,
        max_chars_per_doc: int = -1,
        tokenizer: Tokenizer = SegtokTokenizer(),
        in_memory: bool = True,
        tag_type: str = "class",
    ) -> None:
        """Reads Mongo collections.

        Each collection should contain one document/text per item.

        Each item should have the following format:
        {
        'Beskrivning': 'Abrahamsby. Gård i Gottröra sn, Långhundra hd, Stockholms län, nära Långsjön.',
        'Län':'Stockholms län',
        'Härad': 'Långhundra',
        'Församling': 'Gottröra',
        'Plats': 'Abrahamsby'
        }

        Args:
            query: Query, e.g. {'Län': 'Stockholms län'}
            host: Host, e.g. 'localhost',
            port: Port, e.g. 27017
            database: Database, e.g. 'rosenberg',
            collection: Collection, e.g. 'book',
            text_field: Text field, e.g. 'Beskrivning',
            categories_field: List of category fields, e.g ['Län', 'Härad', 'Tingslag', 'Församling', 'Plats'],
            max_tokens_per_doc: Takes at most this amount of tokens per document. If set to -1 all documents are taken as is.
            max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
            max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
            tokenizer: Custom tokenizer to use (default SegtokTokenizer)
            in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
            tag_type: The tag type to assign labels to.

        Returns: list of sentences
        """
        # first, check if pymongo is installed
        try:
            import pymongo
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "pymongo" is not installed!')
            log.warning('To use MongoDataset, please first install with "pip install pymongo"')
            log.warning("-" * 100)

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
        self.tag_type = tag_type

        start = 0

        if self.in_memory:
            for document in self.__cursor.find(filter=query, skip=start, limit=0):
                sentence = self._parse_document_to_sentence(
                    document[self.text],
                    [document.get(c, "") for c in self.categories],
                    tokenizer,
                )
                if sentence is not None and len(sentence.tokens) > 0:
                    self.sentences.append(sentence)
                    self.total_sentence_count += 1
        else:
            self.indices = self.__cursor.find().distinct("_id")
            self.total_sentence_count = self.__cursor.count_documents()

    def _parse_document_to_sentence(
        self,
        text: str,
        labels: List[str],
        tokenizer: Union[bool, Tokenizer],
    ):
        if self.max_chars_per_doc > 0:
            text = text[: self.max_chars_per_doc]

        if text and labels:
            sentence = Sentence(text, use_tokenizer=tokenizer)
            for label in labels:
                sentence.add_label(self.tag_type, label)

            if self.max_tokens_per_doc > 0:
                sentence.tokens = sentence.tokens[: min(len(sentence), self.max_tokens_per_doc)]

            return sentence
        return None

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self) -> int:
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            return self.sentences[index]
        else:
            document = self.__cursor.find_one({"_id": index})
            sentence = self._parse_document_to_sentence(
                document[self.text],
                [document.get(c, "") for c in self.categories],
                self.tokenizer,
            )
            return sentence


def find_train_dev_test_files(data_folder, dev_file, test_file, train_file, autofind_splits=True):
    if isinstance(data_folder, str):
        data_folder: Path = Path(data_folder)

    if train_file is not None:
        train_file = data_folder / train_file
    if test_file is not None:
        test_file = data_folder / test_file
    if dev_file is not None:
        dev_file = data_folder / dev_file

    suffixes_to_ignore = {".gz", ".swp"}

    # automatically identify train / test / dev files
    if train_file is None and autofind_splits:
        for file in data_folder.iterdir():
            file_name = file.name
            if not suffixes_to_ignore.isdisjoint(file.suffixes):
                continue
            if "train" in file_name and "54019" not in file_name:
                train_file = file
            if "dev" in file_name:
                dev_file = file
            if "testa" in file_name:
                dev_file = file
            if "testb" in file_name:
                test_file = file

        # if no test file is found, take any file with 'test' in name
        if test_file is None and autofind_splits:
            for file in data_folder.iterdir():
                file_name = file.name
                if not suffixes_to_ignore.isdisjoint(file.suffixes):
                    continue
                if "test" in file_name:
                    test_file = file

    log.info(f"Reading data from {data_folder}")
    log.info(f"Train: {train_file}")
    log.info(f"Dev: {dev_file}")
    log.info(f"Test: {test_file}")

    return dev_file, test_file, train_file
