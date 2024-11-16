import csv
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Optional, Union

import flair
from flair.data import (
    Corpus,
    DataPair,
    FlairDataset,
    Sentence,
    Tokenizer,
    _iter_dataset,
)
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path, unpack_file, unzip_file
from flair.tokenization import SegtokTokenizer, SpaceTokenizer

log = logging.getLogger("flair")


class ClassificationCorpus(Corpus):
    """A classification corpus from FastText-formatted text files."""

    def __init__(
        self,
        data_folder: Union[str, Path],
        label_type: str = "class",
        train_file=None,
        test_file=None,
        dev_file=None,
        truncate_to_max_tokens: int = -1,
        truncate_to_max_chars: int = -1,
        filter_if_longer_than: int = -1,
        tokenizer: Union[bool, Tokenizer] = SegtokTokenizer(),
        memory_mode: str = "partial",
        label_name_map: Optional[dict[str, str]] = None,
        skip_labels: Optional[list[str]] = None,
        allow_examples_without_labels=False,
        sample_missing_splits: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        """Instantiates a Corpus from text classification-formatted task data.

        Args:
            data_folder: base folder with the task data
            label_type: name of the label
            train_file: the name of the train file
            test_file: the name of the test file
            dev_file: the name of the dev file, if None, dev data is sampled from train
            truncate_to_max_tokens: If set, truncates each Sentence to a maximum number of tokens
            truncate_to_max_chars: If set, truncates each Sentence to a maximum number of chars
            filter_if_longer_than: If set, filters documents that are longer that the specified number of tokens.
            tokenizer: Tokenizer for dataset, default is SegtokTokenizer
            memory_mode: Set to what degree to keep corpus in memory ('full', 'partial' or 'disk'). Use 'full' if full corpus and all embeddings fits into memory for speedups during training. Otherwise use 'partial' and if even this is too much for your memory, use 'disk'.
            label_name_map: Optionally map label names to different schema.
            allow_examples_without_labels: set to True to allow Sentences without label in the corpus.
            encoding: Default is 'utf-8' but some datasets are in 'latin-1
        """
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        train: FlairDataset = ClassificationDataset(
            train_file,
            label_type=label_type,
            tokenizer=tokenizer,
            truncate_to_max_tokens=truncate_to_max_tokens,
            truncate_to_max_chars=truncate_to_max_chars,
            filter_if_longer_than=filter_if_longer_than,
            memory_mode=memory_mode,
            label_name_map=label_name_map,
            skip_labels=skip_labels,
            allow_examples_without_labels=allow_examples_without_labels,
            encoding=encoding,
        )

        # use test_file to create test split if available
        test = (
            ClassificationDataset(
                test_file,
                label_type=label_type,
                tokenizer=tokenizer,
                truncate_to_max_tokens=truncate_to_max_tokens,
                truncate_to_max_chars=truncate_to_max_chars,
                filter_if_longer_than=filter_if_longer_than,
                memory_mode=memory_mode,
                label_name_map=label_name_map,
                skip_labels=skip_labels,
                allow_examples_without_labels=allow_examples_without_labels,
                encoding=encoding,
            )
            if test_file is not None
            else None
        )

        # use dev_file to create test split if available
        dev = (
            ClassificationDataset(
                dev_file,
                label_type=label_type,
                tokenizer=tokenizer,
                truncate_to_max_tokens=truncate_to_max_tokens,
                truncate_to_max_chars=truncate_to_max_chars,
                filter_if_longer_than=filter_if_longer_than,
                memory_mode=memory_mode,
                label_name_map=label_name_map,
                skip_labels=skip_labels,
                allow_examples_without_labels=allow_examples_without_labels,
                encoding=encoding,
            )
            if dev_file is not None
            else None
        )

        super().__init__(train, dev, test, name=str(data_folder), sample_missing_splits=sample_missing_splits)

        log.info(f"Initialized corpus {self.name} (label type name is '{label_type}')")


class ClassificationDataset(FlairDataset):
    """Dataset for classification instantiated from a single FastText-formatted file."""

    def __init__(
        self,
        path_to_file: Union[str, Path],
        label_type: str,
        truncate_to_max_tokens=-1,
        truncate_to_max_chars=-1,
        filter_if_longer_than: int = -1,
        tokenizer: Union[bool, Tokenizer] = SegtokTokenizer(),
        memory_mode: str = "partial",
        label_name_map: Optional[dict[str, str]] = None,
        skip_labels: Optional[list[str]] = None,
        allow_examples_without_labels=False,
        encoding: str = "utf-8",
    ) -> None:
        """Reads a data file for text classification.

        The file should contain one document/text per line.
        The line should have the following format:
        __label__<class_name> <text>
        If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,
        __label__<class_name_1> __label__<class_name_2> <text>
        :param path_to_file: the path to the data file
        :param label_type: name of the label
        :param truncate_to_max_tokens: If set, truncates each Sentence to a maximum number of tokens
        :param truncate_to_max_chars: If set, truncates each Sentence to a maximum number of chars
        :param filter_if_longer_than: If set, filters documents that are longer that the specified number of tokens.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param memory_mode: Set to what degree to keep corpus in memory ('full', 'partial' or 'disk'). Use 'full'
        if full corpus and all embeddings fits into memory for speedups during training. Otherwise use 'partial' and if
        even this is too much for your memory, use 'disk'.
        :param label_name_map: Optionally map label names to different schema.
        :param allow_examples_without_labels: set to True to allow Sentences without label in the Dataset.
        :param encoding: Default is 'utf-8' but some datasets are in 'latin-1
        :return: list of sentences
        """
        path_to_file = Path(path_to_file)

        assert path_to_file.exists()

        self.label_prefix = "__label__"
        self.label_type = label_type

        self.memory_mode = memory_mode
        self.tokenizer = tokenizer

        if self.memory_mode == "full":
            self.sentences = []
        if self.memory_mode == "partial":
            self.lines = []
        if self.memory_mode == "disk":
            self.indices = []

        self.total_sentence_count: int = 0
        self.truncate_to_max_chars = truncate_to_max_chars
        self.truncate_to_max_tokens = truncate_to_max_tokens
        self.filter_if_longer_than = filter_if_longer_than
        self.label_name_map = label_name_map
        self.allow_examples_without_labels = allow_examples_without_labels

        self.path_to_file = path_to_file

        with open(str(path_to_file), encoding=encoding) as f:
            line = f.readline()
            position = 0
            while line:
                if ("__label__" not in line and not allow_examples_without_labels) or (
                    " " not in line and "\t" not in line
                ):
                    position = f.tell()
                    line = f.readline()
                    continue

                if 0 < self.filter_if_longer_than < len(line.split(" ")):
                    position = f.tell()
                    line = f.readline()
                    continue

                # if data point contains black-listed label, do not use
                if skip_labels:
                    skip = False
                    for skip_label in skip_labels:
                        if "__label__" + skip_label in line:
                            skip = True
                    if skip:
                        line = f.readline()
                        continue

                if self.memory_mode == "full":
                    sentence = self._parse_line_to_sentence(line, self.label_prefix, tokenizer)
                    if sentence is not None and len(sentence.tokens) > 0:
                        self.sentences.append(sentence)
                        self.total_sentence_count += 1

                if self.memory_mode == "partial" or self.memory_mode == "disk":
                    # first check if valid sentence
                    words = line.split()
                    l_len = 0
                    label = False
                    for i in range(len(words)):
                        if words[i].startswith(self.label_prefix):
                            l_len += len(words[i]) + 1
                            label = True
                        else:
                            break
                    text = line[l_len:].strip()

                    # if so, add to indices
                    if text and (label or allow_examples_without_labels):
                        if self.memory_mode == "partial":
                            self.lines.append(line)
                            self.total_sentence_count += 1

                        if self.memory_mode == "disk":
                            self.indices.append(position)
                            self.total_sentence_count += 1

                position = f.tell()
                line = f.readline()

    def _parse_line_to_sentence(self, line: str, label_prefix: str, tokenizer: Union[bool, Tokenizer]):
        words = line.split()

        labels = []
        l_len = 0

        for i in range(len(words)):
            if words[i].startswith(label_prefix):
                l_len += len(words[i]) + 1
                label = words[i].replace(label_prefix, "")

                if self.label_name_map and label in self.label_name_map:
                    label = self.label_name_map[label]

                labels.append(label)
            else:
                break

        text = line[l_len:].strip()

        if self.truncate_to_max_chars > 0:
            text = text[: self.truncate_to_max_chars]

        if text and (labels or self.allow_examples_without_labels):
            sentence = Sentence(text, use_tokenizer=tokenizer)

            for label in labels:
                sentence.add_label(self.label_type, label)

            if sentence is not None and 0 < self.truncate_to_max_tokens < len(sentence):
                sentence.tokens = sentence.tokens[: self.truncate_to_max_tokens]

            return sentence
        return None

    def is_in_memory(self) -> bool:
        return self.memory_mode not in ["disk", "partial"]

    def __len__(self) -> int:
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.memory_mode == "full":
            return self.sentences[index]

        if self.memory_mode == "partial":
            sentence = self._parse_line_to_sentence(self.lines[index], self.label_prefix, self.tokenizer)
            return sentence

        if self.memory_mode == "disk":
            with open(str(self.path_to_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence = self._parse_line_to_sentence(line, self.label_prefix, self.tokenizer)
                return sentence
        raise AssertionError


class CSVClassificationCorpus(Corpus):
    """Classification corpus instantiated from CSV data files."""

    def __init__(
        self,
        data_folder: Union[str, Path],
        column_name_map: dict[int, str],
        label_type: str,
        name: str = "csv_corpus",
        train_file=None,
        test_file=None,
        dev_file=None,
        max_tokens_per_doc=-1,
        max_chars_per_doc=-1,
        tokenizer: Tokenizer = SegtokTokenizer(),
        in_memory: bool = False,
        skip_header: bool = False,
        encoding: str = "utf-8",
        no_class_label=None,
        sample_missing_splits: Union[bool, str] = True,
        **fmtparams,
    ) -> None:
        """Instantiates a Corpus for text classification from CSV column formatted data.

        :param data_folder: base folder with the task data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param label_type: name of the label
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param tokenizer: Tokenizer for dataset, default is SegtokTokenizer
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param skip_header: If True, skips first line because it is header
        :param encoding: Default is 'utf-8' but some datasets are in 'latin-1
        :param fmtparams: additional parameters for the CSV file reader
        :return: a Corpus with annotated train, dev and test data
        """
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

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
            no_class_label=no_class_label,
            **fmtparams,
        )

        test = (
            CSVClassificationDataset(
                test_file,
                column_name_map,
                label_type=label_type,
                tokenizer=tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                max_chars_per_doc=max_chars_per_doc,
                in_memory=in_memory,
                skip_header=skip_header,
                encoding=encoding,
                no_class_label=no_class_label,
                **fmtparams,
            )
            if test_file is not None
            else None
        )

        dev = (
            CSVClassificationDataset(
                dev_file,
                column_name_map,
                label_type=label_type,
                tokenizer=tokenizer,
                max_tokens_per_doc=max_tokens_per_doc,
                max_chars_per_doc=max_chars_per_doc,
                in_memory=in_memory,
                skip_header=skip_header,
                encoding=encoding,
                no_class_label=no_class_label,
                **fmtparams,
            )
            if dev_file is not None
            else None
        )

        super().__init__(train, dev, test, name=name, sample_missing_splits=sample_missing_splits)


class CSVClassificationDataset(FlairDataset):
    """Dataset for text classification from CSV column formatted data."""

    def __init__(
        self,
        path_to_file: Union[str, Path],
        column_name_map: dict[int, str],
        label_type: str,
        max_tokens_per_doc: int = -1,
        max_chars_per_doc: int = -1,
        tokenizer: Tokenizer = SegtokTokenizer(),
        in_memory: bool = True,
        skip_header: bool = False,
        encoding: str = "utf-8",
        no_class_label=None,
        **fmtparams,
    ) -> None:
        """Instantiates a Dataset for text classification from CSV column formatted data.

        :param path_to_file: path to the file with the CSV data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param label_type: name of the label
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param tokenizer: Tokenizer for dataset, default is SegTokTokenizer
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param skip_header: If True, skips first line because it is header
        :param encoding: Most datasets are 'utf-8' but some are 'latin-1'
        :param fmtparams: additional parameters for the CSV file reader
        :return: a Corpus with annotated train, dev and test data
        """
        path_to_file = Path(path_to_file)

        assert path_to_file.exists()

        # variables
        self.path_to_file = path_to_file
        self.in_memory = in_memory
        self.tokenizer = tokenizer
        self.column_name_map = column_name_map
        self.max_tokens_per_doc = max_tokens_per_doc
        self.max_chars_per_doc = max_chars_per_doc
        self.no_class_label = no_class_label

        self.label_type = label_type

        # different handling of in_memory data than streaming data
        if self.in_memory:
            self.sentences = []
        else:
            self.raw_data = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_columns: list[int] = []
        self.pair_columns: list[int] = []
        for column in column_name_map:
            if column_name_map[column] == "text":
                self.text_columns.append(column)
            if column_name_map[column] == "pair":
                self.pair_columns.append(column)

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
                    sentence = self._make_labeled_data_point(row)

                    self.sentences.append(sentence)

                else:
                    self.raw_data.append(row)

                self.total_sentence_count += 1

    def _make_labeled_data_point(self, row):
        # make sentence from text (and filter for length)
        text = " ".join([row[text_column] for text_column in self.text_columns])

        if self.max_chars_per_doc > 0:
            text = text[: self.max_chars_per_doc]

        sentence = Sentence(text, use_tokenizer=self.tokenizer)

        if 0 < self.max_tokens_per_doc < len(sentence):
            sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]

        # if a pair column is defined, make a sentence pair object
        if len(self.pair_columns) > 0:
            text = " ".join([row[pair_column] for pair_column in self.pair_columns])

            if self.max_chars_per_doc > 0:
                text = text[: self.max_chars_per_doc]

            pair = Sentence(text, use_tokenizer=self.tokenizer)

            if 0 < self.max_tokens_per_doc < len(sentence):
                pair.tokens = pair.tokens[: self.max_tokens_per_doc]

            data_point = DataPair(first=sentence, second=pair)

        else:
            data_point = sentence

        for column in self.column_name_map:
            column_value = row[column]
            if (
                self.column_name_map[column].startswith("label")
                and column_value
                and column_value != self.no_class_label
            ):
                data_point.add_label(self.label_type, column_value)

        return data_point

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self) -> int:
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            return self.sentences[index]
        else:
            row = self.raw_data[index]

            sentence = self._make_labeled_data_point(row)

            return sentence


class AMAZON_REVIEWS(ClassificationCorpus):
    """A very large corpus of Amazon reviews with positivity ratings.

    Corpus is downloaded from and documented at
    https://nijianmo.github.io/amazon/index.html.
    We download the 5-core subset which is still tens of millions of
    reviews.
    """

    # noinspection PyDefaultArgument
    def __init__(
        self,
        split_max: int = 30000,
        label_name_map: dict[str, str] = {
            "1.0": "NEGATIVE",
            "2.0": "NEGATIVE",
            "3.0": "NEGATIVE",
            "4.0": "POSITIVE",
            "5.0": "POSITIVE",
        },
        skip_labels=["3.0", "4.0"],
        fraction_of_5_star_reviews: int = 10,
        tokenizer: Tokenizer = SegtokTokenizer(),
        memory_mode="partial",
        **corpusargs,
    ) -> None:
        """Constructs corpus object.

        Split_max indicates how many data points from each of the 28 splits are used, so
        set this higher or lower to increase/decrease corpus size.
        :param label_name_map: Map label names to different schema. By default, the 5-star rating is mapped onto 3
        classes (POSITIVE, NEGATIVE, NEUTRAL)
        :param split_max: Split_max indicates how many data points from each of the 28 splits are used, so
        set this higher or lower to increase/decrease corpus size.
        :param memory_mode: Set to what degree to keep corpus in memory ('full', 'partial' or 'disk'). Use 'full'
        if full corpus and all embeddings fits into memory for speedups during training. Otherwise use 'partial' and if
        even this is too much for your memory, use 'disk'.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param corpusargs: Arguments for ClassificationCorpus
        """
        # dataset name includes the split size
        dataset_name = self.__class__.__name__.lower() + "_" + str(split_max) + "_" + str(fraction_of_5_star_reviews)

        # default dataset folder is the cache root
        data_folder = flair.cache_root / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():
            # download each of the 28 splits
            self.download_and_prepare_amazon_product_file(
                data_folder, "AMAZON_FASHION_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "All_Beauty_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Appliances_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Arts_Crafts_and_Sewing_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Arts_Crafts_and_Sewing_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Automotive_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Books_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "CDs_and_Vinyl_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Cell_Phones_and_Accessories_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Clothing_Shoes_and_Jewelry_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Digital_Music_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Electronics_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Gift_Cards_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Grocery_and_Gourmet_Food_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Home_and_Kitchen_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Industrial_and_Scientific_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Kindle_Store_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Luxury_Beauty_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Magazine_Subscriptions_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Movies_and_TV_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Musical_Instruments_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Office_Products_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Patio_Lawn_and_Garden_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Pet_Supplies_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Prime_Pantry_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Software_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Sports_and_Outdoors_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Tools_and_Home_Improvement_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Toys_and_Games_5.json.gz", split_max, fraction_of_5_star_reviews
            )
            self.download_and_prepare_amazon_product_file(
                data_folder, "Video_Games_5.json.gz", split_max, fraction_of_5_star_reviews
            )

        super().__init__(
            data_folder,
            label_type="sentiment",
            label_name_map=label_name_map,
            skip_labels=skip_labels,
            tokenizer=tokenizer,
            memory_mode=memory_mode,
            **corpusargs,
        )

    def download_and_prepare_amazon_product_file(
        self, data_folder, part_name, max_data_points=None, fraction_of_5_star_reviews=None
    ):
        amazon__path = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall"
        cached_path(f"{amazon__path}/{part_name}", Path("datasets") / "Amazon_Product_Reviews")
        import gzip

        # create dataset directory if necessary
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        with open(data_folder / "train.txt", "a") as train_file:
            write_count = 0
            review_5_count = 0
            # download senteval datasets if necessary und unzip
            with gzip.open(flair.cache_root / "datasets" / "Amazon_Product_Reviews" / part_name, "rb") as f_in:
                for line in f_in:
                    parsed_json = json.loads(line)
                    if "reviewText" not in parsed_json:
                        continue
                    if parsed_json["reviewText"].strip() == "":
                        continue
                    text = parsed_json["reviewText"].replace("\n", "")

                    if fraction_of_5_star_reviews and str(parsed_json["overall"]) == "5.0":
                        review_5_count += 1
                        if review_5_count != fraction_of_5_star_reviews:
                            continue
                        else:
                            review_5_count = 0

                    train_file.write(f"__label__{parsed_json['overall']} {text}\n")

                    write_count += 1
                    if max_data_points and write_count >= max_data_points:
                        break


class IMDB(ClassificationCorpus):
    """Corpus of IMDB movie reviews labeled by sentiment (POSITIVE, NEGATIVE).

    Downloaded from and documented at http://ai.stanford.edu/~amaas/data/sentiment/.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        rebalance_corpus: bool = True,
        tokenizer: Tokenizer = SegtokTokenizer(),
        memory_mode="partial",
        **corpusargs,
    ) -> None:
        """Initialize the IMDB move review sentiment corpus.

        Args:
            base_path: Provide this only if you store the IMDB corpus in a specific folder, otherwise use default.
            tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
            rebalance_corpus: Weather to use a 80/10/10 data split instead of the original 50/0/50 split.
            memory_mode: Set to 'partial' because this is a huge corpus, but you can also set to 'full' for faster
         processing or 'none' for less memory.
            corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower() + "_v4"

        # download data if necessary
        imdb_acl_path = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        if rebalance_corpus:
            dataset_name = dataset_name + "-rebalanced"
        data_folder = base_path / dataset_name
        data_path = flair.cache_root / "datasets" / dataset_name
        train_data_file = data_path / "train.txt"
        test_data_file = data_path / "test.txt"

        if not train_data_file.is_file() or (not rebalance_corpus and not test_data_file.is_file()):
            for file_path in [train_data_file, test_data_file]:
                if file_path.is_file():
                    os.remove(file_path)

            cached_path(imdb_acl_path, Path("datasets") / dataset_name)
            import tarfile

            with tarfile.open(flair.cache_root / "datasets" / dataset_name / "aclImdb_v1.tar.gz", "r:gz") as f_in:
                datasets = ["train", "test"]
                labels = ["pos", "neg"]

                for label in labels:
                    for dataset in datasets:
                        f_in.extractall(
                            data_path, members=[m for m in f_in.getmembers() if f"{dataset}/{label}" in m.name]
                        )
                        data_file = train_data_file
                        if not rebalance_corpus and dataset == "test":
                            data_file = test_data_file

                        with open(data_file, "at") as f_p:
                            current_path = data_path / "aclImdb" / dataset / label
                            for file_name in current_path.iterdir():
                                if file_name.is_file() and file_name.name.endswith(".txt"):
                                    if label == "pos":
                                        sentiment_label = "POSITIVE"
                                    if label == "neg":
                                        sentiment_label = "NEGATIVE"
                                    f_p.write(
                                        f"__label__{sentiment_label} "
                                        + file_name.open("rt", encoding="utf-8").read()
                                        + "\n"
                                    )

        super().__init__(
            data_folder, label_type="sentiment", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs
        )


class NEWSGROUPS(ClassificationCorpus):
    """20 newsgroups corpus, classifying news items into one of 20 categories.

    Downloaded from http://qwone.com/~jason/20Newsgroups


     Each data point is a full news article so documents may be very
     long.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Tokenizer = SegtokTokenizer(),
        memory_mode: str = "partial",
        **corpusargs,
    ) -> None:
        """Instantiates 20 newsgroups corpus.

        :param base_path: Provide this only if you store the IMDB corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param memory_mode: Set to 'partial' because this is a big corpus, but you can also set to 'full' for faster
         processing or 'none' for less memory.
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        twenty_newsgroups_path = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
        data_path = flair.cache_root / "datasets" / dataset_name
        data_file = data_path / "20news-bydate-train.txt"
        if not data_file.is_file():
            cached_path(twenty_newsgroups_path, Path("datasets") / dataset_name / "original")

            import tarfile

            with tarfile.open(
                flair.cache_root / "datasets" / dataset_name / "original" / "20news-bydate.tar.gz", "r:gz"
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
                            members=[m for m in f_in.getmembers() if f"{dataset}/{label}" in m.name],
                        )
                        with open(f"{data_path}/{dataset}.txt", "at", encoding="utf-8") as f_p:
                            current_path = data_path / "original" / dataset / label
                            for file_name in current_path.iterdir():
                                if file_name.is_file():
                                    f_p.write(
                                        f"__label__{label} "
                                        + file_name.open("rt", encoding="latin1").read().replace("\n", " <n> ")
                                        + "\n"
                                    )

        super().__init__(data_folder, tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs)


class AGNEWS(ClassificationCorpus):
    """The AG's News Topic Classification Corpus, classifying news into 4 coarse-grained topics.

    Labels: World, Sports, Business, Sci/Tech.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode="partial",
        **corpusargs,
    ):
        """Instantiates AGNews Classification Corpus with 4 classes.

        :param base_path: Provide this only if you store the AGNEWS corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param memory_mode: Set to 'partial' by default. Can also be 'full' or 'none'.
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data from same source as in huggingface's implementations
        agnews_path = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"

        original_filenames = ["train.csv", "test.csv", "classes.txt"]
        new_filenames = ["train.txt", "test.txt"]

        for original_filename in original_filenames:
            cached_path(f"{agnews_path}{original_filename}", Path("datasets") / dataset_name / "original")

        data_file = data_folder / new_filenames[0]
        label_dict = []
        label_path = original_filenames[-1]

        # read label order
        with open(data_folder / "original" / label_path) as f:
            for line in f:
                line = line.rstrip()
                label_dict.append(line)

        original_filenames = original_filenames[:-1]
        if not data_file.is_file():
            for original_filename, new_filename in zip(original_filenames, new_filenames):
                with (
                    open(data_folder / "original" / original_filename, encoding="utf-8") as open_fp,
                    open(data_folder / new_filename, "w", encoding="utf-8") as write_fp,
                ):
                    csv_reader = csv.reader(
                        open_fp, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
                    )
                    for id_, row in enumerate(csv_reader):
                        label, title, description = row
                        # Original labels are [1, 2, 3, 4] -> ['World', 'Sports', 'Business', 'Sci/Tech']
                        # Re-map to [0, 1, 2, 3].
                        text = " ".join((title, description))

                        new_label = "__label__"
                        new_label += label_dict[int(label) - 1]

                        write_fp.write(f"{new_label} {text}\n")

        super().__init__(data_folder, label_type="topic", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs)


class STACKOVERFLOW(ClassificationCorpus):
    """Stackoverflow corpus classifying questions into one of 20 labels.

    The data will be downloaded from "https://github.com/jacoxu/StackOverflow",

    Each data point is a question.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Tokenizer = SegtokTokenizer(),
        memory_mode: str = "partial",
        **corpusargs,
    ) -> None:
        """Instantiates Stackoverflow corpus.

        :param base_path: Provide this only if you store the IMDB corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param memory_mode: Set to 'partial' because this is a big corpus, but you can also set to 'full' for faster
         processing or 'none' for less memory.
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        stackoverflow_path_data = (
            "https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/title_StackOverflow.txt"
        )
        stackoverflow_path_label = (
            "https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/label_StackOverflow.txt"
        )
        data_path = flair.cache_root / "datasets" / dataset_name
        data_file = data_path / "title_StackOverflow.txt"
        if not data_file.is_file():
            cached_path(stackoverflow_path_data, Path("datasets") / dataset_name / "original")
            cached_path(stackoverflow_path_label, Path("datasets") / dataset_name / "original")

            label_list = []
            labels = [
                "wordpress",
                "oracle",
                "svn",
                "apache",
                "excel",
                "matlab",
                "visual-studio",
                "cocoa",
                "osx",
                "bash",
                "spring",
                "hibernate",
                "scala",
                "sharepoint",
                "ajax",
                "qt",
                "drupal",
                "linq",
                "haskell",
                "magento",
            ]
            # handle labels file
            with open(data_path / "original" / "label_StackOverflow.txt", encoding="latin1") as open_fp:
                for line in open_fp:
                    line = line.rstrip()
                    label_list.append(labels[int(line) - 1])

            # handle data file
            with (
                (data_path / "original" / "title_StackOverflow.txt").open(encoding="latin1") as open_fp,
                (data_folder / "train.txt").open("w", encoding="utf-8") as write_fp,
            ):
                for idx, line in enumerate(open_fp):
                    line = line.rstrip()

                    # Create flair compatible labels
                    label = label_list[idx]
                    write_fp.write(f"__label__{label} {line}\n")

        super().__init__(data_folder, label_type="class", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs)


class SENTIMENT_140(ClassificationCorpus):
    """Twitter sentiment corpus.

    See http://help.sentiment140.com/for-students

    Two sentiments in train data (POSITIVE, NEGATIVE) and three
    sentiments in test data (POSITIVE, NEGATIVE, NEUTRAL).
    """

    def __init__(
        self, label_name_map=None, tokenizer: Tokenizer = SegtokTokenizer(), memory_mode: str = "partial", **corpusargs
    ) -> None:
        """Instantiates twitter sentiment corpus.

        :param label_name_map: By default, the numeric values are mapped to ('NEGATIVE', 'POSITIVE' and 'NEUTRAL')
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param memory_mode: Set to 'partial' because this is a big corpus, but you can also set to 'full' for faster
         processing or 'none' for less memory.
        :param corpusargs: Other args for ClassificationCorpus.
        """
        # by default, map point score to POSITIVE / NEGATIVE values
        if label_name_map is None:
            label_name_map = {"0": "NEGATIVE", "2": "NEUTRAL", "4": "POSITIVE"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = flair.cache_root / "datasets" / dataset_name

        # download data if necessary
        if True:
            # download senteval datasets if necessary und unzip
            sentiment_url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
            cached_path(sentiment_url, Path("datasets") / dataset_name / "raw")
            senteval_folder = flair.cache_root / "datasets" / dataset_name / "raw"
            unzip_file(senteval_folder / "trainingandtestdata.zip", senteval_folder)

            # create dataset directory if necessary
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            # create train.txt file from CSV
            with (
                open(data_folder / "train.txt", "w") as train_file,
                open(senteval_folder / "training.1600000.processed.noemoticon.csv", encoding="latin-1") as csv_train,
            ):
                csv_reader = csv.reader(csv_train)

                for row in csv_reader:
                    label = row[0]
                    text = row[5]
                    train_file.write(f"__label__{label} {text}\n")

            # create test.txt file from CSV
            with (
                (data_folder / "test.txt").open("w", encoding="utf-8") as train_file,
                (senteval_folder / "testdata.manual.2009.06.14.csv").open(encoding="latin-1") as csv_train,
            ):
                csv_reader = csv.reader(csv_train)

                for row in csv_reader:
                    label = row[0]
                    text = row[5]
                    train_file.write(f"__label__{label} {text}\n")

        super().__init__(
            data_folder,
            label_type="sentiment",
            tokenizer=tokenizer,
            memory_mode=memory_mode,
            label_name_map=label_name_map,
            **corpusargs,
        )


class SENTEVAL_CR(ClassificationCorpus):
    """The customer reviews dataset of SentEval, classified into NEGATIVE or POSITIVE sentiment.

    see https://github.com/facebookresearch/SentEval
    """

    def __init__(
        self,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode: str = "full",
        **corpusargs,
    ) -> None:
        """Instantiates SentEval customer reviews dataset.

        :param corpusargs: Other args for ClassificationCorpus.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer())
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        """
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = flair.cache_root / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():
            # download senteval datasets if necessary und unzip
            senteval_path = "https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip"
            cached_path(senteval_path, Path("datasets") / "senteval")
            senteval_folder = flair.cache_root / "datasets" / "senteval"
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

        super().__init__(
            data_folder, label_type="sentiment", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs
        )


class SENTEVAL_MR(ClassificationCorpus):
    """The movie reviews dataset of SentEval, classified into NEGATIVE or POSITIVE sentiment.

    see https://github.com/facebookresearch/SentEval
    """

    def __init__(
        self,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode: str = "full",
        **corpusargs,
    ) -> None:
        """Instantiates SentEval movie reviews dataset.

        :param corpusargs: Other args for ClassificationCorpus.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        """
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = flair.cache_root / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():
            # download senteval datasets if necessary und unzip
            senteval_path = "https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip"
            cached_path(senteval_path, Path("datasets") / "senteval")
            senteval_folder = flair.cache_root / "datasets" / "senteval"
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

        super().__init__(
            data_folder, label_type="sentiment", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs
        )


class SENTEVAL_SUBJ(ClassificationCorpus):
    """The subjectivity dataset of SentEval, classified into SUBJECTIVE or OBJECTIVE sentiment.

    see https://github.com/facebookresearch/SentEval
    """

    def __init__(
        self,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode: str = "full",
        **corpusargs,
    ) -> None:
        """Instantiates SentEval subjectivity dataset.

        :param corpusargs: Other args for ClassificationCorpus.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        """
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = flair.cache_root / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():
            # download senteval datasets if necessary und unzip
            senteval_path = "https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip"
            cached_path(senteval_path, Path("datasets") / "senteval")
            senteval_folder = flair.cache_root / "datasets" / "senteval"
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

        super().__init__(
            data_folder, label_type="objectivity", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs
        )


class SENTEVAL_MPQA(ClassificationCorpus):
    """The opinion-polarity dataset of SentEval, classified into NEGATIVE or POSITIVE polarity.

    see https://github.com/facebookresearch/SentEval
    """

    def __init__(
        self,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode: str = "full",
        **corpusargs,
    ) -> None:
        """Instantiates SentEval opinion polarity dataset.

        :param corpusargs: Other args for ClassificationCorpus.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        """
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = flair.cache_root / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():
            # download senteval datasets if necessary und unzip
            senteval_path = "https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip"
            cached_path(senteval_path, Path("datasets") / "senteval")
            senteval_folder = flair.cache_root / "datasets" / "senteval"
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

        super().__init__(
            data_folder, label_type="sentiment", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs
        )


class SENTEVAL_SST_BINARY(ClassificationCorpus):
    """The Stanford sentiment treebank dataset of SentEval, classified into NEGATIVE or POSITIVE sentiment.

    see https://github.com/facebookresearch/SentEval
    """

    def __init__(
        self,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode: str = "full",
        **corpusargs,
    ) -> None:
        """Instantiates SentEval Stanford sentiment treebank dataset.

        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param corpusargs: Other args for ClassificationCorpus.
        """
        # this dataset name
        dataset_name = self.__class__.__name__.lower() + "_v2"

        # default dataset folder is the cache root
        data_folder = flair.cache_root / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():
            # download senteval datasets if necessary und unzip
            cached_path(
                "https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-train",
                Path("datasets") / dataset_name / "raw",
            )
            cached_path(
                "https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-test",
                Path("datasets") / dataset_name / "raw",
            )
            cached_path(
                "https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-dev",
                Path("datasets") / dataset_name / "raw",
            )

            original_filenames = ["sentiment-train", "sentiment-dev", "sentiment-test"]
            new_filenames = ["train.txt", "dev.txt", "test.txt"]

            # create train dev and test files in fasttext format
            for new_filename, original_filename in zip(new_filenames, original_filenames):
                with (
                    open(data_folder / new_filename, "a") as out_file,
                    open(data_folder / "raw" / original_filename) as in_file,
                ):
                    for line in in_file:
                        fields = line.split("\t")
                        label = "POSITIVE" if fields[1].rstrip() == "1" else "NEGATIVE"
                        out_file.write(f"__label__{label} {fields[0]}\n")

        super().__init__(data_folder, tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs)


class SENTEVAL_SST_GRANULAR(ClassificationCorpus):
    """The Stanford sentiment treebank dataset of SentEval, classified into 5 sentiment classes.

    see https://github.com/facebookresearch/SentEval
    """

    def __init__(
        self,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode: str = "full",
        **corpusargs,
    ) -> None:
        """Instantiates SentEval Stanford sentiment treebank dataset.

        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param corpusargs: Other args for ClassificationCorpus.
        """
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = flair.cache_root / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():
            # download senteval datasets if necessary und unzip
            cached_path(
                "https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/stsa.fine.train",
                Path("datasets") / dataset_name / "raw",
            )
            cached_path(
                "https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/stsa.fine.test",
                Path("datasets") / dataset_name / "raw",
            )
            cached_path(
                "https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/stsa.fine.dev",
                Path("datasets") / dataset_name / "raw",
            )

            # convert to FastText format
            for split in ["train", "dev", "test"]:
                with (
                    (data_folder / f"{split}.txt").open("w", encoding="utf-8") as train_file,
                    (data_folder / "raw" / f"stsa.fine.{split}").open(encoding="latin1") as file,
                ):
                    for line in file:
                        train_file.write(f"__label__{line[0]} {line[2:]}")

        super().__init__(data_folder, tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs)


class GLUE_COLA(ClassificationCorpus):
    """Corpus of Linguistic Acceptability from GLUE benchmark.

    see https://gluebenchmark.com/tasks

    The task is to predict whether an English sentence is grammatically
    correct. Additionaly to the Corpus we have eval_dataset containing
    the unlabeled test data for Glue evaluation.
    """

    def __init__(
        self,
        label_type="acceptability",
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Tokenizer = SegtokTokenizer(),
        **corpusargs,
    ) -> None:
        """Instantiates CoLA dataset.

        :param base_path: Provide this only if you store the COLA corpus in a specific folder.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        dataset_name = "glue"

        data_folder = base_path / dataset_name

        # download data if necessary
        cola_path = "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip"

        data_file = data_folder / "CoLA/train.txt"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            # get the zip file
            zipped_data_path = cached_path(cola_path, Path("datasets") / dataset_name)

            unpack_file(zipped_data_path, data_folder, mode="zip", keep=False)

            # move original .tsv files to another folder
            Path(data_folder / "CoLA/train.tsv").rename(data_folder / "CoLA/original/train.tsv")
            Path(data_folder / "CoLA/dev.tsv").rename(data_folder / "CoLA/original/dev.tsv")
            Path(data_folder / "CoLA/test.tsv").rename(data_folder / "CoLA/original/test.tsv")

            label_map = {0: "not_grammatical", 1: "grammatical"}

            # create train and dev splits in fasttext format
            for split in ["train", "dev"]:
                with (
                    open(data_folder / "CoLA" / (split + ".txt"), "a") as out_file,
                    open(data_folder / "CoLA" / "original" / (split + ".tsv")) as in_file,
                ):
                    for line in in_file:
                        fields = line.rstrip().split("\t")
                        label = int(fields[1])
                        sentence = fields[3]
                        out_file.write(f"__label__{label_map[label]} {sentence}\n")

            # create eval_dataset file with no labels
            with (
                open(data_folder / "CoLA" / "eval_dataset.txt", "a") as out_file,
                open(data_folder / "CoLA" / "original" / "test.tsv") as in_file,
            ):
                for line in in_file:
                    fields = line.rstrip().split("\t")
                    sentence = fields[1]
                    out_file.write(f"{sentence}\n")

        super().__init__(data_folder / "CoLA", label_type=label_type, tokenizer=tokenizer, **corpusargs)

        self.eval_dataset = ClassificationDataset(
            data_folder / "CoLA/eval_dataset.txt",
            label_type=label_type,
            allow_examples_without_labels=True,
            tokenizer=tokenizer,
            memory_mode="full",
        )

    def tsv_from_eval_dataset(self, folder_path: Union[str, Path]):
        """Create eval prediction file.

        This function creates a tsv file with predictions of the eval_dataset (after calling
        classifier.predict(corpus.eval_dataset, label_name='acceptability')). The resulting file
        is called CoLA.tsv and is in the format required for submission to the Glue Benchmark.
        """
        folder_path = Path(folder_path)
        folder_path = folder_path / "CoLA.tsv"

        with open(folder_path, mode="w") as tsv_file:
            tsv_file.write("index\tprediction\n")
            for index, datapoint in enumerate(_iter_dataset(self.eval_dataset)):
                reverse_label_map = {"grammatical": 1, "not_grammatical": 0}
                predicted_label = reverse_label_map[datapoint.get_labels("acceptability")[0].value]
                tsv_file.write(str(index) + "\t" + str(predicted_label) + "\n")


class GLUE_SST2(CSVClassificationCorpus):
    label_map = {0: "negative", 1: "positive"}

    def __init__(
        self,
        label_type: str = "sentiment",
        base_path: Optional[Union[str, Path]] = None,
        max_tokens_per_doc=-1,
        max_chars_per_doc=-1,
        tokenizer: Tokenizer = SegtokTokenizer(),
        in_memory: bool = False,
        encoding: str = "utf-8",
        sample_missing_splits: bool = True,
        **datasetargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        dataset_name = "SST-2"

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.tsv"

        sst2_url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

        if not train_file.is_file():
            # download zip archive
            zipped_data_path = cached_path(sst2_url, data_folder)

            # unpack file in datasets directory (zip archive contains a directory named SST-2)
            unpack_file(zipped_data_path, data_folder.parent, "zip", False)

        kwargs = dict(
            delimiter="\t",
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            tokenizer=tokenizer,
            in_memory=in_memory,
            encoding=encoding,
            skip_header=True,
            **datasetargs,
        )

        super().__init__(
            name=dataset_name,
            data_folder=data_folder,
            label_type=label_type,
            column_name_map={0: "text", 1: "label"},
            train_file=train_file,
            dev_file=data_folder / "dev.tsv",
            sample_missing_splits=sample_missing_splits,
            **kwargs,
        )

        eval_file = data_folder / "test.tsv"

        log.info("Evaluation (no labels): %s", eval_file)
        self.eval_dataset = CSVClassificationDataset(
            eval_file,
            label_type="sentence_index",
            column_name_map={
                0: "label_index",
                1: "text",
            },
            **kwargs,
        )

    def tsv_from_eval_dataset(self, folder_path: Union[str, Path]):
        """Create eval prediction file."""
        folder_path = Path(folder_path)
        folder_path = folder_path / "SST-2.tsv"

        reverse_label_map = {label_name: label_numerical for label_numerical, label_name in self.label_map.items()}

        with open(folder_path, mode="w") as tsv_file:
            tsv_file.write("index\tprediction\n")
            for index, datapoint in enumerate(_iter_dataset(self.eval_dataset)):
                predicted_label = reverse_label_map[datapoint.get_labels(self.eval_dataset.label_type)[0].value]
                tsv_file.write(f"{index}\t{predicted_label}\n")


class GO_EMOTIONS(ClassificationCorpus):
    """GoEmotions dataset containing 58k Reddit comments labeled with 27 emotion categories.

    see https://github.com/google-research/google-research/tree/master/goemotions
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Union[bool, Tokenizer] = SegtokTokenizer(),
        memory_mode: str = "partial",
        **corpusargs,
    ) -> None:
        """Initializes the GoEmotions corpus.

        Parameters
        ----------
        base_path: Union[str, Path]
            Provide this only if you want to store the corpus in a specific folder, otherwise use default.
        tokenizer: Union[bool, Tokenizer]
            Specify which tokenizer to use, the default is SegtokTokenizer().
        memory_mode: str
            Set to what degree to keep corpus in memory ('full', 'partial' or 'disk'). Use 'full'
            if full corpus and all embeddings fits into memory for speedups during training. Otherwise use 'partial' and if
            even this is too much for your memory, use 'disk'.
        """
        label_name_map = {
            "0": "ADMIRATION",
            "1": "AMUSEMENT",
            "2": "ANGER",
            "3": "ANNOYANCE",
            "4": "APPROVAL",
            "5": "CARING",
            "6": "CONFUSION",
            "7": "CURIOSITY",
            "8": "DESIRE",
            "9": "DISAPPOINTMENT",
            "10": "DISAPPROVAL",
            "11": "DISGUST",
            "12": "EMBARRASSMENT",
            "13": "EXCITEMENT",
            "14": "FEAR",
            "15": "GRATITUDE",
            "16": "GRIEF",
            "17": "JOY",
            "18": "LOVE",
            "19": "NERVOUSNESS",
            "20": "OPTIMISM",
            "21": "PRIDE",
            "22": "REALIZATION",
            "23": "RELIEF",
            "24": "REMORSE",
            "25": "SADNESS",
            "26": "SURPRISE",
            "27": "NEUTRAL",
        }

        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = base_path / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():
            # download datasets if necessary
            goemotions_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
            for name in ["train.tsv", "test.tsv", "dev.tsv"]:
                cached_path(goemotions_url + name, Path("datasets") / dataset_name / "raw")

            # create dataset directory if necessary
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            data_path = flair.cache_root / "datasets" / dataset_name / "raw"
            # create correctly formated txt files
            for name in ["train", "test", "dev"]:
                with (
                    (data_folder / (name + ".txt")).open("w", encoding="utf-8") as txt_file,
                    (data_path / (name + ".tsv")).open(encoding="utf-8") as tsv_file,
                ):
                    lines = tsv_file.readlines()
                    for line in lines:
                        row = line.split("\t")
                        text = row[0]
                        # multiple labels are possible
                        labels = row[1].split(",")
                        label_string = ""
                        for label in labels:
                            label_string += "__label__"
                            label_string += label
                            label_string += " "
                        txt_file.write(f"{label_string}{text}\n")

        super().__init__(
            data_folder,
            label_type="emotion",
            tokenizer=tokenizer,
            memory_mode=memory_mode,
            label_name_map=label_name_map,
            **corpusargs,
        )


class TREC_50(ClassificationCorpus):
    """The TREC Question Classification Corpus, classifying questions into 50 fine-grained answer types."""

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode="full",
        **corpusargs,
    ) -> None:
        """Instantiates TREC Question Classification Corpus with 6 classes.

        :param base_path: Provide this only if you store the TREC corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        trec_path = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"

        original_filenames = ["train_5500.label", "TREC_10.label"]
        new_filenames = ["train.txt", "test.txt"]
        for original_filename in original_filenames:
            cached_path(f"{trec_path}{original_filename}", Path("datasets") / dataset_name / "original")

        data_file = data_folder / new_filenames[0]

        if not data_file.is_file():
            for original_filename, new_filename in zip(original_filenames, new_filenames):
                with (
                    (data_folder / "original" / original_filename).open(encoding="latin1") as open_fp,
                    (data_folder / new_filename).open("w", encoding="utf-8") as write_fp,
                ):
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

        super().__init__(data_folder, tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs)


class TREC_6(ClassificationCorpus):
    """The TREC Question Classification Corpus, classifying questions into 6 coarse-grained answer types."""

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode="full",
        **corpusargs,
    ) -> None:
        """Instantiates TREC Question Classification Corpus with 6 classes.

        :param base_path: Provide this only if you store the TREC corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        trec_path = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"

        original_filenames = ["train_5500.label", "TREC_10.label"]
        new_filenames = ["train.txt", "test.txt"]
        for original_filename in original_filenames:
            cached_path(f"{trec_path}{original_filename}", Path("datasets") / dataset_name / "original")

        data_file = data_folder / new_filenames[0]

        if not data_file.is_file():
            for original_filename, new_filename in zip(original_filenames, new_filenames):
                with (
                    (data_folder / "original" / original_filename).open(encoding="latin1") as open_fp,
                    (data_folder / new_filename).open("w", encoding="utf-8") as write_fp,
                ):
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

        super().__init__(
            data_folder, label_type="question_class", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs
        )


class YAHOO_ANSWERS(ClassificationCorpus):
    """The YAHOO Question Classification Corpus, classifying questions into 10 coarse-grained answer types."""

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        memory_mode="partial",
        **corpusargs,
    ) -> None:
        """Instantiates YAHOO Question Classification Corpus with 10 classes.

        :param base_path: Provide this only if you store the YAHOO corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param memory_mode: Set to 'partial' by default since this is a rather big corpus. Can also be 'full' or 'none'.
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        url = "https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz"

        label_map = {
            "1": "Society_&_Culture",
            "2": "Science_&_Mathematics",
            "3": "Health",
            "4": "Education_&_Reference",
            "5": "Computers_&_Internet",
            "6": "Sports",
            "7": "Business_&_Finance",
            "8": "Entertainment_&_Music",
            "9": "Family_&_Relationships",
            "10": "Politics_&_Government",
        }

        original = flair.cache_root / "datasets" / dataset_name / "original"

        if not (data_folder / "train.txt").is_file():
            cached_path(url, original)

            with tarfile.open(original / "yahoo_answers_csv.tgz", "r:gz") as tar:
                members = []

                for member in tar.getmembers():
                    if "test.csv" in member.name or "train.csv" in member.name:
                        members.append(member)

                tar.extractall(original, members=members)

            for name in ["train", "test"]:
                with (
                    (original / "yahoo_answers_csv" / (name + ".csv")).open(encoding="utf-8") as file,
                    (data_folder / (name + ".txt")).open("w", encoding="utf-8") as writer,
                ):
                    reader = csv.reader(file)
                    for row in reader:
                        writer.write("__label__" + label_map[row[0]] + " " + row[1] + "\n")

        super().__init__(
            data_folder, label_type="question_type", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs
        )


class GERMEVAL_2018_OFFENSIVE_LANGUAGE(ClassificationCorpus):
    """GermEval 2018 corpus for identification of offensive language.

    Classifying German tweets into 2 coarse-grained categories OFFENSIVE
    and OTHER or 4 fine-grained categories ABUSE, INSULT, PROFATINTY and
    OTHER.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        tokenizer: Union[bool, Tokenizer] = SegtokTokenizer(),
        memory_mode: str = "full",
        fine_grained_classes: bool = False,
        **corpusargs,
    ) -> None:
        """Instantiates GermEval 2018 Offensive Language Classification Corpus.

        :param base_path: Provide this only if you store the Offensive Language corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        :param fine_grained_classes: Set to True to load the dataset with 4 fine-grained classes
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        offlang_path = "https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/"

        original_filenames = ["germeval2018.training.txt", "germeval2018.test.txt"]
        new_filenames = ["train.txt", "test.txt"]
        for original_filename in original_filenames:
            cached_path(f"{offlang_path}{original_filename}", Path("datasets") / dataset_name / "original")

        task_setting = "coarse_grained"
        if fine_grained_classes:
            task_setting = "fine_grained"

        task_folder = data_folder / task_setting
        data_file = task_folder / new_filenames[0]

        # create a separate directory for different tasks
        if not os.path.exists(task_folder):
            os.makedirs(task_folder)

        if not data_file.is_file():
            for original_filename, new_filename in zip(original_filenames, new_filenames):
                with (
                    (data_folder / "original" / original_filename).open(encoding="utf-8") as open_fp,
                    (data_folder / task_setting / new_filename).open("w", encoding="utf-8") as write_fp,
                ):
                    for line in open_fp:
                        line = line.rstrip()
                        fields = line.split("\t")
                        tweet = fields[0]
                        old_label = fields[2] if task_setting == "fine_grained" else fields[1]
                        new_label = "__label__" + old_label
                        write_fp.write(f"{new_label} {tweet}\n")

        super().__init__(data_folder=task_folder, tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs)


class COMMUNICATIVE_FUNCTIONS(ClassificationCorpus):
    """The Communicative Functions Classification Corpus.

    Classifying sentences from scientific papers into 39 communicative functions.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        memory_mode: str = "full",
        tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
        **corpusargs,
    ) -> None:
        """Instantiates Communicative Functions Classification Corpus with 39 classes.

        :param base_path: Provide this only if you store the Communicative Functions date in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SpaceTokenizer)
        :param memory_mode: Set to 'full' by default since this is a small corpus. Can also be 'partial' or 'none'.
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        original_filenames = ["background.tsv", "discussion.tsv", "introduction.tsv", "method.tsv", "result.tsv"]

        # download data if necessary
        comm_path = "https://raw.githubusercontent.com/Alab-NII/FECFevalDataset/master/sentences/"

        for original_filename in original_filenames:
            cached_path(f"{comm_path}{original_filename}", Path("datasets") / dataset_name / "original")

        data_file = data_folder / "train.txt"

        if not data_file.is_file():  # check if new file already exists
            with open(data_folder / "train.txt", "a+", encoding="utf-8") as write_fp:
                for original_filename in original_filenames[:4]:
                    with open(data_folder / "original" / original_filename, encoding="utf-8") as open_fp:
                        for line in open_fp:
                            liste = line.split("\t")
                            write_fp.write("__label__" + liste[0].replace(" ", "_") + " " + liste[2] + "\n")
                    with open(data_folder / "original" / "result.tsv", encoding="utf-8") as open_fp:
                        for line in open_fp:
                            liste = line.split("\t")
                            if liste[0].split(" ")[-1] == "(again)":
                                write_fp.write("__label__" + liste[0][:-8].replace(" ", "_") + " " + liste[2] + "\n")
                            else:
                                write_fp.write("__label__" + liste[0].replace(" ", "_") + " " + liste[2] + "\n")

        super().__init__(
            data_folder, label_type="communicative_function", tokenizer=tokenizer, memory_mode=memory_mode, **corpusargs
        )


def _download_wassa_if_not_there(emotion, data_folder, dataset_name):
    for split in ["train", "dev", "test"]:
        data_file = data_folder / f"{emotion}-{split}.txt"

        if not data_file.is_file():
            if split == "train":
                url = f"http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/{emotion}-ratings-0to1.train.txt"
            if split == "dev":
                url = f"http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/{emotion}-ratings-0to1.dev.gold.txt"
            if split == "test":
                url = (
                    f"http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/{emotion}-ratings-0to1.test.gold.txt"
                )

            path = cached_path(url, Path("datasets") / dataset_name)

            with open(path, encoding="UTF-8") as f, open(data_file, "w", encoding="UTF-8") as out:
                next(f)
                for line in f:
                    fields = line.split("\t")
                    out.write(f"__label__{fields[3].rstrip()} {fields[1]}\n")

            os.remove(path)


class WASSA_ANGER(ClassificationCorpus):
    """WASSA-2017 anger emotion-intensity corpus.

    see https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html.
    """

    def __init__(
        self, base_path: Optional[Union[str, Path]] = None, tokenizer: Tokenizer = SegtokTokenizer(), **corpusargs
    ) -> None:
        """Instantiates WASSA-2017 anger emotion-intensity corpus.

        :param base_path: Provide this only if you store the WASSA corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("anger", data_folder, dataset_name)

        super().__init__(data_folder, tokenizer=tokenizer, **corpusargs)


class WASSA_FEAR(ClassificationCorpus):
    """WASSA-2017 fear emotion-intensity corpus.

    see https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html.
    """

    def __init__(
        self, base_path: Optional[Union[str, Path]] = None, tokenizer: Tokenizer = SegtokTokenizer(), **corpusargs
    ) -> None:
        """Instantiates WASSA-2017 fear emotion-intensity corpus.

        :param base_path: Provide this only if you store the WASSA corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("fear", data_folder, dataset_name)

        super().__init__(data_folder, tokenizer=tokenizer, **corpusargs)


class WASSA_JOY(ClassificationCorpus):
    """WASSA-2017 joy emotion-intensity dataset corpus.

    see https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
    """

    def __init__(
        self, base_path: Optional[Union[str, Path]] = None, tokenizer: Tokenizer = SegtokTokenizer(), **corpusargs
    ) -> None:
        """Instantiates WASSA-2017 joy emotion-intensity corpus.

        :param base_path: Provide this only if you store the WASSA corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("joy", data_folder, dataset_name)

        super().__init__(data_folder, tokenizer=tokenizer, **corpusargs)


class WASSA_SADNESS(ClassificationCorpus):
    """WASSA-2017 sadness emotion-intensity corpus.

    see https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html.
    """

    def __init__(
        self, base_path: Optional[Union[str, Path]] = None, tokenizer: Tokenizer = SegtokTokenizer(), **corpusargs
    ) -> None:
        """Instantiates WASSA-2017 sadness emotion-intensity dataset.

        :param base_path: Provide this only if you store the WASSA corpus in a specific folder, otherwise use default.
        :param tokenizer: Custom tokenizer to use (default is SegtokTokenizer)
        :param corpusargs: Other args for ClassificationCorpus.
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        _download_wassa_if_not_there("sadness", data_folder, dataset_name)

        super().__init__(data_folder, tokenizer=tokenizer, **corpusargs)
