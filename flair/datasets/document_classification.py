import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Union, Callable


import flair
from flair.data import (
    Sentence,
    Corpus,
    Token,
    FlairDataset,
    space_tokenizer,
    segtok_tokenizer,
)
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path, unzip_file


class ClassificationCorpus(Corpus):
    """
    A classification corpus from FastText-formatted text files.
    """
    def __init__(
            self,
            data_folder: Union[str, Path],
            label_type: str = 'class',
            train_file=None,
            test_file=None,
            dev_file=None,
            truncate_to_max_tokens: int = -1,
            truncate_to_max_chars: int = -1,
            filter_if_longer_than: int = -1,
            tokenizer: Callable[[str], List[Token]] = segtok_tokenizer,
            memory_mode: str = "partial",
            label_name_map: Dict[str, str] = None,
            encoding: str = 'utf-8',
    ):
        """
        Instantiates a Corpus from text classification-formatted task data

        :param data_folder: base folder with the task data
        :param label_type: name of the label
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param truncate_to_max_tokens: If set, truncates each Sentence to a maximum number of tokens
        :param truncate_to_max_chars: If set, truncates each Sentence to a maximum number of chars
        :param filter_if_longer_than: If set, filters documents that are longer that the specified number of tokens.
        :param tokenizer: Tokenizer for dataset, default is segtok
        :param memory_mode: Set to what degree to keep corpus in memory ('full', 'partial' or 'disk'). Use 'full'
        if full corpus and all embeddings fits into memory for speedups during training. Otherwise use 'partial' and if
        even this is too much for your memory, use 'disk'.
        :param label_name_map: Optionally map label names to different schema.
        :param encoding: Default is 'uft-8' but some datasets are in 'latin-1
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        train: FlairDataset = ClassificationDataset(
            train_file,
            label_type=label_type,
            tokenizer=tokenizer,
            truncate_to_max_tokens=truncate_to_max_tokens,
            truncate_to_max_chars=truncate_to_max_chars,
            filter_if_longer_than=filter_if_longer_than,
            memory_mode=memory_mode,
            label_name_map=label_name_map,
            encoding=encoding,
        )

        # use test_file to create test split if available
        test: FlairDataset = ClassificationDataset(
            test_file,
            label_type=label_type,
            tokenizer=tokenizer,
            truncate_to_max_tokens=truncate_to_max_tokens,
            truncate_to_max_chars=truncate_to_max_chars,
            filter_if_longer_than=filter_if_longer_than,
            memory_mode=memory_mode,
            label_name_map=label_name_map,
            encoding=encoding,
        ) if test_file is not None else None

        # use dev_file to create test split if available
        dev: FlairDataset = ClassificationDataset(
            dev_file,
            label_type=label_type,
            tokenizer=tokenizer,
            truncate_to_max_tokens=truncate_to_max_tokens,
            truncate_to_max_chars=truncate_to_max_chars,
            filter_if_longer_than=filter_if_longer_than,
            memory_mode=memory_mode,
            label_name_map=label_name_map,
            encoding=encoding,
        ) if dev_file is not None else None

        super(ClassificationCorpus, self).__init__(
            train, dev, test, name=str(data_folder)
        )


class ClassificationDataset(FlairDataset):
    """
    Dataset for classification instantiated from a single FastText-formatted file.
    """
    def __init__(
            self,
            path_to_file: Union[str, Path],
            label_type: str = 'class',
            truncate_to_max_tokens=-1,
            truncate_to_max_chars=-1,
            filter_if_longer_than: int = -1,
            tokenizer=segtok_tokenizer,
            memory_mode: str = "partial",
            label_name_map: Dict[str, str] = None,
            encoding: str = 'utf-8',
    ):
        """
        Reads a data file for text classification. The file should contain one document/text per line.
        The line should have the following format:
        __label__<class_name> <text>
        If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,
        __label__<class_name_1> __label__<class_name_2> <text>
        :param path_to_file: the path to the data file
        :param label_type: name of the label
        :param truncate_to_max_tokens: If set, truncates each Sentence to a maximum number of tokens
        :param truncate_to_max_chars: If set, truncates each Sentence to a maximum number of chars
        :param filter_if_longer_than: If set, filters documents that are longer that the specified number of tokens.
        :param tokenizer: Tokenizer for dataset, default is segtok
        :param memory_mode: Set to what degree to keep corpus in memory ('full', 'partial' or 'disk'). Use 'full'
        if full corpus and all embeddings fits into memory for speedups during training. Otherwise use 'partial' and if
        even this is too much for your memory, use 'disk'.
        :param label_name_map: Optionally map label names to different schema.
        :param encoding: Default is 'uft-8' but some datasets are in 'latin-1
        :return: list of sentences
        """
        if type(path_to_file) == str:
            path_to_file: Path = Path(path_to_file)

        assert path_to_file.exists()

        self.label_prefix = "__label__"
        self.label_type = label_type

        self.memory_mode = memory_mode
        self.tokenizer = tokenizer

        if self.memory_mode == 'full':
            self.sentences = []
        if self.memory_mode == 'partial':
            self.lines = []
        if self.memory_mode == 'disk':
            self.indices = []

        self.total_sentence_count: int = 0
        self.truncate_to_max_chars = truncate_to_max_chars
        self.truncate_to_max_tokens = truncate_to_max_tokens
        self.filter_if_longer_than = filter_if_longer_than
        self.label_name_map = label_name_map

        self.path_to_file = path_to_file

        with open(str(path_to_file), encoding=encoding) as f:
            line = f.readline()
            position = 0
            while line:
                if "__label__" not in line or " " not in line:
                    position = f.tell()
                    line = f.readline()
                    continue

                if 0 < self.filter_if_longer_than < len(line.split(' ')):
                    position = f.tell()
                    line = f.readline()
                    continue

                if self.memory_mode == 'full':
                    sentence = self._parse_line_to_sentence(
                        line, self.label_prefix, tokenizer
                    )
                    if sentence is not None and len(sentence.tokens) > 0:
                        self.sentences.append(sentence)
                        self.total_sentence_count += 1

                if self.memory_mode == 'partial' or self.memory_mode == 'disk':

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
                    if text and label:

                        if self.memory_mode == 'partial':
                            self.lines.append(line)
                            self.total_sentence_count += 1

                        if self.memory_mode == 'disk':
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

                if self.label_name_map and label in self.label_name_map.keys():
                    label = self.label_name_map[label]

                labels.append(label)
            else:
                break

        text = line[l_len:].strip()

        if self.truncate_to_max_chars > 0:
            text = text[: self.truncate_to_max_chars]

        if text and labels:
            sentence = Sentence(text, use_tokenizer=tokenizer)

            for label in labels:
                sentence.add_label(self.label_type, label)

            if (
                    sentence is not None
                    and 0 < self.truncate_to_max_tokens < len(sentence)
            ):
                sentence.tokens = sentence.tokens[: self.truncate_to_max_tokens]

            return sentence
        return None

    def is_in_memory(self) -> bool:
        if self.memory_mode == 'disk': return False
        if self.memory_mode == 'partial': return False
        return True

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        if self.memory_mode == 'full':
            return self.sentences[index]

        if self.memory_mode == 'partial':
            sentence = self._parse_line_to_sentence(
                self.lines[index], self.label_prefix, self.tokenizer
            )
            return sentence

        if self.memory_mode == 'disk':
            with open(str(self.path_to_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence = self._parse_line_to_sentence(
                    line, self.label_prefix, self.tokenizer
                )
                return sentence


class CSVClassificationCorpus(Corpus):
    """
    Classification corpus instantiated from CSV data files.
    """
    def __init__(
            self,
            data_folder: Union[str, Path],
            column_name_map: Dict[int, str],
            label_type: str = 'class',
            train_file=None,
            test_file=None,
            dev_file=None,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            tokenizer: Callable[[str], List[Token]] = segtok_tokenizer,
            in_memory: bool = False,
            skip_header: bool = False,
            encoding: str = 'utf-8',
            **fmtparams,
    ):
        """
        Instantiates a Corpus for text classification from CSV column formatted data

        :param data_folder: base folder with the task data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param label_type: name of the label
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param tokenizer: Tokenizer for dataset, default is segtok
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param skip_header: If True, skips first line because it is header
        :param encoding: Default is 'uft-8' but some datasets are in 'latin-1
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


class CSVClassificationDataset(FlairDataset):
    """
    Dataset for text classification from CSV column formatted data.
    """
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
        :param label_type: name of the label
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param tokenizer: Tokenizer for dataset, default is segtok
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param skip_header: If True, skips first line because it is header
        :param encoding: Most datasets are 'utf-8' but some are 'latin-1'
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


class AMAZON_REVIEWS(ClassificationCorpus):
    """
    A very large corpus of Amazon reviews with positivity ratings. Corpus is downloaded from and documented at
    https://nijianmo.github.io/amazon/index.html. We download the 5-core subset which is still tens of millions of
    reviews.
    """
    def __init__(
            self,
            label_name_map=None,
            split_max=10000,
            memory_mode='partial',
            **corpusargs
    ):
        """
        Constructs corpus object. Split_max indicates how many data points from each of the 28 splits are used, so
        set this higher or lower to increase/decrease corpus size.
        :param label_name_map: Map label names to different schema. By default, the 5-star rating is mapped onto 3
        classes (POSITIVE, NEGATIVE, NEUTRAL)
        :param split_max: Split_max indicates how many data points from each of the 28 splits are used, so
        set this higher or lower to increase/decrease corpus size.
        :param memory_mode: Set to what degree to keep corpus in memory ('full', 'partial' or 'disk'). Use 'full'
        if full corpus and all embeddings fits into memory for speedups during training. Otherwise use 'partial' and if
        even this is too much for your memory, use 'disk'.
        :param corpusargs: Arguments for ClassificationCorpus
        """

        # by defaut, map point score to POSITIVE / NEGATIVE values
        if label_name_map is None:
            label_name_map = {'1.0': 'NEGATIVE',
                              '2.0': 'NEGATIVE',
                              '3.0': 'NEUTRAL',
                              '4.0': 'POSITIVE',
                              '5.0': 'POSITIVE'}

        # dataset name includes the split size
        dataset_name = self.__class__.__name__.lower() + '_' + str(split_max)

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download each of the 28 splits
            self.download_and_prepare_amazon_product_file(data_folder, "AMAZON_FASHION_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "All_Beauty_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Appliances_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Arts_Crafts_and_Sewing_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Arts_Crafts_and_Sewing_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Automotive_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Books_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "CDs_and_Vinyl_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Cell_Phones_and_Accessories_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Clothing_Shoes_and_Jewelry_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Digital_Music_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Electronics_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Gift_Cards_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Grocery_and_Gourmet_Food_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Home_and_Kitchen_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Industrial_and_Scientific_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Kindle_Store_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Luxury_Beauty_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Magazine_Subscriptions_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Movies_and_TV_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Musical_Instruments_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Office_Products_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Patio_Lawn_and_Garden_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Pet_Supplies_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Prime_Pantry_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Software_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Sports_and_Outdoors_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Tools_and_Home_Improvement_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Toys_and_Games_5.json.gz", split_max)
            self.download_and_prepare_amazon_product_file(data_folder, "Video_Games_5.json.gz", split_max)

        super(AMAZON_REVIEWS, self).__init__(
            data_folder,
            label_type='sentiment',
            tokenizer=segtok_tokenizer,
            label_name_map=label_name_map,
            memory_mode=memory_mode,
            **corpusargs
        )

    def download_and_prepare_amazon_product_file(self, data_folder, part_name, max_data_points = None):
        amazon__path = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall"
        cached_path(f"{amazon__path}/{part_name}", Path("datasets") / 'Amazon_Product_Reviews')
        import gzip
        # create dataset directory if necessary
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        with open(data_folder / "train.txt", "a") as train_file:

            write_count = 0
            # download senteval datasets if necessary und unzip
            with gzip.open(Path(flair.cache_root) / "datasets" / 'Amazon_Product_Reviews' / part_name, "rb", ) as f_in:
                for line in f_in:
                    parsed_json = json.loads(line)
                    if 'reviewText' not in parsed_json:
                        continue
                    if parsed_json['reviewText'].strip() == '':
                        continue
                    text = parsed_json['reviewText'].replace('\n', '')
                    train_file.write(f"__label__{parsed_json['overall']} {text}\n")

                    write_count += 1
                    if max_data_points and write_count >= max_data_points:
                        break


class IMDB(ClassificationCorpus):
    """
    Corpus of IMDB movie reviews labeled by sentiment (POSITIVE, NEGATIVE). Downloaded from and documented at
    http://ai.stanford.edu/~amaas/data/sentiment/.
    """
    def __init__(self, base_path: Union[str, Path] = None, rebalance_corpus: bool = True, **corpusargs):
        """

        :param base_path: Provide this only if you store the IMDB corpus in a specific folder, otherwise use default.
        :param rebalance_corpus: Default splits for this corpus have a strange 50/50 train/test split that are impractical.
        With rebalance_corpus=True (default setting), corpus is rebalanced to a 80/10/10 train/dev/test split. If you
        want to use original splits, set this to False.
        :param corpusargs: Other args for ClassificationCorpus.
        """

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower() + '_v2'

        if rebalance_corpus:
            dataset_name = dataset_name + '-rebalanced'

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
                        with open(f"{data_path}/train-all.txt", "at") as f_p:
                            current_path = data_path / "aclImdb" / dataset / label
                            for file_name in current_path.iterdir():
                                if file_name.is_file() and file_name.name.endswith(
                                        ".txt"
                                ):
                                    if label == "pos": sentiment_label = 'POSITIVE'
                                    if label == "neg": sentiment_label = 'NEGATIVE'
                                    f_p.write(
                                        f"__label__{sentiment_label} "
                                        + file_name.open("rt", encoding="utf-8").read()
                                        + "\n"
                                    )

        super(IMDB, self).__init__(
            data_folder, tokenizer=space_tokenizer, **corpusargs
        )


class NEWSGROUPS(ClassificationCorpus):
    """
    20 newsgroups corpus available at "http://qwone.com/~jason/20Newsgroups", classifying
    news items into one of 20 categories. Each data point is a full news article so documents may be very long.
    """
    def __init__(self, base_path: Union[str, Path] = None, **corpusargs):
        """
        Instantiates 20 newsgroups corpus.
        :param base_path: Provide this only if you store the IMDB corpus in a specific folder, otherwise use default.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, tokenizer=space_tokenizer, **corpusargs,
        )


class SENTIMENT_140(ClassificationCorpus):
    """
    Twitter sentiment corpus downloaded from and documented at http://help.sentiment140.com/for-students. Two sentiments
    in train data (POSITIVE, NEGATIVE) and three sentiments in test data (POSITIVE, NEGATIVE, NEUTRAL).
    """
    def __init__(
            self,
            label_name_map=None,
            **corpusargs,
    ):
        """
        Instantiates twitter sentiment corpus.
        :param label_name_map: By default, the numeric values are mapped to ('NEGATIVE', 'POSITIVE' and 'NEUTRAL')
        :param corpusargs: Other args for ClassificationCorpus.
        """

        # by defaut, map point score to POSITIVE / NEGATIVE values
        if label_name_map is None:
            label_name_map = {'0': 'NEGATIVE',
                              '2': 'NEUTRAL',
                              '4': 'POSITIVE'}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if True or not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            sentiment_url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
            cached_path(sentiment_url, Path("datasets") / dataset_name / 'raw')
            senteval_folder = Path(flair.cache_root) / "datasets" / dataset_name / 'raw'
            unzip_file(senteval_folder / "trainingandtestdata.zip", senteval_folder)

            # create dataset directory if necessary
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            # create train.txt file from CSV
            with open(data_folder / "train.txt", "w") as train_file:

                with open(senteval_folder / "training.1600000.processed.noemoticon.csv", encoding='latin-1') as csv_train:

                    csv_reader = csv.reader(csv_train)

                    for row in csv_reader:

                        label = row[0]
                        text = row[5]
                        train_file.write(f"__label__{label} {text}\n")

            # create test.txt file from CSV
            with open(data_folder / "test.txt", "w") as train_file:

                with open(senteval_folder / "testdata.manual.2009.06.14.csv", encoding='latin-1') as csv_train:

                    csv_reader = csv.reader(csv_train)

                    for row in csv_reader:

                        label = row[0]
                        text = row[5]
                        train_file.write(f"__label__{label} {text}\n")

        super(SENTIMENT_140, self).__init__(
            data_folder, label_type='sentiment', tokenizer=segtok_tokenizer, label_name_map=label_name_map, **corpusargs,
        )


class SENTEVAL_CR(ClassificationCorpus):
    """
    The customer reviews dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified into
    NEGATIVE or POSITIVE sentiment.
    """
    def __init__(
            self,
            **corpusargs,
    ):
        """
        Instantiates SentEval customer reviews dataset.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, label_type='sentiment', tokenizer=segtok_tokenizer, **corpusargs,
        )


class SENTEVAL_MR(ClassificationCorpus):
    """
    The movie reviews dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified into
    NEGATIVE or POSITIVE sentiment.
    """
    def __init__(
            self,
            **corpusargs
    ):
        """
        Instantiates SentEval movie reviews dataset.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, label_type='sentiment', tokenizer=segtok_tokenizer, **corpusargs
        )


class SENTEVAL_SUBJ(ClassificationCorpus):
    """
    The subjectivity dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified into
    SUBJECTIVE or OBJECTIVE sentiment.
    """
    def __init__(
            self,
            **corpusargs,
    ):
        """
        Instantiates SentEval subjectivity dataset.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, label_type='objectivity', tokenizer=segtok_tokenizer, **corpusargs,
        )


class SENTEVAL_MPQA(ClassificationCorpus):
    """
    The opinion-polarity dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified into
    NEGATIVE or POSITIVE polarity.
    """

    def __init__(
            self,
            **corpusargs,
    ):
        """
        Instantiates SentEval opinion polarity dataset.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, label_type='sentiment', tokenizer=segtok_tokenizer, **corpusargs,
        )


class SENTEVAL_SST_BINARY(ClassificationCorpus):
    """
    The Stanford sentiment treebank dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified
    into NEGATIVE or POSITIVE sentiment.
    """

    def __init__(
            self,
            **corpusargs
    ):
        """
        Instantiates SentEval Stanford sentiment treebank dataset.
        :param corpusargs: Other args for ClassificationCorpus.
        """

        # this dataset name
        dataset_name = self.__class__.__name__.lower() + '_v2'

        # default dataset folder is the cache root
        data_folder = Path(flair.cache_root) / "datasets" / dataset_name

        # download data if necessary
        if not (data_folder / "train.txt").is_file():

            # download senteval datasets if necessary und unzip
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-train', Path("datasets") / dataset_name / 'raw')
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-test', Path("datasets") / dataset_name / 'raw')
            cached_path('https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-dev', Path("datasets") / dataset_name / 'raw')

            # create train.txt file by iterating over pos and neg file
            with open(data_folder / "train.txt", "a") as out_file, open(data_folder / 'raw' / "sentiment-train") as in_file:
                for line in in_file:
                    fields = line.split('\t')
                    label = 'POSITIVE' if fields[1].rstrip() == '1' else 'NEGATIVE'
                    out_file.write(f"__label__{label} {fields[0]}\n")

        super(SENTEVAL_SST_BINARY, self).__init__(
            data_folder,
            tokenizer=segtok_tokenizer,
            **corpusargs,
        )


class SENTEVAL_SST_GRANULAR(ClassificationCorpus):
    """
    The Stanford sentiment treebank dataset of SentEval, see https://github.com/facebookresearch/SentEval, classified
    into 5 sentiment classes.
    """

    def __init__(
            self,
            **corpusargs,
    ):
        """
        Instantiates SentEval Stanford sentiment treebank dataset.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            **corpusargs,
        )


class TREC_50(ClassificationCorpus):
    """
    The TREC Question Classification Corpus, classifying questions into 50 fine-grained answer types.
    """

    def __init__(self, base_path: Union[str, Path] = None, **corpusargs):
        """
        Instantiates TREC Question Classification Corpus with 50 classes.
        :param base_path: Provide this only if you store the TREC corpus in a specific folder, otherwise use default.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, tokenizer=space_tokenizer, **corpusargs,
        )


class TREC_6(ClassificationCorpus):
    """
    The TREC Question Classification Corpus, classifying questions into 6 coarse-grained answer types
    (DESC, HUM, LOC, ENTY, NUM, ABBR).
    """

    def __init__(self, base_path: Union[str, Path] = None, **corpusargs):
        """
        Instantiates TREC Question Classification Corpus with 6 classes.
        :param base_path: Provide this only if you store the TREC corpus in a specific folder, otherwise use default.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, label_type='question_type', tokenizer=space_tokenizer, **corpusargs,
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
    """
    WASSA-2017 anger emotion-intensity dataset downloaded from and documented at
     https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
    """
    def __init__(self, base_path: Union[str, Path] = None, **corpusargs):
        """
        Instantiates WASSA-2017 anger emotion-intensity dataset
        :param base_path: Provide this only if you store the WASSA corpus in a specific folder, otherwise use default.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, tokenizer=space_tokenizer, **corpusargs,
        )


class WASSA_FEAR(ClassificationCorpus):
    """
    WASSA-2017 fear emotion-intensity dataset downloaded from and documented at
     https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
    """
    def __init__(self, base_path: Union[str, Path] = None, **corpusargs):
        """
        Instantiates WASSA-2017 fear emotion-intensity dataset
        :param base_path: Provide this only if you store the WASSA corpus in a specific folder, otherwise use default.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, tokenizer=space_tokenizer, **corpusargs
        )


class WASSA_JOY(ClassificationCorpus):
    """
    WASSA-2017 joy emotion-intensity dataset downloaded from and documented at
     https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
    """
    def __init__(self, base_path: Union[str, Path] = None, **corpusargs):
        """
        Instantiates WASSA-2017 joy emotion-intensity dataset
        :param base_path: Provide this only if you store the WASSA corpus in a specific folder, otherwise use default.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, tokenizer=space_tokenizer, **corpusargs,
        )


class WASSA_SADNESS(ClassificationCorpus):
    """
    WASSA-2017 sadness emotion-intensity dataset downloaded from and documented at
     https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
    """
    def __init__(self, base_path: Union[str, Path] = None, **corpusargs):
        """
        Instantiates WASSA-2017 sadness emotion-intensity dataset
        :param base_path: Provide this only if you store the WASSA corpus in a specific folder, otherwise use default.
        :param corpusargs: Other args for ClassificationCorpus.
        """

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
            data_folder, tokenizer=space_tokenizer, **corpusargs,
        )