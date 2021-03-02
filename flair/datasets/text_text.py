import logging
import os
from pathlib import Path
from typing import List, Union
from flair.datasets.base import find_train_dev_test_files

import flair
from flair.data import (
    Sentence,
    Corpus,
    FlairDataset,
    DataPair,
)
from flair.file_utils import cached_path, unpack_file, unzip_file

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


class DataPairCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            columns: List[int] = [0, 1, 2],
            train_file=None,
            test_file=None,
            dev_file=None,
            use_tokenizer: bool = True,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            in_memory: bool = True,
            label_type: str = None,
            autofind_splits=True,
            sample_missing_splits: bool = True,
            skip_first_line: bool = False,
            separator: str = '\t',
            encoding: str = 'utf-8'
    ):
        """
        Corpus for tasks involving pairs of sentences or paragraphs. The data files are expected to be in column format where each line has a colmun 
        for the first sentence/paragraph, the second sentence/paragraph and the labels, respectively. The columns must be separated by a given separator (default: '\t').
        
        :param data_folder: base folder with the task data
        :param columns: List that indicates the columns for the first sentence (first entry in the list), the second sentence (second entry) and label (last entry).
                        default = [0,1,2]
        :param train_file: the name of the train file
        :param test_file: the name of the test file, if None, dev data is sampled from train (if sample_missing_splits is true)
        :param dev_file: the name of the dev file, if None, dev data is sampled from train (if sample_missing_splits is true)
        :param use_tokenizer: Whether or not to use in-built tokenizer
        :param max_tokens_per_doc: If set, shortens sentences to this maximum number of tokens
        :param max_chars_per_doc: If set, shortens sentences to this maximum number of characters
        :param in_memory: If True, data will be saved in list of flair.data.DataPair objects, other wise we use lists with simple strings which needs less space
        :param label_type: Name of the label of the data pairs
        :param autofind_splits: If True, train/test/dev files will be automatically identified in the given data_folder
        :param sample_missing_splits: If True, a missing train/test/dev file will be sampled from the available data
        :param skip_first_line: If True, first line of data files will be ignored
        :param separator: Separator between columns in data files
        :param encoding: Encoding of data files
        
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file, autofind_splits=autofind_splits)

        # create DataPairDataset for train, test and dev file, if they are given

        train: FlairDataset = DataPairDataset(
            train_file,
            columns=columns,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            label_type=label_type,
            skip_first_line=skip_first_line,
            separator=separator,
            encoding=encoding
        ) if train_file is not None else None

        test: FlairDataset = DataPairDataset(
            test_file,
            columns=columns,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            label_type=label_type,
            skip_first_line=skip_first_line,
            separator=separator,
            encoding=encoding
        ) if test_file is not None else None

        dev: FlairDataset = DataPairDataset(
            dev_file,
            columns=columns,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            label_type=label_type,
            skip_first_line=skip_first_line,
            separator=separator,
            encoding=encoding
        ) if dev_file is not None else None

        super(DataPairCorpus, self).__init__(train, dev, test,
                                             sample_missing_splits=sample_missing_splits,
                                             name=str(data_folder))


class DataPairDataset(FlairDataset):
    def __init__(
            self,
            path_to_data: Union[str, Path],
            columns: List[int] = [0, 1, 2],
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            use_tokenizer=True,
            in_memory: bool = True,
            label_type: str = None,
            skip_first_line: bool = False,
            separator: str = '\t',
            encoding: str = 'utf-8',
            label: bool = True
    ):
        """
        Creates a Dataset for pairs of sentences/paragraphs. The file needs to be in a column format, 
        where each line has a column for the first sentence/paragraph, the second sentence/paragraph and the label 
        seperated by e.g. '\t' (just like in the glue RTE-dataset https://gluebenchmark.com/tasks) .
        For each data pair we create a flair.data.DataPair object.
        
        :param path_to_data: path to the data file
        :param columns: list of integers that indicate the respective columns. The first entry is the column
        for the first sentence, the second for the second sentence and the third for the label. Default [0,1,2]
        :param max_tokens_per_doc: If set, shortens sentences to this maximum number of tokens
        :param max_chars_per_doc: If set, shortens sentences to this maximum number of characters
        :param use_tokenizer: Whether or not to use in-built tokenizer
        :param in_memory: If True, data will be saved in list of flair.data.DataPair objects, other wise we use lists with simple strings which needs less space
        :param label_type: Name of the label of the data pairs
        :param skip_first_line: If True, first line of data file will be ignored
        :param separator: Separator between columns in the data file
        :param encoding: Encoding of the data file
        :param label: If False, the dataset expects unlabeled data
        """

        if type(path_to_data) == str:
            path_to_data: Path = Path(path_to_data)

        # stop if file does not exist
        assert path_to_data.exists()

        self.in_memory = in_memory

        self.use_tokenizer = use_tokenizer

        self.max_tokens_per_doc = max_tokens_per_doc

        self.label = label

        self.label_type = label_type

        self.total_data_count: int = 0

        if self.in_memory:
            self.data_pairs: List[DataPair] = []
        else:
            self.first_elements: List[str] = []
            self.second_elements: List[str] = []
            self.labels: List[str] = []

        with open(str(path_to_data), encoding=encoding) as source_file:

            source_line = source_file.readline()

            if skip_first_line:
                source_line = source_file.readline()

            while source_line:

                source_line_list = source_line.strip().split(separator)

                first_element = source_line_list[columns[0]]
                second_element = source_line_list[columns[1]]

                if self.label:
                    pair_label = source_line_list[columns[2]]
                else:
                    pair_label = None

                if max_chars_per_doc > 0:
                    first_element = first_element[:max_chars_per_doc]
                    second_element = second_element[:max_chars_per_doc]

                if self.in_memory:

                    data_pair = self._make_data_pair(first_element, second_element, pair_label)
                    self.data_pairs.append(data_pair)
                else:
                    self.first_elements.append(first_element)
                    self.second_elements.append(second_element)
                    if self.label:
                        self.labels.append(pair_label)

                self.total_data_count += 1

                source_line = source_file.readline()

    # create a DataPair object from strings
    def _make_data_pair(self, first_element: str, second_element: str, label: str = None):

        first_sentence = Sentence(first_element, use_tokenizer=self.use_tokenizer)
        second_sentence = Sentence(second_element, use_tokenizer=self.use_tokenizer)

        if self.max_tokens_per_doc > 0:
            first_sentence.tokens = first_sentence.tokens[: self.max_tokens_per_doc]
            second_sentence.tokens = second_sentence.tokens[: self.max_tokens_per_doc]

        data_pair = DataPair(first_sentence, second_sentence)

        if label:
            data_pair.add_label(label_type=self.label_type, value=label)

        return data_pair

    def is_in_memory(self) -> bool:

        return self.in_memory

    def __len__(self):
        return self.total_data_count

    # if in_memory is True we return a datapair, otherwise we create one from the lists of strings
    def __getitem__(self, index: int = 0) -> DataPair:
        if self.in_memory:
            return self.data_pairs[index]
        elif self.label:
            return self._make_data_pair(
                self.first_elements[index], self.second_elements[index], self.labels[index]
            )
        else:
            return self._make_data_pair(
                self.first_elements[index], self.second_elements[index]
            )


class GLUE_RTE(DataPairCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            use_tokenizer=True,
            in_memory: bool = True,
            sample_missing_splits: bool = True
    ):
        """
        Creates a DataPairCorpus for the Glue Recognizing Textual Entailment (RTE) data (https://gluebenchmark.com/tasks).
        Additionaly to the Corpus we have a eval_dataset containing the test file of the Glue data. 
        This file contains unlabeled test data to evaluate models on the Glue RTE task.
        """

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"

        # if no base_path provided take cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        data_file = data_folder / "RTE/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            # get the zip file
            zipped_data_path = cached_path(
                'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
                Path("datasets") / dataset_name
            )

            unpack_file(
                zipped_data_path,
                data_folder,
                mode="zip",
                keep=False
            )

            # rename test file to eval_dataset, since it has no labels
            os.rename(str(data_folder / "RTE/test.tsv"), str(data_folder / "RTE/eval_dataset.tsv"))

        super(GLUE_RTE, self).__init__(
            data_folder / "RTE",
            columns=[1, 2, 3],
            skip_first_line=True,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            label_type='textual_entailment',
            sample_missing_splits=sample_missing_splits

        )

        self.eval_dataset = DataPairDataset(
            data_folder / "RTE/eval_dataset.tsv",
            columns=[1, 2, 3],
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_first_line=True,
            label=False
        )

    """    
    This function creates a tsv file of the predictions of the eval_dataset (after calling classifier.predict(corpus.eval_dataset, label_name='textual_entailment')).
    The resulting file is called RTE.tsv and is in the format required for submission to the Glue Benchmark.
    """

    def tsv_from_eval_dataset(self, folder_path: Union[str, Path]):

        if type(folder_path) == str:
            folder_path = Path(folder_path)
        folder_path = folder_path / 'RTE.tsv'

        with open(folder_path, mode='w') as tsv_file:
            tsv_file.write("index\tprediction\n")
            for index, datapoint in enumerate(self.eval_dataset):
                tsv_file.write(str(index) + '\t' + datapoint.get_labels('textual_entailment')[0].value + '\n')


class SUPERGLUE_RTE(DataPairCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            use_tokenizer=True,
            in_memory: bool = True,
            sample_missing_splits: bool = True
    ):
        """
        Creates a DataPairCorpus for the SuperGlue Recognizing Textual Entailment (RTE) data (https://super.gluebenchmark.com/tasks).
        Additionaly to the Corpus we have a eval_dataset containing the test file of the SuperGlue data. 
        This file contains unlabeled test data to evaluate models on the SuperGlue RTE task.
        """

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "superglue"

        # if no base_path provided take cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        data_file = data_folder / "RTE/train.tsv"

        # if data not downloaded yet, download it
        if not data_file.is_file():
            # get the zip file
            zipped_data_path = cached_path(
                'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip',
                Path("datasets") / dataset_name
            )

            unpack_file(
                zipped_data_path,
                data_folder,
                mode="zip",
                keep=False
            )

            # the downloaded files have json format, we transform them to tsv
            rte_jsonl_to_tsv(data_folder / "RTE/train.jsonl", remove=True)
            rte_jsonl_to_tsv(data_folder / "RTE/test.jsonl", remove=True, label=False)
            rte_jsonl_to_tsv(data_folder / "RTE/val.jsonl", remove=True)

            os.rename(str(data_folder / "RTE/val.tsv"), str(data_folder / "RTE/dev.tsv"))
            os.rename(str(data_folder / "RTE/test.tsv"), str(data_folder / "RTE/eval_dataset.tsv"))

        super(SUPERGLUE_RTE, self).__init__(
            data_folder / "RTE",
            columns=[0, 1, 2],
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            label_type='textual_entailment',
            sample_missing_splits=sample_missing_splits
        )

        self.eval_dataset = DataPairDataset(
            data_folder / "RTE/eval_dataset.tsv",
            columns=[0, 1, 2],
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_first_line=False,
            label=False
        )

    """    
    Creates JSONL file of the predictions of the eval_dataset (after calling classifier.predict(corpus.eval_dataset, label_name='textual_entailment')).
    The resulting file is called RTE.jsonl and is in the form required for submission to the SuperGlue Benchmark.
    """

    def jsonl_from_eval_dataset(self, folder_path: Union[str, Path]):

        if type(folder_path) == str:
            folder_path = Path(folder_path)
        folder_path = folder_path / 'RTE.jsonl'

        with open(folder_path, mode='w') as jsonl_file:

            for index, datapoint in enumerate(self.eval_dataset):
                entry = {"idx": index, "label": datapoint.get_labels('textual_entailment')[0].value}
                jsonl_file.write(str(entry) + '\n')


# Function to transform JSON file to tsv for Recognizing Textual Entailment Data
def rte_jsonl_to_tsv(file_path: Union[str, Path], label: bool = True, remove: bool = False, encoding='utf-8'):
    import json

    tsv_file = os.path.splitext(file_path)[0] + '.tsv'

    with open(file_path, 'r', encoding=encoding) as jsonl_file:
        with open(tsv_file, 'w', encoding=encoding) as tsv_file:

            line = jsonl_file.readline()

            while line:

                obj = json.loads(line)
                new_line = obj["premise"] + '\t' + obj["hypothesis"]
                if label:
                    new_line += '\t' + obj["label"]
                new_line += '\n'

                tsv_file.write(new_line)

                line = jsonl_file.readline()

    # remove json file
    if remove:
        os.remove(file_path)
