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
        
        
class TextualEntailmentCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            columns: List[int] = [0,1,2],
            train_file=None,
            test_file=None,
            dev_file=None,
            use_tokenizer: bool = True,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            in_memory: bool = True,
            autofind_splits=True,
            sample_missing_splits: bool = True,
            skip_first_line: bool = False,
            separator: str = '\t',
            encoding: str = 'utf-8'
    ):
        """
        Corpus for Recognizing Textual Entailment Tasks. The data files are expected to be in column format with colmuns 
        for the premises, the hypothesises and the labels, respectively. Labels could for example be "entailment" and "not_entailment".
        
        :param data_folder: base folder with the task data
        :param columns: Indicates the columns for premise (first entry in the list), hypothesis (second entry) and label (last entry).
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param use_tokenizer: Whether or not to use in-built tokenizer
        :param max_tokens_per_doc: If set, shortens sentences to this maximum number of tokens
        :param max_chars_per_doc: If set, shortens sentences to this maximum number of characters
        :param in_memory: If True, keeps dataset fully in memory
        :param autofind_splits: If True, train/test/dev files will be automatically identified
        :param sample_missing_splits: If True, a missing train/test/dev file will be sampled from the available data
        :param skip_first_line: If True, first line of data files will be ignored
        :param separator: Separator between columns in data files
        :param encoding: Encoding of data files
        
        :return: a Corpus with annotated train, dev and test data
        """
        
        # find train, dev and test files if not specified
        #hängt Dateinamen and Pfad data_folder
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file, autofind_splits=autofind_splits)
            
        train: FlairDataset = TextualEntailmentDataset(
            train_file,
            columns=columns,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_first_line=skip_first_line,
            separator=separator,
            encoding=encoding
        ) if train_file is not None else None
        
        test: FlairDataset = TextualEntailmentDataset(
            test_file,
            columns=columns,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_first_line=skip_first_line,
            separator=separator,
            encoding=encoding
        ) if test_file is not None else None

        dev: FlairDataset = TextualEntailmentDataset(
            dev_file,
            columns=columns,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_first_line=skip_first_line,
            separator=separator,
            encoding=encoding
        ) if dev_file is not None else None

        super(TextualEntailmentCorpus, self).__init__(train, dev, test,
                                                      sample_missing_splits=sample_missing_splits,
                                                      name=str(data_folder))
        
        
class TextualEntailmentDataset(FlairDataset):
    def __init__(
            self,
            path_to_data: Union[str, Path],
            columns: List[int] = [0,1,2],
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            use_tokenizer=True,
            in_memory: bool = True,
            skip_first_line: bool = False,
            separator: str = '\t',
            encoding: str = 'utf-8',
            label: bool = True
    ):
        """
        Creates a Dataset for Recognizing Textual Entailment (RTE). The file needs to be in a column format, 
        where each line has a column for the premise, the hypothesis and the label (e.g. "entailment"/"not entailment") 
        seperated by e.g. '\t' (just like in the glue RTE-dataset https://gluebenchmark.com/tasks) .
        Each premise-hypothesis pair is stored in a flair.data.DataPair object.
        
        :param path_to_data: path to the data file
        :param columns: list of integers that indicate the respective columns. The first entry is the column
        for the premise, the second for the hypothesis and the third for the label. Default [0,1,2]
        :param max_tokens_per_doc: If set, shortens sentences to this maximum number of tokens
        :param max_chars_per_doc: If set, shortens sentences to this maximum number of characters
        :param use_tokenizer: Whether or not to use in-built tokenizer
        :param in_memory: If True, keeps dataset fully in memory
        :param skip_first_line: If True, first line of data file will be ignored
        :param separator: Separator between columns in the data file
        :param encoding: Encoding of the data file
        :param label: If False, the dataset expects unlabeled data
        """
        
        if type(path_to_data) == str:
            path_to_data: Path = Path(path_to_data)

        assert path_to_data.exists()

        self.in_memory = in_memory

        self.use_tokenizer = use_tokenizer
        
        self.max_tokens_per_doc = max_tokens_per_doc
        
        self.label = label

        self.total_data_count: int = 0

        if self.in_memory:
            self.bi_sentences: List[DataPair] = []
        else:
            self.premises: List[str] = []
            self.hypothesises: List[str] = []
            self.labels: List[str] = []

        with open(str(path_to_data), encoding=encoding) as source_file:

            source_line = source_file.readline()
            
            if skip_first_line:
                source_line = source_file.readline()

            while source_line:
                
                source_line_list = source_line.strip().split(separator)

                premise = source_line_list[columns[0]]
                hypothesis = source_line_list[columns[1]]
                
                if self.label:
                    entailment_label = source_line_list[columns[2]]
                else:
                    entailment_label=None

                if max_chars_per_doc > 0:
                    premise = premise[:max_chars_per_doc]
                    hypothesis = hypothesis[:max_chars_per_doc]

                if self.in_memory:

                    bi_sentence = self._make_entailment_pair(premise, hypothesis, entailment_label)
                    self.bi_sentences.append(bi_sentence)
                else:
                    self.premises.append(premise)
                    self.hypothesises.append(hypothesis)
                    if self.label:
                        self.labels.append(entailment_label)
                    
                self.total_data_count += 1
                
                source_line = source_file.readline()

    def _make_entailment_pair(self, premise: str, hypothesis: str, label: str):

        premise_sentence = Sentence(premise, use_tokenizer=self.use_tokenizer)
        hypothesis_sentence = Sentence(hypothesis, use_tokenizer=self.use_tokenizer)

        if self.max_tokens_per_doc > 0:
            premise_sentence.tokens = premise_sentence.tokens[: self.max_tokens_per_doc]
            hypothesis_sentence.tokens = hypothesis_sentence.tokens[: self.max_tokens_per_doc]

        entailment_pair = DataPair(premise_sentence, hypothesis_sentence)
        
        if label:        
            entailment_pair.add_label(label_type="textual_entailment", value=label)
        
        return entailment_pair
    
    def is_in_memory(self) -> bool:
        
        return self.in_memory

    def __len__(self):
        return self.total_data_count

    def __getitem__(self, index: int = 0) -> DataPair:
        if self.in_memory:
            return self.bi_sentences[index]
        elif self.label:
            return self._make_entailment_pair(
                self.premises[index], self.hypothesises[index], self.labels[index]
            )
        else:
            return self._make_entailment_pair(
                self.premises[index], self.hypothesises[index], None
            )
        
        
class GLUE_RTE(TextualEntailmentCorpus):
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
        Creates a Recognizing Textual Entailment Corpus for the Glue RTE data (https://gluebenchmark.com/tasks).
        Additionaly to the TextualEntailmentCorpus we have a eval_dataset containing the test file of the Glue data. 
        This file contains unlabeled test data to evaluate models on the Glue RTE task.
        """
        
        if type(base_path) == str:
            base_path: Path = Path(base_path)
            
        dataset_name = "glue"

        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name 
        
        data_file = data_folder / "RTE/train.tsv"
        
        if not data_file.is_file():
        
            #get the zip file
            zipped_data_path = cached_path(
                    'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
                    Path("datasets") / dataset_name
                )
            
            unpack_file(
                zipped_data_path , 
                data_folder,
                mode = "zip",
                keep= False 
                )
            
            os.rename(str(data_folder / "RTE/test.tsv"), str(data_folder / "RTE/eval_dataset.tsv"))
            
        super(GLUE_RTE, self).__init__(
            data_folder / "RTE",
            columns=[1,2,3],
            skip_first_line=True,
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            sample_missing_splits=sample_missing_splits

        )
        
        self.eval_dataset =  TextualEntailmentDataset(
            data_folder / "RTE/eval_dataset.tsv",
            columns=[1,2,3],
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_first_line=True,
            label=False
        )

        
        
class SUPERGLUE_RTE(TextualEntailmentCorpus):
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
        Creates a Recognizing Textual Entailment Corpus for the SuperGlue RTE data (https://super.gluebenchmark.com/tasks).
        Additionaly to the TextualEntailmentCorpus we have a eval_dataset containing the test file of the SuperGlue data. 
        This file contains unlabeled test data to evaluate models on the SuperGlue RTE task.
        """
        
        if type(base_path) == str:
            base_path: Path = Path(base_path)
            
        dataset_name = "superglue"

        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name 
        
        data_file = data_folder / "RTE/train.tsv"
        
        if not data_file.is_file():
        
            #get the zip file
            zipped_data_path = cached_path(
                    'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip',
                    Path("datasets") / dataset_name
                )
            
            unpack_file(
                zipped_data_path , 
                data_folder,
                mode = "zip",
                keep= False 
                )
            
            # transform to tsv and delete json files
            rte_jsonl_to_tsv(data_folder / "RTE/train.jsonl", remove = True)
            rte_jsonl_to_tsv(data_folder / "RTE/test.jsonl", remove = True, label=False)
            rte_jsonl_to_tsv(data_folder / "RTE/val.jsonl", remove = True)
            
            os.rename(str(data_folder / "RTE/val.tsv"), str(data_folder / "RTE/dev.tsv"))
            os.rename(str(data_folder / "RTE/test.tsv"), str(data_folder / "RTE/eval_dataset.tsv"))

        super(SUPERGLUE_RTE, self).__init__(
            data_folder / "RTE",
            columns=[0,1,2],
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            sample_missing_splits=sample_missing_splits
        )
        
        
        self.eval_dataset =  TextualEntailmentDataset(
            data_folder / "RTE/eval_dataset.tsv",
            columns=[0,1,2],
            use_tokenizer=use_tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_first_line=False,
            label=False
        )
        
#Function to transform JSON file to tsv for Recognizing Textual Entailment Data
def rte_jsonl_to_tsv(file_path: Union[str, Path], label: bool = True, remove: bool = False, encoding='utf-8'):
    
    import json
    
    tsv_file = os.path.splitext(file_path)[0] + '.tsv'
        
    with open(file_path,'r', encoding=encoding) as jsonl_file:
        with open(tsv_file,'w', encoding=encoding) as tsv_file:
            
            line = jsonl_file.readline()
            
            while line:
                
                obj = json.loads(line)
                new_line = obj["premise"] + '\t' + obj["hypothesis"]
                if label:
                    new_line +=  '\t' + obj["label"]
                new_line += '\n'
                
                tsv_file.write(new_line)
                
                line = jsonl_file.readline()
                
    #remove json file 
    if remove:
        os.remove(file_path)       
    