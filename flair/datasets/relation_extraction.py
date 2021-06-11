import logging
import re
import io
from pathlib import Path
from typing import List, Union, Tuple

import flair
from flair.data import (
    Sentence,
    Corpus,
    Token,
    FlairDataset,
    Relation,
    Span
)
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class CoNLLUCorpus(Corpus):
    def __init__(
            self,
            data_folder: Union[str, Path],
            train_file=None,
            test_file=None,
            dev_file=None,
            in_memory: bool = True,
            split_multiwords: bool = True,
    ):
        """
        Instantiates a Corpus from CoNLL-U column-formatted task data such as the UD corpora

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        :param split_multiwords: If set to True, multiwords are split (default), otherwise kept as single tokens
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        # get train data
        train = CoNLLUDataset(train_file, in_memory=in_memory, split_multiwords=split_multiwords)

        # get test data
        test = CoNLLUDataset(test_file, in_memory=in_memory, split_multiwords=split_multiwords) \
            if test_file is not None else None

        # get dev data
        dev = CoNLLUDataset(dev_file, in_memory=in_memory, split_multiwords=split_multiwords) \
            if dev_file is not None else None

        super(CoNLLUCorpus, self).__init__(
            train, dev, test, name=str(data_folder)
        )


class CoNLLUDataset(FlairDataset):
    def __init__(self, path_to_conllu_file: Union[str, Path], in_memory: bool = True, split_multiwords: bool = True):
        """
        Instantiates a column dataset in CoNLL-U format.

        :param path_to_conllu_file: Path to the CoNLL-U formatted file
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        """
        if type(path_to_conllu_file) is str:
            path_to_conllu_file = Path(path_to_conllu_file)
        assert path_to_conllu_file.exists()

        self.in_memory: bool = in_memory
        self.split_multiwords: bool = split_multiwords

        self.path_to_conllu_file = path_to_conllu_file
        self.total_sentence_count: int = 0

        with open(str(self.path_to_conllu_file), encoding="utf-8") as file:

            # option 1: read only sentence boundaries as offset positions
            if not self.in_memory:
                self.indices: List[int] = []

                line = file.readline()
                position = 0
                while line:
                    line = line.strip()
                    if line == "":
                        self.indices.append(position)
                        position = file.tell()
                    line = file.readline()

                self.total_sentence_count = len(self.indices)

            # option 2: keep everything in memory
            if self.in_memory:
                self.sentences: List[Sentence] = []

                while True:
                    sentence = self._read_next_sentence(file)
                    if not sentence:
                        break
                    self.sentences.append(sentence)

                self.total_sentence_count = len(self.sentences)

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        # if in memory, retrieve parsed sentence
        if self.in_memory:
            sentence = self.sentences[index]

        # else skip to position in file where sentence begins
        else:
            with open(str(self.path_to_conll_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                sentence = self._read_next_sentence(file)

        return sentence

    def _read_next_sentence(self, file):
        line = file.readline()
        sentence: Sentence = Sentence()

        # current token ID
        token_idx = 0

        # handling for the awful UD multiword format
        current_multiword_text = ''
        current_multiword_sequence = ''
        current_multiword_first_token = 0
        current_multiword_last_token = 0

        relation_tuples: List[Tuple[int, int, int, int, str]] = []

        while line:
            line = line.strip()
            fields: List[str] = re.split("\t+", line)

            # end of sentence
            if line == "":
                if len(sentence) > 0:
                    break

            # comments
            elif line.startswith("#"):
                line = file.readline()

                key_maybe_value = line[1:].split('=', 1)
                key = key_maybe_value[0].strip()
                value = None if len(key_maybe_value) == 1 else key_maybe_value[1].strip()

                if key == "relations":
                    for relation in value.split("|"):
                        relation_tuples.append(tuple(relation.split(";")))
                else:
                    continue

            # ellipsis
            elif "." in fields[0]:
                line = file.readline()
                continue

            # if token is a multi-word
            elif "-" in fields[0]:
                line = file.readline()

                current_multiword_first_token = int(fields[0].split('-')[0])
                current_multiword_last_token = int(fields[0].split('-')[1])
                current_multiword_text = fields[1]
                current_multiword_sequence = ''

                if self.split_multiwords:
                    continue
                else:
                    token = Token(fields[1])
                    token.add_label("ner", str(fields[2]))
                    # token.add_label("lemma", str(fields[2]))
                    # if len(fields) > 9 and 'SpaceAfter=No' in fields[9]:
                    #     token.whitespace_after = False
                    sentence.add_token(token)
                    token_idx += 1

            # normal single-word tokens
            else:

                # if we don't split multiwords, skip over component words
                if not self.split_multiwords and token_idx < current_multiword_last_token:
                    token_idx += 1
                    line = file.readline()
                    continue

                # add token
                # token = Token(fields[1], head_id=int(fields[6]))
                token = Token(fields[1])
                token.add_label("ner", str(fields[2]))
                # token.add_label("lemma", str(fields[2]))
                # token.add_label("upos", str(fields[3]))
                # token.add_label("pos", str(fields[4]))
                # token.add_label("dependency", str(fields[7]))

                # if len(fields) > 9 and 'SpaceAfter=No' in fields[9]:
                #     token.whitespace_after = False

                # add morphological tags
                # for morph in str(fields[5]).split("|"):
                #     if "=" not in morph:
                #         continue
                #     token.add_label(morph.split("=")[0].lower(), morph.split("=")[1])

                # if len(fields) > 10 and str(fields[10]) == "Y":
                #     token.add_label("frame", str(fields[11]))

                token_idx += 1

                # derive whitespace logic for multiwords
                if token_idx <= current_multiword_last_token:
                    current_multiword_sequence += token.text

                # print(token)
                # print(current_multiword_last_token)
                # print(current_multiword_first_token)
                # if multi-word equals component tokens, there should be no whitespace
                if token_idx == current_multiword_last_token and current_multiword_sequence == current_multiword_text:
                    # go through all tokens in subword and set whitespace_after information
                    for i in range(current_multiword_last_token - current_multiword_first_token):
                        # print(i)
                        sentence[-(i+1)].whitespace_after = False

                sentence.add_token(token)

            line = file.readline()

        if relation_tuples:
            relations: List[Relation] = []
            for head_start, head_end, tail_start, tail_end, label in relation_tuples:
                head = Span(sentence.tokens[int(head_start)-1:int(head_end)-1])
                tail = Span(sentence.tokens[int(tail_start)-1:int(tail_end)-1])
                relation = Relation(head, tail)
                relation.set_label("label", label)
                relations.append(relation)

            sentence.relations = relations

        return sentence


class SEMEVAL_2010_TASK_8(CoNLLUCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        semeval_2010_task_8_path = (
            "https://github.com/sahitya0000/Relation-Classification/raw/master/corpus/SemEval2010_task8_all_data.zip"
        )
        data_path = flair.cache_root / "datasets" / dataset_name
        data_file = data_path / "semeval2010-task8-train.conllu"
        if not data_file.is_file():
            cached_path(
                semeval_2010_task_8_path, Path("datasets") / dataset_name / "original"
            )
            self.download_and_prepare(data_file=flair.cache_root / "datasets" / dataset_name / "original" / "SemEval2010_task8_all_data.zip", data_folder=data_folder)

        super(SEMEVAL_2010_TASK_8, self).__init__(
            data_folder,
            in_memory=in_memory,
            split_multiwords=True
        )

    def download_and_prepare(self, data_file, data_folder):
        import zipfile

        source_file_paths = [
            "SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
            "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
        ]
        target_filenames = ["semeval2010-task8-train.conllu", "semeval2010-task8-test.conllu"]

        with zipfile.ZipFile(data_file) as zip_file:

            for source_file_path, target_filename in zip(source_file_paths, target_filenames):
                with zip_file.open(source_file_path, mode="r") as source_file:

                    target_file_path = Path(data_folder) / target_filename
                    with open(target_file_path, mode="w", encoding="utf-8") as target_file:
                        raw_lines = []
                        for line in io.TextIOWrapper(source_file, encoding="utf-8"):
                            line = line.strip()

                            if not line:
                                conllu_lines = self._raw_lines_to_conllu_lines(raw_lines)
                                target_file.writelines(conllu_lines)

                                raw_lines = []
                                continue

                            raw_lines.append(line)

    def _raw_lines_to_conllu_lines(self, raw_lines):
        raw_id, raw_text = raw_lines[0].split("\t")
        label = raw_lines[1]
        id_ = int(raw_id)
        raw_text = raw_text.strip('"')

        # Some special cases (e.g., missing spaces before entity marker)
        if id_ in [213, 4612, 6373, 8411, 9867]:
            raw_text = raw_text.replace("<e2>", " <e2>")
        if id_ in [2740, 4219, 4784]:
            raw_text = raw_text.replace("<e1>", " <e1>")
        if id_ == 9256:
            raw_text = raw_text.replace("log- jam", "log-jam")

        # necessary if text should be whitespace tokenizeable
        if id_ in [2609, 7589]:
            raw_text = raw_text.replace("1 1/2", "1-1/2")
        if id_ == 10591:
            raw_text = raw_text.replace("1 1/4", "1-1/4")
        if id_ == 10665:
            raw_text = raw_text.replace("6 1/2", "6-1/2")

        raw_text = re.sub(r"([.,!?()])$", r" \1", raw_text)
        raw_text = re.sub(r"(e[12]>)([',;:\"\(\)])", r"\1 \2", raw_text)
        raw_text = re.sub(r"([',;:\"\(\)])(</?e[12])", r"\1 \2", raw_text)
        raw_text = raw_text.replace("<e1>", "<e1> ")
        raw_text = raw_text.replace("<e2>", "<e2> ")
        raw_text = raw_text.replace("</e1>", " </e1>")
        raw_text = raw_text.replace("</e2>", " </e2>")

        tokens = raw_text.split(" ")

        # Handle case where tail may occur before the head
        head_start = tokens.index("<e1>")
        tail_start = tokens.index("<e2>")
        if head_start < tail_start:
            tokens.pop(head_start)
            head_end = tokens.index("</e1>")
            tokens.pop(head_end)
            tail_start = tokens.index("<e2>")
            tokens.pop(tail_start)
            tail_end = tokens.index("</e2>")
            tokens.pop(tail_end)
        else:
            tokens.pop(tail_start)
            tail_end = tokens.index("</e2>")
            tokens.pop(tail_end)
            head_start = tokens.index("<e1>")
            tokens.pop(head_start)
            head_end = tokens.index("</e1>")
            tokens.pop(head_end)
        
        if label == "Other":
            label = "N"

        lines = []
        lines.append(f"# text = {raw_text}\n")
        lines.append(f"# sentence_id = {id_}\n")
        lines.append(f"# relations = {head_start+1};{head_end+1};{tail_start+1};{tail_end+1};{label}\n")

        for idx, token in enumerate(tokens):
            tag = "O"
            prefix = ""

            if head_start <= idx < head_end:
                prefix = "B-" if idx == head_start else "I-"
                tag = "E1"
            elif tail_start <= idx < tail_end:
                prefix = "B-" if idx == tail_start else "I-"
                tag = "E2"

            lines.append(f"{idx+1}\t{token}\t{prefix}{tag}\n")

        lines.append("\n")

        return lines
