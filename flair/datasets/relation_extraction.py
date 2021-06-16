import logging
import re
import io
import os
from pathlib import Path
from typing import List, Union, Optional, Sequence, Dict

import flair
import gdown
import conllu
from flair.data import Sentence, Corpus, Token, FlairDataset, Relation, Span
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path
from flair.datasets.conllu import CoNLLUCorpus

log = logging.getLogger("flair")


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
        semeval_2010_task_8_url = (
            "https://drive.google.com/uc?id=0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk"
        )
        data_file = data_folder / "semeval2010-task8-train.conllu"

        if not data_file.is_file():
            source_data_folder = data_folder / "original"
            source_data_file = source_data_folder / "SemEval2010_task8_all_data.zip"
            os.makedirs(source_data_folder, exist_ok=True)
            gdown.download(semeval_2010_task_8_url, str(source_data_file))
            self.extract_and_convert_to_conllu(
                data_file=source_data_file,
                data_folder=data_folder,
            )

        super(SEMEVAL_2010_TASK_8, self).__init__(
            data_folder,
            in_memory=in_memory,
        )

    def extract_and_convert_to_conllu(self, data_file, data_folder):
        import zipfile

        source_file_paths = [
            "SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
            "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT",
        ]
        target_filenames = ["semeval2010-task8-train.conllu", "semeval2010-task8-test.conllu"]

        with zipfile.ZipFile(data_file) as zip_file:

            for source_file_path, target_filename in zip(source_file_paths, target_filenames):
                with zip_file.open(source_file_path, mode="r") as source_file:

                    target_file_path = Path(data_folder) / target_filename
                    with open(target_file_path, mode="w", encoding="utf-8") as target_file:
                        # write CoNLL Plus header
                        target_file.write("# global.columns = id form ner\n")

                        raw_lines = []
                        for line in io.TextIOWrapper(source_file, encoding="utf-8"):
                            line = line.strip()

                            if not line:
                                token_list = self._semeval_lines_to_token_list(raw_lines)
                                target_file.write(token_list.serialize())

                                raw_lines = []
                                continue

                            raw_lines.append(line)

    def _semeval_lines_to_token_list(self, raw_lines):
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

        metadata = {
            "text": " ".join(tokens),
            "sentence_id": str(id_),
            "relations": ";".join([str(head_start + 1), str(head_end), str(tail_start + 1), str(tail_end), label]),
        }

        token_dicts = []
        for idx, token in enumerate(tokens):
            tag = "O"
            prefix = ""

            if head_start <= idx < head_end:
                prefix = "B-" if idx == head_start else "I-"
                tag = "E1"
            elif tail_start <= idx < tail_end:
                prefix = "B-" if idx == tail_start else "I-"
                tag = "E2"

            token_dicts.append(
                {
                    "id": str(idx + 1),
                    "form": token,
                    "ner": prefix + tag,
                }
            )

        return conllu.TokenList(tokens=token_dicts, metadata=metadata)
