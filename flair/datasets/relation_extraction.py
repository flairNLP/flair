import logging
import re
import io
import os
from pathlib import Path
from typing import List, Union, Optional, Sequence, Dict, Any, Tuple

import flair
import json
import gdown
import conllu
from flair.file_utils import cached_path
from flair.datasets.conllu import CoNLLUCorpus

log = logging.getLogger("flair")


def convert_ptb_token(token: str) -> str:
    """Convert PTB tokens to normal tokens"""
    return {
        "-lrb-": "(",
        "-rrb-": ")",
        "-lsb-": "[",
        "-rsb-": "]",
        "-lcb-": "{",
        "-rcb-": "}",
    }.get(token.lower(), token)


class SEMEVAL_2010_TASK_8(CoNLLUCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, augment_train: bool = False):
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
        train_file_name = "semeval2010-task8-train-aug.conllu" if augment_train else "semeval2010-task8-train.conllu"
        data_file = data_folder / train_file_name

        # if True:
        if not data_file.is_file():
            source_data_folder = data_folder / "original"
            source_data_file = source_data_folder / "SemEval2010_task8_all_data.zip"
            os.makedirs(source_data_folder, exist_ok=True)
            gdown.download(semeval_2010_task_8_url, str(source_data_file))
            self.extract_and_convert_to_conllu(
                data_file=source_data_file,
                data_folder=data_folder,
                augment_train=augment_train,
            )

        super(SEMEVAL_2010_TASK_8, self).__init__(
            data_folder,
            train_file=train_file_name,
            test_file="semeval2010-task8-test.conllu",
            token_annotation_fields=['ner'],
            in_memory=in_memory,
        )

    def extract_and_convert_to_conllu(self, data_file, data_folder, augment_train):
        import zipfile

        source_file_paths = [
            "SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
            "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT",
        ]
        train_filename = "semeval2010-task8-train-aug.conllu" if augment_train else "semeval2010-task8-train.conllu"
        target_filenames = [train_filename, "semeval2010-task8-test.conllu"]

        with zipfile.ZipFile(data_file) as zip_file:

            for source_file_path, target_filename in zip(source_file_paths, target_filenames):
                with zip_file.open(source_file_path, mode="r") as source_file:

                    target_file_path = Path(data_folder) / target_filename
                    with open(target_file_path, mode="w", encoding="utf-8") as target_file:
                        # write CoNLL-U Plus header
                        target_file.write("# global.columns = id form ner\n")

                        raw_lines = []
                        for line in io.TextIOWrapper(source_file, encoding="utf-8"):
                            line = line.strip()

                            if not line:
                                token_list = self._semeval_lines_to_token_list(raw_lines,
                                                                               augment_relations=augment_train if "train" in target_filename else False)
                                target_file.write(token_list.serialize())

                                raw_lines = []
                                continue

                            raw_lines.append(line)

    def _semeval_lines_to_token_list(self, raw_lines, augment_relations):
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
        subj_start = tokens.index("<e1>")
        obj_start = tokens.index("<e2>")
        if subj_start < obj_start:
            tokens.pop(subj_start)
            subj_end = tokens.index("</e1>")
            tokens.pop(subj_end)
            obj_start = tokens.index("<e2>")
            tokens.pop(obj_start)
            obj_end = tokens.index("</e2>")
            tokens.pop(obj_end)
        else:
            tokens.pop(obj_start)
            obj_end = tokens.index("</e2>")
            tokens.pop(obj_end)
            subj_start = tokens.index("<e1>")
            tokens.pop(subj_start)
            subj_end = tokens.index("</e1>")
            tokens.pop(subj_end)

        relation = ";".join([str(subj_start + 1), str(subj_end), str(obj_start + 1), str(obj_end), label])

        if augment_relations:
            label_inverted = label.replace("e1", "e3")
            label_inverted = label_inverted.replace("e2", "e1")
            label_inverted = label_inverted.replace("e3", "e2")
            relation_inverted = ";".join([str(obj_start + 1), str(obj_end), str(subj_start + 1), str(subj_end), label_inverted])

        metadata = {
            "text": " ".join(tokens),
            "sentence_id": str(id_),
            "relations": relation + "|" + relation_inverted if augment_relations else relation,
        }

        token_dicts = []
        for idx, token in enumerate(tokens):
            tag = "O"
            prefix = ""

            if subj_start <= idx < subj_end:
                prefix = "B-" if idx == subj_start else "I-"
                tag = "E1"
            elif obj_start <= idx < obj_end:
                prefix = "B-" if idx == obj_start else "I-"
                tag = "E2"

            token_dicts.append(
                {
                    "id": str(idx + 1),
                    "form": token,
                    "ner": prefix + tag,
                }
            )

        return conllu.TokenList(tokens=token_dicts, metadata=metadata)


class TACRED(CoNLLUCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        data_file = data_folder / "tacred-train.conllu"

        if not data_file.is_file():
            source_data_folder = data_folder / "original"
            source_data_file = source_data_folder / "TACRED_LDC.zip"
            os.makedirs(source_data_folder, exist_ok=True)
            self.extract_and_convert_to_conllu(
                data_file=source_data_file,
                data_folder=data_folder,
            )

        super(TACRED, self).__init__(
            data_folder,
            token_annotation_fields=['ner'],
            in_memory=in_memory,
        )

    def extract_and_convert_to_conllu(self, data_file, data_folder):
        import zipfile

        source_file_paths = [
            "tacred/data/json/train.json",
            "tacred/data/json/dev.json",
            "tacred/data/json/test.json",
        ]
        target_filenames = ["tacred-train.conllu", "tacred-dev.conllu", "tacred-test.conllu"]

        with zipfile.ZipFile(data_file) as zip_file:

            for source_file_path, target_filename in zip(source_file_paths, target_filenames):
                with zip_file.open(source_file_path, mode="r") as source_file:

                    target_file_path = Path(data_folder) / target_filename
                    with open(target_file_path, mode="w", encoding="utf-8") as target_file:
                        # write CoNLL-U Plus header
                        target_file.write("# global.columns = id form ner\n")

                        for example in json.load(source_file):
                            token_list = self._tacred_example_to_token_list(example)
                            target_file.write(token_list.serialize())

    def _tacred_example_to_token_list(self, example: Dict[str, Any]) -> conllu.TokenList:
        id_ = example["id"]
        tokens = example["token"]
        ner = example["stanford_ner"]

        subj_start = example["subj_start"]
        subj_end = example["subj_end"]
        obj_start = example["obj_start"]
        obj_end = example["obj_end"]

        subj_tag = example["subj_type"]
        obj_tag = example["obj_type"]

        label = example["relation"]

        metadata = {
            "text": " ".join(tokens),
            "sentence_id": str(id_),
            "relations": ";".join(
                [str(subj_start + 1), str(subj_end + 1), str(obj_start + 1), str(obj_end + 1), label]
            ),
        }

        prev_tag = None
        token_dicts = []
        for idx, (token, tag) in enumerate(zip(tokens, ner)):
            if subj_start <= idx <= subj_end:
                tag = subj_tag

            if obj_start <= idx <= obj_end:
                tag = obj_tag

            prefix = ""
            if tag != "O":
                if tag != prev_tag:
                    prefix = "B-"
                else:
                    prefix = "I-"

            prev_tag = tag

            token_dicts.append(
                {
                    "id": str(idx + 1),
                    "form": convert_ptb_token(token),
                    "ner": prefix + tag,
                }
            )

        return conllu.TokenList(tokens=token_dicts, metadata=metadata)


class CoNLL04(CoNLLUCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # TODO: change data source to original CoNLL04 -- this dataset has span formatting errors
        # download data if necessary
        conll04_url = (
            "https://raw.githubusercontent.com/bekou/multihead_joint_entity_relation_extraction/master/data/CoNLL04/"
        )
        data_file = data_folder / "conll04-train.conllu"

        if True or not data_file.is_file():
            source_data_folder = data_folder / "original"
            cached_path(f"{conll04_url}train.txt", source_data_folder)
            cached_path(f"{conll04_url}dev.txt", source_data_folder)
            cached_path(f"{conll04_url}test.txt", source_data_folder)

            self.convert_to_conllu(
                source_data_folder=source_data_folder,
                data_folder=data_folder,
            )

        super(CoNLL04, self).__init__(
            data_folder,
            token_annotation_fields=['ner'],
            in_memory=in_memory,
        )

    def _parse_incr(self, source_file) -> Sequence[conllu.TokenList]:
        fields = ["id", "form", "ner", "relations", "relation_heads"]
        field_parsers = {
            "relations": lambda line, i: json.loads(line[i].replace("'", '"')),
            "relation_heads": lambda line, i: json.loads(line[i]),
        }
        metadata_parsers = {"__fallback__": lambda k, v: tuple(k.split())}

        lines = []
        for index, line in enumerate(source_file):
            if index > 0 and line.startswith("#"):
                source_str = "".join(lines)
                src_token_list = conllu.parse(
                    source_str, fields=fields, field_parsers=field_parsers, metadata_parsers=metadata_parsers
                )
                lines = []
                yield src_token_list[0]

            lines.append(line)

        source_str = "".join(lines)
        src_token_list = conllu.parse(
            source_str, fields=fields, field_parsers=field_parsers, metadata_parsers=metadata_parsers
        )
        yield src_token_list[0]

    def convert_to_conllu(self, source_data_folder, data_folder):
        source_filenames = [
            "train.txt",
            "dev.txt",
            "test.txt",
        ]
        target_filenames = ["conll04-train.conllu", "conll04-dev.conllu", "conll04-test.conllu"]

        for source_filename, target_filename in zip(source_filenames, target_filenames):
            with open(source_data_folder / source_filename, mode="r") as source_file:

                with open(data_folder / target_filename, mode="w", encoding="utf-8") as target_file:
                    # write CoNLL-U Plus header
                    target_file.write("# global.columns = id form ner\n")

                    for src_token_list in self._parse_incr(source_file):
                        token_list = self._src_token_list_to_token_list(src_token_list)
                        target_file.write(token_list.serialize())

    def _bio_tags_to_spans(self, tags: List[str]) -> List[Tuple[int, int]]:
        spans = []
        span_start = 0
        span_end = 0
        active_conll_tag = None
        for index, tag in enumerate(tags):
            bio_tag = tag[0]
            conll_tag = tag[2:]
            if bio_tag == "O":
                # The span has ended.
                if active_conll_tag is not None:
                    spans.append((span_start, span_end))
                active_conll_tag = None
                continue
            elif bio_tag == "B" or (bio_tag == "I" and conll_tag != active_conll_tag):
                # We are entering a new span; reset indices
                # and active tag to new span.
                if active_conll_tag is not None:
                    spans.append((span_start, span_end))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
            elif bio_tag == "I" and conll_tag == active_conll_tag:
                # We're inside a span.
                span_end += 1
            else:
                raise Exception("That should never happen.")

        # Last token might have been a part of a valid span.
        if active_conll_tag is not None:
            spans.append((span_start, span_end))

        return spans

    def _src_token_list_to_token_list(self, src_token_list):
        tokens = []
        token_dicts = []
        ner_tags = []
        for index, token in enumerate(src_token_list, start=1):
            text = token["form"]
            ner_tag = token["ner"]
            tokens.append(text)
            ner_tags.append(ner_tag)

            token_dicts.append(
                {
                    "id": str(index),
                    "form": text,
                    "ner": ner_tag,
                }
            )

        span_end_to_span = {end: (start, end) for start, end in self._bio_tags_to_spans(ner_tags)}

        relations = []
        for index, token in enumerate(src_token_list):
            for relation, head in zip(token["relations"], token["relation_heads"]):
                if relation == "N":
                    continue

                subj_start, subj_end = span_end_to_span[index]
                obj_start, obj_end = span_end_to_span[head]
                relations.append((subj_start, subj_end, obj_start, obj_end, relation))

        doc_id = src_token_list.metadata["doc"]

        metadata = {
            "text": " ".join(tokens),
            "sentence_id": doc_id,
            "relations": "|".join(
                [
                    ";".join([str(subj_start + 1), str(subj_end + 1), str(obj_start + 1), str(obj_end + 1), relation])
                    for subj_start, subj_end, obj_start, obj_end, relation in relations
                ]
            ),
        }

        return conllu.TokenList(tokens=token_dicts, metadata=metadata)
