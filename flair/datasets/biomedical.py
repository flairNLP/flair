import os
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Union, Callable, Dict, List, Tuple

from lxml import etree

import flair
from flair.datasets import ColumnCorpus
from flair.file_utils import cached_path, unzip_file


class Entity:
    def __init__(self, span: Tuple[int, int], entity_type: str):
        self.span = range(*span)
        self.type = entity_type


class InternalHunerDataset:
    def __init__(
        self, documents: Dict[str, str], entities_per_document: Dict[str, List[Entity]]
    ):
        self.documents = documents
        self.entities_per_document = entities_per_document

    def get_subset(self, split_path: Path):
        with split_path.open() as f:
            ids = {l.strip() for l in f if l.strip()}
            ids = sorted(id_ for id_ in ids if id_ in self.documents)

        return InternalHunerDataset(
            documents={k: self.documents[k] for k in ids},
            entities_per_document={k: self.entities_per_document[k] for k in ids},
        )


def overlap(entity1, entity2):
    return range(max(entity1[0], entity2[0]), min(entity1[1], entity2[1]))


def merge_overlapping_entities(entities):
    entities = list(entities)

    entity_set_stable = False
    while not entity_set_stable:
        for e1, e2 in combinations(entities, 2):
            if overlap(e1, e2):
                merged_entity = (min(e1[0], e2[0]), max(e1[1], e2[1]))
                entities.remove(e1)
                entities.remove(e2)
                entities.append(merged_entity)
                break
        else:
            entity_set_stable = True

    return entities


class CoNLLWriter:
    def __init__(
        self,
        tokenizer: Callable[[str], Tuple[List[str], List[int]]],
        sentence_splitter: Callable[[str], Tuple[List[str], List[int]]],
    ):
        """
        :param tokenizer: Callable that segments a sentence into words
        :param sentence_splitter: Callable that segments a document into sentences
        """
        self.tokenizer = tokenizer
        self.sentence_splitter = sentence_splitter

    def process_dataset(self, datasets: Dict[str, InternalHunerDataset], out_dir: Path):
        self.write_to_conll(datasets["train"], out_dir / "train.conll")
        self.write_to_conll(datasets["dev"], out_dir / "dev.conll")
        self.write_to_conll(datasets["test"], out_dir / "test.conll")

    def write_to_conll(self, dataset: InternalHunerDataset, output_file: Path):
        os.makedirs(output_file.parent, exist_ok=True)
        with output_file.open("w") as f:
            for document_id in dataset.documents.keys():

                document_text = dataset.documents[document_id]
                sentences, sentence_offsets = self.sentence_splitter(document_text)
                entities = dataset.entities_per_document[document_id]

                for sentence, sentence_offset in zip(sentences, sentence_offsets):
                    in_entity = False
                    tokens, token_offsets = self.tokenizer(sentence)
                    for token, token_offset in zip(tokens, token_offsets):
                        offset = sentence_offset + token_offset

                        for entity in entities:
                            if offset in entity.span:
                                if in_entity != entity:
                                    tag = "B-" + entity.type
                                    in_entity = entity
                                else:
                                    tag = "I-" + entity.type
                                break
                        else:
                            tag = "O"
                            in_entity = False

                        f.write(" ".join([token, tag]) + "\n")
                    f.write("\n")


class SciSpacyTokenizer:
    def __init__(self):
        import spacy

        self.nlp = spacy.load(
            "en_core_sci_sm", disable=["tagger", "ner", "parser", "textcat"]
        )

    def __call__(self, sentence: str):
        sentence = self.nlp(sentence)
        tokens = [str(tok) for tok in sentence]
        offsets = [tok.idx for tok in sentence]

        return tokens, offsets


class SciSpacySentenceSplitter:
    def __init__(self):
        import spacy

        self.nlp = spacy.load("en_core_sci_sm", disable=["tagger", "ner", "textcat"])

    def __call__(self, text: str):
        doc = self.nlp(text)
        sentences = [str(sent) for sent in doc.sents]
        offsets = [sent.start_char for sent in doc.sents]

        return sentences, offsets


class HunerDataset(ColumnCorpus, ABC):
    """
    Base class for HUNER Datasets

    Every subclass has to implement the following methods:
      - `to_internal', which produces a dictionary of InternalHUNERDatasets.
        This dictionary is composed as follows:
            dict['train'] -> train split
            dict['dev'] -> development split
            dict['test'] -> test split
    """

    @staticmethod
    @abstractmethod
    def to_internal(data_folder: Path) -> Dict[str, InternalHunerDataset]:
        pass

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        tokenizer: Callable[[str], Tuple[List[str], List[int]]] = None,
        sentence_splitter: Callable[[str], Tuple[List[str], List[int]]] = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param tokenizer: Callable that segments a sentence into words,
                          defaults to scispacy
        :param sentence_splitter: Callable that segments a document into sentences,
                                  defaults to scispacy
        """

        if tokenizer is None:
            try:
                import spacy

                tokenizer = SciSpacyTokenizer()
            except ImportError:
                raise ValueError(
                    "Default tokenizer is scispacy."
                    " Install packages 'scispacy' and"
                    " 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
                    "/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz' via pip"
                    " or choose a different tokenizer"
                )

        if sentence_splitter is None:
            try:
                import spacy

                sentence_splitter = SciSpacySentenceSplitter()
            except ImportError:
                raise ValueError(
                    "Default sentence splitter is scispacy."
                    " Install packages 'scispacy' and"
                    "'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
                    "/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz' via pip"
                    " or choose a different sentence splitter"
                )

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter,
            )
            internal_datasets = self.to_internal(data_folder)
            writer.process_dataset(internal_datasets, data_folder)

        super(HunerDataset, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )


class HunerProteinBioInfer(HunerDataset):
    @staticmethod
    def to_internal(data_dir: Path) -> Dict[str, InternalHunerDataset]:
        documents = {}
        entities_per_document = defaultdict(list)
        data_url = "http://mars.cs.utu.fi/BioInfer/files/BioInfer_corpus_1.1.1.zip"
        data_path = cached_path(data_url, data_dir)
        train_split = cached_path(
            "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bioinfer.train",
            data_dir,
        )
        dev_split = cached_path(
            "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bioinfer.dev",
            data_dir,
        )
        test_split = cached_path(
            "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bioinfer.test",
            data_dir,
        )
        unzip_file(data_path, data_dir)

        tree = etree.parse(str(data_dir / "BioInfer_corpus_1.1.1.xml"))
        sentence_elems = tree.xpath("//sentence")
        for sentence_id, sentence in enumerate(sentence_elems):
            sentence_id = str(sentence_id)
            token_ids = []
            token_offsets = []
            sentence_text = ""

            all_entity_token_ids = []
            entities = (
                sentence.xpath(".//entity[@type='Individual_protein']")
                + sentence.xpath(".//entity[@type='Gene/protein/RNA']")
                + sentence.xpath(".//entity[@type='Gene']")
                + sentence.xpath(".//entity[@type='DNA_family_or_group']")
            )
            for entity in entities:
                valid_entity = True
                entity_token_ids = set()
                for subtoken in entity.xpath(".//nestedsubtoken"):
                    token_id = ".".join(subtoken.attrib["id"].split(".")[1:3])
                    entity_token_ids.add(token_id)

                if valid_entity:
                    all_entity_token_ids.append(entity_token_ids)

            for token in sentence.xpath(".//token"):
                token_text = "".join(token.xpath(".//subtoken/@text"))
                token_id = ".".join(token.attrib["id"].split(".")[1:])
                token_ids.append(token_id)

                if not sentence_text:
                    token_offsets.append(0)
                    sentence_text = token_text
                else:
                    token_offsets.append(len(sentence_text) + 1)
                    sentence_text += " " + token_text

            documents[sentence_id] = sentence_text

            for entity_token_ids in all_entity_token_ids:
                entity_start = None
                for token_idx, (token_id, token_offset) in enumerate(
                    zip(token_ids, token_offsets)
                ):
                    if token_id in entity_token_ids:
                        if entity_start is None:
                            entity_start = token_offset
                    else:
                        if entity_start is not None:
                            entities_per_document[sentence_id].append(
                                Entity((entity_start, token_offset - 1), "Protein")
                            )
                            entity_start = None
        dataset = InternalHunerDataset(
            documents=documents, entities_per_document=entities_per_document
        )
        train_dataset = dataset.get_subset(train_split)
        dev_dataset = dataset.get_subset(dev_split)
        test_dataset = dataset.get_subset(test_split)
        return {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
