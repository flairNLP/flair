import inspect
import json
import logging
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from copy import copy
from operator import attrgetter
from pathlib import Path
from tarfile import (
    CompressionError,
    ExtractError,
    HeaderError,
    ReadError,
    StreamError,
    TarError,
)
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Union
from warnings import warn
from zipfile import BadZipFile, LargeZipFile

import ftfy
from lxml import etree
from lxml.etree import XMLSyntaxError

import flair
from flair.data import MultiCorpus, Tokenizer
from flair.datasets.sequence_labeling import ColumnCorpus, ColumnDataset
from flair.file_utils import Tqdm, cached_path, unpack_file
from flair.tokenization import (
    NewlineSentenceSplitter,
    NoSentenceSplitter,
    SciSpacySentenceSplitter,
    SciSpacyTokenizer,
    SentenceSplitter,
    SpaceTokenizer,
    TagSentenceSplitter,
)

DISEASE_TAG = "Disease"
CHEMICAL_TAG = "Chemical"
CELL_LINE_TAG = "CellLine"
GENE_TAG = "Gene"
SPECIES_TAG = "Species"

SENTENCE_TAG = "[__SENT__]"

logger = logging.getLogger("flair")


class Entity:
    """
    Internal class to represent entities while converting biomedical NER corpora to a standardized format
    (only used for pre-processing purposes!). Each entity consists of the char span it addresses in
    the original text as well as the type of entity (e.g. Chemical, Gene, and so on).
    """

    def __init__(self, char_span: Tuple[int, int], entity_type: str):
        assert char_span[0] < char_span[1]
        self.char_span = range(*char_span)
        self.type = entity_type

    def __str__(self):
        return self.type + "(" + str(self.char_span.start) + "," + str(self.char_span.stop) + ")"

    def __repr__(self):
        return str(self)

    def is_before(self, other_entity) -> bool:
        """
        Checks whether this entity is located before the given one

        :param other_entity: Entity to check
        """
        return self.char_span.stop <= other_entity.char_span.start

    def contains(self, other_entity) -> bool:
        """
        Checks whether the given entity is fully contained in this entity

        :param other_entity: Entity to check
        """
        return (
            other_entity.char_span.start >= self.char_span.start and other_entity.char_span.stop <= self.char_span.stop
        )

    def overlaps(self, other_entity) -> bool:
        """
        Checks whether this and the given entity overlap

        :param other_entity: Entity to check
        """
        return (self.char_span.start <= other_entity.char_span.start < self.char_span.stop) or (
            self.char_span.start < other_entity.char_span.stop <= self.char_span.stop
        )


class InternalBioNerDataset:
    """
    Internal class to represent a corpus and it's entities.
    """

    def __init__(self, documents: Dict[str, str], entities_per_document: Dict[str, List[Entity]]):
        self.documents = documents
        self.entities_per_document = entities_per_document


class DpEntry(NamedTuple):
    position_end: int
    entity_count: int
    entity_lengths_sum: int
    last_entity: Optional[Entity]


def merge_datasets(data_sets: Iterable[InternalBioNerDataset]):
    all_documents = {}
    all_entities = {}

    for ds in data_sets:
        all_documents.update(ds.documents)
        all_entities.update(ds.entities_per_document)

    return InternalBioNerDataset(documents=all_documents, entities_per_document=all_entities)


def filter_and_map_entities(
    dataset: InternalBioNerDataset, entity_type_to_canonical: Dict[str, str]
) -> InternalBioNerDataset:
    """
    :param entity_type_to_canonical: Maps entity type in dataset to canonical type
                                     if entity type is not present in map it is discarded
    """
    mapped_entities_per_document = {}
    for id, entities in dataset.entities_per_document.items():
        new_entities = []
        for entity in entities:
            if entity.type in entity_type_to_canonical:
                new_entity = copy(entity)
                new_entity.type = entity_type_to_canonical[entity.type]
                new_entities.append(new_entity)
            else:
                logging.debug(f"Skip entity type {entity.type}")
                pass
        mapped_entities_per_document[id] = new_entities

    return InternalBioNerDataset(documents=dataset.documents, entities_per_document=mapped_entities_per_document)


def filter_nested_entities(dataset: InternalBioNerDataset) -> None:
    num_entities_before = sum([len(x) for x in dataset.entities_per_document.values()])

    for document_id, entities in dataset.entities_per_document.items():
        # Uses dynamic programming approach to calculate maximum independent set in interval graph
        # with sum of all entity lengths as secondary key
        dp_array = [DpEntry(position_end=0, entity_count=0, entity_lengths_sum=0, last_entity=None)]
        for entity in sorted(entities, key=lambda x: x.char_span.stop):
            i = len(dp_array) - 1
            while dp_array[i].position_end > entity.char_span.start:
                i -= 1
            if dp_array[i].entity_count + 1 > dp_array[-1].entity_count or (
                dp_array[i].entity_count + 1 == dp_array[-1].entity_count
                and dp_array[i].entity_lengths_sum + len(entity.char_span) > dp_array[-1].entity_lengths_sum
            ):
                dp_array += [
                    DpEntry(
                        entity.char_span.stop,
                        dp_array[i].entity_count + 1,
                        dp_array[i].entity_lengths_sum + len(entity.char_span),
                        entity,
                    )
                ]
            else:
                dp_array += [dp_array[-1]]

        independent_set = []
        p = dp_array[-1].position_end
        for dp_entry in dp_array[::-1]:
            if dp_entry.last_entity is None:
                break
            if dp_entry.position_end <= p:
                independent_set += [dp_entry.last_entity]
                p -= len(dp_entry.last_entity.char_span)

        dataset.entities_per_document[document_id] = independent_set

    num_entities_after = sum([len(x) for x in dataset.entities_per_document.values()])
    if num_entities_before != num_entities_after:
        removed = num_entities_before - num_entities_after
        warn(f"Corpus modified by filtering nested entities. Removed {removed} entities.")


def bioc_to_internal(bioc_file: Path):
    """
    Helper function to parse corpora that are given in BIOC format. See

        http://bioc.sourceforge.net/

    for details.
    """
    tree = etree.parse(str(bioc_file))
    texts_per_document = {}
    entities_per_document = {}
    documents = tree.xpath(".//document")

    all_entities = 0
    non_matching = 0

    for document in Tqdm.tqdm(documents, desc="Converting to internal"):
        document_id = document.xpath("./id")[0].text
        texts: List[str] = []
        entities = []

        for passage in document.xpath("passage"):
            passage_texts = passage.xpath("text/text()")
            if len(passage_texts) == 0:
                continue
            text = passage_texts[0]

            passage_offset = int(passage.xpath("./offset/text()")[0])  # from BioC annotation

            # calculate offset without current text
            # because we stick all passages of a document together
            document_text = " ".join(texts)
            document_offset = len(document_text)

            texts.append(text)
            document_text += " " + text

            for annotation in passage.xpath(".//annotation"):

                entity_types = [
                    i.text.replace(" ", "_")
                    for i in annotation.xpath("./infon")
                    if i.attrib["key"] in {"type", "class"}
                ]

                start = int(annotation.xpath("./location")[0].get("offset")) - passage_offset
                # TODO For split entities we also annotate everything inbetween which might be a bad idea?
                final_length = int(annotation.xpath("./location")[-1].get("length"))
                final_offset = int(annotation.xpath("./location")[-1].get("offset")) - passage_offset
                if final_length <= 0:
                    continue
                end = final_offset + final_length

                start += document_offset
                end += document_offset

                true_entity = annotation.xpath(".//text")[0].text
                annotated_entity = " ".join(texts)[start:end]

                # Try to fix incorrect annotations
                if annotated_entity.lower() != true_entity.lower():
                    max_shift = min(3, len(true_entity))
                    for i in range(max_shift):
                        index = annotated_entity.lower().find(true_entity[0 : max_shift - i].lower())
                        if index != -1:
                            start += index
                            end += index
                            break

                annotated_entity = " ".join(texts)[start:end]
                if not annotated_entity.lower() == true_entity.lower():
                    non_matching += 1

                all_entities += 1

                for entity_type in entity_types:
                    entities.append(Entity((start, end), entity_type))

        texts_per_document[document_id] = " ".join(texts)
        entities_per_document[document_id] = entities

    # print(
    #     f"Found {non_matching} non-matching entities ({non_matching/all_entities}%) in {bioc_file}"
    # )

    return InternalBioNerDataset(documents=texts_per_document, entities_per_document=entities_per_document)


def brat_to_internal(corpus_dir: Path, ann_file_suffixes=None) -> InternalBioNerDataset:
    """
    Helper function to parse corpora that are annotated using BRAT. See

        https://brat.nlplab.org/

    for details.

    """
    if ann_file_suffixes is None:
        ann_file_suffixes = [".ann"]

    text_files = list(corpus_dir.glob("*.txt"))
    documents = {}
    entities_per_document = defaultdict(list)
    for text_file in text_files:
        document_text = open(str(text_file), encoding="utf8").read().strip()
        document_id = text_file.stem

        for suffix in ann_file_suffixes:
            with open(str(text_file.with_suffix(suffix)), "r", encoding="utf8") as ann_file:
                for line in ann_file:
                    fields = line.strip().split("\t")

                    # Ignore empty lines or relation annotations
                    if not fields or len(fields) <= 2:
                        continue

                    ent_type, char_start, char_end = fields[1].split()
                    start = int(char_start)
                    end = int(char_end)

                    # FIX annotation of whitespaces (necessary for PDR)
                    while document_text[start:end].startswith(" "):
                        start += 1

                    while document_text[start:end].endswith(" "):
                        end -= 1

                    entities_per_document[document_id].append(
                        Entity(
                            char_span=(start, end),
                            entity_type=ent_type,
                        )
                    )

                    assert document_text[start:end].strip() == fields[2].strip()

        documents[document_id] = document_text

    return InternalBioNerDataset(documents=documents, entities_per_document=dict(entities_per_document))


class CoNLLWriter:
    """
    Class which implements the output CONLL file generation of corpora given as instances of
    :class:`InternalBioNerDataset`.
    """

    def __init__(
        self,
        sentence_splitter: SentenceSplitter,
    ):
        """
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which
        segments the text into sentences and tokens
        """
        self.sentence_splitter = sentence_splitter

    def process_dataset(self, datasets: Dict[str, InternalBioNerDataset], out_dir: Path):
        self.write_to_conll(datasets["train"], out_dir / "train.conll")
        self.write_to_conll(datasets["dev"], out_dir / "dev.conll")
        self.write_to_conll(datasets["test"], out_dir / "test.conll")

    def write_to_conll(self, dataset: InternalBioNerDataset, output_file: Path):
        os.makedirs(str(output_file.parent), exist_ok=True)
        filter_nested_entities(dataset)

        with output_file.open("w", encoding="utf8") as f:
            for document_id in Tqdm.tqdm(
                dataset.documents.keys(),
                total=len(dataset.documents),
                desc="Converting to CoNLL",
            ):
                document_text = ftfy.fix_text(dataset.documents[document_id])
                document_text = re.sub(r"[\u2000-\u200B]", " ", document_text)  # replace unicode space characters!
                document_text = document_text.replace("\xa0", " ")  # replace non-break space

                entities = deque(
                    sorted(
                        dataset.entities_per_document[document_id],
                        key=attrgetter("char_span.start", "char_span.stop"),
                    )
                )
                current_entity = entities.popleft() if entities else None

                sentences = self.sentence_splitter.split(document_text)

                for sentence in sentences:
                    in_entity = False
                    sentence_had_tokens = False

                    for flair_token in sentence.tokens:
                        token = flair_token.text.strip()
                        assert sentence.start_pos is not None
                        assert flair_token.start_pos is not None
                        offset = sentence.start_pos + flair_token.start_pos

                        if current_entity and offset >= current_entity.char_span.stop:
                            in_entity = False

                            # One token may contain multiple entities -> deque all of them
                            while current_entity and offset >= current_entity.char_span.stop:
                                current_entity = entities.popleft() if entities else None

                        if current_entity and offset in current_entity.char_span:
                            if not in_entity:
                                tag = "B-" + current_entity.type
                                in_entity = True
                            else:
                                tag = "I-" + current_entity.type
                        else:
                            tag = "O"
                            in_entity = False

                        whitespace_after = "+" if flair_token.whitespace_after > 0 else "-"
                        if len(token) > 0:
                            f.write(" ".join([token, tag, whitespace_after]) + "\n")
                            sentence_had_tokens = True

                    if sentence_had_tokens:
                        f.write("\n")


class HunerDataset(ColumnCorpus, ABC):
    """
    Base class for HUNER datasets.

    Every subclass has to implement the following methods:
      - `to_internal', which reads the complete data set (incl. train, dev, test) and returns the corpus
        as InternalBioNerDataset
      - `split_url', which returns the base url (i.e. without '.train', '.dev', '.test') to the HUNER split files

    For further information see:
      - Weber et al.: 'HUNER: improving biomedical NER with pretraining'
        https://academic.oup.com/bioinformatics/article-abstract/36/1/295/5523847?redirectedFrom=fulltext
      - HUNER github repository:
        https://github.com/hu-ner/huner
    """

    @abstractmethod
    def to_internal(self, data_folder: Path) -> InternalBioNerDataset:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def split_url() -> str:
        raise NotImplementedError()

    def get_corpus_sentence_splitter(self) -> Optional[SentenceSplitter]:
        """
        If the corpus has a pre-defined sentence splitting, then this method returns
        the sentence splitter to be used to reconstruct the original splitting.
        If the corpus has no pre-defined sentence splitting None will be returned.
        """
        return None

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Custom implementation of :class:`SentenceSplitter` which
            segments the text into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        self.sentence_splitter = self.get_corpus_sentence_splitter()
        if not self.sentence_splitter:
            self.sentence_splitter = sentence_splitter if sentence_splitter else SciSpacySentenceSplitter()
        else:
            if sentence_splitter:
                warn(
                    f"The corpus {self.__class__.__name__} has a pre-defined sentence splitting, "
                    f"thus just the tokenizer of the given sentence splitter ist used"
                )
                self.sentence_splitter.tokenizer = sentence_splitter.tokenizer

        # Create tokenization-dependent CONLL files. This is necessary to prevent
        # from caching issues (e.g. loading the same corpus with different sentence splitters)
        train_file = data_folder / f"{self.sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{self.sentence_splitter.name}_dev.conll"
        test_file = data_folder / f"{self.sentence_splitter.name}_test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            splits_dir = data_folder / "splits"
            os.makedirs(splits_dir, exist_ok=True)

            writer = CoNLLWriter(sentence_splitter=self.sentence_splitter)
            internal_dataset = self.to_internal(data_folder)

            train_data = self.get_subset(internal_dataset, "train", splits_dir)
            writer.write_to_conll(train_data, train_file)

            dev_data = self.get_subset(internal_dataset, "dev", splits_dir)
            writer.write_to_conll(dev_data, dev_file)

            test_data = self.get_subset(internal_dataset, "test", splits_dir)
            writer.write_to_conll(test_data, test_file)

        super(HunerDataset, self).__init__(
            data_folder=data_folder,
            train_file=train_file.name,
            dev_file=dev_file.name,
            test_file=test_file.name,
            column_format=columns,
            tag_to_bioes="ner",
            in_memory=in_memory,
        )

    def get_subset(self, dataset: InternalBioNerDataset, split: str, split_dir: Path):
        split_file = cached_path(f"{self.split_url()}.{split}", split_dir)

        with split_file.open(encoding="utf8") as f:
            ids = [line.strip() for line in f if line.strip()]
            ids = sorted(id_ for id_ in ids if id_ in dataset.documents)

        return InternalBioNerDataset(
            documents={k: dataset.documents[k] for k in ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in ids},
        )


class BIO_INFER(ColumnCorpus):
    """
    Original BioInfer corpus

    For further information see Pyysalo et al.:
       BioInfer: a corpus for information extraction in the biomedical domain
       https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-50
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"

        if not (train_file.exists()):
            corpus_folder = self.download_dataset(data_folder)
            corpus_data = self.parse_dataset(corpus_folder)

            sentence_splitter = NoSentenceSplitter(tokenizer=SpaceTokenizer())

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(corpus_data, train_file)

        super(BIO_INFER, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_dataset(cls, data_dir: Path) -> Path:
        data_url = "http://mars.cs.utu.fi/BioInfer/files/BioInfer_corpus_1.1.1.zip"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

        return data_dir / "BioInfer_corpus_1.1.1.xml"

    @classmethod
    def parse_dataset(cls, original_file: Path):
        documents: Dict[str, str] = {}
        entities_per_document: Dict[str, List[Entity]] = {}

        tree = etree.parse(str(original_file))
        sentence_elems = tree.xpath("//sentence")
        for s_id, sentence in enumerate(sentence_elems):
            sentence_id = str(s_id)
            token_id_to_span = {}
            sentence_text = ""
            entities_per_document[sentence_id] = []

            for token in sentence.xpath(".//token"):
                token_text = "".join(token.xpath(".//subtoken/@text"))
                token_id = ".".join(token.attrib["id"].split(".")[1:])

                if not sentence_text:
                    token_id_to_span[token_id] = (0, len(token_text))
                    sentence_text = token_text
                else:
                    token_id_to_span[token_id] = (
                        len(sentence_text) + 1,
                        len(token_text) + len(sentence_text) + 1,
                    )
                    sentence_text += " " + token_text
            documents[sentence_id] = sentence_text

            entities = [
                e for e in sentence.xpath(".//entity") if not e.attrib["type"].isupper()
            ]  # all caps entity type apparently marks event trigger

            for entity in entities:
                token_nums = []
                entity_character_starts = []
                entity_character_ends = []

                for subtoken in entity.xpath(".//nestedsubtoken"):
                    token_id_parts = subtoken.attrib["id"].split(".")
                    token_id = ".".join(token_id_parts[1:3])

                    token_nums.append(int(token_id_parts[2]))
                    entity_character_starts.append(token_id_to_span[token_id][0])
                    entity_character_ends.append(token_id_to_span[token_id][1])

                if token_nums and entity_character_starts and entity_character_ends:
                    entity_tokens = list(zip(token_nums, entity_character_starts, entity_character_ends))

                    start_token = entity_tokens[0]
                    last_entity_token = entity_tokens[0]
                    for entity_token in entity_tokens[1:]:
                        if not (entity_token[0] - 1) == last_entity_token[0]:
                            entities_per_document[sentence_id].append(
                                Entity(
                                    char_span=(start_token[1], last_entity_token[2]),
                                    entity_type=entity.attrib["type"],
                                )
                            )
                            start_token = entity_token

                        last_entity_token = entity_token

                    if start_token:
                        entities_per_document[sentence_id].append(
                            Entity(
                                char_span=(start_token[1], last_entity_token[2]),
                                entity_type=entity.attrib["type"],
                            )
                        )

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_GENE_BIO_INFER(HunerDataset):
    """
    HUNER version of the BioInfer corpus containing only gene/protein annotations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bioinfer"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        original_file = BIO_INFER.download_dataset(data_dir)
        corpus = BIO_INFER.parse_dataset(original_file)

        entity_type_mapping = {
            "Individual_protein": GENE_TAG,
            "Gene/protein/RNA": GENE_TAG,
            "Gene": GENE_TAG,
            "DNA_family_or_group": GENE_TAG,
        }

        return filter_and_map_entities(corpus, entity_type_mapping)


class JNLPBA(ColumnCorpus):
    """
    Original corpus of the JNLPBA shared task.

    For further information see Kim et al.:
      Introduction to the Bio-Entity Recognition Task at JNLPBA
      https://www.aclweb.org/anthology/W04-1213.pdf
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and test_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)

            train_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz"
            train_data_path = cached_path(train_data_url, download_dir)
            unpack_file(train_data_path, download_dir)

            train_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz"
            train_data_path = cached_path(train_data_url, download_dir)
            unpack_file(train_data_path, download_dir)

            train_file = download_dir / "Genia4ERtask2.iob2"
            shutil.copy(train_file, data_folder / "train.conll")

            test_file = download_dir / "Genia4EReval2.iob2"
            shutil.copy(test_file, data_folder / "test.conll")

        super(JNLPBA, self).__init__(
            data_folder,
            columns,
            tag_to_bioes="ner",
            in_memory=in_memory,
            comment_symbol="#",
        )


class HunerJNLPBA(object):
    @classmethod
    def download_and_prepare_train(cls, data_folder: Path, sentence_tag: str) -> InternalBioNerDataset:
        train_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz"
        train_data_path = cached_path(train_data_url, data_folder)
        unpack_file(train_data_path, data_folder)

        train_input_file = data_folder / "Genia4ERtask2.iob2"
        return cls.read_file(train_input_file, sentence_tag)

    @classmethod
    def download_and_prepare_test(cls, data_folder: Path, sentence_tag: str) -> InternalBioNerDataset:
        test_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz"
        test_data_path = cached_path(test_data_url, data_folder)
        unpack_file(test_data_path, data_folder)

        test_input_file = data_folder / "Genia4EReval2.iob2"
        return cls.read_file(test_input_file, sentence_tag)

    @classmethod
    def read_file(cls, input_iob_file: Path, sentence_tag: str) -> InternalBioNerDataset:
        documents: Dict[str, str] = {}
        entities_per_document: Dict[str, List[Entity]] = defaultdict(list)

        with open(str(input_iob_file), "r", encoding="utf8") as file_reader:
            document_id: Optional[str] = None
            document_text: Optional[str] = None

            entities: List[Entity] = []
            entity_type: Optional[str] = None
            entity_start = 0

            for line in file_reader:
                line = line.strip()
                if line[:3] == "###":
                    if not (document_id is None or document_text is None):
                        documents[document_id] = document_text
                        entities_per_document[document_id] = entities

                    document_id = line.split(":")[-1]
                    document_text = None

                    entities = []
                    entity_type = None
                    entity_start = 0

                    file_reader.__next__()
                    continue

                if line:
                    parts = line.split()
                    token = parts[0].strip()
                    tag = parts[1].strip()

                    if tag.startswith("B-"):
                        if entity_type is not None and document_text is not None:
                            entities.append(Entity((entity_start, len(document_text)), entity_type))

                        entity_start = len(document_text) + 1 if document_text else 0
                        entity_type = tag[2:]

                    elif tag == "O" and entity_type is not None and document_text is not None:
                        entities.append(Entity((entity_start, len(document_text)), entity_type))
                        entity_type = None

                    document_text = (document_text + " " + token) if document_text is not None else token

                else:
                    if document_text is not None:
                        document_text += sentence_tag

                        # Edge case: last token starts a new entity
                        if entity_type is not None:
                            entities.append(Entity((entity_start, len(document_text)), entity_type))

            # Last document in file
            if not (document_id is None or document_text is None):
                documents[document_id] = document_text
                entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_GENE_JNLPBA(HunerDataset):
    """
    HUNER version of the JNLPBA corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/genia"

    def get_corpus_sentence_splitter(self) -> SentenceSplitter:
        return TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        orig_folder = data_dir / "original"
        os.makedirs(str(orig_folder), exist_ok=True)

        sentence_separator = " "
        if isinstance(self.sentence_splitter, TagSentenceSplitter):
            sentence_separator = self.sentence_splitter.tag

        train_data = HunerJNLPBA.download_and_prepare_train(orig_folder, sentence_separator)
        train_data = filter_and_map_entities(train_data, {"protein": GENE_TAG})

        test_data = HunerJNLPBA.download_and_prepare_test(orig_folder, sentence_separator)
        test_data = filter_and_map_entities(test_data, {"protein": GENE_TAG})

        return merge_datasets([train_data, test_data])


class HUNER_CELL_LINE_JNLPBA(HunerDataset):
    """
    HUNER version of the JNLPBA corpus containing cell line annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/genia"

    def get_corpus_sentence_splitter(self) -> SentenceSplitter:
        return TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        sentence_separator = " "
        if isinstance(self.sentence_splitter, TagSentenceSplitter):
            sentence_separator = self.sentence_splitter.tag

        train_data = HunerJNLPBA.download_and_prepare_train(download_folder, sentence_separator)
        train_data = filter_and_map_entities(train_data, {"cell_line": CELL_LINE_TAG})

        test_data = HunerJNLPBA.download_and_prepare_test(download_folder, sentence_separator)
        test_data = filter_and_map_entities(test_data, {"cell_line": CELL_LINE_TAG})

        return merge_datasets([train_data, test_data])


class CELL_FINDER(ColumnCorpus):
    """
    Original CellFinder corpus containing cell line, species and gene annotations.

    For futher information see Neves et al.:
        Annotating and evaluating text for stem cell research
        https://pdfs.semanticscholar.org/38e3/75aeeeb1937d03c3c80128a70d8e7a74441f.pdf
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Custom implementation of :class:`SentenceSplitter` which segments
            the text into sentences and tokens.
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        data_folder = base_path / dataset_name

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        if not (train_file.exists()):
            train_corpus = self.download_and_prepare(data_folder)

            writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            writer.write_to_conll(train_corpus, train_file)

        super(CELL_FINDER, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_and_prepare(cls, data_folder: Path) -> InternalBioNerDataset:
        data_url = (
            "https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/cellfinder1_brat.tar.gz"
        )
        data_path = cached_path(data_url, data_folder)
        unpack_file(data_path, data_folder)

        return cls.read_folder(data_folder)

    @classmethod
    def read_folder(cls, data_folder: Path) -> InternalBioNerDataset:
        ann_files = list(data_folder.glob("*.ann"))
        documents = {}
        entities_per_document = defaultdict(list)
        for ann_file in ann_files:
            with ann_file.open(encoding="utf8") as f_ann, ann_file.with_suffix(".txt").open(encoding="utf8") as f_txt:
                document_text = f_txt.read().strip()

                document_id = ann_file.stem
                documents[document_id] = document_text

                for line in f_ann:
                    fields = line.strip().split("\t")
                    if not fields:
                        continue
                    ent_type, char_start, char_end = fields[1].split()
                    entities_per_document[document_id].append(
                        Entity(
                            char_span=(int(char_start), int(char_end)),
                            entity_type=ent_type,
                        )
                    )

                    assert document_text[int(char_start) : int(char_end)] == fields[2]

        return InternalBioNerDataset(documents=documents, entities_per_document=dict(entities_per_document))


class HUNER_CELL_LINE_CELL_FINDER(HunerDataset):
    """
    HUNER version of the CellFinder corpus containing only cell line annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/cellfinder_cellline"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        data = CELL_FINDER.download_and_prepare(data_dir)
        data = filter_and_map_entities(data, {"CellLine": CELL_LINE_TAG})

        return data


class HUNER_SPECIES_CELL_FINDER(HunerDataset):
    """
    HUNER version of the CellFinder corpus containing only species annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/cellfinder_species"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        data = CELL_FINDER.download_and_prepare(data_dir)
        data = filter_and_map_entities(data, {"Species": SPECIES_TAG})

        return data


class HUNER_GENE_CELL_FINDER(HunerDataset):
    """
    HUNER version of the CellFinder corpus containing only gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/cellfinder_protein"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        data = CELL_FINDER.download_and_prepare(data_dir)
        data = filter_and_map_entities(data, {"GeneProtein": GENE_TAG})

        return data


class MIRNA(ColumnCorpus):
    """
    Original miRNA corpus.

    For further information see Bagewadi et al.:
        Detecting miRNA Mentions and Relations in Biomedical Literature
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4602280/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param tokenizer: Callable that segments a sentence into words,
                          defaults to scispacy
        :param sentence_splitter: Callable that segments a document into sentences,
                                  defaults to scispacy
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        sentence_separator = " "
        if sentence_splitter is None:
            sentence_separator = SENTENCE_TAG
            sentence_splitter = TagSentenceSplitter(tag=sentence_separator, tokenizer=SciSpacyTokenizer())

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        test_file = data_folder / f"{sentence_splitter.name}_test.conll"

        if not (train_file.exists() and test_file.exists()):
            download_folder = data_folder / "original"
            os.makedirs(str(download_folder), exist_ok=True)

            writer = CoNLLWriter(sentence_splitter=sentence_splitter)

            train_corpus = self.download_and_prepare_train(download_folder, sentence_separator)
            writer.write_to_conll(train_corpus, train_file)

            test_corpus = self.download_and_prepare_test(download_folder, sentence_separator)
            writer.write_to_conll(test_corpus, test_file)

        super(MIRNA, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_and_prepare_train(cls, data_folder: Path, sentence_separator: str):
        data_url = (
            "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/miRNA/miRNA-Train-Corpus.xml"
        )
        data_path = cached_path(data_url, data_folder)

        return cls.parse_file(data_path, "train", sentence_separator)

    @classmethod
    def download_and_prepare_test(cls, data_folder: Path, sentence_separator):
        data_url = (
            "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/miRNA/miRNA-Test-Corpus.xml"
        )
        data_path = cached_path(data_url, data_folder)

        return cls.parse_file(data_path, "test", sentence_separator)

    @classmethod
    def parse_file(cls, input_file: Path, split: str, sentence_separator: str) -> InternalBioNerDataset:
        tree = etree.parse(str(input_file))

        documents = {}
        entities_per_document = {}

        for document in tree.xpath(".//document"):
            document_id = document.get("id") + "-" + split
            entities = []

            document_text = ""
            for sentence in document.xpath(".//sentence"):
                if document_text:
                    document_text += sentence_separator

                sentence_offset = len(document_text)
                document_text += sentence.get("text") if document_text else sentence.get("text")

                for entity in sentence.xpath(".//entity"):
                    start, end = entity.get("charOffset").split("-")
                    entities.append(
                        Entity(
                            (
                                sentence_offset + int(start),
                                sentence_offset + int(end) + 1,
                            ),
                            entity.get("type"),
                        )
                    )

            documents[document_id] = document_text
            entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HunerMiRNAHelper(object):
    @staticmethod
    def get_mirna_subset(dataset: InternalBioNerDataset, split_url: str, split_dir: Path):
        split_file = cached_path(split_url, split_dir)

        with split_file.open(encoding="utf8") as f:
            ids = [line.strip() for line in f if line.strip()]
            ids = [id + "-train" for id in ids] + [id + "-test" for id in ids]
            ids = sorted(id_ for id_ in ids if id_ in dataset.documents)

        return InternalBioNerDataset(
            documents={k: dataset.documents[k] for k in ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in ids},
        )


class HUNER_GENE_MIRNA(HunerDataset):
    """
    HUNER version of the miRNA corpus containing protein / gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/miRNA"

    def get_subset(self, dataset: InternalBioNerDataset, split: str, split_dir: Path):
        # In the huner split files there is no information whether a given id originates
        # from the train or test file of the original corpus - so we have to adapt corpus
        # splitting here
        return HunerMiRNAHelper.get_mirna_subset(dataset, f"{self.split_url()}.{split}", split_dir)

    def get_corpus_sentence_splitter(self):
        return TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        sentence_separator = " "
        if isinstance(self.sentence_splitter, TagSentenceSplitter):
            sentence_separator = self.sentence_splitter.tag

        train_data = MIRNA.download_and_prepare_train(download_folder, sentence_separator)
        train_data = filter_and_map_entities(train_data, {"Genes/Proteins": GENE_TAG})

        test_data = MIRNA.download_and_prepare_test(download_folder, sentence_separator)
        test_data = filter_and_map_entities(test_data, {"Genes/Proteins": GENE_TAG})

        return merge_datasets([train_data, test_data])


class HUNER_SPECIES_MIRNA(HunerDataset):
    """
    HUNER version of the miRNA corpus containing species annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/miRNA"

    def get_subset(self, dataset: InternalBioNerDataset, split: str, split_dir: Path):
        # In the huner split files there is no information whether a given id originates
        # from the train or test file of the original corpus - so we have to adapt corpus
        # splitting here
        return HunerMiRNAHelper.get_mirna_subset(dataset, f"{self.split_url()}.{split}", split_dir)

    def get_corpus_sentence_splitter(self) -> SentenceSplitter:
        return TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        sentence_separator = " "
        if isinstance(self.sentence_splitter, TagSentenceSplitter):
            sentence_separator = self.sentence_splitter.tag

        train_data = MIRNA.download_and_prepare_train(download_folder, sentence_separator)
        train_data = filter_and_map_entities(train_data, {"Species": SPECIES_TAG})

        test_data = MIRNA.download_and_prepare_test(download_folder, sentence_separator)
        test_data = filter_and_map_entities(test_data, {"Species": SPECIES_TAG})

        return merge_datasets([train_data, test_data])


class HUNER_DISEASE_MIRNA(HunerDataset):
    """
    HUNER version of the miRNA corpus containing disease annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/miRNA"

    def get_subset(self, dataset: InternalBioNerDataset, split: str, split_dir: Path):
        # In the huner split files there is no information whether a given id originates
        # from the train or test file of the original corpus - so we have to adapt corpus
        # splitting here
        return HunerMiRNAHelper.get_mirna_subset(dataset, f"{self.split_url()}.{split}", split_dir)

    def get_corpus_sentence_splitter(self) -> SentenceSplitter:
        return TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        sentence_separator = " "
        if isinstance(self.sentence_splitter, TagSentenceSplitter):
            sentence_separator = self.sentence_splitter.tag

        train_data = MIRNA.download_and_prepare_train(download_folder, sentence_separator)
        train_data = filter_and_map_entities(train_data, {"Diseases": DISEASE_TAG})

        test_data = MIRNA.download_and_prepare_test(download_folder, sentence_separator)
        test_data = filter_and_map_entities(test_data, {"Diseases": DISEASE_TAG})

        return merge_datasets([train_data, test_data])


class KaewphanCorpusHelper:
    """Helper class for the corpora from Kaewphan et al., i.e. CLL and Gellus"""

    @staticmethod
    def download_cll_dataset(data_folder: Path):
        data_url = "http://bionlp-www.utu.fi/cell-lines/CLL_corpus.tar.gz"
        data_path = cached_path(data_url, data_folder)
        unpack_file(data_path, data_folder)

    @staticmethod
    def prepare_and_save_dataset(nersuite_folder: Path, output_file: Path):
        writer = open(str(output_file), "w", encoding="utf8")
        out_newline = False

        for file in os.listdir(str(nersuite_folder)):
            if not file.endswith(".nersuite"):
                continue

            annotations = []
            with open(os.path.join(str(nersuite_folder), file), "r", encoding="utf8") as reader:
                for line in reader.readlines():
                    columns = line.split("\t")
                    annotations.append(columns[:4])

            num_annotations = len(annotations)
            for i, annotation in enumerate(annotations):
                if len(annotation) == 1:
                    assert annotation[0] == "\n"
                    if not out_newline:
                        writer.write("\n")
                    out_newline = True
                    continue

                has_whitespace = "+"

                next_annotation = (
                    annotations[i + 1] if (i + 1) < num_annotations and len(annotations[i + 1]) > 1 else None
                )
                if next_annotation and next_annotation[1] == annotation[2]:
                    has_whitespace = "-"

                writer.write(" ".join([annotation[3], annotation[0], has_whitespace]) + "\n")
                out_newline = False

            if not out_newline:
                writer.write("\n")
                out_newline = True

        writer.close()

    @staticmethod
    def download_gellus_dataset(data_folder: Path):
        data_url = "http://bionlp-www.utu.fi/cell-lines/Gellus_corpus.tar.gz"
        data_path = cached_path(data_url, data_folder)
        unpack_file(data_path, data_folder)

    @staticmethod
    def read_dataset(nersuite_folder: Path, sentence_separator: str) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}
        for file in os.listdir(str(nersuite_folder)):
            if not file.endswith(".nersuite"):
                continue

            document_id = file.replace(".nersuite", "")

            with open(os.path.join(str(nersuite_folder), file), "r", encoding="utf8") as reader:
                document_text = ""
                entities = []

                entity_start = None
                entity_type = None

                for line in reader.readlines():
                    line = line.strip()
                    if line:
                        tag, _, _, _, token = line.split("\t")[:5]
                        if tag.startswith("B-"):
                            if entity_type is not None and entity_start is not None:
                                entities.append(Entity((entity_start, len(document_text)), entity_type))

                            entity_start = len(document_text) + 1 if document_text else 0
                            entity_type = tag[2:]

                        elif tag == "O" and entity_type is not None and entity_start is not None:
                            entities.append(
                                Entity(
                                    (entity_start, len(document_text)),
                                    entity_type,
                                )
                            )
                            entity_type = None

                        document_text = document_text + " " + token if document_text else token
                    else:
                        # Edge case: last token starts a new entity
                        if entity_type is not None and entity_start is not None:
                            entities.append(Entity((entity_start, len(document_text)), entity_type))
                        document_text += sentence_separator

                if document_text.endswith(sentence_separator):
                    document_text = document_text[: -len(sentence_separator)]

                documents[document_id] = document_text
                entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class CLL(ColumnCorpus):
    """
    Original CLL corpus containing cell line annotations.

    For further information, see Kaewphan et al.:
        Cell line name recognition in support of the identification of synthetic lethality in cancer from text
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4708107/
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"

        if not (train_file.exists()):
            KaewphanCorpusHelper.download_cll_dataset(data_folder)

            nersuite_folder = data_folder / "CLL-1.0.2" / "nersuite"
            KaewphanCorpusHelper.prepare_and_save_dataset(nersuite_folder, train_file)

        super(CLL, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class HUNER_CELL_LINE_CLL(HunerDataset):
    """
    HUNER version of the CLL corpus containing cell line annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/cll"

    def get_corpus_sentence_splitter(self) -> SentenceSplitter:
        return TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        KaewphanCorpusHelper.download_cll_dataset(data_dir)

        sentence_separator = " "
        if isinstance(self.sentence_splitter, TagSentenceSplitter):
            sentence_separator = self.sentence_splitter.tag

        nersuite_folder = data_dir / "CLL-1.0.2" / "nersuite"
        orig_dataset = KaewphanCorpusHelper.read_dataset(nersuite_folder, sentence_separator)

        return filter_and_map_entities(orig_dataset, {"CL": CELL_LINE_TAG})


class GELLUS(ColumnCorpus):
    """
    Original Gellus corpus containing cell line annotations.

    For further information, see Kaewphan et al.:
        Cell line name recognition in support of the identification of synthetic lethality in cancer from text
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4708107/
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            KaewphanCorpusHelper.download_gellus_dataset(data_folder)

            nersuite_train = data_folder / "GELLUS-1.0.3" / "nersuite" / "train"
            KaewphanCorpusHelper.prepare_and_save_dataset(nersuite_train, train_file)

            nersuite_dev = data_folder / "GELLUS-1.0.3" / "nersuite" / "devel"
            KaewphanCorpusHelper.prepare_and_save_dataset(nersuite_dev, dev_file)

            nersuite_test = data_folder / "GELLUS-1.0.3" / "nersuite" / "test"
            KaewphanCorpusHelper.prepare_and_save_dataset(nersuite_test, test_file)

        super(GELLUS, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class HUNER_CELL_LINE_GELLUS(HunerDataset):
    """
    HUNER version of the Gellus corpus containing cell line annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/gellus"

    def get_corpus_sentence_splitter(self) -> SentenceSplitter:
        return TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        KaewphanCorpusHelper.download_gellus_dataset(data_dir)

        sentence_separator = " "
        if isinstance(self.sentence_splitter, TagSentenceSplitter):
            sentence_separator = self.sentence_splitter.tag

        splits = []
        for folder in ["train", "devel", "test"]:
            nersuite_folder = data_dir / "GELLUS-1.0.3" / "nersuite" / folder
            splits.append(KaewphanCorpusHelper.read_dataset(nersuite_folder, sentence_separator))

        full_dataset = merge_datasets(splits)
        return filter_and_map_entities(full_dataset, {"Cell-line-name": CELL_LINE_TAG})


class LOCTEXT(ColumnCorpus):
    """
    Original LOCTEXT corpus containing species annotations.

    For further information see Cejuela et al.:
        LocText: relation extraction of protein localizations to assist database curation
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2021-9
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Custom implementation of :class:`SentenceSplitter`
            that segments a document into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            self.download_dataset(data_folder)
            full_dataset = self.parse_dataset(data_folder)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(full_dataset, train_file)

        super(LOCTEXT, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = "http://pubannotation.org/downloads/LocText-annotations.tgz"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

    @staticmethod
    def parse_dataset(data_dir: Path) -> InternalBioNerDataset:
        loctext_json_folder = data_dir / "LocText"

        entity_type_mapping = {
            "go": "protein",
            "uniprot": "protein",
            "taxonomy": "species",
        }

        documents = {}
        entities_per_document = {}

        for file in os.listdir(str(loctext_json_folder)):
            document_id = file.strip(".json")
            entities = []

            with open(os.path.join(str(loctext_json_folder), file), "r", encoding="utf8") as f_in:
                data = json.load(f_in)
                document_text = data["text"].strip()
                document_text = document_text.replace("\n", " ")

                if "denotations" in data.keys():
                    for ann in data["denotations"]:
                        start = int(ann["span"]["begin"])
                        end = int(ann["span"]["end"])

                        original_entity_type = ann["obj"].split(":")[0]
                        if original_entity_type not in entity_type_mapping:
                            continue

                        entity_type = entity_type_mapping[original_entity_type]
                        entities.append(Entity((start, end), entity_type))

                documents[document_id] = document_text
                entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_SPECIES_LOCTEXT(HunerDataset):
    """
    HUNER version of the Loctext corpus containing species annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/loctext"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        LOCTEXT.download_dataset(data_dir)
        dataset = LOCTEXT.parse_dataset(data_dir)

        return filter_and_map_entities(dataset, {"species": SPECIES_TAG})


class HUNER_GENE_LOCTEXT(HunerDataset):
    """
    HUNER version of the Loctext corpus containing protein annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/loctext"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        LOCTEXT.download_dataset(data_dir)
        dataset = LOCTEXT.parse_dataset(data_dir)

        return filter_and_map_entities(dataset, {"protein": GENE_TAG})


class CHEMDNER(ColumnCorpus):
    """
    Original corpus of the CHEMDNER shared task.

    For further information see Krallinger et al.:
      The CHEMDNER corpus of chemicals and drugs and its annotation principles
      https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Custom implementation of :class:`SentenceSplitter` which
            segements documents into sentences and tokens
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{sentence_splitter.name}_dev.conll"
        test_file = data_folder / f"{sentence_splitter.name}_test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            train_data = bioc_to_internal(download_dir / "chemdner_corpus" / "training.bioc.xml")
            dev_data = bioc_to_internal(download_dir / "chemdner_corpus" / "development.bioc.xml")
            test_data = bioc_to_internal(download_dir / "chemdner_corpus" / "evaluation.bioc.xml")

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)

            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(CHEMDNER, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2014/chemdner_corpus.tar.gz"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)


class HUNER_CHEMICAL_CHEMDNER(HunerDataset):
    """
    HUNER version of the CHEMDNER corpus containing chemical annotations.
    """

    def __init__(self, *args, download_folder=None, **kwargs):
        self.download_folder = download_folder or CHEMDNER.default_dir / "original"
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/chemdner"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        os.makedirs(str(self.download_folder), exist_ok=True)
        CHEMDNER.download_dataset(self.download_folder)
        train_data = bioc_to_internal(self.download_folder / "chemdner_corpus" / "training.bioc.xml")
        dev_data = bioc_to_internal(self.download_folder / "chemdner_corpus" / "development.bioc.xml")
        test_data = bioc_to_internal(self.download_folder / "chemdner_corpus" / "evaluation.bioc.xml")
        all_data = merge_datasets([train_data, dev_data, test_data])
        all_data = filter_and_map_entities(
            all_data,
            {
                "ABBREVIATION": CHEMICAL_TAG,
                "FAMILY": CHEMICAL_TAG,
                "FORMULA": CHEMICAL_TAG,
                "IDENTIFIER": CHEMICAL_TAG,
                "MULTIPLE": CHEMICAL_TAG,
                "NO_CLASS": CHEMICAL_TAG,
                "SYSTEMATIC": CHEMICAL_TAG,
                "TRIVIAL": CHEMICAL_TAG,
            },
        )

        return all_data


class IEPA(ColumnCorpus):
    """
    IEPA corpus as provided by http://corpora.informatik.hu-berlin.de/
    (Original corpus is 404)

    For further information see Ding, Berleant, Nettleton, Wurtele:
      Mining MEDLINE: abstracts, sentences, or phrases?
      https://www.ncbi.nlm.nih.gov/pubmed/11928487
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        tokenizer: Tokenizer = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param tokenizer: Custom implementation of :class:`Tokenizer` which
             segments sentences into tokens (default :class:`SciSpacyTokenizer`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if tokenizer is None:
            tokenizer = SciSpacyTokenizer()

        sentence_splitter = NewlineSentenceSplitter(tokenizer=tokenizer)

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            all_data = bioc_to_internal(download_dir / "iepa_bioc.xml")

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(all_data, train_file)

        super(IEPA, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = "http://corpora.informatik.hu-berlin.de/corpora/brat2bioc/iepa_bioc.xml.zip"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)


class HUNER_GENE_IEPA(HunerDataset):
    """
    HUNER version of the IEPA corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/iepa"

    def get_corpus_sentence_splitter(self) -> SentenceSplitter:
        return NewlineSentenceSplitter(tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        os.makedirs(str(data_dir), exist_ok=True)
        IEPA.download_dataset(data_dir)

        all_data = bioc_to_internal(data_dir / "iepa_bioc.xml")
        all_data = filter_and_map_entities(all_data, {"Protein": GENE_TAG})

        return all_data


class LINNEAUS(ColumnCorpus):
    """
    Original LINNEAUS corpus containing species annotations.

    For further information see Gerner et al.:
         LINNAEUS: a species name identification system for biomedical literature
         https://www.ncbi.nlm.nih.gov/pubmed/20149233
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        tokenizer: Tokenizer = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param tokenizer: Custom implementation of :class:`Tokenizer` which segments
             sentence into tokens (default :class:`SciSpacyTokenizer`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if tokenizer is None:
            tokenizer = SciSpacyTokenizer()

        sentence_splitter = TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=tokenizer)

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            dataset = self.download_and_parse_dataset(data_folder)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(dataset, train_file)

        super(LINNEAUS, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_and_parse_dataset(data_dir: Path):
        data_url = "https://iweb.dl.sourceforge.net/project/linnaeus/Corpora/manual-corpus-species-1.0.tar.gz"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

        documents = {}
        entities_per_document = defaultdict(list)

        # Read texts
        texts_directory = data_dir / "manual-corpus-species-1.0" / "txt"
        for filename in os.listdir(str(texts_directory)):
            document_id = filename.strip(".txt")

            with open(os.path.join(str(texts_directory), filename), "r", encoding="utf8") as file:
                documents[document_id] = file.read().strip()

        # Read annotations
        tag_file = data_dir / "manual-corpus-species-1.0" / "filtered_tags.tsv"
        with open(str(tag_file), "r", encoding="utf8") as file:
            next(file)  # Ignore header row

            for line in file:
                if not line:
                    continue

                document_id, _start, _end, text = line.strip().split("\t")[1:5]
                start, end = int(_start), int(_end)

                entities_per_document[document_id].append(Entity((start, end), SPECIES_TAG))

                document_text = documents[document_id]
                if document_text[start:end] != text:
                    raise AssertionError()

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_SPECIES_LINNEAUS(HunerDataset):
    """
    HUNER version of the LINNEAUS corpus containing species annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/linneaus"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        return LINNEAUS.download_and_parse_dataset(data_dir)


class CDR(ColumnCorpus):
    """
    CDR corpus as provided by https://github.com/JHnlp/BioCreative-V-CDR-Corpus

    For further information see Li et al.:
      BioCreative V CDR task corpus: a resource for chemical disease relation extraction
      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments
            documents into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{sentence_splitter.name}_dev.conll"
        test_file = data_folder / f"{sentence_splitter.name}_test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            train_data = bioc_to_internal(download_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml")
            dev_data = bioc_to_internal(
                download_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_DevelopmentSet.BioC.xml"
            )
            test_data = bioc_to_internal(download_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.BioC.xml")

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(CDR, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = "https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)


class HUNER_DISEASE_CDR(HunerDataset):
    """
    HUNER version of the IEPA corpus containing disease annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/CDRDisease"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        os.makedirs(str(data_dir), exist_ok=True)
        CDR.download_dataset(data_dir)
        train_data = bioc_to_internal(data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml")
        dev_data = bioc_to_internal(data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_DevelopmentSet.BioC.xml")
        test_data = bioc_to_internal(data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.BioC.xml")
        all_data = merge_datasets([train_data, dev_data, test_data])
        all_data = filter_and_map_entities(all_data, {"Disease": DISEASE_TAG})

        return all_data


class HUNER_CHEMICAL_CDR(HunerDataset):
    """
    HUNER version of the IEPA corpus containing chemical annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/CDRChem"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        os.makedirs(str(data_dir), exist_ok=True)
        CDR.download_dataset(data_dir)
        train_data = bioc_to_internal(data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml")
        dev_data = bioc_to_internal(data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_DevelopmentSet.BioC.xml")
        test_data = bioc_to_internal(data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.BioC.xml")
        all_data = merge_datasets([train_data, dev_data, test_data])
        all_data = filter_and_map_entities(all_data, {"Chemical": CHEMICAL_TAG})

        return all_data


class VARIOME(ColumnCorpus):
    """
    Variome corpus as provided by http://corpora.informatik.hu-berlin.de/corpora/brat2bioc/hvp_bioc.xml.zip

    For further information see Verspoor et al.:
      Annotating the biomedical literature for the human variome
      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3676157/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments
             documents into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            all_data = self.parse_corpus(download_dir / "hvp_bioc.xml")

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(all_data, train_file)

        super(VARIOME, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = "http://corpora.informatik.hu-berlin.de/corpora/brat2bioc/hvp_bioc.xml.zip"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

    @staticmethod
    def parse_corpus(corpus_xml: Path) -> InternalBioNerDataset:
        corpus = bioc_to_internal(corpus_xml)

        cleaned_documents = {}
        cleaned_entities_per_document = {}

        for id, document_text in corpus.documents.items():
            entities = corpus.entities_per_document[id]
            original_length = len(document_text)

            text_cleaned = document_text.replace("** IGNORE LINE **\n", "")
            offset = original_length - len(text_cleaned)

            if offset != 0:
                new_entities = []
                for entity in entities:
                    new_start = entity.char_span.start - offset
                    new_end = entity.char_span.stop - offset

                    new_entities.append(Entity((new_start, new_end), entity.type))

                    orig_text = document_text[entity.char_span.start : entity.char_span.stop]
                    new_text = text_cleaned[new_start:new_end]
                    assert orig_text == new_text

                entities = new_entities
                document_text = text_cleaned

            cleaned_documents[id] = document_text
            cleaned_entities_per_document[id] = entities

        return InternalBioNerDataset(
            documents=cleaned_documents,
            entities_per_document=cleaned_entities_per_document,
        )


class HUNER_GENE_VARIOME(HunerDataset):
    """
    HUNER version of the Variome corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/variome_gene"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        os.makedirs(str(data_dir), exist_ok=True)
        VARIOME.download_dataset(data_dir)
        all_data = VARIOME.parse_corpus(data_dir / "hvp_bioc.xml")
        all_data = filter_and_map_entities(all_data, {"gene": GENE_TAG})

        return all_data


class HUNER_DISEASE_VARIOME(HunerDataset):
    """
    HUNER version of the Variome corpus containing disease annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/variome_disease"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        os.makedirs(str(data_dir), exist_ok=True)
        VARIOME.download_dataset(data_dir)
        all_data = VARIOME.parse_corpus(data_dir / "hvp_bioc.xml")
        all_data = filter_and_map_entities(all_data, {"Disorder": DISEASE_TAG, "disease": DISEASE_TAG})

        return all_data


class HUNER_SPECIES_VARIOME(HunerDataset):
    """
    HUNER version of the Variome corpus containing species annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/variome_species"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        os.makedirs(str(data_dir), exist_ok=True)
        VARIOME.download_dataset(data_dir)
        all_data = VARIOME.parse_corpus(data_dir / "hvp_bioc.xml")
        all_data = filter_and_map_entities(all_data, {"Living_Beings": SPECIES_TAG})

        return all_data


class NCBI_DISEASE(ColumnCorpus):
    """
    Original NCBI disease corpus containing disease annotations.

    For further information see Dogan et al.:
       NCBI disease corpus: a resource for disease name recognition and concept normalization
       https://www.ncbi.nlm.nih.gov/pubmed/24393765
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments
             documents into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{sentence_splitter.name}_dev.conll"
        test_file = data_folder / f"{sentence_splitter.name}_test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            orig_folder = self.download_corpus(data_folder)

            train_data = self.parse_input_file(orig_folder / "NCBItrainset_patched.txt")
            dev_data = self.parse_input_file(orig_folder / "NCBIdevelopset_corpus.txt")
            test_data = self.parse_input_file(orig_folder / "NCBItestset_corpus.txt")

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(NCBI_DISEASE, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_corpus(cls, data_dir: Path) -> Path:
        original_folder = data_dir / "original"
        os.makedirs(str(original_folder), exist_ok=True)

        data_urls = [
            "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip",
            "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip",
            "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip",
        ]

        for url in data_urls:
            data_path = cached_path(url, original_folder)
            unpack_file(data_path, original_folder)

        # We need to apply a patch to correct the original training file
        orig_train_file = original_folder / "NCBItrainset_corpus.txt"
        patched_train_file = original_folder / "NCBItrainset_patched.txt"
        cls.patch_training_file(orig_train_file, patched_train_file)

        return original_folder

    @staticmethod
    def patch_training_file(orig_train_file: Path, patched_file: Path):
        patch_lines = {
            3249: '10923035\t711\t761\tgeneralized epilepsy and febrile seizures " plus "\tSpecificDisease\tD004829+D003294\n'
        }
        with open(str(orig_train_file), "r", encoding="utf8") as input:
            with open(str(patched_file), "w", encoding="utf8") as output:
                line_no = 1

                for line in input:
                    output.write(patch_lines[line_no] if line_no in patch_lines else line)
                    line_no += 1

    @staticmethod
    def parse_input_file(input_file: Path):
        documents = {}
        entities_per_document = {}

        with open(str(input_file), "r", encoding="utf8") as file:
            document_id = ""
            document_text = ""
            entities: List[Entity] = []

            c = 1
            for line in file:
                line = line.strip()
                if not line:
                    if document_id and document_text:
                        documents[document_id] = document_text
                        entities_per_document[document_id] = entities

                    document_id, document_text, entities = "", "", []
                    c = 1
                    continue
                if c == 1:
                    # Articles title
                    document_text = line.split("|")[2] + " "
                    document_id = line.split("|")[0]
                elif c == 2:
                    # Article abstract
                    document_text += line.split("|")[2]
                else:
                    # Entity annotations
                    columns = line.split("\t")
                    start = int(columns[1])
                    end = int(columns[2])
                    entity_text = columns[3]

                    assert document_text[start:end] == entity_text
                    entities.append(Entity((start, end), DISEASE_TAG))
                c += 1

            if c != 1 and document_id and document_text:
                documents[document_id] = document_text
                entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_DISEASE_NCBI(HunerDataset):
    """
    HUNER version of the NCBI corpus containing disease annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/ncbi"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        orig_folder = NCBI_DISEASE.download_corpus(data_dir)

        train_data = NCBI_DISEASE.parse_input_file(orig_folder / "NCBItrainset_patched.txt")
        dev_data = NCBI_DISEASE.parse_input_file(orig_folder / "NCBIdevelopset_corpus.txt")
        test_data = NCBI_DISEASE.parse_input_file(orig_folder / "NCBItestset_corpus.txt")

        return merge_datasets([train_data, dev_data, test_data])


class ScaiCorpus(ColumnCorpus):
    """Base class to support the SCAI chemicals and disease corpora"""

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter:  Implementation of :class:`SentenceSplitter` which segments
             documents into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            dataset_file = self.download_corpus(data_folder)
            train_data = self.parse_input_file(dataset_file)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)

        super(ScaiCorpus, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    def download_corpus(self, data_folder: Path) -> Path:
        raise NotImplementedError()

    @staticmethod
    def parse_input_file(input_file: Path):
        documents: Dict[str, str] = {}
        entities_per_document: Dict[str, List[Entity]] = {}

        with open(str(input_file), "r", encoding="iso-8859-1") as file:
            document_id = None
            document_text = ""
            entities: List[Entity] = []
            entity_type = None
            entity_start = 0

            for line in file:
                line = line.strip()
                if not line:
                    continue

                if line[:3] == "###":
                    # Edge case: last token starts a new entity
                    if entity_type is not None:
                        entities.append(Entity((entity_start, len(document_text)), entity_type))

                    if not (document_id is None or document_text is None):
                        documents[document_id] = document_text
                        entities_per_document[document_id] = entities

                    document_id = line.strip("#").strip()
                    document_text = ""
                    entities = []
                else:
                    columns = line.strip().split("\t")
                    token = columns[0].strip()
                    tag = columns[4].strip().split("|")[1]

                    if tag.startswith("B-"):
                        if entity_type is not None:
                            entities.append(Entity((entity_start, len(document_text)), entity_type))

                        entity_start = len(document_text) + 1 if document_text else 0
                        entity_type = tag[2:]

                    elif tag == "O" and entity_type is not None:
                        entities.append(Entity((entity_start, len(document_text)), entity_type))
                        entity_type = None

                    document_text = document_text + " " + token if document_text is not None else token

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class SCAI_CHEMICALS(ScaiCorpus):
    """
    Original SCAI chemicals corpus containing chemical annotations.

    For further information see Kolářik et al.:
         Chemical Names: Terminological Resources and Corpora Annotation
         https://pub.uni-bielefeld.de/record/2603498
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download_corpus(self, data_dir: Path) -> Path:
        return self.perform_corpus_download(data_dir)

    @staticmethod
    def perform_corpus_download(data_dir: Path) -> Path:
        original_directory = data_dir / "original"
        os.makedirs(str(original_directory), exist_ok=True)

        url = "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/Corpora-for-Chemical-Entity-Recognition/chemicals-test-corpus-27-04-2009-v3_iob.gz"
        data_path = cached_path(url, original_directory)
        corpus_file = original_directory / "chemicals-test-corpus-27-04-2009-v3.iob"
        unpack_file(data_path, corpus_file)

        return corpus_file


class SCAI_DISEASE(ScaiCorpus):
    """
    Original SCAI disease corpus containing disease annotations.

    For further information see Gurulingappa et al.:
     An Empirical Evaluation of Resources for the Identification of Diseases and Adverse Effects in Biomedical Literature
     https://pub.uni-bielefeld.de/record/2603398
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download_corpus(self, data_dir: Path) -> Path:
        return self.perform_corpus_download(data_dir)

    @staticmethod
    def perform_corpus_download(data_dir: Path) -> Path:
        original_directory = data_dir / "original"
        os.makedirs(str(original_directory), exist_ok=True)

        url = "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/Disease-ae-corpus.iob"
        data_path = cached_path(url, original_directory)

        return data_path


class HUNER_CHEMICAL_SCAI(HunerDataset):
    """
    HUNER version of the SCAI chemicals corpus containing chemical annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/scai_chemicals"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        original_file = SCAI_CHEMICALS.perform_corpus_download(data_dir)
        corpus = ScaiCorpus.parse_input_file(original_file)

        # Map all entities to chemicals
        entity_mapping = {
            "FAMILY": CHEMICAL_TAG,
            "TRIVIALVAR": CHEMICAL_TAG,
            "PARTIUPAC": CHEMICAL_TAG,
            "TRIVIAL": CHEMICAL_TAG,
            "ABBREVIATION": CHEMICAL_TAG,
            "IUPAC": CHEMICAL_TAG,
            "MODIFIER": CHEMICAL_TAG,
            "SUM": CHEMICAL_TAG,
        }

        return filter_and_map_entities(corpus, entity_mapping)


class HUNER_DISEASE_SCAI(HunerDataset):
    """
    HUNER version of the SCAI chemicals corpus containing chemical annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/scai_disease"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        original_file = SCAI_DISEASE.perform_corpus_download(data_dir)
        corpus = ScaiCorpus.parse_input_file(original_file)

        # Map all entities to disease
        entity_mapping = {"DISEASE": DISEASE_TAG, "ADVERSE": DISEASE_TAG}

        return filter_and_map_entities(corpus, entity_mapping)


class OSIRIS(ColumnCorpus):
    """
    Original OSIRIS corpus containing variation and gene annotations.

    For further information see Furlong et al.:
       Osiris v1.2: a named entity recognition system for sequence variants of genes in biomedical literature
       https://www.ncbi.nlm.nih.gov/pubmed/18251998
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
        load_original_unfixed_annotation=False,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which
             segments documents into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        :param load_original_unfixed_annotation: The original annotation of Osiris
             erroneously annotates two sentences as a protein. Set to True if you don't
             want the fixed version.
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            corpus_folder = self.download_dataset(data_folder)
            corpus_data = self.parse_dataset(corpus_folder, fix_annotation=not load_original_unfixed_annotation)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(corpus_data, train_file)

        super(OSIRIS, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_dataset(cls, data_dir: Path) -> Path:
        url = "http://ibi.imim.es/OSIRIScorpusv02.tar"
        data_path = cached_path(url, data_dir)
        unpack_file(data_path, data_dir)

        return data_dir / "OSIRIScorpusv02"

    @classmethod
    def parse_dataset(cls, corpus_folder: Path, fix_annotation=True):
        documents = {}
        entities_per_document = {}

        input_files = [
            file for file in os.listdir(str(corpus_folder)) if file.endswith(".txt") and not file.startswith("README")
        ]
        for text_file in input_files:

            with open(os.path.join(str(corpus_folder), text_file), encoding="utf8") as text_reader:
                document_text = text_reader.read()
                if not document_text:
                    continue

                article_parts = document_text.split("\n\n")
                document_id = article_parts[0]
                text_offset = document_text.find(article_parts[1])
                document_text = (article_parts[1] + "  " + article_parts[2]).strip()

            with open(os.path.join(str(corpus_folder), text_file + ".ann"), encoding="utf8") as ann_file:
                entities = []

                tree = etree.parse(ann_file)
                for annotation in tree.xpath(".//Annotation"):
                    entity_type = annotation.get("type")
                    if entity_type == "file":
                        continue

                    start, end = annotation.get("span").split("..")
                    start, end = int(start), int(end)

                    if fix_annotation and text_file == "article46.txt" and start == 289 and end == 644:
                        end = 295

                    entities.append(Entity((start - text_offset, end - text_offset), entity_type))

            documents[document_id] = document_text
            entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_GENE_OSIRIS(HunerDataset):
    """
    HUNER version of the OSIRIS corpus containing (only) gene annotations.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/osiris"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        original_file = OSIRIS.download_dataset(data_dir)
        corpus = OSIRIS.parse_dataset(original_file)

        entity_type_mapping = {"ge": GENE_TAG}
        return filter_and_map_entities(corpus, entity_type_mapping)


class S800(ColumnCorpus):
    """
    S800 corpus
    For further information see Pafilis et al.:
      The SPECIES and ORGANISMS Resources for Fast and Accurate Identification of Taxonomic Names in Text
      http://www.plosone.org/article/info:doi%2F10.1371%2Fjournal.pone.0065390
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments documents
             into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            all_data = self.parse_dataset(download_dir)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(all_data, train_file)

        super(S800, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = "https://species.jensenlab.org/files/S800-1.0.tar.gz"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

    @staticmethod
    def parse_dataset(data_dir: Path) -> InternalBioNerDataset:
        entities_per_document = defaultdict(list)
        texts_per_document = {}
        with (data_dir / "S800.tsv").open(encoding="utf8") as f:
            for line in f:
                fields = line.strip().split("\t")
                if not fields:
                    continue
                fname, pmid = fields[1].split(":")
                start, end = int(fields[2]), int(fields[3])

                if start == end:
                    continue

                entities_per_document[fname].append(Entity((start, end), "Species"))

        for fname in entities_per_document:
            with (data_dir / "abstracts" / fname).with_suffix(".txt").open(encoding="utf8") as f:
                texts_per_document[fname] = f.read()

        return InternalBioNerDataset(documents=texts_per_document, entities_per_document=entities_per_document)


class HUNER_SPECIES_S800(HunerDataset):
    """
    HUNER version of the S800 corpus containing species annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/s800"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        S800.download_dataset(data_dir)
        data = S800.parse_dataset(data_dir)
        data = filter_and_map_entities(data, {"Species": SPECIES_TAG})

        return data


class GPRO(ColumnCorpus):
    """
    Original GPRO corpus containing gene annotations.

    For further information see:
         https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/gpro-detailed-task-description/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments documents
             into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{sentence_splitter.name}_dev.conll"

        if not (train_file.exists() and dev_file.exists()):
            train_folder = self.download_train_corpus(data_folder)
            train_text_file = train_folder / "chemdner_patents_train_text.txt"
            train_ann_file = train_folder / "chemdner_gpro_gold_standard_train_v02.tsv"
            train_data = self.parse_input_file(train_text_file, train_ann_file)

            dev_folder = self.download_dev_corpus(data_folder)
            dev_text_file = dev_folder / "chemdner_patents_development_text.txt"
            dev_ann_file = dev_folder / "chemdner_gpro_gold_standard_development.tsv"
            dev_data = self.parse_input_file(dev_text_file, dev_ann_file)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)

        super(GPRO, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_train_corpus(cls, data_dir: Path) -> Path:
        corpus_dir = data_dir / "original"
        os.makedirs(str(corpus_dir), exist_ok=True)

        train_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2015/gpro_training_set_v02.tar.gz"
        data_path = cached_path(train_url, corpus_dir)
        unpack_file(data_path, corpus_dir)

        return corpus_dir / "gpro_training_set_v02"

    @classmethod
    def download_dev_corpus(cls, data_dir) -> Path:
        corpus_dir = data_dir / "original"
        os.makedirs(str(corpus_dir), exist_ok=True)

        dev_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2015/gpro_development_set.tar.gz"
        data_path = cached_path(dev_url, corpus_dir)
        unpack_file(data_path, corpus_dir)

        return corpus_dir / "gpro_development_set"

    @staticmethod
    def parse_input_file(text_file: Path, ann_file: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document: Dict[str, List[Entity]] = {}

        document_title_length = {}

        with open(str(text_file), "r", encoding="utf8") as text_reader:
            for line in text_reader:
                if not line:
                    continue

                document_id, title, abstract = line.split("\t")
                documents[document_id] = title + " " + abstract
                document_title_length[document_id] = len(title) + 1

                entities_per_document[document_id] = []

        with open(str(ann_file), "r", encoding="utf8") as ann_reader:
            for line in ann_reader:
                if not line:
                    continue

                columns = line.split("\t")
                document_id = columns[0]
                start, end = int(columns[2]), int(columns[3])

                if columns[1] == "A":
                    start = start + document_title_length[document_id]
                    end = end + document_title_length[document_id]

                entities_per_document[document_id].append(Entity((start, end), GENE_TAG))

                document_text = documents[document_id]
                assert columns[4] == document_text[start:end]

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_GENE_GPRO(HunerDataset):
    """
    HUNER version of the GPRO corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/gpro"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        train_folder = GPRO.download_train_corpus(data_dir)
        train_text_file = train_folder / "chemdner_patents_train_text.txt"
        train_ann_file = train_folder / "chemdner_gpro_gold_standard_train_v02.tsv"
        train_data = GPRO.parse_input_file(train_text_file, train_ann_file)

        dev_folder = GPRO.download_dev_corpus(data_dir)
        dev_text_file = dev_folder / "chemdner_patents_development_text.txt"
        dev_ann_file = dev_folder / "chemdner_gpro_gold_standard_development.tsv"
        dev_data = GPRO.parse_input_file(dev_text_file, dev_ann_file)

        return merge_datasets([train_data, dev_data])


class DECA(ColumnCorpus):
    """
    Original DECA corpus containing gene annotations.

    For further information see Wang et al.:
       Disambiguating the species of biomedical named entities using natural language parsers
       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2828111/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments
             documents into sentences and tokens (default BioSpacySentenceSpliiter)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / "train.conll"

        if not train_file.exists():
            corpus_dir = self.download_corpus(data_folder)
            text_dir = corpus_dir / "text"
            gold_file = corpus_dir / "gold.txt"

            corpus_data = self.parse_corpus(text_dir, gold_file)
            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(corpus_data, train_file)

        super(DECA, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_corpus(cls, data_dir: Path) -> Path:
        url = "http://www.nactem.ac.uk/deca/species_corpus_0.2.tar.gz"
        data_path = cached_path(url, data_dir)
        unpack_file(data_path, data_dir)

        return data_dir / "species_corpus_0.2"

    @staticmethod
    def parse_corpus(text_dir: Path, gold_file: Path) -> InternalBioNerDataset:
        documents: Dict[str, str] = {}
        entities_per_document: Dict[str, List[Entity]] = {}

        text_files = [file for file in os.listdir(str(text_dir)) if not file.startswith(".")]

        for file in text_files:
            document_id = file.strip(".txt")
            with open(os.path.join(str(text_dir), file), "r", encoding="utf8") as text_file:
                documents[document_id] = text_file.read().strip()
                entities_per_document[document_id] = []

        with open(str(gold_file), "r", encoding="utf8") as gold_reader:
            for line in gold_reader:
                if not line:
                    continue
                columns = line.strip().split("\t")

                document_id = columns[0].strip(".txt")
                start, end = int(columns[1]), int(columns[2])

                entities_per_document[document_id].append(Entity((start, end), GENE_TAG))

                document_text = documents[document_id]
                assert document_text[start:end] == columns[3]

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_GENE_DECA(HunerDataset):
    """
    HUNER version of the DECA corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/deca"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        corpus_dir = DECA.download_corpus(data_dir)
        text_dir = corpus_dir / "text"
        gold_file = corpus_dir / "gold.txt"

        return DECA.parse_corpus(text_dir, gold_file)


class FSU(ColumnCorpus):
    """
    Original FSU corpus containing protein and derived annotations.

    For further information see Hahn et al.:
      A proposal for a configurable silver standard
      https://www.aclweb.org/anthology/W10-1838/
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        sentence_splitter = TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SpaceTokenizer())
        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not train_file.exists():
            corpus_dir = self.download_corpus(data_folder)
            corpus_data = self.parse_corpus(corpus_dir, SENTENCE_TAG)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(corpus_data, train_file)

        super(FSU, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_corpus(cls, data_dir: Path) -> Path:
        url = "https://julielab.de/downloads/resources/fsu_prge_release_v1_0.tgz"
        data_path = cached_path(url, data_dir)
        unpack_file(data_path, data_dir, mode="targz")

        return data_dir / "fsu-prge-release-v1.0"

    @staticmethod
    def parse_corpus(corpus_dir: Path, sentence_separator: str) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}

        for subcorpus in corpus_dir.iterdir():
            if not subcorpus.is_dir():
                continue
            for doc in (subcorpus / "mmax").iterdir():
                if not doc.is_dir():
                    continue
                try:
                    with open(doc / "Basedata" / "Basedata.xml", "r", encoding="utf8") as word_f:
                        word_tree = etree.parse(word_f)
                    with open(doc / "Markables" / "sentence.xml", "r", encoding="utf8") as sentence_f:
                        sentence_tree = etree.parse(sentence_f).getroot()
                    with open(doc / "Markables" / "proteins.xml", "r", encoding="utf8") as protein_f:
                        protein_tree = etree.parse(protein_f).getroot()
                    with open(doc / "Basedata.uri", "r", encoding="utf8") as id_f:
                        document_id = id_f.read().strip()
                except FileNotFoundError:
                    # Incomplete article
                    continue
                except XMLSyntaxError:
                    # Invalid XML syntax
                    continue

                word_to_id = {}
                words = []
                for i, token in enumerate(word_tree.xpath(".//word")):
                    words += [token.text]
                    word_to_id[token.get("id")] = i
                word_pos = [(0, 0) for _ in words]

                sentences_id_span = sorted(
                    [(int(sentence.get("id").split("_")[-1]), sentence.get("span")) for sentence in sentence_tree]
                )

                sentences = []
                for j, sentence in enumerate(sentences_id_span):
                    tmp_sentence = []
                    akt_pos = 0
                    start = word_to_id[sentence[1].split("..")[0]]
                    end = word_to_id[sentence[1].split("..")[1]]
                    for i in range(start, end + 1):
                        tmp_sentence += [words[i]]
                        word_pos[i] = (j, akt_pos)
                        akt_pos += len(words[i]) + 1
                    sentences += [tmp_sentence]

                pre_entities: List[List[Tuple[int, int, str]]] = [[] for _ in sentences]
                for protein in protein_tree:
                    for span in protein.get("span").split(","):
                        start = word_to_id[span.split("..")[0]]
                        end = word_to_id[span.split("..")[-1]]
                        pre_entities[word_pos[start][0]] += [
                            (
                                word_pos[start][1],
                                word_pos[end][1] + len(words[end]),
                                protein.get("proteins"),
                            )
                        ]

                sentence_texts = [" ".join(sentence) for sentence in sentences]
                document = sentence_separator.join(sentence_texts)

                entities = []
                sent_offset = 0
                for sent, sent_entities in zip(sentence_texts, pre_entities):
                    entities += [
                        Entity(
                            (start + sent_offset, end + sent_offset),
                            ent_type,
                        )
                        for (start, end, ent_type) in sent_entities
                    ]
                    sent_offset += len(sent) + len(sentence_separator)

                documents[document_id] = document
                entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_GENE_FSU(HunerDataset):
    """
    HUNER version of the FSU corpus containing (only) gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/fsu"

    def get_corpus_sentence_splitter(self) -> SentenceSplitter:
        return TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=SciSpacyTokenizer())

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        corpus_dir = FSU.download_corpus(data_dir)

        sentence_separator = " "
        if isinstance(self.sentence_splitter, TagSentenceSplitter):
            sentence_separator = self.sentence_splitter.tag

        corpus = FSU.parse_corpus(corpus_dir, sentence_separator)

        entity_type_mapping = {
            "protein": GENE_TAG,
            "protein_familiy_or_group": GENE_TAG,
            "protein_complex": GENE_TAG,
            "protein_variant": GENE_TAG,
            "protein_enum": GENE_TAG,
        }
        return filter_and_map_entities(corpus, entity_type_mapping)


class CRAFT(ColumnCorpus):
    """
    Original CRAFT corpus (version 2.0) containing all but the coreference and sections/typography annotations.

    For further information see Bada et al.:
      Concept annotation in the craft corpus
      https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-161
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments documents
             into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not train_file.exists():
            corpus_dir = self.download_corpus(data_folder)
            corpus_data = self.parse_corpus(corpus_dir)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(corpus_data, train_file)

        super(CRAFT, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_corpus(cls, data_dir: Path) -> Path:
        url = "http://sourceforge.net/projects/bionlp-corpora/files/CRAFT/v2.0/craft-2.0.tar.gz/download"
        data_path = cached_path(url, data_dir)
        unpack_file(data_path, data_dir, mode="targz")

        return data_dir / "craft-2.0"

    @staticmethod
    def parse_corpus(corpus_dir: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}

        text_dir = corpus_dir / "articles" / "txt"
        document_texts = [doc for doc in text_dir.iterdir() if doc.name[-4:] == ".txt"]
        annotation_dirs = [
            path
            for path in (corpus_dir / "xml").iterdir()
            if path.name not in ["sections-and-typography", "coreference"]
        ]

        for doc in Tqdm.tqdm(document_texts, desc="Converting to internal"):
            document_id = doc.name.split(".")[0]

            with open(doc, "r", encoding="utf8") as f_txt:
                documents[document_id] = f_txt.read()

            entities = []

            for annotation_dir in annotation_dirs:
                with open(
                    annotation_dir / (doc.name + ".annotations.xml"),
                    "r",
                    encoding="utf8",
                ) as f_ann:
                    ann_tree = etree.parse(f_ann)
                for annotation in ann_tree.xpath("//annotation"):
                    for span in annotation.xpath("span"):
                        start = int(span.get("start"))
                        end = int(span.get("end"))
                        entities += [Entity((start, end), annotation_dir.name)]

            entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class BIOSEMANTICS(ColumnCorpus):
    """
    Original Biosemantics corpus.

    For further information see Akhondi et al.:
      Annotated chemical patent corpus: a gold standard for text mining
      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4182036/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments documents
            into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            corpus_dir = self.download_dataset(data_folder)
            full_dataset = self.parse_dataset(corpus_dir)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(full_dataset, train_file)

        super(BIOSEMANTICS, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path) -> Path:
        data_url = "http://biosemantics.erasmusmc.nl/PatentCorpus/Patent_Corpus.rar"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

        return data_dir / "Patent_Corpus"

    @staticmethod
    def parse_dataset(data_dir: Path) -> InternalBioNerDataset:
        base_folder = data_dir / "Full_set"

        dirs = [file for file in os.listdir(str(base_folder)) if os.path.isdir(os.path.join(str(base_folder), file))]

        text_files = []
        for directory in dirs:
            text_files += [
                os.path.join(str(base_folder), directory, file)
                for file in os.listdir(os.path.join(str(base_folder), directory))
                if file[-4:] == ".txt"
            ]
        text_files = sorted(text_files)

        documents: Dict[str, str] = {}
        entities_per_document: Dict[str, List[Entity]] = {}

        for text_file in sorted(text_files):
            document_id = os.path.basename(text_file).split("_")[0]
            with open(text_file, "r", encoding="utf8") as file_reader:
                file_text = file_reader.read().replace("\n", " ")

            offset = 0
            document_text = ""
            if document_id in documents:
                document_text = documents[document_id] + " "
                offset = len(document_text)

            tmp_document_text = document_text + file_text

            entities = []
            dirty_file = False
            with open(text_file[:-4] + ".ann", encoding="utf8") as file_reader:
                for line in file_reader:
                    if line[-1] == "\n":
                        line = line[:-1]
                    if not line:
                        continue

                    columns = line.split("\t")
                    mid = columns[1].split()
                    # if len(mid) != 3:
                    #     continue

                    entity_type, _start, _end = mid[:3]
                    start, end = int(_start.split(";")[0]), int(_end.split(";")[0])

                    if start == end:
                        continue

                    # Try to fix entity offsets
                    if tmp_document_text[offset + start : offset + end] != columns[2]:
                        alt_text = tmp_document_text[offset + start : offset + start + len(columns[2])]
                        if alt_text == columns[2]:
                            end = start + len(columns[2])

                    if file_text[start:end] != columns[2]:
                        dirty_file = True
                        continue

                    if tmp_document_text[offset + start : offset + end] != columns[2]:
                        dirty_file = True
                        continue

                    entities.append(Entity((offset + start, offset + end), entity_type))

            if not dirty_file:
                documents[document_id] = tmp_document_text
                if document_id in entities_per_document:
                    entities_per_document[document_id] += entities
                else:
                    entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class BC2GM(ColumnCorpus):
    """
    Original BioCreative-II-GM corpus containing gene annotations.

    For further information see Smith et al.:
        Overview of BioCreative II gene mention recognition
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2559986/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments documents
            into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        test_file = data_folder / f"{sentence_splitter.name}_test.conll"

        if not (train_file.exists() and test_file.exists()):
            data_folder = self.download_dataset(data_folder)
            train_data = self.parse_train_dataset(data_folder)
            test_data = self.parse_test_dataset(data_folder)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(BC2GM, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path) -> Path:
        data_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2011/bc2GMtrain_1.1.tar.gz"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

        data_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2011/bc2GMtest_1.0.tar.gz"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

        return data_dir

    @classmethod
    def parse_train_dataset(cls, data_folder: Path) -> InternalBioNerDataset:
        train_text_file = data_folder / "bc2geneMention" / "train" / "train.in"
        train_ann_file = data_folder / "bc2geneMention" / "train" / "GENE.eval"

        return cls.parse_dataset(train_text_file, train_ann_file)

    @classmethod
    def parse_test_dataset(cls, data_folder: Path) -> InternalBioNerDataset:
        test_text_file = data_folder / "BC2GM" / "test" / "test.in"
        test_ann_file = data_folder / "BC2GM" / "test" / "GENE.eval"

        return cls.parse_dataset(test_text_file, test_ann_file)

    @staticmethod
    def parse_dataset(text_file: Path, ann_file: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document: Dict[str, List[Entity]] = {}

        with open(str(text_file), "r", encoding="utf8") as text_file_reader:
            for line in text_file_reader:
                line = line.strip()
                offset = line.find(" ")
                document_id = line[:offset]
                document_text = line[offset + 1 :]
                documents[document_id] = document_text
                entities_per_document[document_id] = []

        with open(str(ann_file), "r", encoding="utf8") as ann_file_reader:
            for line in ann_file_reader:
                columns = line.strip().split("|")
                document_id = columns[0]
                document_text = documents[document_id]

                start_idx, end_idx = [int(i) for i in columns[1].split()]

                non_whitespaces_chars = 0
                new_start_idx = None
                new_end_idx = None
                for i, char in enumerate(document_text):
                    if char != " ":
                        non_whitespaces_chars += 1
                    if new_start_idx is None and non_whitespaces_chars == start_idx + 1:
                        new_start_idx = i
                    if non_whitespaces_chars == end_idx + 1:
                        new_end_idx = i + 1
                        break
                assert new_start_idx is not None
                assert new_end_idx is not None
                mention_text = document_text[new_start_idx:new_end_idx]
                if mention_text != columns[2] and mention_text.startswith("/"):
                    # There is still one illegal annotation in the file ..
                    new_start_idx += 1

                entities_per_document[document_id].append(Entity((new_start_idx, new_end_idx), GENE_TAG))

                assert document_text[new_start_idx:new_end_idx] == columns[2]

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_GENE_BC2GM(HunerDataset):
    """
    HUNER version of the BioCreative-II-GM corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bc2gm"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        data_dir = BC2GM.download_dataset(data_dir)
        train_data = BC2GM.parse_train_dataset(data_dir)
        test_data = BC2GM.parse_test_dataset(data_dir)

        return merge_datasets([train_data, test_data])


class CEMP(ColumnCorpus):
    """
    Original CEMP corpus containing chemical annotations.

    For further information see:
         https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/cemp-detailed-task-description/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments
             documents into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{sentence_splitter.name}_dev.conll"

        if not (train_file.exists() and dev_file.exists()):
            train_folder = self.download_train_corpus(data_folder)
            train_text_file = train_folder / "chemdner_patents_train_text.txt"
            train_ann_file = train_folder / "chemdner_cemp_gold_standard_train.tsv"
            train_data = self.parse_input_file(train_text_file, train_ann_file)

            dev_folder = self.download_dev_corpus(data_folder)
            dev_text_file = dev_folder / "chemdner_patents_development_text.txt"
            dev_ann_file = dev_folder / "chemdner_cemp_gold_standard_development_v03.tsv"
            dev_data = self.parse_input_file(dev_text_file, dev_ann_file)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)

        super(CEMP, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_train_corpus(cls, data_dir: Path) -> Path:
        corpus_dir = data_dir / "original"
        os.makedirs(str(corpus_dir), exist_ok=True)

        train_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2015/cemp_training_set.tar.gz"
        data_path = cached_path(train_url, corpus_dir)
        unpack_file(data_path, corpus_dir)

        return corpus_dir / "cemp_training_set"

    @classmethod
    def download_dev_corpus(cls, data_dir) -> Path:
        corpus_dir = data_dir / "original"
        os.makedirs(str(corpus_dir), exist_ok=True)

        dev_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2015/cemp_development_set_v03.tar.gz"
        data_path = cached_path(dev_url, corpus_dir)
        unpack_file(data_path, corpus_dir)

        return corpus_dir / "cemp_development_set_v03"

    @staticmethod
    def parse_input_file(text_file: Path, ann_file: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document: Dict[str, List[Entity]] = {}
        document_abstract_length = {}

        with open(str(text_file), "r", encoding="utf8") as text_reader:
            for line in text_reader:
                if not line:
                    continue

                document_id, title, abstract = line.split("\t")

                # Abstract first, title second to prevent issues with sentence splitting
                documents[document_id] = abstract + " " + title
                document_abstract_length[document_id] = len(abstract) + 1

                entities_per_document[document_id] = []

        with open(str(ann_file), "r", encoding="utf8") as ann_reader:
            for line in ann_reader:
                if not line:
                    continue

                columns = line.split("\t")
                document_id = columns[0]
                start, end = int(columns[2]), int(columns[3])

                if columns[1] == "T":
                    start = start + document_abstract_length[document_id]
                    end = end + document_abstract_length[document_id]

                entities_per_document[document_id].append(Entity((start, end), columns[5].strip()))

                document_text = documents[document_id]
                assert columns[4] == document_text[start:end]

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_CHEMICAL_CEMP(HunerDataset):
    """
    HUNER version of the CEMP corpus containing chemical annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/cemp"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        train_folder = CEMP.download_train_corpus(data_dir)
        train_text_file = train_folder / "chemdner_patents_train_text.txt"
        train_ann_file = train_folder / "chemdner_cemp_gold_standard_train.tsv"
        train_data = CEMP.parse_input_file(train_text_file, train_ann_file)

        dev_folder = CEMP.download_dev_corpus(data_dir)
        dev_text_file = dev_folder / "chemdner_patents_development_text.txt"
        dev_ann_file = dev_folder / "chemdner_cemp_gold_standard_development_v03.tsv"
        dev_data = CEMP.parse_input_file(dev_text_file, dev_ann_file)

        dataset = merge_datasets([train_data, dev_data])
        entity_type_mapping = {
            x: CHEMICAL_TAG
            for x in [
                "ABBREVIATION",
                "FAMILY",
                "FORMULA",
                "IDENTIFIERS",
                "MULTIPLE",
                "SYSTEMATIC",
                "TRIVIAL",
            ]
        }
        return filter_and_map_entities(dataset, entity_type_mapping)


class CHEBI(ColumnCorpus):
    """
    Original CHEBI corpus containing all annotations.

    For further information see Shardlow et al.:
         A New Corpus to Support Text Mining for the Curation of Metabolites in the ChEBI Database
         http://www.lrec-conf.org/proceedings/lrec2018/pdf/229.pdf
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
        annotator: int = 0,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments documents
                into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        :param annotator: The abstracts have been annotated by two annotators, which can be
                selected by choosing annotator 1 or 2. If annotator is 0, the union of both annotations is used.
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not (train_file.exists()):
            corpus_dir = self.download_dataset(data_folder)
            full_dataset = self.parse_dataset(corpus_dir, annotator=annotator)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(full_dataset, train_file)

        super(CHEBI, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    def download_dataset(data_dir: Path) -> Path:
        data_url = "http://www.nactem.ac.uk/chebi/ChEBI.zip"
        data_path = cached_path(data_url, data_dir)
        unpack_file(data_path, data_dir)

        return data_dir / "ChEBI"

    @staticmethod
    def parse_dataset(data_dir: Path, annotator: int) -> InternalBioNerDataset:
        abstract_folder = data_dir / "abstracts"
        fulltext_folder = data_dir / "fullpapers"

        if annotator == 0:
            annotation_dirs = ["Annotator1", "Annotator2"]
        elif annotator <= 2:
            annotation_dirs = [f"Annotator{annotator}"]
        else:
            raise ValueError("Invalid value for annotator")

        documents = {}
        entities_per_document = {}

        abstract_ids = [x.name[:-4] for x in (abstract_folder / annotation_dirs[0]).iterdir() if x.name[-4:] == ".txt"]
        fulltext_ids = [x.name[:-4] for x in fulltext_folder.iterdir() if x.name[-4:] == ".txt"]

        for abstract_id in abstract_ids:
            abstract_id_output = abstract_id + "_A"
            with open(
                abstract_folder / annotation_dirs[0] / f"{abstract_id}.txt",
                "r",
                encoding="utf8",
            ) as f:
                documents[abstract_id_output] = f.read()

            for annotation_dir in annotation_dirs:
                with open(
                    abstract_folder / annotation_dir / f"{abstract_id}.ann",
                    "r",
                    encoding="utf8",
                ) as f:
                    entities = CHEBI.get_entities(f)
            entities_per_document[abstract_id_output] = entities

        for fulltext_id in fulltext_ids:
            fulltext_id_output = fulltext_id + "_F"
            with open(fulltext_folder / f"{fulltext_id}.txt", "r", encoding="utf8") as f:
                documents[fulltext_id_output] = f.read()

            with open(fulltext_folder / f"{fulltext_id}.ann", "r", encoding="utf8") as f:
                entities = CHEBI.get_entities(f)
            entities_per_document[fulltext_id_output] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)

    @staticmethod
    def get_entities(f):
        entities = []
        for line in f:
            if not line.strip() or line[0] != "T":
                continue
            parts = line.split("\t")[1].split()
            entity_type = parts[0]
            char_offsets = " ".join(parts[1:])
            for start_end in char_offsets.split(";"):
                start, end = start_end.split(" ")
                entities += [Entity((int(start), int(end)), entity_type)]

        return entities


class HUNER_CHEMICAL_CHEBI(HunerDataset):
    """
    HUNER version of the CHEBI corpus containing chemical annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/chebi_new"

    def to_internal(self, data_dir: Path, annotator: int = 0) -> InternalBioNerDataset:
        corpus_dir = CHEBI.download_dataset(data_dir)
        dataset = CHEBI.parse_dataset(corpus_dir, annotator=annotator)
        entity_type_mapping = {"Chemical": CHEMICAL_TAG}
        return filter_and_map_entities(dataset, entity_type_mapping)


class HUNER_GENE_CHEBI(HunerDataset):
    """
    HUNER version of the CHEBI corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/chebi_new"

    def to_internal(self, data_dir: Path, annotator: int = 0) -> InternalBioNerDataset:
        corpus_dir = CHEBI.download_dataset(data_dir)
        dataset = CHEBI.parse_dataset(corpus_dir, annotator=annotator)
        entity_type_mapping = {"Protein": GENE_TAG}
        return filter_and_map_entities(dataset, entity_type_mapping)


class HUNER_SPECIES_CHEBI(HunerDataset):
    """
    HUNER version of the CHEBI corpus containing species annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/chebi_new"

    def to_internal(self, data_dir: Path, annotator: int = 0) -> InternalBioNerDataset:
        corpus_dir = CHEBI.download_dataset(data_dir)
        dataset = CHEBI.parse_dataset(corpus_dir, annotator=annotator)
        entity_type_mapping = {"Species": SPECIES_TAG}
        return filter_and_map_entities(dataset, entity_type_mapping)


class BioNLPCorpus(ColumnCorpus):
    """
    Base class for corpora from BioNLP event extraction shared tasks

    For further information see:
         http://2013.bionlp-st.org/Intro
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments documents
             into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{sentence_splitter.name}_dev.conll"
        test_file = data_folder / f"{sentence_splitter.name}_test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            train_folder, dev_folder, test_folder = self.download_corpus(data_folder / "original")

            train_data = self.parse_input_files(train_folder)
            dev_data = self.parse_input_files(dev_folder)
            test_data = self.parse_input_files(test_folder)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(BioNLPCorpus, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    @abstractmethod
    def download_corpus(data_folder: Path) -> Tuple[Path, Path, Path]:
        pass

    @staticmethod
    def parse_input_files(input_folder: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}

        for txt_file in input_folder.glob("*.txt"):
            name = txt_file.with_suffix("").name
            a1_file = txt_file.with_suffix(".a1")

            with txt_file.open(encoding="utf8") as f:
                documents[name] = f.read()

            with a1_file.open(encoding="utf8") as ann_reader:
                entities = []

                for line in ann_reader:
                    fields = line.strip().split("\t")
                    if fields[0].startswith("T"):
                        ann_type, start, end = fields[1].split()
                        entities.append(Entity(char_span=(int(start), int(end)), entity_type=ann_type))
                entities_per_document[name] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class BIONLP2013_PC(BioNLPCorpus):
    """
    Corpus of the BioNLP'2013 Pathway Curation shared task

    For further information see Ohta et al.
        Overview of the pathway curation (PC) task of bioNLP shared task 2013.
        https://www.aclweb.org/anthology/W13-2009/
    """

    @staticmethod
    def download_corpus(download_folder: Path) -> Tuple[Path, Path, Path]:
        train_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_training_data.tar.gz"
        dev_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_development_data.tar.gz"
        test_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_test_data.tar.gz"

        cached_path(train_url, download_folder)
        cached_path(dev_url, download_folder)
        cached_path(test_url, download_folder)

        unpack_file(
            download_folder / "BioNLP-ST_2013_PC_training_data.tar.gz",
            download_folder,
            keep=False,
        )
        unpack_file(
            download_folder / "BioNLP-ST_2013_PC_development_data.tar.gz",
            download_folder,
            keep=False,
        )
        unpack_file(
            download_folder / "BioNLP-ST_2013_PC_test_data.tar.gz",
            download_folder,
            keep=False,
        )

        train_folder = download_folder / "BioNLP-ST_2013_PC_training_data"
        dev_folder = download_folder / "BioNLP-ST_2013_PC_development_data"
        test_folder = download_folder / "BioNLP-ST_2013_PC_test_data"

        return train_folder, dev_folder, test_folder


class BIONLP2013_CG(BioNLPCorpus):
    """
    Corpus of the BioNLP'2013 Cancer Genetics shared task

    For further information see Pyysalo, Ohta & Ananiadou 2013
        Overview of the Cancer Genetics (CG) task of BioNLP Shared Task 2013
        https://www.aclweb.org/anthology/W13-2008/
    """

    @staticmethod
    def download_corpus(download_folder: Path) -> Tuple[Path, Path, Path]:
        train_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_training_data.tar.gz"
        dev_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_development_data.tar.gz"
        test_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_test_data.tar.gz"

        download_folder = download_folder / "original"

        cached_path(train_url, download_folder)
        cached_path(dev_url, download_folder)
        cached_path(test_url, download_folder)

        unpack_file(
            download_folder / "BioNLP-ST_2013_CG_training_data.tar.gz",
            download_folder,
            keep=False,
        )
        unpack_file(
            download_folder / "BioNLP-ST_2013_CG_development_data.tar.gz",
            download_folder,
            keep=False,
        )
        unpack_file(
            download_folder / "BioNLP-ST_2013_CG_test_data.tar.gz",
            download_folder,
            keep=False,
        )

        train_folder = download_folder / "BioNLP-ST_2013_CG_training_data"
        dev_folder = download_folder / "BioNLP-ST_2013_CG_development_data"
        test_folder = download_folder / "BioNLP-ST_2013_CG_test_data"

        return train_folder, dev_folder, test_folder


class ANAT_EM(ColumnCorpus):
    """
    Corpus for anatomical named entity mention recognition.

    For further information see Pyysalo and Ananiadou:
      Anatomical entity mention recognition at literature scale
      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3957068/
      http://nactem.ac.uk/anatomytagger/#AnatEM
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        tokenizer: Tokenizer = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`Tokenizer` which segments
             sentences into tokens (default :class:`SciSpacyTokenizer`)
        """
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if tokenizer is None:
            tokenizer = SciSpacyTokenizer()

        sentence_splitter = TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=tokenizer)

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{sentence_splitter.name}_dev.conll"
        test_file = data_folder / f"{sentence_splitter.name}_test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            corpus_folder = self.download_corpus(data_folder)

            train_data = self.parse_input_files(corpus_folder / "nersuite" / "train", SENTENCE_TAG)
            dev_data = self.parse_input_files(corpus_folder / "nersuite" / "devel", SENTENCE_TAG)
            test_data = self.parse_input_files(corpus_folder / "nersuite" / "test", SENTENCE_TAG)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(ANAT_EM, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @staticmethod
    @abstractmethod
    def download_corpus(data_folder: Path):
        corpus_url = "http://nactem.ac.uk/anatomytagger/AnatEM-1.0.2.tar.gz"
        corpus_archive = cached_path(corpus_url, data_folder)

        unpack_file(
            corpus_archive,
            data_folder,
            keep=True,
            mode="targz",
        )

        return data_folder / "AnatEM-1.0.2"

    @staticmethod
    def parse_input_files(input_dir: Path, sentence_separator: str) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}

        input_files = [
            file for file in os.listdir(str(input_dir)) if file.endswith(".nersuite") and not file.startswith("._")
        ]

        for input_file in input_files:
            document_id = input_file.replace(".nersuite", "")
            document_text = ""

            entities = []
            entity_type = None
            entity_start = None

            sent_offset = 0
            last_offset = 0

            with open(input_dir / input_file, "r", encoding="utf8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        document_text += sentence_separator
                        sent_offset += len(sentence_separator)
                        last_offset += len(sentence_separator)
                        continue
                    tag, _start, _end, word, _, _, _ = line.split("\t")

                    start = int(_start) + sent_offset
                    end = int(_end) + sent_offset

                    document_text += " " * (start - last_offset)
                    document_text += word

                    if tag.startswith("B-"):
                        if entity_type is not None:
                            entities.append(Entity((entity_start, last_offset), entity_type))

                        entity_start = start
                        entity_type = tag[2:]

                    elif tag == "O" and entity_type is not None and entity_start is not None:
                        entities.append(Entity((entity_start, last_offset), entity_type))
                        entity_type = None

                    last_offset = end

                    assert word == document_text[start:end]

            documents[document_id] = document_text
            entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class BioBertHelper(ColumnCorpus):
    """
    Helper class to convert corpora and the respective train, dev and test split
    used by BioBERT.

    For further details see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    @staticmethod
    def download_corpora(download_dir: Path):
        from google_drive_downloader import GoogleDriveDownloader as gdd

        gdd.download_file_from_google_drive(
            file_id="1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh",
            dest_path=str(download_dir / "NERdata.zip"),
            unzip=True,
        )

    @staticmethod
    def convert_and_write(download_folder, data_folder, tag_type):
        data_folder.mkdir(parents=True, exist_ok=True)
        with (download_folder / "train.tsv").open(encoding="utf8") as f_in, (data_folder / "train.conll").open(
            "w", encoding="utf8"
        ) as f_out:
            for line in f_in:
                if not line.strip():
                    f_out.write("\n")
                    continue

                token, tag = line.strip().split("\t")
                if tag != "O":
                    tag = tag + "-" + tag_type
                f_out.write(f"{token} {tag}\n")

        with (download_folder / "devel.tsv").open(encoding="utf8") as f_in, (data_folder / "dev.conll").open(
            "w", encoding="utf8"
        ) as f_out:
            for line in f_in:
                if not line.strip():
                    f_out.write("\n")
                    continue
                token, tag = line.strip().split("\t")
                if tag != "O":
                    tag = tag + "-" + tag_type
                f_out.write(f"{token} {tag}\n")

        with (download_folder / "test.tsv").open(encoding="utf8") as f_in, (data_folder / "test.conll").open(
            "w", encoding="utf8"
        ) as f_out:
            for line in f_in:
                if not line.strip():
                    f_out.write("\n")
                    continue
                token, tag = line.strip().split("\t")
                if tag != "O":
                    tag = tag + "-" + tag_type
                f_out.write(f"{token} {tag}\n")


class BIOBERT_CHEMICAL_BC4CHEMD(ColumnCorpus):
    """
    BC4CHEMD corpus with chemical annotations as used in the evaluation
    of BioBERT.

    For further details regarding BioBERT and it's evaluation, see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        columns = {0: "text", 1: "ner"}
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            common_path = base_path / "biobert_common"
            if not (common_path / "BC4CHEMD").exists():
                BioBertHelper.download_corpora(common_path)

            BioBertHelper.convert_and_write(common_path / "BC4CHEMD", data_folder, tag_type=CHEMICAL_TAG)
        super(BIOBERT_CHEMICAL_BC4CHEMD, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class BIOBERT_GENE_BC2GM(ColumnCorpus):
    """
    BC4CHEMD corpus with gene annotations as used in the evaluation
    of BioBERT.

    For further details regarding BioBERT and it's evaluation, see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        columns = {0: "text", 1: "ner"}
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            common_path = base_path / "biobert_common"
            if not (common_path / "BC2GM").exists():
                BioBertHelper.download_corpora(common_path)
            BioBertHelper.convert_and_write(common_path / "BC2GM", data_folder, tag_type=GENE_TAG)
        super(BIOBERT_GENE_BC2GM, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class BIOBERT_GENE_JNLPBA(ColumnCorpus):
    """
    JNLPBA corpus with gene annotations as used in the evaluation
    of BioBERT.

    For further details regarding BioBERT and it's evaluation, see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        columns = {0: "text", 1: "ner"}
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            common_path = base_path / "biobert_common"
            if not (common_path / "JNLPBA").exists():
                BioBertHelper.download_corpora(common_path)
            BioBertHelper.convert_and_write(common_path / "JNLPBA", data_folder, tag_type=GENE_TAG)
        super(BIOBERT_GENE_JNLPBA, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class BIOBERT_CHEMICAL_BC5CDR(ColumnCorpus):
    """
    BC5CDR corpus with chemical annotations as used in the evaluation
    of BioBERT.

    For further details regarding BioBERT and it's evaluation, see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        columns = {0: "text", 1: "ner"}
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            common_path = base_path / "biobert_common"
            if not (common_path / "BC5CDR-chem").exists():
                BioBertHelper.download_corpora(common_path)
            BioBertHelper.convert_and_write(common_path / "BC5CDR-chem", data_folder, tag_type=CHEMICAL_TAG)
        super(BIOBERT_CHEMICAL_BC5CDR, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class BIOBERT_DISEASE_BC5CDR(ColumnCorpus):
    """
    BC5CDR corpus with disease annotations as used in the evaluation
    of BioBERT.

    For further details regarding BioBERT and it's evaluation, see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        columns = {0: "text", 1: "ner"}
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            common_path = base_path / "biobert_common"
            if not (common_path / "BC5CDR-disease").exists():
                BioBertHelper.download_corpora(common_path)
            BioBertHelper.convert_and_write(common_path / "BC5CDR-disease", data_folder, tag_type=DISEASE_TAG)
        super(BIOBERT_DISEASE_BC5CDR, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class BIOBERT_DISEASE_NCBI(ColumnCorpus):
    """
    NCBI disease corpus as used in the evaluation of BioBERT.

    For further details regarding BioBERT and it's evaluation, see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        columns = {0: "text", 1: "ner"}
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            common_path = base_path / "biobert_common"
            if not (common_path / "NCBI-disease").exists():
                BioBertHelper.download_corpora(common_path)
            BioBertHelper.convert_and_write(common_path / "NCBI-disease", data_folder, tag_type=DISEASE_TAG)
        super(BIOBERT_DISEASE_NCBI, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class BIOBERT_SPECIES_LINNAEUS(ColumnCorpus):
    """
    Linneaeus corpus with species annotations as used in the evaluation
    of BioBERT.

    For further details regarding BioBERT and it's evaluation, see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        columns = {0: "text", 1: "ner"}
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            common_path = base_path / "biobert_common"
            if not (common_path / "linnaeus").exists():
                BioBertHelper.download_corpora(common_path)
            BioBertHelper.convert_and_write(common_path / "linnaeus", data_folder, tag_type=SPECIES_TAG)
        super(BIOBERT_SPECIES_LINNAEUS, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class BIOBERT_SPECIES_S800(ColumnCorpus):
    """
    S800 corpus with species annotations as used in the evaluation
    of BioBERT.

    For further details regarding BioBERT and it's evaluation, see Lee et al.:
        https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
        https://github.com/dmis-lab/biobert
    """

    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True):
        columns = {0: "text", 1: "ner"}
        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            common_path = base_path / "biobert_common"
            if not (common_path / "s800").exists():
                BioBertHelper.download_corpora(common_path)
            BioBertHelper.convert_and_write(common_path / "s800", data_folder, tag_type=SPECIES_TAG)
        super(BIOBERT_SPECIES_S800, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)


class CRAFT_V4(ColumnCorpus):
    """
    Version 4.0.1 of the CRAFT corpus containing all but the co-reference and structural annotations.

    For further information see:
      https://github.com/UCDenver-ccp/CRAFT
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which segments
             documents into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"
        dev_file = data_folder / f"{sentence_splitter.name}_dev.conll"
        test_file = data_folder / f"{sentence_splitter.name}_test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            corpus_dir = self.download_corpus(data_folder)
            corpus_data = self.parse_corpus(corpus_dir)

            # Filter for specific entity types, by default no entities will be filtered
            corpus_data = self.filter_entities(corpus_data)

            train_data, dev_data, test_data = self.prepare_splits(data_folder, corpus_data)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(CRAFT_V4, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    def filter_entities(self, corpus: InternalBioNerDataset) -> InternalBioNerDataset:
        return corpus

    @classmethod
    def download_corpus(cls, data_dir: Path) -> Path:
        url = "https://github.com/UCDenver-ccp/CRAFT/archive/v4.0.1.tar.gz"
        data_path = cached_path(url, data_dir)
        unpack_file(data_path, data_dir, mode="targz")

        return data_dir / "CRAFT-4.0.1"

    @staticmethod
    def prepare_splits(
        data_dir: Path, corpus: InternalBioNerDataset
    ) -> Tuple[InternalBioNerDataset, InternalBioNerDataset, InternalBioNerDataset]:
        splits_dir = data_dir / "splits"
        os.makedirs(str(splits_dir), exist_ok=True)

        # Get original HUNER splits to retrieve a list of all document ids contained in V2
        split_urls = [
            "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/craft.train",
            "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/craft.dev",
            "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/craft.test",
        ]

        splits = {}
        for url in split_urls:
            split_file = cached_path(url, splits_dir)
            with open(str(split_file), "r", encoding="utf8") as split_reader:
                splits[url.split(".")[-1]] = [line.strip() for line in split_reader if line.strip()]

        train_documents, train_entities = {}, {}
        dev_documents, dev_entities = {}, {}
        test_documents, test_entities = {}, {}

        for document_id, document_text in corpus.documents.items():
            if document_id in splits["train"] or document_id in splits["dev"]:
                # train and dev split of V2 will be train in V4
                train_documents[document_id] = document_text
                train_entities[document_id] = corpus.entities_per_document[document_id]
            elif document_id in splits["test"]:
                # test split of V2 will be dev in V4
                dev_documents[document_id] = document_text
                dev_entities[document_id] = corpus.entities_per_document[document_id]
            else:
                # New documents in V4 will become test documents
                test_documents[document_id] = document_text
                test_entities[document_id] = corpus.entities_per_document[document_id]

        train_corpus = InternalBioNerDataset(documents=train_documents, entities_per_document=train_entities)
        dev_corpus = InternalBioNerDataset(documents=dev_documents, entities_per_document=dev_entities)
        test_corpus = InternalBioNerDataset(documents=test_documents, entities_per_document=test_entities)

        return train_corpus, dev_corpus, test_corpus

    @staticmethod
    def parse_corpus(corpus_dir: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}

        text_dir = corpus_dir / "articles" / "txt"
        document_texts = [doc for doc in text_dir.iterdir() if doc.name[-4:] == ".txt"]
        annotation_dirs = [
            path
            for path in (corpus_dir / "concept-annotation").iterdir()
            if path.name not in ["sections-and-typography", "coreference"] and path.is_dir()
        ]

        for doc in Tqdm.tqdm(document_texts, desc="Converting to internal"):
            document_id = doc.name.split(".")[0]

            with open(doc, "r", encoding="utf8") as f_txt:
                documents[document_id] = f_txt.read()

            entities = []

            for annotation_dir in annotation_dirs:
                with open(
                    annotation_dir / annotation_dir.parts[-1] / "knowtator" / (doc.name + ".knowtator.xml"),
                    "r",
                    encoding="utf8",
                ) as f_ann:
                    ann_tree = etree.parse(f_ann)
                for annotation in ann_tree.xpath("//annotation"):
                    for span in annotation.xpath("span"):
                        start = int(span.get("start"))
                        end = int(span.get("end"))
                        entities += [Entity((start, end), annotation_dir.name.lower())]

            entities_per_document[document_id] = entities

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class HUNER_CHEMICAL_CRAFT_V4(HunerDataset):
    """
    HUNER version of the CRAFT corpus containing (only) chemical annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/craft_v4"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        corpus_dir = CRAFT_V4.download_corpus(data_dir)
        corpus = CRAFT_V4.parse_corpus(corpus_dir)

        entity_type_mapping = {"chebi": CHEMICAL_TAG}
        return filter_and_map_entities(corpus, entity_type_mapping)


class HUNER_GENE_CRAFT_V4(HunerDataset):
    """
    HUNER version of the CRAFT corpus containing (only) gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/craft_v4"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        corpus_dir = CRAFT_V4.download_corpus(data_dir)
        corpus = CRAFT_V4.parse_corpus(corpus_dir)

        entity_type_mapping = {"pr": GENE_TAG}
        return filter_and_map_entities(corpus, entity_type_mapping)


class HUNER_SPECIES_CRAFT_V4(HunerDataset):
    """
    HUNER version of the CRAFT corpus containing (only) species annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/craft_v4"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        corpus_dir = CRAFT_V4.download_corpus(data_dir)
        corpus = CRAFT_V4.parse_corpus(corpus_dir)

        entity_type_mapping = {"ncbitaxon": SPECIES_TAG}
        return filter_and_map_entities(corpus, entity_type_mapping)


class HUNER_CHEMICAL_BIONLP2013_CG(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bionlp2013_cg"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        train_dir, dev_dir, test_dir = BIONLP2013_CG.download_corpus(data_dir)
        train_corpus = BioNLPCorpus.parse_input_files(train_dir)
        dev_corpus = BioNLPCorpus.parse_input_files(dev_dir)
        test_corpus = BioNLPCorpus.parse_input_files(test_dir)
        corpus = merge_datasets([train_corpus, dev_corpus, test_corpus])

        entity_type_mapping = {"Simple_chemical": CHEMICAL_TAG}
        return filter_and_map_entities(corpus, entity_type_mapping)


class HUNER_DISEASE_BIONLP2013_CG(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bionlp2013_cg"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        train_dir, dev_dir, test_dir = BIONLP2013_CG.download_corpus(data_dir)
        train_corpus = BioNLPCorpus.parse_input_files(train_dir)
        dev_corpus = BioNLPCorpus.parse_input_files(dev_dir)
        test_corpus = BioNLPCorpus.parse_input_files(test_dir)
        corpus = merge_datasets([train_corpus, dev_corpus, test_corpus])

        entity_type_mapping = {"Cancer": DISEASE_TAG}
        return filter_and_map_entities(corpus, entity_type_mapping)


class HUNER_GENE_BIONLP2013_CG(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bionlp2013_cg"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        train_dir, dev_dir, test_dir = BIONLP2013_CG.download_corpus(data_dir)
        train_corpus = BioNLPCorpus.parse_input_files(train_dir)
        dev_corpus = BioNLPCorpus.parse_input_files(dev_dir)
        test_corpus = BioNLPCorpus.parse_input_files(test_dir)
        corpus = merge_datasets([train_corpus, dev_corpus, test_corpus])

        entity_type_mapping = {"Gene_or_gene_product": GENE_TAG}
        return filter_and_map_entities(corpus, entity_type_mapping)


class HUNER_SPECIES_BIONLP2013_CG(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bionlp2013_cg"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        train_dir, dev_dir, test_dir = BIONLP2013_CG.download_corpus(data_dir)
        train_corpus = BioNLPCorpus.parse_input_files(train_dir)
        dev_corpus = BioNLPCorpus.parse_input_files(dev_dir)
        test_corpus = BioNLPCorpus.parse_input_files(test_dir)
        corpus = merge_datasets([train_corpus, dev_corpus, test_corpus])

        entity_type_mapping = {"Organism": SPECIES_TAG}
        return filter_and_map_entities(corpus, entity_type_mapping)


class AZDZ(ColumnCorpus):
    """
    Arizona Disease Corpus from the Biomedical Informatics Lab at Arizona State University.

     For further information see:
       http://diego.asu.edu/index.php
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        tokenizer: Tokenizer = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param tokenizer: Implementation of :class:`Tokenizer` which segments sentences
             into tokens (default :class:`SciSpacyTokenizer`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        if tokenizer is None:
            tokenizer = SciSpacyTokenizer()
        sentence_splitter = TagSentenceSplitter(tag=SENTENCE_TAG, tokenizer=tokenizer)

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not train_file.exists():
            corpus_file = self.download_corpus(data_folder)
            corpus_data = self.parse_corpus(corpus_file)

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(corpus_data, train_file)

        super(AZDZ, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_corpus(cls, data_dir: Path) -> Path:
        url = "http://diego.asu.edu/downloads/AZDC_6-26-2009.txt"
        data_path = cached_path(url, data_dir)

        return data_path

    @staticmethod
    def parse_corpus(input_file: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}

        with open(str(input_file), "r", encoding="iso-8859-1") as azdz_reader:
            prev_document_id: Optional[str] = None
            prev_sentence_id: Optional[str] = None

            document_text: Optional[str] = None
            entities: List[Entity] = []
            offset: Optional[int] = None

            for line in azdz_reader:
                line = line.strip()
                if not line or line.startswith("Doc Id"):
                    continue

                pmid, sentence_no, text, entity_start, entity_end = line.split("\t")

                document_id = pmid
                sentence_id = document_id + "_" + sentence_no

                if document_id != prev_document_id and document_text:
                    documents[document_id] = document_text
                    entities_per_document[document_id] = entities

                    document_text = None
                    entities = []
                    offset = None

                if sentence_id != prev_sentence_id:
                    offset = offset + len(SENTENCE_TAG) if offset is not None else 0
                    document_text = document_text + SENTENCE_TAG + text.strip() if document_text is not None else text

                if offset is None:
                    continue

                try:
                    start = offset + int(entity_start) - 1
                    end = offset + int(entity_end)
                except ValueError:
                    continue

                if end == 0:
                    continue

                entities.append(Entity((start, end), DISEASE_TAG))

        return InternalBioNerDataset(documents=documents, entities_per_document=entities_per_document)


class PDR(ColumnCorpus):
    """
    Corpus of plant-disease relations from Kim et al., consisting of named entity annotations
    for plants and disease.

      For further information see Kim et al.:
        A corpus of plant-disease relations in the biomedical domain
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221582
        http://gcancer.org/pdr/
    """

    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        sentence_splitter: SentenceSplitter = None,
    ):
        """
        :param base_path: Path to the corpus on your machine
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param sentence_splitter: Implementation of :class:`SentenceSplitter` which
             segments documents into sentences and tokens (default :class:`SciSpacySentenceSplitter`)
        """

        if base_path is None:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner", 2: ColumnDataset.SPACE_AFTER_KEY}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root

        data_folder = base_path / dataset_name

        if sentence_splitter is None:
            sentence_splitter = SciSpacySentenceSplitter()

        train_file = data_folder / f"{sentence_splitter.name}_train.conll"

        if not train_file.exists():
            corpus_folder = self.download_corpus(data_folder)
            corpus_data = brat_to_internal(corpus_folder, ann_file_suffixes=[".ann", ".ann2"])

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(corpus_data, train_file)

        super(PDR, self).__init__(data_folder, columns, tag_to_bioes="ner", in_memory=in_memory)

    @classmethod
    def download_corpus(cls, data_dir: Path) -> Path:
        url = "http://gcancer.org/pdr/Plant-Disease_Corpus.tar.gz"
        data_path = cached_path(url, data_dir)
        unpack_file(data_path, data_dir)

        return data_dir / "Plant-Disease_Corpus"


class HUNER_DISEASE_PDR(HunerDataset):
    """
    PDR Dataset with only Disease annotations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/pdr"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        corpus_folder = PDR.download_corpus(data_dir)
        corpus_data = brat_to_internal(corpus_folder, ann_file_suffixes=[".ann", ".ann2"])
        corpus_data = filter_and_map_entities(corpus_data, {"Disease": DISEASE_TAG})

        return corpus_data


class HunerMultiCorpus(MultiCorpus):
    """
    Base class to build the union of all HUNER data sets considering a particular entity type.
    """

    def __init__(self, entity_type: str, sentence_splitter: SentenceSplitter = None):
        self.entity_type = entity_type

        def entity_type_predicate(member):
            return f"HUNER_{entity_type}_" in str(member) and inspect.isclass(member)

        self.huner_corpora_classes = inspect.getmembers(sys.modules[__name__], predicate=entity_type_predicate)
        self.huner_corpora = []
        for name, constructor_func in self.huner_corpora_classes:
            try:
                if not sentence_splitter:
                    corpus = constructor_func()
                else:
                    corpus = constructor_func(sentence_splitter=sentence_splitter)

                self.huner_corpora.append(corpus)
            except (CompressionError, ExtractError, HeaderError, ReadError, StreamError, TarError):
                logger.exception(
                    f"Error while processing Tar file from corpus {name}:\n{sys.exc_info()[1]}\n\n", exc_info=False
                )
            except (BadZipFile, LargeZipFile):
                logger.exception(
                    f"Error while processing Zip file from corpus {name}:\n{sys.exc_info()[1]}\n\n", exc_info=False
                )
            except IOError:
                logger.exception(
                    f"Error while downloading data for corpus {name}:\n{sys.exc_info()[1]}\n\n", exc_info=False
                )
            except shutil.Error:
                logger.exception(
                    f"Error while copying data files for corpus {name}:\n{sys.exc_info()[1]}\n\n", exc_info=False
                )
            except etree.LxmlError:
                logger.exception(
                    f"Error while processing XML file from corpus {name}:\n{sys.exc_info()[1]}\n\n", exc_info=False
                )
            except json.JSONDecodeError:
                logger.exception(
                    f"Error while processing JSON file from corpus {name}:\n{sys.exc_info()[1]}\n\n", exc_info=False
                )
            except (FileNotFoundError, OSError, ValueError):
                logger.exception(f"Error while preparing corpus {name}:\n{sys.exc_info()[1]}\n\n", exc_info=False)

        super(HunerMultiCorpus, self).__init__(corpora=self.huner_corpora, name=f"HUNER-{entity_type}")


class HUNER_CELL_LINE(HunerMultiCorpus):
    """
    Union of all HUNER cell line data sets.
    """

    def __init__(self, sentence_splitter: SentenceSplitter = None):
        super(HUNER_CELL_LINE, self).__init__(entity_type="CELL_LINE", sentence_splitter=sentence_splitter)


class HUNER_CHEMICAL(HunerMultiCorpus):
    """
    Union of all HUNER chemical data sets.
    """

    def __init__(self, sentence_splitter: SentenceSplitter = None):
        super(HUNER_CHEMICAL, self).__init__(entity_type="CHEMICAL", sentence_splitter=sentence_splitter)


class HUNER_DISEASE(HunerMultiCorpus):
    """
    Union of all HUNER disease data sets.
    """

    def __init__(self, sentence_splitter: SentenceSplitter = None):
        super(HUNER_DISEASE, self).__init__(entity_type="DISEASE", sentence_splitter=sentence_splitter)


class HUNER_GENE(HunerMultiCorpus):
    """
    Union of all HUNER gene data sets.
    """

    def __init__(self, sentence_splitter: SentenceSplitter = None):
        super(HUNER_GENE, self).__init__(entity_type="GENE", sentence_splitter=sentence_splitter)


class HUNER_SPECIES(HunerMultiCorpus):
    """
    Union of all HUNER species data sets.
    """

    def __init__(self, sentence_splitter: SentenceSplitter = None):
        super(HUNER_SPECIES, self).__init__(entity_type="SPECIES", sentence_splitter=sentence_splitter)
