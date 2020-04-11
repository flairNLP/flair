import json
import os
import shutil

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from copy import copy
from functools import cmp_to_key
from itertools import combinations
from operator import attrgetter
from pathlib import Path
from typing import Union, Callable, Dict, List, Tuple, Iterable

import ftfy
from lxml import etree

import flair
from flair.datasets import ColumnCorpus
from flair.file_utils import cached_path, unzip_file, unzip_targz_file, Tqdm

DISEASE_TAG = "Disease"
CHEMICAL_TAG = "Chemical"
CELL_LINE_TAG = "CellLine"
GENE_TAG = "Gene"
SPECIES_TAG = "Species"


class Entity:
    def __init__(self, char_span: Tuple[int, int], entity_type: str):
        self.char_span = range(*char_span)
        self.type = entity_type

    def __str__(self):
        return (
            self.type
            + "("
            + str(self.char_span.start)
            + ","
            + str(self.char_span.stop)
            + ")"
        )

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
            other_entity.char_span.start >= self.char_span.start
            and other_entity.char_span.stop <= self.char_span.stop
        )

    def overlaps(self, other_entity) -> bool:
        """
        Checks whether this and the given entity overlap

        :param other_entity: Entity to check
        """
        return (
            self.char_span.start <= other_entity.char_span.start < self.char_span.stop
        ) or (self.char_span.start < other_entity.char_span.stop <= self.char_span.stop)


class NestedEntity(Entity):
    def __init__(
        self,
        char_span: Tuple[int, int],
        entity_type: str,
        nested_entities: Iterable[Entity],
    ):
        super(NestedEntity, self).__init__(char_span, entity_type)
        self.nested_entities = nested_entities


class InternalBioNerDataset:
    def __init__(
        self, documents: Dict[str, str], entities_per_document: Dict[str, List[Entity]]
    ):
        self.documents = documents
        self.entities_per_document = entities_per_document


def overlap(entity1, entity2):
    return range(max(entity1[0], entity2[0]), min(entity1[1], entity2[1]))


def compare_by_start_and_length(entity1, entity2):
    start_offset = entity1.char_span.start - entity2.char_span.start
    return (
        start_offset
        if start_offset != 0
        else len(entity2.char_span) - len(entity1.char_span)
    )


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


def merge_datasets(data_sets: Iterable[InternalBioNerDataset]):
    all_documents = {}
    all_entities = {}

    for ds in data_sets:
        all_documents.update(ds.documents)
        all_entities.update(ds.entities_per_document)

    return InternalBioNerDataset(
        documents=all_documents, entities_per_document=all_entities
    )


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
        mapped_entities_per_document[id] = new_entities

    return InternalBioNerDataset(
        documents=dataset.documents, entities_per_document=mapped_entities_per_document
    )


def find_overlapping_entities(
    entities: Iterable[Entity],
) -> List[Tuple[Entity, Entity]]:
    # Sort the entities by their start offset
    entities = sorted(entities, key=lambda e: e.char_span.start)

    overlapping_entities = []
    for i in range(0, len(entities)):
        current_entity = entities[i]
        for other_entity in entities[i + 1 :]:
            if current_entity.overlaps(other_entity):
                # Entities overlap!
                overlapping_entities.append((current_entity, other_entity))
            else:
                # Second entity is located after the current one!
                break

    return overlapping_entities


def find_nested_entities(entities: Iterable[Entity]) -> List[NestedEntity]:
    # Sort entities by start offset and length (i.e. rank longer entity spans first)
    entities = sorted(entities, key=cmp_to_key(compare_by_start_and_length))

    # Initial list with entities and whether they are already contained in a nested entity
    entities = [(entity, False) for entity in entities]

    nested_entities = []
    for i in range(0, len(entities)):
        current_entity, is_part_of_other_entity = entities[i]
        if is_part_of_other_entity:
            continue

        contained_entities = []
        for j in range(i + 1, len(entities)):
            other_entity, _ = entities[j]

            if current_entity.is_before(other_entity):
                # other_entity is located after the current one
                break

            elif current_entity.contains(other_entity):
                # other_entity is contained in current_entity
                contained_entities.append(other_entity)
                entities[j] = (other_entity, True)

        if len(contained_entities) > 0:
            nested_entities.append(
                NestedEntity(
                    (current_entity.char_span.start, current_entity.char_span.stop),
                    current_entity.type,
                    contained_entities,
                )
            )

    return nested_entities


def normalize_entity_spans(entities: Iterable[Entity]) -> List[Entity]:
    # Sort entities by start offset and length (i.e. rank longer entity spans first)
    entities = sorted(entities, key=cmp_to_key(compare_by_start_and_length))

    for i in range(0, len(entities)):
        current_entity = entities[i]
        if current_entity is None:
            continue

        contained_entities = []
        for j in range(i + 1, len(entities)):
            other_entity = entities[j]
            if other_entity is None:
                continue

            if current_entity.is_before(other_entity):
                # other_entity is located after the current one
                break

            elif current_entity.contains(other_entity):
                # other entity is nested in the current one
                contained_entities.append((other_entity, j))

            elif current_entity.overlaps(other_entity):
                # Shift overlapping entities
                shifted_entity = Entity(
                    (current_entity.char_span.stop, other_entity.char_span.stop),
                    other_entity.type,
                )
                entities[j] = shifted_entity

        if len(contained_entities) == 1:
            # Only one smaller entity span is contained -> take the longer one and erase the shorter one
            contained_entity, position = contained_entities[0]
            entities[position] = None

        elif len(contained_entities) > 1:
            # Wrapper for sorting entries by start offset and length
            def compare_entries(entry1, entry2):
                return compare_by_start_and_length(entry1[0], entry2[0])

            contained_entities = sorted(
                contained_entities, key=cmp_to_key(compare_entries)
            )

            # Keep first nested entity
            current_contained_entity = contained_entities[0][0]

            # Fill the complete span successively with non-overlapping entities
            for other_contained_entity, position in contained_entities[1:]:
                if current_contained_entity.is_before(other_contained_entity):
                    current_contained_entity = other_contained_entity
                else:
                    # Entities overlap - erase other contained entity!
                    # FIXME: Shift overlapping entity alternatively?
                    entities[position] = None

            # Erase longer entity
            entities[i] = None

    return [entity for entity in entities if entity is not None]


def bioc_to_internal(bioc_file: Path):
    tree = etree.parse(str(bioc_file))
    texts_per_document = {}
    entities_per_document = {}
    documents = tree.xpath(".//document")

    for document in Tqdm.tqdm(documents, desc="Converting to internal"):
        document_id = document.xpath("./id")[0].text
        texts = []
        entities = []

        for passage in document.xpath("passage"):
            text = passage.xpath("text/text()")[0]
            passage_offset = int(
                passage.xpath("./offset/text()")[0]
            )  # from BioC annotation
            document_offset = len(
                " ".join(texts)
            )  # because we stick all passages of a document together

            texts.append(text)  # calculate offset without current text

            for annotation in passage.xpath(".//annotation"):

                entity_types = [
                    i.text.replace(" ", "_")
                    for i in annotation.xpath("./infon")
                    if i.attrib["key"] in {"type", "class"}
                ]

                start = (
                    int(annotation.xpath("./location")[0].get("offset"))
                    - passage_offset
                )
                # TODO For split entities we also annotate everything inbetween which might be a bad idea?
                final_length = int(annotation.xpath("./location")[-1].get("length"))
                final_offset = (
                    int(annotation.xpath("./location")[-1].get("offset"))
                    - passage_offset
                )
                if final_length <= 0:
                    continue
                end = final_offset + final_length
                annotated_entity = text[start:end]
                true_entity = annotation.xpath(".//text")[0].text
                # assert annotated_entity.lower() == true_entity.lower()

                for entity_type in entity_types:
                    entities.append(
                        Entity(
                            (start + document_offset, end + document_offset),
                            entity_type,
                        )
                    )
        texts_per_document[document_id] = " ".join(texts)
        entities_per_document[document_id] = entities

    return InternalBioNerDataset(
        documents=texts_per_document, entities_per_document=entities_per_document
    )


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

    def process_dataset(
        self, datasets: Dict[str, InternalBioNerDataset], out_dir: Path
    ):
        self.write_to_conll(datasets["train"], out_dir / "train.conll")
        self.write_to_conll(datasets["dev"], out_dir / "dev.conll")
        self.write_to_conll(datasets["test"], out_dir / "test.conll")

    def write_to_conll(self, dataset: InternalBioNerDataset, output_file: Path):
        os.makedirs(str(output_file.parent), exist_ok=True)

        with output_file.open("w") as f:
            for document_id in Tqdm.tqdm(dataset.documents.keys(),
                                         total=len(dataset.documents),
                                         desc="Converting to CoNLL"):
                document_text = ftfy.fix_text(dataset.documents[document_id])
                sentences, sentence_offsets = self.sentence_splitter(document_text)
                entities = deque(
                    sorted(
                        dataset.entities_per_document[document_id],
                        key=attrgetter("char_span.start", "char_span.stop"),
                    )
                )

                current_entity = entities.popleft() if entities else None

                in_entity = False
                for sentence, sentence_offset in zip(sentences, sentence_offsets):
                    tokens, token_offsets = self.tokenizer(sentence)
                    for token, token_offset in zip(tokens, token_offsets):
                        offset = sentence_offset + token_offset

                        if current_entity and offset >= current_entity.char_span.stop:
                            in_entity = False

                            # One token may contain multiple entities -> deque all of them
                            while (
                                current_entity
                                and offset >= current_entity.char_span.stop
                            ):
                                current_entity = (
                                    entities.popleft() if entities else None
                                )

                        # FIXME This assumes that entities aren't nested, we have to ensure that beforehand
                        if current_entity and offset in current_entity.char_span:
                            if not in_entity:
                                tag = "B-" + current_entity.type
                                in_entity = True
                            else:
                                tag = "I-" + current_entity.type
                        else:
                            tag = "O"
                            in_entity = False

                        f.write(" ".join([token, tag]) + "\n")
                    f.write("\n")


def whitespace_tokenize(text):
    offset = 0
    tokens = []
    offsets = []
    for token in text.split():
        tokens.append(token)
        offsets.append(offset)
        offset += len(token) + 1

    return tokens, offsets


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


def build_spacy_tokenizer() -> SciSpacyTokenizer:
    try:
        import spacy

        return SciSpacyTokenizer()
    except ImportError:
        raise ValueError(
            "Default tokenizer is scispacy."
            " Install packages 'scispacy' and"
            " 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
            "/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz' via pip"
            " or choose a different tokenizer"
        )


def build_spacy_sentence_splitter() -> SciSpacySentenceSplitter:
    try:
        import spacy

        return SciSpacySentenceSplitter()
    except ImportError:
        raise ValueError(
            "Default sentence splitter is scispacy."
            " Install packages 'scispacy' and"
            "'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
            "/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz' via pip"
            " or choose a different sentence splitter"
        )


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
            internal_dataset = self.to_internal(data_folder)

            splits_dir = data_folder / "splits"
            os.makedirs(splits_dir, exist_ok=True)

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter,
            )

            train_data = self.get_subset(internal_dataset, "train", splits_dir)
            writer.write_to_conll(train_data, train_file)

            dev_data = self.get_subset(internal_dataset, "dev", splits_dir)
            writer.write_to_conll(dev_data, dev_file)

            test_data = self.get_subset(internal_dataset, "test", splits_dir)
            writer.write_to_conll(test_data, test_file)

        super(HunerDataset, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    def get_subset(self, dataset: InternalBioNerDataset, split: str, split_dir: Path):
        split_file = cached_path(f"{self.split_url()}.{split}", split_dir)

        with split_file.open() as f:
            ids = [l.strip() for l in f if l.strip()]
            ids = sorted(id_ for id_ in ids if id_ in dataset.documents)

        return InternalBioNerDataset(
            documents={k: dataset.documents[k] for k in ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in ids},
        )


class HUNER_GENE_BIO_INFER(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bioinfer"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = defaultdict(list)

        data_url = "http://mars.cs.utu.fi/BioInfer/files/BioInfer_corpus_1.1.1.zip"
        data_path = cached_path(data_url, data_dir)
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
                                Entity((entity_start, token_offset - 1), "protein")
                            )
                            entity_start = None
        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
        )


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
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and test_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)

            train_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz"
            train_data_path = cached_path(train_data_url, download_dir)
            unzip_targz_file(train_data_path, download_dir)

            train_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz"
            train_data_path = cached_path(train_data_url, download_dir)
            unzip_targz_file(train_data_path, download_dir)

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


class HunerJNLPBA:
    @classmethod
    def download_and_prepare_train(cls, data_folder: Path) -> InternalBioNerDataset:
        train_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz"
        train_data_path = cached_path(train_data_url, data_folder)
        unzip_targz_file(train_data_path, data_folder)

        train_input_file = data_folder / "Genia4ERtask2.iob2"
        return cls.read_file(train_input_file)

    @classmethod
    def download_and_prepare_test(cls, data_folder: Path) -> InternalBioNerDataset:
        test_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz"
        test_data_path = cached_path(test_data_url, data_folder)
        unzip_targz_file(test_data_path, data_folder)

        test_input_file = data_folder / "Genia4EReval2.iob2"
        return cls.read_file(test_input_file)

    @classmethod
    def read_file(cls, input_iob_file: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = defaultdict(list)

        with open(str(input_iob_file), "r") as file_reader:
            document_id = None
            document_text = None

            entities = []
            entity_type = None
            entity_start = 0

            for line in file_reader:
                line = line.strip()
                if line[:3] == "###":
                    if not (document_id is None and document_text is None):
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
                        if entity_type is not None:
                            entities.append(
                                Entity((entity_start, len(document_text)), entity_type)
                            )

                        entity_start = len(document_text) + 1 if document_text else 0
                        entity_type = tag[2:]

                    elif tag == "O" and entity_type is not None:
                        entities.append(
                            Entity((entity_start, len(document_text)), entity_type)
                        )
                        entity_type = None

                    document_text = (
                        document_text + " " + token if document_text else token
                    )

                else:
                    # Edge case: last token starts a new entity
                    if entity_type is not None:
                        entities.append(
                            Entity((entity_start, len(document_text)), entity_type)
                        )

            # Last document in file
            if not (document_id is None and document_text is None):
                documents[document_id] = document_text
                entities_per_document[document_id] = entities

        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
        )


class HUNER_GENE_JNLPBA(HunerDataset):
    """
        HUNER version of the JNLPBA corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/genia"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        train_data = HunerJNLPBA.download_and_prepare_train(download_folder)
        train_data = filter_and_map_entities(train_data, {"protein": GENE_TAG})

        test_data = HunerJNLPBA.download_and_prepare_test(download_folder)
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

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        train_data = HunerJNLPBA.download_and_prepare_train(download_folder)
        train_data = filter_and_map_entities(train_data, {"cell_line": CELL_LINE_TAG})

        test_data = HunerJNLPBA.download_and_prepare_test(download_folder)
        test_data = filter_and_map_entities(test_data, {"cell_line": CELL_LINE_TAG})

        return merge_datasets([train_data, test_data])


class CELL_FINDER(ColumnCorpus):
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
        if not (train_file.exists()):
            train_corpus = self.download_and_prepare(data_folder)

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter,
            )
            writer.write_to_conll(train_corpus, train_file)
        super(CELL_FINDER, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @classmethod
    def download_and_prepare(cls, data_folder: Path) -> InternalBioNerDataset:
        data_url = "https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/cellfinder1_brat.tar.gz"
        data_path = cached_path(data_url, data_folder)
        unzip_targz_file(data_path, data_folder)

        return cls.read_folder(data_folder)

    @classmethod
    def read_folder(cls, data_folder: Path) -> InternalBioNerDataset:
        ann_files = list(data_folder.glob("*.ann"))
        documents = {}
        entities_per_document = defaultdict(list)
        for ann_file in ann_files:
            with ann_file.open() as f_ann, ann_file.with_suffix(".txt").open() as f_txt:
                document_id = ann_file.stem
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
                documents[document_id] = f_txt.read()

        return InternalBioNerDataset(
            documents=documents, entities_per_document=dict(entities_per_document)
        )


class HUNER_CELL_LINE_CELL_FINDER(HunerDataset):
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
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and test_file.exists()):
            download_folder = data_folder / "original"
            os.makedirs(str(download_folder), exist_ok=True)

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter,
            )

            train_corpus = self.download_and_prepare_train(download_folder)
            writer.write_to_conll(train_corpus, train_file)

            test_corpus = self.download_and_prepare_test(download_folder)
            writer.write_to_conll(test_corpus, test_file)

        super(MIRNA, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @classmethod
    def download_and_prepare_train(cls, data_folder: Path):
        data_url = "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/miRNA/miRNA-Train-Corpus.xml"
        data_path = cached_path(data_url, data_folder)

        return cls.parse_file(data_path)

    @classmethod
    def download_and_prepare_test(cls, data_folder: Path):
        data_url = "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/miRNA/miRNA-Test-Corpus.xml"
        data_path = cached_path(data_url, data_folder)

        return cls.parse_file(data_path)

    @classmethod
    def parse_file(cls, input_file: Path) -> InternalBioNerDataset:
        tree = etree.parse(str(input_file))

        documents = {}
        entities_per_document = {}

        for document in tree.xpath(".//document"):
            document_id = document.get("id")
            entities = []

            document_text = ""
            for sentence in document.xpath(".//sentence"):
                sentence_offset = len(document_text)
                document_text += sentence.get("text")

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

        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
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

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        train_data = MIRNA.download_and_prepare_train(download_folder)
        train_data = filter_and_map_entities(train_data, {"Genes/Proteins": GENE_TAG})

        test_data = MIRNA.download_and_prepare_test(download_folder)
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

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        train_data = MIRNA.download_and_prepare_train(download_folder)
        train_data = filter_and_map_entities(train_data, {"Species": SPECIES_TAG})

        test_data = MIRNA.download_and_prepare_test(download_folder)
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

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        train_data = MIRNA.download_and_prepare_train(download_folder)
        train_data = filter_and_map_entities(train_data, {"Diseases": DISEASE_TAG})

        test_data = MIRNA.download_and_prepare_test(download_folder)
        test_data = filter_and_map_entities(test_data, {"Diseases": DISEASE_TAG})

        return merge_datasets([train_data, test_data])


class KaewphanCorpusHelper:
    """ Helper class for the corpora from Kaewphan et al., i.e. CLL and Gellus"""

    @staticmethod
    def download_cll_dataset(data_folder: Path):
        data_url = "http://bionlp-www.utu.fi/cell-lines/CLL_corpus.tar.gz"
        data_path = cached_path(data_url, data_folder)
        unzip_targz_file(data_path, data_folder)

    @staticmethod
    def prepare_and_save_dataset(conll_folder: Path, output_file: Path):
        sentences = []
        for file in os.listdir(str(conll_folder)):
            if not file.endswith(".conll"):
                continue

            with open(os.path.join(str(conll_folder), file), "r") as reader:
                sentences.append(reader.read())

        with open(str(output_file), "w", encoding="utf8") as writer:
            writer.writelines(sentences)

    @staticmethod
    def download_gellus_dataset(data_folder: Path):
        data_url = "http://bionlp-www.utu.fi/cell-lines/Gellus_corpus.tar.gz"
        data_path = cached_path(data_url, data_folder)
        unzip_targz_file(data_path, data_folder)

    @staticmethod
    def read_dataset(
        conll_folder: Path, tag_column: int, token_column: int
    ) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}
        for file in os.listdir(str(conll_folder)):
            if not file.endswith(".conll"):
                continue

            document_id = file.replace(".conll", "")

            with open(os.path.join(str(conll_folder), file), "r") as reader:
                document_text = ""
                entities = []

                entity_start = None
                entity_type = None

                for line in reader.readlines():
                    line = line.strip()
                    if line:
                        columns = line.split("\t")
                        tag = columns[tag_column]
                        token = columns[token_column]
                        if tag.startswith("B-"):
                            if entity_type is not None:
                                entities.append(
                                    Entity(
                                        (entity_start, len(document_text)), entity_type
                                    )
                                )

                            entity_start = (
                                len(document_text) + 1 if document_text else 0
                            )
                            entity_type = tag[2:]

                        elif tag == "O" and entity_type is not None:
                            entities.append(
                                Entity((entity_start, len(document_text)), entity_type,)
                            )
                            entity_type = None

                        document_text = (
                            document_text + " " + token if document_text else token
                        )
                    else:
                        # Edge case: last token starts a new entity
                        if entity_type is not None:
                            entities.append(
                                Entity((entity_start, len(document_text)), entity_type)
                            )

                documents[document_id] = document_text
                entities_per_document[document_id] = entities

        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
        )


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
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "ner", 1: "text"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"

        if not (train_file.exists()):
            KaewphanCorpusHelper.download_cll_dataset(data_folder)

            conll_folder = data_folder / "CLL-1.0.2" / "conll"
            KaewphanCorpusHelper.prepare_and_save_dataset(conll_folder, train_file)

        super(CLL, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )


class HUNER_CELL_LINE_CLL(HunerDataset):
    """
        HUNER version of the CLL corpus containing cell line annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/cll"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        KaewphanCorpusHelper.download_cll_dataset(data_dir)
        conll_folder = data_dir / "CLL-1.0.2" / "conll"
        # FIXME: Normalize entity type name!
        return KaewphanCorpusHelper.read_dataset(
            conll_folder=conll_folder, tag_column=0, token_column=1
        )


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
            KaewphanCorpusHelper.download_gellus_dataset(data_folder)

            conll_train = data_folder / "GELLUS-1.0.3" / "conll" / "train"
            KaewphanCorpusHelper.prepare_and_save_dataset(conll_train, train_file)

            conll_dev = data_folder / "GELLUS-1.0.3" / "conll" / "devel"
            KaewphanCorpusHelper.prepare_and_save_dataset(conll_dev, dev_file)

            conll_test = data_folder / "GELLUS-1.0.3" / "conll" / "test"
            KaewphanCorpusHelper.prepare_and_save_dataset(conll_test, test_file)

        super(GELLUS, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )


class HUNER_CELL_LINE_GELLUS(HunerDataset):
    """
        HUNER version of the Gellus corpus containing cell line annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/gellus"

    def to_internal(self, data_dir: Path) -> InternalBioNerDataset:
        KaewphanCorpusHelper.download_gellus_dataset(data_dir)

        splits = []
        for folder in ["train", "devel", "test"]:
            conll_folder = data_dir / "GELLUS-1.0.3" / "conll" / folder
            split_data = KaewphanCorpusHelper.read_dataset(
                conll_folder=conll_folder, tag_column=1, token_column=0
            )
            split_data = filter_and_map_entities(
                split_data, {"Cell-line-name": CELL_LINE_TAG}
            )
            splits.append(split_data)

        return merge_datasets(splits)


class LOCTEXT(ColumnCorpus):
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

        if not (train_file.exists()):
            self.download_dataset(data_folder)
            full_dataset = self.parse_dataset(data_folder)

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            conll_writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter
            )
            conll_writer.write_to_conll(full_dataset, train_file)

        super(LOCTEXT, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = "http://pubannotation.org/downloads/LocText-annotations.tgz"
        data_path = cached_path(data_url, data_dir)
        unzip_targz_file(data_path, data_dir)

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

            with open(os.path.join(str(loctext_json_folder), file), "r") as f_in:
                data = json.load(f_in)
                document_text = data["text"].strip()
                document_text = document_text.replace("\n", " ")

                if "denotations" in data.keys():
                    for ann in data["denotations"]:
                        start = int(ann["span"]["begin"])
                        end = int(ann["span"]["end"])

                        original_entity_type = ann["obj"].split(":")[0]
                        if not original_entity_type in entity_type_mapping:
                            continue

                        entity_type = entity_type_mapping[original_entity_type]
                        entities.append(Entity((start, end), entity_type))

                documents[document_id] = document_text
                entities_per_document[document_id] = entities

        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
        )


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

    default_dir = Path(flair.cache_root) / "datasets" / "CHEMDNER"

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

        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            # download file is huge => make default_dir visible so that derivative
            # corpora can all use the same download file
            data_folder = self.default_dir
        else:
            data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and dev_file.exists() and test_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            train_data = bioc_to_internal(
                download_dir / "chemdner_corpus" / "training.bioc.xml"
            )
            dev_data = bioc_to_internal(
                download_dir / "chemdner_corpus" / "development.bioc.xml"
            )
            test_data = bioc_to_internal(
                download_dir / "chemdner_corpus" / "evaluation.bioc.xml"
            )

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            conll_writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter
            )
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(CHEMDNER, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2014/chemdner_corpus.tar.gz"
        data_path = cached_path(data_url, data_dir)
        unzip_targz_file(data_path, data_dir)


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
        train_data = bioc_to_internal(
            self.download_folder / "chemdner_corpus" / "training.bioc.xml"
        )
        dev_data = bioc_to_internal(
            self.download_folder / "chemdner_corpus" / "development.bioc.xml"
        )
        test_data = bioc_to_internal(
            self.download_folder / "chemdner_corpus" / "evaluation.bioc.xml"
        )
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

        if not (train_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            all_data = bioc_to_internal(download_dir / "iepa_bioc.xml")

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            conll_writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter
            )
            conll_writer.write_to_conll(all_data, train_file)

        super(IEPA, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = (
            "http://corpora.informatik.hu-berlin.de/corpora/brat2bioc/iepa_bioc.xml.zip"
        )
        data_path = cached_path(data_url, data_dir)
        unzip_file(data_path, data_dir)


class HUNER_GENE_IEPA(HunerDataset):
    """
        HUNER version of the IEPA corpus containing gene annotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/iepa"

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

        if not (train_file.exists()):
            dataset = self.download_and_parse_dataset(data_folder)

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            conll_writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter
            )
            conll_writer.write_to_conll(dataset, train_file)
        super(LINNEAUS, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @staticmethod
    def download_and_parse_dataset(data_dir: Path):
        data_url = "https://iweb.dl.sourceforge.net/project/linnaeus/Corpora/manual-corpus-species-1.0.tar.gz"
        data_path = cached_path(data_url, data_dir)
        unzip_targz_file(data_path, data_dir)

        documents = {}
        entities_per_document = defaultdict(list)

        # Read texts
        texts_directory = data_dir / "manual-corpus-species-1.0" / "txt"
        for filename in os.listdir(str(texts_directory)):
            document_id = filename.strip(".txt")

            with open(os.path.join(str(texts_directory), filename), "r") as file:
                documents[document_id] = file.read().strip()

        # Read annotations
        tag_file = data_dir / "manual-corpus-species-1.0" / "filtered_tags.tsv"
        with open(str(tag_file), "r") as file:
            next(file)  # Ignore header row

            for line in file:
                if not line:
                    continue

                document_id, start, end, text = line.strip().split("\t")[1:5]
                start, end = int(start), int(end)

                entities_per_document[document_id].append(
                    Entity((start, end), SPECIES_TAG)
                )

                document_text = documents[document_id]
                if document_text[start:end] != text:
                    raise AssertionError()

        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
        )


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
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            train_data = bioc_to_internal(
                download_dir
                / "CDR_Data"
                / "CDR.Corpus.v010516"
                / "CDR_TrainingSet.BioC.xml"
            )
            dev_data = bioc_to_internal(
                download_dir
                / "CDR_Data"
                / "CDR.Corpus.v010516"
                / "CDR_DevelopmentSet.BioC.xml"
            )
            test_data = bioc_to_internal(
                download_dir
                / "CDR_Data"
                / "CDR.Corpus.v010516"
                / "CDR_TestSet.BioC.xml"
            )

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            conll_writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter
            )
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)
            conll_writer.write_to_conll(test_data, test_file)

        super(CDR, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = (
            "https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip"
        )
        data_path = cached_path(data_url, data_dir)
        unzip_file(data_path, data_dir)


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
        train_data = bioc_to_internal(
            data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml"
        )
        dev_data = bioc_to_internal(
            data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_DevelopmentSet.BioC.xml"
        )
        test_data = bioc_to_internal(
            data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.BioC.xml"
        )
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
        train_data = bioc_to_internal(
            data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml"
        )
        dev_data = bioc_to_internal(
            data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_DevelopmentSet.BioC.xml"
        )
        test_data = bioc_to_internal(
            data_dir / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.BioC.xml"
        )
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

        if not (train_file.exists()):
            download_dir = data_folder / "original"
            os.makedirs(download_dir, exist_ok=True)
            self.download_dataset(download_dir)

            all_data = bioc_to_internal(download_dir / "hvp_bioc.xml")

            if tokenizer is None:
                tokenizer = build_spacy_tokenizer()

            if sentence_splitter is None:
                sentence_splitter = build_spacy_sentence_splitter()

            conll_writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter
            )
            conll_writer.write_to_conll(all_data, train_file)

        super(VARIOME, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @staticmethod
    def download_dataset(data_dir: Path):
        data_url = (
            "http://corpora.informatik.hu-berlin.de/corpora/brat2bioc/hvp_bioc.xml.zip"
        )
        data_path = cached_path(data_url, data_dir)
        unzip_file(data_path, data_dir)


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
        all_data = bioc_to_internal(data_dir / "hvp_bioc.xml")
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
        all_data = bioc_to_internal(data_dir / "hvp_bioc.xml")
        all_data = filter_and_map_entities(
            all_data, {"Disorder": DISEASE_TAG, "disease": DISEASE_TAG}
        )

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
        all_data = bioc_to_internal(data_dir / "hvp_bioc.xml")
        all_data = filter_and_map_entities(all_data, {"Living_Beings": SPECIES_TAG})

        return all_data
