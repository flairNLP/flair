import os
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Union, Callable, Dict, List, Tuple, Iterable

from lxml import etree

import flair
from flair.datasets import ColumnCorpus
from flair.file_utils import cached_path, unzip_file, unzip_targz_file


class Entity:
    def __init__(self, char_span: Tuple[int, int], entity_type: str):
        self.char_span = range(*char_span)
        self.type = entity_type


class InternalBioNerDataset:
    def __init__(
        self, documents: Dict[str, str], entities_per_document: Dict[str, List[Entity]]
    ):
        self.documents = documents
        self.entities_per_document = entities_per_document


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


def merge_datasets(data_sets: Iterable[InternalBioNerDataset]):
    all_documents = {}
    all_entities = {}

    for ds in data_sets:
        all_documents.update(ds.documents)
        all_entities.update(ds.entities_per_document)

    return InternalBioNerDataset(
        documents=all_documents, entities_per_document=all_entities
    )


def filter_entities(
    dataset: InternalBioNerDataset, target_types: Iterable[str]
) -> InternalBioNerDataset:
    """
    FIXME Map to canonical type names
    """
    target_entities_per_document = {
        id: [e for e in entities if e.type in target_types]
        for id, entities in dataset.entities_per_document.items()
    }

    return InternalBioNerDataset(
        documents=dataset.documents, entities_per_document=target_entities_per_document
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

                        # FIXME The runtime complexity of this is unneccessarily high
                        # FIXME This assumes that entities aren't nested, we have to ensure that beforehand
                        for entity in entities:
                            if offset in entity.char_span:
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
    def to_internal(data_folder: Path) -> InternalBioNerDataset:
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
            internal_dataset = self.to_internal(data_folder)

            splits_dir = data_folder / "splits"
            os.makedirs(splits_dir, exist_ok=True)

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


class HUNER_PROTEIN_BIO_INFER(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bioinfer"

    @staticmethod
    def to_internal(data_dir: Path) -> InternalBioNerDataset:
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
        test_file = data_folder / "test.conll"

        if not (train_file.exists() and test_file.exists()):
            writer = CoNLLWriter(
                tokenizer=tokenizer, sentence_splitter=sentence_splitter,
            )

            download_folder = data_folder / "original"
            os.makedirs(str(download_folder), exist_ok=True)

            train_corpus = JNLPBA.download_and_prepare_train(download_folder)
            writer.write_to_conll(train_corpus, train_file)

            test_corpus = JNLPBA.download_and_prepare_test(download_folder)
            writer.write_to_conll(test_corpus, test_file)

        super(JNLPBA, self).__init__(
            data_folder, columns, tag_to_bioes="ner", in_memory=in_memory
        )

    @staticmethod
    def download_and_prepare_train(data_folder: Path) -> InternalBioNerDataset:
        train_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz"
        train_data_path = cached_path(train_data_url, data_folder)
        unzip_targz_file(train_data_path, data_folder)

        train_input_file = data_folder / "Genia4ERtask2.iob2"
        return JNLPBA.read_file(train_input_file)

    @staticmethod
    def download_and_prepare_test(data_folder: Path) -> InternalBioNerDataset:
        train_data_url = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz"
        train_data_path = cached_path(train_data_url, data_folder)
        unzip_targz_file(train_data_path, data_folder)

        test_input_file = data_folder / "Genia4EReval2.iob2"
        return JNLPBA.read_file(test_input_file)

    @staticmethod
    def read_file(input_iob_file: Path) -> InternalBioNerDataset:
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

                    entity = parts[1].strip()
                    if entity.startswith("B-"):
                        if entity_type is not None:
                            entities.append(
                                Entity((entity_start, len(document_text)), entity_type)
                            )

                        entity_start = len(document_text) + 1 if document_text else 0
                        entity_type = entity[2:]

                    elif entity == "O" and entity_type is not None:
                        entities.append(
                            Entity((entity_start, len(document_text)), entity_type)
                        )
                        entity_type = None

                    token = parts[0].strip()
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


class HUNER_PROTEIN_JNLPBA(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/genia"

    @staticmethod
    def to_internal(data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        train_data = JNLPBA.download_and_prepare_train(download_folder)
        train_data = filter_entities(train_data, "protein")

        test_data = JNLPBA.download_and_prepare_test(download_folder)
        test_data = filter_entities(test_data, "protein")

        return merge_datasets([train_data, test_data])


class HUNER_CELL_LINE_JNLPBA(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/genia"

    @staticmethod
    def to_internal(data_dir: Path) -> InternalBioNerDataset:
        download_folder = data_dir / "original"
        os.makedirs(str(download_folder), exist_ok=True)

        train_data = JNLPBA.download_and_prepare_train(download_folder)
        train_data = filter_entities(train_data, "cell_line")

        test_data = JNLPBA.download_and_prepare_test(download_folder)
        test_data = filter_entities(test_data, "cell_line")

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
        if not (train_file.exists()):
            train_corpus = self.download_and_prepare(data_folder)
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

    @staticmethod
    def to_internal(data_dir: Path) -> InternalBioNerDataset:
        data = CELL_FINDER.download_and_prepare(data_dir)
        data = filter_entities(data, "CellLine")

        return data


class HUNER_SPECIES_CELL_FINDER(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/cellfinder_species"

    @staticmethod
    def to_internal(data_dir: Path) -> InternalBioNerDataset:
        data = CELL_FINDER.download_and_prepare(data_dir)
        data = filter_entities(data, "Species")

        return data


class HUNER_PROTEIN_CELL_FINDER(HunerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_url() -> str:
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/cellfinder_protein"

    @staticmethod
    def to_internal(data_dir: Path) -> InternalBioNerDataset:
        data = CELL_FINDER.download_and_prepare(data_dir)
        data = filter_entities(data, "GeneProtein")

        return data
