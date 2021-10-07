import logging

from pathlib import Path
from typing import List, Union, Optional, Sequence, Dict, Tuple, Any

import conllu

from flair.data import Sentence, Corpus, Token, FlairDataset, Span, RelationLabel, SpanLabel
from flair.datasets.base import find_train_dev_test_files

log = logging.getLogger("flair")

DEFAULT_FIELDS: Tuple[str, ...] = ("id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc")

DEFAULT_TOKEN_ANNOTATION_FIELDS: Tuple[str, ...] = ("lemma", "upos", "xpos", "feats", "head", "deprel")

# noinspection PyProtectedMember
DEFAULT_METADATA_PARSERS: Dict[str, conllu._MetadataParserType] = {
    **conllu.parser.DEFAULT_METADATA_PARSERS,
    **{"relations": lambda key, value: parse_relation_tuple_list(key, value, list_sep="|", value_sep=";")}
}


def parse_relation_tuple_list(key: str,
                              value: Optional[str] = None,
                              list_sep: str = "|",
                              value_sep: str = ";") -> Optional[Tuple[str, List[Tuple[int, int, int, int, str]]]]:
    if value is None:
        return value

    relation_tuples: List[Tuple[int, int, int, int, str]] = []
    for relation in value.split(list_sep):
        head_start, head_end, tail_start, tail_end, label = relation.split(value_sep)
        relation_tuples.append((int(head_start), int(head_end), int(tail_start), int(tail_end), label))

    return key, relation_tuples


class CoNLLUCorpus(Corpus):

    # noinspection PyProtectedMember
    def __init__(self,
                 data_folder: Union[str, Path],
                 train_file=None,
                 test_file=None,
                 dev_file=None,
                 in_memory: bool = True,
                 fields: Optional[Sequence[str]] = None,
                 token_annotation_fields: Optional[Sequence[str]] = None,
                 field_parsers: Optional[Dict[str, conllu._FieldParserType]] = None,
                 metadata_parsers: Optional[Dict[str, conllu._MetadataParserType]] = None,
                 sample_missing_splits: bool = True):
        """
        Instantiates a Corpus from CoNLL-U (Plus) column-formatted task data

        Universal dependencies corpora that contain multi-word tokens are not supported yet.
        The annotation of flair sentences with the "deps" column is not yet supported as well.
        Please consider using the "UniversalDependenciesCorpus" instead.

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        :param token_annotation_fields: A subset of the fields parameter for token level annotations
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        # get train data
        train = CoNLLUDataset(
            train_file,
            in_memory=in_memory,
            fields=fields,
            token_annotation_fields=token_annotation_fields,
            field_parsers=field_parsers,
            metadata_parsers=metadata_parsers,
        )

        # get test data
        test = (
            CoNLLUDataset(
                test_file,
                in_memory=in_memory,
                fields=fields,
                token_annotation_fields=token_annotation_fields,
                field_parsers=field_parsers,
                metadata_parsers=metadata_parsers,
            )
            if test_file is not None
            else None
        )

        # get dev data
        dev = (
            CoNLLUDataset(
                dev_file,
                in_memory=in_memory,
                fields=fields,
                token_annotation_fields=token_annotation_fields,
                field_parsers=field_parsers,
                metadata_parsers=metadata_parsers,
            )
            if dev_file is not None
            else None
        )

        super(CoNLLUCorpus, self).__init__(train, dev, test, name=str(data_folder),
                                           sample_missing_splits=sample_missing_splits)


class CoNLLUDataset(FlairDataset):

    # noinspection PyProtectedMember
    def __init__(self,
                 path_to_conllu_file: Union[str, Path],
                 in_memory: bool = True,
                 fields: Optional[Sequence[str]] = None,
                 token_annotation_fields: Optional[Sequence[str]] = None,
                 field_parsers: Optional[Dict[str, conllu._FieldParserType]] = None,
                 metadata_parsers: Optional[Dict[str, conllu._MetadataParserType]] = None):
        """
        Instantiates a column dataset in CoNLL-U (Plus) format.

        Universal dependencies datasets that contain multi-word tokens are not supported yet.
        The annotation of flair sentences with the "deps" column is not yet supported as well.
        Please consider using the "UniversalDependenciesDataset" instead.

        :param path_to_conllu_file: Path to the CoNLL-U formatted file
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        :param token_annotation_fields: A subset of the fields parameter for token level annotations
        """
        if type(path_to_conllu_file) is str:
            path_to_conllu_file = Path(path_to_conllu_file)
        assert path_to_conllu_file.exists()

        self.path_to_conllu_file = path_to_conllu_file
        self.in_memory = in_memory

        # if no fields specified, check if the file is CoNLL plus formatted and get fields
        if fields is None:
            with open(str(self.path_to_conllu_file), encoding="utf-8") as file:
                fields = conllu.parser.parse_conllu_plus_fields(file)

        self.fields = fields or DEFAULT_FIELDS
        self.token_annotation_fields = token_annotation_fields or DEFAULT_TOKEN_ANNOTATION_FIELDS

        # Validate fields and token_annotation_fields
        if not set(self.token_annotation_fields).issubset(self.fields):
            raise ValueError(f"The token annotation fields {repr(self.token_annotation_fields)} "
                             f"are not a subset of the parsed fields {repr(self.fields)}.")

        # noinspection PyProtectedMember
        augmented_default_field_parsers: Dict[str, conllu._FieldParserType] = {
            **{
                field: lambda line_, i: conllu.parser.parse_nullable_value(line_[i])
                for field in self.token_annotation_fields
            },
            **conllu.parser.DEFAULT_FIELD_PARSERS
        }

        self.field_parsers = field_parsers or augmented_default_field_parsers
        self.metadata_parsers = metadata_parsers or DEFAULT_METADATA_PARSERS

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

                self.indices.append(position)
                self.total_sentence_count = len(self.indices)

            # option 2: keep everything in memory
            if self.in_memory:
                self.sentences: List[Sentence] = [
                    self.token_list_to_sentence(token_list)
                    for token_list in conllu.parse_incr(
                        file,
                        fields=self.fields,
                        field_parsers=self.field_parsers,
                        metadata_parsers=self.metadata_parsers,
                    )
                ]

                # pointer to previous
                previous_sentence = None

                for sentence in self.sentences:

                    sentence._previous_sentence = previous_sentence
                    sentence._next_sentence = None
                    if previous_sentence: previous_sentence._next_sentence = sentence
                    previous_sentence = sentence

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
            with open(str(self.path_to_conllu_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                token_list = next(conllu.parse_incr(file, self.fields, self.field_parsers, self.metadata_parsers))
                sentence = self.token_list_to_sentence(token_list)

        return sentence

    def token_list_to_sentence(self, token_list: conllu.TokenList) -> Sentence:
        sentence: Sentence = Sentence()

        # Build the sentence tokens and add the annotations.
        for conllu_token in token_list:
            token = Token(conllu_token["form"])

            for field in self.token_annotation_fields:
                field_value: Any = conllu_token[field]
                if isinstance(field_value, dict):
                    # For fields that contain key-value annotations,
                    # we add the key as label type-name and the value as the label value.
                    for key, value in field_value.items():
                        token.add_label(typename=key, value=str(value))
                else:
                    token.add_label(typename=field, value=str(field_value))

            if conllu_token.get("misc") is not None:
                space_after: Optional[str] = conllu_token["misc"].get("SpaceAfter")
                if space_after == "No":
                    token.whitespace_after = False

            sentence.add_token(token)

        if "sentence_id" in token_list.metadata:
            sentence.add_label("sentence_id", token_list.metadata["sentence_id"])

        if "relations" in token_list.metadata:
            for head_start, head_end, tail_start, tail_end, label in token_list.metadata["relations"]:
                # head and tail span indices are 1-indexed and end index is inclusive
                head = Span(sentence.tokens[head_start - 1: head_end])
                tail = Span(sentence.tokens[tail_start - 1: tail_end])

                sentence.add_complex_label("relation", RelationLabel(value=label, head=head, tail=tail))

        # determine all NER label types in sentence and add all NER spans as sentence-level labels
        ner_label_types = []
        for token in sentence.tokens:
            for annotation in token.annotation_layers.keys():
                if annotation.startswith("ner") and annotation not in ner_label_types:
                    ner_label_types.append(annotation)

        for label_type in ner_label_types:
            spans = sentence.get_spans(label_type)
            for span in spans:
                sentence.add_complex_label("entity", label=SpanLabel(span=span, value=span.tag, score=span.score))

        return sentence
