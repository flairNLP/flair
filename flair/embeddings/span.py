from typing import List
import csv
import torch
import time
import logging
import numpy as np
import os

from torch import Tensor

import flair
from flair.data import Span, DT
from flair.embeddings import Embeddings
from typing import Dict, Generic

log = logging.getLogger("flair")


class SpanGazetteer(Embeddings[Span]):
    def __init__(self,
                 gazetteer: Dict[str, List[int]],
                 add_lower_case_lookup: bool = False,
                 ):
        """
        :param add_lower_case_lookup: concatenating a look-up for title cased version of span text, ege WASHINGTON --> Washington
        """

        self.name: str = "dictionary_gazetteer"
        self.gazetteer: Dict[str, List[int]] = gazetteer
        self.static_embeddings = True

        super().__init__()

        self.add_lower_case_lookup = add_lower_case_lookup

        self.__gazetteer_vector_length = len(next(iter(self.gazetteer.values())))  # one entry in gaz to get its size
        self.__embedding_type = "span-level"

        # length of gazetteer embedding
        self.__embedding_length = self.__gazetteer_vector_length
        if self.add_lower_case_lookup:
            self.__embedding_length += self.__gazetteer_vector_length

        self.to(flair.device)

    def get_gazetteer_embedding(self, span_string: str) -> torch.Tensor:

        if span_string in self.gazetteer:
            gaz_vector = torch.tensor(self.gazetteer[span_string], device=flair.device, dtype=torch.float)
        else:
            gaz_vector = torch.zeros(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float)

        return gaz_vector

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    def _add_embeddings_internal(self, spans: List[Span]):
        for span in spans:
            embeddings = [self.get_gazetteer_embedding(span.text)]
            if self.add_lower_case_lookup:
                embeddings.append(self.get_gazetteer_embedding(span.text.title()))
            span.set_embedding(self.name, torch.cat(embeddings))

    def __str__(self):
        return self.name


class SpanEmbeddingFromExternal(Embeddings[Span], Generic[DT]):
    def __init__(self,
                 global_lowercasing: bool = True,
                 add_lower_case_lookup: bool = False,
                 add_substring_gazetteer_lookup: bool = False,
                 add_first_last_token_gazetteer_lookup: bool = False,
                 **kwargs,
                 ):
        """
        :param global_lowercasing: make lookup case insensitive
        :param add_lower_case_lookup: concatenating a look-up for title cased version of span text, ege WASHINGTON --> Washington
        :param add_substring_gazetteer_lookup: concatenating mean vector of look-up for each token in span
        :param add_first_last_token_gazetteer_lookup: concatenatinv look-up for first and last token in span
        """

        if not hasattr(self, "name"):
            self.name: str = "unnamed_embedding"
        if not hasattr(self, "static_embeddings"):
            self.static_embeddings = False

        super().__init__()

        self.static_embeddings = True

        self.global_lowercasing = global_lowercasing
        self.add_lower_case_lookup = add_lower_case_lookup
        self.add_first_last_token_gazetteer_lookup = add_first_last_token_gazetteer_lookup
        self.add_substring_gazetteer_lookup = add_substring_gazetteer_lookup

        self.gazetteer = self._prepare_gazetteer()

        if self.global_lowercasing:
            print(f"--- Converting keys to lower case ---")
            self.gazetteer = {k.lower(): v for k, v in self.gazetteer.items()}

        self.__gazetteer_vector_length = len(next(iter(self.gazetteer.values())))  # one entry in gaz to get its size
        self.__embedding_length = self.__gazetteer_vector_length
        self.__embedding_type = "span-level"
        if self.add_lower_case_lookup:
            self.__embedding_length += self.__gazetteer_vector_length
        if self.add_first_last_token_gazetteer_lookup:
            self.__embedding_length += (self.__gazetteer_vector_length * 2)
        if self.add_substring_gazetteer_lookup:
            self.__embedding_length += self.__gazetteer_vector_length

        self.to(flair.device)

    def _prepare_gazetteer(self):
        raise NotImplementedError

    def get_gazetteer_embedding(self, span_string: str) -> torch.Tensor:

        if self.global_lowercasing:
            span_string = span_string.lower()

        if span_string in self.gazetteer:
            gaz_vector = torch.tensor(self.gazetteer[span_string], device=flair.device, dtype=torch.float)
        else:
            # gaz_vector = torch.neg(torch.ones(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float))
            gaz_vector = torch.zeros(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float)

        return gaz_vector

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    def _add_embeddings_internal(self, spans: List[Span]):
        for span in spans:
            embeddings = [self.get_gazetteer_embedding(span.text)]
            if self.add_lower_case_lookup:
                embeddings.append(self.get_gazetteer_embedding(span.text.title()))
            if self.add_first_last_token_gazetteer_lookup:
                embeddings.append(self.get_gazetteer_embedding(span[0].text))
                embeddings.append(self.get_gazetteer_embedding(span[-1].text))
            if self.add_substring_gazetteer_lookup:
                substring_mean = torch.mean(torch.stack([self.get_gazetteer_embedding(t.text)
                                                         for t in span.tokens]), dim=0)
                embeddings.append(substring_mean)

            span.set_embedding(self.name, torch.cat(embeddings))

    def __str__(self):
        return self.name


class SpanGazetteerEmbeddingsFromFiles(SpanEmbeddingFromExternal[Span]):

    def __init__(self,
                 gazetteer_files_directory: str = None,
                 exclude_files: list = None,
                 **kwargs,
                 ):
        """
        :param gazetteer_files_directory: path to folder containing gazetteer files
        :param exclude_files: filenames to exclude during preparation of gazetteer
        """

        self.name = gazetteer_files_directory
        self.gazetteer_files_directory = gazetteer_files_directory
        self.exclude_files = exclude_files

        super().__init__(**kwargs,
                         )

    def _prepare_gazetteer(self):
        gazetteer_files = []
        files = list([f for f in os.listdir(self.gazetteer_files_directory + '/') if
                      not os.path.isdir(os.path.join(self.gazetteer_files_directory, f))])
        if self.exclude_files:
            files = [f for f in files if str(f) not in self.exclude_files]
        for index, file in enumerate(files):
            gazetteer_files.append((str(index), file))
        count_files = len(gazetteer_files)
        gazetteer = {}
        for (label_id, gazetteer_file) in gazetteer_files:
            print(f"--- Now processing file nr {label_id}: {gazetteer_file} ---")
            with open(f"{self.gazetteer_files_directory}/{gazetteer_file}", 'r', encoding='utf-8',
                      errors='strict') as src:
                for line in src:
                    if len(line) == 0:
                        continue
                    elif len(line.rstrip("\n")) > 0:
                        line = line.rstrip("\n")

                    if line not in gazetteer:
                        gazetteer[line] = [0.0] * count_files

                    gazetteer[line][int(label_id)] = 1.0

        print(f"--- Nr of entries in gazetteer: {len(gazetteer)} ---")

        return gazetteer


class SpanEmbeddingGEMNET(SpanEmbeddingFromExternal[Span]):

    def __init__(self,
                 path: str = None,
                 **kwargs,
                 ):
        """
        :param path: the local path to the 'wikigaz-tsv' file from Meng et al. 2020, can be downloaded here:
        https://lowcontext-ner-gaz.s3.amazonaws.com/readme.html
        """

        self.path = path

        super().__init__(**kwargs,
                         )

    def _prepare_gazetteer(self):
        log.info(f"---- Reading raw gazetteer file: {self.path}")

        gazetteer = {}
        with open(self.path, mode='r') as inp:
            log.info(f"---- Gazetteer file contains {sum(1 for line in inp)} lines...")
            inp.seek(0)  # to start at beginning again
            label2id = {"CORP": 0,
                        "CW": 1,
                        "GRP": 2,
                        "LOC": 3,
                        "PER": 4,
                        "PROD": 5
                        }
            reader = csv.reader(inp, delimiter='\t')

            for row in reader:
                if len(row) == 1:  # some ~10 rows have comma instead of tab, they get ignored here
                    try:
                        row = row[0].split(",")
                    except:
                        continue
                if len(row) < 4:
                    continue
                span = " ".join(row[3:])  # hack: in some rows the span got wrongly separated
                label = row[1]
                label_id = label2id[label]
                if span not in gazetteer:
                    gazetteer[span] = [0.0] * len(label2id)
                gazetteer[span][label_id] = 1.0

        print(f"--- Nr of entries in gazetteer: {len(gazetteer)} ---")

        return gazetteer


class SpanGazetteerEmbeddings(SpanEmbeddingFromExternal[Span]):

    def __init__(self,
                 gazetteer_file: str = None,
                 counts_for_max_confidence=100,  # TODO: what to choose here? make parameter
                 gazetteer_prepare_method: str = "normalize_confidence_ratio",
                 **kwargs,
                 ):
        """
        :param gazetteer_file: path to a csv file containing a gazetteer list with span strings in rows, label names in columns, counts in cells
        :param counts_for_max_confidence: #TODO
        :param gazetteer_prepare_method: #TODO
        """
        self.name = gazetteer_file
        self.gazetteer_file = gazetteer_file
        self.static_embeddings = True
        self.gazetteer_prepare_method = gazetteer_prepare_method
        self.counts_for_max_confidence = counts_for_max_confidence

        super().__init__(**kwargs,
                         )

    def _prepare_gazetteer(self):

        log.info(f"---- Reading raw gazetteer file: {self.gazetteer_file}")
        with open(self.gazetteer_file, mode='r') as inp:
            log.info(f"---- Gazetteer file contains {sum(1 for line in inp)} lines...")
            inp.seek(0)  # to start at beginning again
            reader = csv.reader(inp)
            header = next(reader)  # header line
            log.info(f"---- Header is: {header}")

            gazetteer = {row[0]: list(map(float, row[1:])) for row in reader}  # read rest in dict

        print(f"---- Length of used gazetteer:\t", len(gazetteer))

        global_start = time.time()
        start = time.time()

        for nr, (key, vector) in enumerate(gazetteer.items()):

            if self.gazetteer_prepare_method == "normalize_confidence_ratio":
                if time.time() - start >= 30:  # print progress every 30 seconds
                    print("done with \t", round(nr / len(gazetteer) * 100, 2), " % of gazetteer", end="\n")
                    start = time.time()

                # Filter tagged counts and get tagged sum
                vector_tag_counts = vector[:4]
                sum_tagged = np.sum(vector_tag_counts)  # sum
                ratio_tagged_untagged = vector[6]

                # compute normalized tag counts and confidence
                normalized_tag_counts = vector_tag_counts / sum_tagged
                confidence = np.array([min(sum_tagged / self.counts_for_max_confidence, 1)])

                rt_vector = np.concatenate((normalized_tag_counts, confidence, [ratio_tagged_untagged]), 0)
                # rt_vector = np.round(rt_vector, 4)

                gazetteer[key] = np.around(rt_vector, decimals=5).tolist()

            if self.gazetteer_prepare_method == "normalize":
                # Filter tagged counts and get tagged sum
                vector_tag_counts = vector[:4]
                sum_tagged = np.sum(vector_tag_counts)  # sum
                ratio_tagged_untagged = vector[6]

                # compute normalized tag counts and confidence
                normalized_tag_counts = vector_tag_counts / sum_tagged

                gazetteer[key] = np.around(np.array(normalized_tag_counts), decimals=5).tolist()

        global_end = time.time()
        print(f"---- Converting took {round(global_end - global_start, 2)} seconds")

        return gazetteer


# TODO: refactor so that it inherits from SpanEmbeddingFromExternal class as well as the others
class SpanGazetteerFeaturePrediction(Embeddings[Span]):

    def __init__(self,
                 prediction_model=None,
                 fine_tune: bool = False,
                 ):
        """
        :param prediction_model: the trained model to be used for predicting gazetteer features per span,
        the prediction_model needs to have a predict method that returns a feature vector per string
        """
        super().__init__()
        self.prediction_model = prediction_model
        self.name = "SpanGazetteerFeaturePrediction"  # TODO how to get a nice descriptive name from the model
        self.fine_tune = fine_tune
        if self.fine_tune:
            self.static_embeddings = False
        else:
            self.static_embeddings = True

        self.__gazetteer_vector_length = self.prediction_model.predict(["some string"]).size(1)
        self.__embedding_length = self.__gazetteer_vector_length
        self.__embedding_type = "span-level"

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    def predict_feature_vector(self, spans: List[Span]):
        if self.fine_tune:
            self.prediction_model.train()
            return self.prediction_model.predict([s.text for s in spans])
        else:
            self.prediction_model.eval()
            with torch.no_grad():
                return self.prediction_model.predict([s.text for s in spans])

    def _add_embeddings_internal(self, spans: List[Span]):
        embeddings = self.predict_feature_vector(spans)
        for i, span in enumerate(spans):
            span.set_embedding(self.name, embeddings[i])

    def __str__(self):
        return self.name


class StackedSpanEmbeddings(Embeddings[Span]):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[Embeddings[Span]]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            embedding.name = f"{str(i)}-{embedding.name}"
            self.add_module(f"list_embedding_{str(i)}", embedding)

        self.name: str = "Stack"
        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(self, spans: List[Span], static_embeddings: bool = True):
        if type(spans) is Span:
            spans = [spans]

        for embedding in self.embeddings:
            embedding.embed(spans)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, spans: List[Span]) -> List[Span]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(spans)

        return spans

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

    def get_names(self) -> List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        names = []
        for embedding in self.embeddings:
            names.extend(embedding.get_names())
        return names

    def get_named_embeddings_dict(self) -> Dict:

        named_embeddings_dict = {}
        for embedding in self.embeddings:
            named_embeddings_dict.update(embedding.get_named_embeddings_dict())

        return named_embeddings_dict
