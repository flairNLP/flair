from typing import List
import csv
import torch
import time
import logging
import numpy as np

import flair
from flair.data import Span
from flair.embeddings import Embeddings, TokenEmbeddings
from flair.nn import DefaultClassifier
from typing import Dict

log = logging.getLogger("flair")

# class LogitEmbeddings(TokenEmbeddings):
#
#     def __init__(self,
#                  classifier: DefaultClassifier,
#                  ):
#         self.classifier_model = classifier




class SpanGazetteerEmbeddings(Embeddings[Span]):

    def __init__(self,
                 gazetteer_file: str = None,
                 add_lower_case_lookup: bool = False,
                 add_substring_gazetteer_lookup: bool = False,
                 add_first_last_token_gazetteer_lookup: bool = False,
                 counts_for_max_confidence=100,  # TODO: what to choose here? make parameter
                 gazetteer_prepare_method: str = "normalize_confidence_ratio",
                 ):
        """
        :param gazetteer_file: path to a csv file containing a gazetteer list with span strings in rows, label names in columns, counts in cells
        :param add_lower_case_lookup: concatenating a look-up for title cased version of span text, ege WASHINGTON --> Washington
        :param add_substring_gazetteer_lookup: concatenating mean vector of look-up for each token in span
        :param add_first_last_token_gazetteer_lookup: concatenatinv look-up for first and last token in span
        :param counts_for_max_confidence: #TODO
        :param gazetteer_prepare_method: #TODO
        """
        super().__init__()
        self.name = gazetteer_file
        self.static_embeddings = True
        self.add_lower_case_lookup = add_lower_case_lookup
        self.add_first_last_token_gazetteer_lookup = add_first_last_token_gazetteer_lookup
        self.add_substring_gazetteer_lookup = add_substring_gazetteer_lookup
        self.gazetteer_prepare_method = gazetteer_prepare_method

        self.gazetteer = self._prepare_gazetteer(gazetteer_file,
                                                 counts_for_max_confidence,
                                                 prepare_method=self.gazetteer_prepare_method)

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

    def _prepare_gazetteer(self,
                           gazetteer_file,
                           counts_for_max_confidence,
                           prepare_method = "normalize_confidence_ratio"):

        log.info(f"---- Reading raw gazetteer file: {gazetteer_file}")
        with open(gazetteer_file, mode='r') as inp:
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

            if prepare_method == "normalize_confidence_ratio":
                if time.time() - start >= 30:  # print progress every 30 seconds
                    print("done with \t", round(nr / len(gazetteer) * 100, 2), " % of gazetteer", end="\n")
                    start = time.time()

                # Filter tagged counts and get tagged sum
                vector_tag_counts = vector[:4]
                sum_tagged = np.sum(vector_tag_counts)  # sum
                ratio_tagged_untagged = vector[6]

                # compute normalized tag counts and confidence
                normalized_tag_counts = vector_tag_counts / sum_tagged
                confidence = np.array([min(sum_tagged / counts_for_max_confidence, 1)])

                rt_vector = np.concatenate((normalized_tag_counts, confidence, [ratio_tagged_untagged]), 0)
                #rt_vector = np.round(rt_vector, 4)

                gazetteer[key] = np.around(rt_vector, decimals=5).tolist()

            if prepare_method == "normalize":
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

    def get_gazetteer_embedding(self, span_string: str) -> torch.Tensor:

        if span_string in self.gazetteer:
            gaz_vector = torch.tensor(self.gazetteer[span_string], device=flair.device, dtype=torch.float)
        else:
            #gaz_vector = torch.neg(torch.ones(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float))
            gaz_vector = torch.zeros(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float)

        return gaz_vector

    def has_entry(self, span_string: str) -> bool:
        return span_string in self.gazetteer

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
                                                         for t in span.tokens]),dim=0)
                embeddings.append(substring_mean)

            span.set_embedding(self.name, torch.cat(embeddings))

    def __str__(self):
        return self.name


class SpanGazetteerFeaturePrediction(Embeddings[Span]):

    def __init__(self,
                 prediction_model = None,
                 fine_tune: bool = False,
                 ):
        """
        :param prediction_model: the trained model to be used for predicting gazetteer features per span,
        the prediction_model needs to have a predict method that returns a feature vector per string
        """
        super().__init__()
        self.prediction_model = prediction_model
        self.name = "SpanGazetteerFeaturePrediction" # TODO how to get a nice descriptive name from the model
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
