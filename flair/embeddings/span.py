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
                 counts_for_max_confidence=100,  # TODO: what to choose here? make parameter
                 ):
        """
        :param gazetteer_file: path to a csv file containing a gazetteer list with span strings in rows, label names in columns, counts in cells
        :param gazetteer_untagged_label_name: give column name in gazetteer for "O" (untagged) counts (defaults to None, eg. not existing)
        :param gazetteer_include_untagged: set to True if you want to include untagged in gazetteer feature value (if existing)
        """
        super().__init__()
        self.name = gazetteer_file
        self.static_embeddings = True
        self.add_lower_case_lookup = add_lower_case_lookup
        self.add_substring_gazetteer_lookup = add_substring_gazetteer_lookup

        self.gazetteer = self._prepare_gazetteer(gazetteer_file,
                                                 counts_for_max_confidence)

        self.__gazetteer_vector_length = len(next(iter(self.gazetteer.values())))  # one entry in gaz to get its size
        self.__embedding_length = self.__gazetteer_vector_length
        if self.add_lower_case_lookup:
            self.__embedding_length += self.__gazetteer_vector_length
        if self.add_substring_gazetteer_lookup:
            self.__embedding_length += (self.__gazetteer_vector_length * 2)

        self.to(flair.device)

    def _prepare_gazetteer(self,
                           gazetteer_file,
                           counts_for_max_confidence):

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

            if time.time() - start >= 30:  # print progress every 30 seconds
                print("done with \t", round(nr / len(gazetteer) * 100, 2), " % of gazetteer", end="\n")
                start = time.time()

            # Filter untagged and get tagged sum
            vector_without_untagged = vector[:4]
            sum_tagged = np.sum(vector_without_untagged)  # sum without counting the "O" counts

            # compute normalized tag counts and confidence
            normalized_tag_counts = vector_without_untagged / sum_tagged
            confidence = np.array([min(sum_tagged / counts_for_max_confidence, 1)])

            rt_vector = np.concatenate((normalized_tag_counts, confidence, [vector[6]]), 0)
            rt_vector = np.round(rt_vector, 4)

            gazetteer[key] = np.around(rt_vector, decimals=5).tolist()

        global_end = time.time()
        print(f"---- Converting took {round(global_end - global_start, 2)} seconds")

        return gazetteer

    def get_gazetteer_embedding(self, span_string: str) -> torch.Tensor:

        if span_string in self.gazetteer:
            gaz_vector = torch.tensor(self.gazetteer[span_string], device=flair.device, dtype=torch.float)
        else:
            gaz_vector = torch.neg(torch.ones(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float))

        return gaz_vector

    def has_entry(self, span_string: str) -> bool:
        return span_string in self.gazetteer

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @property
    def embedding_type(self) -> int:
        return self.__embedding_type

    def _add_embeddings_internal(self, spans: List[Span]):
        for span in spans:
            embeddings = [self.get_gazetteer_embedding(span.text)]
            if self.add_lower_case_lookup:
                embeddings.append(self.get_gazetteer_embedding(span.text.title()))
            if self.add_substring_gazetteer_lookup:
                embeddings.append(self.get_gazetteer_embedding(span[0].text))
                embeddings.append(self.get_gazetteer_embedding(span[-1].text))

            span.set_embedding(self.name, torch.cat(embeddings))

    def __str__(self):
        return self.name



class SpanGazetteerFeaturePrediction(Embeddings[Span]):

    def __init__(self,
                 prediction_model = None, # TODO needs to be a model that has a predict-method
                 ):
        """
        :param prediction_model: the trained model to be used for predicting gazetteer features per span
        """
        super().__init__()
        self.prediction_model = prediction_model
        self.name = "predicted-gazetteer-features" # TODO get a nice descriptive name from the model
        self.static_embeddings = True

        self.__gazetteer_vector_length = 4 # TODO get length somehow
        self.__embedding_length = self.__gazetteer_vector_length

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @property
    def embedding_type(self) -> int:
        return self.__embedding_type

    def predict_feature_vector(self, spans: List[Span]):
        feature_vector = self.prediction_model.predict([s.text for s in spans])
        #feature_vector = feature_vector.squeeze().detach()  # TODO necessary?

        return feature_vector

    def _add_embeddings_internal(self, spans: List[Span]):
        embeddings = self.predict_feature_vector(spans)
        for i, span in enumerate(spans):
            span.set_embedding(self.name, embeddings[i])

    def __str__(self):
        return self.name

    # TODO das hier ist eigentlich Quatsch, wird nur gerade von der error_analysis gebraucht, daher hack
    # die error_analysis sollte also flexibler werden
    def has_entry(self, span_string: str) -> bool:
        return True

