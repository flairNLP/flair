from typing import List
import csv
import torch
import torch.optim as optim

import time
import logging
import numpy as np
import os
import random

import flair
from flair.data import Span, DT, Token
from flair.embeddings import Embeddings, FlairEmbeddings
from typing import Dict, Generic, Union

log = logging.getLogger("flair")


class SpanEmbeddingFromExternal(Embeddings[Span], Generic[DT]):
    def __init__(self,
                 global_lowercasing: bool = True,
                 add_lower_case_lookup: bool = False,
                 add_substring_gazetteer_lookup: bool = False,
                 add_first_last_token_gazetteer_lookup: bool = False,
                 transform_gazetteer_to_latent: bool = False,
                 latent_gazetteer_embedding_dimension: int = 10,
                 #train_gazetteer_model_meanwhile: Embeddings[Token] = None,
                 gazetteer_calculation_method: str = "normalize_using_counts",
                 add_confidence = True, # TODO change back?
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
        self.transform_gazetteer_to_latent = transform_gazetteer_to_latent
        self.latent_gazetteer_embedding_dimension = latent_gazetteer_embedding_dimension
        self.gazetteer_calculation_method = gazetteer_calculation_method
        self.add_confidence = add_confidence


        #self.train_gazetteer_model_meanwhile = train_gazetteer_model_meanwhile

        self.gazetteer = self._prepare_gazetteer()

        if self.global_lowercasing:
            print(f"--- Converting keys to lower case ---")
            self.gazetteer =  {k.lower(): v for k, v in self.gazetteer.items()}

        self.__gazetteer_vector_length = len(next(iter(self.gazetteer.values())))  # one entry in gaz to get its size
        #if self.add_confidence:
        #    self.__gazetteer_vector_length +=1
        self.__embedding_length = self.__gazetteer_vector_length
        if self.add_confidence:
            self.__embedding_length +=1
        if self.add_lower_case_lookup:
            self.__embedding_length += self.__gazetteer_vector_length
            if self.add_confidence:
                self.__embedding_length += 1
        if self.add_first_last_token_gazetteer_lookup:
            self.__embedding_length += (self.__gazetteer_vector_length * 2)
            if self.add_confidence:
                self.__embedding_length += 2
        if self.add_substring_gazetteer_lookup:
            self.__embedding_length += self.__gazetteer_vector_length
            if self.add_confidence:
                self.__embedding_length += 1

        if self.transform_gazetteer_to_latent:
            self.gazetteer_to_latent_layer = torch.nn.Linear(self.__embedding_length,
                                                             self.latent_gazetteer_embedding_dimension)

            self.__embedding_length = self.latent_gazetteer_embedding_dimension # resetting to latent dim

        #if self.train_gazetteer_model_meanwhile:
        #    self.__embedding_length = self.train_gazetteer_model_meanwhile.embedding_length

        self.__embedding_type = "span-level"

        self.to(flair.device)

    def _prepare_gazetteer(self):
        raise NotImplementedError

    def get_gazetteer_embedding(self, span_string: str) -> torch.Tensor:

        if self.global_lowercasing:
            span_string = span_string.lower()

        if self.gazetteer_calculation_method == "normalize_using_counts":
            count_vector = self.get_count_vector(span_string)
            if torch.sum(count_vector) > 0:
                gaz_vector = count_vector / torch.sum(count_vector)
            else:
                gaz_vector = count_vector

            if self.add_confidence:
                confidence = torch.minimum(torch.sum(count_vector) / 50, torch.tensor([1.0], device=flair.device, dtype=torch.float))
                gaz_vector = torch.cat([gaz_vector, confidence], dim = 0)

        elif self.gazetteer_calculation_method == "direct":
            if span_string in self.gazetteer:
                gaz_vector = self.gazetteer[span_string]
            else:
                gaz_vector = torch.zeros(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float)

        else:
            raise NotImplementedError

        return gaz_vector

    def get_count_vector(self, span_string: str) -> torch.Tensor:
        if self.global_lowercasing:
            span_string = span_string.lower()

        if span_string in self.count_dict:
            count_vector = self.count_dict[span_string]

        else:
            count_vector = torch.zeros(self.__gazetteer_vector_length,  device=flair.device, dtype=torch.float)

        return count_vector


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

            embeddings = torch.cat(embeddings)

            if self.transform_gazetteer_to_latent:
                embeddings = self.gazetteer_to_latent_layer(embeddings)

            span.set_embedding(self.name, embeddings)

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
        self.label2idx: Dict[str, int] = {}

        super().__init__(**kwargs,
                         )

        # use gazetteer also as count_dict (proxy: one-hot interpreted as "seen once")
        self.count_dict = {}
        self.count_dict.update(self.gazetteer)


    def _prepare_gazetteer(self):
        #gazetteer_files = []
        files = list([f for f in os.listdir(self.gazetteer_files_directory + '/') if not os.path.isdir(os.path.join(self.gazetteer_files_directory,f))])
        if self.exclude_files:
            files = [f for f in files if str(f) not in self.exclude_files]
        gazetteer_files = files
        count_files = len(gazetteer_files)
        gazetteer = {}
        for (label_id, gazetteer_file) in enumerate(gazetteer_files):
            print(f"--- Now processing file nr {label_id}: {gazetteer_file} ---")
            label_name_proxy = gazetteer_file # using the file name as name for label type
            self.label2idx.update({label_name_proxy: label_id})
            with open(f"{self.gazetteer_files_directory}/{gazetteer_file}", 'r', encoding='utf-8', errors='strict') as src:
                idx = self.label2idx[label_name_proxy]
                for line in src:
                    if len(line) == 0:
                        continue
                    elif len(line.rstrip("\n")) > 0:
                        line = line.rstrip("\n")

                    if line not in gazetteer:
                        #gazetteer[line] = torch.zeros(count_files) # [0.0] * count_files
                        gazetteer[line] = torch.zeros(count_files, device=flair.device, dtype=torch.float)

                    gazetteer[line][idx] = 1.0

        print(f"--- Nr of entries in gazetteer: {len(gazetteer)} ---")
        print(f"--- Mapping label2idx: {self.label2idx} ---")

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
                if len(row) ==1: # some ~10 rows have comma instead of tab, they get ignored here
                    try:
                        row = row[0].split(",")
                    except:
                        continue
                if len(row) < 4:
                    continue
                span = " ".join(row[3:]) # hack: in some rows the span got wrongly separated
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
                 gazetteer_calculation_method: str = "normalize_confidence_ratio",
                 **kwargs,
                 ):
        """
        :param gazetteer_file: path to a csv file containing a gazetteer list with span strings in rows, label names in columns, counts in cells
        :param counts_for_max_confidence: #TODO
        :param gazetteer_calculation_method: #TODO
        """
        self.name = gazetteer_file
        self.gazetteer_file = gazetteer_file
        self.static_embeddings = True
        self.gazetteer_calculation_method = gazetteer_calculation_method
        self.counts_for_max_confidence = counts_for_max_confidence
        self.add_confidence = False
        #if self.gazetteer_calculation_method == "normalize_using_counts":
        #    self.add_confidence = True # TODO add_confidence change
        self.label2idx: Dict[str, int] = {}

        super().__init__(**kwargs,
                         )


    def _prepare_gazetteer(self):

        log.info(f"---- Reading raw gazetteer file: {self.gazetteer_file}")
        with open(self.gazetteer_file, mode='r') as inp:
            log.info(f"---- Gazetteer file contains {sum(1 for line in inp)} lines...")
            inp.seek(0)  # to start at beginning again
            reader = csv.reader(inp)
            header = next(reader)  # header line
            self.label2idx = { label: i for (i, label) in enumerate(header[1:])}

            log.info(f"---- Header is: {header}")

            gazetteer = {row[0]: list(map(float, row[1:])) for row in reader}  # read rest in dict
            #gazetteer = {row[0]: torch.tensor(map(float, row[1:]), device=flair.device, dtype=torch.float) for row in reader}  # read rest in dict

        print(f"---- Length of used gazetteer:\t", len(gazetteer))

        global_start = time.time()
        start = time.time()

        if self.gazetteer_calculation_method == "normalize_confidence_ratio":
            self.label2idx.pop("abs_span_freq")
            self.label2idx.pop("tagged_frequency")
            self.label2idx.pop("tagged_ratio")
            self.label2idx.update({"confidence": 4,
                                   "ratio": 5})

        if self.gazetteer_calculation_method == "normalize":
            self.label2idx.pop("abs_span_freq")
            self.label2idx.pop("tagged_frequency")
            self.label2idx.pop("tagged_ratio")

        if self.gazetteer_calculation_method == "normalize_using_counts":
            self.label2idx.pop("untagged")
            # here I'm using a different csv file! not applicable for other gazetteers at the moment!
            # store the absolute counts and calculate the normalized vector on the fly
            self.count_dict: Dict[str, torch.Tensor] = {}

        self.__gazetteer_vector_length = len(self.label2idx)

        for nr, (key, vector) in enumerate(gazetteer.items()):

            if self.gazetteer_calculation_method == "normalize_confidence_ratio":
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
                #rt_vector = np.round(rt_vector, 4)

                #gazetteer[key] = np.around(rt_vector, decimals=5).tolist()
                gazetteer[key] = torch.tensor(np.around(rt_vector, decimals=5), device=flair.device, dtype=torch.float)

            if self.gazetteer_calculation_method == "normalize":
                # Filter tagged counts and get tagged sum
                vector_tag_counts = vector[:4]
                sum_tagged = np.sum(vector_tag_counts)  # sum
                ratio_tagged_untagged = vector[6]

                # compute normalized tag counts and confidence
                normalized_tag_counts = vector_tag_counts / sum_tagged

                #gazetteer[key] = np.around(np.array(normalized_tag_counts), decimals=5).tolist()
                gazetteer[key] = torch.tensor(np.around(np.array(normalized_tag_counts), decimals=5), device=flair.device, dtype=torch.float)

            if self.gazetteer_calculation_method == "normalize_using_counts":
                vector_tag_counts = vector[:4]
                self.count_dict[key] = torch.tensor(np.around(np.array(vector_tag_counts), decimals=5),
                                                    device=flair.device, dtype=torch.float)

                gazetteer[key] = self.count_dict[key] # todo: this is not really needed and confusing, but to make other methods happy (embedding_length, mapping of labels, ...)

        global_end = time.time()
        print(f"---- Converting took {round(global_end - global_start, 2)} seconds")

        return gazetteer



# TODO: refactor so that it inherits from SpanEmbeddingFromExternal class as well as the others
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


class ExpandingGazetteerSpanEmbeddings(Embeddings[Span], Generic[DT]):
    def __init__(self,
                 label2idx: Dict[str, int],
                 reset_after_each_epoch: bool = True,
                 starting_with_gazetteer: Embeddings[Span] = None,
                 confidence_threshold: float = 0.8,
                 update_gazetteer_embeddings_method: str = "replace",
                 skip_first_epoch: bool = False, # TODO not implemented
                 pooling: str = "min",
                 global_lowercasing: bool = False,
                 mapping_corpus_label_to_initial_gazetteer: Dict[str, str] = None,
                 transform_gazetteer_to_latent: bool = False, # learn an intermediate linear layer transforming original gaz vector (e.g. one-hot) to latent vector
                 latent_gazetteer_embedding_dimension: int = 10,
                 train_gazetteer_model_meanwhile: Union[bool, Embeddings[Token]] = False,
                 use_gazetteer_model_predictions_mode: str = "as_backup",
                 gazetteer_calculation_method: str = "normalize_using_counts",
                 **kwargs,
                 ):
        """
        :param
        """

        if not hasattr(self, "name"):
            self.name: str = "unnamed_embedding"
        if not hasattr(self, "static_embeddings"):
            self.static_embeddings = False

        super().__init__()

        self.static_embeddings = False

        # these fields are for the embedding memory
        self.label2idx = label2idx
        if "O" in self.label2idx:
            self.label2idx.pop("O")

        self.gazetteer: Dict[str, torch.Tensor] = {}
        self.count_dict: Dict[str, torch.Tensor] = {}
        self.updated_partition: Dict[str, torch.Tensor] = {}
        self.skip_first_epoch = skip_first_epoch
        self.global_lowercasing = global_lowercasing
        self.updated_partition: Dict[str, torch.Tensor] = {}
        self.update_gazetteer_embeddings_method= update_gazetteer_embeddings_method


        self.transform_gazetteer_to_latent = transform_gazetteer_to_latent
        self.latent_gazetteer_embedding_dimension = latent_gazetteer_embedding_dimension

        if train_gazetteer_model_meanwhile:
            if type(train_gazetteer_model_meanwhile) == Embeddings[Token]:
                self.train_gazetteer_model_meanwhile = train_gazetteer_model_meanwhile
            else:
                self.train_gazetteer_model_meanwhile = flair.embeddings.FlairEmbeddings("news-forward-fast")

            self.use_gazetteer_model_predictions_mode = use_gazetteer_model_predictions_mode
            if self.use_gazetteer_model_predictions_mode not in ["never", "always", "as_backup"]:
                raise AssertionError("use_gazetteer_model_predictions_mode must be one of 'never', 'always' or 'as_backup'")
        else:
            self.train_gazetteer_model_meanwhile = False

        # set the memory method
        self.pooling = pooling

        self.reset_after_each_epoch = reset_after_each_epoch
        self.starting_with_gazetteer = starting_with_gazetteer
        self.confidence_threshold = confidence_threshold
        self.mapping_corpus_label_to_initial_gazetteer = mapping_corpus_label_to_initial_gazetteer
        self.initial_gazetteer = {}
        self.gazetteer_calculation_method = gazetteer_calculation_method
        self.add_confidence = False

        if self.starting_with_gazetteer:
            self.gazetteer_calculation_method = self.starting_with_gazetteer.gazetteer_calculation_method
            if hasattr(self.starting_with_gazetteer, "add_confidence"):
                self.add_confidence = self.starting_with_gazetteer.add_confidence

            if self.global_lowercasing != self.starting_with_gazetteer.global_lowercasing:
                raise AssertionError("Attention: Lowercasing (global_lowercasing) is inconsistent between initial gazetteer and dynamic gazetteer!")

            self.initial_gazetteer.update(self.starting_with_gazetteer.gazetteer)
            log.info(f"Starting with initial gazetteer with length {len(self.initial_gazetteer)}")
            log.info(f"With initial label2idx mapping: {self.starting_with_gazetteer.label2idx}")
            label2idx_combined = {}

            # adding the original from initial gazetteer
            idx_counter = 0
            for k in self.starting_with_gazetteer.label2idx.keys():
                label2idx_combined[k] = idx_counter
                idx_counter +=1

            # adding the ones from corpus, in their matched/mapped form
            for k in self.label2idx.keys():
                if self.mapping_corpus_label_to_initial_gazetteer and k in self.mapping_corpus_label_to_initial_gazetteer:
                    corpus_equivalent = self.mapping_corpus_label_to_initial_gazetteer[k]
                else:
                    corpus_equivalent = k
                if corpus_equivalent not in label2idx_combined:
                    label2idx_combined[corpus_equivalent] = idx_counter
                    idx_counter +=1

            log.info("Restructuring the initial gazetteer to match new mapping (adding Columns and Zeros)...")
            idx2label_initial = { v:k for k,v in self.starting_with_gazetteer.label2idx.items() }

            initial_gazetteer_tmp = {}
            for k,v in self.initial_gazetteer.items():
                original_vector = v
                new_vector = torch.zeros(len(label2idx_combined), device=flair.device, dtype=torch.float)
                for old_idx, entry in enumerate(original_vector):
                    label_name = idx2label_initial[old_idx]
                    new_idx = label2idx_combined[label_name]
                    new_vector[new_idx] = entry
                initial_gazetteer_tmp[k] = new_vector

            self.initial_gazetteer = initial_gazetteer_tmp
            del initial_gazetteer_tmp

            # same for count_dict:
            if hasattr(self.starting_with_gazetteer, "count_dict"):
                count_dict_tmp = {}
                for k, v in self.starting_with_gazetteer.count_dict.items():
                    original_vector = v
                    new_vector = torch.zeros(len(label2idx_combined), device=flair.device, dtype=torch.float)
                    for old_idx, entry in enumerate(original_vector):
                        label_name = idx2label_initial[old_idx]
                        new_idx = label2idx_combined[label_name]
                        new_vector[new_idx] = entry
                    count_dict_tmp[k] = new_vector

                self.count_dict = count_dict_tmp
                del count_dict_tmp

            self.starting_with_gazetteer.label2idx = label2idx_combined
            self.gazetteer.update(self.initial_gazetteer)

            self.label2idx = self.starting_with_gazetteer.label2idx
            log.info("After matching and adding the labels from corpus:")
            log.info(f"{self.label2idx}")

            # take the gazetteer vectors also as count vectors in case there is no count_dict from starting_gazetteer (assuming these are one-hot then...)
            #if not self.starting_with_gazetteer or not hasattr(self.starting_with_gazetteer, "count_dict"):
            #    from copy import copy
            #    count_dict = copy(self.gazetteer)
            #    self.count_dict = count_dict
            #    del count_dict

        # take the gazetteer vectors also as count vectors in case there is no count_dict from starting_gazetteer (assuming these are one-hot then...)
        #if not self.count_dict:
        # Achtung, auch das Count Dict muss mapping durchlaufen!
        #from copy import copy
        #count_dict = copy(self.gazetteer)
        #self.count_dict = count_dict
        #del count_dict

        self.__gazetteer_vector_length = len(self.label2idx)
        self.__embedding_length = self.__gazetteer_vector_length
        if self.add_confidence:
            self.__embedding_length += 1

        #if self.starting_with_gazetteer:
        #    self.__embedding_length = starting_with_gazetteer.embedding_length # necessary because of latent dim # TODO why was that necessary...?

        if self.transform_gazetteer_to_latent:
            self.gazetteer_to_latent_layer = torch.nn.Linear(self.__embedding_length,
                                                             self.latent_gazetteer_embedding_dimension)
            self.__embedding_length = self.latent_gazetteer_embedding_dimension

        if self.train_gazetteer_model_meanwhile:

            self.gaz_model = GazetteerModel(model = self.train_gazetteer_model_meanwhile,
                                            out_dim = self.__embedding_length)
            self.gaz_model.to(flair.device)

            self.gaz_model.pretrain = True  # set FlairEmbeddings into fine tune mode
            self.gaz_model.update = True

            if self.gaz_model.pretrain:
                pass

                # pretraining the self.gaz_model with the current/initial gazetteer
                # problem: where to get corpus_for_negative_sampling from?

                #self._fine_tune_gaz_model_on_gazetteer(gazetteer= self.gazetteer,
                #                                       corpus_for_negative_sampling= None, # where to get this from?
                #                                       )

        self.__embedding_type = "span-level"

        self.to(flair.device)

    def _fine_tune_gaz_model_on_gazetteer(self, gazetteer, corpus_for_negative_sampling,
                                          ratio_negative_to_positive =1,
                                          downsample = 1.0,
                                          max_span_length = 5,
                                          epochs=5, batch_size=32,lr=0.01, loss =torch.nn.MSELoss()):
        if not self.gaz_model:
            pass

        if self.gaz_model.update:

            train_data = {}
            if downsample != 1.0:
                gazetteer = dict(random.sample(gazetteer.items(), int(len(gazetteer)*downsample)))
            train_data.update(gazetteer)

            # now sample some (same size as positive) negative spans with zero vectors
            size_positive = len(train_data)
            negative_spans = {}
            counter = 0
            while counter <= size_positive*ratio_negative_to_positive:
                sentence = random.choice(corpus_for_negative_sampling)
                rand_span_size = random.randint(1, max_span_length)
                if rand_span_size > len(sentence):
                    continue
                rand_start = random.randint(0, len(sentence) - rand_span_size)
                random_span_text = sentence[rand_start:rand_start + rand_span_size].text
                if random_span_text in train_data:
                    continue
                negative_spans[random_span_text] = torch.zeros(self.embedding_length,
                                                               device=flair.device, dtype=torch.float)
                counter += 1

            train_data.update(negative_spans.items())  # add negatives to train data

            self.gaz_model.model.fine_tune = False  # TODO do I want to train the FlairEmbeddings as well or just the linear map ?
            self.gaz_model.train()

            optimizer = optim.AdamW(self.gaz_model.parameters(), lr=lr)

            def chunks(lst, n):
                """Split list into n-sized chunks."""
                rt = []
                for i in range(0, len(lst), n):
                    rt.append(lst[i:i + n])
                return rt

            log.info(f"---- Train Gaz Model ----")

            for epoch in range(epochs):
                log.info(f"---- Train Gaz Model, Epoch {epoch+1} ----")

                strings = []
                targets = []
                gazetteer_entries = list(gazetteer.items())
                random.shuffle(gazetteer_entries)

                for key, value in gazetteer_entries:
                    strings.append(key)
                    targets.append(value)

                strings_batches = chunks(strings, batch_size)
                targets_batches = chunks(targets, batch_size)

                for nr, batch in enumerate(zip(strings_batches, targets_batches)):
                    print(f"------- Train Gaz Model, Epoch {epoch+1}/{epochs}, Batch {nr+1}/{len(strings_batches)}")
                    self.gaz_model.train()

                    strings = batch[0]
                    targets = batch[1]

                    output = self.gaz_model(strings)
                    batch_loss = loss(output, torch.stack(targets))

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

            self.gaz_model.model.fine_tune = False  # set FlairEmbeddings into fine tune mode
            self.gaz_model.eval()

            # TODO hier bin ich gerade pass

    def _add_current_gazetteer_to_initial(self):
        log.info(f"--- Adding the current gazetteer state to initial gazetteer:")
        len_before = len(self.initial_gazetteer)
        self.initial_gazetteer.update(self.gazetteer)
        len_after = len(self.initial_gazetteer)
        log.info(f"--- Initial gazetteer size \t {len_before} --> \t {len_after}")

    def _add_updated_gazetteer_to_initial(self):
        log.info(f"--- Adding the current updated gazetteer to initial gazetteer:")
        len_before = len(self.initial_gazetteer)
        self.initial_gazetteer.update(self.updated_partition)
        len_after = len(self.initial_gazetteer)
        log.info(f"--- Initial gazetteer size \t {len_before} --> \t {len_after}")

    def _add_initial_gazetteer_to_current(self):
        log.info(f"--- Adding the initial gazetteer to current gazetteer:")
        len_before = len(self.gazetteer)
        self.gazetteer.update(self.initial_gazetteer)
        len_after = len(self.gazetteer)
        log.info(f"--- Gazetteer size \t {len_before} --> \t {len_after}")

    def set_update_mode_latent_layer(self, mode):
        if not self.gazetteer_to_latent_layer:
            print("--- Trying to reset latent layer but not existent! ---")
            return
        else:
            if mode == False:
                for p in self.gazetteer_to_latent_layer.parameters():
                    p.requires_grad = False
            if mode == True:
                for p in self.gazetteer_to_latent_layer.parameters():
                    p.requires_grad = True

    def _freeze_latent_layer_weights(self):
        for p in self.gazetteer_to_latent_layer.parameters():
            p.requires_grad = False


    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            if self.reset_after_each_epoch:
                # memory is wiped each time we do a training run
                len_before = len(self.gazetteer)
                self.gazetteer = {}
                self.updated_partition = {}

                if self.starting_with_gazetteer:
                    log.info("train mode resetting embeddings to just include initial gazetteer")
                    #self.gazetteer.update(self.initial_gazetteer)
                    self._add_initial_gazetteer_to_current()
                else:
                    log.info("train mode resetting embeddings")
                log.info(f"--> resetting from {len_before} to {len(self.gazetteer)} entries")
            else:
                log.info("train mode, keeping embeddings")
                log.info(f"--> now including {len(self.gazetteer)} entries")

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    def get_gazetteer_embedding(self, span_string: str,
                                ) -> torch.Tensor:

        if self.global_lowercasing:
            span_string = span_string.lower()

        if self.gazetteer_calculation_method == "normalize_using_counts":
            count_vector = self.get_count_vector(span_string)
            if torch.sum(count_vector) > 0:
                gaz_vector = count_vector / torch.sum(count_vector)
            else:
                gaz_vector = count_vector

            if self.add_confidence:
                confidence = torch.minimum(torch.sum(count_vector) / 50, torch.tensor([1.0], device=flair.device, dtype=torch.float))
                gaz_vector = torch.cat([gaz_vector, confidence], dim = 0)


        elif self.gazetteer_calculation_method == "direct":
            if span_string in self.gazetteer:
                gaz_vector = self.gazetteer[span_string]
            else:
                gaz_vector = torch.zeros(self.__embedding_length, device=flair.device, dtype=torch.float)

        else:
            raise NotImplementedError

        if self.train_gazetteer_model_meanwhile and self.use_gazetteer_model_predictions_mode == "always":
            with torch.no_grad():
                predicted_embedding = self.gaz_model(span_string)
                gaz_vector = predicted_embedding.squeeze(0)

        if self.train_gazetteer_model_meanwhile and self.use_gazetteer_model_predictions_mode == "as_backup":
            raise NotImplementedError

        if not isinstance(gaz_vector, torch.Tensor):
            raise ValueError("Couldn't calculate a gaz_vector...")
        else:
            return gaz_vector

            # else:
            #    if self.train_gazetteer_model_meanwhile and self.use_gazetteer_model_predictions_mode == "as_backup":
            #        with torch.no_grad():
            #            predicted_embedding = self.gaz_model(span_string)
            #            gaz_vector = predicted_embedding.squeeze(0)
            #    else:
            #        gaz_vector = torch.zeros(self.__gazetteer_vector_length,  device=flair.device, dtype=torch.float)



    def get_count_vector(self, span_string: str) -> torch.Tensor:
        if self.global_lowercasing:
            span_string = span_string.lower()

        if span_string in self.count_dict:
            count_vector = self.count_dict[span_string]

        else:
            count_vector = torch.zeros(self.__gazetteer_vector_length,  device=flair.device, dtype=torch.float)

        return count_vector


    def update_gazetteer_embeddings(self, span_string, label_from_corpus):
        if self.mapping_corpus_label_to_initial_gazetteer and label_from_corpus in self.mapping_corpus_label_to_initial_gazetteer:
            label_from_corpus = self.mapping_corpus_label_to_initial_gazetteer[label_from_corpus]

        if label_from_corpus not in self.label2idx:
            raise ValueError("label seems to not be in the label2idx dict. Something wrong with mapping? A typo?")

        if self.global_lowercasing:
            span_string = span_string.lower()

        current_embedding = self.get_gazetteer_embedding(span_string)

        if self.update_gazetteer_embeddings_method == "replace":
            new_embedding = current_embedding.detach().clone()
            #new_embedding = torch.zeros(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float)
            new_embedding[self.label2idx[label_from_corpus]] = 1.0

        elif self.update_gazetteer_embeddings_method == "normalize_using_counts":
            # 1. get the ABSOLUTE COUNTS so far
            current_counts = self.get_count_vector(span_string)
            # 2. add the one seen from prediction/gold label
            new_counts = current_counts.detach().clone()
            new_counts[self.label2idx[label_from_corpus]] += 1
            # 3. update the absolute counts
            self.count_dict[span_string] = new_counts
            # 4. normalize newly
            #new_embedding = new_counts / torch.sum(new_counts)
            new_embedding = self.get_gazetteer_embedding(span_string)

        else:
            raise NotImplementedError

        if False in torch.eq(current_embedding, new_embedding):
            #print(f"Updating embedding entry for \t{span_string} from \t {current_embedding} --> \t {new_embedding}")
            if self.update_gazetteer_embeddings_method == "normalize_using_counts":
                # with this method the important thins is the count dict!
                self.gazetteer[span_string] = new_counts
                self.updated_partition[span_string] = new_counts
            else:
                self.gazetteer[span_string] = new_embedding
                self.updated_partition[span_string] = new_embedding

    def _add_embeddings_internal(self, spans: List[Span]):

        for span in spans:
            embeddings = [self.get_gazetteer_embedding(span.text)]
            embeddings = torch.cat(embeddings)

            if self.transform_gazetteer_to_latent:
                embeddings = self.gazetteer_to_latent_layer(embeddings)

            span.set_embedding(self.name, embeddings)

    def _get_updated_partition_of_gazetteer(self):
        return self.updated_partition

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


class GazetteerModel(torch.nn.Module):
    def __init__(self, model, out_dim):
        super().__init__()
        self.model = model
        self.linear = torch.nn.Linear(self.model.embedding_length, out_dim)

    def forward(self, strings):
        if not isinstance(strings, list):
            strings = [strings]
        strings_as_flair_sentences = flair.data.Sentence(strings)
        self.model.embed(strings_as_flair_sentences)
        embedding = [s.embedding for s in strings_as_flair_sentences]  # each span seen as one token
        embedding = torch.stack(embedding)
        out = self.linear(embedding)
        return out
