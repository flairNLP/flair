from typing import List
import csv
import torch
import time
import logging
import numpy as np
import os

import flair
from flair.data import Span, DT
from flair.embeddings import Embeddings
from typing import Dict, Generic

log = logging.getLogger("flair")


class SpanEmbeddingFromExternal(Embeddings[Span], Generic[DT]):
    def __init__(self,
                 global_lowercasing: bool = True,
                 add_lower_case_lookup: bool = False,
                 add_substring_gazetteer_lookup: bool = False,
                 add_first_last_token_gazetteer_lookup: bool = False,
                 transform_gazetteer_to_latent: bool = False,
                 latent_gazetteer_embedding_dimension: int = 10,
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

        self.gazetteer = self._prepare_gazetteer()

        if self.global_lowercasing:
            print(f"--- Converting keys to lower case ---")
            self.gazetteer =  {k.lower(): v for k, v in self.gazetteer.items()}

        self.__gazetteer_vector_length = len(next(iter(self.gazetteer.values())))  # one entry in gaz to get its size
        self.__embedding_length = self.__gazetteer_vector_length
        if self.add_lower_case_lookup:
            self.__embedding_length += self.__gazetteer_vector_length
        if self.add_first_last_token_gazetteer_lookup:
            self.__embedding_length += (self.__gazetteer_vector_length * 2)
        if self.add_substring_gazetteer_lookup:
            self.__embedding_length += self.__gazetteer_vector_length

        if self.transform_gazetteer_to_latent:
            self.gazetteer_to_latent_layer = torch.nn.Linear(self.__embedding_length,
                                                             self.latent_gazetteer_embedding_dimension)

            self.__embedding_length = self.latent_gazetteer_embedding_dimension # resetting to latent dim

        self.__embedding_type = "span-level"

        self.to(flair.device)

    def _prepare_gazetteer(self):
        raise NotImplementedError

    def get_gazetteer_embedding(self, span_string: str) -> torch.Tensor:

        if self.global_lowercasing:
            span_string = span_string.lower()

        if span_string in self.gazetteer:
            #gaz_vector = torch.tensor(self.gazetteer[span_string], device=flair.device, dtype=torch.float)
            gaz_vector = self.gazetteer[span_string]
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

        if self.gazetteer_prepare_method == "normalize_confidence_ratio":
            self.label2idx.pop("abs_span_freq")
            self.label2idx.pop("tagged_frequency")
            self.label2idx.pop("tagged_ratio")
            self.label2idx.update({"confidence": 4,
                                   "ratio": 5})

        if self.gazetteer_prepare_method == "normalize":
            self.label2idx.pop("abs_span_freq")
            self.label2idx.pop("tagged_frequency")
            self.label2idx.pop("tagged_ratio")

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
                #rt_vector = np.round(rt_vector, 4)

                #gazetteer[key] = np.around(rt_vector, decimals=5).tolist()
                gazetteer[key] = torch.tensor(np.around(rt_vector, decimals=5), device=flair.device, dtype=torch.float)

            if self.gazetteer_prepare_method == "normalize":
                # Filter tagged counts and get tagged sum
                vector_tag_counts = vector[:4]
                sum_tagged = np.sum(vector_tag_counts)  # sum
                ratio_tagged_untagged = vector[6]

                # compute normalized tag counts and confidence
                normalized_tag_counts = vector_tag_counts / sum_tagged

                #gazetteer[key] = np.around(np.array(normalized_tag_counts), decimals=5).tolist()
                gazetteer[key] = torch.tensor(np.around(np.array(normalized_tag_counts), decimals=5), device=flair.device, dtype=torch.float)

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
                 starting_with_gazetteer = None, #TODO implement this
                 confidence_threshold: float = 0.8,
                 skip_first_epoch: bool = False, # TODO maybe makes sense to leave first epoch out
                 pooling: str = "min",
                 global_lowercasing: bool = False,
                 mapping_corpus_label_to_initial_gazetteer: Dict[str, str] = None,
                 transform_gazetteer_to_latent: bool = False, # learn an intermediate linear layer transforming original gaz vector (e.g. one-hot) to latent vector
                 latent_gazetteer_embedding_dimension: int = 10,

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
        self.updated_partition: Dict[str, torch.Tensor] = {}
        self.skip_first_epoch = skip_first_epoch
        self.global_lowercasing = global_lowercasing
        self.updated_partition: Dict[str, torch.Tensor] = {}

        self.transform_gazetteer_to_latent = transform_gazetteer_to_latent
        self.latent_gazetteer_embedding_dimension = latent_gazetteer_embedding_dimension

        # set the memory method
        self.pooling = pooling

        self.reset_after_each_epoch = reset_after_each_epoch
        self.starting_with_gazetteer = starting_with_gazetteer
        self.confidence_threshold = confidence_threshold

        if self.starting_with_gazetteer:
            if self.global_lowercasing != self.starting_with_gazetteer.global_lowercasing:
                raise AssertionError("Attention: Lowercasing (global_lowercasing) is inconsistent between initial gazetteer and dynamic gazetteer!")

            self.initial_gazetteer = self.starting_with_gazetteer.gazetteer
            self.mapping_corpus_label_to_initial_gazetteer = mapping_corpus_label_to_initial_gazetteer
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
                new_vector = torch.zeros(len(label2idx_combined),  device=flair.device, dtype=torch.float)
                for old_idx, entry in enumerate(original_vector):
                    label_name = idx2label_initial[old_idx]
                    new_idx = label2idx_combined[label_name]
                    new_vector[new_idx] = entry
                initial_gazetteer_tmp[k] = new_vector

            self.initial_gazetteer = initial_gazetteer_tmp
            del initial_gazetteer_tmp
            self.starting_with_gazetteer.label2idx = label2idx_combined
            self.gazetteer.update(self.initial_gazetteer)

            self.label2idx = self.starting_with_gazetteer.label2idx
            log.info("After matching and adding the labels from corpus:")
            log.info(f"{self.label2idx}")


        self.__gazetteer_vector_length = len(self.label2idx)
        self.__embedding_length = self.__gazetteer_vector_length
        if self.starting_with_gazetteer:
            self.__embedding_length = starting_with_gazetteer.embedding_length # necessary because of latent dim

        if self.transform_gazetteer_to_latent:
            self.gazetteer_to_latent_layer = torch.nn.Linear(self.__embedding_length,
                                                             self.latent_gazetteer_embedding_dimension)
            self.__embedding_length = self.latent_gazetteer_embedding_dimension

        self.__embedding_type = "span-level"

        self.to(flair.device)

    def _add_current_gazetteer_to_initial(self):
        log.info(f"--- Adding the current gazetteer state to initial gazetteer:")
        len_before = len(self.initial_gazetteer)
        self.initial_gazetteer.update(self.gazetteer)
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

    def get_gazetteer_embedding(self, span_string: str) -> torch.Tensor:
        if self.global_lowercasing:
            span_string = span_string.lower()

        if span_string in self.gazetteer:
            #gaz_vector = torch.tensor(self.gazetteer[span_string], device=flair.device, dtype=torch.float)
            gaz_vector = self.gazetteer[span_string]

        else:
            gaz_vector = torch.zeros(self.__gazetteer_vector_length,  device=flair.device, dtype=torch.float)

        return gaz_vector

    def update_gazetteer_embeddings(self, span_string, predicted_label, method = "replace"):
        if self.mapping_corpus_label_to_initial_gazetteer and predicted_label in self.mapping_corpus_label_to_initial_gazetteer:
            predicted_label = self.mapping_corpus_label_to_initial_gazetteer[predicted_label]

        if predicted_label not in self.label2idx:
            raise ValueError("label seems to not be in the label2idx dict. Something wrong with mapping? A typo?")

        if self.global_lowercasing:
            span_string = span_string.lower()

        current_embedding = self.get_gazetteer_embedding(span_string)
        if method == "replace":
            new_embedding = current_embedding.detach().clone()
            #new_embedding = torch.zeros(self.__gazetteer_vector_length, device=flair.device, dtype=torch.float)
            new_embedding[self.label2idx[predicted_label]] = 1.0
            # TODO Beware: blindly setting 1.0 to the predicted label!
            #  Think about using logits or at least think about what to do with not 1-hot gazetteer!
        else:
            raise NotImplementedError

        if False in torch.eq(current_embedding, new_embedding):
            #print(f"Updating embedding entry for \t{span_string} from \t {current_embedding} --> \t {new_embedding}")
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
