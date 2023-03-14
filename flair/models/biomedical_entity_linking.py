import os
import stat
import re
import pickle
import logging
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
)
from huggingface_hub import hf_hub_url, cached_download
from string import punctuation
import flair
from flair.data import Sentence, EntityLinkingLabel
from flair.datasets import (
    NEL_CTD_CHEMICAL_DICT,
    NEL_CTD_DISEASE_DICT,
    NEL_NCBI_HUMAN_GENE_DICT,
    NEL_NCBI_TAXONOMY_DICT,
)
from flair.file_utils import cached_path
from typing import List, Tuple, Union
from pathlib import Path
import subprocess
import tempfile
import faiss
from flair.embeddings import TransformerDocumentEmbeddings
from string import punctuation

log = logging.getLogger("flair")


class BigramTfIDFVectorizer:
    """
    Class to encode a list of mentions into a sparse tensor.

    Slightly modified from Sung et al. 2020
    Biomedical Entity Representations with Synonym Marginalization
    https://github.com/dmis-lab/BioSyn/tree/master/src/biosyn/sparse_encoder.py#L8
    """

    def __init__(self) -> None:
        self.encoder = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))

    def transform(self, mentions: list) -> torch.Tensor:
        vec = self.encoder.transform(mentions).toarray()
        vec = torch.FloatTensor(vec)
        return vec

    def __call__(self, mentions: list):
        return self.transform(mentions)

    def save_encoder(self, path: Path):
        with open(path, "wb") as fout:
            pickle.dump(self.encoder, fout)
            log.info("Sparse encoder saved in {}".format(path))

    @classmethod
    def load(cls, path: Path):
        newVectorizer = cls()
        with open(path, "rb") as fin:
            newVectorizer.encoder = pickle.load(fin)
            log.info("Sparse encoder loaded from {}".format(path))

        return newVectorizer


class TextPreprocess:
    """
    Text Preprocess module
    Support lowercase, removing punctuation, typo correction

    Slightly modifed from Sung et al. 2020
    Biomedical Entity Representations with Synonym Marginalization
    https://github.com/dmis-lab/BioSyn/blob/master/src/biosyn/preprocesser.py#L5
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
    ) -> None:
        """
        :param typo_path str: path of known typo dictionary
        """
        self.lowercase = lowercase
        self.rmv_puncts = remove_punctuation
        self.punctuation = punctuation
        self.rmv_puncts_regex = re.compile(
            r"[\s{}]+".format(re.escape(self.punctuation))
        )

    def remove_punctuation(self, phrase: str) -> str:
        phrase = self.rmv_puncts_regex.split(phrase)
        phrase = " ".join(phrase).strip()

        return phrase

    def run(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()

        if self.rmv_puncts:
            text = self.remove_punctuation(text)

        text = text.strip()

        return text


class Ab3P:
    """
    Module for the Abbreviation Resolver Ab3P
    https://github.com/ncbi-nlp/Ab3P
    """

    def __init__(self, ab3p_path: Path, word_data_dir: Path) -> None:
        self.ab3p_path = ab3p_path
        self.word_data_dir = word_data_dir

    @classmethod
    def load(cls, ab3p_path: Path = None):
        data_dir = os.path.join(flair.cache_root, "ab3p")
        if not os.path.exists(data_dir):
            os.mkdir(os.path.join(data_dir))
        word_data_dir = os.path.join(data_dir, "word_data/")
        if not os.path.exists(word_data_dir):
            os.mkdir(word_data_dir)
        if ab3p_path is None:
            ab3p_path = cls.download_ab3p(data_dir, word_data_dir)
        return cls(ab3p_path, word_data_dir)

    @classmethod
    def download_ab3p(cls, data_dir: Path, word_data_dir: Path) -> Path:
        # download word data for Ab3P if not already downloaded
        ab3p_url = (
            "https://raw.githubusercontent.com/dmis-lab/BioSyn/master/Ab3P/WordData/"
        )
        ab3p_files = [
            "Ab3P_prec.dat",
            "Lf1chSf",
            "SingTermFreq.dat",
            "cshset_wrdset3.ad",
            "cshset_wrdset3.ct",
            "cshset_wrdset3.ha",
            "cshset_wrdset3.nm",
            "cshset_wrdset3.str",
            "hshset_Lf1chSf.ad",
            "hshset_Lf1chSf.ha",
            "hshset_Lf1chSf.nm",
            "hshset_Lf1chSf.str",
            "hshset_stop.ad",
            "hshset_stop.ha",
            "hshset_stop.nm",
            "hshset_stop.str",
            "stop",
        ]
        for file in ab3p_files:
            data_path = cached_path(ab3p_url + file, word_data_dir)
        # download ab3p executable
        ab3p_path = cached_path(
            "https://github.com/dmis-lab/BioSyn/raw/master/Ab3P/identify_abbr", data_dir
        )
        os.chmod(ab3p_path, stat.S_IEXEC)
        return ab3p_path

    def build_abbreviation_dict(self, sentences: List[flair.data.Sentence]) -> dict:
        abbreviation_dict = {}
        # tempfile to store the data to pass to the ab3p executable
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as temp_file:
            for sentence in sentences:
                temp_file.write(sentence.to_tokenized_string() + "\n")
            temp_file.flush()
            # temporarily create path file in the current working directory for Ab3P
            with open(os.path.join(os.getcwd(), "path_Ab3P"), "w") as path_file:
                path_file.write(self.word_data_dir + "\n")
            # run ab3p with the temp file containing the dataset
            try:
                result = subprocess.run(
                    [self.ab3p_path, temp_file.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except:
                log.error(
                    """The abbreviation resolver Ab3P could not be run on your system. To ensure maximum accuracy, please
                install Ab3P yourself. See https://github.com/ncbi-nlp/Ab3P"""
                )
            else:
                line = result.stdout.decode("utf-8")
                if "Path file for type cshset does not exist!" in line:
                    log.error(
                        "Error when using Ab3P for abbreviation resolution. A file named path_Ab3p needs to exist in your current directory containing the path to the WordData directory for Ab3P to work!"
                    )
                elif "Cannot open" in line:
                    log.error(
                        "Error when using Ab3P for abbreviation resolution. Could not open the WordData directory for Ab3P!"
                    )
                elif "failed to open" in line:
                    log.error(
                        "Error when using Ab3P for abbreviation resolution. Could not open the WordData directory for Ab3P!"
                    )

                lines = line.split("\n")
                for line in lines:
                    if len(line.split("|")) == 3:
                        sf, lf, _ = line.split("|")
                        sf = sf.strip().lower()
                        lf = lf.strip().lower()
                        abbreviation_dict[sf] = lf
            finally:
                # remove the path file
                os.remove(os.path.join(os.getcwd(), "path_Ab3P"))

        return abbreviation_dict


class DictionaryDataset:
    """
    A class used to load dictionary data from a custom dictionary file.
    Every line in the file must be formatted as follows:
    concept_unique_id||concept_name
    with one line per concept name. Multiple synonyms for the same concept should
    be in seperate lines with the same concept_unique_id.

    Slightly modifed from Sung et al. 2020
    Biomedical Entity Representations with Synonym Marginalization
    https://github.com/dmis-lab/BioSyn/blob/master/src/biosyn/data_loader.py#L89
    """

    def __init__(
        self, dictionary_path: Union[Path, str], load_into_memory: True
    ) -> None:
        """
        :param dictionary_path str: The path of the dictionary
        """
        log.info("Loading Dictionary from {}".format(dictionary_path))
        if load_into_memory:
            self.data = self.load_data(dictionary_path)
        else:
            self.data = self.get_data(dictionary_path)

    def load_data(self, dictionary_path) -> np.ndarray:
        data = []
        with open(dictionary_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Loading dictionary"):
                line = line.strip()
                if line == "":
                    continue
                cui, name = line.split("||")
                name = name.lower()
                data.append((name, cui))

        data = np.array(data)
        return data

    # generator version
    def get_data(self, dictionary_path):
        data = []
        with open(dictionary_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Loading dictionary"):
                line = line.strip()
                if line == "":
                    continue
                cui, name = line.split("||")
                name = name.lower()
                yield (name, cui)


class BiEncoderEntityLiker:
    """
    A class to load a model and use it to encode a dictionary and entities
    """

    def __init__(
        self, use_sparse_embeds: bool, max_length: int, index_use_cuda: bool
    ) -> None:
        self.use_sparse_embeds = use_sparse_embeds
        self.max_length = max_length

        self.tokenizer = None
        self.encoder = None

        self.sparse_encoder = None
        self.sparse_weight = None

        self.index_use_cuda = index_use_cuda and flair.device.type == "cuda"

        self.dense_dictionary_index = None
        self.dict_sparse_embeds = None

        self.dictionary = None

    def load_model(
        self,
        model_name_or_path: Union[str, Path],
        dictionary_name_or_path: str,
        batch_size: int = 1024,
        use_cosine: bool = True,
    ):
        """
        Load the model and embed the dictionary
        :param model_name_or_path: The path of the model
        :param dictionary_name_or_path: The path of the dictionary
        :param batch_size: The batch size for embedding the dictionary
        """
        self.load_dense_encoder(model_name_or_path)
        self.use_cosine = True

        if self.use_sparse_embeds:
            self.load_sparse_encoder(model_name_or_path)
            self.load_sparse_weight(model_name_or_path)

        self.embed_dictionary(
            model_name_or_path=model_name_or_path,
            dictionary_name_or_path=dictionary_name_or_path,
            batch_size=batch_size,
        )

        return self

    def load_dense_encoder(
        self, model_name_or_path: Union[str, Path]
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

        self.encoder = TransformerDocumentEmbeddings(
            model_name_or_path, is_token_embedding=False
        )

        return self.encoder

    def load_sparse_encoder(
        self, model_name_or_path: Union[str, Path]
    ) -> BigramTfIDFVectorizer:
        sparse_encoder_path = os.path.join(model_name_or_path, "sparse_encoder.pk")
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            # download from huggingface hub and cache it
            sparse_encoder_url = hf_hub_url(
                model_name_or_path, filename="sparse_encoder.pk"
            )
            sparse_encoder_path = cached_download(
                url=sparse_encoder_url,
                cache_dir=flair.cache_root / "models" / model_name_or_path,
            )

        self.sparse_encoder = BigramTfIDFVectorizer.load(path=sparse_encoder_path)

        return self.sparse_encoder

    def load_sparse_weight(self, model_name_or_path: Union[str, Path]) -> torch.Tensor:
        sparse_weight_path = os.path.join(model_name_or_path, "sparse_weight.pt")
        # check file exists
        if not os.path.isfile(sparse_weight_path):
            # download from huggingface hub and cache it
            sparse_weight_url = hf_hub_url(
                model_name_or_path, filename="sparse_weight.pt"
            )
            sparse_weight_path = cached_download(
                url=sparse_weight_url,
                cache_dir=flair.cache_root / "models" / model_name_or_path,
            )

        self.sparse_weight = torch.load(sparse_weight_path, map_location="cpu")

        return self.sparse_weight

    def get_sparse_weight(self) -> torch.Tensor:
        assert self.sparse_weight is not None

        return self.sparse_weight

    def embed_sparse(self, names: list) -> np.ndarray:
        """
        Embedding data into sparse representations
        :param names np.array: An array of names
        :returns sparse_embeds np.array: A list of sparse embeddings
        """
        sparse_embeds = self.sparse_encoder(names)
        sparse_embeds = sparse_embeds.numpy()

        return sparse_embeds

    def embed_dense(
        self,
        names: np.ndarray,
        show_progress: bool = False,
        batch_size: int = 2048,
    ) -> np.ndarray:
        """
        Embedding data into dense representations for SapBert
        :param names: np.array of names
        :param show_progress: bool to toggle progress bar
        :param batch_size: batch size
        :return dense_embeds: list of dense embeddings of the names
        """
        self.encoder.eval()  # prevent dropout

        dense_embeds = []

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(
                    range(0, len(names), batch_size),
                    desc="Calculating dense embeddings for dictionary",
                )
            else:
                iterations = range(0, len(names), batch_size)

            for start in iterations:

                # make batch
                end = min(start + batch_size, len(names))
                batch = [Sentence(name) for name in names[start:end]]

                # embed batch
                self.encoder.embed(batch)

                dense_embeds += [
                    name.embedding.cpu().detach().numpy() for name in batch
                ]

                if flair.device.type == "cuda":
                    torch.cuda.empty_cache()

        return np.array(dense_embeds)

    def get_sparse_similarity_scores(
        self,
        query_embeds: np.ndarray,
        dict_embeds: np.ndarray,
        cosine: bool = False,
        normalise: bool = False,
    ) -> np.ndarray:
        """
        Return score matrix
        :param query_embeds: 2d numpy array of query embeddings
        :param dict_embeds: 2d numpy array of query embeddings
        :param score_matrix: 2d numpy array of scores
        """
        if cosine:
            score_matrix = cosine_similarity(query_embeds, dict_embeds)
        else:
            score_matrix = np.matmul(query_embeds, dict_embeds.T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min()) / (
                score_matrix.max() - score_matrix.min()
            )

        return score_matrix

    def retrieve_sparse_topk_candidates(
        self,
        score_matrix: np.ndarray,
        topk: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return sorted topk idxes (descending order)
        :param score_matrix: 2d numpy array of scores
        :param topk: number of candidates to retrieve
        :return res: d numpy array of ids [# of query , # of dict]
        :return scores: numpy array of top scores
        """

        def indexing_2d(arr, cols):
            rows = np.repeat(
                np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1
            )
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)
        topk_scores = indexing_2d(score_matrix, topk_idxs)

        return topk_idxs, topk_scores

    def get_predictions(
        self,
        mention: str,
        topk: int,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Return the topk predictions for a mention and their scores
        :param mention: string of the mention to find candidates for
        :param topk: number of candidates
        :return res: d numpy array of ids [# of query , # of dict]
        :return scores: numpy array of top predictions and their scores
        """
        # get dense embeds for mention
        mention_dense_embeds = self.embed_dense(names=[mention])

        if self.use_cosine:
            # normalize mention vector
            faiss.normalize_L2(mention_dense_embeds)

        assert (
            self.dense_dictionary_index is not None
        ), "Index not built yet, please run load_model to embed your dictionary before calling get_predictions"

        # if using sparse embeds: calculate hybrid scores with dense and sparse embeds
        if self.use_sparse_embeds:
            assert (
                self.dict_sparse_embeds is not None
            ), "Index not built yet, please run load_model to embed your dictionary before calling get_predictions"
            # search for more than topk candidates to use them when combining with sparse scores
            # get candidates from dense embeddings
            dense_scores, dense_ids = self.dense_dictionary_index.search(
                x=mention_dense_embeds, k=topk + 10
            )
            # get sparse embeds for mention
            mention_sparse_embeds = self.embed_sparse(names=[mention])
            if self.use_cosine:
                # normalize mention vector
                faiss.normalize_L2(mention_sparse_embeds)

            # get candidates from sprase embeddings
            sparse_weight = self.get_sparse_weight().item()
            sparse_score_matrix = self.get_sparse_similarity_scores(
                query_embeds=mention_sparse_embeds, dict_embeds=self.dict_sparse_embeds
            )
            sparse_ids, sparse_distances = self.retrieve_sparse_topk_candidates(
                score_matrix=sparse_score_matrix, topk=topk + 10
            )

            # combine dense and sparse scores
            hybrid_ids = []
            hybrid_scores = []
            # for every embedded mention
            for (
                top_dense_ids,
                top_dense_scores,
                top_sparse_ids,
                top_sparse_distances,
            ) in zip(dense_ids, dense_scores, sparse_ids, sparse_distances):
                ids = top_dense_ids
                distances = top_dense_scores
                for sparse_id, sparse_distance in zip(
                    top_sparse_ids, top_sparse_distances
                ):
                    if sparse_id not in ids:
                        ids = np.append(ids, sparse_id)
                        distances = np.append(
                            distances, sparse_weight * sparse_distance
                        )
                    else:
                        index = np.where(ids == sparse_id)[0][0]
                        distances[index] = (
                            sparse_weight * sparse_distance
                        ) + distances[index]

                sorted_indizes = np.argsort(-distances)
                ids = ids[sorted_indizes][:topk]
                distances = distances[sorted_indizes][:topk]
                hybrid_ids.append(ids.tolist())
                hybrid_scores.append(distances.tolist())

            return [
                np.append(self.dictionary[ind], score)
                for ind, score in zip(hybrid_ids, hybrid_scores)
            ]
        # use only dense embeds
        else:
            dense_distances, dense_ids = self.dense_dictionary_index.search(
                x=mention_dense_embeds, k=topk
            )
            return [
                np.append(self.dictionary[ind], score)
                for ind, score in zip(dense_ids, dense_distances)
            ]

    def embed_dictionary(
        self, model_name_or_path: str, dictionary_name_or_path: str, batch_size: int
    ):
        # check for embedded dictionary in cache
        dictionary_name = os.path.splitext(os.path.basename(dictionary_name_or_path))[0]
        cache_folder = os.path.join(flair.cache_root, "datasets")
        file_name = f"bio_nen_{model_name_or_path.split('/')[-1]}_{dictionary_name}"
        cached_dictionary_path = os.path.join(
            cache_folder,
            f"{file_name}.pk",
        )
        self.load_dictionary(dictionary_name_or_path)

        # If exists, load the cached dictionary indices
        if os.path.exists(cached_dictionary_path):
            self.load_cached_dictionary(cached_dictionary_path)

        # else, load and embed
        else:

            # get names from dictionary and remove punctuation
            punctuation_regex = re.compile(r"[\s{}]+".format(re.escape(punctuation)))
            dictionary_names = []
            for row in self.dictionary:
                name = punctuation_regex.split(row[0])
                name = " ".join(name).strip().lower()
                dictionary_names.append(name)

            # create dense and sparse embeddings
            if self.use_sparse_embeds:
                self.dict_dense_embeds = self.embed_dense(
                    names=dictionary_names,
                    batch_size=batch_size,
                    show_progress=True
                )
                self.dict_sparse_embeds = self.embed_sparse(names=dictionary_names)

            # create only dense embeddings
            else:
                self.dict_dense_embeds = self.embed_dense(
                    names=dictionary_names, show_progress=True
                )
                self.dict_sparse_embeds = None

            # build dense index
            dimension = self.dict_dense_embeds.shape[1]
            if self.use_cosine:
                # to use cosine similarity, we normalize the vectors and then use inner product
                faiss.normalize_L2(self.dict_dense_embeds)

            self.dense_dictionary_index = faiss.IndexFlatIP(dimension)
            if self.index_use_cuda:
                self.dense_dictionary_index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.dense_dictionary_index
                )
            self.dense_dictionary_index.add(self.dict_dense_embeds)

            if self.index_use_cuda:
                # create a cpu version of the index to cache (IO does not work with gpu index)
                cache_index_dense = faiss.index_gpu_to_cpu(self.dense_dictionary_index)
                # cache dictionary
                self.cache_dictionary(
                    cache_index_dense, cache_folder, cached_dictionary_path
                )
            else:
                self.cache_dictionary(
                    self.dense_dictionary_index, cache_folder, cached_dictionary_path
                )

    def load_cached_dictionary(self, cached_dictionary_path: str):
        with open(cached_dictionary_path, "rb") as cached_file:
            cached_dictionary = pickle.load(cached_file)
            log.info(
                "Loaded dictionary from cached file {}".format(cached_dictionary_path)
            )

            (self.dictionary, self.dict_sparse_embeds, self.dense_dictionary_index,) = (
                cached_dictionary["dictionary"],
                cached_dictionary["sparse_dictionary_embeds"],
                cached_dictionary["dense_dictionary_index"],
            )
            if self.index_use_cuda:
                self.dense_dictionary_index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.dense_dictionary_index
                )

    def load_dictionary(self, dictionary_name_or_path: str):
        # use provided dictionary
        if dictionary_name_or_path == "ctd-disease":
            self.dictionary = NEL_CTD_DISEASE_DICT().data
        elif dictionary_name_or_path == "ctd-chemical":
            self.dictionary = NEL_CTD_CHEMICAL_DICT().data
        elif dictionary_name_or_path == "ncbi-gene":
            self.dictionary = NEL_NCBI_HUMAN_GENE_DICT().data
        elif dictionary_name_or_path == "ncbi-taxonomy":
            self.dictionary = NEL_NCBI_TAXONOMY_DICT().data
        # use custom dictionary file
        else:
            self.dictionary = DictionaryDataset(
                dictionary_path=dictionary_name_or_path
            ).data

    def cache_dictionary(self, cache_index_dense, cache_folder, cached_dictionary_path):
        cached_dictionary = {
            "dictionary": self.dictionary,
            "sparse_dictionary_embeds": self.dict_sparse_embeds,
            "dense_dictionary_index": cache_index_dense,
        }
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        with open(cached_dictionary_path, "wb") as cache_file:
            pickle.dump(cached_dictionary, cache_file)
        print("Saving dictionary into cached file {}".format(cache_folder))


class ExactStringMatchEntityLinker:
    def __init__(self):
        self.dictionary = None

    def load_model(
        self,
        dictionary_name_or_path: str,
    ):
        # use provided dictionary
        if dictionary_name_or_path == "ctd-disease":
            dictionary_data = NEL_CTD_DISEASE_DICT().data
        elif dictionary_name_or_path == "ctd-chemical":
            dictionary_data = NEL_CTD_CHEMICAL_DICT().data
        elif dictionary_name_or_path == "ncbi-gene":
            dictionary_data = NEL_NCBI_HUMAN_GENE_DICT().data
        elif dictionary_name_or_path == "ncbi-taxonomy":
            dictionary_data = NEL_NCBI_TAXONOMY_DICT().data
        # use custom dictionary file
        else:
            dictionary_data = DictionaryDataset(
                dictionary_path=dictionary_name_or_path
            ).data

        # make dictionary from array of tuples (entity, entity_id)
        self.dictionary = {name: cui for name, cui in dictionary_data}

    def get_predictions(self, mention: str, topk) -> np.ndarray:
        if mention in self.dictionary:
            return np.array([(mention, self.dictionary[mention], 1.0)])
        else:
            return []


class MultiBiEncoderEntityLinker:
    """
    Biomedical Entity Linker for HunFlair
    Can predict top k entities on sentences annotated with biomedical entity mentions
    """

    def __init__(
        self,
        models: List[BiEncoderEntityLiker],
        ab3p=None,
    ) -> None:
        """
        Initalize class, called by classmethod load
        :param models: list of objects containing the dense and sparse encoders
        :param ab3p_path: path to ab3p model
        """
        self.models = models
        self.text_preprocessor = TextPreprocess()
        self.ab3p = ab3p

    @classmethod
    def load(
        cls,
        model_names: Union[List[str], str],
        dictionary_names_or_paths: Union[str, Path, List[str], List[Path]] = None,
        use_sparse_and_dense_embeds: bool = True,
        max_length=25,
        batch_size=1024,
        index_use_cuda=False,
        use_cosine: bool = True,
        use_ab3p: bool = True,
        ab3p_path: Path = None,
    ):
        """
        Load a model for biomedical named entity normalization on sentences annotated with
        biomedical entity mentions
        :param model_names: List of names of pretrained models to use. Possible values for pretrained models are:
        chemical, disease, gene, sapbert-bc5cdr-dissaease, sapbert-ncbi-disease, sapbert-bc5cdr-chemical, biobert-bc5cdr-disease,
        biobert-ncbi-disease, biobert-bc5cdr-chemical, biosyn-biobert-bc2gn, biosyn-sapbert-bc2gn, sapbert, exact-string-match
        :param dictionary_path: Name of one of the provided dictionaries listing all possible ids and their synonyms
        or a path to a dictionary file with each line in the format: id||name, with one line for each name of a concept.
        Possible values for dictionaries are: chemical, ctd-chemical, disease, bc5cdr-disease, gene, cnbci-gene,
        taxonomy and ncbi-taxonomy
        :param use_sparse_and_dense_embeds: If True, uses a combinations of sparse and dense embeddings for the dictionary and the mentions
        If False, uses only dense embeddings
        :param batch_size: Batch size for the dense encoder
        :param index_use_cuda: If True, uses GPU for the dense encoder
        :param use_cosine: If True, uses cosine similarity for the dense encoder. If False, uses inner product
        :param use_ab3p: If True, uses ab3p to resolve abbreviations
        :param ab3p_path: Optional: oath to ab3p on your machine
        """
        # validate input: check that amount of models and dictionaries match
        if isinstance(model_names, str):
            model_names = [model_names]

        # case one dictionary for all models
        if isinstance(dictionary_names_or_paths, str) or isinstance(
            dictionary_names_or_paths, Path
        ):
            dictionary_names_or_paths = [dictionary_names_or_paths] * len(model_names)
        # case no dictionary provided
        elif dictionary_names_or_paths is None:
            dictionary_names_or_paths = [None] * len(model_names)
        # case one model, multiple dictionaries
        elif len(model_names) == 1:
            model_names = model_names * len(dictionary_names_or_paths)
        # case mismatching amount of models and dictionaries
        elif len(model_names) != len(dictionary_names_or_paths):
            raise ValueError(
                "Amount of models and dictionaries must match. Got {} models and {} dictionaries".format(
                    len(model_names), len(dictionary_names_or_paths)
                )
            )
        assert len(model_names) == len(dictionary_names_or_paths)

        models = []

        for model_name, dictionary_name_or_path in zip(
            model_names, dictionary_names_or_paths
        ):
            # get the paths for the model and dictionary
            model_path = cls.__get_model_path(model_name, use_sparse_and_dense_embeds)
            dictionary_path = cls.__get_dictionary_path(
                dictionary_name_or_path, model_name=model_name
            )

            if model_path == "exact-string-match":
                model = ExactStringMatchEntityLinker()
                model.load_model(dictionary_path)

            else:
                model = BiEncoderEntityLiker(
                    use_sparse_embeds=use_sparse_and_dense_embeds,
                    max_length=max_length,
                    index_use_cuda=index_use_cuda,
                )

                model.load_model(
                    model_name_or_path=model_path,
                    dictionary_name_or_path=dictionary_path,
                    batch_size=batch_size,
                    use_cosine=use_cosine,
                )

            models.append(model)

        # load ab3p model
        ab3p = Ab3P.load(ab3p_path) if use_ab3p else None

        return cls(models, ab3p)

    @staticmethod
    def __get_model_path(
        model_name: str, use_sparse_and_dense_embeds
    ) -> BiEncoderEntityLiker:
        model_name = model_name.lower()
        model_path = model_name

        # if a provided model is used,
        # modify model name to huggingface path

        if model_name in [
            "sapbert-bc5cdr-disease",
            "sapbert-ncbi-disease",
            "sapbert-bc5cdr-chemical",
            "biobert-bc5cdr-disease",
            "biobert-ncbi-disease",
            "biobert-bc5cdr-chemical",
            "biosyn-biobert-bc2gn",
            "biosyn-sapbert-bc2gn",
        ]:
            model_path = "dmis-lab/biosyn-" + model_name
        elif model_name == "sapbert":
            model_path = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        elif model_name == "exact-string-match":
            model_path = "exact-string-match"
        elif use_sparse_and_dense_embeds:
            if model_name == "disease":
                model_path = "dmis-lab/biosyn-sapbert-bc5cdr-disease"
            elif model_name == "chemical":
                model_path = "dmis-lab/biosyn-sapbert-bc5cdr-chemical"
            elif model_name == "gene":
                model_path = "dmis-lab/biosyn-sapbert-bc2gn"
        else:
            if model_name == "disease":
                model_path = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            elif model_name == "chemical":
                model_path = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            elif model_name == "gene":
                raise ValueError(
                    "No trained model for gene entity linking using only dense embeddings."
                )
        return model_path

    @staticmethod
    def __get_dictionary_path(dictionary_path: str, model_name: str):
        # determine dictionary to use
        if dictionary_path == "disease":
            dictionary_path = "ctd-disease"
        if dictionary_path == "chemical":
            dictionary_path = "ctd-chemical"
        if dictionary_path == "gene":
            dictionary_path = "ncbi-gene"
        if dictionary_path == "taxonomy":
            dictionary_path = "ncbi-taxonomy"
        if dictionary_path is None:
            # disease
            if model_name in [
                "sapbert-bc5cdr-disease",
                "sapbert-ncbi-disease",
                "biobert-bc5cdr-disease",
                "biobert-ncbi-disease",
                "disease",
            ]:
                dictionary_path = "ctd-disease"
            # chemical
            elif model_name in [
                "sapbert-bc5cdr-chemical",
                "biobert-bc5cdr-chemical",
                "chemical",
            ]:
                dictionary_path = "ctd-chemical"
            # gene
            elif model_name in ["gene", "biosyn-biobert-bc2gn", "biosyn-sapbert-bc2gn"]:
                dictionary_path = "ncbi-gene"
            # error
            else:
                log.error(
                    """When using a custom model you need to specify a dictionary. 
                Available options are: 'disease', 'chemical', 'gene' and 'taxonomy'.
                Or provide a path to a dictionary file."""
                )
                raise ValueError("Invalid dictionary")

        return dictionary_path

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        input_entity_annotation_layer: str = None,
        topk: int = 1,
    ) -> None:
        """
        On one or more sentences, predict the cui on all named entites annotated with a tag of type input_entity_annotation_layer.
        Annotates the top k predictions.
        :param sentences: one or more sentences to run the predictions on
        :param input_entity_annotation_layer: only entities with in this annotation layer will be annotated
        :param topk: number of predicted cui candidates to add to annotation
        :param abbreviation_dict: dictionary with abbreviations and their expanded form or a boolean value indicating whether
        abbreviations should be expanded using Ab3P
        """
        # make sure sentences is a list of sentences
        if not isinstance(sentences, list):
            sentences = [sentences]

        # use Ab3P to build abbreviation dictionary
        if self.ab3p is not None:
            abbreviation_dict = self.ab3p.build_abbreviation_dict(sentences)

        for model in self.models:

            for sentence in sentences:
                for entity in sentence.get_labels(input_entity_annotation_layer):
                    # preprocess mention
                    if abbreviation_dict is not None:
                        parsed_tokens = []
                        for token in entity.data_point.tokens:
                            token = self.text_preprocessor.run(token.text)
                            if token in abbreviation_dict:
                                parsed_tokens.append(abbreviation_dict[token.lower()])
                            elif len(token) != 0:
                                parsed_tokens.append(token)
                        mention = " ".join(parsed_tokens)
                    else:
                        mention = self.text_preprocessor.run(entity.span.text)

                    # get predictions from dictionary
                    predictions = model.get_predictions(mention, topk)

                    # add predictions to entity
                    label_name = (
                        (input_entity_annotation_layer + "_nen")
                        if (input_entity_annotation_layer is not None)
                        else "nen"
                    )
                    for prediction in predictions:
                        # if concept unique id is made up of mulitple ids, seperated by '|'
                        # seperate it into cui and additional_labels
                        cui = prediction[1]
                        if "|" in cui:
                            labels = cui.split("|")
                            cui = labels[0]
                            additional_labels = labels[1:]
                        else:
                            additional_labels = None
                        # determine database:
                        if ":" in cui:
                            cui_parts = cui.split(":")
                            database = ":".join(cui_parts[0:-1])
                            cui = cui_parts[-1]
                        else:
                            database = None
                        sentence.add_label(
                            typename=label_name,
                            value_or_label=EntityLinkingLabel(
                                data_point=entity.data_point,
                                id=cui,
                                concept_name=prediction[0],
                                additional_ids=additional_labels,
                                database=database,
                                score=prediction[2].astype(float),
                            ),
                        )
