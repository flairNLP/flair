import os
import stat
import re
import pickle
import logging
from statistics import mode
from xmlrpc.client import Boolean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    default_data_collator,
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
    NEL_NCBI_GENE_DICT,
    NEL_NCBI_TAXONOMY_DICT,
)
from flair.file_utils import cached_path
from typing import List, Tuple, Union
from pathlib import Path
import subprocess
import tempfile
import faiss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer

log = logging.getLogger("flair")


class SparseEncoder:
    """
    Class to encode a list of mentions into a sparse tensor.

    Slightly modified from Sung et al. 2020
    Biomedical Entity Representations with Synonym Marginalization
    https://github.com/dmis-lab/BioSyn/tree/master/src/biosyn/sparse_encoder.py#L8
    """

    def __init__(self, use_cuda: bool = False) -> None:
        self.encoder = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        self.use_cuda = use_cuda

    def transform(self, mentions: list) -> torch.Tensor:
        vec = self.encoder.transform(mentions).toarray()
        vec = torch.FloatTensor(vec)  # return torch float tensor
        if self.use_cuda:
            vec = vec.cuda()
        return vec

    def cuda(self):
        self.use_cuda = True

        return self

    def cpu(self):
        self.use_cuda = False
        return self

    def __call__(self, mentions: list):
        return self.transform(mentions)

    def save_encoder(self, path: Path):
        with open(path, "wb") as fout:
            pickle.dump(self.encoder, fout)
            log.info("Sparse encoder saved in {}".format(path))

    def load_encoder(self, path: Path):
        with open(path, "rb") as fin:
            self.encoder = pickle.load(fin)
            log.info("Sparse encoder loaded from {}".format(path))

        return self


class TextPreprocess:
    """
    Text Preprocess module
    Support lowercase, removing punctuation, typo correction

    SSlightly modifed from Sung et al. 2020
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


class NamesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings) -> None:
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


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

    def __init__(self, dictionary_path: Union[Path, str]) -> None:
        """
        :param dictionary_path str: The path of the dictionary
        """
        log.info("Loading Dictionary from {}".format(dictionary_path))
        self.data = self.load_data(dictionary_path)

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


class BioEntityLinkingModel:
    def __init__(self, use_sparse_embeds: bool, max_length: int) -> None:
        self.use_sparse_embeds = use_sparse_embeds
        self.max_length = max_length

        self.tokenizer = None
        self.encoder = None

        self.sparse_encoder = None
        self.sparse_weight = None

        self.use_cuda = torch.cuda.is_available()

    def load_model(
        self,
        model_name_or_path: Union[str, Path],
        dictionary_name_or_path: str,
        mean_centering: bool,
    ):
        self.load_dense_encoder(model_name_or_path)

        if self.use_sparse_embeds:
            self.load_sparse_encoder(model_name_or_path)
            self.load_sparse_weight(model_name_or_path)

        # self.embed_dictionary(
        #     model_name_or_path=model_name_or_path,
        #     dictionary_name_or_path=dictionary_name_or_path,
        #     mean_centering=mean_centering,
        # )
        self.embed_dictionary_faiss(
            model_name_or_path=model_name_or_path,
            dictionary_name_or_path=dictionary_name_or_path,
            mean_centering=mean_centering,
        )

        return self

    # always used
    def load_dense_encoder(
        self, model_name_or_path: Union[str, Path]
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        # load dense encoder from path or from hugging face
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, do_lower_case=True
        )
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")

        return self.encoder, self.tokenizer

    def load_sparse_encoder(
        self, model_name_or_path: Union[str, Path]
    ) -> SparseEncoder:
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

        self.sparse_encoder = SparseEncoder().load_encoder(path=sparse_encoder_path)

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

        self.sparse_weight = torch.load(sparse_weight_path)

        return self.sparse_weight

    def get_sparse_weight(self) -> torch.Tensor:
        assert self.sparse_weight is not None

        return self.sparse_weight

    def embed_sparse(self, names: list, show_progress: bool = False) -> np.ndarray:
        """
        Embedding data into sparse representations
        :param names np.array: An array of names
        :returns sparse_embeds np.array: A list of sparse embeddings
        """
        batch_size = 1024
        sparse_embeds = []

        if show_progress:
            iterations = tqdm(
                range(0, len(names), batch_size),
                desc="Calculating sparse embeddings for dictionary",
            )
        else:
            iterations = range(0, len(names), batch_size)

        for start in iterations:
            end = min(start + batch_size, len(names))
            batch = names[start:end]
            batch_sparse_embeds = self.sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def biosyn_embed_dense(
        self, names: list, show_progress: bool = False, batch_size: int = 1024
    ) -> np.ndarray:
        """
        Embedding data into dense representations for BioSyn
        :param names list: An array of names
        :returns dense_embeds list: A list of dense embeddings
        """
        self.encoder.eval()  # prevent dropout

        batch_size = 1024
        dense_embeds = []

        name_encodings = self.tokenizer(
            names,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        if self.use_cuda:
            name_encodings = name_encodings.to("cuda")
        name_dataset = NamesDataset(name_encodings)
        name_dataloader = DataLoader(
            name_dataset,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=batch_size,
        )

        with torch.no_grad():
            for batch in tqdm(
                name_dataloader,
                disable=not show_progress,
                desc="Calculating dense embeddings for dictionary",
            ):
                outputs = self.encoder(**batch)
                batch_dense_embeds = (
                    outputs[0][:, 0].cpu().detach().numpy()
                )  # [CLS] representations
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)

        return dense_embeds

    # Attention: uses cuda
    def sapbert_embed_dense(
        self,
        names: np.ndarray,
        show_progress: bool = False,
        batch_size: int = 2048,
        agg_mode: str = "cls",
    ) -> np.ndarray:
        """
        Embedding data into dense representations for SapBert
        :param names: np.array of names
        :param show_progress: bool to toggle progress bar
        :param batch_size: batch size
        :param agg_mode: options are 'cls', 'mean_all_tol' and 'mean'
        :return dense_embeds: list of dense embeddings
        """
        self.encoder.eval()  # prevent dropout

        # Difference: batch size given as parameter
        batch_size = batch_size
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
                end = min(start + batch_size, len(names))
                batch = names[start:end]
                batch_tokenized_names = self.tokenizer.batch_encode_plus(
                    batch,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch_tokenized_names_cuda = {}
                for k, v in batch_tokenized_names.items():
                    batch_tokenized_names_cuda[k] = v.cuda()

                last_hidden_state = self.encoder(**batch_tokenized_names_cuda)[0]
                if agg_mode == "cls":
                    batch_dense_embeds = last_hidden_state[:, 0, :]  # [CLS]
                elif agg_mode == "mean_all_tok":
                    batch_dense_embeds = last_hidden_state.mean(1)  # pooling
                elif agg_mode == "mean":
                    batch_dense_embeds = (
                        last_hidden_state
                        * batch_tokenized_names_cuda["attention_mask"].unsqueeze(-1)
                    ).sum(1) / batch_tokenized_names_cuda["attention_mask"].sum(
                        -1
                    ).unsqueeze(
                        -1
                    )
                else:
                    raise ValueError("agg_mode must be 'cls', 'mean_all_tok' or 'mean'")

                batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)

        return dense_embeds

    def get_score_matrix(
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

    def retrieve_candidate(
        self, score_matrix: np.ndarray, topk: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return sorted topk indxes (descending order)
        :param score_matrix np.array: 2d numpy array of scores
        :param topk int: The number of candidates
        :returns topk_idxs np.array: 2d numpy array of scores [# of query , # of dict]
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
        topk_scores = np.sort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs, -topk_scores

    def retrieve_candidate_cuda(
        self,
        score_matrix: np.ndarray,
        topk: int,
        batch_size: int = 128,
        show_progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return sorted topk idxes (descending order)
        :param score_matrix: 2d numpy array of scores
        :param topk: number of candidates
        :return res: d numpy array of ids [# of query , # of dict]
        :return scores: numpy array of top scores
        """

        res = None
        scores = None
        for i in tqdm(
            np.arange(0, score_matrix.shape[0], batch_size), disable=not show_progress
        ):
            score_matrix_tmp = torch.tensor(score_matrix[i : i + batch_size]).cuda()
            sorted_values, matrix_sorted = torch.sort(
                score_matrix_tmp, dim=1, descending=True
            )
            matrix_sorted = matrix_sorted[:, :topk].cpu()
            sorted_values = sorted_values[:, :topk].cpu()
            if res is None:
                res = matrix_sorted
                scores = sorted_values
            else:
                res = torch.cat([res, matrix_sorted], axis=0)
                scores = torch.cat([scores, sorted_values], axis=0)

        return res.numpy(), scores.numpy()

    # possible values for agg_mode: cls|mean_pool|nospec
    def get_predictions(
        self,
        mention: str,
        topk: int,
        tgt_space_mean_vec: np.ndarray = None,
        agg_mode: str = "cls",
    ) -> np.ndarray:
        # get dense embeds for mention
        if self.use_sparse_embeds:
            mention_dense_embeds = self.biosyn_embed_dense(names=[mention])
        else:
            mention_dense_embeds = self.sapbert_embed_dense(
                names=[mention], agg_mode=agg_mode
            )
        if tgt_space_mean_vec is not None:
            mention_dense_embeds -= tgt_space_mean_vec
        dense_score_matrix = self.get_score_matrix(
            query_embeds=mention_dense_embeds, dict_embeds=self.dict_dense_embeds
        )

        # when using sparse embeds: calculate hybrid scores with dense and sparse embeds
        if self.use_sparse_embeds:
            # get sparse embeds for mention
            mention_sparse_embeds = self.embed_sparse(names=[mention])
            sparse_score_matrix = self.get_score_matrix(
                query_embeds=mention_sparse_embeds, dict_embeds=self.dict_sparse_embeds
            )
            sparse_weight = self.get_sparse_weight().item()
            # calculate hybrid score matrix
            score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            candidate_idxs, candidate_scores = self.retrieve_candidate(
                score_matrix=score_matrix, topk=topk
            )
        # for only dense embeddings
        else:
            score_matrix = dense_score_matrix
            candidate_idxs, candidate_scores = self.retrieve_candidate_cuda(
                score_matrix=score_matrix, topk=topk, batch_size=16, show_progress=False
            )

        ids = candidate_idxs[0].tolist()
        scores = candidate_scores[0].tolist()

        return [
            np.append(self.dictionary[ind], score) for ind, score in zip(ids, scores)
        ]

    def get_predictions_faiss(
        self,
        mention: str,
        topk: int,
        tgt_space_mean_vec: np.ndarray = None,
        agg_mode: str = "cls",
    ) -> np.ndarray:

        # get dense embeds for mention
        if self.use_sparse_embeds:
            mention_dense_embeds = self.biosyn_embed_dense(names=[mention])
        else:
            mention_dense_embeds = self.sapbert_embed_dense(
                names=[mention], agg_mode=agg_mode
            )
        # remove mean centering
        if tgt_space_mean_vec is not None:
            mention_dense_embeds -= tgt_space_mean_vec

        # normalize mention vector
        faiss.normalize_L2(mention_dense_embeds)

        # if using sparse embeds: calculate hybrid scores with dense and sparse embeds
        if self.use_sparse_embeds:
            # search for more than topk candidates to use them when combining with sparse scores
            dense_scores, dense_ids = self.dense_dictionary_index.search(
                x=mention_dense_embeds, k=topk + 10
            )
            # get sparse embeds for mention
            mention_sparse_embeds = self.embed_sparse(names=[mention])
            # normalize mention vector
            faiss.normalize_L2(mention_sparse_embeds)

            sparse_weight = self.get_sparse_weight().item()
            sparse_distances, sparse_ids = self.sparse_dictionary_index.kneighbors(
                mention_sparse_embeds, n_neighbors=topk + 10
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
                            distances, sparse_weight * (1 - sparse_distance)
                        )
                    else:
                        index = np.where(ids == sparse_id)[0][0]
                        distances[index] = (
                            sparse_weight * (1 - sparse_distance) + distances[index]
                        )

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
        self,
        model_name_or_path: str,
        dictionary_name_or_path: str,
        mean_centering: bool,
    ):
        # check for embedded dictionary in cache
        dictionary_name = os.path.splitext(os.path.basename(dictionary_name_or_path))[0]
        cache_folder = os.path.join(flair.cache_root, "datasets")
        file_name = f"bio_nen_{model_name_or_path.split('/')[-1]}_{dictionary_name}"
        cached_dictionary_path = os.path.join(
            cache_folder,
            f"{file_name}.pk",
        )

        # If exists, load the cached dictionary indizes
        if os.path.exists(cached_dictionary_path):
            with open(cached_dictionary_path, "rb") as cached_file:
                cached_dictionary = pickle.load(cached_file)
            log.info(
                "Loaded dictionary from cached file {}".format(cached_dictionary_path)
            )

            (self.dictionary, self.dict_sparse_embeds, self.dict_dense_embeds,) = (
                cached_dictionary["dictionary"],
                cached_dictionary["sparse_dictionary_embeds"],
                cached_dictionary["dense_dictionary_embeds"],
            )

        # else, load and embed
        else:
            # use provided dictionary
            if dictionary_name_or_path == "ctd-disease":
                self.dictionary = NEL_CTD_DISEASE_DICT().data
            elif dictionary_name_or_path == "ctd-chemical":
                self.dictionary = NEL_CTD_CHEMICAL_DICT().data
            elif dictionary_name_or_path == "ncbi-gene":
                self.dictionary = NEL_NCBI_GENE_DICT().data
            elif dictionary_name_or_path == "ncbi-taxonomy":
                self.dictionary = NEL_NCBI_TAXONOMY_DICT().data
            # use custom dictionary file
            else:
                self.dictionary = DictionaryDataset(
                    dictionary_path=dictionary_name_or_path
                ).data

            # embed dictionary
            dictionary_names = [row[0] for row in self.dictionary]

            # create dense and sparse embeddings
            if self.use_sparse_embeds:
                self.dict_dense_embeds = self.biosyn_embed_dense(
                    names=dictionary_names, show_progress=True
                )
                self.dict_sparse_embeds = self.embed_sparse(
                    names=dictionary_names, show_progress=True
                )

            # create only dense embeddings
            else:
                self.dict_dense_embeds = self.sapbert_embed_dense(
                    names=dictionary_names, show_progress=True
                )
                self.dict_sparse_embeds = None
                self.sparse_dictionary_index = None

            # cache dictionary
            cached_dictionary = {
                "dictionary": self.dictionary,
                "sparse_dictionary_embeds": self.dict_sparse_embeds,
                "dense_dictionary_embeds": self.dict_dense_embeds,
            }
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)
            with open(cached_dictionary_path, "wb") as cache_file:
                pickle.dump(cached_dictionary, cache_file)
            print("Saving dictionary into cached file {}".format(cache_folder))

        # apply mean centering
        if mean_centering:
            self.tgt_space_mean_vec = self.dict_dense_embeds.mean(0)
            self.dict_dense_embeds -= self.tgt_space_mean_vec
        else:
            self.tgt_space_mean_vec = None

    def embed_dictionary_faiss(
        self,
        model_name_or_path: str,
        dictionary_name_or_path: str,
        mean_centering: bool,
    ):
        # check for embedded dictionary in cache
        dictionary_name = os.path.splitext(os.path.basename(dictionary_name_or_path))[0]
        cache_folder = os.path.join(flair.cache_root, "datasets")
        file_name = f"bio_nen_{model_name_or_path.split('/')[-1]}_{dictionary_name}"
        cached_dictionary_path = os.path.join(
            cache_folder,
            f"{file_name}.pk",
        )

        # If exists, load the cached dictionary indices
        if os.path.exists(cached_dictionary_path):
            with open(cached_dictionary_path, "rb") as cached_file:
                cached_dictionary = pickle.load(cached_file)
            log.info(
                "Loaded dictionary from cached file {}".format(cached_dictionary_path)
            )

            (
                self.dictionary,
                self.sparse_dictionary_index,
                self.dense_dictionary_index,
            ) = (
                cached_dictionary["dictionary"],
                cached_dictionary["sparse_dictionary_index"],
                cached_dictionary["dense_dictionary_index"],
            )
            self.dense_dictionary_index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.dense_dictionary_index
            )

        # else, load and embed
        else:
            # use provided dictionary
            if dictionary_name_or_path == "ctd-disease":
                self.dictionary = NEL_CTD_DISEASE_DICT().data
            elif dictionary_name_or_path == "ctd-chemical":
                self.dictionary = NEL_CTD_CHEMICAL_DICT().data
            elif dictionary_name_or_path == "ncbi-gene":
                self.dictionary = NEL_NCBI_GENE_DICT().data
            elif dictionary_name_or_path == "ncbi-taxonomy":
                self.dictionary = NEL_NCBI_TAXONOMY_DICT().data
            # use custom dictionary file
            else:
                self.dictionary = DictionaryDataset(
                    dictionary_path=dictionary_name_or_path
                ).data

            # embed dictionary
            dictionary_names = [row[0] for row in self.dictionary]

            # create dense and sparse embeddings
            if self.use_sparse_embeds:
                self.dict_dense_embeds = self.biosyn_embed_dense(
                    names=dictionary_names, show_progress=True
                )
                self.dict_sparse_embeds = self.embed_sparse(
                    names=dictionary_names, show_progress=True
                )
                faiss.normalize_L2(self.dict_sparse_embeds)

                # build sparse index scikit
                self.sparse_dictionary_index = NearestNeighbors(metric="cosine")
                self.sparse_dictionary_index.fit(self.dict_sparse_embeds)

            # create only dense embeddings
            else:
                self.dict_dense_embeds = self.sapbert_embed_dense(
                    names=dictionary_names, show_progress=True
                )
                self.dict_sparse_embeds = None
                self.sparse_dictionary_index = None

            # build dense index
            dimension = self.dict_dense_embeds.shape[1]
            # to use cosine similarity, we normalize the vectors and then use inner product
            faiss.normalize_L2(self.dict_dense_embeds)
            dense_index = faiss.IndexFlatIP(dimension)
            self.dense_dictionary_index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, dense_index
            )
            self.dense_dictionary_index.add(self.dict_dense_embeds)

            # create a cpu version of the index to cache (IO does not work with gpu index)
            cache_index_dense = faiss.index_gpu_to_cpu(self.dense_dictionary_index)

            # cache dictionary
            cached_dictionary = {
                "dictionary": self.dictionary,
                "sparse_dictionary_index": self.sparse_dictionary_index,
                "dense_dictionary_index": cache_index_dense,
            }
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)
            with open(cached_dictionary_path, "wb") as cache_file:
                pickle.dump(cached_dictionary, cache_file)
            print("Saving dictionary into cached file {}".format(cache_folder))

        # apply mean centering
        if mean_centering:
            self.tgt_space_mean_vec = self.dict_dense_embeds.mean(0)
            self.dict_dense_embeds -= self.tgt_space_mean_vec
        else:
            self.tgt_space_mean_vec = None


class BiomedicalEntityLinking:
    """
    Biomedical Entity Linker for HunFlair
    Can predict top k entities on sentences annotated with biomedical entity mentions
    """

    def __init__(
        self,
        model,
        ab3p,
    ) -> None:
        """
        Initalize class, called by classmethod load
        :param model: object containing the dense and sparse encoders
        :param ab3p_path: path to ab3p model
        """
        self.model = model
        self.text_preprocessor = TextPreprocess()
        self.ab3p = ab3p

    @classmethod
    def load(
        cls,
        model_name,
        dictionary_path: Union[str, Path] = None,
        use_sparse_and_dense_embeds: bool = True,
        max_length=25,
        use_ab3p: bool = True,
        ab3p_path: Path = None,
    ):
        """
        Load a model for biomedical named entity normalization on sentences annotated with
        biomedical entity mentions
        :param model_name: Name of pretrained model to use. Possible values for pretrained models are:
        chemical, disease, gene, sapbert-bc5cdr-disease, sapbert-ncbi-disease, sapbert-bc5cdr-chemical, biobert-bc5cdr-disease,
        biobert-ncbi-disease, biobert-bc5cdr-chemical, biosyn-biobert-bc2gn, biosyn-sapbert-bc2gn, sapbert
        :param dictionary_path: Name of one of the provided dictionaries listing all possible ids and their synonyms
        or a path to a dictionary file with each line in the format: id||name, with one line for each name of a concept.
        Possible values for dictionaries are: chemical, ctd-chemical, disease, bc5cdr-disease, gene, cnbci-gene,
        taxonomy and ncbi-taxonomy
        :param use_sparse_and_dense_embeds: If True, uses a combinations of sparse and dense embeddings for the dictionary and the mentions
        If False, uses only dense embeddings
        """
        model_name = model_name.lower()
        model_path = model_name

        # if a provided model is used,
        # modify model name to huggingface path

        # BioSyn
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
            use_sparse_and_dense_embeds = False
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

        # Initalize model
        model = BioEntityLinkingModel(
            use_sparse_embeds=use_sparse_and_dense_embeds, max_length=max_length
        )

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

        model.load_model(
            model_name_or_path=model_path,
            dictionary_name_or_path=dictionary_path,
            mean_centering=False,
        )

        # load ab3p model
        ab3p = Ab3P.load(ab3p_path) if use_ab3p else None

        return cls(model, ab3p)

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

        for sentence in sentences:
            for entity in sentence.get_labels(input_entity_annotation_layer):
                # preprocess mention
                if abbreviation_dict is not None:
                    parsed_tokens = []
                    for token in entity.span:
                        token = self.text_preprocessor.run(token.text)
                        if token in abbreviation_dict:
                            parsed_tokens.append(abbreviation_dict[token.lower()])
                        elif len(token) != 0:
                            parsed_tokens.append(token)
                    mention = " ".join(parsed_tokens)
                else:
                    mention = self.text_preprocessor.run(entity.span.text)

                # get predictions from dictionary
                # predictions = self.model.get_predictions(mention, topk)

                predictions = self.model.get_predictions_faiss(mention, topk)

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
                    # determine ontology:
                    if ":" in cui:
                        cui_parts = cui.split(":")
                        ontology = ":".join(cui_parts[0:-1])
                        cui = cui_parts[-1]
                    else:
                        ontology = None
                    sentence.add_complex_label(
                        typename=label_name,
                        label=EntityLinkingLabel(
                            span=entity.span,
                            id=cui,
                            concept_name=prediction[0],
                            additional_ids=additional_labels,
                            ontology=ontology,
                            score=prediction[2].astype(float),
                        ),
                    )
