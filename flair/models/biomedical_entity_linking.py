import flair
import faiss
import logging
import numpy as np
import os
import pickle
import re
import subprocess
import stat
import string
import tempfile
import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from flair.data import Sentence, EntityLinkingLabel, DataPoint, Label
from flair.datasets import (
    NEL_CTD_CHEMICAL_DICT,
    NEL_CTD_DISEASE_DICT,
    NEL_NCBI_HUMAN_GENE_DICT,
    NEL_NCBI_TAXONOMY_DICT,
)
from flair.embeddings import TransformerDocumentEmbeddings
from flair.file_utils import cached_path
from huggingface_hub import hf_hub_url, cached_download
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Dict, Iterable

log = logging.getLogger("flair")


class MentionPreprocessor:
    """
    A mention preprocessor is used to transform / clean an entity mention (recognized by
    an entity recognition model in the original text). This can include removing certain characters
    (e.g. punctuation) or converting characters (e.g. HTML-encoded characters) as well as
    (more sophisticated) domain-specific procedures.

    This class provides the basic interface for such transformations and should be extended by
    subclasses that implement the concrete transformations.
    """
    def process(self, entity_mention: Union[DataPoint, str], sentence: Sentence) -> str:
        """
        Processes the given entity mention and applies the transformation procedure to it.

        :param entity_mention: entity mention either given as DataPoint or str
        :param sentence: sentence in which the entity mentioned occurred
        :result: Cleaned / transformed string representation of the given entity mention
        """
        raise NotImplementedError()

    def initialize(self, sentences: List[Sentence]) -> None:
        """
        Initializes the pre-processor for a batch of sentences, which is may be necessary for
        more sophisticated transformations.

        :param sentences: List of sentences that will be processed.
        """
        # Do nothing by default
        pass


class BasicMentionPreprocessor(MentionPreprocessor):
    """
    Basic implementation of MentionPreprocessor, which supports lowercasing, typo correction
     and removing of punctuation characters.

    Implementation is adapted from:
        Sung et al. 2020, Biomedical Entity Representations with Synonym Marginalization
        https://github.com/dmis-lab/BioSyn/blob/master/src/biosyn/preprocesser.py#L5
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        punctuation_symbols: str = string.punctuation
    ) -> None:
        """
        Initializes the mention preprocessor.

        :param lowercase: Indicates whether to perform lowercasing or not (True by default)
        :param remove_punctuation: Indicates whether to perform removal punctuations symbols (True by default)
        :param punctuation_symbols: String containing all punctuation symbols that should be removed
            (default is given by string.punctuation)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.rmv_puncts_regex = re.compile(
            r"[\s{}]+".format(re.escape(punctuation_symbols))
        )

    def process(self, entity_mention: Union[DataPoint, str], sentence: Sentence) -> str:
        mention_text = entity_mention if isinstance(entity_mention, str) else entity_mention.text

        if self.lowercase:
            mention_text = mention_text.lower()

        if self.remove_punctuation:
            mention_text = self.rmv_puncts_regex.split(mention_text)
            mention_text = " ".join(mention_text).strip()

        mention_text = mention_text.strip()

        return mention_text


class Ab3PMentionPreprocessor(MentionPreprocessor):
    """
    Implementation of MentionPreprocessor which utilizes Ab3P, an (biomedical)abbreviation definition detector,
    given in:
        https://github.com/ncbi-nlp/Ab3P

    Ab3P applies a set of rules reflecting simple patterns such as Alpha Beta (AB) as well as more involved cases.
    The algorithm is described in detail in the following paper:

        Abbreviation definition identification based on automatic precision estimates.
        Sohn S, Comeau DC, Kim W, Wilbur WJ. BMC Bioinformatics. 2008 Sep 25;9:402.
        PubMed ID: 18817555
    """

    def __init__(
            self,
            ab3p_path: Path,
            word_data_dir: Path,
            mention_preprocessor: Optional[MentionPreprocessor] = None
    ) -> None:
        """
        Creates the mention pre-processor

        :param ab3p_path: Path to the folder containing the Ab3P implementation
        :param word_data_dir: Path to the word data directory
        :param mention_preprocessor: Mention text preprocessor that is used before trying to link
            the mention text to an abbreviation.

        """
        self.ab3p_path = ab3p_path
        self.word_data_dir = word_data_dir
        self.mention_preprocessor = mention_preprocessor

    def initialize(self, sentences: List[Sentence]) -> None:
        self.abbreviation_dict = self._build_abbreviation_dict(sentences)

    def process(self, entity_mention: Union[Label, str], sentence: Sentence) -> str:
        sentence_text = sentence.to_tokenized_string().strip()

        tokens = (
            [token.text for token in entity_mention.data_point.tokens]
            if isinstance(entity_mention, Label)
            else [entity_mention] # FIXME: Maybe split mention on whitespaces here??
        )

        parsed_tokens = []
        for token in tokens:
            if self.mention_preprocessor is not None:
                token = self.mention_preprocessor.process(token, sentence)

            if sentence_text in self.abbreviation_dict:
                if token.lower() in self.abbreviation_dict[sentence_text]:
                    parsed_tokens.append(self.abbreviation_dict[sentence_text][token.lower()])
                    continue

            if len(token) != 0:
                parsed_tokens.append(token)

        return " ".join(parsed_tokens)

    @classmethod
    def load(
            cls,
            ab3p_path: Path = None,
            preprocessor: Optional[MentionPreprocessor] = None
    ):
        data_dir = flair.cache_root / "ab3p"
        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        word_data_dir = data_dir / "word_data"
        if not word_data_dir.exists():
            word_data_dir.mkdir()

        if ab3p_path is None:
            ab3p_path = cls.download_ab3p(data_dir, word_data_dir)

        return cls(ab3p_path, word_data_dir, preprocessor)

    @classmethod
    def download_ab3p(cls, data_dir: Path, word_data_dir: Path) -> Path:
        """
            Downloads the Ab3P tool and all necessary data files.
        """

        # Download word data for Ab3P if not already downloaded
        ab3p_url = "https://raw.githubusercontent.com/dmis-lab/BioSyn/master/Ab3P/WordData/"

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
            cached_path(ab3p_url + file, word_data_dir)

        # Download Ab3P executable
        ab3p_path = cached_path(
            "https://github.com/dmis-lab/BioSyn/raw/master/Ab3P/identify_abbr", data_dir
        )

        ab3p_path.chmod(ab3p_path.stat().st_mode | stat.S_IXUSR)
        return ab3p_path

    def _build_abbreviation_dict(self, sentences: List[flair.data.Sentence]) -> Dict[str, Dict[str, str]]:
        """
        Processes the given sentences with the Ab3P tool. The function returns a (nested) dictionary
        containing the abbreviations found for each sentence, e.g.:

        {
            "Respiratory syncytial viruses ( RSV ) are a subgroup of the paramyxoviruses.":
                {"RSV": "Respiratory syncytial viruses"},
            "Rous sarcoma virus ( RSV ) is a retrovirus.":
                {"RSV": "Rous sarcoma virus"}
        }
        """
        abbreviation_dict = defaultdict(dict)

        # Create a temp file which holds the sentences we want to process with ab3p
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as temp_file:
            for sentence in sentences:
                temp_file.write(sentence.to_tokenized_string() + "\n")
            temp_file.flush()

            # Temporarily create path file in the current working directory for Ab3P
            with open(os.path.join(os.getcwd(), "path_Ab3P"), "w") as path_file:
                path_file.write(str(self.word_data_dir) + "/\n")

            # Run ab3p with the temp file containing the dataset
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
                cur_sentence = None
                for line in lines:
                    if len(line.split("|")) == 3:
                        if cur_sentence is None:
                            continue

                        sf, lf, _ = line.split("|")
                        sf = sf.strip().lower()
                        lf = lf.strip().lower()
                        abbreviation_dict[cur_sentence][sf] = lf

                    elif len(line.strip()) > 0:
                        cur_sentence = line
                    else:
                        cur_sentence = None

            finally:
                # remove the path file
                os.remove(os.path.join(os.getcwd(), "path_Ab3P"))

        return abbreviation_dict


class BigramTfIDFVectorizer:
    """
    Helper class to encode a list of entity mentions or dictionary entries into a sparse tensor.

    Implementation adapted from:
        Sung et al.: Biomedical Entity Representations with Synonym Marginalization, 2020
        https://github.com/dmis-lab/BioSyn/tree/master/src/biosyn/sparse_encoder.py#L8
    """

    def __init__(self) -> None:
        self.encoder = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))

    def transform(self, mentions: List[str]) -> torch.Tensor:
        vec = self.encoder.transform(mentions).toarray()
        vec = torch.FloatTensor(vec)
        return vec

    def __call__(self, mentions: List[str]) -> torch.Tensor:
        return self.transform(mentions)

    def save_encoder(self, path: Path) -> None:
        with path.open("wb") as fout:
            pickle.dump(self.encoder, fout)
            log.info("Sparse encoder saved in {}".format(path))

    @classmethod
    def load(cls, path: Path) -> "BigramTfIDFVectorizer":
        newVectorizer = cls()
        with open(path, "rb") as fin:
            newVectorizer.encoder = pickle.load(fin)
            log.info("Sparse encoder loaded from {}".format(path))

        return newVectorizer


class DictionaryDataset:
    """
    A class used to load dictionary data from a custom dictionary file.
    Every line in the file must be formatted as follows:
    concept_unique_id||concept_name
    with one line per concept name. Multiple synonyms for the same concept should
    be in separate lines with the same concept_unique_id.

    Slightly modifed from Sung et al. 2020
    Biomedical Entity Representations with Synonym Marginalization
    https://github.com/dmis-lab/BioSyn/blob/master/src/biosyn/data_loader.py#L89
    """

    def __init__(
            self,
            dictionary_path: Union[Path, str],
            load_into_memory: bool = True
    ) -> None:
        """
        :param dictionary_path str: Path to the dictionary file
        :param load_into_memory bool: Indicates whether the dictionary entries should be loaded in
            memory or not (Default True)
        """
        log.info("Loading dictionary from {}".format(dictionary_path))
        if load_into_memory:
            self.data = self.load_data(dictionary_path)
        else:
            self.data = self.get_data(dictionary_path)

    def load_data(self, dictionary_path: Union[Path, str]) -> np.ndarray:
        data = []
        with open(dictionary_path, mode="r", encoding="utf-8") as file:
            lines = file.readlines()
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
    def get_data(self, dictionary_path: Union[Path, str]) -> Iterable[Tuple]:
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

    @classmethod
    def load(cls, dictionary_name_or_path: Union[Path, str]):
        if isinstance(dictionary_name_or_path, str):
            # use provided dictionary
            if dictionary_name_or_path == "ctd-disease":
                return NEL_CTD_DISEASE_DICT()
            elif dictionary_name_or_path == "ctd-chemical":
                return NEL_CTD_CHEMICAL_DICT()
            elif dictionary_name_or_path == "ncbi-gene":
                return NEL_NCBI_HUMAN_GENE_DICT()
            elif dictionary_name_or_path == "ncbi-taxonomy":
                return NEL_NCBI_TAXONOMY_DICT()
        else:
            # use custom dictionary file
            return DictionaryDataset(dictionary_path=dictionary_name_or_path)


class EntityRetrieverModel(ABC):
    """
    An entity retriever model is used to find the top-k entities / concepts of a knowledge base /
    dictionary for a given entity mention in text.
    """

    @abstractmethod
    def get_top_k(
            self,
            entity_mention: str,
            top_k: int
    ) -> List[Tuple[str, str, float]]:
        """
        Returns the top-k entity / concept identifiers for the given entity mention.

        :param entity_mention: Entity mention text under investigation
        :param top_k: Number of (best-matching) entities from the knowledge base to return
        :result: List of tuples highlighting the top-k entities. Each tuple has the following
            structure (entity / concept name, concept ids, score).
        """
        raise NotImplementedError()


class ExactStringMatchingRetrieverModel(EntityRetrieverModel):
    """
    Implementation of an entity retriever model which uses exact string matching to
    find the entity / concept identifier for a given entity mention.
    """
    def __init__(self, dictionary: DictionaryDataset):
        # Build index which maps concept / entity names to concept / entity ids
        self.name_to_id_index = {name: cui for name, cui in dictionary.data}

    @classmethod
    def load_model(
        cls,
        dictionary_name_or_path: str,
    ):
        # Load dictionary
        return cls(DictionaryDataset.load(dictionary_name_or_path))

    def get_top_k(
            self,
            entity_mention: str,
            top_k: int
    ) -> List[Tuple[str, str, float]]:
        """
        Returns the top-k entity / concept identifiers for the given entity mention. Note that
        the model either return the entity with an identical name in the knowledge base / dictionary
        or none.

        :param entity_mention: Entity mention under investigation
        :param top_k: Number of (best-matching) entities from the knowledge base to return
        :result: List of tuples highlighting the top-k entities. Each tuple has the following
            structure (entity / concept name, concept ids, score).
        """
        if entity_mention in self.name_to_id_index:
            return [(entity_mention, self.name_to_id_index[entity_mention], 1.0)]
        else:
            return []


class BiEncoderEntityRetrieverModel(EntityRetrieverModel):
    """
    Implementation of EntityRetrieverModel which uses dense (transformer-based) embeddings and (optionally)
    sparse character-based representations, for normalizing an entity mention to specific identifiers
    in a knowledge base / dictionary.

    To this end, the model embeds the entity mention text and all concept names from the knowledge
    base and outputs the k best-matching concepts based on embedding similarity.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        dictionary_name_or_path: str,
        use_sparse_embeddings: bool,
        use_cosine: bool,
        max_length: int,
        batch_size: int,
        index_use_cuda: bool,
        top_k_extra_dense: int = 10,
        top_k_extra_sparse: int = 10
    ) -> None:
        """
            Initializes the BiEncoderEntityRetrieverModel.

            :param model_name_or_path: Name of or path to the transformer model to be used.
            :param dictionary_name_or_path: Name of or path to the transformer model to be used.
            :param use_sparse_embeddings: Indicates whether to use sparse embeddings or not
            :param use_cosine: Indicates whether to use cosine similarity (instead of inner product)
            :param max_length: Maximal number of tokens used for embedding an entity mention / concept name
            :param batch_size: Batch size used during embedding of the dictionary and top-k prediction
            :param index_use_cuda: Indicates whether to use CUDA while indexing the dictionary / knowledge base
            :param top_k_extra_sparse: Number of extra entities (resp. their sparse embeddings) which should be
                retrieved while combining sparse and dense scores
            :param top_k_extra_dense: Number of extra entities (resp. their dense embeddings) which should be
                retrieved while combining sparse and dense scores
        """
        self.use_sparse_embeds = use_sparse_embeddings
        self.use_cosine = use_cosine
        self.max_length = max_length
        self.batch_size = batch_size
        self.top_k_extra_dense = top_k_extra_dense
        self.top_k_extra_sparse = top_k_extra_sparse
        self.index_use_cuda = index_use_cuda and flair.device.type == "cuda"

        # Load dense encoder
        self.dense_encoder = TransformerDocumentEmbeddings(
            model=model_name_or_path,
            is_token_embedding=False
        )

        # Load sparse encoder
        if self.use_sparse_embeds:
            #FIXME: What happens if sparse encoder isn't pre-trained???
            self._load_sparse_encoder(model_name_or_path)
            self._load_sparse_weight(model_name_or_path)

        self._embed_dictionary(
            model_name_or_path=model_name_or_path,
            dictionary_name_or_path=dictionary_name_or_path,
            batch_size=batch_size,
        )

    def _load_sparse_encoder(
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

    def _load_sparse_weight(self, model_name_or_path: Union[str, Path]) -> float:
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

        self.sparse_weight = torch.load(sparse_weight_path, map_location="cpu").item()

        return self.sparse_weight

    def _embed_sparse(self, entity_names: np.ndarray) -> np.ndarray:
        """
        Embeds the given numpy array of entity names, either originating from the knowledge base
        or recognized in a text, into sparse representations.

        :param entity_names: An array of entity / concept names
        :returns sparse_embeds np.array: Numpy array containing the sparse embeddings
        """
        sparse_embeds = self.sparse_encoder(entity_names)
        sparse_embeds = sparse_embeds.numpy()

        if self.use_cosine:
            faiss.normalize_L2(sparse_embeds)

        return sparse_embeds

    def _embed_dense(
        self,
        names: np.ndarray,
        batch_size: int = 2048,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embeds the given numpy array of entity / concept names, either originating from the
        knowledge base or recognized in a text, into dense representations using a
        TransformerDocumentEmbedding model.

        :param names: Numpy array of entity / concept names
        :param batch_size: Batch size used while embedding the name
        :param show_progress: bool to toggle progress bar
        :return: Numpy array containing the dense embeddings of the names
        """
        self.dense_encoder.eval()  # prevent dropout

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
                # Create batch
                end = min(start + batch_size, len(names))
                batch = [Sentence(name) for name in names[start:end]]

                # embed batch
                self.dense_encoder.embed(batch)

                dense_embeds += [
                    name.embedding.cpu().detach().numpy() for name in batch
                ]

                if flair.device.type == "cuda":
                    torch.cuda.empty_cache()

        dense_embeds = np.array(dense_embeds)
        if self.use_cosine:
            faiss.normalize_L2(dense_embeds)

        return dense_embeds

    def _embed_dictionary(
            self,
            model_name_or_path: str,
            dictionary_name_or_path: str,
            batch_size: int
    ):
        """
            Computes the embeddings for the given knowledge base / dictionary.
        """
        # Load dictionary
        self.dictionary = DictionaryDataset.load(dictionary_name_or_path).data

        # Check for embedded dictionary in cache
        dictionary_name = os.path.splitext(os.path.basename(dictionary_name_or_path))[0]
        file_name = f"bio_nen_{model_name_or_path.split('/')[-1]}_{dictionary_name}"

        cache_folder = flair.cache_root / "datasets"
        emb_dictionary_cache_file = cache_folder / f"{file_name}.pk"

        # If exists, load the cached dictionary indices
        if emb_dictionary_cache_file.exists():
            self._load_cached_dense_emb_dictionary(emb_dictionary_cache_file)

        else:
            # get names from dictionary and remove punctuation
            # FIXME: Why doing this here????
            punctuation_regex = re.compile(r"[\s{}]+".format(re.escape(string.punctuation)))

            dictionary_names = []
            for row in self.dictionary:
                name = punctuation_regex.split(row[0])
                name = " ".join(name).strip().lower()
                dictionary_names.append(name)
            dictionary_names = np.array(dictionary_names)

            # Compute dense embeddings (if necessary)
            self.dict_dense_embeddings = self._embed_dense(
                names=dictionary_names,
                batch_size=batch_size,
                show_progress=True
            )

            # To use cosine similarity, we normalize the vectors and then use inner product
            if self.use_cosine:
                faiss.normalize_L2(self.dict_dense_embeddings)

            # Compute sparse embeddings (if necessary)
            if self.use_sparse_embeds:
                self.dict_sparse_embeddings = self._embed_sparse(entity_names=dictionary_names)
            else:
                self.dict_sparse_embeddings = None

            # Build dense embedding index using faiss
            dimension = self.dict_dense_embeddings.shape[1]
            self.dense_dictionary_index = faiss.IndexFlatIP(dimension)
            self.dense_dictionary_index.add(self.dict_dense_embeddings)

            # Store the pre-computed index on disk for later re-use
            cached_dictionary = {
                "dictionary": self.dictionary,
                "sparse_dictionary_embeds": self.dict_sparse_embeddings,
                "dense_dictionary_index": self.dense_dictionary_index,
            }

            if not cache_folder.exists():
                cache_folder.mkdir(parents=True)

            log.info(f"Saving dictionary into cached file {cache_folder}")
            with emb_dictionary_cache_file.open("wb") as cache_file:
                pickle.dump(cached_dictionary, cache_file)

            # If we use CUDA - move index to GPU
            if self.index_use_cuda:
                self.dense_dictionary_index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.dense_dictionary_index
                )

    def _load_cached_dense_emb_dictionary(self, cached_dictionary_path: Path):
        """
            Loads pre-computed dense dictionary embedding from disk.
        """
        with cached_dictionary_path.open("rb") as cached_file:
            log.info("Loaded dictionary from cached file {}".format(cached_dictionary_path))
            cached_dictionary = pickle.load(cached_file)

            self.dictionary, self.dict_sparse_embeddings, self.dense_dictionary_index = (
                cached_dictionary["dictionary"],
                cached_dictionary["sparse_dictionary_embeds"],
                cached_dictionary["dense_dictionary_index"]
            )

            if self.index_use_cuda:
                self.dense_dictionary_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.dense_dictionary_index)

    def retrieve_sparse_topk_candidates(
            self,
            mention_embeddings: np.ndarray,
            dict_concept_embeddings: np.ndarray,
            top_k: int,
            normalise: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns top-k indexes (in descending order) for the given entity mentions resp. mention
        embeddings.

        :param score_matrix: 2d numpy array of scores
        :param top_k: number of candidates to retrieve
        :return res: d numpy array of ids [# of query , # of dict]
        :return scores: numpy array of top scores
        """
        if self.use_cosine:
            score_matrix = cosine_similarity(mention_embeddings, dict_concept_embeddings)
        else:
            score_matrix = np.matmul(mention_embeddings, dict_concept_embeddings.T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min()) / (
                score_matrix.max() - score_matrix.min()
            )

        def indexing_2d(arr, cols):
            rows = np.repeat(
                np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1
            )
            return arr[rows, cols]

        # Get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix, -top_k)[:, -top_k:]

        # Get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)
        topk_scores = indexing_2d(score_matrix, topk_idxs)

        return (topk_idxs, topk_scores)

    def get_top_k(
            self,
            entity_mention: str,
            top_k: int
    ) -> List[Tuple[str, str, float]]:
        """
        Returns the top-k entities for a given entity mention.

        :param entity_mention: Entity mention text under investigation
        :param top_k: Number of (best-matching) entities from the knowledge base to return
        :result: List of tuples highlighting the top-k entities. Each tuple has the following
            structure (entity / concept name, concept ids, score).
        """

        # Compute dense embedding for the given entity mention
        mention_dense_embeds = self._embed_dense(
            names=np.array([entity_mention]),
            batch_size=self.batch_size
        )

        # Search for more than top-k candidates if combining them with sparse scores
        top_k_dense = top_k if not self.use_sparse_embeds else top_k + self.top_k_extra_dense

        # Get candidates from dense embeddings
        dense_scores, dense_ids = self.dense_dictionary_index.search(
            x=mention_dense_embeds,
            k=top_k_dense
        )

        # If using sparse embeds: calculate hybrid scores with dense and sparse embeds
        if self.use_sparse_embeds:
            # Get sparse embeddings for the entity mention
            mention_sparse_embeds = self._embed_sparse(entity_names=np.array([entity_mention]))

            # Get candidates from sparse embeddings
            sparse_ids, sparse_distances = self.retrieve_sparse_topk_candidates(
                mention_embeddings=mention_sparse_embeds,
                dict_concept_embeddings=self.dict_sparse_embeddings,
                top_k=top_k + self.top_k_extra_sparse
            )

            # Combine dense and sparse scores
            sparse_weight = self.sparse_weight
            hybrid_ids = []
            hybrid_scores = []

            # For every embedded mention
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
                ids = ids[sorted_indizes][:top_k]
                distances = distances[sorted_indizes][:top_k]
                hybrid_ids.append(ids.tolist())
                hybrid_scores.append(distances.tolist())

        else:
            # Use only dense embedding results
            hybrid_ids = dense_ids
            hybrid_scores = dense_scores

        return [
            tuple(self.dictionary[entity_index].reshape(1, -1)[0]) + (score[0],)
            for entity_index, score in zip(hybrid_ids, hybrid_scores)
        ]


class BiomedicalEntityLinker:
    """
    Entity linking model which expects text/sentences with annotated entity mentions and predicts
    entity / concept to these mentions according to a knowledge base / dictionary.
    """
    def __init__(
            self,
            retriever_model: EntityRetrieverModel,
            mention_preprocessor: MentionPreprocessor
    ):
        self.preprocessor = mention_preprocessor
        self.retriever_model = retriever_model

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        input_entity_annotation_layer: str = None,
        top_k: int = 1,
    ) -> None:
        """
        Predicts the best matching top-k entity / concept identifiers of all named entites annotated
        with tag input_entity_annotation_layer.

        :param sentences: One or more sentences to run the prediction on
        :param input_entity_annotation_layer: Entity type to run the prediction on
        :param top_k: Number of best-matching entity / concept identifiers which should be predicted
            per entity mention
        """
        # make sure sentences is a list of sentences
        if not isinstance(sentences, list):
            sentences = [sentences]

        if self.preprocessor is not None:
            self.preprocessor.initialize(sentences)

        # Build label name
        label_name = (
            input_entity_annotation_layer + "_nen"
            if (input_entity_annotation_layer is not None)
            else "nen"
        )

        # For every sentence ..
        for sentence in sentences:
            # ... process every mentioned entity
            for entity in sentence.get_labels(input_entity_annotation_layer):
                # Pre-process entity mention (if necessary)
                mention_text = (
                    self.preprocessor.process(entity, sentence)
                    if self.preprocessor is not None
                    else entity.data_point.text
                )

                # Retrieve top-k concept / entity candidates
                predictions = self.retriever_model.get_top_k(mention_text, top_k)

                # Add a label annotation for each candidate
                for prediction in predictions:
                    # if concept identifier is made up of multiple ids, separated by '|'
                    # separate it into cui and additional_labels
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
                            concept_id=cui,
                            concept_name=prediction[0],
                            additional_ids=additional_labels,
                            database=database,
                            score=prediction[2],
                        ),
                    )

    @classmethod
    def load(
        cls,
        model_name_or_path: Union[str, Path],
        dictionary_name_or_path: Union[str, Path] = None,
        use_sparse_embeddings: bool = True,
        max_length: int = 25,
        batch_size: int = 1024,
        index_use_cuda: bool = False,
        use_cosine: bool = True,
        preprocessor: MentionPreprocessor = Ab3PMentionPreprocessor.load(preprocessor=BasicMentionPreprocessor())
    ):
        """
        Loads a model for biomedical named entity normalization.

        :param model_name_or_path: Name of or path to a pretrained model to use. Possible values for pretrained
            models are:
            chemical, disease, gene, sapbert-bc5cdr-dissaease, sapbert-ncbi-disease, sapbert-bc5cdr-chemical,
            biobert-bc5cdr-disease,biobert-ncbi-disease, biobert-bc5cdr-chemical, biosyn-biobert-bc2gn,
            biosyn-sapbert-bc2gn, sapbert, exact-string-match
        :param dictionary_name_or_path: Name of or path to a dictionary listing all possible entity / concept
            identifiers and their concept names / synonyms.  Pre-defined dictionaries are:
                chemical, ctd-chemical, disease, bc5cdr-disease, gene, cnbci-gene, taxonomy and ncbi-taxonomy
        :param use_sparse_embeddings: Indicates whether to use sparse embeddings for inference. If True,
            uses a combinations of sparse and dense embeddings. If False, uses only dense embeddings
        :param: max_length: Maximal number of tokens for an entity mention or concept name
        :param batch_size: Batch size for the dense encoder
        :param index_use_cuda: If True, uses GPU for the dense encoding
        :param use_cosine: If True, uses cosine similarity for the dense encoder. If False, inner product is used.
        :param preprocessor: Implementation of MentionPreprocessor to use for pre-processing the entity
            mention text and dictionary entries
        """
        dictionary_path = dictionary_name_or_path
        if dictionary_name_or_path is None or isinstance(dictionary_name_or_path, str):
            dictionary_path = cls.__get_dictionary_path(dictionary_name_or_path, model_name_or_path)

        retriever_model = None
        if isinstance(model_name_or_path, str):
            if model_name_or_path == "exact-string-match":
                retriever_model = ExactStringMatchingRetrieverModel.load_model(dictionary_path)
            else:
                model_path = cls.__get_model_path(model_name_or_path, use_sparse_embeddings)
                retriever_model = BiEncoderEntityRetrieverModel(
                    model_name_or_path=model_path,
                    dictionary_name_or_path=dictionary_path,
                    use_sparse_embeddings=use_sparse_embeddings,
                    use_cosine=use_cosine,
                    max_length=max_length,
                    batch_size=batch_size,
                    index_use_cuda=index_use_cuda,
                )

        return cls(
            retriever_model=retriever_model,
            mention_preprocessor=preprocessor
        )

    @staticmethod
    def __get_model_path(
        model_name: str,
        use_sparse_and_dense_embeds: bool
    ) -> str:

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
    def __get_dictionary_path(
            dictionary_path: str,
            model_name: str
    ) -> str:
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
