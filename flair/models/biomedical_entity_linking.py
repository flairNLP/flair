import logging
import os
import pickle
import re
import stat
import string
import subprocess
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import flair
from flair.data import EntityLinkingCandidate, EntityLinkingLabel, Label, Sentence, Span
from flair.datasets import (
    CTD_CHEMICALS_DICTIONARY,
    CTD_DISEASES_DICTIONARY,
    NCBI_GENE_HUMAN_DICTIONARY,
    NCBI_TAXONOMY_DICTIONARY,
)
from flair.datasets.biomedical import (
    AbstractBiomedicalEntityLinkingDictionary,
    ParsedBiomedicalEntityLinkingDictionary,
)
from flair.embeddings import TransformerDocumentEmbeddings
from flair.file_utils import cached_path

FAISS_VERSION = "1.7.4"

try:
    import faiss
except ImportError as error:
    raise ImportError(
        f"You need to install to run the biomedical entity linking: `pip faiss faiss-cpu=={FAISS_VERSION}`"
    ) from error

logger = logging.getLogger("flair")


PRETRAINED_DENSE_MODELS = [
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
]

# Dense + sparse retrieval
PRETRAINED_HYBRID_MODELS = {
    "dmis-lab/biosyn-sapbert-bc5cdr-disease": "disease",
    "dmis-lab/biosyn-sapbert-ncbi-disease": "disease",
    "dmis-lab/biosyn-sapbert-bc5cdr-chemical": "chemical",
    "dmis-lab/biosyn-biobert-bc5cdr-disease": "disease",
    "dmis-lab/biosyn-biobert-ncbi-disease": "disease",
    "dmis-lab/biosyn-biobert-bc5cdr-chemical": "chemical",
    "dmis-lab/biosyn-biobert-bc2gn": "gene",
    "dmis-lab/biosyn-sapbert-bc2gn": "gene",
}

PRETRAINED_MODELS = list(PRETRAINED_HYBRID_MODELS) + PRETRAINED_DENSE_MODELS

# just in case we add: fuzzy search, Levenstein, ...
STRING_MATCHING_MODELS = ["exact-string-match"]

MODELS = PRETRAINED_MODELS + STRING_MATCHING_MODELS

ENTITY_TYPES = ["disease", "chemical", "gene", "species"]

ENTITY_TYPE_TO_LABELS = {
    "disease": "diseases",
    "gene": "genes",
    "species": "species",
    "chemical": "chemical",
}

ENTITY_TYPE_TO_HYBRID_MODEL = {
    "disease": "dmis-lab/biosyn-sapbert-bc5cdr-disease",
    "chemical": "dmis-lab/biosyn-sapbert-bc5cdr-chemical",
    "gene": "dmis-lab/biosyn-sapbert-bc2gn",
}

# for now we always fall back to SapBERT,
# but we should train our own models at some point
ENTITY_TYPE_TO_DENSE_MODEL = {
    entity_type: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" for entity_type in ENTITY_TYPES
}


ENTITY_TYPE_TO_DICTIONARY = {
    "gene": "ncbi-gene",
    "species": "ncbi-taxonomy",
    "disease": "ctd-diseases",
    "chemical": "ctd-chemicals",
}

ENTITY_TYPE_TO_ANNOTATION_LAYER = {
    "disease": "diseases",
    "gene": "genes",
    "chemical": "chemicals",
    "species": "species",
}

BIOMEDICAL_DICTIONARIES = {
    "ctd-diseases": CTD_DISEASES_DICTIONARY,
    "ctd-chemicals": CTD_CHEMICALS_DICTIONARY,
    "ncbi-gene": NCBI_GENE_HUMAN_DICTIONARY,
    "ncbi-taxonomy": NCBI_TAXONOMY_DICTIONARY,
}

MODEL_NAME_TO_DICTIONARY = {
    "dmis-lab/biosyn-sapbert-bc5cdr-disease": "ctd-disease",
    "dmis-lab/biosyn-sapbert-ncbi-disease": "ctd-disease",
    "dmis-lab/biosyn-sapbert-bc5cdr-chemical": "ctd-chemical",
    "dmis-lab/biosyn-biobert-bc5cdr-disease": "ctd-chemical",
    "dmis-lab/biosyn-biobert-ncbi-disease": "ctd-disease",
    "dmis-lab/biosyn-biobert-bc5cdr-chemical": "ctd-chemical",
    "dmis-lab/biosyn-biobert-bc2gn": "ncbi-gene",
    "dmis-lab/biosyn-sapbert-bc2gn": "ncbi-gene",
}


DEFAULT_SPARSE_WEIGHT = 0.5


class SimilarityMetric(Enum):
    """Similarity metrics"""

    INNER_PRODUCT = faiss.METRIC_INNER_PRODUCT
    # L2 = faiss.METRIC_L2
    COSINE = auto()


class AbstractEntityPreprocessor(ABC):
    """
    A pre-processor used to transform / clean both entity mentions and entity names
    This class provides the basic interface for such transformations
    and must provide a `name` attribute to uniquely identify the type of preprocessing applied.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        This is needed to correctly cache different multiple version of the dictionary
        """

    @abstractmethod
    def process_mention(self, entity_mention: Label, sentence: Sentence) -> str:
        """
        Processes the given entity mention and applies the transformation procedure to it.

        :param entity_mention: entity mention under investigation
        :param sentence: sentence in which the entity mentioned occurred
        :result: Cleaned / transformed string representation of the given entity mention
        """

    @abstractmethod
    def process_entity_name(self, entity_name: str) -> str:
        """
        Processes the given entity name (originating from a knowledge base / ontology) and
        applies the transformation procedure to it.

        :param entity_name: entity mention given as DataPoint
        :result: Cleaned / transformed string representation of the given entity mention
        """

    @abstractmethod
    def initialize(self, sentences: List[Sentence]):
        """
        Initializes the pre-processor for a batch of sentences, which is may be necessary for
        more sophisticated transformations.

        :param sentences: List of sentences that will be processed.
        """


class EntityPreprocessor(AbstractEntityPreprocessor):
    """
    Entity preprocessor adapted from:
        Sung et al. 2020, Biomedical Entity Representations with Synonym Marginalization
        https://github.com/dmis-lab/BioSyn/blob/master/src/biosyn/preprocesser.py#L5
    """

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        """
        Initializes the mention preprocessor.

        :param lowercase: Indicates whether to perform lowercasing or not (True by default)
        :param remove_punctuation: Indicates whether to perform removal punctuations symbols (True by default)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.rmv_puncts_regex = re.compile(r"[\s{}]+".format(re.escape(string.punctuation)))

    @property
    def name(self):

        return "biosyn"

    def initialize(self, sentences):
        pass

    def process_entity_name(self, entity_name: str) -> str:
        if self.lowercase:
            entity_name = entity_name.lower()

        if self.remove_punctuation:
            entity_name = self.rmv_puncts_regex.split(entity_name)
            entity_name = " ".join(entity_name).strip()

        return entity_name.strip()

    def process_mention(self, entity_mention: Label, sentence: Sentence) -> str:
        return self.process_entity_name(entity_mention.data_point.text)


class Ab3PEntityPreprocessor(AbstractEntityPreprocessor):
    """
    Entity preprocessor which uses Ab3P, an (biomedical)abbreviation definition detector:
        Abbreviation definition identification based on automatic precision estimates.
        Sohn S, Comeau DC, Kim W, Wilbur WJ. BMC Bioinformatics. 2008 Sep 25;9:402.
        PubMed ID: 18817555
        https://github.com/ncbi-nlp/Ab3P
    """

    def __init__(
        self, ab3p_path: Path, word_data_dir: Path, preprocessor: Optional[AbstractEntityPreprocessor] = None
    ) -> None:
        """
        Creates the mention pre-processor

        :param ab3p_path: Path to the folder containing the Ab3P implementation
        :param word_data_dir: Path to the word data directory
        :param preprocessor: Basic entity preprocessor
        """
        self.ab3p_path = ab3p_path
        self.word_data_dir = word_data_dir
        self.preprocessor = preprocessor
        self.abbreviation_dict = {}

    @property
    def name(self):

        return f"ab3p_{self.preprocessor.name}"

    def initialize(self, sentences: List[Sentence]) -> None:
        self.abbreviation_dict = self._build_abbreviation_dict(sentences)

    def process_mention(self, entity_mention: Label, sentence: Sentence) -> str:
        sentence_text = sentence.to_tokenized_string().strip()
        tokens = [token.text for token in entity_mention.data_point.tokens]

        parsed_tokens = []
        for token in tokens:
            if self.preprocessor is not None:
                token = self.preprocessor.process_entity_name(token)

            if sentence_text in self.abbreviation_dict:
                if token.lower() in self.abbreviation_dict[sentence_text]:
                    parsed_tokens.append(self.abbreviation_dict[sentence_text][token.lower()])
                    continue

            if len(token) != 0:
                parsed_tokens.append(token)

        return " ".join(parsed_tokens)

    def process_entity_name(self, entity_name: str) -> str:
        # Ab3P works on sentence-level and not on a single entity mention / name
        # - so we just apply the wrapped text pre-processing here (if configured)
        if self.preprocessor is not None:
            return self.preprocessor.process_entity_name(entity_name)

        return entity_name

    @classmethod
    def load(cls, ab3p_path: Path = None, preprocessor: Optional[AbstractEntityPreprocessor] = None):
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
        """Downloads the Ab3P tool and all necessary data files."""

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
        ab3p_path = cached_path("https://github.com/dmis-lab/BioSyn/raw/master/Ab3P/identify_abbr", data_dir)

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

        :param sentences: list of senternces
        :result abbreviation_dict: abbreviations and their resolution detected in each input sentence
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
            # https://pylint.pycqa.org/en/latest/user_guide/messages/warning/subprocess-run-check.html
            try:
                result = subprocess.run(
                    [self.ab3p_path, temp_file.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            except subprocess.CalledProcessError:
                logger.error(
                    """The abbreviation resolver Ab3P could not be run on your system. To ensure maximum accuracy, please
                install Ab3P yourself. See https://github.com/ncbi-nlp/Ab3P"""
                )
            else:
                line = result.stdout.decode("utf-8")
                if "Path file for type cshset does not exist!" in line:
                    logger.error(
                        "Error when using Ab3P for abbreviation resolution. A file named path_Ab3p needs to exist in your current directory containing the path to the WordData directory for Ab3P to work!"
                    )
                elif "Cannot open" in line:
                    logger.error(
                        "Error when using Ab3P for abbreviation resolution. Could not open the WordData directory for Ab3P!"
                    )
                elif "failed to open" in line:
                    logger.error(
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


class BiomedicalEntityLinkingDictionary:
    """
    Load dictionary: either pre-definded or from path
    Every line in the file must be formatted as follows: concept_id||concept_name
    If multiple concept ids are associated to a given name
    they must be separated by a `|`.
    """

    def __init__(
        self, reader: Union[AbstractBiomedicalEntityLinkingDictionary, ParsedBiomedicalEntityLinkingDictionary]
    ):
        self.reader = reader

    @classmethod
    def load(
        cls, dictionary_name_or_path: Union[Path, str], database_name: Optional[str] = None
    ) -> "EntityLinkingDictionary":
        """Load dictionary: either pre-definded or from path"""

        if isinstance(dictionary_name_or_path, str):
            if (
                dictionary_name_or_path not in ENTITY_TYPE_TO_DICTIONARY
                and dictionary_name_or_path not in BIOMEDICAL_DICTIONARIES
            ):
                raise ValueError(
                    f"Unkwnon dictionary `{dictionary_name_or_path}`!"
                    f" Available dictionaries are: {tuple(BIOMEDICAL_DICTIONARIES)}"
                    " If you want to pass a local path please use the `Path` class, i.e. `model_name_or_path=Path(my_path)`"
                )

            dictionary_name_or_path = ENTITY_TYPE_TO_DICTIONARY.get(dictionary_name_or_path, dictionary_name_or_path)

            reader = BIOMEDICAL_DICTIONARIES[dictionary_name_or_path]()

        else:
            # use custom dictionary file
            assert (
                database_name is not None
            ), "When providing a path to a custom dictionary you must specify the `database_name`!"
            reader = ParsedBiomedicalEntityLinkingDictionary(path=dictionary_name_or_path, database_name=database_name)

        return cls(reader=reader)

    @property
    def database_name(self) -> str:
        """Database name of the dictionary"""

        return self.reader.database_name

    def stream(self) -> Iterator[Tuple[str, str]]:
        """
        Stream entries from preprocessed dictionary
        """

        for entry in self.reader.stream():
            yield entry


class AbstractCandidateGenerator(ABC):
    """
    Base class for a candidate genertor
    """

    @abstractmethod
    def search(self, entity_mentions: List[str], top_k: int) -> List[List[EntityLinkingCandidate]]:
        """
        Returns the top-k entity / concept identifiers for the each entity mention.

        :param entity_mentions: Entity mentions
        :param top_k: Number of best-matching entities from the knowledge base to return
        :result: list of tuples in the form: (entity / concept name, concept ids, similarity score).
        """

    def build_candidate(self, candidate: Tuple[str, str, float]) -> EntityLinkingCandidate:
        """Get nice container with all info about entity linking candidate"""

        concept_name = candidate[0]
        concept_id = candidate[1]
        score = candidate[2]
        database_name = self.dictionary.database_name

        if "|" in concept_id:
            labels = concept_id.split("|")
            concept_id = labels[0]
            additional_labels = labels[1:]
        else:
            additional_labels = None

        return EntityLinkingCandidate(
            concept_id=concept_id,
            concept_name=concept_name,
            score=score,
            additional_ids=additional_labels,
            database_name=database_name,
        )


class ExactMatchCandidateGenerator(AbstractCandidateGenerator):
    """
    Candidate generator using exact string matching as search criterion
    """

    def __init__(self, dictionary: BiomedicalEntityLinkingDictionary):
        # Build index which maps concept / entity names to concept / entity ids
        self.name_to_id_index = dict(list(dictionary.stream()))

    @classmethod
    def load(cls, dictionary_name_or_path: str) -> "ExactStringMatchingRetrieverModel":
        """Compatibility function"""
        return cls(BiomedicalEntityLinkingDictionary.load(dictionary_name_or_path))

    def search(self, entity_mentions: List[str], top_k: int) -> List[List[EntityLinkingCandidate]]:
        """
        Returns the top-k entity / concept identifiers for the each entity mention.

        :param entity_mentions: Entity mentions
        :param top_k: Number of best-matching entities from the knowledge base to return
        :result: list of tuples in the form: (entity / concept name, concept ids, similarity score).
        """

        return [[self.build_candidate((em, self.name_to_id_index.get(em), 1.0))] for em in entity_mentions]


class BigramTfIDFVectorizer:
    """
    Wrapper for sklearn TfIDFVectorizer w/ fixed ngram range at the character level
    Implementation adapted from:
        Sung et al.: Biomedical Entity Representations with Synonym Marginalization, 2020
        https://github.com/dmis-lab/BioSyn/tree/master/src/biosyn/sparse_encoder.py#L8
    """

    def __init__(self):
        self.encoder = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))

    def fit(self, names: List[str]):
        """Learn vocabulary"""
        self.encoder.fit(names)
        return self

    def transform(self, names: List[str]) -> torch.Tensor:
        """Convert strings to sparse vectors"""
        vec = self.encoder.transform(names).toarray()
        vec = torch.FloatTensor(vec)
        return vec

    def __call__(self, mentions: List[str]) -> torch.Tensor:
        """Short for `transform`"""
        return self.transform(mentions)

    @classmethod
    def load(cls, path: Path) -> "BigramTfIDFVectorizer":
        """Instantiate from path"""
        newVectorizer = cls()

        with open(path, "rb") as fin:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                newVectorizer.encoder = pickle.load(fin)
            # logger.info("Sparse encoder loaded from %s", path)

        return newVectorizer


class BiEncoderCandidateGenerator(AbstractCandidateGenerator):
    """
    Candidate generator using both dense (transformer-based)
    and (optionally) sparse vector representations,
    to search candidates in a knowledge base / dictionary.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        dictionary_name_or_path: str,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        preprocessor: AbstractEntityPreprocessor = Ab3PEntityPreprocessor.load(preprocessor=EntityPreprocessor()),
        max_length: int = 25,
        batch_size: int = 1024,
        hybrid_search: bool = False,
        sparse_weight: Optional[float] = None,
        force_hybrid_search: bool = False,
        dictionary: Optional[BiomedicalEntityLinkingDictionary] = None,
    ):
        """
        Initializes the BiEncoderEntityRetrieverModel.

        :param model_name_or_path: Name of or path to the transformer model to be used.
        :param dictionary_name_or_path: Name of or path to the transformer model to be used.
        :param similarity_metric: which metric to use to compute similarity
        :param preprocessor: Preprocessing for entity mentions and names
        :param max_length: Maximum number of input tokens to transformer model
        :param batch_size: how many entity mentions/names to embed in one forward pass
        :param hybrid_search: Indicates whether to use sparse embeddings or not
        :param sparse_weight: default sparse weight
        :param force_hybrid_search: if pre-trained model is not hybrid (dense+sparse) fit a sparse encoder
        :param dictionary: optionally pass a dictionary
        """
        self.model_name_or_path = model_name_or_path
        self.dictionary_name_or_path = dictionary_name_or_path
        self.preprocessor = preprocessor
        self.similarity_metric = similarity_metric
        self.max_length = max_length
        self.batch_size = batch_size
        self.hybrid_search = hybrid_search
        self.sparse_weight = sparse_weight
        self.force_hybrid_search = force_hybrid_search

        # allow to pass custom dictionary
        if dictionary is not None:
            self.dictionary = dictionary
        else:
            self.dictionary = BiomedicalEntityLinkingDictionary.load(dictionary_name_or_path)

        self.dictionary_data = list(self.dictionary.stream())

        # Load encoders
        self.dense_encoder = TransformerDocumentEmbeddings(model=model_name_or_path, is_token_embedding=False)
        self.sparse_encoder: Optional[BigramTfIDFVectorizer] = None
        self.sparse_weight: Optional[float] = None
        if self.hybrid_search:
            self._set_sparse_weigth_and_encoder(
                model_name_or_path=model_name_or_path, dictionary_name_or_path=dictionary_name_or_path
            )

        self.embeddings = self._load_embeddings(
            model_name_or_path=model_name_or_path,
            dictionary_name_or_path=dictionary_name_or_path,
            batch_size=self.batch_size,
        )

        self.dense_index = self.build_dense_index(self.embeddings["dense"])

    @property
    def higher_is_better(self):
        """
        Determine if similarity is proportional to score.
        E.g. for L2 lower is better, while INNER_PRODUCT higher is better
        """

        return self.similarity_metric in [SimilarityMetric.COSINE, SimilarityMetric.INNER_PRODUCT]

    # separate method to allow more sophisticated logic in the future,
    # e.g. ANN with IndexIP, HNSW...
    def build_dense_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Initialize FAISS index"""

        dense_index = faiss.IndexFlatIP(embeddings.shape[1])
        dense_index.add(embeddings)

        return dense_index

    def _fit_sparse_encoder(self):
        """Fit sparse encoder to current dictionary"""

        logger.info(
            "Hybrid model has no pretrained sparse encoder. Fit to dictionary `%s` (sparse_weight=%s)",
            self.dictionary_name_or_path,
            self.sparse_weight,
        )
        self.sparse_encoder = BigramTfIDFVectorizer().fit([name for name, cui in self.dictionary])
        # sparse_encoder.save(Path(sparse_encoder_path))
        # torch.save(torch.FloatTensor(self.sparse_weight), sparse_weight_path)

    def _set_sparse_weigth_and_encoder(
        self, model_name_or_path: Union[str, Path], dictionary_name_or_path: Union[str, Path]
    ):

        sparse_encoder_path = os.path.join(model_name_or_path, "sparse_encoder.pk")
        sparse_weight_path = os.path.join(model_name_or_path, "sparse_weight.pt")

        if isinstance(model_name_or_path, str):

            # check file exists
            if model_name_or_path in PRETRAINED_HYBRID_MODELS:

                if not os.path.exists(sparse_encoder_path):

                    sparse_encoder_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename="sparse_encoder.pk",
                        cache_dir=flair.cache_root / "models" / model_name_or_path,
                    )

                    self.sparse_encoder = BigramTfIDFVectorizer.load(path=sparse_encoder_path)

                if not os.path.exists(sparse_weight_path):

                    sparse_weight_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename="sparse_weight.pt",
                        cache_dir=flair.cache_root / "models" / model_name_or_path,
                    )
                    self.sparse_weight = torch.load(sparse_weight_path, map_location="cpu").item()
            else:
                if self.force_hybrid_search:
                    self.sparse_weight = self.sparse_weight if self.sparse_weight is not None else DEFAULT_SPARSE_WEIGHT
                    self._fit_sparse_encoder()
                else:
                    raise ValueError(
                        f"A: Hybrid model has no pretrained sparse encoder. Please pass `force_hybrid_search=True` if you want to fit a sparse model to dictionary `{dictionary_name_or_path}`"
                    )
        else:
            if self.force_hybrid_search:
                self.sparse_weight = self.sparse_weight if self.sparse_weight is not None else DEFAULT_SPARSE_WEIGHT
                self._fit_sparse_encoder()
            else:
                raise ValueError(
                    f"Local hybrid model has no pretrained sparse encoder. Please pass `force_hybrid_search=True` if you want to fit a sparse model to dictionary `{dictionary_name_or_path}`"
                )

    def embed_sparse(self, inputs: np.ndarray) -> np.ndarray:
        """
        Create sparse embeddings from array of entity mentions/names.

        :param entity_names: An array of entity / concept names
        :returns sparse_embeds np.array: Numpy array containing the sparse embeddings
        """
        sparse_embeds = self.sparse_encoder(inputs)
        sparse_embeds = sparse_embeds.numpy()

        if self.similarity_metric == SimilarityMetric.COSINE:
            faiss.normalize_L2(sparse_embeds)

        return sparse_embeds

    def embed_dense(self, inputs: np.ndarray, batch_size: int = 1024, show_progress: bool = False) -> np.ndarray:
        """

        Create dense embeddings from array of entity mentions/names.

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
                    range(0, len(inputs), batch_size),
                    desc=f"Embedding `{self.dictionary.database_name}` dictionary:",
                )
            else:
                iterations = range(0, len(inputs), batch_size)

            for start in iterations:
                # Create batch
                end = min(start + batch_size, len(inputs))
                batch = [Sentence(name) for name in inputs[start:end]]

                # embed batch
                self.dense_encoder.embed(batch)

                dense_embeds += [name.embedding.cpu().detach().numpy() for name in batch]

                if flair.device.type == "cuda":
                    torch.cuda.empty_cache()

        dense_embeds = np.array(dense_embeds)

        return dense_embeds

    def _load_embeddings(self, model_name_or_path: str, dictionary_name_or_path: str, batch_size: int):
        """Compute and cache the embeddings for the given knowledge base / dictionary."""

        # Check for embedded dictionary in cache
        dictionary_name = os.path.splitext(os.path.basename(dictionary_name_or_path))[0]
        file_name = f"bio_nen_{model_name_or_path.split('/')[-1]}_{dictionary_name}"

        cache_folder = flair.cache_root / "datasets"

        pp_name = self.preprocessor.name if self.preprocessor is not None else "null"

        embeddings_cache_file = cache_folder / f"{file_name}-{pp_name}.pk"

        # If exists, load the cached dictionary indices
        if embeddings_cache_file.exists():

            with embeddings_cache_file.open("rb") as fp:
                logger.info("Load cached emebddings from:  %s", embeddings_cache_file)
                embeddings = pickle.load(fp)

        else:

            cache_folder.mkdir(parents=True, exist_ok=True)

            names = [self.preprocessor.process_entity_name(name) for name, cui in self.dictionary_data]

            # Compute dense embeddings (if necessary)
            dense_embeddings = self.embed_dense(inputs=names, batch_size=batch_size, show_progress=True)

            sparse_embeddings = self.embed_sparse(inputs=names) if self.hybrid_search else None

            # Store the pre-computed index on disk for later re-use
            embeddings = {
                "dense": dense_embeddings,
                "sparse": sparse_embeddings,
            }

            logger.info("Caching dictionary emebddings into %s", embeddings_cache_file)
            with embeddings_cache_file.open("wb") as fp:
                pickle.dump(embeddings, fp)

        if self.similarity_metric == SimilarityMetric.COSINE:
            faiss.normalize_L2(embeddings["dense"])

        return embeddings

    def search_sparse(
        self,
        entity_mentions: List[str],
        top_k: int = 1,
        normalise: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find candidates with sparse representations

        :param entity_mentions: list of entity mentions (queries)
        :param top_k: number of candidates to retrieve
        :param normalise: normalise scores
        """

        mention_embeddings = self.sparse_encoder(entity_mentions)

        if self.similarity_metric == SimilarityMetric.COSINE:
            score_matrix = cosine_similarity(mention_embeddings, self.embeddings["sparse"])
        else:
            score_matrix = np.matmul(mention_embeddings, self.embeddings["sparse"].T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min()) / (score_matrix.max() - score_matrix.min())

        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
            return arr[rows, cols]

        # Get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix, -top_k)[:, -top_k:]

        # Get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)
        topk_scores = indexing_2d(score_matrix, topk_idxs)

        return topk_scores, topk_idxs

    def search_dense(self, entity_mentions: List[str], top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find candidates with dense representations (FAISS)

        :param entity_mentions: list of entity mentions (queries)
        :param top_k: number of candidates to retrieve
        """

        # Compute dense embedding for the given entity mention
        mention_dense_embeds = self.embed_dense(inputs=np.array(entity_mentions), batch_size=self.batch_size)

        if self.similarity_metric == SimilarityMetric.COSINE:
            faiss.normalize_L2(mention_dense_embeds)

        # Get candidates from dense embeddings
        dists, ids = self.dense_index.search(mention_dense_embeds, top_k)

        return dists, ids

    def combine_dense_and_sparse_results(
        self,
        dense_ids: np.ndarray,
        dense_scores: np.ndarray,
        sparse_ids: np.ndarray,
        sparse_scores: np.ndarray,
        top_k: int = 1,
    ):
        """
        Expand dense resutls with sparse ones (that are not already in the dense)
        and re-weight the score as: dense_score + sparse_weight * sparse_scores
        """

        hybrid_ids = []
        hybrid_scores = []
        for i in range(dense_ids.shape[0]):

            mention_ids = dense_ids[i]
            mention_scores = dense_scores[i]

            mention_spare_ids = sparse_ids[i]
            mention_sparse_scores = sparse_scores[i]

            for sparse_id, sparse_score in zip(mention_spare_ids, mention_sparse_scores):
                if sparse_id not in mention_ids:
                    mention_ids = np.append(mention_ids, sparse_id)
                    mention_scores = np.append(mention_scores, self.sparse_weight * sparse_score)
                else:
                    index = np.where(mention_ids == sparse_id)[0][0]
                    mention_scores[index] += self.sparse_weight * sparse_score

            rerank_indices = np.argsort(-mention_scores if self.higher_is_better else mention_scores)
            mention_ids = mention_ids[rerank_indices][:top_k]
            mention_scores = mention_scores[rerank_indices][:top_k]
            hybrid_ids.append(mention_ids.tolist())
            hybrid_scores.append(mention_scores.tolist())

        return hybrid_scores, hybrid_ids

    def search(self, entity_mentions: List[str], top_k: int) -> List[List[EntityLinkingCandidate]]:
        """
        Returns the top-k entity / concept identifiers for the each entity mention.

        :param entity_mentions: Entity mentions
        :param top_k: Number of best-matching entities from the knowledge base to return
        :result: list of tuples in the form: (entity / concept name, concept ids, similarity score).
        """

        scores, ids = self.search_dense(entity_mentions=entity_mentions, top_k=top_k)

        if self.hybrid_search:

            sparse_scores, sparse_ids = self.search_sparse(entity_mentions=entity_mentions, top_k=top_k)

            scores, ids = self.combine_dense_and_sparse_results(
                dense_ids=ids,
                dense_scores=scores,
                sparse_scores=sparse_scores,
                sparse_ids=sparse_ids,
                top_k=top_k,
            )

        return [
            [
                self.build_candidate(tuple(self.dictionary_data[i]) + (score,))
                for i, score in zip(mention_ids, mention_scores)
            ]
            for mention_ids, mention_scores in zip(ids, scores)
        ]


class BiomedicalEntityLinker:
    """Entity linking model for the biomedical domain"""

    def __init__(
        self,
        candidate_generator: AbstractCandidateGenerator,
        preprocessor: AbstractEntityPreprocessor,
        entity_type: str,
    ):
        self.preprocessor = preprocessor
        self.candidate_generator = candidate_generator
        self.entity_type = entity_type
        self.annotation_layers = [ENTITY_TYPE_TO_ANNOTATION_LAYER.get(self.entity_type, "ner")]

    def extract_mentions(
        self,
        sentences: List[Sentence],
        annotation_layers: Optional[List[str]] = None,
    ) -> Tuple[List[int], List[Span], List[str]]:
        """Unpack all mentions in sentences for batch search."""

        source = []
        data_points = []
        mentions = []
        mention_annotation_layers = []

        # use default annotation layers only if are not provided
        annotation_layers = annotation_layers if annotation_layers is not None else self.annotation_layers

        for i, sentence in enumerate(sentences):
            for annotation_layer in annotation_layers:
                for entity in sentence.get_labels(annotation_layer):
                    source.append(i)
                    data_points.append(entity.data_point)
                    mentions.append(
                        self.preprocessor.process_mention(entity, sentence)
                        if self.preprocessor is not None
                        else entity.data_point.text,
                    )
                    mention_annotation_layers.append(annotation_layer)

        # assert len(mentions) > 0, f"There are no entity mentions of type `{self.entity_type}`"

        return source, data_points, mentions, mention_annotation_layers

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        annotation_layers: Optional[List[str]] = None,
        top_k: int = 1,
    ) -> None:
        """
        Predicts the best matching top-k entity / concept identifiers of all named entites annotated
        with tag input_entity_annotation_layer.

        :param sentences: One or more sentences to run the prediction on
        :param annotation_layers: list of annotation layers to extract entity mentions
        :param top_k: Number of best-matching entity / concept identifiers
        """
        # make sure sentences is a list of sentences
        if not isinstance(sentences, list):
            sentences = [sentences]

        if self.preprocessor is not None:
            self.preprocessor.initialize(sentences)

        # Build label name
        # label_name = input_entity_annotation_layer + "_nen" if (input_entity_annotation_layer is not None) else "nen"

        source, data_points, mentions, mentions_annotation_layers = self.extract_mentions(
            sentences=sentences, annotation_layers=annotation_layers
        )

        # Retrieve top-k concept / entity candidates
        candidates = self.candidate_generator.search(entity_mentions=mentions, top_k=top_k)

        # Add a label annotation for each candidate
        for i, data_point, mention_candidates, mentions_annotation_layer in zip(
            source, data_points, candidates, mentions_annotation_layers
        ):

            sentences[i].add_label(
                typename=mentions_annotation_layer,
                value_or_label=EntityLinkingLabel(data_point=data_point, candidates=mention_candidates),
            )

    @classmethod
    def load(
        cls,
        model_name_or_path: Union[str, Path],
        dictionary_name_or_path: Optional[Union[str, Path]] = None,
        hybrid_search: bool = True,
        max_length: int = 25,
        batch_size: int = 1024,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        preprocessor: AbstractEntityPreprocessor = Ab3PEntityPreprocessor.load(preprocessor=EntityPreprocessor()),
        force_hybrid_search: bool = False,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        entity_type: Optional[str] = None,
        dictionary: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Loads a model for biomedical named entity normalization.
        See __init__ method for detailed docstring on arguments
        """

        if dictionary_name_or_path is None or isinstance(dictionary_name_or_path, str):
            dictionary_name_or_path = cls.__get_dictionary_path(
                model_name_or_path=model_name_or_path, dictionary_name_or_path=dictionary_name_or_path
            )

        if isinstance(model_name_or_path, str):
            model_name_or_path, entity_type = cls.__get_model_path_and_entity_type(
                model_name_or_path=model_name_or_path,
                entity_type=entity_type,
                hybrid_search=hybrid_search,
                force_hybrid_search=force_hybrid_search,
            )
        else:
            assert entity_type is not None, "When using a custom model you must specify `entity_type`"
            assert entity_type in ENTITY_TYPES, f"Invalid entity type `{entity_type}! Must be one of: {ENTITY_TYPES}"

        if model_name_or_path == "exact-string-match":
            candidate_generator = ExactMatchCandidateGenerator.load(dictionary_name_or_path)
        else:
            candidate_generator = BiEncoderCandidateGenerator(
                model_name_or_path=model_name_or_path,
                dictionary_name_or_path=dictionary_name_or_path,
                hybrid_search=hybrid_search,
                similarity_metric=similarity_metric,
                max_length=max_length,
                batch_size=batch_size,
                sparse_weight=sparse_weight,
                preprocessor=preprocessor,
                force_hybrid_search=force_hybrid_search,
                dictionary=dictionary,
            )

        logger.info(
            "BiomedicalEntityLinker predicts: Dictionary `%s` (entity type: %s) with %s classes",
            dictionary_name_or_path,
            entity_type,
            len(candidate_generator.dictionary_data),
        )

        return cls(candidate_generator=candidate_generator, preprocessor=preprocessor, entity_type=entity_type)

    @staticmethod
    def __get_model_path_and_entity_type(
        model_name_or_path: Union[str, Path],
        entity_type: Optional[str] = None,
        hybrid_search: bool = False,
        force_hybrid_search: bool = False,
    ) -> Tuple[str, str]:
        """
        Try to figure out what model the user wants
        """

        if model_name_or_path not in MODELS and model_name_or_path not in ENTITY_TYPES:
            raise ValueError(
                f"Unknown model `{model_name_or_path}`!"
                f" Available entity types are: {ENTITY_TYPES}"
                " If you want to pass a local path please use the `Path` class, i.e. `model_name_or_path=Path(my_path)`"
            )

        if model_name_or_path == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext":
            assert entity_type is not None, f"For model {model_name_or_path} you must specify `entity_type`"

        if hybrid_search:
            # load model by entity_type
            if model_name_or_path in ENTITY_TYPES:
                # check if we have a hybrid pre-trained model
                if model_name_or_path in ENTITY_TYPE_TO_HYBRID_MODEL:
                    entity_type = model_name_or_path
                    model_name_or_path = ENTITY_TYPE_TO_HYBRID_MODEL[model_name_or_path]
                else:
                    # check if user really wants to use hybrid search anyway
                    if not force_hybrid_search:
                        raise ValueError(
                            f"""
                            Model for entity type `{model_name_or_path}` was not trained for hybrid search!
                            If you want to proceed anyway please pass `force_hybrid_search=True`:
                            we will fit a sparse encoder for you. The default value of `sparse_weight` is `{DEFAULT_SPARSE_WEIGHT}`.
                            """
                        )
                    model_name_or_path = ENTITY_TYPE_TO_DENSE_MODEL[model_name_or_path]
            else:
                if model_name_or_path not in PRETRAINED_HYBRID_MODELS and not force_hybrid_search:
                    raise ValueError(
                        f"""
                        Model `{model_name_or_path}` was not trained for hybrid search!
                        If you want to proceed anyway please pass `force_hybrid_search=True`:
                        we will fit a sparse encoder for you. The default value of `sparse_weight` is `{DEFAULT_SPARSE_WEIGHT}`.
                        """
                    )
                entity_type = PRETRAINED_HYBRID_MODELS[model_name_or_path]

        else:
            if model_name_or_path in ENTITY_TYPES:
                model_name_or_path = ENTITY_TYPE_TO_DENSE_MODEL[model_name_or_path]

        assert (
            entity_type is not None
        ), f"Impossible to determine entity type for model `{model_name_or_path}`: please specify via `entity_type`"

        return model_name_or_path, entity_type

    @staticmethod
    def __get_dictionary_path(
        model_name_or_path: str,
        dictionary_name_or_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Try to figure out what dictionary (depending on the model) the user wants
        """

        if model_name_or_path in STRING_MATCHING_MODELS and dictionary_name_or_path is None:
            raise ValueError(
                "When using a string-matching candidate generator you must specify `dictionary_name_or_path`!"
            )

        if dictionary_name_or_path is not None:
            if dictionary_name_or_path in ENTITY_TYPES:
                dictionary_name_or_path = ENTITY_TYPE_TO_DICTIONARY[dictionary_name_or_path]
        else:
            if model_name_or_path in MODEL_NAME_TO_DICTIONARY:
                dictionary_name_or_path = MODEL_NAME_TO_DICTIONARY[model_name_or_path]
            elif model_name_or_path in ENTITY_TYPE_TO_DICTIONARY:
                dictionary_name_or_path = ENTITY_TYPE_TO_DICTIONARY[model_name_or_path]
            else:
                raise ValueError(
                    f"When using a custom model you need to specify a dictionary. Available options are: {ENTITY_TYPES}. Or provide a path to a dictionary file."
                )

        return dictionary_name_or_path
