import inspect
import logging
import os
import re
import stat
import string
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
from tqdm import tqdm

import flair
from flair.class_utils import get_state_subclass_by_name
from flair.data import Label, Sentence, Span
from flair.datasets import (
    CTD_CHEMICALS_DICTIONARY,
    CTD_DISEASES_DICTIONARY,
    NCBI_GENE_HUMAN_DICTIONARY,
    NCBI_TAXONOMY_DICTIONARY,
    HunerEntityLinkingDictionary,
    KnowledgebaseLinkingDictionary,
)
from flair.datasets.knowledgebase import InMemoryEntityLinkingDictionary
from flair.embeddings import DocumentEmbeddings, DocumentTFIDFEmbeddings, TransformerDocumentEmbeddings
from flair.file_utils import cached_path, load_torch_state

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

ENTITY_TYPES = ["diseases", "chemical", "genes", "species"]

ENTITY_TYPE_TO_HYBRID_MODEL = {
    "diseases": "dmis-lab/biosyn-sapbert-bc5cdr-disease",
    "chemical": "dmis-lab/biosyn-sapbert-bc5cdr-chemical",
    "genes": "dmis-lab/biosyn-sapbert-bc2gn",
}

# for now we always fall back to SapBERT,
# but we should train our own models at some point
ENTITY_TYPE_TO_DENSE_MODEL = {
    entity_type: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" for entity_type in ENTITY_TYPES
}

ENTITY_TYPE_TO_DICTIONARY = {
    "genes": "ncbi-gene",
    "species": "ncbi-taxonomy",
    "diseases": "ctd-diseases",
    "chemical": "ctd-chemicals",
}

BIOMEDICAL_DICTIONARIES: Dict[str, Type] = {
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


def load_dictionary(
    dictionary_name_or_path: Union[Path, str], dataset_name: Optional[str] = None
) -> KnowledgebaseLinkingDictionary:
    """Load dictionary: either pre-defined or from path."""
    if isinstance(dictionary_name_or_path, str) and (
        dictionary_name_or_path in ENTITY_TYPE_TO_DICTIONARY or dictionary_name_or_path in BIOMEDICAL_DICTIONARIES
    ):
        dictionary_name_or_path = ENTITY_TYPE_TO_DICTIONARY.get(dictionary_name_or_path, dictionary_name_or_path)

        return BIOMEDICAL_DICTIONARIES[str(dictionary_name_or_path)]()

    if dataset_name is None:
        raise ValueError("When loading a custom dictionary, you need to specify a dataset_name!")
    return HunerEntityLinkingDictionary(path=dictionary_name_or_path, dataset_name=dataset_name)


class SimilarityMetric(Enum):
    """Similarity metrics."""

    INNER_PRODUCT = auto()
    COSINE = auto()


class EntityPreprocessor(ABC):
    """A pre-processor used to transform / clean both entity mentions and entity names."""

    def initialize(self, sentences: List[Sentence]) -> None:
        """Initializes the pre-processor for a batch of sentences.

        This may be necessary for more sophisticated transformations.

        Args:
            sentences: List of sentences that will be processed.
        """

    def process_mention(self, entity_mention: Label, sentence: Sentence) -> str:
        """Processes the given entity mention and applies the transformation procedure to it.

        Usually just forwards the entity_mention to :meth:`EntityPreprocessor.process_entity_name`, but can be implemented
        to preprocess mentions on a sentence level instead.

        Args:
            entity_mention: entity mention under investigation
            sentence: sentence in which the entity mentioned occurred

        Returns:
            Cleaned / transformed string representation of the given entity mention
        """
        return self.process_entity_name(entity_mention.data_point.text)

    @abstractmethod
    def process_entity_name(self, entity_name: str) -> str:
        """Processes the given entity name and applies the transformation procedure to it.

        Args:
            entity_name: the text of the entity mention

        Returns:
            Cleaned / transformed string representation of the given entity mention
        """

    @classmethod
    def _from_state(cls, state_dict: Dict[str, Any]) -> "EntityPreprocessor":
        if inspect.isabstract(cls):
            cls_name = state_dict.pop("__cls__", None)
            return get_state_subclass_by_name(cls, cls_name)._from_state(state_dict)
        else:
            return cls(**state_dict)

    def _get_state(self) -> Dict[str, Any]:
        return {"__cls__": self.__class__.__name__}


class BioSynEntityPreprocessor(EntityPreprocessor):
    """Entity preprocessor using Synonym Marginalization.

    Adapted from:
        Sung et al. 2020, Biomedical Entity Representations with Synonym Marginalization
        https://github.com/dmis-lab/BioSyn/blob/master/src/biosyn/preprocesser.py#L5.

    The preprocessor provides basic string transformation options including lower-casing,
    removal of punctuations symbols, etc.
    """

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        """Initializes the mention preprocessor.

        Args:
            lowercase: Indicates whether to perform lowercasing or not
            remove_punctuation: Indicates whether to perform removal punctuations symbols
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.rmv_puncts_regex = re.compile(rf"[\s{re.escape(string.punctuation)}]+")

    def process_entity_name(self, entity_name: str) -> str:
        if self.lowercase:
            entity_name = entity_name.lower()

        if self.remove_punctuation:
            name_parts = self.rmv_puncts_regex.split(entity_name)
            entity_name = " ".join(name_parts).strip()

        return entity_name.strip()

    def _get_state(self) -> Dict[str, Any]:
        return {
            **super()._get_state(),
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
        }


class Ab3PEntityPreprocessor(EntityPreprocessor):
    """Entity preprocessor which uses Ab3P, an (biomedical) abbreviation definition detector.

    Abbreviation definition identification based on automatic precision estimates.
    Sohn S, Comeau DC, Kim W, Wilbur WJ. BMC Bioinformatics. 2008 Sep 25;9:402.
    PubMed ID: 18817555
    https://github.com/ncbi-nlp/Ab3P.
    """

    def __init__(self, ab3p_path: Path, word_data_dir: Path, preprocessor: Optional[EntityPreprocessor] = None) -> None:
        """Creates the mention pre-processor.

        Args:
            ab3p_path: Path to the folder containing the Ab3P implementation
            word_data_dir: Path to the word data directory
            preprocessor: Basic entity preprocessor
        """
        self.ab3p_path = ab3p_path
        self.word_data_dir = word_data_dir
        self.preprocessor = preprocessor
        self.abbreviation_dict: Dict[str, Dict[str, str]] = {}

    def initialize(self, sentences: List[Sentence]) -> None:
        self.abbreviation_dict = self._build_abbreviation_dict(sentences)

    def process_mention(self, entity_mention: Label, sentence: Sentence) -> str:
        sentence_text = sentence.to_tokenized_string().strip()
        tokens = [token.text for token in cast(Span, entity_mention.data_point).tokens]

        parsed_tokens = []
        for token in tokens:
            if self.preprocessor is not None:
                token = self.preprocessor.process_entity_name(token)

            if sentence_text in self.abbreviation_dict and token.lower() in self.abbreviation_dict[sentence_text]:
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
    def load_biosyn(cls, preprocessor: Optional[EntityPreprocessor] = None):
        data_dir = flair.cache_root / "ab3p_biosyn"
        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        word_data_dir = data_dir / "word_data"
        if not word_data_dir.exists():
            word_data_dir.mkdir()

        ab3p_path = cls._download_biosyn_ab3p(data_dir, word_data_dir)

        return cls(ab3p_path, word_data_dir, preprocessor)

    @classmethod
    def _download_biosyn_ab3p(cls, data_dir: Path, word_data_dir: Path) -> Path:
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

        # Make Ab3P executable
        ab3p_path.chmod(ab3p_path.stat().st_mode | stat.S_IXUSR)
        return ab3p_path

    def _build_abbreviation_dict(self, sentences: List[flair.data.Sentence]) -> Dict[str, Dict[str, str]]:
        """Processes the given sentences with the Ab3P tool.

        The function returns a (nested) dictionary containing the abbreviations found for each sentence, e.g.:

        {
            "Respiratory syncytial viruses ( RSV ) are a subgroup of the paramyxoviruses.":
                {"RSV": "Respiratory syncytial viruses"},
            "Rous sarcoma virus ( RSV ) is a retrovirus.":
                {"RSV": "Rous sarcoma virus"}
        }

        Args:
            sentences: list of sentences

        Returns:
            abbreviation_dict: abbreviations and their resolution detected in each input sentence
        """
        abbreviation_dict: Dict = defaultdict(dict)

        # Create a temp file which holds the sentences we want to process with Ab3P
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as temp_file:
            for sentence in sentences:
                temp_file.write(sentence.to_tokenized_string() + "\n")
            temp_file.flush()

            # Temporarily create path file in the current working directory for Ab3P
            with open(os.path.join(os.getcwd(), "path_Ab3P"), "w") as path_file:
                path_file.write(str(self.word_data_dir) + "/\n")

            # Run Ab3P with the temp file containing the dataset
            # https://pylint.pycqa.org/en/latest/user_guide/messages/warning/subprocess-run-check.html
            try:
                result = subprocess.run(
                    [self.ab3p_path, temp_file.name],
                    capture_output=True,
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
                elif "Cannot open" in line or "failed to open" in line:
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

    def _get_state(self) -> Dict[str, Any]:
        return {
            **super()._get_state(),
            "ab3p_path": str(self.ab3p_path),
            "word_data_dir": str(self.word_data_dir),
            "preprocessor": None if self.preprocessor is None else self.preprocessor._get_state(),
        }

    @classmethod
    def _from_state(cls, state_dict: Dict[str, Any]) -> "EntityPreprocessor":
        return cls(
            ab3p_path=Path(state_dict["ad3p_path"]),
            word_data_dir=Path(state_dict["word_data_dir"]),
            preprocessor=None
            if state_dict["preprocessor"] is None
            else EntityPreprocessor._from_state(state_dict["preprocessor"]),
        )


class CandidateSearchIndex(ABC):
    """Base class for a candidate generator.

    Given a mention of an entity, find matching entries from the dictionary.
    """

    @abstractmethod
    def index(
        self, dictionary: KnowledgebaseLinkingDictionary, preprocessor: Optional[EntityPreprocessor] = None
    ) -> None:
        """Index a dictionary to prepare for search.

        Args:
            dictionary: The data to index.
            preprocessor: If given, preprocess the concept name and synonyms before indexing.
        """

    @abstractmethod
    def search(self, entity_mentions: List[str], top_k: int) -> List[List[Tuple[str, float]]]:
        """Returns the top-k entity / concept identifiers for each entity mention.

        Args:
            entity_mentions: Entity mentions
            top_k: Number of best-matching entities from the knowledge base to return

        Returns:
            List containing a list of entity linking candidates per entity mention from the input
        """

    @classmethod
    def _from_state(cls, state_dict: Dict[str, Any]) -> "CandidateSearchIndex":
        if inspect.isabstract(cls):
            cls_name = state_dict.pop("__cls__", None)
            return get_state_subclass_by_name(cls, cls_name)._from_state(state_dict)
        else:
            return cls(**state_dict)

    def _get_state(self) -> Dict[str, Any]:
        return {"__cls__": self.__class__.__name__}


class ExactMatchCandidateSearchIndex(CandidateSearchIndex):
    """Candidate generator using exact string matching as search criterion."""

    def __init__(self):
        """Candidate generator using exact string matching as search criterion.

        Args:
            name_to_id_index: internal state, should only be set when loading an initialized index.
        """
        self.name_to_id_index: Dict[str, str] = {}

    def index(
        self, dictionary: KnowledgebaseLinkingDictionary, preprocessor: Optional[EntityPreprocessor] = None
    ) -> None:
        def p(text: str) -> str:
            return preprocessor.process_entity_name(text) if preprocessor is not None else text

        for candidate in dictionary.candidates:
            self.name_to_id_index[p(candidate.concept_name)] = candidate.concept_id
            for synonym in candidate.synonyms:
                self.name_to_id_index[p(synonym)] = candidate.concept_id

    def search(self, entity_mentions: List[str], top_k: int) -> List[List[Tuple[str, float]]]:
        results: List[List[Tuple[str, float]]] = []
        for mention in entity_mentions:
            dict_entry = self.name_to_id_index.get(mention)
            if dict_entry is None:
                results.append([])
                continue
            results.append([(dict_entry, 1.0)])

        return results

    @classmethod
    def _from_state(cls, state_dict: Dict[str, Any]) -> "CandidateSearchIndex":
        index = cls()
        index.name_to_id_index = state_dict["name_to_id_index"]
        return index

    def _get_state(self) -> Dict[str, Any]:
        return {
            **super()._get_state(),
            "name_to_id_index": self.name_to_id_index,
        }


class SemanticCandidateSearchIndex(CandidateSearchIndex):
    """Candidate generator using both dense and (optionally) sparse vector representations, to search candidates."""

    def __init__(
        self,
        embeddings: List[DocumentEmbeddings],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        weights: Optional[List[float]] = None,
        batch_size: int = 128,
        show_progress: bool = True,
    ):
        """Initializes the EncoderCandidateSearchIndex.

        Args:
            embeddings: A list of embeddings used for search.
            weights: Weight the embedding's importance.
            similarity_metric: The metric used to define similarity.
            batch_size: The batch size used for indexing embeddings.
            show_progress: show the progress while indexing.
        """
        if weights is None:
            weights = [1.0 for _ in embeddings]
        if len(weights) != len(embeddings):
            raise ValueError("Weights have to be of the same length as embeddings")

        self.embeddings = embeddings
        self.weights = weights
        self.similarity_metric = similarity_metric
        self.show_progress = show_progress
        self.batch_size = batch_size

        self.ids: List[str] = []
        self._precomputed_embeddings: np.ndarray = np.array([])

    @classmethod
    def bi_encoder(
        cls,
        model_name_or_path: str,
        hybrid_search: bool,
        similarity_metric: SimilarityMetric,
        batch_size: int = 128,
        show_progress: bool = True,
        sparse_weight: float = 0.5,
        preprocessor: Optional[EntityPreprocessor] = None,
        dictionary: Optional[KnowledgebaseLinkingDictionary] = None,
    ) -> "SemanticCandidateSearchIndex":
        embeddings: List[DocumentEmbeddings] = [TransformerDocumentEmbeddings(model_name_or_path)]
        weights = [1.0]
        if hybrid_search:
            if dictionary is None:
                raise ValueError("Require dictionary to be set on hybrid search.")

            texts = []

            for candidate in dictionary.candidates:
                texts.append(candidate.concept_name)
                texts.extend(candidate.synonyms)

            if preprocessor is not None:
                texts = [preprocessor.process_entity_name(t) for t in texts]

            embeddings.append(
                DocumentTFIDFEmbeddings(
                    [Sentence(t) for t in texts],
                    analyzer="char",
                    ngram_range=(1, 2),
                )
            )
            weights = [1.0, sparse_weight]
        return cls(
            embeddings,
            similarity_metric=similarity_metric,
            weights=weights,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def index(
        self, dictionary: KnowledgebaseLinkingDictionary, preprocessor: Optional[EntityPreprocessor] = None
    ) -> None:
        def p(text: str) -> str:
            return preprocessor.process_entity_name(text) if preprocessor is not None else text

        texts: List[str] = []
        self.ids = []
        for candidate in dictionary.candidates:
            texts.append(p(candidate.concept_name))
            self.ids.append(candidate.concept_id)
            for synonym in candidate.synonyms:
                texts.append(p(synonym))
                self.ids.append(candidate.concept_id)

        precomputed_embeddings = []

        with torch.no_grad():
            if self.show_progress:
                iterations = tqdm(
                    range(0, len(texts), self.batch_size),
                    desc=f"Embedding `{dictionary.database_name}`",
                )
            else:
                iterations = range(0, len(texts), self.batch_size)

            for start in iterations:
                end = min(start + self.batch_size, len(texts))
                batch = [Sentence(name) for name in texts[start:end]]

                for embedding in self.embeddings:
                    embedding.embed(batch)

                for sent in batch:
                    embs = []
                    for embedding, weight in zip(self.embeddings, self.weights):
                        emb = sent.get_embedding(embedding.get_names())
                        if self.similarity_metric == SimilarityMetric.COSINE:
                            emb = emb / torch.norm(emb)
                        embs.append(emb * weight)

                    precomputed_embeddings.append(torch.cat(embs, dim=0).cpu().numpy())
                    sent.clear_embeddings()
                if flair.device.type == "cuda":
                    torch.cuda.empty_cache()

        self._precomputed_embeddings = np.stack(precomputed_embeddings, axis=0)

    def emb_search(self, entity_mentions: List[str]) -> np.ndarray:
        embeddings = []

        with torch.no_grad():
            for start in range(0, len(entity_mentions), self.batch_size):
                end = min(start + self.batch_size, len(entity_mentions))
                batch = [Sentence(name) for name in entity_mentions[start:end]]

                for embedding in self.embeddings:
                    embedding.embed(batch)

                for sent in batch:
                    embs = []
                    for embedding in self.embeddings:
                        emb = sent.get_embedding(embedding.get_names())
                        if self.similarity_metric == SimilarityMetric.COSINE:
                            emb = emb / torch.norm(emb)
                        embs.append(emb)

                    embeddings.append(torch.cat(embs, dim=0).cpu().numpy())
                    sent.clear_embeddings()
                if flair.device.type == "cuda":
                    torch.cuda.empty_cache()

        return np.stack(embeddings, axis=0)

    def search(self, entity_mentions: List[str], top_k: int) -> List[List[Tuple[str, float]]]:
        """Returns the top-k entity / concept identifiers for each entity mention.

        Args:
            entity_mentions: Entity mentions
            top_k: Number of best-matching entities from the knowledge base to return

        Returns:
            List containing a list of entity linking candidates per entity mention from the input
        """
        mention_embs = self.emb_search(entity_mentions)
        all_scores = mention_embs @ self._precomputed_embeddings.T
        selected_indices = np.argsort(all_scores, axis=1)[:, :top_k]
        scores = np.take_along_axis(all_scores, selected_indices, axis=1)

        results = []
        for i in range(selected_indices.shape[0]):
            results.append(
                [(self.ids[selected_indices[i, j]], float(scores[i, j])) for j in range(selected_indices.shape[1])]
            )

        return results

    @classmethod
    def _from_state(cls, state_dict: Dict[str, Any]) -> "CandidateSearchIndex":
        index = cls(
            embeddings=[DocumentEmbeddings.load_embedding(emb) for emb in state_dict["embeddings"]],
            similarity_metric=SimilarityMetric(state_dict["similarity_metric"]),
            weights=state_dict["weights"],
            batch_size=state_dict["batch_size"],
            show_progress=state_dict["show_progress"],
        )
        index.ids = state_dict["ids"]
        index._precomputed_embeddings = state_dict["precomputed_embeddings"]
        return index

    def _get_state(self) -> Dict[str, Any]:
        return {
            **super()._get_state(),
            "embeddings": [emb.save_embeddings() for emb in self.embeddings],
            "similarity_metric": self.similarity_metric.value,
            "weights": self.weights,
            "batch_size": self.batch_size,
            "show_progress": self.show_progress,
            "ids": self.ids,
            "precomputed_embeddings": self._precomputed_embeddings,
        }


class EntityMentionLinker:
    """Entity linking model for the biomedical domain."""

    def __init__(
        self,
        candidate_generator: CandidateSearchIndex,
        preprocessor: EntityPreprocessor,
        entity_label_type: str,
        label_type: str,
        dictionary: KnowledgebaseLinkingDictionary,
    ):
        self.preprocessor = preprocessor
        self.candidate_generator = candidate_generator
        self.entity_label_type = entity_label_type
        self._label_type = label_type
        self._dictionary = dictionary

    @property
    def label_type(self):
        return self._label_type

    @property
    def dictionary(self) -> KnowledgebaseLinkingDictionary:
        return self._dictionary

    def extract_mentions(
        self,
        sentences: List[Sentence],
    ) -> Tuple[List[Span], List[str]]:
        """Unpack all mentions in sentences for batch search."""
        data_points = []
        mentions = []

        for sentence in sentences:
            for entity in sentence.get_labels(self.entity_label_type):
                data_points.append(entity.data_point)
                mentions.append(
                    self.preprocessor.process_mention(entity, sentence)
                    if self.preprocessor is not None
                    else entity.data_point.text,
                )

        return data_points, mentions

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        top_k: int = 1,
    ) -> None:
        """Predicts the best matching top-k entity / concept identifiers of all named entities annotated with tag input_entity_annotation_layer.

        Args:
            sentences: One or more sentences to run the prediction on
            top_k: Number of best-matching entity / concept identifiers
        """
        # make sure sentences is a list of sentences
        if not isinstance(sentences, list):
            sentences = [sentences]

        if self.preprocessor is not None:
            self.preprocessor.initialize(sentences)

        data_points, mentions = self.extract_mentions(sentences=sentences)

        # no mentions: nothing to do here
        if len(mentions) > 0:
            # Retrieve top-k concept / entity candidates
            candidates = self.candidate_generator.search(entity_mentions=mentions, top_k=top_k)

            # Add a label annotation for each candidate
            for data_point, mention_candidates in zip(data_points, candidates):
                for candidate_id, confidence in mention_candidates:
                    data_point.add_label(self.label_type, candidate_id, confidence)

    @staticmethod
    def _fetch_model(model_name: str) -> str:
        if Path(model_name).exists():
            return model_name

        raise NotImplementedError()

    def save(self, model_path: Union[str, Path]) -> None:
        pass

    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "EntityMentionLinker":
        if isinstance(model_path, str):
            model_path = cls._fetch_model(model_path)

        if isinstance(model_path, dict):
            state = model_path
        else:
            state = load_torch_state(str(model_path))

        candidate_generator = CandidateSearchIndex._from_state(state["candidate_search_index"])
        preprocessor = EntityPreprocessor._from_state("entity_preprocessor")
        entity_label_type = state["entity_label_type"]
        label_type = state["label_type"]
        dictionary = InMemoryEntityLinkingDictionary.from_state(state["dictionary"])

        return cls(candidate_generator, preprocessor, entity_label_type, label_type, dictionary)

    @classmethod
    def build(
        cls,
        model_name_or_path: Union[str, Path],
        label_type: str,
        dictionary_name_or_path: Optional[Union[str, Path]] = None,
        hybrid_search: bool = True,
        batch_size: int = 128,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        preprocessor: EntityPreprocessor = BioSynEntityPreprocessor(),
        force_hybrid_search: bool = False,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        entity_type: Optional[str] = None,
        dictionary: Optional[KnowledgebaseLinkingDictionary] = None,
        dataset_name: Optional[str] = None,
    ) -> "EntityMentionLinker":
        """Loads a model for biomedical named entity normalization.

        See __init__ method for detailed docstring on arguments.
        """
        if not isinstance(model_name_or_path, str):
            raise AssertionError(f"String matching model name has to be an string (and not {type(model_name_or_path)}")
        model_name_or_path = cast(str, model_name_or_path)

        if dictionary is None:
            if dictionary_name_or_path is None or isinstance(dictionary_name_or_path, str):
                dictionary_name_or_path = cls.__get_dictionary_path(
                    model_name_or_path=model_name_or_path, dictionary_name_or_path=dictionary_name_or_path
                )
            dictionary = load_dictionary(dictionary_name_or_path, dataset_name=dataset_name)

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
            candidate_generator: CandidateSearchIndex = ExactMatchCandidateSearchIndex()
        else:
            candidate_generator = SemanticCandidateSearchIndex.bi_encoder(
                model_name_or_path=str(model_name_or_path),
                hybrid_search=hybrid_search,
                similarity_metric=similarity_metric,
                batch_size=batch_size,
                sparse_weight=sparse_weight,
                preprocessor=preprocessor,
                dictionary=dictionary,
            )

        candidate_generator.index(dictionary, preprocessor)

        logger.info(
            "BiomedicalEntityLinker predicts: Dictionary `%s` (entity type: %s)", dictionary_name_or_path, entity_type
        )

        return cls(
            candidate_generator=candidate_generator,
            preprocessor=preprocessor,
            entity_label_type=entity_type,
            label_type=label_type,
            dictionary=dictionary,
        )

    @staticmethod
    def __get_model_path_and_entity_type(
        model_name_or_path: Union[str, Path],
        entity_type: Optional[str] = None,
        hybrid_search: bool = False,
        force_hybrid_search: bool = False,
    ) -> Tuple[Union[str, Path], str]:
        """Try to figure out what model the user wants."""
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
            if isinstance(model_name_or_path, str) and model_name_or_path in ENTITY_TYPES:
                model_name_or_path = cast(str, model_name_or_path)

                # check if we have a hybrid pre-trained model
                if model_name_or_path in ENTITY_TYPE_TO_HYBRID_MODEL:
                    entity_type = model_name_or_path
                    model_name_or_path = ENTITY_TYPE_TO_HYBRID_MODEL[model_name_or_path]
                else:
                    # check if user really wants to use hybrid search anyway
                    if not force_hybrid_search:
                        logger.warning(
                            "BiEncoderCandidateGenerator: model for entity type `%s` was not trained for"
                            " hybrid search: no sparse search will be performed."
                            " If you want to use sparse search please pass `force_hybrid_search=True`:"
                            " we will fit a sparse encoder for you. The default value of `sparse_weight` is `%s`.",
                            model_name_or_path,
                            DEFAULT_SPARSE_WEIGHT,
                        )
                    model_name_or_path = ENTITY_TYPE_TO_DENSE_MODEL[model_name_or_path]
            else:
                if model_name_or_path not in PRETRAINED_HYBRID_MODELS and not force_hybrid_search:
                    logger.warning(
                        "BiEncoderCandidateGenerator: model `%s` was not trained for hybrid search: no sparse"
                        " search will be performed."
                        " If you want to use sparse search please pass `force_hybrid_search=True`:"
                        " we will fit a sparse encoder for you. The default value of `sparse_weight` is `%s`.",
                        model_name_or_path,
                        DEFAULT_SPARSE_WEIGHT,
                    )

                model_name_or_path = cast(str, model_name_or_path)
                entity_type = PRETRAINED_HYBRID_MODELS[model_name_or_path]

        else:
            if isinstance(model_name_or_path, str) and model_name_or_path in ENTITY_TYPES:
                model_name_or_path = cast(str, model_name_or_path)
                model_name_or_path = ENTITY_TYPE_TO_DENSE_MODEL[model_name_or_path]

        assert (
            entity_type is not None
        ), f"Impossible to determine entity type for model `{model_name_or_path}`: please specify via `entity_type`"

        return model_name_or_path, entity_type

    @staticmethod
    def __get_dictionary_path(
        model_name_or_path: str,
        dictionary_name_or_path: Optional[Union[str, Path]] = None,
    ) -> Union[str, Path]:
        """Try to figure out what dictionary (depending on the model) the user wants."""
        if model_name_or_path in STRING_MATCHING_MODELS and dictionary_name_or_path is None:
            raise ValueError(
                "When using a string-matching candidate generator you must specify `dictionary_name_or_path`!"
            )

        if dictionary_name_or_path is not None and isinstance(dictionary_name_or_path, str):
            dictionary_name_or_path = cast(str, dictionary_name_or_path)

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
