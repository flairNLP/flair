import importlib.util
import inspect
import logging
import re
import string
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, cast

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset
from tqdm import tqdm

import flair
from flair.class_utils import get_state_subclass_by_name
from flair.data import DT, Dictionary, Label, Sentence, _iter_dataset
from flair.datasets import (
    CTD_CHEMICALS_DICTIONARY,
    CTD_DISEASES_DICTIONARY,
    NCBI_GENE_HUMAN_DICTIONARY,
    NCBI_TAXONOMY_DICTIONARY,
    EntityLinkingDictionary,
    HunerEntityLinkingDictionary,
)
from flair.datasets.entity_linking import InMemoryEntityLinkingDictionary
from flair.embeddings import DocumentEmbeddings, DocumentTFIDFEmbeddings, TransformerDocumentEmbeddings
from flair.embeddings.base import load_embeddings
from flair.file_utils import hf_download
from flair.training_utils import Result

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

# fetched from original repo to avoid download
HYBRID_MODELS_SPARSE_WEIGHT = {
    "dmis-lab/biosyn-sapbert-bc5cdr-disease": 0.09762775897979736,
    "dmis-lab/biosyn-sapbert-ncbi-disease": 0.40971508622169495,
    "dmis-lab/biosyn-sapbert-bc5cdr-chemical": 0.07534809410572052,
    "dmis-lab/biosyn-biobert-bc5cdr-disease": 1.5729279518127441,
    "dmis-lab/biosyn-biobert-ncbi-disease": 1.7646825313568115,
    "dmis-lab/biosyn-biobert-bc2gn": 1.5786927938461304,
    "dmis-lab/biosyn-sapbert-bc2gn": 0.0288906991481781,
}

PRETRAINED_MODELS = list(PRETRAINED_HYBRID_MODELS) + PRETRAINED_DENSE_MODELS


# just in case we add: fuzzy search, Levenstein, ...
STRING_MATCHING_MODELS = ["exact-string-match"]

MODELS = PRETRAINED_MODELS + STRING_MATCHING_MODELS

ENTITY_TYPES = ["disease", "chemical", "gene", "species"]

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

BIOMEDICAL_DICTIONARIES: Dict[str, Type] = {
    "ctd-diseases": CTD_DISEASES_DICTIONARY,
    "ctd-chemicals": CTD_CHEMICALS_DICTIONARY,
    "ncbi-gene": NCBI_GENE_HUMAN_DICTIONARY,
    "ncbi-taxonomy": NCBI_TAXONOMY_DICTIONARY,
}

MODEL_NAME_TO_DICTIONARY = {
    "dmis-lab/biosyn-sapbert-bc5cdr-disease": "ctd-diseases",
    "dmis-lab/biosyn-sapbert-ncbi-disease": "ctd-diseases",
    "dmis-lab/biosyn-sapbert-bc5cdr-chemical": "ctd-chemicals",
    "dmis-lab/biosyn-sapbert-bc2gn": "ncbi-gene",
    "dmis-lab/biosyn-biobert-bc5cdr-disease": "ctd-chemicals",
    "dmis-lab/biosyn-biobert-ncbi-disease": "ctd-diseases",
    "dmis-lab/biosyn-biobert-bc5cdr-chemical": "ctd-chemicals",
    "dmis-lab/biosyn-biobert-bc2gn": "ncbi-gene",
}

DEFAULT_SPARSE_WEIGHT = 0.5


class SimilarityMetric(Enum):
    """Similarity metrics."""

    INNER_PRODUCT = auto()
    COSINE = auto()


PRETRAINED_MODEL_TO_SIMILARITY_METRIC = {m: SimilarityMetric.INNER_PRODUCT for m in PRETRAINED_MODELS}


def normalize_entity_type(entity_type: str) -> str:
    """Normalize entity type to ease interoperability."""
    entity_type = entity_type.lower()

    if entity_type == "diseases":
        entity_type = "disease"
    elif entity_type == "genes":
        entity_type = "gene"

    return entity_type


def load_dictionary(
    dictionary_name_or_path: Union[Path, str], dataset_name: Optional[str] = None
) -> EntityLinkingDictionary:
    """Load dictionary: either pre-defined or from path."""
    if isinstance(dictionary_name_or_path, str) and (
        dictionary_name_or_path in ENTITY_TYPE_TO_DICTIONARY or dictionary_name_or_path in BIOMEDICAL_DICTIONARIES
    ):
        dictionary_name_or_path = ENTITY_TYPE_TO_DICTIONARY.get(dictionary_name_or_path, dictionary_name_or_path)

        return BIOMEDICAL_DICTIONARIES[str(dictionary_name_or_path)]()

    if dataset_name is None:
        raise ValueError("When loading a custom dictionary, you need to specify a dataset_name!")
    return HunerEntityLinkingDictionary(path=dictionary_name_or_path, dataset_name=dataset_name)


class EntityPreprocessor(ABC):
    """A pre-processor used to transform / clean both entity mentions and entity names."""

    def initialize(self, sentences: List[Sentence]) -> None:
        """Initializes the pre-processor for a batch of sentences.

        This may be necessary for more sophisticated transformations.

        Args:
            sentences: List of sentences that will be processed.
        """

    def process_mention(self, entity_mention: str, sentence: Optional[Sentence] = None) -> str:
        """Processes the given entity mention and applies the transformation procedure to it.

        Usually just forwards the entity_mention to :meth:`EntityPreprocessor.process_entity_name`, but can be implemented
        to preprocess mentions on a sentence level instead.

        Args:
            entity_mention: entity mention under investigation
            sentence: sentence in which the entity mentioned occurred

        Returns:
            Cleaned / transformed string representation of the given entity mention
        """
        return self.process_entity_name(entity_mention)

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
        original = entity_name

        if self.lowercase:
            entity_name = entity_name.lower()

        if self.remove_punctuation:
            name_parts = self.rmv_puncts_regex.split(entity_name)
            entity_name = " ".join(name_parts).strip()

        entity_name = entity_name.strip()

        # NOTE: Avoid emtpy string if mentions are just punctutations (e.g. `-` or `(`)
        entity_name = original if len(entity_name) == 0 else entity_name

        return entity_name

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

    def __init__(
        self,
        preprocessor: Optional[EntityPreprocessor] = None,
    ) -> None:
        """Creates the mention pre-processor.

        Args:
            preprocessor: Basic entity preprocessor
        """
        try:
            import pyab3p
        except ImportError:
            raise ImportError("Please install pyab3p to use the `Ab3PEntityPreprocessor`")
        self.ab3p = pyab3p.Ab3p()

        self.preprocessor = preprocessor
        self.abbreviation_dict: Dict[str, Dict[str, str]] = {}

    def initialize(self, sentences: List[Sentence]) -> None:
        self.abbreviation_dict = self._build_abbreviation_dict(sentences)

    def process_mention(self, entity_mention: str, sentence: Optional[Sentence] = None) -> str:
        assert (
            sentence is not None
        ), "Ab3P requires the sentence where `entity_mention` was found for abbreviation resolution"

        original = entity_mention

        sentence_text = sentence.to_original_text()

        if entity_mention in self.abbreviation_dict.get(sentence_text, {}):
            entity_mention = self.abbreviation_dict[sentence_text][entity_mention]

        if self.preprocessor is not None:
            entity_mention = self.preprocessor.process_entity_name(entity_mention)

        # NOTE: Avoid emtpy string if mentions are just punctuations (e.g. `-` or `(`)
        entity_mention = original if len(entity_mention) == 0 else entity_mention

        return entity_mention

    def process_entity_name(self, entity_name: str) -> str:
        # Ab3P works on sentence-level and not on a single entity mention / name
        # - so we just apply the wrapped text pre-processing here (if configured)
        if self.preprocessor is not None:
            return self.preprocessor.process_entity_name(entity_name)

        return entity_name

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
        abbreviation_dict: Dict[str, Dict[str, str]] = {}

        for sentence in sentences:
            sentence_text = sentence.to_original_text()
            abbreviation_dict[sentence_text] = {
                abbr_out.short_form: abbr_out.long_form for abbr_out in self.ab3p.get_abbrs(sentence_text)
            }

        return abbreviation_dict

    def _get_state(self) -> Dict[str, Any]:
        return {
            **super()._get_state(),
            "preprocessor": None if self.preprocessor is None else self.preprocessor._get_state(),
        }

    @classmethod
    def _from_state(cls, state_dict: Dict[str, Any]) -> "EntityPreprocessor":
        return cls(
            preprocessor=(
                None
                if state_dict["preprocessor"] is None
                else EntityPreprocessor._from_state(state_dict["preprocessor"])
            ),
        )


class CandidateSearchIndex(ABC):
    """Base class for a candidate generator.

    Given a mention of an entity, find matching entries from the dictionary.
    """

    @abstractmethod
    def index(self, dictionary: EntityLinkingDictionary, preprocessor: Optional[EntityPreprocessor] = None) -> None:
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

    def index(self, dictionary: EntityLinkingDictionary, preprocessor: Optional[EntityPreprocessor] = None) -> None:
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
        embeddings: Dict[str, DocumentEmbeddings],
        hybrid_search: bool,
        similarity_metric: SimilarityMetric = SimilarityMetric.INNER_PRODUCT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        batch_size: int = 128,
        show_progress: bool = True,
    ):
        """Initializes the SemanticCandidateSearchIndex.

        Args:
            embeddings: A list of embeddings used for search.
            hybrid_search: combine sparse and dense embeddings
            sparse_weight: Weight for sparse embeddings.
            similarity_metric: The metric used to define similarity.
            batch_size: The batch size used for indexing embeddings.
            show_progress: show the progress while indexing.
        """
        self.embeddings = embeddings
        self.hybrid_search = hybrid_search
        self.sparse_weight = sparse_weight
        self.similarity_metric = similarity_metric
        self.show_progress = show_progress
        self.batch_size = batch_size

        self.ids: List[str] = []
        self._precomputed_embeddings: Dict[str, np.ndarray] = {"sparse": np.array([]), "dense": np.array([])}

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
        dictionary: Optional[EntityLinkingDictionary] = None,
    ) -> "SemanticCandidateSearchIndex":
        # NOTE: ensure correct similarity metric for pretrained model
        if model_name_or_path in PRETRAINED_MODELS:
            similarity_metric = PRETRAINED_MODEL_TO_SIMILARITY_METRIC[model_name_or_path]

        embeddings: Dict[str, DocumentEmbeddings] = {"dense": TransformerDocumentEmbeddings(model_name_or_path)}

        if hybrid_search:
            if dictionary is None:
                raise ValueError("Require dictionary to be set on hybrid search.")

            texts = []

            for candidate in dictionary.candidates:
                texts.append(candidate.concept_name)
                texts.extend(candidate.synonyms)

            if preprocessor is not None:
                texts = [preprocessor.process_entity_name(t) for t in texts]

            embeddings["sparse"] = DocumentTFIDFEmbeddings(
                [Sentence(t) for t in texts],
                analyzer="char",
                ngram_range=(1, 2),
            )

        sparse_weight = HYBRID_MODELS_SPARSE_WEIGHT.get(model_name_or_path, sparse_weight)

        return cls(
            embeddings,
            similarity_metric=similarity_metric,
            sparse_weight=sparse_weight,
            batch_size=batch_size,
            show_progress=show_progress,
            hybrid_search=hybrid_search,
        )

    def index(self, dictionary: EntityLinkingDictionary, preprocessor: Optional[EntityPreprocessor] = None) -> None:
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

        dense_embeddings = []
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

                self.embeddings["dense"].embed(batch)
                for sent in batch:
                    emb = sent.get_embedding()
                    if self.similarity_metric == SimilarityMetric.COSINE:
                        emb = emb / torch.norm(emb)
                    dense_embeddings.append(emb.cpu().numpy())
                    sent.clear_embeddings()

                # empty cuda cache if device is a cuda device
                if flair.device.type == "cuda":
                    torch.cuda.empty_cache()

        self._precomputed_embeddings["dense"] = np.stack(dense_embeddings, axis=0)

        if self.hybrid_search:
            sparse_embs = []
            batch = [Sentence(name) for name in texts]
            self.embeddings["sparse"].embed(batch)
            for sent in batch:
                sparse_emb = sent.get_embedding()
                if self.similarity_metric == SimilarityMetric.COSINE:
                    sparse_emb = sparse_emb / torch.norm(sparse_emb)
                sparse_embs.append(sparse_emb.cpu().numpy())
                sent.clear_embeddings()
            self._precomputed_embeddings["sparse"] = np.stack(sparse_embs, axis=0)

    def embed(self, entity_mentions: List[str]) -> Dict[str, np.ndarray]:
        query_embeddings: Dict[str, List] = {"dense": []}

        inputs = [Sentence(name) for name in entity_mentions]

        with torch.no_grad():
            for start in range(0, len(entity_mentions), self.batch_size):
                end = min(start + self.batch_size, len(entity_mentions))
                batch = inputs[start:end]
                self.embeddings["dense"].embed(batch)
                for sent in batch:
                    emb = sent.get_embedding(self.embeddings["dense"].get_names())
                    if self.similarity_metric == SimilarityMetric.COSINE:
                        emb = emb / torch.norm(emb)
                    query_embeddings["dense"].append(emb.cpu().numpy())
                    sent.clear_embeddings(self.embeddings["dense"].get_names())

                # Sanity conversion: if flair.device was set as a string, convert to torch.device
                if isinstance(flair.device, str):
                    flair.device = torch.device(flair.device)

                if flair.device.type == "cuda":
                    torch.cuda.empty_cache()

        if self.hybrid_search:
            query_embeddings["sparse"] = []
            self.embeddings["sparse"].embed(inputs)
            for sent in inputs:
                sparse_emb = sent.get_embedding(self.embeddings["sparse"].get_names())
                if self.similarity_metric == SimilarityMetric.COSINE:
                    sparse_emb = sparse_emb / torch.norm(sparse_emb)
                query_embeddings["sparse"].append(sparse_emb.cpu().numpy())
                sent.clear_embeddings(self.embeddings["sparse"].get_names())

        return {k: np.stack(v, axis=0) for k, v in query_embeddings.items()}

    def search(self, entity_mentions: List[str], top_k: int) -> List[List[Tuple[str, float]]]:
        """Returns the top-k entity / concept identifiers for each entity mention.

        Args:
            entity_mentions: Entity mentions
            top_k: Number of best-matching entities from the knowledge base to return

        Returns:
            List containing a list of entity linking candidates per entity mention from the input
        """
        mention_embs = self.embed(entity_mentions)

        scores = mention_embs["dense"] @ self._precomputed_embeddings["dense"].T

        if self.hybrid_search:
            query = sparse.csr_matrix(mention_embs["sparse"])
            index = sparse.csr_matrix(self._precomputed_embeddings["sparse"])
            sparse_scores = query.dot(index.T).toarray()
            scores += self.sparse_weight * sparse_scores

        num_mentions = scores.shape[0]
        unsorted_indices = np.argpartition(scores, -top_k)[:, -top_k:]
        unsorted_scores = scores[np.arange(num_mentions)[:, None], unsorted_indices]
        sorted_score_matrix_indices = np.argsort(-unsorted_scores)
        topk_idxs = unsorted_indices[np.arange(num_mentions)[:, None], sorted_score_matrix_indices]
        topk_scores = unsorted_scores[np.arange(num_mentions)[:, None], sorted_score_matrix_indices]

        results = []
        for i in range(num_mentions):
            results.append([(self.ids[j], s) for j, s in zip(topk_idxs[i, :], topk_scores[i, :])])

        return results

    @classmethod
    def _from_state(cls, state_dict: Dict[str, Any]) -> "SemanticCandidateSearchIndex":
        index = cls(
            embeddings=cast(
                Dict[str, DocumentEmbeddings], {k: load_embeddings(emb) for k, emb in state_dict["embeddings"].items()}
            ),
            similarity_metric=SimilarityMetric(state_dict["similarity_metric"]),
            sparse_weight=state_dict["sparse_weight"],
            batch_size=state_dict["batch_size"],
            hybrid_search=state_dict["hybrid_search"],
            show_progress=state_dict["show_progress"],
        )
        index.ids = state_dict["ids"]
        index._precomputed_embeddings = state_dict["precomputed_embeddings"]
        return index

    def _get_state(self) -> Dict[str, Any]:
        return {
            **super()._get_state(),
            "embeddings": {k: emb.save_embeddings() for k, emb in self.embeddings.items()},
            "similarity_metric": self.similarity_metric.value,
            "sparse_weight": self.sparse_weight,
            "batch_size": self.batch_size,
            "show_progress": self.show_progress,
            "hybrid_search": self.hybrid_search,
            "ids": self.ids,
            "precomputed_embeddings": self._precomputed_embeddings,
        }


class EntityMentionLinker(flair.nn.Model[Sentence]):
    """Entity linking model for the biomedical domain."""

    def __init__(
        self,
        candidate_generator: CandidateSearchIndex,
        preprocessor: EntityPreprocessor,
        entity_label_types: Union[str, Sequence[str], Dict[str, Set[str]]],
        label_type: str,
        dictionary: EntityLinkingDictionary,
        batch_size: int = 1024,
    ):
        """Initializes an entity mention linker.

        Args:
            candidate_generator: Strategy to find matching entities for a given mention
            preprocessor: Pre-processing strategy to transform / clean entity mentions
            entity_label_types: A label type or sequence of label types of the required entities.
                                You can also specify a label filter in a dictionary with the label type as key and the valid entity labels as values in a set.
                                E.g. to use only 'disease' and 'chemical' labels from a NER-tagger: `{'ner': {'disease', 'chemical'}}`.
                                To use all labels from 'ner', pass 'ner'
            label_type: The label under which the predictions of the linker should be stored
            dictionary: The dictionary listing all entities
            batch_size: Batch size to encode mentions/dictionary names
        """
        self.preprocessor = preprocessor
        self.candidate_generator = candidate_generator
        self.entity_label_types = self.get_entity_label_types(entity_label_types)
        self._label_type = label_type
        self._dictionary = dictionary
        self.batch_size = batch_size
        self._warned_legacy_sequence_tagger = False
        super().__init__()

    def get_entity_label_types(
        self, entity_label_types: Union[str, Sequence[str], Dict[str, Set[str]]]
    ) -> Dict[str, Set[str]]:
        """Find out what NER labels to extract from sentence.

        Args:
            entity_label_types: A label type or sequence of label types of the required entities.
                                You can also specify a label filter in a dictionary with the label type as key and the valid entity labels as values in a set.
                                E.g. to use only 'disease' and 'chemical' labels from a NER-tagger: `{'ner': {'disease', 'chemical'}}`.
                                To use all labels from 'ner', pass 'ner'
        """
        if isinstance(entity_label_types, str):
            entity_label_types = cast(Dict[str, Set[str]], {entity_label_types: {}})
        elif isinstance(entity_label_types, Sequence):
            entity_label_types = cast(Dict[str, Set[str]], {label: {} for label in entity_label_types})

        entity_label_types = {
            label: {normalize_entity_type(e) for e in entity_types}
            for label, entity_types in entity_label_types.items()
        }

        return entity_label_types

    @property
    def label_type(self):
        return self._label_type

    @property
    def dictionary(self) -> EntityLinkingDictionary:
        return self._dictionary

    def extract_entities_mentions(self, sentence: Sentence, entity_label_types: Dict[str, Set[str]]) -> List[Label]:
        """Extract tagged mentions from sentences."""
        entities_mentions: List[Label] = []

        # NOTE: This is a hacky workaround for the fact that
        # the `label_type`s in `Classifier.load('hunflair)` are
        # 'diseases', 'genes', 'species', 'chemical' instead of 'ner'.
        # We warn users once they need to update SequenceTagger model
        # See: https://github.com/flairNLP/flair/pull/3387
        if any(label in ["diseases", "genes", "species", "chemical"] for label in sentence.annotation_layers):
            if not self._warned_legacy_sequence_tagger:
                logger.warning(
                    "It appears that the sentences have been annotated with HunFlair (version 1). "
                    "Consider using HunFlair2 for improved extraction performance: Classifier.load('hunflair2')."
                    "See https://github.com/flairNLP/flair/blob/master/resources/docs/HUNFLAIR2.md for further "
                    "information."
                )
                self._warned_legacy_sequence_tagger = True

            entity_types = {e for sublist in entity_label_types.values() for e in sublist}
            entities_mentions = [
                label for label in sentence.get_labels() if normalize_entity_type(label.value) in entity_types
            ]
        else:
            for label_type, entity_types in entity_label_types.items():
                labels = sentence.get_labels(label_type)
                if len(entity_types) > 0:
                    labels = [label for label in labels if normalize_entity_type(label.value) in entity_types]
                entities_mentions.extend(labels)

        return entities_mentions

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        top_k: int = 1,
        pred_label_type: Optional[str] = None,
        entity_label_types: Optional[Union[str, Sequence[str], Dict[str, Set[str]]]] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Predicts the best matching top-k entity / concept identifiers of all named entities annotated with tag input_entity_annotation_layer.

        Args:
            sentences: One or more sentences to run the prediction on
            top_k: Number of best-matching entity / concept identifiers
            entity_label_types: A label type or sequence of label types of the required entities.
                                You can also specify a label filter in a dictionary with the label type as key and the valid entity labels as values in a set.
                                E.g. to use only 'disease' and 'chemical' labels from a NER-tagger: `{'ner': {'disease', 'chemical'}}`.
                                To use all labels from 'ner', pass 'ner'
            pred_label_type: The label under which the predictions of the linker should be stored
            batch_size: Batch size to encode mentions/dictionary names
        """
        # make sure sentences is a list of sentences
        if not isinstance(sentences, list):
            sentences = [sentences]
        if batch_size is None:
            batch_size = self.batch_size

        # Make sure entity label types are represented as dict
        entity_label_types = (
            self.get_entity_label_types(entity_label_types)
            if entity_label_types is not None
            else self.entity_label_types
        )

        pred_label_type = pred_label_type if pred_label_type is not None else self.label_type

        if self.preprocessor is not None:
            self.preprocessor.initialize(sentences)

        data_points = []
        mentions = []

        for sentence in sentences:
            # Collect all entities based on entity type labels configuration
            entities_mentions = self.extract_entities_mentions(sentence, entity_label_types)

            # Preprocess entity mentions
            for entity in entities_mentions:
                data_points.append(entity.data_point)
                mentions.append(
                    (
                        self.preprocessor.process_mention(entity.data_point.text, sentence)
                        if self.preprocessor is not None
                        else entity.data_point.text
                    ),
                )

        # Retrieve top-k concept / entity candidates
        for i in range(0, len(mentions), batch_size):
            candidates = self.candidate_generator.search(entity_mentions=mentions[i : i + batch_size], top_k=top_k)

            # Add a label annotation for each candidate
            for data_point, mention_candidates in zip(data_points[i : i + batch_size], candidates):
                for candidate_id, confidence in mention_candidates:
                    data_point.add_label(
                        pred_label_type, candidate_id, confidence, name=self.dictionary[candidate_id].concept_name
                    )

    @staticmethod
    def _fetch_model(model_name: str) -> str:
        if Path(model_name).exists():
            return model_name

        bio_base_repo = "hunflair"
        hf_model_map = {
            "gene-linker": f"{bio_base_repo}/biosyn-sapbert-bc2gn",
            "gene-linker-no-ab3p": f"{bio_base_repo}/biosyn-sapbert-bc2gn-no-ab3p",
            "disease-linker": f"{bio_base_repo}/biosyn-sapbert-bc5cdr-disease",
            "disease-linker-no-ab3p": f"{bio_base_repo}/biosyn-sapbert-bc5cdr-disease-no-ab3p",
            "chemical-linker": f"{bio_base_repo}/biosyn-sapbert-bc5cdr-chemical",
            "chemical-linker-no-ab3p": f"{bio_base_repo}/biosyn-sapbert-bc5cdr-chemical-no-ab3p",
            "species-linker": f"{bio_base_repo}/sapbert-ncbi-taxonomy",
            "species-linker-no-ab3p": f"{bio_base_repo}/sapbert-ncbi-taxonomy-no-ab3p",
        }

        if model_name in hf_model_map:
            model_name = hf_model_map[model_name]

            if not model_name.endswith("-no-ab3p") and importlib.util.find_spec("pyab3p") is None:
                logger.warning(
                    "'pyab3p' is not found, switching to a model without abbreviation resolution. "
                    "This might impact the model performance. To reach full performance, please install"
                    "pyab3p by running:"
                    "   pip install pyab3p"
                )
                model_name += "-no-ab3p"

        return hf_download(model_name)

    @classmethod
    def _init_model_with_state_dict(cls, state: Dict[str, Any], **kwargs) -> "EntityMentionLinker":
        candidate_generator = CandidateSearchIndex._from_state(state["candidate_search_index"])
        preprocessor = EntityPreprocessor._from_state(state["entity_preprocessor"])
        entity_label_types = state["entity_label_types"]
        label_type = state["label_type"]
        dictionary = InMemoryEntityLinkingDictionary.from_state(state["dictionary"])
        batch_size = state.get("batch_size", 128)
        return cls(candidate_generator, preprocessor, entity_label_types, label_type, dictionary, batch_size=batch_size)

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        return {
            **super()._get_state_dict(),
            "label_type": self.label_type,
            "entity_label_types": self.entity_label_types,
            "entity_preprocessor": self.preprocessor._get_state(),
            "candidate_search_index": self.candidate_generator._get_state(),
            "dictionary": self.dictionary.to_in_memory_dictionary().to_state(),
            "batch_size": self.batch_size,
        }

    @classmethod
    def build(
        cls,
        model_name_or_path: str,
        label_type: str = "link",
        dictionary_name_or_path: Optional[Union[str, Path]] = None,
        hybrid_search: bool = True,
        batch_size: int = 128,
        similarity_metric: SimilarityMetric = SimilarityMetric.INNER_PRODUCT,
        preprocessor: Optional[EntityPreprocessor] = None,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        entity_type: Optional[str] = None,
        dictionary: Optional[EntityLinkingDictionary] = None,
        dataset_name: Optional[str] = None,
    ) -> "EntityMentionLinker":
        """Builds a model for biomedical named entity normalization.

        Args:
            model_name_or_path: the name to an transformer embedding model on the huggingface hub or "exact-string-match"
            label_type: the label-type the predictions should be assigned to
            dictionary_name_or_path: the name or path to a dictionary. If the model name is a common biomedical model, the dictionary name is asigned by default. Otherwise you can pass any of "gene", "species", "disease", "chemical" to get the respective biomedical dictionary.
            hybrid_search: if True add a character-ngram-tfidf embedding on top of the transformer embedding model.
            batch_size: the batch_size used when indexing the dictionary.
            similarity_metric: the metric used to compare similarity between two embeddings.
            preprocessor: The preprocessor used to preprocess. If None is passed, it used an AD3P processor.
            sparse_weight: if hybrid_search is added, the sparse weight will weight the importance of the character-ngram-tfidf embedding. For the common models, this will be overwritten with a specific value.
            entity_type: the entity type of the mentions
            dictionary: the dictionary provided in memory. If None, the dictionary is loaded from dictionary_name_or_path.
            dataset_name: the name to assign the dictionary for reference.
        """
        if dictionary is None:
            if dictionary_name_or_path is None or isinstance(dictionary_name_or_path, str):
                dictionary_name_or_path = cls.__get_dictionary_path(
                    model_name_or_path=model_name_or_path, dictionary_name_or_path=dictionary_name_or_path
                )
            dictionary = load_dictionary(dictionary_name_or_path, dataset_name=dataset_name)

        model_name_or_path, entity_type = cls.__get_model_path_and_entity_type(
            model_name_or_path=model_name_or_path,
            entity_type=entity_type,
            hybrid_search=hybrid_search,
        )

        preprocessor = (
            preprocessor
            if preprocessor is not None
            else Ab3PEntityPreprocessor(preprocessor=BioSynEntityPreprocessor())
        )

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
            "EntityMentionLinker predicts: Dictionary `%s` (entity type: %s)", dictionary_name_or_path, entity_type
        )

        return cls(
            candidate_generator=candidate_generator,
            preprocessor=preprocessor,
            entity_label_types={"ner": {entity_type}},
            label_type=label_type,
            dictionary=dictionary,
        )

    @staticmethod
    def __get_model_path_and_entity_type(
        model_name_or_path: str,
        entity_type: Optional[str] = None,
        hybrid_search: bool = False,
    ) -> Tuple[str, str]:
        """Try to figure out what model the user wants."""
        if model_name_or_path not in MODELS and model_name_or_path not in ENTITY_TYPES:
            raise ValueError(
                f"Unknown model `{model_name_or_path}`!"
                " If you want to pass a local path please use the `Path` class, i.e. `model_name_or_path=Path(my_path)`"
            )

        if model_name_or_path == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext":
            assert entity_type is not None, f"For model {model_name_or_path} you must specify `entity_type`"
            entity_type = normalize_entity_type(entity_type)

        if hybrid_search:
            # load model by entity_type
            if model_name_or_path in ENTITY_TYPES:
                model_name_or_path = cast(str, model_name_or_path)
                entity_type = model_name_or_path

                # check if we have a hybrid pre-trained model
                if model_name_or_path in ENTITY_TYPE_TO_HYBRID_MODEL:
                    model_name_or_path = ENTITY_TYPE_TO_HYBRID_MODEL[model_name_or_path]
                else:
                    logger.warning(
                        "EntityMentionLinker: `hybrid_search=True` but model for entity type `%s` was not trained for hybrid search."
                        " Results may be poor.",
                        model_name_or_path,
                    )
                    model_name_or_path = ENTITY_TYPE_TO_DENSE_MODEL[model_name_or_path]
            elif model_name_or_path not in PRETRAINED_HYBRID_MODELS:
                logger.warning(
                    "EntityMentionLinker: `hybrid_search=True` but model `%s` was not trained for hybrid search."
                    " Results may be poor.",
                    model_name_or_path,
                )
                assert (
                    entity_type is not None
                ), f"For non-hybrid model `{model_name_or_path}` with `hybrid_search=True` you must specify `entity_type`"
            else:
                model_name_or_path = cast(str, model_name_or_path)
                entity_type = PRETRAINED_HYBRID_MODELS[model_name_or_path]
        elif model_name_or_path in ENTITY_TYPES:
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
                    f"When using a custom model you need to specify a dictionary. Available options are: {list(ENTITY_TYPE_TO_DICTIONARY.values())}. "
                    "Or provide a path to a dictionary file."
                )

        return dictionary_name_or_path

    def forward_loss(self, data_points: List[DT]) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError("The EntityLinker cannot be trained")

    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "EntityMentionLinker":
        from typing import cast

        return cast("EntityMentionLinker", super().load(model_path=model_path))

    def evaluate(
        self,
        data_points: Union[List[Sentence], Dataset],
        gold_label_type: str,
        out_path: Optional[Union[str, Path]] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        main_evaluation_metric: Tuple[str, str] = ("accuracy", "f1-score"),
        exclude_labels: Optional[List[str]] = None,
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        k: int = 1,
        **kwargs,
    ) -> Result:
        exclude_labels = exclude_labels if exclude_labels is not None else []
        if gold_label_dictionary is not None:
            raise NotImplementedError("evaluating an EntityMentionLinker with a gold_label_dictionary is not supported")

        if isinstance(data_points, Dataset):
            data_points = list(_iter_dataset(data_points))

        self.predict(
            data_points,
            top_k=k,
            pred_label_type="predicted",
            entity_label_types=gold_label_type,
            batch_size=mini_batch_size,
        )

        hits = 0
        total = 0
        for sentence in data_points:
            spans = sentence.get_spans(gold_label_type)
            for span in spans:
                exps = {exp.value for exp in span.get_labels(gold_label_type) if exp.value not in exclude_labels}

                predictions = {pred.value for pred in span.get_labels("predicted")}
                total += 1
                if exps & predictions:
                    hits += 1
            sentence.remove_labels("predicted")
        accuracy = hits / total

        detailed_results = f"Accuracy@{k}: {accuracy:0.2%}"
        scores = {"accuracy": accuracy, f"accuracy@{k}": accuracy, "loss": 0.0}
        return Result(main_score=accuracy, detailed_results=detailed_results, scores=scores)
