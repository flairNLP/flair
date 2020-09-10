# Expose base classses
from .base import DataLoader
from .base import SentenceDataset
from .base import StringDataset
from .base import MongoDataset

# Expose all sequence labeling datasets
from .sequence_labeling import ColumnCorpus
from .sequence_labeling import ColumnDataset
from .sequence_labeling import BIOFID
from .sequence_labeling import BIOSCOPE
from .sequence_labeling import CONLL_03
from .sequence_labeling import CONLL_03_GERMAN
from .sequence_labeling import CONLL_03_DUTCH
from .sequence_labeling import TWITTER_NER
from .sequence_labeling import CONLL_03_SPANISH
from .sequence_labeling import CONLL_2000
from .sequence_labeling import DANE
from .sequence_labeling import EUROPARL_NER_GERMAN
from .sequence_labeling import GERMEVAL_14
from .sequence_labeling import INSPEC
from .sequence_labeling import LER_GERMAN
from .sequence_labeling import NER_BASQUE
from .sequence_labeling import NER_FINNISH
from .sequence_labeling import NER_SWEDISH
from .sequence_labeling import SEMEVAL2010
from .sequence_labeling import SEMEVAL2017
from .sequence_labeling import WIKINER_ENGLISH
from .sequence_labeling import WIKINER_GERMAN
from .sequence_labeling import WIKINER_DUTCH
from .sequence_labeling import WIKINER_FRENCH
from .sequence_labeling import WIKINER_ITALIAN
from .sequence_labeling import WIKINER_SPANISH
from .sequence_labeling import WIKINER_PORTUGUESE
from .sequence_labeling import WIKINER_POLISH
from .sequence_labeling import WIKINER_RUSSIAN
from .sequence_labeling import WNUT_17
from .sequence_labeling import MIT_RESTAURANTS

# Expose all document classification datasets
from .document_classification import ClassificationCorpus
from .document_classification import ClassificationDataset
from .document_classification import CSVClassificationCorpus
from .document_classification import CSVClassificationDataset
from .document_classification import AMAZON_REVIEWS
from .document_classification import IMDB
from .document_classification import NEWSGROUPS
from .document_classification import SENTIMENT_140
from .document_classification import SENTEVAL_CR
from .document_classification import SENTEVAL_MR
from .document_classification import SENTEVAL_MPQA
from .document_classification import SENTEVAL_SUBJ
from .document_classification import SENTEVAL_SST_BINARY
from .document_classification import SENTEVAL_SST_GRANULAR
from .document_classification import TREC_50
from .document_classification import TREC_6
from .document_classification import COMMUNICATIVE_FUNCTIONS
from .document_classification import WASSA_ANGER
from .document_classification import WASSA_FEAR
from .document_classification import WASSA_JOY
from .document_classification import WASSA_SADNESS

# Expose all treebanks
from .treebanks import UniversalDependenciesCorpus
from .treebanks import UniversalDependenciesDataset
from .treebanks import UD_ENGLISH
from .treebanks import UD_GERMAN
from .treebanks import UD_GERMAN_HDT
from .treebanks import UD_DUTCH
from .treebanks import UD_FRENCH
from .treebanks import UD_ITALIAN
from .treebanks import UD_SPANISH
from .treebanks import UD_PORTUGUESE
from .treebanks import UD_ROMANIAN
from .treebanks import UD_CATALAN
from .treebanks import UD_POLISH
from .treebanks import UD_CZECH
from .treebanks import UD_SLOVAK
from .treebanks import UD_SWEDISH
from .treebanks import UD_DANISH
from .treebanks import UD_NORWEGIAN
from .treebanks import UD_FINNISH
from .treebanks import UD_SLOVENIAN
from .treebanks import UD_CROATIAN
from .treebanks import UD_SERBIAN
from .treebanks import UD_BULGARIAN
from .treebanks import UD_ARABIC
from .treebanks import UD_HEBREW
from .treebanks import UD_TURKISH
from .treebanks import UD_PERSIAN
from .treebanks import UD_RUSSIAN
from .treebanks import UD_HINDI
from .treebanks import UD_INDONESIAN
from .treebanks import UD_JAPANESE
from .treebanks import UD_CHINESE
from .treebanks import UD_KOREAN
from .treebanks import UD_BASQUE

# Expose all text-text datasets
from .text_text import ParallelTextCorpus
from .text_text import ParallelTextDataset
from .text_text import OpusParallelCorpus

# Expose all text-image datasets
from .text_image import FeideggerCorpus
from .text_image import FeideggerDataset

# Expose all biomedical data sets
from .biomedical import ANAT_EM
from .biomedical import AZDZ
from .biomedical import BIONLP2013_PC
from .biomedical import BIONLP2013_CG
from .biomedical import BIO_INFER
from .biomedical import BIOSEMANTICS
from .biomedical import BC2GM
from .biomedical import CELL_FINDER
from .biomedical import CEMP
from .biomedical import CDR
from .biomedical import CHEMDNER
from .biomedical import CRAFT
from .biomedical import CRAFT_V4
from .biomedical import CLL
from .biomedical import DECA
from .biomedical import FSU
from .biomedical import GELLUS
from .biomedical import GPRO
from .biomedical import IEPA
from .biomedical import JNLPBA
from .biomedical import LOCTEXT
from .biomedical import LINNEAUS
from .biomedical import NCBI_DISEASE
from .biomedical import MIRNA
from .biomedical import OSIRIS
from .biomedical import PDR
from .biomedical import S800
from .biomedical import SCAI_CHEMICALS
from .biomedical import SCAI_DISEASE
from .biomedical import VARIOME

# Expose all biomedical data sets using the HUNER splits
from .biomedical import HUNER_CHEMICAL
from .biomedical import HUNER_CHEMICAL_CHEBI
from .biomedical import HUNER_CHEMICAL_CHEMDNER
from .biomedical import HUNER_CHEMICAL_CDR
from .biomedical import HUNER_CHEMICAL_CEMP
from .biomedical import HUNER_CHEMICAL_SCAI
from .biomedical import HUNER_CHEMICAL_CRAFT_V4
# -
from .biomedical import HUNER_CELL_LINE
from .biomedical import HUNER_CELL_LINE_CLL
from .biomedical import HUNER_CELL_LINE_CELL_FINDER
from .biomedical import HUNER_CELL_LINE_GELLUS
from .biomedical import HUNER_CELL_LINE_JNLPBA
# -
from .biomedical import HUNER_DISEASE
from .biomedical import HUNER_DISEASE_CDR
from .biomedical import HUNER_DISEASE_MIRNA
from .biomedical import HUNER_DISEASE_NCBI
from .biomedical import HUNER_DISEASE_SCAI
from .biomedical import HUNER_DISEASE_VARIOME
from .biomedical import HUNER_DISEASE_PDR
# -
from .biomedical import HUNER_GENE
from .biomedical import HUNER_GENE_BIO_INFER
from .biomedical import HUNER_GENE_BC2GM
from .biomedical import HUNER_GENE_CHEBI
from .biomedical import HUNER_GENE_CRAFT_V4
from .biomedical import HUNER_GENE_CELL_FINDER
from .biomedical import HUNER_GENE_DECA
from .biomedical import HUNER_GENE_FSU
from .biomedical import HUNER_GENE_GPRO
from .biomedical import HUNER_GENE_IEPA
from .biomedical import HUNER_GENE_JNLPBA
from .biomedical import HUNER_GENE_LOCTEXT
from .biomedical import HUNER_GENE_MIRNA
from .biomedical import HUNER_GENE_OSIRIS
from .biomedical import HUNER_GENE_VARIOME
# -
from .biomedical import HUNER_SPECIES
from .biomedical import HUNER_SPECIES_CELL_FINDER
from .biomedical import HUNER_SPECIES_CHEBI
from .biomedical import HUNER_SPECIES_CRAFT_V4
from .biomedical import HUNER_SPECIES_LOCTEXT
from .biomedical import HUNER_SPECIES_LINNEAUS
from .biomedical import HUNER_SPECIES_MIRNA
from .biomedical import HUNER_SPECIES_S800
from .biomedical import HUNER_SPECIES_VARIOME

# Expose all biomedical data sets used for the evaluation of BioBERT
from .biomedical import BIOBERT_CHEMICAL_BC4CHEMD
from .biomedical import BIOBERT_CHEMICAL_BC5CDR
from .biomedical import BIOBERT_DISEASE_NCBI
from .biomedical import BIOBERT_DISEASE_BC5CDR
from .biomedical import BIOBERT_SPECIES_LINNAEUS
from .biomedical import BIOBERT_SPECIES_S800
from .biomedical import BIOBERT_GENE_BC2GM
from .biomedical import BIOBERT_GENE_JNLPBA
