# Training the models from the paper
Here's how we trained the models that we evaluated in [our paper](https://arxiv.org/abs/2008.07347).

```python
# 1. define corpora
from flair.datasets import biomedical

CELLLINE_CORPORA = [
    biomedical.HUNER_CELL_LINE_CELL_FINDER(),
    biomedical.HUNER_CELL_LINE_CLL(),
    biomedical.HUNER_CELL_LINE_GELLUS(),
    biomedical.HUNER_CELL_LINE_JNLPBA()
]

CHEMICAL_CORPORA = [
    biomedical.HUNER_CHEMICAL_CDR(),
    biomedical.HUNER_CHEMICAL_CEMP(),
    biomedical.HUNER_CHEMICAL_CHEBI(),
    biomedical.HUNER_CHEMICAL_CHEMDNER(),
    biomedical.HUNER_CHEMICAL_SCAI()
]

DISEASE_CORPORA = [
    biomedical.HUNER_DISEASE_CDR(),
    biomedical.HUNER_DISEASE_MIRNA(),
    biomedical.HUNER_DISEASE_NCBI(),
    biomedical.HUNER_DISEASE_SCAI(),
    biomedical.HUNER_DISEASE_VARIOME()
]

GENE_CORPORA = [
    biomedical.HUNER_GENE_BC2GM(),
    biomedical.HUNER_GENE_BIO_INFER(),
    biomedical.HUNER_GENE_CELL_FINDER(),
    biomedical.HUNER_GENE_CHEBI(),
    biomedical.HUNER_GENE_DECA(),
    biomedical.HUNER_GENE_FSU(),
    biomedical.HUNER_GENE_GPRO(),
    biomedical.HUNER_GENE_IEPA(),
    biomedical.HUNER_GENE_JNLPBA(),
    biomedical.HUNER_GENE_LOCTEXT(),
    biomedical.HUNER_GENE_MIRNA(),
    biomedical.HUNER_GENE_OSIRIS(),
    biomedical.HUNER_GENE_VARIOME()
]


SPECIES_CORPORA = [
    biomedical.HUNER_SPECIES_CELL_FINDER(),
    biomedical.HUNER_SPECIES_CHEBI(),
    biomedical.HUNER_SPECIES_LINNEAUS(),
    biomedical.HUNER_SPECIES_LOCTEXT(),
    biomedical.HUNER_SPECIES_MIRNA(),
    biomedical.HUNER_SPECIES_S800(),
    biomedical.HUNER_SPECIES_VARIOME()
]

# 2. initialize embeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
embedding_types = [
    WordEmbeddings("pubmed"),
    FlairEmbeddings("pubmed-forward"),
    FlairEmbeddings("pubmed-backward"),

]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 3. Initialize corpus
# We also train on the test portions of the corpora, because we evaluate on held-out corpora
from flair.data import MultiCorpus
from torch.utils.data import ConcatDataset
corpus = MultiCorpus(GENE_CORPORA)
corpus._train = ConcatDataset([corpus._train, corpus._test])

# 4. Initialize sequence tagger
from flair.models import SequenceTagger
tag_dictionary = corpus.make_tag_dictionary(tag_type="ner")

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type="ner",
    use_crf=True,
    locked_dropout=0.5
)

# 5. train the model
from flair.trainers import ModelTrainer
trainer = ModelTrainer(tagger, corpus)

trainer.train(
    base_path="taggers/hunflair-gene",
    train_with_dev=False,
    max_epochs=200,
    learning_rate=0.1,
    mini_batch_size=32
)
```

The taggers for the other entity types are trained analogously.
Details on the evaluation can be found in a [dedicated github repository](https://github.com/hu-ner/hunflair-experiments).
