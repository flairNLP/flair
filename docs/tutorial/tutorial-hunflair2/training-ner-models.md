# HunFlair2 Tutorial 3: Training NER models

This part of the tutorial shows how you can train your own biomedical named entity recognition models
using state-of-the-art pretrained Transformers embeddings.

For this tutorial, we assume that you're familiar with the [base types](project:../tutorial-basics/basic-types.md) of Flair
and how [transformers_word embeddings](project:../tutorial-training/how-to-train-sequence-tagger.md).
You should also know how to [load a corpus](project:../tutorial-training/how-to-load-prepared-dataset.md).

## Train a biomedical NER model from scratch: single entity type

Here is example code for a biomedical NER model trained on the [`NCBI_DISEASE`](#flair.datasets.biomedical.NCBI_DISEASE) corpus using Transformer word embeddings.
This will result in a tagger specialized for a single entity type, i.e. "Disease".

```python
from flair.datasets import NCBI_DISEASE

# 1. get the corpus
corpus = NCBI_DISEASE()
print(corpus)

# 2. make the tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type="ner", add_unk=False)

# 3. initialize embeddings
from flair.embeddings import TransformerWordEmbeddings

embeddings: TransformerWordEmbeddings = TransformerWordEmbeddings(
    "michiyasunaga/BioLinkBERT-base",
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
    model_max_length=512,
)

# 4. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_format="BIOES",
    tag_type="ner",
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)

# 5. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.fine_tune(
    base_path="taggers/ncbi-disease",
    train_with_dev=False,
    max_epochs=16,
    learning_rate=2.0e-5,
    mini_batch_size=16,
    shuffle=False,
)
```

Once the model is trained you can use it to predict tags for new sentences.
Just call the predict method of the model.

```python
# load the model you trained
model = SequenceTagger.load("taggers/ncbi-disease/best-model.pt")

# create example sentence
from flair.data import Sentence
sentence = Sentence("Women who smoke 20 cigarettes a day are four times more likely to develop breast cancer.")

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
```

If the model works well, it will correctly tag "breast cancer" as disease in this example:

```
Women who smoke 20 cigarettes a day are four times more likely to develop breast <B-Disease> cancer <E-Disease> .
```

## Train a biomedical NER model: multiple entity types

If you are dealing with multiple entity types, e.g. "Disease" and "Chemicals", you can opt
to train a single model capable of handling multiple entity types at once.
This can be achieved by using the  [`PrefixedSequenceTagger()`](#flair.models.PrefixedSequenceTagger) class, which implements the method described in \[1\].
This model uses prompting, i.e. it adds a prefix (hence the name) string in front of specifying the
entity types requested for tagging: `[Tag <entity-type-0>, <entity-type-1>, ...]`.
This is especially useful for training, as it allows to combine multiple corpora even if they cover different subsets of entity types.

```python
# 1. get the corpora
from flair.datasets.biomedical import HUNER_ALL_CDR, HUNER_CHEMICAL_NLM_CHEM
corpora = (HUNER_ALL_CDR(), HUNER_CHEMICAL_NLM_CHEM())

# 2. add prefixed strings to each corpus by prepending its tagged entity
#    types "[Tag <entity-type-0>, <entity-type-1>, ...]"
from flair.data import MultiCorpus
from flair.models.prefixed_tagger import EntityTypeTaskPromptAugmentationStrategy
from flair.datasets.biomedical import (
    BIGBIO_NER_CORPUS,
    CELL_LINE_TAG,
    CHEMICAL_TAG,
    DISEASE_TAG,
    GENE_TAG,
    SPECIES_TAG,
)

mapping = {
    CELL_LINE_TAG: "cell lines",
    CHEMICAL_TAG: "chemicals",
    DISEASE_TAG: "diseases",
    GENE_TAG: "genes",
    SPECIES_TAG: "species",
}

prefixed_corpora = []
all_entity_types = set()
for corpus in corpora:
    entity_types = sorted(
        set(
            [
                mapping[tag]
                for tag in corpus.get_entity_type_mapping().values()
            ]
        )
    )
    all_entity_types.update(set(entity_types))

    print(f"Entity types in {corpus}: {entity_types}")

    augmentation_strategy = EntityTypeTaskPromptAugmentationStrategy(
        entity_types
    )
    prefixed_corpora.append(
        augmentation_strategy.augment_corpus(corpus)
    )

corpora = MultiCorpus(prefixed_corpora)
all_entity_types = sorted(all_entity_types)

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type="ner")

# 4. the final model will on default predict all the entity types seen
#    in the training corpora, e.g., disease and chemicals here
augmentation_strategy = EntityTypeTaskPromptAugmentationStrategy(
    all_entity_types
)

# 5. initialize embeddings
from flair.embeddings import TransformerWordEmbeddings

embeddings: TransformerWordEmbeddings = TransformerWordEmbeddings(
    "michiyasunaga/BioLinkBERT-base",
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
    model_max_length=512,
)

# 4. initialize sequence tagger
from flair.models.prefixed_tagger import PrefixedSequenceTagger

tagger: SequenceTagger = PrefixedSequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_format="BIOES",
    tag_type="ner",
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
    augmentation_strategy=augmentation_strategy,
)

# 5. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.fine_tune(
    base_path="taggers/cdr_nlm_chem",
    train_with_dev=False,
    max_epochs=16,
    learning_rate=2.0e-5,
    mini_batch_size=16,
    shuffle=False,
)
```

## Training HunFlair2 from scratch

*HunFlair2* uses the `PrefixedSequenceTagger()` class as defined above but adds the following corpora to the training set instead:

```python
from flair.datasets.biomedical import (
    HUNER_ALL_BIORED, HUNER_GENE_NLM_GENE,
    HUNER_GENE_GNORMPLUS, HUNER_ALL_SCAI,
    HUNER_CHEMICAL_NLM_CHEM, HUNER_SPECIES_LINNEAUS,
    HUNER_SPECIES_S800, HUNER_DISEASE_NCBI
)

corpora = (
    HUNER_ALL_BIORED(), HUNER_GENE_NLM_GENE(),
    HUNER_GENE_GNORMPLUS(), HUNER_ALL_SCAI(),
    HUNER_CHEMICAL_NLM_CHEM(), HUNER_SPECIES_LINNEAUS(),
    HUNER_SPECIES_S800(), HUNER_DISEASE_NCBI()
)

```

## References

\[1\] Luo, L., Wei, C. H., Lai, P. T., Leaman, R., Chen, Q., & Lu, Z. (2023). AIONER: all-in-one scheme-based biomedical named entity recognition using deep learning. Bioinformatics, 39(5), btad310.
