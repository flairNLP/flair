# Train a Span Classifier

Span Classification models are used to model problems such as entity linking, where you already have extracted some
relevant spans within the `Sentence` and want to predict some more fine-grained labels.

This tutorial section show you how to train models using the [Span Classifier](#flair.models.SpanClassifier) in Flair.

## Training an entity linker (NEL) model with transformers

For a state-of-the-art NER system you should fine-tune transformer embeddings, and use full document context
(see our [FLERT](https://arxiv.org/abs/2011.06993) paper for details).

Use the following script:

```python
from flair.datasets import ZELDA
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SpanClassifier
from flair.models.entity_linker_model import CandidateGenerator
from flair.trainers import ModelTrainer
from flair.nn.decoder import PrototypicalDecoder


# 1. get the corpus
corpus = ZELDA()
print(corpus)

# 2. what label do we want to predict?
label_type = 'nel'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True)
print(label_dict)

# 4. initialize fine-tuneable transformer embeddings WITH document context
embeddings = TransformerWordEmbeddings(
    model="bert-base-uncased",
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SpanClassifier(
    embeddings=embeddings,
    label_dictionary=label_dict,
    label_type=label_type,
    decoder=PrototypicalDecoder(
        num_prototypes=len(label_dict),
        embeddings_size=embeddings.embedding_length * 2, # we use "first_last" encoding for spans
        distance_function="dot_product",
    ),
    candidates=CandidateGenerator("zelda"),
)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. run fine-tuning
trainer.fine_tune(
    "resources/taggers/zelda-nel",
    learning_rate=5.0e-6,
    mini_batch_size=4,
    mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
)
```

As you can see, we use [`TransformerWordEmbeddings`](#flair.embeddings.token.TransformerWordEmbeddings) based on [bert-base-uncased](https://huggingface.co/bert-base-uncased) embeddings. We enable fine-tuning and set `use_context` to True.
We use [Prototypical Networks](https://arxiv.org/abs/1703.05175), to generalize bettwer in the few-shot classification setting.
Also, we set a `CandidateGenerator` in the [`SpanClassifier`](#flair.models.SpanClassifier).
This way we limit the classification to a small set of candidates that are chosen depending on the text of the respective span.

## Loading a ColumnCorpus

In cases you want to train over a custom named entity linking dataset, you can load them with the [`ColumnCorpus`](#flair.datasets.sequence_labeling.ColumnCorpus) object.
Most sequence labeling datasets in NLP use some sort of column format in which each line is a word and each column is
one level of linguistic annotation. See for instance this sentence:

```console
George B-George_Washington
Washington I-George_Washington
went O
to O
Washington B-Washington_D_C

Sam B-Sam_Houston
Houston I-Sam_Houston
stayed O
home O
```

The first column is the word itself, the second BIO-annotated tags used to specify the spans that will be classified. To read such a
dataset, define the column structure as a dictionary and instantiate a [`ColumnCorpus`](#flair.datasets.sequence_labeling.ColumnCorpus).

```python
from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
columns = {0: "text", 1: "nel"}

# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns)
```

## Next 

Next, learn [how to train a multitask model in Flair](how-to-train-multitask-model.md).