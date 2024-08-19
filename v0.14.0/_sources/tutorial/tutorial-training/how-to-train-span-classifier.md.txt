# Train a span classifier

Span Classification models are used to model problems such as entity linking, where you already have extracted some
relevant spans
within the {term}`Sentence` and want to predict some more fine-grained labels.

This tutorial section show you how to train models using the [Span Classifier](#flair.models.SpanClassifier) in Flair.

## Training an entity linker (NEL) model with transformers

For a state-of-the-art NER sytem you should fine-tune transformer embeddings, and use full document context
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

## constructing a dataset in memory

If you have a pipeline where you need to construct your dataset from a different data source,
you can always construct a [Corpus](#flair.data.Corpus) with [FlairDatapointDataset](#flair.datasets.base.FlairDatapointDataset) by hand.
Let's assume you create a function `create_datapoint(datapoint) -> Sentence` that looks somewhat like this:
```python
from flair.data import Sentence

def create_sentence(datapoint) -> Sentence:
    tokens = ...  # calculate the tokens from your internal data structure (e.g. pandas dataframe or json dictionary)
    spans = ...  # create a list of tuples (start_token, end_token, label) from your data structure
    sentence = Sentence(tokens)
    for (start, end, label) in spans:
        sentence[start:end+1].add_label("nel", label)
```
Then you can use this function to create a full dataset:
```python
from flair.data import Corpus
from flair.datasets import FlairDatapointDataset

def construct_corpus(data):
    return Corpus(
        train=FlairDatapointDataset([create_sentence(datapoint for datapoint in data["train"])]),
        dev=FlairDatapointDataset([create_sentence(datapoint for datapoint in data["dev"])]),
        test=FlairDatapointDataset([create_sentence(datapoint for datapoint in data["test"])]),
    )
```
And use this to construct a corpus instead of loading a dataset.


## Combining NEL with Mention Detection

often, you don't just want to use a Named Entity Linking model alone, but combine it with a Mention Detection or Named Entity Recognition model.
For this, you can use a [Multitask Model](#flair.models.MultitaskModel) to combine a [SequenceTagger](#flair.models.SequenceTagger) and a [Span Classifier](#flair.models.SpanClassifier).

```python
from flair.datasets import NER_MULTI_WIKINER, ZELDA
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger, SpanClassifier
from flair.models.entity_linker_model import CandidateGenerator
from flair.trainers import ModelTrainer
from flair.nn import PrototypicalDecoder
from flair.nn.multitask import make_multitask_model_and_corpus

# 1. get the corpus
ner_corpus = NER_MULTI_WIKINER()
nel_corpus = ZELDA(column_format={0: "text", 2: "nel"})  # need to set the label type to be the same as the ner one

# --- Embeddings that are shared by both models --- #
shared_embeddings = TransformerWordEmbeddings("distilbert-base-uncased", fine_tune=True)

ner_label_dict = ner_corpus.make_label_dictionary("ner", add_unk=False)

ner_model = SequenceTagger(
    embeddings=shared_embeddings,
    tag_dictionary=ner_label_dict,
    tag_type="ner",
    use_rnn=False,
    use_crf=False,
    reproject_embeddings=False,
)


nel_label_dict = nel_corpus.make_label_dictionary("nel", add_unk=True)

nel_model = SpanClassifier(
    embeddings=shared_embeddings,
    label_dictionary=nel_label_dict,
    label_type="nel",
    span_label_type="ner",
    decoder=PrototypicalDecoder(
        num_prototypes=len(nel_label_dict),
        embeddings_size=shared_embeddings.embedding_length * 2, # we use "first_last" encoding for spans
        distance_function="dot_product",
    ),
    candidates=CandidateGenerator("zelda"),
)


# -- Define mapping (which tagger should train on which model) -- #
multitask_model, multicorpus = make_multitask_model_and_corpus(
    [
        (ner_model, ner_corpus),
        (nel_model, nel_corpus),
    ]
)

# -- Create model trainer and train -- #
trainer = ModelTrainer(multitask_model, multicorpus)
trainer.fine_tune(f"resources/taggers/zelda_with_mention")
```

Here, the [make_multitask_model_and_corpus](#flair.nn.multitask.make_multitask_model_and_corpus) method creates a multitask model and a multicorpus where each sub-model is aligned for a sub-corpus.

### Multitask with aligned training data

If you have sentences with both annotations for ner and for nel, you might want to use a single corpus for both models.

This means, that you need to manually the `multitask_id` to the sentences:

```python
from flair.data import Sentence

def create_sentence(datapoint) -> Sentence:
    tokens = ...  # calculate the tokens from your internal data structure (e.g. pandas dataframe or json dictionary)
    spans = ...  # create a list of tuples (start_token, end_token, label) from your data structure
    sentence = Sentence(tokens)
    for (start, end, ner_label, nel_label) in spans:
        sentence[start:end+1].add_label("ner", ner_label)
        sentence[start:end+1].add_label("nel", nel_label)
    sentence.add_label("multitask_id", "Task_0")  # Task_0 for the NER model
    sentence.add_label("multitask_id", "Task_1")  # Task_1 for the NEL model
```

Then you can run the multitask training script with the exception that you create the [MultitaskModel](#flair.models.MultitaskModel) directly.

```python
...
multitask_model = MultitaskModel([ner_model, nel_model], use_all_tasks=True)
```

Here, setting `use_all_tasks=True` means that we will jointly train on both tasks at the same time. This will save a lot of training time,
as the shared embedding will be calculated once but used twice (once for each model).

