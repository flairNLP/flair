# How to load a custom dataset

This part of the tutorial shows how you can load a corpus for training a model. 

## loading a ColumnCorpus

In cases you want to train over a sequence labeling dataset that is not in the above list, you can load them with the [`ColumnCorpus`](#flair.datasets.sequence_labeling.ColumnCorpus) object.
Most sequence labeling datasets in NLP use some sort of column format in which each line is a word and each column is
one level of linguistic annotation. See for instance this sentence:

```console
George N B-PER
Washington N I-PER
went V O
to P O
Washington N B-LOC

Sam N B-PER
Houston N I-PER
stayed V O
home N O
```

The first column is the word itself, the second coarse PoS tags, and the third BIO-annotated NER tags. Empty line separates sentences. To read such a
dataset, define the column structure as a dictionary and instantiate a [`ColumnCorpus`](#flair.datasets.sequence_labeling.ColumnCorpus).

```python
from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
columns = {0: 'text', 1: 'pos', 2: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')

```

This gives you a [`Corpus`](#flair.data.Corpus) object that contains the train, dev and test splits, each has a list of [`Sentence`](#flair.data.Sentence).
So, to check how many sentences there are in the training split, do

```python
len(corpus.train)
```

You can also access a sentence and check out annotations. Lets assume that the training split is
read from the example above, then executing these commands

```python
print(corpus.train[0].to_tagged_string('ner'))
print(corpus.train[1].to_tagged_string('pos'))
```

will print the sentences with different layers of annotation:

```console
George <B-PER> Washington <I-PER> went to Washington <B-LOC> .

Sam <N> Houston <N> stayed <V> home <N>
```

## Using a Hugging Face DatasetDict

You can build a [`Corpus`](#flair.data.Corpus) directly from the splits of a Hugging Face
`DatasetDict`. Install `datasets` to run this example with CoNLL-2003. It keeps the original
tokens, attaches part-of-speech tags to individual tokens, and converts BIO entity tags into
labeled spans. The dataset creates a Flair sentence when you access a row.

```python
from datasets import load_dataset
from flair.data import Corpus, FlairDataset, Sentence, get_spans_from_bio


class HuggingFaceSentences(FlairDataset):
    def __init__(self, split, to_sentence):
        self.split = split
        self.features = split.features
        self.to_sentence = to_sentence

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        return self.to_sentence(self.split[index], self.features)

    def is_in_memory(self):
        return False


def tagged_sentence(row, features):
    sentence = Sentence(row["tokens"])
    for token, tag_id in zip(sentence, row["pos_tags"]):
        token.add_label("pos", features["pos_tags"].feature.int2str(tag_id))
    tags = [features["ner_tags"].feature.int2str(tag_id) for tag_id in row["ner_tags"]]
    for indices, _, entity_type in get_spans_from_bio(tags):
        sentence[indices[0] : indices[-1] + 1].add_label("ner", entity_type)
    return sentence


data = load_dataset("tomaarsen/conll2003")
corpus = Corpus(
    train=HuggingFaceSentences(data["train"], tagged_sentence),
    dev=HuggingFaceSentences(data["validation"], tagged_sentence),
    test=HuggingFaceSentences(data["test"], tagged_sentence),
)

print(corpus.train[0].get_spans("ner"))
```

For text classification, pass `classified_sentence` as the conversion function for each
split. This example attaches the class to the whole sentence for a dataset with a `text`
field and an integer `label` backed by a Hugging Face `ClassLabel`.

```python
def classified_sentence(row, features):
    sentence = Sentence(row["text"])
    sentence.add_label("topic", features["label"].int2str(row["label"]))
    return sentence
```

## Reading a text classification dataset

If you want to use your own text classification dataset, there are currently two methods to go about this:
load specified text and labels from a simple CSV file or format your data to the
[FastText format](https://fasttext.cc/docs/en/supervised-tutorial.html).

### Load from simple CSV file

Many text classification datasets are distributed as simple CSV files in which each row corresponds to a data point and
columns correspond to text, labels, and other metadata.  You can load a CSV format classification dataset using
[`CSVClassificationCorpus`](#flair.datasets.document_classification.CSVClassificationCorpus) by passing in a column format (like in [`ColumnCorpus`](#flair.datasets.sequence_labeling.ColumnCorpus) above).  This column format indicates
which column(s) in the CSV holds the text and which field(s) the label(s). By default, Python's CSV library assumes that
your files are in Excel CSV format, but [you can specify additional parameters](https://docs.python.org/3/library/csv.html#csv-fmt-params)
if you use custom delimiters or quote characters.

Note: You will need to save your split CSV data files in the `data_folder` path with each file titled appropriately i.e.
`train.csv` `test.csv` `dev.csv`.   This is because the corpus initializers will automatically search for the train,
dev, test splits in a folder.

```python
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data'

# column format indicating which columns hold the text and label(s)
column_name_map = {4: "text", 1: "label_topic", 2: "label_subtopic"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         skip_header=True,
                                         delimiter='\t',    # tab-separated files
)
```


### FastText format
If using [`CSVClassificationCorpus`](#flair.datasets.document_classification.CSVClassificationCorpus) is not practical, you may format your data to the FastText format, in which each line in the file represents a text document. A document can have one or multiple labels that are defined at the beginning of the line starting with the prefix `__label__`. This looks like this:

```bash
__label__<label_1> <text>
__label__<label_1> __label__<label_2> <text>
```

As previously mentioned, to create a [`Corpus`](#flair.data.Corpus) for a text classification task, you need to have three files (train, dev, and test) in the
above format located in one folder. This data folder structure could, for example, look like this for the IMDB task:
```text
/resources/tasks/imdb/train.txt
/resources/tasks/imdb/dev.txt
/resources/tasks/imdb/test.txt
```
Now create a [`CSVClassificationCorpus`](#flair.datasets.document_classification.CSVClassificationCorpus) by pointing to this folder (`/resources/tasks/imdb`).
Thereby, each line in a file is converted to a [`Sentence`](#flair.data.Sentence) object annotated with the labels.

```{important}
A text in a line can have multiple sentences. Thus, a [`Sentence`](#flair.data.Sentence) object can actually consist of multiple sentences.
```

```python
from flair.data import Corpus
from flair.datasets import ClassificationCorpus

# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# load corpus containing training, test and dev data
corpus: Corpus = ClassificationCorpus(data_folder,
                                      test_file='test.txt',
                                      dev_file='dev.txt',
                                      train_file='train.txt',
                                      label_type='topic',
                                      )
```

Note again that our corpus initializers have methods to automatically look for train, dev and test splits in a folder. So in
most cases you don't need to specify the file names yourself. Often, this is enough:

```python
# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# load corpus by pointing to folder. Train, dev and test gets identified automatically.
corpus: Corpus = ClassificationCorpus(data_folder,
                                      label_type='topic',
                                      )
```

Since the FastText format does not have columns, you must manually define a name for the annotations. In this
example we chose `label_type='topic'` to denote that we are loading a corpus with topic labels.



## Next 

Next, learn [how to train a sequence tagger](how-to-train-sequence-tagger.md).
