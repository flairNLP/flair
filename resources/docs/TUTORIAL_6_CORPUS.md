# Tutorial 6: Loading Training Data

This part of the tutorial shows how you can load a corpus for training a model. We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
library.


## The Corpus Object

The `Corpus` represents a dataset that you use to train a model. It consists of a list of `train` sentences,
a list of `dev` sentences, and a list of `test` sentences, which correspond to the training, validation and testing 
split during model training.

The following example snippet instantiates the Universal Dependency Treebank for English as a corpus object: 

```python
import flair.datasets
corpus = flair.datasets.UD_ENGLISH()
```

The first time you call this snippet, it triggers a download of the Universal Dependency Treebank for English onto your
hard drive. It then reads the train, test and dev splits into the `Corpus` corpus which it returns. Check the length of 
the three splits to see how many Sentences are there:

```python
# print the number of Sentences in the train split
print(len(corpus.train))

# print the number of Sentences in the test split
print(len(corpus.test))

# print the number of Sentences in the dev split
print(len(corpus.dev))
```

You can also access the Sentence objects in each split directly. For instance, let us look at the first Sentence in 
the training split of the English UD: 

```python
# print the first Sentence in the training split
print(corpus.test[0])
```
This prints: 
```console
Sentence: "What if Google Morphed Into GoogleOS ?" - 7 Tokens
```

The sentence is fully tagged with syntactic and morphological information. For instance, print the sentence with
PoS tags: 

```python
# print the first Sentence in the training split
print(corpus.test[0].to_tagged_string('pos'))
```

This should print: 

```console
What <WP> if <IN> Google <NNP> Morphed <VBD> Into <IN> GoogleOS <NNP> ? <.>
```

So the corpus is tagged and ready for training.

### Helper functions

A `Corpus` contains a bunch of useful helper functions.
For instance, you can downsample the data by calling `downsample()` and passing a ratio. So, if you normally get a 
corpus like this:

```python
import flair.datasets
corpus = flair.datasets.UD_ENGLISH()
```

then you can downsample the corpus, simply like this:

```python
import flair.datasets
downsampled_corpus = flair.datasets.UD_ENGLISH().downsample(0.1)
```

If you print both corpora, you see that the second one has been downsampled to 10% of the data.

```python
print("--- 1 Original ---")
print(corpus)

print("--- 2 Downsampled ---")
print(downsampled_corpus)
```

This should print:

```console
--- 1 Original ---
Corpus: 12543 train + 2002 dev + 2077 test sentences

--- 2 Downsampled ---
Corpus: 1255 train + 201 dev + 208 test sentences
```

For many learning tasks you need to create a "dictionary" that contains all the labels you want to predict.
You can generate this dictionary directly out of the `Corpus` by calling the method `make_label_dictionary`. 
If a corpus has multiple label types, you additionally need to specify for which label you want to produce the dictionary: 

```python
# create tag dictionary for a PoS task
corpus = flair.datasets.UD_ENGLISH()
print(corpus.make_label_dictionary('upos'))

# create tag dictionary for an NER task
corpus = flair.datasets.CONLL_03_DUTCH()
print(corpus.make_label_dictionary('ner'))

# create label dictionary for a text classification task
corpus = flair.datasets.TREC_6()
print(corpus.make_label_dictionary())
```

This should print out different label dictionaries for different datasets and tasks.

Another useful function is `obtain_statistics()` which returns you a python dictionary with useful statistics about your
dataset. Using it, for example, on the IMDB dataset like this

```python
import flair.datasets 
corpus = flair.datasets.TREC_6()
stats = corpus.obtain_statistics()
print(stats)
```
outputs detailed information on the dataset, each split, and the distribution of class labels.

### The MultiCorpus Object

If you want to train multiple tasks at once, you can use the `MultiCorpus` object.
To initiate the `MultiCorpus` you first need to create any number of `Corpus` objects. Afterwards, you can pass
a list of `Corpus` to the `MultiCorpus` object. For instance, the following snippet loads a combination corpus
consisting of the English, German and Dutch Universal Dependency Treebanks.

```python
english_corpus = flair.datasets.UD_ENGLISH()
german_corpus = flair.datasets.UD_GERMAN()
dutch_corpus = flair.datasets.UD_DUTCH()

# make a multi corpus consisting of three UDs
from flair.data import MultiCorpus
multi_corpus = MultiCorpus([english_corpus, german_corpus, dutch_corpus])
```

The `MultiCorpus` inherits from `Corpus`, so you can use it like any other corpus to train your models. 

## Prepared Datasets

Flair supports a growing list of prepared datasets out of the box. That is, it automatically downloads and sets up the 
data the first time you call the corresponding constructor ID. The following datasets are supported: 


#### Chunking

| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'CONLL_2000' | English  |  [CoNLL-2000]((https://www.clips.uantwerpen.be/conll2000/chunking/)) syntactic chunking |


#### Named Entity Recognition

| ID(s) | Languages | Description |
| -------------    | ------------- |------------- 
| 'ANER_CORP' | Arabic  |  [Arabic Named Entity Recognition Corpus](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp) 4-class NER |
| 'BIOFID' | German  |  [CoNLL-03](https://www.aclweb.org/anthology/K19-1081/) Biodiversity literature NER |
| 'BIOSCOPE' | English  |  [BioScope](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S11-S9/) biomedical texts annotated for uncertainty, negation and their scopes |
| 'BUSINESS_HUN' | Hungarian | NER on Hungarian business news |
| 'CONLL_03_DUTCH' | Dutch  |  [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER |
| 'CONLL_03_SPANISH' | Spanish  |  [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER |
| 'DANE' | Danish | [DaNE dataset](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank) | 
| 'EUROPARL_NER_GERMAN' | German | [German Europarl dataset](https://nlpado.de/~sebastian/software/ner_german.shtml) NER in German EU parliament speeches | 
| 'JAPANESE_NER' | Japanese | [https://github.com/Hironsan/IOB2Corpus] Japanese NER dataset automatically generated from Wikipedia |
| 'LER_GERMAN' | German | [Legal Entity Recognition](https://github.com/elenanereiss/Legal-Entity-Recognition) NER in German Legal Documents |
| 'LUGANDA_NER' | Luganda | [LUGANDA_NER](https://github.com/masakhane-io/masakhane-ner/tree/main/data/lug) |
| 'MIT_MOVIE_NER_SIMPLE' | English  |  [NER dataset for movie reviews](https://groups.csail.mit.edu/sls/downloads/movie/) - simple NER |
| 'MIT_MOVIE_NER_COMPLEX' | English  |  [NER dataset for movie reviews](https://groups.csail.mit.edu/sls/downloads/movie/) - complex NER |
| 'MIT_RESTAURANT_NER' | English  |  [NER dataset for restaurant reviews](https://groups.csail.mit.edu/sls/downloads/restaurant/) |
| 'NER_BASQUE' | Basque  |  [NER dataset for Basque](http://ixa2.si.ehu.eus/eiec/) |
| 'NER_FINNISH' | Finnish | [Finer-data](https://github.com/mpsilfve/finer-data) | 
| 'NER_SWEDISH' | Swedish | [Swedish Spraakbanken NER](https://github.com/klintan/swedish-ner-corpus/) 4-class NER |
| 'STACKOVERFLOW_NER' | English  | NER on StackOverflow posts |
| 'TURKU_NER' | Finnish | [TURKU_NER](https://github.com/TurkuNLP/turku-ner-corpus) NER corpus created by the Turku NLP Group, University of Turku, Finland |
| 'TWITTER_NER' | English  |  [Twitter NER dataset](https://github.com/aritter/twitter_nlp/) |
| 'WEIBO_NER' | Chinese  | [Weibo NER corpus](https://paperswithcode.com/sota/chinese-named-entity-recognition-on-weibo-ner/).  |
| 'WIKIANN' | 282 languages  | Gigantic [corpus for cross-lingual NER derived from Wikipedia](https://elisa-ie.github.io/wikiann/).  |
| 'WIKIGOLD_NER' | English  |  [Wikigold](https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold) a manually annotated collection of Wikipedia text |
| 'WIKINER_ENGLISH' | English  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_GERMAN'  | German  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_FRENCH'  | French  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_ITALIAN'  | Italian  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_SPANISH' | Spanish  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_PORTUGUESE' | Portuguese  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_POLISH' | Polish  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WIKINER_RUSSIAN'  | Russian  |  [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia |
| 'WNUT_17' | English  |  [WNUT-17](https://noisy-text.github.io/2017/files/) emerging entity detection |
| 'WNUT_2020_NER' | English  |  [WNUT-20](https://github.com/jeniyat/WNUT_2020_NER) named entity extraction |
| 'XTREME' | 176 languages  |  [Xtreme](https://github.com/google-research/xtreme) corpus by Google Research for cross-lingual NER consisting of datasets of a total of 176 languages |


#### Biomedical Named Entity Recognition

We support 31 biomedical NER datasets, listed [here](HUNFLAIR_CORPORA.md).


#### Universal Proposition Banks 

We now also support loading the [Universal Proposition Banks](https://github.com/System-T/UniversalPropositions)
for the purpose of training multilingual frame detection systems. 

| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'UP_CHINESE' | Chinese  |  Universal Propositions for [Chinese](https://github.com/System-T/UniversalPropositions/tree/master/UP_Chinese) |
| 'UP_ENGLISH'| English  |  Universal Propositions for [English](https://github.com/System-T/UniversalPropositions/tree/master/UP_English-EWT) |
| 'UP_FINNISH'| Finnish  |  Universal Propositions for [Finnish](https://github.com/System-T/UniversalPropositions/tree/master/UP_Finnish)
| 'UP_FRENCH'| French  |  Universal Propositions for [French](https://github.com/System-T/UniversalPropositions/tree/master/UP_French)
| 'UP_GERMAN'| German  |  Universal Propositions for [German](https://github.com/System-T/UniversalPropositions/tree/master/UP_German) |
| 'UP_ITALIAN', | Italian  |  Universal Propositions for [Italian](https://github.com/System-T/UniversalPropositions/tree/master/UP_Italian) |
| 'UP_SPANISH' | Spanish  |  Universal Propositions for [Spanish](https://github.com/System-T/UniversalPropositions/tree/master/UP_Spanish) |
| 'UP_SPANISH_ANCORA' | Spanish (Ancora Corpus)  |  Universal Propositions for [Spanish](https://github.com/System-T/UniversalPropositions/tree/master/UP_Spanish-AnCora) |


#### Universal Dependency Treebanks

| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'UD_ARABIC'| Arabic  |  Universal Dependency Treebank for [Arabic](https://github.com/UniversalDependencies/UD_Arabic-PADT) |
| 'UD_BASQUE'| Basque  |  Universal Dependency Treebank for [Basque](https://github.com/UniversalDependencies/UD_Basque-BDT) |
| 'UD_BULGARIAN'| Bulgarian  |  Universal Dependency Treebank for [Bulgarian](https://github.com/UniversalDependencies/UD_Bulgarian-BTB)
| 'UD_CATALAN', | Catalan  |  Universal Dependency Treebank for [Catalan](https://github.com/UniversalDependencies/UD_Catalan-AnCora) |
| 'UD_CHINESE' | Chinese  |  Universal Dependency Treebank for [Chinese](https://github.com/UniversalDependencies/UD_Chinese-GSD) |
| 'UD_CHINESE_KYOTO' | Classical Chinese  |  Universal Dependency Treebank for Classical [Chinese](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/dev) |
| 'UD_CROATIAN' | Croatian  |  Universal Dependency Treebank for [Croatian](https://github.com/UniversalDependencies/UD_Croatian-SET) |
| 'UD_CZECH' | Czech  |  Very large Universal Dependency Treebank for [Czech](https://github.com/UniversalDependencies/UD_Czech-PDT) |
| 'UD_DANISH' | Danish  |  Universal Dependency Treebank for [Danish](https://github.com/UniversalDependencies/UD_Danish-DDT) |
| 'UD_DUTCH' | Dutch  |  Universal Dependency Treebank for [Dutch](https://github.com/UniversalDependencies/UD_Dutch-Alpino) |
| 'UD_ENGLISH' | English  |  Universal Dependency Treebank for [English](https://github.com/UniversalDependencies/UD_English-EWT) |
| 'UD_FINNISH' | Finnish  |  Universal Dependency Treebank for [Finnish](https://github.com/UniversalDependencies/UD_Finnish-TDT) |
| 'UD_FRENCH' | French  |  Universal Dependency Treebank for [French](https://github.com/UniversalDependencies/UD_French-GSD) |
|'UD_GERMAN' | German  |  Universal Dependency Treebank for [German](https://github.com/UniversalDependencies/UD_German-GSD) |
|'UD_GERMAN-HDT' | German  |  Very large Universal Dependency Treebank for [German](https://github.com/UniversalDependencies/UD_German-HDT) |
|'UD_HEBREW' | Hebrew  |  Universal Dependency Treebank for [Hebrew](https://github.com/UniversalDependencies/UD_Hebrew-HTB) |
|'UD_HINDI' | Hindi  |  Universal Dependency Treebank for [Hindi](https://github.com/UniversalDependencies/UD_Hindi-HDTB) |
|'UD_INDONESIAN' | Indonesian  |  Universal Dependency Treebank for [Indonesian](https://github.com/UniversalDependencies/UD_Indonesian-GSD) |
| 'UD_ITALIAN' | Italian  |  Universal Dependency Treebank for [Italian](https://github.com/UniversalDependencies/UD_Italian-ISDT) |
| 'UD_JAPANESE'| Japanese  |  Universal Dependency Treebank for [Japanese](https://github.com/UniversalDependencies/UD_Japanese-GSD) |
|'UD_KOREAN' | Korean  |  Universal Dependency Treebank for [Korean](https://github.com/UniversalDependencies/UD_Korean-Kaist) |
| 'UD_NORWEGIAN',  | Norwegian  |  Universal Dependency Treebank for [Norwegian](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal) |
|  'UD_PERSIAN' | Persian / Farsi  |  Universal Dependency Treebank for [Persian](https://github.com/UniversalDependencies/UD_Persian-Seraji) |
| 'UD_POLISH'  |  Polish |  Universal Dependency Treebank for [Polish](https://github.com/UniversalDependencies/UD_Polish-LFG) |
|'UD_PORTUGUESE' | Portuguese  |  Universal Dependency Treebank for [Portuguese](https://github.com/UniversalDependencies/UD_Portuguese-Bosque) |
| 'UD_ROMANIAN' | Romanian  |  Universal Dependency Treebank for [Romanian](https://github.com/UniversalDependencies/UD_Romanian-RRT)  |
| 'UD_RUSSIAN' | Russian  |  Universal Dependency Treebank for [Russian](https://github.com/UniversalDependencies/UD_Russian-SynTagRus) |
| 'UD_SERBIAN' | Serbian  |  Universal Dependency Treebank for [Serbian](https://github.com/UniversalDependencies/UD_Serbian-SET)|
| 'UD_SLOVAK' | Slovak  |  Universal Dependency Treebank for [Slovak](https://github.com/UniversalDependencies/UD_Slovak-SNK) |
| 'UD_SLOVENIAN' | Slovenian  |  Universal Dependency Treebank for [Slovenian](https://github.com/UniversalDependencies/UD_Slovenian-SSJ) |
| 'UD_SPANISH'  | Spanish  |  Universal Dependency Treebank for [Spanish](https://github.com/UniversalDependencies/UD_Spanish-GSD) |
|  'UD_SWEDISH' | Swedish  |  Universal Dependency Treebank for [Swedish](https://github.com/UniversalDependencies/UD_Swedish-Talbanken) |
|  'UD_TURKISH' | Turkish  |  Universal Dependency Treebank for [Tturkish](https://github.com/UniversalDependencies/UD_Turkish-IMST) |

#### Keyword / Keyphrase Extraction
| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'INSPEC' | English |  [INSPEC](https://www.aclweb.org/anthology/W03-1028) original corpus (2000 docs) from INSPEC database, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |
| 'SEMEVAL2017' | English | The original [SEMEVAL2017](https://arxiv.org/abs/1704.02853) dataset (500 docs) from ScienceDirect, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |
| 'SEMEVAL2010' | English | The original [SEMEVAL2010](https://www.aclweb.org/anthology/S10-1004/) dataset (~250 docs) from ACM Digital Library, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |

#### Text Classification
| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'AMAZON_REVIEWS' | English |  [Amazon product reviews](https://nijianmo.github.io/amazon/index.html/) dataset with sentiment annotation |
| 'COMMUNICATIVE_FUNCTIONS' | English |  [Communicative functions](https://github.com/Alab-NII/FECFevalDataset) of sentences in scholarly papers |
| 'GERMEVAL_2018_OFFENSIVE_LANGUAGE' | German | Offensive language detection for German |
| 'GO_EMOTIONS' | English | [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) Reddit comments labeled with 27 emotions |
| 'IMDB' | English |  [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/) dataset of movie reviews with sentiment annotation  |
| 'NEWSGROUPS' | English | The popular [20 newsgroups](http://qwone.com/~jason/20Newsgroups/) classification dataset |
| 'YAHOO_ANSWERS' | English | The [10 largest main categories](https://course.fast.ai/datasets#nlp) from the Yahoo! Answers |
| 'SENTIMENT_140' | English | [Tweets dataset](http://help.sentiment140.com/for-students/) with sentiment annotation |
| 'SENTEVAL_CR' | English | Customer reviews dataset of [SentEval](https://github.com/facebookresearch/SentEval) with sentiment annotation |
| 'SENTEVAL_MR' | English | Movie reviews dataset of [SentEval](https://github.com/facebookresearch/SentEval) with sentiment annotation |
| 'SENTEVAL_SUBJ' | English | Subjectivity dataset of [SentEval](https://github.com/facebookresearch/SentEval) |
| 'SENTEVAL_MPQA' | English | Opinion-polarity dataset of [SentEval](https://github.com/facebookresearch/SentEval) with opinion-polarity annotation |
| 'SENTEVAL_SST_BINARY' | English | Stanford sentiment treebank dataset of of [SentEval](https://github.com/facebookresearch/SentEval) with sentiment annotation |
| 'SENTEVAL_SST_GRANULAR' | English | Stanford sentiment treebank dataset of of [SentEval](https://github.com/facebookresearch/SentEval) with fine-grained sentiment annotation |
| 'TREC_6', 'TREC_50' | English | The [TREC](http://cogcomp.org/Data/QA/QC/) question classification dataset |


#### Text Regression
| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'WASSA_ANGER' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (anger) |
| 'WASSA_FEAR' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (fear) |
| 'WASSA_JOY' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (joy) |
| 'WASSA_SADNESS' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (sadness) |

#### Recognizing Textual Entailment
| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'GLUE_RTE' | English | The RTE task from the GLUE benchmark |
| 'SUPERGLUE_RTE' | English | The RTE task from the SuperGLUE benchmark |


#### Experimental: Similarity Learning
| ID(s) | Languages | Description |
| -------------    | ------------- |------------- |
| 'FeideggerCorpus' | German |  [Feidegger](https://github.com/zalandoresearch/feidegger/) dataset fashion images and German-language descriptions  |
| 'OpusParallelCorpus' | Any language pair | Parallel corpora of the [OPUS](http://opus.nlpl.eu/) project, currently supports only Tatoeba corpus |


So to load the IMDB corpus for sentiment text classification, simply do:

```python
import flair.datasets
corpus = flair.datasets.IMDB()
```

This downloads and sets up everything you need to train your model. 


## Reading Your Own Sequence Labeling Dataset

In cases you want to train over a sequence labeling dataset that is not in the above list, you can load them with the ColumnCorpus object.
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
dataset, define the column structure as a dictionary and instantiate a `ColumnCorpus`.

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

This gives you a `Corpus` object that contains the train, dev and test splits, each has a list of `Sentence`.
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

## Reading a Text Classification Dataset

If you want to use your own text classification dataset, there are currently two methods to go about this: 
load specified text and labels from a simple CSV file or format your data to the 
[FastText format](https://fasttext.cc/docs/en/supervised-tutorial.html).

#### Load from simple CSV file

Many text classification datasets are distributed as simple CSV files in which each row corresponds to a data point and
columns correspond to text, labels, and other metadata.  You can load a CSV format classification dataset using 
`CSVClassificationCorpus` by passing in a column format (like in `ColumnCorpus` above).  This column format indicates
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


#### FastText Format
If using `CSVClassificationCorpus` is not practical, you may format your data to the FastText format, in which each line in the file represents a text document. A document can have one or multiple labels that are defined at the beginning of the line starting with the prefix `__label__`. This looks like this:

```bash
__label__<label_1> <text>
__label__<label_1> __label__<label_2> <text>
```

As previously mentioned, to create a `Corpus` for a text classification task, you need to have three files (train, dev, and test) in the 
above format located in one folder. This data folder structure could, for example, look like this for the IMDB task:
```text
/resources/tasks/imdb/train.txt
/resources/tasks/imdb/dev.txt
/resources/tasks/imdb/test.txt
```
Now create a `ClassificationCorpus` by pointing to this folder (`/resources/tasks/imdb`). 
Thereby, each line in a file is converted to a `Sentence` object annotated with the labels.

Attention: A text in a line can have multiple sentences. Thus, a `Sentence` object can actually consist of multiple
sentences.

```python
from flair.data import Corpus
from flair.datasets import ClassificationCorpus

# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# load corpus containing training, test and dev data
corpus: Corpus = ClassificationCorpus(data_folder,
                                      test_file='test.txt',
                                      dev_file='dev.txt',
                                      train_file='train.txt')
```

Note again that our corpus initializers have methods to automatically look for train, dev and test splits in a folder. So in 
most cases you don't need to specify the file names yourself. Often, this is enough: 

```python
# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# load corpus by pointing to folder. Train, dev and test gets identified automatically. 
corpus: Corpus = ClassificationCorpus(data_folder)
```

## Next

You can now look into [training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).
