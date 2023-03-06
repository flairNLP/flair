# Tutorial 4.1: Loading a Prepared Corpus

This part of the tutorial shows how you can load a corpus for training a model. 
We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this
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
hard drive. It then reads the train, test and dev splits into the `Corpus` which it returns. Check the length of
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
# get the first Sentence in the training split
sentence = corpus.test[0]

# print with all annotations
print(sentence)

# print only with POS annotations (better readability)
print(sentence.to_tagged_string('pos'))
```

The sentence is fully tagged with syntactic and morphological information. With the latter line,
you print out only the POS tags:

```console
Sentence: "What if Google Morphed Into GoogleOS ?" → ["What"/WP, "if"/IN, "Google"/NNP, "Morphed"/VBD, "Into"/IN, "GoogleOS"/NNP, "?"/.]
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

### Creating label dictionaries

For many learning tasks you need to create a "dictionary" that contains all the labels you want to predict.
You can generate this dictionary directly out of the `Corpus` by calling the method `make_label_dictionary`
and passing the desired `label_type`.

For instance, the UD_ENGLISH corpus instantiated above has multiple layers of annotation like regular
POS tags ('pos'), universal POS tags ('upos'), morphological tags ('tense', 'number'..) and so on.
Create label dictionaries for universal POS tags by passing `label_type='upos'` like this:

```python
# create label dictionary for a Universal Part-of-Speech tagging task
upos_dictionary = corpus.make_label_dictionary(label_type='upos')

# print dictionary
print(upos_dictionary)
```

This will print out the created dictionary:

```console
Dictionary with 17 tags: PROPN, PUNCT, ADJ, NOUN, VERB, DET, ADP, AUX, PRON, PART, SCONJ, NUM, ADV, CCONJ, X, INTJ, SYM
```

#### Dictionaries for other label types

If you don't know the label types in a corpus, just call `make_label_dictionary` with
any random label name (e.g. `corpus.make_label_dictionary(label_type='abcd')`). This will print
out statistics on all label types in the corpus:

```console
The corpus contains the following label types: 'lemma' (in 12543 sentences), 'upos' (in 12543 sentences), 'pos' (in 12543 sentences), 'dependency' (in 12543 sentences), 'number' (in 12036 sentences), 'verbform' (in 10122 sentences), 'prontype' (in 9744 sentences), 'person' (in 9381 sentences), 'mood' (in 8911 sentences), 'tense' (in 8747 sentences), 'degree' (in 7148 sentences), 'definite' (in 6851 sentences), 'case' (in 6486 sentences), 'gender' (in 2824 sentences), 'numtype' (in 2771 sentences), 'poss' (in 2516 sentences), 'voice' (in 1085 sentences), 'typo' (in 399 sentences), 'extpos' (in 185 sentences), 'abbr' (in 168 sentences), 'reflex' (in 98 sentences), 'style' (in 31 sentences), 'foreign' (in 5 sentences)
```

This means that you can create dictionaries for any of these label types for the UD_ENGLISH corpus. Let's create dictionaries for regular part of speech tags
and a morphological number tagging task:

```python
# create label dictionary for a regular POS tagging task
pos_dictionary = corpus.make_label_dictionary(label_type='pos')

# create label dictionary for a morphological number tagging task
tense_dictionary = corpus.make_label_dictionary(label_type='number')
```

If you print these dictionaries, you will find that the POS dictionary contains 50 tags and the number dictionary only 2 for this corpus (singular and plural).


#### Dictionaries for other corpora types

The method `make_label_dictionary` can be used for any corpus, including text classification corpora:

```python
# create label dictionary for a text classification task
corpus = flair.datasets.TREC_6()
corpus.make_label_dictionary('question_class')
```

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

## Datasets included in Flair

Flair supports many datasets out of the box. It automatically downloads and sets up the
data the first time you call the corresponding constructor ID.

The following datasets are supported:

#### Named Entity Recognition

| Object                      | Languages                | Description                                                                                                                                                                                       |
|-----------------------------|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 'CONLL_03'                  | English                  | [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER (requires manual download)                                                                                                 |
| 'CONLL_03_GERMAN'           | German                   | [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER (requires manual download)                                                                                                 |
| 'CONLL_03_DUTCH'            | Dutch                    | [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER                                                                                                                            |
| 'CONLL_03_SPANISH'          | Spanish                  | [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER                                                                                                                            |
| 'ONTONOTES'                 | Arabic, English, Chinese | [Ontonotes](https://paperswithcode.com/dataset/ontonotes-5-0/) 18-class NER                                                                                                                          |
| 'FEWNERD'                   | English                  | [FewNERD](https://ningding97.github.io/fewnerd/) 66-class NER                                                                                                                             |
| 'NER_ARABIC_ANER'           | Arabic                   | [Arabic Named Entity Recognition Corpus](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp) 4-class NER                                                                                       |
| 'NER_ARABIC_AQMAR'          | Arabic                   | [American and Qatari Modeling of Arabic](http://www.cs.cmu.edu/~ark/AQMAR/) 4-class NER (modified)                                                                                                |
| 'NER_BASQUE'                | Basque                   | [NER dataset for Basque](http://ixa2.si.ehu.eus/eiec/)                                                                                                                                            |
| 'NER_CHINESE_WEIBO'         | Chinese                  | [Weibo NER corpus](https://paperswithcode.com/sota/chinese-named-entity-recognition-on-weibo-ner/).                                                                                               |
| 'NER_DANISH_DANE'           | Danish                   | [DaNE dataset](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank)                                                                                    |
| 'NER_ENGLISH_MOVIE_SIMPLE'  | English                  | [NER dataset for movie reviews](https://groups.csail.mit.edu/sls/downloads/movie/) - simple NER                                                                                                   |
| 'NER_ENGLISH_MOVIE_COMPLEX' | English                  | [NER dataset for movie reviews](https://groups.csail.mit.edu/sls/downloads/movie/) - complex NER                                                                                                  |
| 'NER_ENGLISH_PERSON'        | English                  | [PERSON_NER](https://github.com/das-sudeshna/genid) NER with person names                                                                                                                         |
| 'NER_ENGLISH_RESTAURANT'    | English                  | [NER dataset for restaurant reviews](https://groups.csail.mit.edu/sls/downloads/restaurant/)                                                                                                      |
| 'NER_ENGLISH_SEC_FILLINGS'  | English                  | [SEC-fillings](https://github.com/juand-r/entity-recognition-datasets) with 4-class NER labels from (Alvarado et al, 2015)[https://aclanthology.org/U15-1010/] here                               |
| 'NER_ENGLISH_STACKOVERFLOW' | English                  | NER on StackOverflow posts                                                                                                                                                                        |
| 'NER_ENGLISH_TWITTER'       | English                  | [Twitter NER dataset](https://github.com/aritter/twitter_nlp/)                                                                                                                                    |
| 'NER_ENGLISH_WIKIGOLD'      | English                  | [Wikigold](https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold) a manually annotated collection of Wikipedia text                                                    |
| 'NER_ENGLISH_WNUT_2020'     | English                  | [WNUT-20](https://github.com/jeniyat/WNUT_2020_NER) named entity extraction                                                                                                                       |
| 'NER_ENGLISH_WEBPAGES'      | English                  | 4-class NER on web pages from [Ratinov and Roth (2009)](https://aclanthology.org/W09-1119/)                                                                                                       |
| 'NER_FINNISH'               | Finnish                  | [Finer-data](https://github.com/mpsilfve/finer-data)                                                                                                                                              |
| 'NER_GERMAN_BIOFID'         | German                   | [CoNLL-03](https://www.aclweb.org/anthology/K19-1081/) Biodiversity literature NER                                                                                                                |
| 'NER_GERMAN_EUROPARL'       | German                   | [German Europarl dataset](https://nlpado.de/~sebastian/software/ner_german.shtml) NER in German EU parliament speeches                                                                            |
| 'NER_GERMAN_GERMEVAL'       | German                   | [GermEval 14 NER](https://sites.google.com/site/germeval2014ner/data/) corpus                                                                                                                     |
| 'NER_GERMAN_LEGAL'          | German                   | [Legal Entity Recognition](https://github.com/elenanereiss/Legal-Entity-Recognition) NER in German Legal Documents                                                                                |
| 'NER_GERMAN_POLITICS'       | German                   | [NEMGP](https://www.thomas-zastrow.de/nlp/) corpus                                                                                                                                                |
| 'NER_HIPE_2022'             | 5 languages              | NER dataset for [HIPE-2022](https://hipe-eval.github.io/HIPE-2022/) (Identifying Historical People, Places and other Entities)                                                                    |
| 'NER_HUNGARIAN'             | Hungarian                | NER on Hungarian business news                                                                                                                                                                    |
| 'NER_ICELANDIC'             | Icelandic                | NER on Icelandic                                                                                                                                                                                  |
| 'NER_JAPANESE'              | Japanese                 | [Japanese NER](https://github.com/Hironsan/IOB2Corpus) dataset automatically generated from Wikipedia                                                                                             |
| 'NER_MASAKHANE'             | 10 languages             | [MasakhaNER: Named Entity Recognition for African Languages](https://github.com/masakhane-io/masakhane-ner) corpora                                                                               |
| 'NER_SWEDISH'               | Swedish                  | [Swedish Spraakbanken NER](https://github.com/klintan/swedish-ner-corpus/) 4-class NER                                                                                                            |
| 'NER_TURKU'                 | Finnish                  | [TURKU_NER](https://github.com/TurkuNLP/turku-ner-corpus) NER corpus created by the Turku NLP Group, University of Turku, Finland                                                                 |
| 'NER_UKRAINIAN'             | Ukrainian                | [lang-uk](https://github.com/lang-uk/flair-ner) NER corpus created by the [Lang-uk community](https://lang.org.ua/en/)                                                                            |
| 'NER_MULTI_WIKIANN'         | 282 languages            | Gigantic [corpus for cross-lingual NER derived from Wikipedia](https://elisa-ie.github.io/wikiann/).                                                                                              |
| 'NER_MULTI_WIKINER'         | 8 languages              | [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia (English, German, French, Italian, Spanish, Portuguese, Polish, Russian) |
| 'NER_MULTI_XTREME'          | 176 languages            | [Xtreme](https://github.com/google-research/xtreme) corpus by Google Research for cross-lingual NER consisting of datasets of a total of 176 languages                                            |
| 'WNUT_17'                   | English                  | [WNUT-17](https://noisy-text.github.io/2017/files/) emerging entity detection                                                                                                                     |

#### Biomedical Named Entity Recognition

We support 31 biomedical NER datasets, listed [here](HUNFLAIR_CORPORA.md).

#### Entity Linking
| Object | Languages | Description |
| -------------    | ------------- |-------------  |
| 'NEL_ENGLISH_AIDA' | English  |  [AIDA CoNLL-YAGO Entity Linking corpus](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads) on the CoNLL-03 corpus |
| 'NEL_ENGLISH_AQUAINT' | English  | Aquaint Entity Linking corpus introduced in [Milne and Witten (2008)](https://www.cms.waikato.ac.nz/~ihw/papers/08-DNM-IHW-LearningToLinkWithWikipedia.pdf) |
| 'NEL_ENGLISH_IITB' | English  | ITTB Entity Linking corpus introduced in [Sayali et al. (2009)](https://dl.acm.org/doi/10.1145/1557019.1557073) |
| 'NEL_ENGLISH_REDDIT' | English  | Reddit Entity Linking corpus introduced in [Botzer et al. (2021)](https://arxiv.org/abs/2101.01228v2) (only gold annotations)|
| 'NEL_ENGLISH_TWEEKI' | English  | ITTB Entity Linking corpus introduced in [Harandizadeh and Singh (2020)](https://aclanthology.org/2020.wnut-1.29.pdf) |
| 'NEL_GERMAN_HIPE' | German  | [HIPE](https://impresso.github.io/CLEF-HIPE-2020/) Entity Linking corpus for historical German as a [sentence-segmented version](https://github.com/stefan-it/clef-hipe) |

#### Relation Extraction
| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'RE_ENGLISH_CONLL04' | English  |  [CoNLL-04](https://github.com/bekou/multihead_joint_entity_relation_extraction/tree/master/data/CoNLL04) Relation Extraction |
| 'RE_ENGLISH_SEMEVAL2010' | English  |  [SemEval-2010 Task 8](https://aclanthology.org/S10-1006.pdf) on Multi-Way Classification of Semantic Relations Between Pairs of Nominals |
| 'RE_ENGLISH_TACRED' | English  |  [TAC Relation Extraction Dataset](https://nlp.stanford.edu/projects/tacred/) with 41 relations (download required) |
| 'RE_ENGLISH_DRUGPROT' | English  |  [DrugProt corpus: Biocreative VII Track 1](https://zenodo.org/record/5119892#.YSdSaVuxU5k/) - drug and chemical-protein interactions |


#### GLUE Benchmark
| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'GLUE_COLA' | English | The Corpus of Linguistic Acceptability from GLUE benchmark |
| 'GLUE_MNLI' | English | The Multi-Genre Natural Language Inference Corpus from the GLUE benchmark |
| 'GLUE_RTE' | English | The RTE task from the GLUE benchmark |
| 'GLUE_QNLI' | English | The Stanford Question Answering Dataset formated as NLI task from the GLUE benchmark |
| 'GLUE_WNLI' | English | The Winograd Schema Challenge formated as NLI task from the GLUE benchmark |
| 'GLUE_MRPC' | English | The MRPC task from GLUE benchmark |
| 'GLUE_QQP' | English | The Quora Question Pairs dataset where the task is to determine whether a pair of questions are semantically equivalent |
| 'SUPERGLUE_RTE' | English | The RTE task from the SuperGLUE benchmark |

#### Universal Proposition Banks

We also support loading the [Universal Proposition Banks](https://github.com/System-T/UniversalPropositions)
for the purpose of training multilingual frame detection systems.

| Object | Languages | Description |
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

| Object             | Languages         | Description                                                                                                                         |
|--------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| 'UD_ARABIC'        | Arabic            | Universal Dependency Treebank for [Arabic](https://github.com/UniversalDependencies/UD_Arabic-PADT)                                 |
| 'UD_BASQUE'        | Basque            | Universal Dependency Treebank for [Basque](https://github.com/UniversalDependencies/UD_Basque-BDT)                                  |
| 'UD_BULGARIAN'     | Bulgarian         | Universal Dependency Treebank for [Bulgarian](https://github.com/UniversalDependencies/UD_Bulgarian-BTB)                            
| 'UD_CATALAN',      | Catalan           | Universal Dependency Treebank for [Catalan](https://github.com/UniversalDependencies/UD_Catalan-AnCora)                             |
| 'UD_CHINESE'       | Chinese           | Universal Dependency Treebank for [Chinese](https://github.com/UniversalDependencies/UD_Chinese-GSD)                                |
| 'UD_CHINESE_KYOTO' | Classical Chinese | Universal Dependency Treebank for Classical [Chinese](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/dev) |
| 'UD_CROATIAN'      | Croatian          | Universal Dependency Treebank for [Croatian](https://github.com/UniversalDependencies/UD_Croatian-SET)                              |
| 'UD_CZECH'         | Czech             | Very large Universal Dependency Treebank for [Czech](https://github.com/UniversalDependencies/UD_Czech-PDT)                         |
| 'UD_DANISH'        | Danish            | Universal Dependency Treebank for [Danish](https://github.com/UniversalDependencies/UD_Danish-DDT)                                  |
| 'UD_DUTCH'         | Dutch             | Universal Dependency Treebank for [Dutch](https://github.com/UniversalDependencies/UD_Dutch-Alpino)                                 |
| 'UD_ENGLISH'       | English           | Universal Dependency Treebank for [English](https://github.com/UniversalDependencies/UD_English-EWT)                                |
| 'UD_FINNISH'       | Finnish           | Universal Dependency Treebank for [Finnish](https://github.com/UniversalDependencies/UD_Finnish-TDT)                                |
| 'UD_FRENCH'        | French            | Universal Dependency Treebank for [French](https://github.com/UniversalDependencies/UD_French-GSD)                                  |
| 'UD_GERMAN'        | German            | Universal Dependency Treebank for [German](https://github.com/UniversalDependencies/UD_German-GSD)                                  |
| 'UD_GERMAN-HDT'    | German            | Very large Universal Dependency Treebank for [German](https://github.com/UniversalDependencies/UD_German-HDT)                       |
| 'UD_HEBREW'        | Hebrew            | Universal Dependency Treebank for [Hebrew](https://github.com/UniversalDependencies/UD_Hebrew-HTB)                                  |
| 'UD_HINDI'         | Hindi             | Universal Dependency Treebank for [Hindi](https://github.com/UniversalDependencies/UD_Hindi-HDTB)                                   |
| 'UD_INDONESIAN'    | Indonesian        | Universal Dependency Treebank for [Indonesian](https://github.com/UniversalDependencies/UD_Indonesian-GSD)                          |
| 'UD_ITALIAN'       | Italian           | Universal Dependency Treebank for [Italian](https://github.com/UniversalDependencies/UD_Italian-ISDT)                               |
| 'UD_JAPANESE'      | Japanese          | Universal Dependency Treebank for [Japanese](https://github.com/UniversalDependencies/UD_Japanese-GSD)                              |
| 'UD_KOREAN'        | Korean            | Universal Dependency Treebank for [Korean](https://github.com/UniversalDependencies/UD_Korean-Kaist)                                |
| 'UD_NORWEGIAN',    | Norwegian         | Universal Dependency Treebank for [Norwegian](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal)                        |
| 'UD_PERSIAN'       | Persian / Farsi   | Universal Dependency Treebank for [Persian](https://github.com/UniversalDependencies/UD_Persian-Seraji)                             |
| 'UD_POLISH'        | Polish            | Universal Dependency Treebank for [Polish](https://github.com/UniversalDependencies/UD_Polish-LFG)                                  |
| 'UD_PORTUGUESE'    | Portuguese        | Universal Dependency Treebank for [Portuguese](https://github.com/UniversalDependencies/UD_Portuguese-Bosque)                       |
| 'UD_ROMANIAN'      | Romanian          | Universal Dependency Treebank for [Romanian](https://github.com/UniversalDependencies/UD_Romanian-RRT)                              |
| 'UD_RUSSIAN'       | Russian           | Universal Dependency Treebank for [Russian](https://github.com/UniversalDependencies/UD_Russian-SynTagRus)                          |
| 'UD_SERBIAN'       | Serbian           | Universal Dependency Treebank for [Serbian](https://github.com/UniversalDependencies/UD_Serbian-SET)                                |
| 'UD_SLOVAK'        | Slovak            | Universal Dependency Treebank for [Slovak](https://github.com/UniversalDependencies/UD_Slovak-SNK)                                  |
| 'UD_SLOVENIAN'     | Slovenian         | Universal Dependency Treebank for [Slovenian](https://github.com/UniversalDependencies/UD_Slovenian-SSJ)                            |
| 'UD_SPANISH'       | Spanish           | Universal Dependency Treebank for [Spanish](https://github.com/UniversalDependencies/UD_Spanish-GSD)                                |
| 'UD_SWEDISH'       | Swedish           | Universal Dependency Treebank for [Swedish](https://github.com/UniversalDependencies/UD_Swedish-Talbanken)                          |
| 'UD_TURKISH'       | Turkish           | Universal Dependency Treebank for [Tturkish](https://github.com/UniversalDependencies/UD_Turkish-IMST)                              |
| 'UD_UKRAINIAN'     | Ukrainian         | Universal Dependency Treebank for [Ukrainian](https://github.com/UniversalDependencies/UD_Ukrainian-IU)                             |

#### Text Classification
| Object | Languages | Description |
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
| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'WASSA_ANGER' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (anger) |
| 'WASSA_FEAR' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (fear) |
| 'WASSA_JOY' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (joy) |
| 'WASSA_SADNESS' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (sadness) |

#### Other Sequence Labeling

| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'CONLL_2000' | English  | Syntactic chunking with [CoNLL-2000]((https://www.clips.uantwerpen.be/conll2000/chunking/))  |
| 'BIOSCOPE' | English  | Negation and speculation scoping wih [BioScope](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S11-S9/) biomedical texts annotated for uncertainty, negation and their scopes |
| 'KEYPHRASE_INSPEC' | English | Keyphrase dectection with [INSPEC](https://www.aclweb.org/anthology/W03-1028) original corpus (2000 docs) from INSPEC database, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |
| 'KEYPHRASE_SEMEVAL2017' | English | Keyphrase dectection with [SEMEVAL2017](https://arxiv.org/abs/1704.02853) dataset (500 docs) from ScienceDirect, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |
| 'KEYPHRASE_SEMEVAL2010' | English | Keyphrase dectection with [SEMEVAL2010](https://www.aclweb.org/anthology/S10-1004/) dataset (~250 docs) from ACM Digital Library, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |

#### Experimental: Similarity Learning
| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'FeideggerCorpus' | German |  [Feidegger](https://github.com/zalandoresearch/feidegger/) dataset fashion images and German-language descriptions  |
| 'OpusParallelCorpus' | Any language pair | Parallel corpora of the [OPUS](http://opus.nlpl.eu/) project, currently supports only Tatoeba corpus |

## Next

Learn [how to load your own dataset](/resources/docs/TUTORIAL_CORPUS_CUSTOM.md) if
you want to use a dataset that is not prepared in Flair.

Alternatively, jump directly into the sections that show how to 
[train your own **Sequence Labeling** model](/resources/docs/TUTORIAL_TRAINING_SEQUENCE_LABELER.md) 
or how to [train your own **Text Classification** model](/resources/docs/TUTORIAL_TRAINING_TEXT_CLASSIFIER.md)

