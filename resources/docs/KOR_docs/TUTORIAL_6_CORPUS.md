# 튜토리얼 6 : 훈련 데이터 불러오기
이번 튜토리얼은 모델을 훈련하기 위해 말뭉치(corpus)를 로드하는 내용을 다룹니다. 
이번 튜토리얼은 여러분이 라이브러리의 [기본 유형](/resources/docs/KOR_docs/TUTORIAL_1_BASICS.md)에 익숙하다 가정하고 진행됩니다.

## 말뭉치 오브젝트
`corpus`는 모델을 훈련하는데 사용되는 데이터 세트입니다. 이는 모델 훈련 중 훈련, 검증 및 테스트 분할에 사용되는 문장들, 개발을 위한 문장 목록 및 테스트 문장 목록으로 구성됩니다.

다음 예제는 the Universal Dependency Treebank for English를 말뭉치 오브젝트로 초기화하는 코드입니다.
```python
import flair.datasets
corpus = flair.datasets.UD_ENGLISH()
```
위 코드를 처음 실행한다면 the Universal Dependency Treebank for English를 하드디스크에 다운로드합니다.
그 다음 훈련, 테스트, 개발을 위한 `corpus`로 분할합니다. 아래 코드를 통해 각각의 `corpus`에 몇개의 문장이 들어있는지 확인할 수 있습니다.
```python
# 몇개의 문장이 train split에 있는지 출력합니다.
print(len(corpus.train))

# 몇개의 문장이 test split에 있는지 출력합니다.
print(len(corpus.test))

# 몇개의 문장이 dev split에 있는지 출력합니다.
print(len(corpus.dev))
```

각 split의 객체에 직접 접근할 수 있습니다. 아래의 코드는 test split의 처음 문장을 출력합니다 :
```python
# training split의 처음 문장을 출력합니다.
print(corpus.test[0])
```
결과입니다 : 
```console
Sentence: "What if Google Morphed Into GoogleOS ?" - 7 Tokens
```

이 문장은 통사적, 형태학적 정보가 tag되어 있습니다. POS 태그를 사용해 문장을 인쇄해보겠습니다 :
```python
# 훈련중인 split에서 첫 번째 문장을 출력합니다.
print(corpus.test[0].to_tagged_string('pos'))
```
결과입니다 : 
```console
What <WP> if <IN> Google <NNP> Morphed <VBD> Into <IN> GoogleOS <NNP> ? <.>
```
이 말뭉치는 tag되어 있고 훈련에 사용할 수 있습니다.

### 도움을 주는 함수들
`corpus`는 유용한 도움 함수들이 많이 포함되어 있습니다. `downsample()`을 호출하고 비율을 정해 데이터를 다운샘플링 할 수 있습니다. 
우선 말뭉치를 얻습니다.
```python
import flair.datasets
corpus = flair.datasets.UD_ENGLISH()
```
그리고 말뭉치를 다운샘플링합니다.
```python
import flair.datasets
downsampled_corpus = flair.datasets.UD_ENGLISH().downsample(0.1)
```
두 말뭉치를 출력하는 것을 통해 10%를 다운 샘플링 한 것을 확인할 수 있습니다.
```python
print("--- 1 Original ---")
print(corpus)

print("--- 2 Downsampled ---")
print(downsampled_corpus)
```
결과입니다 :
```console
--- 1 Original ---
Corpus: 12543 train + 2002 dev + 2077 test sentences

--- 2 Downsampled ---
Corpus: 1255 train + 201 dev + 208 test sentences
```

### 레이블 사전 만들기
다수의 경우 예측할 레이블이 포함되어 있는 "사전"이 필요합니다. `make_label_dictionary` 메소드를 호출하고 `label_type`을 전달해 `corpus`에서 바로 사전을 만들 수 있습니다.

예를 들어, 위에서 인스턴스화된 UD_ENGLISH 말뭉치들은 일반 POS tags('POS'), 범용 POS tags('upos'), 형태학적 tags('tense', 'number'...) 등 여러 레이어의 주석을 가지고 있습니다. 다음 코드는 `label_type='upos'`를 인자로 사용하는 예시입니다.
```python
# 범용 POS tag 작업에 대한 레이블 사전을 만듭니다.
upos_dictionary = corpus.make_label_dictionary(label_type='upos')

# 사전을 출력합니다.
print(upos_dictionary)
```
결과입니다 :
```console
Dictionary with 17 tags: PROPN, PUNCT, ADJ, NOUN, VERB, DET, ADP, AUX, PRON, PART, SCONJ, NUM, ADV, CCONJ, X, INTJ, SYM
```

#### 다른 레이블 유형에 대한 사전
위의 예에서 `make_label_dictionary`를 호출하면 동일한 말뭉치에 있는 모든 레이블 유형에 대한 통계가 인쇄됩니다.
```console
Corpus contains the labels: upos (#204585), lemma (#204584), pos (#204584), dependency (#204584), number (#68023), verbform (#35412), prontype (#33584), person (#21187), tense (#20238), mood (#16547), degree (#13649), definite (#13300), case (#12091), numtype (#4266), gender (#4038), poss (#3039), voice (#1205), typo (#332), abbr (#126), reflex (#100), style (#33), foreign (#18)
```
UD_ENGLISH 말뭉치는 이런 레이블을 가지고 있으며 이에 대한 사전을 만들 수 있습니다. 아래의 예시는 일반 POS tags와 형태학적 숫자 tags에 관한 사전을 만드는 예시입니다.
```python
# 일반 POS tags를 위한 사전을 만듭니다.
pos_dictionary = corpus.make_label_dictionary(label_type='pos')

# 형태학적 숫자 tags를 위한 사전을 만듭니다.
tense_dictionary = corpus.make_label_dictionary(label_type='number')
```
만약 위 사전을 출력한다면 POS 사전에는 50개의 태그가 있고 이 말뭉치에 대한 숫자 사전이 2개(단수 및 복수)만 포함되어 있습니다.

#### 다른 말뭉치를 위한 사전
`make_label_dictionary` 메소드는 텍스트 분류 말뭉치를 포함하여 모든 말뭉치에 사용할 수 있습니다 :
```python
# 텍스트 분류 작업을 위해 레이블 사전을 만듭니다.
corpus = flair.datasets.TREC_6()
print(corpus.make_label_dictionary('question_class'))
```

### 다중 말뭉치(MultiCorpus) 오브젝트
한 번에 여러 태스크를 훈련시키려면 `MultiCorpus` 오브젝트를 사용하세요. `MultiCorpus`를 초기화하기 위해선 원하는 만큼의 `Corpus`오브젝트를 먼저 만들어야 합니다. 그 다음 `Corpus` 리스트를 `MultiCorpus` 오브젝트에 넘겨주면 됩니다. 아래 코드는 Universal Dependency Treebanks 형식의 영어, 독일어, 네덜란드어가 결합된 말뭉치를 로드합니다 :
```python
english_corpus = flair.datasets.UD_ENGLISH()
german_corpus = flair.datasets.UD_GERMAN()
dutch_corpus = flair.datasets.UD_DUTCH()

# 세 언어로 구성된 다중 말뭉치를 만듭니다.
from flair.data import MultiCorpus
multi_corpus = MultiCorpus([english_corpus, german_corpus, dutch_corpus])
```
`MultiCorpus`는 `Corpus`를 상속하기 때문에 모델을 교육하는데 사용할 수 있습니다.

## Flair가 포함하고 있는 데이터셋
Flair는 많은 데이터셋을 지원합니다. 사용자가 해당 생성자 ID를 처음 호출할 때 자동으로 데이터를 다운로드하고 설정합니다.

아래는 지원되는 데이터셋입니다. (**클릭하면 확장됩니다.**)
<details>
  <summary>Named Entity Recognition (NER) datasets</summary>

#### Named Entity Recognition

| Object | Languages | Description |
| -------------    | ------------- |-------------  |
| 'CONLL_03' | English  |  [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER (requires manual download) |
| 'CONLL_03_GERMAN' | German  |  [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER (requires manual download) |
| 'CONLL_03_DUTCH' | Dutch  |  [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER |
| 'CONLL_03_SPANISH' | Spanish  |  [CoNLL-03](https://www.clips.uantwerpen.be/conll2002/ner/) 4-class NER |
| 'NER_ARABIC_ANER' | Arabic  |  [Arabic Named Entity Recognition Corpus](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp) 4-class NER |
| 'NER_ARABIC_AQMAR' | Arabic  |  [American and Qatari Modeling of Arabic](http://www.cs.cmu.edu/~ark/AQMAR/) 4-class NER (modified) |
| 'NER_BASQUE' | Basque  |  [NER dataset for Basque](http://ixa2.si.ehu.eus/eiec/) |
| 'NER_CHINESE_WEIBO' | Chinese  | [Weibo NER corpus](https://paperswithcode.com/sota/chinese-named-entity-recognition-on-weibo-ner/).  |
| 'NER_DANISH_DANE' | Danish | [DaNE dataset](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank) | 
| 'NER_ENGLISH_MOVIE_SIMPLE' | English  |  [NER dataset for movie reviews](https://groups.csail.mit.edu/sls/downloads/movie/) - simple NER |
| 'NER_ENGLISH_MOVIE_COMPLEX' | English  |  [NER dataset for movie reviews](https://groups.csail.mit.edu/sls/downloads/movie/) - complex NER |
| 'NER_ENGLISH_PERSON' | English | [PERSON_NER](https://github.com/das-sudeshna/genid) NER with person names | 
| 'NER_ENGLISH_RESTAURANT' | English  |  [NER dataset for restaurant reviews](https://groups.csail.mit.edu/sls/downloads/restaurant/) |
| 'NER_ENGLISH_SEC_FILLINGS' | English | [SEC-fillings](https://github.com/juand-r/entity-recognition-datasets) with 4-class NER labels from (Alvarado et al, 2015)[https://aclanthology.org/U15-1010/] here | 
| 'NER_ENGLISH_STACKOVERFLOW' | English  | NER on StackOverflow posts |
| 'NER_ENGLISH_TWITTER' | English  |  [Twitter NER dataset](https://github.com/aritter/twitter_nlp/) |
| 'NER_ENGLISH_WIKIGOLD' | English  |  [Wikigold](https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold) a manually annotated collection of Wikipedia text |
| 'NER_ENGLISH_WNUT_2020' | English  |  [WNUT-20](https://github.com/jeniyat/WNUT_2020_NER) named entity extraction |
| 'NER_ENGLISH_WEBPAGES' | English  | 4-class NER on web pages from [Ratinov and Roth (2009)](https://aclanthology.org/W09-1119/) |
| 'NER_FINNISH' | Finnish | [Finer-data](https://github.com/mpsilfve/finer-data) | 
| 'NER_GERMAN_BIOFID' | German  |  [CoNLL-03](https://www.aclweb.org/anthology/K19-1081/) Biodiversity literature NER |
| 'NER_GERMAN_EUROPARL' | German | [German Europarl dataset](https://nlpado.de/~sebastian/software/ner_german.shtml) NER in German EU parliament speeches | 
| 'NER_GERMAN_GERMEVAL' | German  |  [GermEval 14 NER](https://sites.google.com/site/germeval2014ner/data/) corpus |
| 'NER_GERMAN_LEGAL' | German | [Legal Entity Recognition](https://github.com/elenanereiss/Legal-Entity-Recognition) NER in German Legal Documents |
| 'NER_GERMAN_POLITICS' | German | [NEMGP](https://www.thomas-zastrow.de/nlp/) corpus |
| 'NER_HUNGARIAN' | Hungarian | NER on Hungarian business news |
| 'NER_ICELANDIC' | Icelandic | NER on Icelandic |
| 'NER_JAPANESE' | Japanese | [Japanese NER](https://github.com/Hironsan/IOB2Corpus) dataset automatically generated from Wikipedia |
| 'NER_MASAKHANE' | 10 languages | [MasakhaNER: Named Entity Recognition for African Languages](https://github.com/masakhane-io/masakhane-ner) corpora |
| 'NER_SWEDISH' | Swedish | [Swedish Spraakbanken NER](https://github.com/klintan/swedish-ner-corpus/) 4-class NER |
| 'NER_TURKU' | Finnish | [TURKU_NER](https://github.com/TurkuNLP/turku-ner-corpus) NER corpus created by the Turku NLP Group, University of Turku, Finland |
| 'NER_MULTI_WIKIANN' | 282 languages  | Gigantic [corpus for cross-lingual NER derived from Wikipedia](https://elisa-ie.github.io/wikiann/).  |
| 'NER_MULTI_WIKINER' | 8 languages | [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) NER dataset automatically generated from Wikipedia (English, German, French, Italian, Spanish, Portuguese, Polish, Russian) |
| 'NER_MULTI_XTREME' | 176 languages  |  [Xtreme](https://github.com/google-research/xtreme) corpus by Google Research for cross-lingual NER consisting of datasets of a total of 176 languages |
| 'WNUT_17' | English  |  [WNUT-17](https://noisy-text.github.io/2017/files/) emerging entity detection |

</details>

<details>
  <summary>Biomedical Named Entity Recognition (BioNER) datasets</summary>

#### Biomedical Named Entity Recognition

We support 31 biomedical NER datasets, listed [here](HUNFLAIR_CORPORA.md).

</details>

<details>
  <summary>Entity Linking (NEL) datasets</summary>

#### Entity Linking
| Object | Languages | Description |
| -------------    | ------------- |-------------  |
| 'NEL_ENGLISH_AIDA' | English  |  [AIDA CoNLL-YAGO Entity Linking corpus](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads) on the CoNLL-03 corpus |
| 'NEL_ENGLISH_AQUAINT' | English  | Aquaint Entity Linking corpus introduced in [Milne and Witten (2008)](https://www.cms.waikato.ac.nz/~ihw/papers/08-DNM-IHW-LearningToLinkWithWikipedia.pdf) |
| 'NEL_ENGLISH_IITB' | English  | ITTB Entity Linking corpus introduced in [Sayali et al. (2009)](https://dl.acm.org/doi/10.1145/1557019.1557073) |
| 'NEL_ENGLISH_REDDIT' | English  | Reddit Entity Linking corpus introduced in [Botzer et al. (2021)](https://arxiv.org/abs/2101.01228v2) (only gold annotations)|
| 'NEL_ENGLISH_TWEEKI' | English  | ITTB Entity Linking corpus introduced in [Harandizadeh and Singh (2020)](https://aclanthology.org/2020.wnut-1.29.pdf) |
| 'NEL_GERMAN_HIPE' | German  | [HIPE](https://impresso.github.io/CLEF-HIPE-2020/) Entity Linking corpus for historical German as a [sentence-segmented version](https://github.com/stefan-it/clef-hipe) |

</details>


<details>
  <summary>Relation Extraction (RE) datasets</summary>

#### Relation Extraction
| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'RE_ENGLISH_CONLL04' | English  |  [CoNLL-04](https://github.com/bekou/multihead_joint_entity_relation_extraction/tree/master/data/CoNLL04) Relation Extraction |
| 'RE_ENGLISH_SEMEVAL2010' | English  |  [SemEval-2010 Task 8](https://aclanthology.org/S10-1006.pdf) on Multi-Way Classification of Semantic Relations Between Pairs of Nominals |
| 'RE_ENGLISH_TACRED' | English  |  [TAC Relation Extraction Dataset](https://nlp.stanford.edu/projects/tacred/) with 41 relations (download required) |
| 'RE_ENGLISH_DRUGPROT' | English  |  [DrugProt corpus: Biocreative VII Track 1](https://zenodo.org/record/5119892#.YSdSaVuxU5k/) - drug and chemical-protein interactions |

</details>

<details>
  <summary>GLUE Benchmark datasets</summary>

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

</details>

<details>
  <summary>Universal Proposition Banks (UP) datasets</summary>

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

</details>

<details>
  <summary>Universal Dependency Treebanks (UD) datasets</summary>

#### Universal Dependency Treebanks

| Object | Languages | Description |
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

</details>

<details>
  <summary>Text Classification datasets</summary>

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

</details>

<details>
  <summary>Text Regression datasets</summary>

#### Text Regression
| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'WASSA_ANGER' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (anger) |
| 'WASSA_FEAR' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (fear) |
| 'WASSA_JOY' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (joy) |
| 'WASSA_SADNESS' | English | The [WASSA](https://competitions.codalab.org/competitions/16380#learn_the_details) emotion-intensity detection challenge (sadness) |

</details>

<details>
  <summary>Other Sequence Labeling datasets</summary>

#### Other Sequence Labeling

| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'CONLL_2000' | English  | Syntactic chunking with [CoNLL-2000]((https://www.clips.uantwerpen.be/conll2000/chunking/))  |
| 'BIOSCOPE' | English  | Negation and speculation scoping wih [BioScope](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S11-S9/) biomedical texts annotated for uncertainty, negation and their scopes |
| 'KEYPHRASE_INSPEC' | English | Keyphrase dectection with [INSPEC](https://www.aclweb.org/anthology/W03-1028) original corpus (2000 docs) from INSPEC database, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |
| 'KEYPHRASE_SEMEVAL2017' | English | Keyphrase dectection with [SEMEVAL2017](https://arxiv.org/abs/1704.02853) dataset (500 docs) from ScienceDirect, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |
| 'KEYPHRASE_SEMEVAL2010' | English | Keyphrase dectection with [SEMEVAL2010](https://www.aclweb.org/anthology/S10-1004/) dataset (~250 docs) from ACM Digital Library, adapted by [midas-research](https://arxiv.org/abs/1910.08840) |

</details>

<details>
  <summary>Similarity Learning datasets</summary>

#### Experimental: Similarity Learning
| Object | Languages | Description |
| -------------    | ------------- |------------- |
| 'FeideggerCorpus' | German |  [Feidegger](https://github.com/zalandoresearch/feidegger/) dataset fashion images and German-language descriptions  |
| 'OpusParallelCorpus' | Any language pair | Parallel corpora of the [OPUS](http://opus.nlpl.eu/) project, currently supports only Tatoeba corpus |

</details>

만약 감정 텍스트 분류를 위해 IMDB를 로드하고자 한다면 아래의 코드를 참조해주세요 :
```python
import flair.datasets
corpus = flair.datasets.IMDB()
```
위 코드를 통해 모델 교육에 위한 다운로드와 설정이 완료됩니다.

## 고유한 Sequence Labeling Dataset 읽기
만약 위의 리스트에 없는 Sequence Labeling Dataset에 대해 학습을 원한다면 `ColumnCorpus` 오브젝트로 불러올 수 있습니다. NLP에서 대부분의 Sequence Labeling Dataset은 각 행이 단어이며 각 열은 언어 주석인 형태를 가지고 있습니다. 다음 문장을 보시면 :
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

첫 번째 열은 단어, 두 번째 열은 단순한 POS 태그, 세 번째는 BIO-NER tag입니다. 빈 줄은 문장의 구분을 나타냅니다. 이러한 데이터셋을 읽기 위해서 열 구조를 사전으로 정의하고 `ColumnCorpus`를 인스턴스화하면 됩니다.
```python
from flair.data import Corpus
from flair.datasets import ColumnCorpus

# 열을 정의합니다.
columns = {0: 'text', 1: 'pos', 2: 'ner'}

# 훈련, 테스트, 개발 파일이 있는 폴더입니다.
data_folder = '/path/to/data/folder'

# 열의 형식, 데이터 폴더, 훈련, 테스트, 개발 파일을 사용해 말뭉치를 초기화합니다.
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')

```
위 코드를 통해 훈련, 개발, 테스트를 위해 분리된 `Sentence`의 리스트를 가지고 있는 말뭉치 오브젝트를 만들었습니다. 아래 코드를 통해 훈련 split에 몇개의 문장이 있는지 확인할 수 있습니다 :
```python
len(corpus.train)
```
또한 문장에 접근해 주석을 확인할 수 있습니다 :
```python
print(corpus.train[0].to_tagged_string('ner'))
print(corpus.train[1].to_tagged_string('pos'))
```
이는 아래와 같은 결과를 보여줍니다 : 
```console
George <B-PER> Washington <I-PER> went to Washington <B-LOC> .

Sam <N> Houston <N> stayed <V> home <N>
```

## Text Classification dataset 
자신의 text Classification dataset을 사용하는 두 가지 방법이 있습니다.
텍스트와 레이블을 simple CSV 파일 혹은 [FastText 형식](https://fasttext.cc/docs/en/supervised-tutorial.html)으로 데이터를 포맷하면 됩니다.

#### simple CSV file로 로드하기
많은 text classification dataset은 simple CSV 파일로 배포됩니다. simple CSV 파일은 각 행이 데이터 포인트에 해당하고 열이 텍스트, 레이블, 기타 메타 데이터인 형식을 가지고 있습니다. `CSVClassificationCorpus`에 위에서 본 `ColumnCorpus`같은 열 형식을 전달하는 것으로 CSV 포멧의 classification dataset을 로드할 수 있습니다. 열 형식은 CSV에서 텍스트를 보관하는 열과 레이블을 보관하는 영역을 나타냅니다. 파이썬 CSV 라이브러리는 Excel CSV 포멧을 기본으로 하고있지만 추가적인 파라미터를 통해 [사용자 지정 구분 문자](https://docs.python.org/3/library/csv.html#csv-fmt-params) 혹은 따옴표를 사용할 수 있습니다.
참고 : 말뭉치 initializer는 자동으로 훈련, 개발, 테스트 split을 폴더에서 검색하기 때문에 분할 CSV 파일들은 각 이름이 적절하게 지정되어야 합니다.(예: `train.csv` `test.csv` `dev.csv`)
```python
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

# 훈련, 테스트, 개발 파일이 있는 폴더의 경로입니다.
data_folder = '/path/to/data'

# 열 형식은 텍스트와 레이블을 포함하는 형식입니다.
column_name_map = {4: "text", 1: "label_topic", 2: "label_subtopic"}

# 훈련, 테스트, 개발 데이터가 포함된 말뭉치를 로드합니다. 만약 CSV 헤더가 있다면 스킵합니다.
corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         skip_header=True,
                                         delimiter='\t',    # tab-separated files
) 
```

#### FastText 형식으로 로드하기
`CSVClassificationCorpus`가 효과적이지 않은 경우 파일의 각 줄이 텍스트 문서를 나타내는 형식의 FastText을 사용합니다.
문서에는 접두사 `__label__`로 시작하는 한개 이상의 레이블이 있을 수 있습니다. 아래를 참고해주세요 :
```bash
__label__<label_1> <text>
__label__<label_1> __label__<label_2> <text>
```
앞에서 언급한 바와 같이 text classification을 하기 위한 `Corpus`를 만들기 위해선 훈련, 개발, 테스트 세 개의 파일이 필요합니다.
아래의 코드는 IMDB작업을 하는 예시입니다 : 
```text
/resources/tasks/imdb/train.txt
/resources/tasks/imdb/dev.txt
/resources/tasks/imdb/test.txt
```
`/resources/tasks/imdb`를 선택해 `ClassificationCorpus`를 만듭니다.
파일의 각 줄은 레이블 주석이 있는 `Sentence` 오브젝트로 변환됩니다.

주의 : 한 줄의 텍스트는 여러 문장을 포함하고 있을 수 있기 때문에 `Sentence` 오브젝트는 여러개의 문장으로 구성될 수 있습니다.
```python
from flair.data import Corpus
from flair.datasets import ClassificationCorpus

# 훈련, 테스트, 개발 파일이 있는 폴더의 경로입니다.
data_folder = '/path/to/data/folder'

# 훈련, 테스트, 개발 데이터가 포함된 말뭉치를 로드합니다.
corpus: Corpus = ClassificationCorpus(data_folder,
                                      test_file='test.txt',
                                      dev_file='dev.txt',
                                      train_file='train.txt',                                       
                                      label_type='topic',
                                      )
```
대부분의 경우 말뭉치 initializer는 자동으로 폴더의 훈련, 개발, 테스트 split을 찾습니다. 그렇기 때문에 파일 이름을 직접 지정할 필요가 없습니다. 이 정도면 충분합니다.
```python
# 훈련, 테스트, 개발 파일이 있는 폴더의 경로입니다.
data_folder = '/path/to/data/folder'

# 폴더에서 자동으로 훈련, 개발, 테스트 split을 식별합니다. 
corpus: Corpus = ClassificationCorpus(data_folder,                                                                            
                                      label_type='topic',
                                      )
```
`FastText` 형식은 열이 없기 때문에 주석의 이름을 직접 정의해야 합니다. 위 예제는 `label_type='topic'`인 말뭉치를 로드하고 있음을 나타냅니다.

## 다음 튜토리얼
이제 [나만의 모델을 훈련](/resources/docs/KOR_docs/TUTORIAL_7_TRAINING_A_MODEL.md)을 알아보겠습니다.
