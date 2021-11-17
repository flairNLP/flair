# 튜토리얼 2: 텍스트 태깅

튜토리얼 2부는 1부를 어느정도 학습하였다고 가정하고 진행하겠습니다. 
여기서는 사전 훈련된 모델을 사용하여 텍스트에 태그를 지정합니다.

## 사전 훈련된 모델을 사용하여 태깅

개체명 인식(NER)에 대해 사전 훈련된 모델을 사용하겠습니다.
이 모델은 영어 CoNLL-03 과제를 통해 교육되었으며 4개의 다른 실체를 인식할 수 있습니다.

```python
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')
```

문장에서 tagger의 메소드인 predict()를 사용할 수 있습니다.
예측 태그를 토큰에 추가합니다. 여기서는 문장에 두 개의 명명된 엔터티가 있는 문장을 사용하겠습니다.

```python
sentence = Sentence('George Washington went to Washington.')

# NER 태그 예측
tagger.predict(sentence)

# 예측된 태그가 있는 문장 출력
print(sentence.to_tagged_string())
```

출력 결과: 
```console
George <B-PER> Washington <E-PER> went to Washington <S-LOC> . 
```

### 주석이 달린 범위 받기

많은 시퀀스 라벨링 방법은 여러 단어로 구성된 범위에 주석을 달게 됩니다. (예 : "조지 워싱턴")
다음과 같이 태그가 지정된 문장에서 이러한 범위를 직접 얻을 수 있습니다.

```python
for entity in sentence.get_spans('ner'):
    print(entity)
```

출력 결과:
```console
Span [1,2]: "George Washington"   [− Labels: PER (0.9968)]
Span [5]: "Washington"   [− Labels: LOC (0.9994)]
```

이것은 "조지 워싱턴"이 사람이고 "워싱턴"이 사람임을 나타냅니다.
위치(LOC)에는 각각 문장과 문장 내 위치, 라벨이 있고, 값 및 점수(예측에 대한 신뢰)가 있어야 합니다.
또한 저희는 위치 오프셋과 같은 추가 정보를 얻을 수 있습니다.
문장의 각 실체는 다음을 호출합니다.

```python
print(sentence.to_dict(tag_type='ner'))
```

출력 결과:
```console
{'text': 'George Washington went to Washington.',
    'entities': [
        {'text': 'George Washington', 'start_pos': 0, 'end_pos': 17, 'type': 'PER', 'confidence': 0.999},
        {'text': 'Washington', 'start_pos': 26, 'end_pos': 36, 'type': 'LOC', 'confidence': 0.998}
    ]}
```


### 멀티 태깅

예를 들어 NER 및 POS(Part-of-Speech) 태그와 같은 여러 유형의 주석을 한 번에 예측하려는 경우도 있습니다.
이를 위해 다음과 같이 새로운 멀티태거 개체를 사용할 수 있습니다.

```python
from flair.models import MultiTagger

# POS와 NER 용 tagger 로드
tagger = MultiTagger.load(['pos', 'ner'])

# 예시 문장 만들기
sentence = Sentence("George Washington went to Washington.")

# 두 모델로 예측
tagger.predict(sentence)

print(sentence)
``` 

이 문장에는 두 가지 유형의 주석이 있습니다. POS와 NER입니다.

### 사전 훈련된 시퀀스 태거 모델 목록

적절한 교육을 통과하여 로드할 사전 교육 모델을 선택합니다.
`SequenceTagger` 클래스의 `load()` 메서드에 문자열을 지정합니다.

현재 및 커뮤니티 기반 모델의 전체 목록은 [__model hub__](https://huggingface.co/models?library=flair&sort=downloads)에서 찾아볼 수 있다.
최소한 다음과 같은 사전 교육 모델이 제공됩니다(자세한 정보를 보려면 ID 링크 클릭).
모델 및 온라인 데모):

#### 영어 모델들

| ID | 태스크 | 언어 | 훈련 데이터셋 | 정확도 | 참고사항 |
| -------------    | ------------- |------------- |------------- | ------------- | ------------- |
| '[ner](https://huggingface.co/flair/ner-english)' | NER (4-class) |  English | Conll-03  |  **93.03** (F1) |
| '[ner-fast](https://huggingface.co/flair/ner-english-fast)' | NER (4-class)  |  English  |  Conll-03  |  **92.75** (F1) | (fast model)
| '[ner-large](https://huggingface.co/flair/ner-english-large)' | NER (4-class)  |  English  |  Conll-03  |  **94.09** (F1) | (large model)
| 'ner-pooled' | NER (4-class)  |  English |  Conll-03  |  **93.24** (F1) | (memory inefficient)
| '[ner-ontonotes](https://huggingface.co/flair/ner-english-ontonotes)' | NER (18-class) |  English | Ontonotes  |  **89.06** (F1) |
| '[ner-ontonotes-fast](https://huggingface.co/flair/ner-english-ontonotes-fast)' | NER (18-class) |  English | Ontonotes  |  **89.27** (F1) | (fast model)
| '[ner-ontonotes-large](https://huggingface.co/flair/ner-english-ontonotes-large)' | NER (18-class) |  English | Ontonotes  |  **90.93** (F1) | (large model)
| '[chunk](https://huggingface.co/flair/chunk-english)' |  Chunking   |  English | Conll-2000     |  **96.47** (F1) |
| '[chunk-fast](https://huggingface.co/flair/chunk-english-fast)' |   Chunking   |  English | Conll-2000     |  **96.22** (F1) |(fast model)
| '[pos](https://huggingface.co/flair/pos-english)' |  POS-tagging |   English |  Ontonotes     |**98.19** (Accuracy) |
| '[pos-fast](https://huggingface.co/flair/pos-english-fast)' |  POS-tagging |   English |  Ontonotes     |  **98.1** (Accuracy) |(fast model)
| '[upos](https://huggingface.co/flair/upos-english)' |  POS-tagging (universal) | English | Ontonotes     |  **98.6** (Accuracy) |
| '[upos-fast](https://huggingface.co/flair/upos-english-fast)' |  POS-tagging (universal) | English | Ontonotes     |  **98.47** (Accuracy) | (fast model)
| '[frame](https://huggingface.co/flair/frame-english)'  |   Frame Detection |  English | Propbank 3.0     |  **97.54** (F1) |
| '[frame-fast](https://huggingface.co/flair/frame-english-fast)'  |  Frame Detection |  English | Propbank 3.0     |  **97.31** (F1) | (fast model)
| 'negation-speculation'  | Negation / speculation |English |  Bioscope | **80.2** (F1) |

### 다국어 모델

단일 모델 내에서 여러 언어로 텍스트를 처리할 수 있는 새로운 모델을 배포합니다.

NER 모델은 4개 언어(영어, 독일어, 네덜란드어 및 스페인어) 이상, PoS 모델은 12개 언어(영어, 독일어, 프랑스어, 이탈리아어, 네덜란드어, 폴란드어, 스페인어, 스웨덴어, 덴마크어, 노르웨이어, 핀란드어 및 체코어)이 존재합니다.

| ID | 태스크 | 언어 | 훈련 데이터셋 | 정확도 | 참고사항 |
| -------------    | ------------- |------------- |------------- | ------------- | ------------- |
| '[ner-multi](https://huggingface.co/flair/ner-multi)' | NER (4-class) | Multilingual | Conll-03   |  **89.27**  (average F1) | (4 languages)
| '[ner-multi-fast](https://huggingface.co/flair/ner-multi-fast)' | NER (4-class)|  Multilingual |  Conll-03   |  **87.91**  (average F1) | (4 languages)
| '[pos-multi](https://huggingface.co/flair/upos-multi)' |  POS-tagging   |  Multilingual |  UD Treebanks  |  **96.41** (average acc.) |  (12 languages)
| '[pos-multi-fast](https://huggingface.co/flair/upos-multi-fast)' |  POS-tagging |  Multilingual |  UD Treebanks  |  **92.88** (average acc.) | (12 languages) 

이러한 언어로 된 텍스트를 모델에 전달할 수 있습니다. 특히, NER는 프랑스어와 같이 훈련되지 않은 언어에도 적용되었습니다.

#### 다른 언어들을 위한 모델들

| ID | 태스크 | 언어 | 훈련 데이터셋 | 정확도 | 참고사항 |
| -------------    | ------------- |------------- |------------- |------------- | ------------ |
| '[ar-ner](https://huggingface.co/megantosh/flair-arabic-multi-ner)' | NER (4-class) | Arabic | AQMAR & ANERcorp (curated) |  **86.66** (F1) | |
| '[ar-pos](https://huggingface.co/megantosh/flair-arabic-dialects-codeswitch-egy-lev)' | NER (4-class) | Arabic (+dialects)| combination of corpora |  | |
| '[de-ner](https://huggingface.co/flair/ner-german)' | NER (4-class) |  German | Conll-03  |  **87.94** (F1) | |
| '[de-ner-large](https://huggingface.co/flair/ner-german-large)' | NER (4-class) |  German | Conll-03  |  **92,31** (F1) | |
| 'de-ner-germeval' | NER (4-class) | German | Germeval  |  **84.90** (F1) | |
| '[de-ner-legal](https://huggingface.co/flair/ner-german-legal)' | NER (legal text) |  German | [LER](https://github.com/elenanereiss/Legal-Entity-Recognition) dataset  |  **96.35** (F1) | |
| 'de-pos' | POS-tagging | German | UD German - HDT  |  **98.50** (Accuracy) | |
| 'de-pos-tweets' | POS-tagging | German | German Tweets  |  **93.06** (Accuracy) | [stefan-it](https://github.com/stefan-it/flair-experiments/tree/master/pos-twitter-german) |
| 'de-historic-indirect' | historical indirect speech | German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-direct' | historical direct speech |  German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-reported' | historical reported speech | German |  @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| 'de-historic-free-indirect' | historical free-indirect speech | German | @redewiedergabe project |  **87.94** (F1) | [redewiedergabe](https://github.com/redewiedergabe/tagger) | |
| '[fr-ner](https://huggingface.co/flair/ner-french)' | NER (4-class) | French | [WikiNER (aij-wikiner-fr-wp3)](https://github.com/dice-group/FOX/tree/master/input/Wikiner)  |  **95.57** (F1) | [mhham](https://github.com/mhham) |
| '[es-ner-large](https://huggingface.co/flair/ner-spanish-large)' | NER (4-class) | Spanish | CoNLL-03  |  **90,54** (F1) | [mhham](https://github.com/mhham) |
| '[nl-ner](https://huggingface.co/flair/ner-dutch)' | NER (4-class) | Dutch |  [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **92.58** (F1) |  |
| '[nl-ner-large](https://huggingface.co/flair/ner-dutch-large)' | NER (4-class) | Dutch | Conll-03 |  **95,25** (F1) |  |
| 'nl-ner-rnn' | NER (4-class) | Dutch | [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **90.79** (F1) | |
| '[da-ner](https://huggingface.co/flair/ner-danish)' | NER (4-class) | Danish |  [Danish NER dataset](https://github.com/alexandrainst/danlp)  |   | [AmaliePauli](https://github.com/AmaliePauli) |
| 'da-pos' | POS-tagging | Danish | [Danish Dependency Treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md)  |  | [AmaliePauli](https://github.com/AmaliePauli) |
| 'ml-pos' | POS-tagging | Malayalam | 30000 Malayalam sentences  | **83** | [sabiqueqb](https://github.com/sabiqueqb) |
| 'ml-upos' | POS-tagging | Malayalam | 30000 Malayalam sentences | **87** | [sabiqueqb](https://github.com/sabiqueqb) |
| 'pt-pos-clinical' | POS-tagging | Portuguese | [PUCPR](https://github.com/HAILab-PUCPR/portuguese-clinical-pos-tagger) | **92.39** | [LucasFerroHAILab](https://github.com/LucasFerroHAILab) for clinical texts |


### 독일어 문장 태그 지정

위 목록에 표시된 것처럼 영어 이외의 언어에 대한 사전 교육 모델도 제공합니다. 독일어 문장에 태그를 지정하려면 적절한 모델을 로드하면 됩니다.

```python

# 모델 로드
tagger = SequenceTagger.load('de-ner')

# 독일어 문장 만들기
sentence = Sentence('George Washington ging nach Washington.')

# NER 태그 예측
tagger.predict(sentence)

# 예측된 태그가 있는 문장 출력
print(sentence.to_tagged_string())
```

출력 결과: 
```console
George <B-PER> Washington <E-PER> ging nach Washington <S-LOC> .
```

### 아랍어 문장 태그 지정

Flair는 또한 오른쪽에서 왼쪽으로 쓰는 언어에서도 작동한다. 아랍어 문장에 태그를 지정하려면 적절한 모델을 로드하면 됩니다.
```python

# 모델 로드
tagger = SequenceTagger.load('ar-ner')

# 아랍어 문장 만들기
sentence = Sentence("احب برلين")

# NER 태그 예측
tagger.predict(sentence)

# 예측된 태그가 있는 문장 출력
for entity in sentence.get_labels('ner'):
    print(entity)
```

출력 : 
```console
LOC [برلين (2)] (0.9803) 
```

### 다국어 텍스트 태그 지정

여러 언어(예: 영어 및 독일어)의 텍스트가 있는 경우, 새로운 다국어 모델을 사용할 수 있습니다.

```python

# 모델 로드
tagger = SequenceTagger.load('pos-multi')

# 영어와 독일어 문장으로 된 텍스트
sentence = Sentence('George Washington went to Washington. Dort kaufte er einen Hut.')

# PoS 태그 예측
tagger.predict(sentence)

# 예측된 태그가 있는 문장 출력
print(sentence.to_tagged_string())
```

출력 결과: 
```console
George <PROPN> Washington <PROPN> went <VERB> to <ADP> Washington <PROPN> . <PUNCT>

Dort <ADV> kaufte <VERB> er <PRON> einen <DET> Hut <NOUN> . <PUNCT>
```

그래서 이 문장에서는 'went'와 'kaufte'가 모두 동사로 식별된다.

### 실험: 시맨틱 프레임 탐지

영어의 경우 Propbank 3.0 프레임을 사용하여 학습된 텍스트 의미 프레임을 감지하는 사전 교육 모델을 제공합니다.
이것은 단어를 연상시키는 틀에 대한 일종의 단어 감각의 모호함을 제공합니다.

예를 들어 보겠습니다.

```python
# 모델 로드
tagger = SequenceTagger.load('frame')

# 영어 문장 만들기
sentence_1 = Sentence('George returned to Berlin to return his hat.')
sentence_2 = Sentence('He had a look at different hats.')

# NER 태그 예측하기
tagger.predict(sentence_1)
tagger.predict(sentence_2)

# 예측된 태그가 있는 문장 출력
print(sentence_1.to_tagged_string())
print(sentence_2.to_tagged_string())
```
출력 결과: 

```console
George returned <return.01> to Berlin to return <return.02> his hat .

He had <have.LV> a look <look.01> at different hats .
```

우리가 볼 수 있듯이, 프레임 감지기는 문장 1에서 '반환'이라는 단어의 두 가지 다른 의미 사이를 구별합니다.
'return.01'은 위치로 돌아가는 것을 의미하고, 'return.02'는 무언가를 돌려주는 것을 의미합니다.

비슷하게, 문장 2에서 프레임 탐지기는 'have'가 라이트 동사인 경동사 구조를 찾습니다.
look은 단어를 연상시키는 틀입니다.

### 문장 목록 태그 지정

종종 전체 텍스트 말뭉치에 태그를 지정할 수 있습니다. 이 경우에, 당신은 말뭉치를 문장으로 나누고 통과시킬 필요가 있다.
.predict() 메서드에 대한 'Sentence' 개체 목록입니다.

예를 들어 segtok의 문장 분할기를 사용하여 텍스트를 분할할 수 있습니다.

```python
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

# 많은 문장이 포함된 예제 텍스트
text = "This is a sentence. This is another sentence. I love Berlin."

# 문장 스플리터 초기화
splitter = SegtokSentenceSplitter()

# 스플리터를 사용하여 텍스트를 문장 목록으로 분할
sentences = splitter.split(text)

# 문장에 대한 태그 예측
tagger = SequenceTagger.load('ner')
tagger.predict(sentences)

# 문장을 반복하고 예측된 레이블을 출력
for sentence in sentences:
    print(sentence.to_tagged_string())
```

`.predict()` 메서드의 `mini_batch_size` 매개 변수를 사용하여, 다음에 전달된 미니 배치의 크기를 설정할 수 있습니다.
태그거. 리소스에 따라 이 매개 변수를 사용하여 속도를 최적화할 수 있습니다.


## 사전 교육된 텍스트 분류 모델을 사용한 태그 지정

긍정 또는 부정 의견을 탐지하기 위해 사전 훈련된 모델을 사용하겠습니다.
이 모델은 제품과 영화 리뷰 데이터셋의 혼합에 대해 교육되었으며 긍정적인 것을 인식할 수 있습니다.
그리고 영어 본문에는 부정적인 정서가 있습니다.

```python
from flair.models import TextClassifier

# tagger 로드
classifier = TextClassifier.load('sentiment')
```

여러분은 문장에서 분류자의 `predict()`방법만 사용하면 됩니다. 예측 레이블에 추가하고, 긍정적인 느낌의 문장을 사용해봅시다.

```python
# 예시 문장 만들기
sentence = Sentence("enormously entertaining for moviegoers of any age.")

# predict 호출
classifier.predict(sentence)

# 예측 확인하기
print(sentence)
```

출력 결과:
```console
Sentence: "enormously entertaining for moviegoers of any age."   [− Tokens: 8  − Sentence-Labels: {'class': [POSITIVE (0.9976)]}]
```

POSITION이라는 라벨이 문장에 추가되어 이 문장이 긍정적인 감정을 가지고 있음을 나타냅니다.

### 사전 교육 텍스트 분류 모델 목록

적절한 교육을 통과하여 로드할 사전 교육 모델을 선택합니다.
문자열은 `TextClassifier` 클래스의 `load()` 메서드로 이동합니다. 현재 다음과 같은 사전 교육 모델 제공됨:

| ID | 언어 | 태스크 | 훈련 데이터셋 | 정확도 |
| ------------- | ---- | ------------- |------------- |------------- |
| 'sentiment' | English | detecting positive and negative sentiment (transformer-based) | movie and product reviews |  **98.87** |
| 'sentiment-fast' | English | detecting positive and negative sentiment (RNN-based) | movie and product reviews |  **96.83**|
| 'communicative-functions' | English | detecting function of sentence in research paper (BETA) | scholarly papers |  |
| 'de-offensive-language' | German | detecting offensive language | [GermEval 2018 Task 1](https://projects.fzai.h-da.de/iggsa/projekt/) |  **75.71** (Macro F1) |

## 교육 데이터 없이 새 클래스 태그 지정

포함되지 않은 클래스에 레이블을 지정해야 하는 경우
사전 훈련된 제로샷 분류기 TARS를 사용해 볼 수도 있습니다.
([제로샷 튜토리얼](/resources/docs/KOR_docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md)로 건너뛰기)    
TARS는 임의 클래스에 대해 텍스트 분류를 수행할 수 있습니다.

## 다음

이제 텍스트를 포함하기 위해 다른 [워드 임베딩](/resources/docs/KOR_docs/TUTORIAL_3_WORD_EMBEDDING.md)을 사용하는 방법에 대해 알아보겠습니다.
