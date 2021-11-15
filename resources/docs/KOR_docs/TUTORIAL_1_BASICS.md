# 튜토리얼 1: NLP 기본 타입들

튜토리얼 1부에서는 이 라이브러리에서 사용되는 몇 가지 기본 유형을 살펴볼 것 입니다.

## 문장 생성

flair 라이브러리의 중심에는 'Sentence'과 'Token'이라는 두 가지 유형의 오브젝트가 존재합니다.
문장(Sentence)은 본문 문장(Token)을 담고 있으며 본질적으로 토큰(Token)의 목록입니다.

먼저 예문 'Sentence' 를 만드는 것으로 시작하겠습니다.
```python
# 문장 객체는 특정 태그를 지정할 수 있는 문장을 포함합니다.
from flair.data import Sentence

# 문장 객체를 만드는 모습
sentence = Sentence('The grass is green.')

# 문자열이 포함된 Sentence 객체 출력
print(sentence)
```

출력 결과:

```console
Sentence: "The grass is green ."   [− Tokens: 5]
```

출력문에서 문장이 5개의 토큰으로 구성되어 있음을 알 수 있습니다.
다음과 같이 토큰의 ID 나 인덱스를 통해 문장의 토큰에 액세스할 수도 있습니다.

```python
# 토큰 id
print(sentence.get_token(4))
# 인덱스 자체
print(sentence[3])
```

두 출력문은 아래와 같은 결과가 나오게 됩니다.

```console
Token: 4 green
```

위 출력문에는 토큰 ID(4)와 토큰의 어휘 값("green")이 포함됩니다. 또한 문장의 모든 토큰에 대해 반복하여 출력이 가능합니다.

```python
for token in sentence:
    print(token)
```

출력 결과:

```console
Token: 1 The
Token: 2 grass
Token: 3 is
Token: 4 green
Token: 5 .
```

## 토큰화

위와 같이 'Sentence'를 생성하면 텍스트는 [세그톡 라이브러리](https://pypi.org/project/segtok/)에 의해 자동 토큰화됩니다.

### 토큰화를 사용 안하기

이 토큰나이저를 사용하지 않으려면 `use_tokenizer` 플래그를 `False`로 설정하십시오.

```python
from flair.data import Sentence

# 'use_tokenizer' flag를 false로 설정하여 토큰화하지 않는 모습
untokenized_sentence = Sentence('The grass is green.', use_tokenizer=False)

# 출력
print(untokenized_sentence)
```

이 경우 토큰화가 수행되지 않고 텍스트가 공백으로 분할되므로 토큰이 4개만 생성됩니다.

### 다른 토크나이저를 사용하는 경우

사용자 지정 토큰나이저를 초기화 방법에 전달할 수도 있습니다. 
예를 들어 일본어를 토큰화하려는 경우, 문장 대신 다음과 같이 'janome' 토큰나이저를 사용할 수 있습니다.

```python
from flair.data import Sentence
from flair.tokenization import JapaneseTokenizer

# 일본어 토크나이저 초기화
tokenizer = JapaneseTokenizer("janome"

# 문장 생성
japanese_sentence = Sentence("私はベルリンが好き", use_tokenizer=tokenizer)

# 문장 출력
print(japanese_sentence)
```

출력 결과:

```console
Sentence: "私 は ベルリン が 好き"   [− Tokens: 5]
```

다음과 같이 토큰화 루틴을 직접 작성할 수 있습니다. 

### 사전 토큰화된 시퀀스 사용
사전 토큰화된 시퀀스를 단어 목록으로 전달할 수 있습니다.

```python
from flair.data import Sentence
sentence = Sentence(['The', 'grass', 'is', 'green', '.'])
print(sentence)
```

출력 결과:

```console
Sentence: "The grass is green ."   [− Tokens: 5]
```


## 라벨 추가

### 토큰에 라벨 추가

Flair에서는 모든 데이터 점에 레이블을 지정할 수 있습니다. 예를 들어 단어에 레이블을 지정하거나 문장에 레이블을 지정할 수 있습니다.

```python
# 문장 속 단어에 대한 태그 추가
sentence[3].add_tag('ner', 'color')

# 문장의 모든 태그 출력
print(sentence.to_tagged_string())
```

출력 결과:

```console
The grass is green <color> .
```

라벨 클래스의 각 태그는 다음과 같이 옆에 score가 표시됩니다.

```python
# 3번째 인덱스의 토큰 가져오기
token = sentence[3]

# 토큰의 ner 태그를 가져오기
tag = token.get_tag('ner')

# 토큰 출력
print(f'"{token}" is tagged as "{tag.value}" with confidence score "{tag.score}"')
```

출력 결과:

```console
"Token: 4 green" is tagged as "color" with confidence score "1.0"
```

방금의 color 태그는 수동으로 추가했기 때문에 1.0점입니다. 태그가 다음 항목에 의해 예측되는 경우
시퀀스 레이블러, 점수 값은 분류자 신뢰도를 나타냅니다.

### 문장에 라벨 추가

전체 문장에 라벨도 추가할 수 있습니다.
예를 들어, 아래 예제는 문장에 '스포츠'라는 레이블을 추가하는 방법을 보여줍니다.


```python
sentence = Sentence('France is the current world cup winner.')

# 문장에 라벨 추가
sentence.add_label('topic', 'sports')

print(sentence)

# 또는 한 줄에 레이블이 있는 문장을 만들 수도 있습니다.
sentence = Sentence('France is the current world cup winner.').add_label('topic', 'sports')

print(sentence)
```

출력 결과: 

```console
Sentence: "France is the current world cup winner."   [− Tokens: 7  − Sentence-Labels: {'topic': [sports (1.0)]}]
```

위 문장은 완벽하게 '스포츠' 항목에 속함을 나타냅니다.

### 다중 레이블

모든 데이터에 대해 여러 번 레이블을 지정할 수 있습니다. 예를 들어 문장은 두 가지 주제에 속할 수 있습니다. 이 경우 레이블 이름이 같은 레이블 두 개를 추가합니다.

```python
sentence = Sentence('France is the current world cup winner.')

# 이 문장에는 여러 주제 레이블들이 있습니다.
sentence.add_label('topic', 'sports')
sentence.add_label('topic', 'soccer')
```

동일한 문장에 대해 다른 주석 계층을 추가할 수 있습니다. 주제 옆에서 문장의 "언어"를 예측할 수도 있습니다. 이 경우 다른 레이블 이름을 가진 레이블을 추가합니다.

```python
sentence = Sentence('France is the current world cup winner.')

# 이 문장에는 여러 "주제"의 레이블들이 있습니다.
sentence.add_label('topic', 'sports')
sentence.add_label('topic', 'soccer')

# 이 문장에는 "언어" 레이블이 있습니다.
sentence.add_label('language', 'English')

print(sentence)
```

출력 결과: 

```console
Sentence: "France is the current world cup winner."   [− Tokens: 7  − Sentence-Labels: {'topic': [sports (1.0), soccer (1.0)], 'language': [English (1.0)]}]
```

이 문장에 두 개의 "주제" 라벨과 하나의 "언어" 라벨이 있음을 나타냅니다.

### 문장의 레이블에 액세스

다음과 같은 레이블에 액세스할 수 있습니다.

```python
for label in sentence.labels:
    print(label)
```

각 라벨은 'Label' 개체이므로 라벨의 `val` 및 `score` 필드에 직접 액세스할 수도 있습니다.

```python
print(sentence.to_plain_string())
for label in sentence.labels:
    print(f' - classified as "{label.value}" with score {label.score}')
```

출력 결과:

```console
France is the current world cup winner.
 - classified as "sports" with score 1.0
 - classified as "soccer" with score 1.0
 - classified as "English" with score 1.0
```

한 레이어의 레이블에만 관심이 있는 경우 다음과 같이 액세스할 수 있습니다.

```python
for label in sentence.get_labels('topic'):
    print(label)
```

위의 예제는 topic 라벨만 제공합니다 

## 다음 튜토리얼

지금까지 문장을 만들고 수동으로 라벨을 붙이는 방법에 대해 알아보았습니다.

이제 [사전 교육된 모델](/resources/docs/KOR_docs/TUTORIAL_2_TAGGING.md)을 사용하여 텍스트에 태그를 지정하는 방법에 대해 알아보겠습니다.
