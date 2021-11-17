# 튜토리얼 3: 워드 임베딩

다양한 방법으로 문장에 단어를 삽입할 수 있는 일련의 수업을 제공합니다.
그 이전의 튜토리얼 1,2에 대해 어느 정도 학습하셨다고 가정하고 진행하겠습니다.


## 임베딩

모든 단어 임베딩 클래스는 `TokenEmbeddings` 클래스에서 상속되며 필요한 `embed()` 를 호출하여 텍스트를 포함합니다.
필요한 임베딩 클래스를 인스턴스화하고 `embed()`로 호출하여 텍스트를 임베딩합니다.
우리의 방법으로 생산된 모든 임베딩은 PyTorch 벡터이기 때문에 즉시 훈련에 사용될 수 있고 미세 조정이 가능합니다.

이 튜토리얼에서는 몇 가지 일반적인 임베딩을 소개하고 사용 방법을 보여줍니다. 
이러한 임베딩에 대한 자세한 내용과 지원되는 모든 임베딩에 대한 개요는 [여기](/resources/docs/KOR_docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)를 참조하세요.

## 클래식 워드 임베딩

고전적인 단어 임베딩은 정적인 성향과 단어 수준을 가지며, 이는 각 개별 단어가 정확히 하나의 사전 계산된 임베딩을 얻는다는 것을 의미합니다.
널리 사용되는 GloVe 또는 Komninos 임베딩을 포함한 대부분의 임베딩이 이 클래스에 속합니다.

우선 `WordEmbedings` 클래스를 인스턴스화하고 로드할 임베딩의 문자열 식별자를 전달합니다.
GloVe 임베딩을 사용하려면 'glove' 문자열을 생성자에게 전달하십시오.

```python
from flair.embeddings import WordEmbeddings
from flair.data import Sentence

# 임베딩 초기화
glove_embedding = WordEmbeddings('glove')
```

이제 예제 문장을 만들고 임베딩의 `embed()` 메서드를 호출합니다. 일부 임베딩 유형은 속도를 높이기 위해 배치를 사용하기 때문에 문장 목록을 이 방법으로 전달할 수도 있습니다.

```python
# 문장 만들기
sentence = Sentence('The grass is green .')

# glove를 사용하여 문장 삽입
glove_embedding.embed(sentence)

# 이제 포함된 토큰을 확인하십시오.
for token in sentence:
    print(token)
    print(token.embedding)
```


이렇게 하면 토큰과 임베딩이 출력됩니다. GloVe 임베딩은 차수 100의 PyTorch 벡터입니다.

적절한 내용을 전달하여 로드하는 사전 교육된 임베딩을 선택합니다.
`WordEmbedings` 클래스의 생성자에 대한 ID 문자열입니다. 일반적으로
**두 글자로 된 언어 코드**는 임베딩을 시작하므로 영어의 'en'과
독일어 등을 나타내는 'de'입니다. 기본적으로 Wikipedia를 통해 학습된 FastText 임베딩이 초기화됩니다.
또한 '-crawl'로 인스턴스화하여 웹 크롤을 통해 언제든지 FastText 임베딩을 사용할 수 있습니다. 
따라서 독일 웹 크롤을 통해 학습된 임베딩을 사용하기 위해 'de-crawl'을 사용합니다.

```python
german_embedding = WordEmbeddings('de-crawl')
```

이 클래스에 대한 자세한 설명과 함께 [여기](/docs/embeddings/CLASSIC_WORD_EMBEDings.md) 모든 워드 임베딩 모델의 전체 목록을 확인할 수 있습니다.
일반적으로 FastText 임베딩 또는 GloVe를 사용하는 것이 좋습니다.


## Flair 임베딩

상황별 문자열 임베딩은 [powerful embeddings](https://www.aclweb.org/anthology/C18-1139/)
표준 단어 임베딩을 넘어서는 잠재적인 구문 분석 정보를 캡처합니다. 주요 차이점은 
(1) 단어에 대한 명확한 개념 없이 훈련되고 따라서 기본적으로 단어를 문자 시퀀스로 모델링합니다. 
(2) 주변 텍스트에 의해 **contextualized**됩니다. 이는 *동일 단어의 문맥적 용도에 따라* 다른 임베딩이 있음을 의미합니다.

Flair를 사용할 때, 표준 단어 임베딩과 같은 적절한 임베딩 클래스를 인스턴스화하기만 하면 이러한 임베딩을 사용할 수 있습니다.

```python
from flair.embeddings import FlairEmbeddings

# 임베딩 초기화
flair_embedding_forward = FlairEmbeddings('news-forward')

# 문장 만들기
sentence = Sentence('The grass is green .')

# 문장에 단어 삽입
flair_embedding_forward.embed(sentence)
```

`FlairEmbedings` 클래스의 생성자에게 적절한 문자열을 전달하여 로드할 임베딩을 선택합니다. 
지원되는 모든 언어에는 전진 및 후진 모델이 있습니다. 
**2글자 언어 코드**에 이어 하이픈 및 **앞으로** 또는 **뒤로**를 사용하여 언어의 모델을 로드할 수 있습니다. 
독일어 Flair 모델을 앞뒤로 로드하려면 다음과 같이 하십시오.

```python
# init forward embedding for German
flair_embedding_forward = FlairEmbeddings('de-forward')
flair_embedding_backward = FlairEmbeddings('de-backward')
```

표준 사용에 대한 자세한 정보와 함께 사전 훈련된 모든 FlairEmbedings 모델 [여기](/리소스/docs/embeddings/FLAIR_EMBEDDINGS.md)의 전체 목록을 확인하십시오.

## 스택 임베딩

스택형 임베딩은 이 라이브러리의 가장 중요한 개념 중 하나입니다. 예를 들어, 두 개의 기존 임베딩을 모두 상황에 맞는 문자열과 함께 사용하려는 경우 이러한 임베딩을 사용하여 서로 다른 임베딩을 함께 결합할 수 있습니다.
스택형 임베딩을 사용하면 혼합 및 일치시킬 수 있습니다.

`StackedEmbedings` 클래스를 사용하여 결합하고자 하는 임베딩 목록을 전달하여 인스턴스화하기만 하면 됩니다.

예를 들어, 고전적인 GloVe 임베딩을 전방 및 후방 Flair 임베딩과 결합해봅시다. 이는 특히 시퀀스 라벨링에 대해 일반적으로 권장하는 조합입니다.

먼저 결합할 두 개의 임베딩을 인스턴스화합니다.

```python
from flair.embeddings import WordEmbeddings, FlairEmbeddings

# 표준 GloVe 임베딩 초기화
glove_embedding = WordEmbeddings('glove')

# Flair 정방향 및 역방향 임베딩 초기화
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')
```

이제 'StackedEmbedings' 클래스를 인스턴스화하고 두 개의 임베딩이 포함된 목록을 전달합니다.

```python
from flair.embeddings import StackedEmbeddings

# glove와 정방향 및 역방향 Flair 임베딩을 결합하는 StackedEmbedding 개체를 만들기
stacked_embeddings = StackedEmbeddings([
                                        glove_embedding,
                                        flair_embedding_forward,
                                        flair_embedding_backward,
                                       ])
```

이제 이 임베딩을 다른 모든 임베딩과 마찬가지로 사용하면 됩니다. 즉, 문장 위에 'embed()' 방법을 사용합니다.

```python
sentence = Sentence('The grass is green .')

# 단일 임베딩에서와 같이 StackedEmbedding을 사용하여 문장을 임베드하기만 하면 됩니다.
stacked_embeddings.embed(sentence)

# 이제 임베딩된 토큰을 확인하십시오.
for token in sentence:
    print(token)
    print(token.embedding)
```

단어들은 이제 세 가지 다른 임베딩의 연결을 사용하여 내장됩니다. 이는 결과 임베딩 벡터가 여전히 단일 PyTorch 벡터임을 의미합니다.

## Next 
이러한 임베딩에 대한 자세한 내용과 지원되는 모든 단어 임베딩에 대한 전체 개요를 보려면 다음을 참조하십시오.
[TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING](/resources/docs/KOR_docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md). 
