# 튜토리얼 5: 문서 임베딩
우리가 앞서 살펴본 [단어 임베딩](/resources/docs/KOR_docs/TUTORIAL_3_WORD_EMBEDDING.md)은 개별 단어에 대한 임베딩을 제공했습니다. 이번에 살펴볼 문서 임베딩은 전체 텍스트에 대해 하나의 임베딩을 제공합니다.

이번 튜토리얼은 여러분이 라이브러리의 [기본 유형](/resources/docs/KOR_docs/TUTORIAL_1_BASICS.md)과 [단어 임베딩](/resources/docs/KOR_docs/TUTORIAL_3_WORD_EMBEDDING.md)이 어떻게 동작하는지 익숙하다 가정하고 진행됩니다.

## 임베딩
모든 문서 임베딩 클래스들은 `DocumentEmbeddings` 클래스를 상속하며 텍스트를 임베드 하기 위해 `embed()` 메소드를 호출합니다.
Flair를 사용하는 대부분의 경우 다양하고 복잡한 embedding 과정이 인터페이스 뒤로 숨겨져 있습니다.

Flair에는 4가지 주요 문서 임베딩이 있습니다.

 1. `DocumentPoolEmbeddings` 문장 속 모든 단어의 평균을 단순하게 도출합니다.
 2. `DocumentRNNEmbeddings` 문장 속 모든 단어들로 RNN을 훈련시킵니다.
 3. `TransformerDocumentEmbeddings` 미리 훈련된 변환기를 사용합니다. 대부분의 텍스트 분류 작업에 **권장**합니다.
 4. `SentenceTransformerDocumentEmbeddings` 미리 훈련된 변환기를 사용합니다. 문장의 벡터 표현을 필요로 할 때 *권장*합니다.

네 가지 옵션 중 하나를 선택해 초기화하고 `embed()` 메서드를 호출해 텍스트를 임베드 합니다.

아래는 네 가지 문서 임베딩에 대한 세부정보입니다 :

## Documnet Pool Embeddings
문서 임베딩 중 가장 단순한 유형입니다. 전체 문장에 대한 임베딩을 얻기 위해 문장 속 모든 단어 임베딩에 대해 풀링 연산을 합니다.
디폴트는 평균 풀링입니다. 이는 모든 단어 임베딩의 평균을 사용합니다.

인스턴스화 하기 위해 다음과 같은 임베딩 리스트를 사용합니다.
```python
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

# 워드 임베딩을 초기화합니다.
glove_embedding = WordEmbeddings('glove')

# 문서 임베딩을 초기화합니다. mode = mean
document_embeddings = DocumentPoolEmbeddings([glove_embedding])
```
이제 `embed()` 메소드를 호출해 문장을 임베드 합니다.
```python
# 예시 문장입니다.
sentence = Sentence('The grass is green . And the sky is blue .')

# 문서 임베딩에 문장을 임베드합니다.
document_embeddings.embed(sentence)

# 임베드된 문장을 확인합니다.
print(sentence.embedding)
```
문서 임베딩은 단어 임베딩에서 파생됩니다. 그렇기 때문에 단어 임베딩의 차원에 따라 문서의 차원이 달라집니다. 더 자세한 내용은 [여기](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/DOCUMENT_POOL_EMBEDDINGS.md)를 참조해주세요.

`DocumentPoolEmbeddings`은 교육할 필요가 없으며 즉각적으로 문서를 임베딩해 사용할 수 있는 장점이 있습니다.

## Document RNN Embeddings
RNN 임베딩을 사용하기 위해 문장의 모든 단어에 대해 RNN을 실행하고 RNN의 최종 state를 전체 문서에 대한 임베딩으로 사용합니다.
이를 사용하기 위해 `DocumentRNNEmbeddings`를 토큰 임베딩 목록을 전달하는 것을 통해 초기화합니다.

```python
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

glove_embedding = WordEmbeddings('glove')

document_embeddings = DocumentRNNEmbeddings([glove_embedding])
```
디폴트로 GRU-type RNN이 인스턴스화됩니다. 예제 문장을 만들고 임베딩의 `embed()` 메소드를 호출합니다.

```python
# 예시 문장입니다.
sentence = Sentence('The grass is green . And the sky is blue .')

# 문서 임베딩에 문장을 임베드합니다.
document_embeddings.embed(sentence)

# 임베드된 문장을 확인합니다.
print(sentence.get_embedding())
```
결과물은 전체 문장에 대한 단일 임베딩입니다. 임베딩 차원은 hidden state의 개수와 RNN이 양방향인지 아닌지에 따라 달라집니다. 더 자세한 내용은 [여기](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/DOCUMENT_RNN_EMBEDDINGS.md)를 참조해주세요.

**주의** RNN 임베딩을 초기화하면 RNN 가중치가 무작위로 초기화됩니다! 사용을 위해서 사용자의 의도에 알맞게 훈련돼야 합니다.

## TransformerDocumentEmbeddings
이미 훈련된 [변환기](https://github.com/huggingface/transformers)를 통해 전체 문장을 임베딩합니다. 임베딩의 식별자를 통해 다른 변환기를 사용할 수 있습니다.

표준 BERT 변환 모델의 예시입니다:
```python
from flair.embeddings import TransformerDocumentEmbeddings

# 임베딩을 초기화합니다.
embedding = TransformerDocumentEmbeddings('bert-base-uncased')

# 예시 문장입니다.
sentence = Sentence('The grass is green .')

# 문장을 임베딩합니다.
embedding.embed(sentence)
```

RoBERTa의 예시입니다:
```python
from flair.embeddings import TransformerDocumentEmbeddings

# 임베딩을 초기화합니다.
embedding = TransformerDocumentEmbeddings('roberta-base')

# 예시 문장입니다.
sentence = Sentence('The grass is green .')

# 문장을 임베딩합니다.
embedding.embed(sentence)
```

[여기](https://huggingface.co/transformers/pretrained_models.html)에서 모든 모델들의 리스트를 확인할 수 있습니다(BERT, RoBERTa, XLM, XLNet 기타 등등). 이 클래스를 통해 모델들을 사용할 수 있습니다.

## SentenceTransformerDocumentEmbeddings
[`sentence-transformer`](https://github.com/UKPLab/sentence-transformers) 라이브러리에서 다른 임베딩을 사용할 수도 있습니다. 이 모델들은 사전 훈련된 것으로 범용 벡터 표현을 제공합니다.
```python
from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings

# 임베딩을 초기화합니다.
embedding = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')

# 예시 문장입니다.
sentence = Sentence('The grass is green .')

# 문장을 임베딩합니다.
embedding.embed(sentence)
```
[여기](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0) 에서 사전 훈련된 모델의 전체 리스트를 확인할 수 있습니다.

**참고**: 이 임베딩을 사용하기 위해서 `sentence-transformers`를 설치해야 합니다.

`pip install sentence-transformers`. 

## 다음 튜토리얼
[나만의 모델을 훈련](/resources/docs/KOR_docs/TUTORIAL_7_TRAINING_A_MODEL.md)하기 위한 전제조건인 [말뭉치(corpus)를 로드](/resources/docs/KOR_docs/TUTORIAL_6_CORPUS.md)하는 튜토리얼이 준비되어 있습니다.
