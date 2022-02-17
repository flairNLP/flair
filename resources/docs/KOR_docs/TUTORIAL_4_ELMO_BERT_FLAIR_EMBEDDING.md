# 튜토리얼 4 : Word Embedding의 종류
이번 챕터는 튜토리얼이라기보다 Flair에서 지원하는 Embedding의 종류를 소개합니다. 아래 테이블의 Embedding을 클릭해 사용법을 볼 수 있습니다. 설명들은 [base types](/resources/docs/KOR_docs/TUTORIAL_1_BASICS.md)과 [standard word embeddings](/resources/docs/KOR_docs/TUTORIAL_3_WORD_EMBEDDING.md), 그리고 `StackedEmbeddings`클래스에 익숙하다는 전제로 작성되어 있습니다.

## 개요
모든 word embedding 클래스들은 `TokenEmbeddings` 클래스를 상속하고 있으며 텍스트를 임베드 하기 위해 `embed()` 메소드를 호출합니다. Flair를 사용하는 대부분의 경우 다양하고 복잡한 embedding 과정이 인터페이스 뒤로 숨겨져 있습니다. 사용자는 단순히 필요한 embedding 클래스를 인스턴스화하고 `embed()`를 호출해 텍스트를 임베드 하면 됩니다.

현재 지원하고 있는 임베딩의 종류입니다 :

| Class | Type | Paper | 
| ------------- | -------------  | -------------  | 
| [`BytePairEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md) | Subword-level word embeddings | [Heinzerling and Strube (2018)](https://www.aclweb.org/anthology/L18-1473)  |
| [`CharacterEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/CHARACTER_EMBEDDINGS.md) | Task-trained character-level embeddings of words | [Lample et al. (2016)](https://www.aclweb.org/anthology/N16-1030) |
| [`ELMoEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/ELMO_EMBEDDINGS.md) | Contextualized word-level embeddings | [Peters et al. (2018)](https://aclweb.org/anthology/N18-1202)  |
| [`FastTextEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/FASTTEXT_EMBEDDINGS.md) | Word embeddings with subword features | [Bojanowski et al. (2017)](https://aclweb.org/anthology/Q17-1010)  |
| [`FlairEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md) | Contextualized character-level embeddings | [Akbik et al. (2018)](https://www.aclweb.org/anthology/C18-1139/)  |
| [`OneHotEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/ONE_HOT_EMBEDDINGS.md) | Standard one-hot embeddings of text or tags | - |
| [`PooledFlairEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md) | Pooled variant of `FlairEmbeddings` |  [Akbik et al. (2019)](https://www.aclweb.org/anthology/N19-1078/)  | 
| [`TransformerWordEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md) | Embeddings from pretrained [transformers](https://huggingface.co/transformers/pretrained_models.html) (BERT, XLM, GPT, RoBERTa, XLNet, DistilBERT etc.) | [Devlin et al. (2018)](https://www.aclweb.org/anthology/N19-1423/) [Radford et al. (2018)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  [Liu et al. (2019)](https://arxiv.org/abs/1907.11692) [Dai et al. (2019)](https://arxiv.org/abs/1901.02860) [Yang et al. (2019)](https://arxiv.org/abs/1906.08237) [Lample and Conneau (2019)](https://arxiv.org/abs/1901.07291) |  
| [`WordEmbeddings`](https://github.com/flairNLP/flair/tree/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md) | Classic word embeddings |  |

## BERT와 Flair 조합하기
우리는 Flair, ELMo, BERT 그리고 고전적 word embedding을 쉽게 결합할 수 있습니다. 조합하려는 임베딩을 각각 인스턴스화하고 `StackedEmbedding`에서 사용하면 됩니다.
아래는 다국어 Flair와 BERT 임베딩을 사용해 강력한 다국어 다운스트림 작업 모델을 훈련하는 예시입니다.

우선 조합하고자 하는 임베딩을 인스턴스화합니다.
```python
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings

# Flair 임베딩 초기화
flair_forward_embedding = FlairEmbeddings('multi-forward')
flair_backward_embedding = FlairEmbeddings('multi-backward')

# 다국어 BERT 초기화
bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased')
```

이제 `StackedEmbeddings` 클래스를 초기화 하고 앞에서 초기화한 세가지 임베딩이 포함된 목록을 전달합니다.

```python
from flair.embeddings import StackedEmbeddings

# 앞에서 초기화 한 임베딩을 결합한 StackedEmbedding 객체를 생성합니다.
stacked_embeddings = StackedEmbeddings(
    embeddings=[flair_forward_embedding, flair_backward_embedding, bert_embedding])
```

완성입니다! 다른 임베딩을 사용하는 것과 마찬가지로 문장에 대해 `embed()` 메서드를 호출하면 됩니다.

```python
sentence = Sentence('The grass is green .')

# 단일 임베딩을 사용하는 것과 마찬가지로 StackedEmbedding을 사용합니다.
stacked_embeddings.embed(sentence)

# 문장에 대한 Token을 확인합니다.
for token in sentence:
    print(token)
    print(token.embedding)
```

단어들은 세 가지 다른 임베딩이 조합된 것으로 임베드 되었습니다. output은 여전히 PyTorch 벡터입니다.

## 다음 튜토리얼
텍스트 분류와 같은 작업을 위해 전체 텍스트 [문서를 임베드](/resources/docs/KOR_docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md)하는 튜토리얼 혹은 [나만의 모델을 훈련](/resources/docs/KOR_docs/TUTORIAL_7_TRAINING_A_MODEL.md)하기 위한 전제조건인 [말뭉치(corpus)를 로드](/resources/docs/KOR_docs/TUTORIAL_6_CORPUS.md)하는 튜토리얼이 준비되어 있습니다.
