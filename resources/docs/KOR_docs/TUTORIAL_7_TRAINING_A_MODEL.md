# 튜토리얼 7: 모델 훈련하기

튜토리얼 7에서는 최첨단 word embedding을 사용하여 여러분의 시퀀스 레이블링(sequence labeling)과 텍스트 분류(text classification) 모델을
훈련하는 방법을 살펴볼 것입니다.

이 튜토리얼을 학습하기 전에, 다음의 항목들을 이미 알고있다고 가정할 것입니다.
* Base types: [TUTORIAL_1_BASICS](/resources/docs/KOR_docs/TUTORIAL_1_BASICS.md)
* Word embeddings: [TUTORIAL_3_WORD_EMBEDDING](/resources/docs/KOR_docs/TUTORIAL_3_WORD_EMBEDDING.md)
* Flair embeddings: [TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING](/resources/docs/KOR_docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)
* Load a corpus: [TUTORIAL_6_CORPUS](/resources/docs/KOR_docs/TUTORIAL_6_CORPUS.md)



## 품사 Tagging 모델 훈련

다음 예제는 간단한 글로브(Glove) 임베딩을 이용하여 UD_ENGLISH (English universal dependency treebank) 데이터를 통해 훈련된 작은 품사 tagger 모델에 대한 코드입니다.
이 예제에서는 더 빠르게 작동시키기 위해 기존 데이터의 10%로 다운샘플링하여 진행했지만, 보통의 경우에는 전체 데이터셋으로 훈련시켜야 합니다:

```python
from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
# 1. 말뭉치 가져오기
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)
# 2. 어떤 레이블을 예측하고 싶으신가요?
label_type = 'upos'
# 3. 말뭉치에서 레이블 사전 만들기
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)
# 4. 임베딩 초기화하기
embedding_types = [
    WordEmbeddings('glove'),
    # comment in this line to use character embeddings
    # CharacterEmbeddings(),
    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]
embeddings = StackedEmbeddings(embeddings=embedding_types)
# 5. 시퀀스 tagger 초기화하기
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)
# 6. 트레이너 초기화하기
trainer = ModelTrainer(tagger, corpus)
# 7. 훈련 시작
trainer.train('resources/taggers/example-upos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)
```


또는 전체 데이터에 대해 FlairEmbeddings 및 GloVe와 함께 누적된 임베딩을 150 epochs 만큼 (전체 데이터를 150번 훈련) 사용해 보세요.
그렇게 하면 [Akbik et al. (2018)](https://aclanthology.org/C18-1139.pdf)에 보고된 최신 정확도를 얻을 수 있습니다.

모델이 학습되면 이를 사용하여 새 문장의 태그를 예측할 수 있습니다. 모델의 'predict' 메서드를 호출하기만 하면 됩니다.

```python
# 훈련한 모델 로드하기
model = SequenceTagger.load('resources/taggers/example-pos/final-model.pt')
# 예시 문장 만들기
sentence = Sentence('I love Berlin')
# 태그 예측하고 출력하기
model.predict(sentence)
print(sentence.to_tagged_string())
```

모델이 잘 작동한다면 이 예에서 동사로 'love'를 올바르게 태그할 것입니다.

## Flair Embedding으로 개채명 인식 (NER) 모델 훈련하기


NER에 대한 시퀀스 레이블링 모델을 훈련하려면 위의 스크립트를 약간만 수정하면 됩니다.
CONLL_03(데이터를 수동으로 다운로드하거나 [different NER corpus](/resources/docs/KOR_docs/TUTORIAL_6_CORPUS.md#datasets-included-in-flair) 사용)과 같은 NER corpus를 로드하고,
`label_type'을 'ner'로 변경한 후, GloVe 및 Flair로 구성된 'StackedEmbedding'을 사용하세요:

```python
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
# 1. 말뭉치 가져오기
corpus = CONLL_03()
print(corpus)
# 2. 어떤 레이블을 예측하고 싶으신가요?
label_type = 'ner'
# 3. 말뭉치에서 레이블 사전 만들기
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)
# 4. Flair 및 GloVe로 임베딩 스택 초기화하기
embedding_types = [
    WordEmbeddings('glove'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]
embeddings = StackedEmbeddings(embeddings=embedding_types)
# 5. 시퀀스 tagger 초기화하기
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)
# 6. 트레이너 초기화하기
trainer = ModelTrainer(tagger, corpus)
# 7. 훈련 시작
trainer.train('resources/taggers/sota-ner-flair',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
```


그렇게 하면 [Akbik et al. (2018)](https://aclanthology.org/C18-1139.pdf)에 보고된 것과 유사한 최근 숫자들이 나올 것입니다.

## 변환기를 사용하여 개체명 인식 (NER) 모델 훈련하기

임베딩으로 변환기를 사용하고 미세 조정하고 전체 문서 컨텍스트를 사용하면 **훨씬 더 나은 수치**를 얻을 수 있습니다. (자세한 내용은 [FLERT](https://arxiv.org/abs/2011.06993) 문서 참조)
이는 최신식이지만 위의 모델보다 훨씬 느립니다.

변환기 임베딩을 사용하도록 스크립트를 변경하고 SGD 대신 AdamW optimizer 및 작은 학습률로 미세 조정하도록 훈련 루틴을 변경하세요:

```python
from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import torch
from torch.optim.lr_scheduler import OneCycleLR
# 1. 말뭉치 가져오기
corpus = CONLL_03()
print(corpus)
# 2. 어떤 레이블을 예측하고 싶으신가요?
label_type = 'ner'
# 3. 말뭉치에서 레이블 사전 만들기
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)
# 4. 문서 컨텍스트로 미세 조정 가능한 변환기 임베딩 초기화
embeddings = TransformerWordEmbeddings(
    model='xlm-roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)
# 5. bare-bones 시퀀스 태거 초기화하기 (CRF 없음, RNN 없음, 재투영 없음)
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type='ner',
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)
# 6. AdamW 옵티마이저로 트레이너 초기화하기
trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)
# 7. XLM 매개변수로 훈련 실행(20 epochs, 작은 LR, 1주기 학습률 스케줄링)
trainer.train('resources/taggers/sota-ner-flert',
              learning_rate=5.0e-6,
              mini_batch_size=4,
              mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
              max_epochs=20,  # 10 is also good
              scheduler=OneCycleLR,
              embeddings_storage_mode='none',
              weight_decay=0.,
              )
```

이는 [Schweter and Akbik (2021)](https://arxiv.org/abs/2011.06993)에 보고된 최근 수치와 비슷하게 나올 것입니다.

## 텍스트 분류 모델 훈련하기

다른 유형의 모델을 훈련시키는 것은 위의 시퀀스 레이블러를 교육하기 위한 스크립트와 매우 유사합니다. 텍스트 분류의 경우 적절한 말뭉치를 사용하고
word-level 임베딩 대신 document-level 임베딩을 사용하세요. (차이점은 이 둘에 대한 튜토리얼을 참조하세요.) 나머지는 이전과 동일합니다!

텍스트 분류에서 가장 좋은 결과는 아래 코드와 같이 `TransformerDocumentEmbeddings`와 함께 미세 조정된 변환기를 사용합니다:

(변환기를 미세 조정할 수 있는 큰 GPU가 없는 경우 대신 `DocumentPoolEmbeddings` 또는 `DocumentRNNEmbeddings`를 사용해 보세요.
가끔 제대로 작동하기도 합니다!)

```python
import torch
from torch.optim.lr_scheduler import OneCycleLR
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
# 1. 말뭉치 가져오기
corpus: Corpus = TREC_6()
# 2. 어떤 레이블을 예측하고 싶으신가요?
label_type = 'question_class'
# 3. 레이블 사전 만들기
label_dict = corpus.make_label_dictionary(label_type=label_type)
# 4. 변환기 문서 임베딩 초기화하기 (많은 모델 사용 가능)
document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)
# 5. 텍스트 분류 만들기
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)
# 6. AdamW 옵티마이저로 트레이너 초기화하기
trainer = ModelTrainer(classifier, corpus, optimizer=torch.optim.AdamW)
# 7. 미세 조정으로 훈련 실행
trainer.train('resources/taggers/question-classification-with-transformer',
              learning_rate=5.0e-5,
              mini_batch_size=4,
              max_epochs=10,
              scheduler=OneCycleLR,
              embeddings_storage_mode='none',
              weight_decay=0.,
              )
```

모델이 학습되면 이것을 로드하여 새로운 문장의 클래스를 예측할 수 있습니다. 모델의 'predict' 메서드를 호출하기만 하면 됩니다.

```python
classifier = TextClassifier.load('resources/taggers/question-classification-with-transformer/final-model.pt')
# 예시 문장 만들기
sentence = Sentence('Who built the Eiffel Tower ?')
# 클래스를 예측하고 출력하기
classifier.predict(sentence)
print(sentence.labels)
```

## 멀티 데이터셋 훈련하기

이제 영어와 독일어로 텍스트에 PoS 태그를 지정할 수 있는 단일 모델을 훈련해 보겠습니다. 이를 위해 영어 및 독일어 UD 말뭉치를 로드하고 멀티 말뭉치 개체를 만듭니다. 이 작업을 위해 새로운 다국어 Flair 임베딩을 사용할 것입니다. 나머지는 모두 이전과 동일합니다.
e.g.:

```python
from flair.data import MultiCorpus
from flair.datasets import UD_ENGLISH, UD_GERMAN
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
# 1. 말뭉치 가져오기 - 영어 및 독일어 UD
corpus = MultiCorpus([UD_ENGLISH(), UD_GERMAN()]).downsample(0.1)
# 2. 어떤 레이블을 예측하고 싶으신가요?
label_type = 'upos'
# 3. 말뭉치에서 레이블 사전 만들기
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)
# 4. 임베딩 초기화하기
embedding_types = [
    # we use multilingual Flair embeddings in this task
    FlairEmbeddings('multi-forward'),
    FlairEmbeddings('multi-backward'),
]
embeddings = StackedEmbeddings(embeddings=embedding_types)
# 5. 시퀀스 tagger 초기화하기
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)
# 6. 트레이너 초기화하기
trainer = ModelTrainer(tagger, corpus)
# 7. 훈련 시작
trainer.train('resources/taggers/example-universal-pos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              )
```

이는 다국어 모델을 제공합니다. 더 많은 언어로 실험해 보세요!

## 훈련 곡선 및 가중치 Plotting

Flair에는 신경망에서 훈련 곡선과 가중치를 표시하는 도우미 메서드가 포함되어 있습니다. `ModelTrainer`는 결과 폴더에 `loss.tsv`를 자동으로
생성합니다. 훈련 중에 `write_weights=True`로 설정하면 `weights.txt` 파일도 생성됩니다.

훈련 후 plotter가 다음 파일을 가리킬 것입니다:

```python
# 가중치를 쓰려면 write_weights를 True로 설정하세요.
trainer.train('resources/taggers/example-universal-pos',
              ...
write_weights = True,
                ...
)
# 가시화하기
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('loss.tsv')
plotter.plot_weights('weights.txt')
```

결과 폴더에 PNG 플롯이 생성될 것입니다.

## 훈런 재개

만약 특정 시점에서 훈련을 중지하고 나중에 다시 시작하려면 'checkpoint' 매개변수를 'True'로 설정하여 학습해야 합니다. 그렇게 하면 매 epoch 후에
모델과 훈련 매개변수를 저장할 것입니다.
따라서 나중에 언제든지 모델과 트레이너를 로드하고 남은 위치에서 정확히 훈련을 계속할 수 있습니다.

아래 예제 코드는 `SequenceTagger`의 훈련, 중지 및 계속 훈련 방법을 보여줍니다. 'TextClassifier'의 경우도 마찬가지입니다.

```python
from flair.data import Corpus
from flair.datasets import WNUT_17
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
# 1. 말뭉치 가져오기
corpus: Corpus = WNUT_17().downsample(0.1)
# 2. 어떤 레이블을 예측하고 싶으신가요?
label_type = 'ner'
# 3. 말뭉치에서 레이블 사전 만들기
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
# 4. 임베딩 초기화하기
embedding_types: list[TokenEmbeddings] = [
    WordEmbeddings('glove')
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
# 5. 시퀀스 tagger 초기화하기
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=label_dict,
                                        tag_type=label_type,
                                        use_crf=True)
# 6. 트레이너 초기화하기
trainer: ModelTrainer = ModelTrainer(tagger, corpus)
# 7. 훈련 시작
trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10,
              checkpoint=True)
# 8. 언제든지 훈련을 중단하세요.
# 9. 나중에 트레이너를 계속하세요.
checkpoint = 'resources/taggers/example-ner/checkpoint.pt'
trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)
trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              checkpoint=True)
```

## Scalability: 대규모 데이터셋으로 훈련하기

Flair의 많은 임베딩은 런타임 측면에서 생성하는 데 다소 비용이 많이 들고 큰 벡터를 가질 수 있습니다. 이에 대한 예는 Flair 및 Transformer 기반 임베딩입니다. 설정에 따라 훈련 시간을 최적화하는 옵션을 설정할 수 있습니다.

### Mini-Batch 크기 설정

가장 중요한 것은 `mini_batch_size`입니다. GPU가 속도 향상을 위해 처리할 수 있는 경우 더 높은 값으로 설정하세요.
그러나 데이터 세트가 매우 작은 경우 너무 높게 설정하지 마세요. 그렇지 않으면 Epoch당 학습 단계가 충분하지 않을 것입니다.

유사한 매개변수는 `mini_batch_chunk_size`입니다. 이 매개변수는 미니 배치를 청크로 더 분할하여 속도를 늦추지만 GPU 메모리 효율성을
향상시킵니다. 표준은 이것을 None으로 설정하는 것입니다 - GPU가 원하는 미니 배치 크기를 처리할 수 없는 경우에만 이것을 설정하세요.
이는 `mini_batch_size`의 반대이므로 계산 속도가 느려질 것입니다.

### Embedding의 저장 모드 설정

설정해야 하는 또 다른 주요 매개변수는 `ModelTrainer`의 `train()` 메서드에 있는 `embeddings_storage_mode`입니다.
다음 세 가지 값 중 하나를 가질 수 있습니다:

1. **'none'**: `embeddings_storage_mode='none'`으로 설정하면 임베딩이 메모리에 저장되지 않습니다. 대신 (**훈련** 동안) 각 훈련 미니 배치에서 즉석에서 생성됩니다. 주요한 이점은 메모리 요구 사항을 낮게 유지한다는 것입니다. 변압기를 미세 조정하는 경우 항상 이것을 설정하세요.



2. **'cpu'**: `embeddings_storage_mode='cpu'`를 설정하면 임베딩이 일반 메모리에 저장될 것입니다.

* during *training*: 임베딩은 첫 번째 epoch에서만 계산되고 그 후에는 메모리에서 검색되기 때문에 많은 경우에 속도가 크게 빨라집니다. 이것의 단점은 메모리 요구 사항이 증가한다는 것입니다. 데이터셋의 크기와 메모리 설정에 따라 이 옵션이 불가능할 수 있습니다.
* during *inference*: GPU 메모리에서 일반 메모리로 임베딩을 이동해야 하므로 GPU와 함께 사용할 때 추론 속도가 느려집니다. 추론 중에 이 옵션을 사용하는 유일한 이유는 예측뿐만 아니라 예측 후 임베딩도 사용하기 위해서입니다.

3. **'gpu'**: `embeddings_storage_mode='gpu'`로 설정하면 임베딩은 CUDA 메모리에 저장될 것입니다. 이는 CPU에서 CUDA로 텐서를 계속해서 섞을 필요가 없기 때문에 가장 빠른 경우가 많습니다. 물론 CUDA 메모리는 종종 제한되어 있어 큰 데이터셋은 CUDA 메모리에 맞지 않습니다. 하지만 데이터셋이 CUDA 메모리에 맞는 경우에는 이 옵션이 가장 빠릅니다.


## Next

훈련 데이터가 없거나 아주 적은 경우 TARS 접근 방식이 가장 적합할 수 있습니다.
[TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL](/resources/docs/KOR_docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md): Few-shot and zero-shot classification에 대한 TARS 튜토리얼을 확인하세요.

또는
[TUTORIAL_9_TRAINING_LM_EMBEDDINGS](/resources/docs/KOR_docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md): Training your own embeddings을 살펴보세요.
