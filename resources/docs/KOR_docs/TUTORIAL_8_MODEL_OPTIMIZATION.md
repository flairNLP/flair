# 튜토리얼 8: 모델 튜닝

튜토리얼 8에서는 올바른 모델 및 하이퍼 매개변수 셋을 선택하여 모델의 품질을 향상시킬 수 있는 방법을 살펴볼 것입니다.

## Hyper Parameter 선택하기

Flair에는 잘 알려진 하이퍼 매개변수 선택 도구인 [hyperopt](https://github.com/hyperopt/hyperopt)에 대한 래퍼가 포함되어 있습니다.

먼저 말뭉치를 로드해야 합니다. 다음 예에서 사용된 [AGNews corpus](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)를 로드하려면
먼저 다운로드하여 올바른 형식으로 변환하세요.
자세한 내용은 [tutorial 6](/resources/docs/KOR_docs/TUTORIAL_6_CORPUS.md)을 확인하세요.

```python
from flair.datasets import WNUT_17

# 당신의 말뭉치를 로드하세요.
corpus = WNUT_17().downsample(0.1)
```

두 번째로는 매개변수의 검색 공간을 정의해야 합니다. 이를 통해 hyperopt에서 정의한 모든 [parameter expressions](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions)을 사용할 수 있습니다.

```python
from hyperopt import hp
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.hyperparameter.param_selection import SearchSpace, Parameter

# 검색 공간을 정의하세요.
search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    WordEmbeddings('en'),
    StackedEmbeddings([FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')])
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])
```

Attention: 항상 검색 공간에 임베딩을 추가해야 합니다(위 그림 참조). 다른 종류의 임베딩을 테스트하지 않으려면 검색 공간에 하나의 임베딩 옵션만 전달하면 됩니다. 그러면 모든 테스트 실행에서 사용될 것입니다.

마지막 단계에서 실제 매개변수 선택기를 생성해야 합니다.
작업에 따라 `TextClassifierParamSelector` 또는 `SequenceTaggerParamSelector`를 정의하고 최적화를 시작해야 합니다.
hyperopt가 수행해야 하는 최대 평가 실행 횟수를 정의할 수 있습니다(`max_evals`). 평가 실행은 지정된 수의 epoch(`max_epochs`)를 수행합니다.
시끄러운 평가 점수 문제를 극복하기 위해 평가 실행에서 마지막 세 평가 점수('dev_score' 또는 'dev_loss')에 대한 평균을 취합니다. 이 점수는 최종 점수를 나타내며 hyperopt에 전달됩니다.
또한 평가 실행당 실행 횟수(`training_runs`)를 지정할 수 있습니다.
둘 이상의 훈련 실행을 지정하는 경우 하나의 평가 실행이 지정된 횟수만큼 실행됩니다.
최종 평가 점수는 모든 실행에 대한 평균이 됩니다.

```python
from flair.hyperparameter.param_selection import SequenceTaggerParamSelector

# 매개변수 선택기 생성
param_selector = SequenceTaggerParamSelector(corpus,
                                             'ner',
                                             'resources/results',
                                             training_runs=3,
                                             max_epochs=50
                                             )

# 최적화 시작
param_selector.optimize(search_space, max_evals=100)
```

매개변수 설정 및 평가 점수는 결과 디렉토리의 'param_selection.txt'에 기록됩니다.
최상의 매개변수 조합을 선택하는 동안 어떤 모델도 디스크에 저장하지 않으며 테스트 실행 또한 수행하지 않습니다.
훈련 중에 로깅 목적으로 테스트 세트에 대한 훈련 후 모델을 한 번만 평가합니다.

## 최고의 학습률 찾기


학습률은 가장 중요한 하이퍼 매개변수 중 하나이며 기본적으로 모델의 아키텍처와 모델이 사용하는 교육 데이터를 통한 손실 환경의 토폴로지에 따라 다릅니다.
최적의 학습은 훈련 속도를 향상시키고 더 나은 성능의 모델을 제공할 것입니다. Leslie Smith가 설명한 간단한 기술
[Cyclical Learning Rates for Training](https://arxiv.org/abs/1506.01186) 논문은 매우 낮은 학습률로 시작하여
SGD의 모든 배치 업데이트에서 학습률을 기하급수적으로 증가시키는 모델을 학습시키는 것입니다. 우리는 손실을 플로팅하여
학습률과 관련하여 일반적으로 세 가지 별개의 단계를 관찰할 것입니다:
낮은 학습률의 경우 손실이 개선되지 않으며, 손실이 가장 급격하게 떨어지는 최적의 학습률 범위와 학습률이 너무 커지면 손실이 폭발하는 최종 단계입니다.
이러한 플롯을 사용하면 최적의 학습률을 선택하는 것이 최적의 단계에서 가장 높은 것을 선택하는 것만큼 쉽습니다.

이러한 실험을 실행하려면 초기화된 'ModelTrainer'로 시작하고 학습률과 손실을 기록할 파일 이름과 'base_path'와 함께 'find_learning_rate()'를 호출하십시오.
그런 다음 `Plotter`의 `plot_learning_rate()` 함수를 통해 생성된 결과를 플롯하고 `learning_rate.png` 이미지를 보고 최적의 학습률을 선택하세요:

```python
from flair.datasets import WNUT_17
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.trainers import ModelTrainer
from typing import List
from torch.optim.adam import Adam
# 1. 말뭉치 가져오기
corpus = WNUT_17().downsample(0.1)
print(corpus)
# 2. 우리는 예측하고 싶은 태그는 무엇인가요?
tag_type = 'ner'
# 3. 말뭉치에서 태그 사전 만들기
tag_dictionary = corpus.make_label_dictionary(label_type=tag_type, add_unk=False)
print(tag_dictionary.idx2item)
# 4. 임베딩 초기화하기
embedding_types: list[TokenEmbeddings] = [
    WordEmbeddings('glove'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
# 5. 시퀀스 tagger 초기화하기
from flair.models import SequenceTagger
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
# 6. 트레이너 초기화하기
trainer: ModelTrainer = ModelTrainer(tagger, corpus)
# 7. 학습률 찾기
learning_rate_tsv = trainer.find_learning_rate('resources/taggers/example-ner', Adam)
# 8. 학습률 찾기 곡선 그리기
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_learning_rate(learning_rate_tsv)
```

## Next

다음 튜토리얼에서는 [training your own embeddings](/resources/docs/KOR_docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md)에 대해 살펴볼 것입니다,
