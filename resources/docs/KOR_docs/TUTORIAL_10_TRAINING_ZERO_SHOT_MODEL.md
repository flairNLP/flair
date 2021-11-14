# 튜토리얼 10: Few-Shot과 Zero-Shot 분류 (TARS)

TARS(Task-aware representation of sentence)는 [Halder et al. (2020)](https://kishaloyhalder.github.io/pdfs/tars_coling2020.pdf)이 **텍스트 분류를 위한 퓨샷 및 제로샷 학습**을 위한 간단하고 효과적인 방법으로 도입했습니다.
이것은 훈련 예제 없이도 텍스트를 분류할 수 있음을 의미합니다.
이 모델은 Flair에서 'TASClassifier' 클래스로 구현됩니다.
 
이번 튜토리얼에서는 TARS를 사용하는 다양한 방법을 보여줄 것입니다:

    
## 사용 사례 #1: 훈련 데이터 없이 텍스트 분류(Zero-Shot)

때로 우리는 해결하려는 텍스트 분류 작업에 대한 훈련 데이터가 없을 때가 있습니다. 이 경우 기본 TARS 모델을 로드하고 제로샷 예측을 수행할 수 있습니다.   
즉, TARS의 `predict_zero_shot` 방법을 사용하고 레이블 이름 목록을 제공하는 것입니다. 그런 다음 TARS는 이러한 레이블 중 하나를 텍스트와 일치시키려고 시도할 것입니다.

예를 들어 텍스트가 "행복"인지 "슬픔"인지 예측하고 싶지만 이에 대한 교육 데이터가 없다고 가정해 보겠습니다.
이 스니펫과 함께 TARS를 사용하기만 하면 됩니다:

```python
from flair.models import TARSClassifier
from flair.data import Sentence
# 1. 영어로 사전 훈련된 TARS 모델 로드
tars = TARSClassifier.load('tars-base')
# 2. 테스트 문장 준비
sentence = Sentence("I am so glad you liked it!")
# 3. 서술적인 이름을 사용하여 예측하려는 일부 클래스 정의
classes = ["happy", "sad"]
# 4. 이 클래스들에 대한 예측
tars.predict_zero_shot(sentence, classes)
# 5. 예측된 레이블이 있는 문장 출력
print(sentence)
```

출력은 다음과 같습니다:   

```console
Sentence: "I am so glad you liked it !"   [− Tokens: 8  − Sentence-Labels: {'label': [happy (0.9312)]}]
```

이 문장에는 "happy"라는 레이블이 선택되었습니다.

다른 라벨과 함께 사용해 보세요! 제로샷 예측은 때때로 (*항상 그런 것은 아니지만*) 매우 잘 작동합니다.

## 사용 사례 #2: TARS를 사용한 제로샷 NER(Named Entity Recognition)

TARS 제로샷 학습 접근 방식을 시퀀스 라벨링으로 확장하고 영어 NER에 대해 사전 훈련된 모델을 제공합니다. 일부 클래스를 정의하고 모델이 클래스를 찾을 수 있는지 확인하세요:

```python
from flair.models import TARSTagger
from flair.data import Sentence
# 1. 제로샷 NER tagger 로드
tars = TARSTagger.load('tars-ner')
# 2. 테스트 문장 준비
sentences = [
    Sentence("The Humboldt University of Berlin is situated near the Spree in Berlin, Germany"),
    Sentence("Bayern Munich played against Real Madrid"),
    Sentence("I flew with an Airbus A380 to Peru to pick up my Porsche Cayenne"),
    Sentence("Game of Thrones is my favorite series"),
]
# 3. "축구팀", "TV 프로그램" 및 "강"과 같은 명명된 엔터티의 일부 클래스 정의
labels = ["Soccer Team", "University", "Vehicle", "River", "City", "Country", "Person", "Movie", "TV Show"]
tars.add_and_switch_to_new_task('task 1', labels, label_type='ner')
# 4. 이 클래스에 대한 예측 및 결과 출력
for sentence in sentences:
    tars.predict(sentence)
    print(sentence.to_tagged_string("ner"))
```

다음과 같이 출력될 것입니다:

```console
The Humboldt <B-University> University <I-University> of <I-University> Berlin <E-University> is situated near the Spree <S-River> in Berlin <S-City> , Germany <S-Country>

Bayern <B-Soccer Team> Munich <E-Soccer Team> played against Real <B-Soccer Team> Madrid <E-Soccer Team>

I flew with an Airbus <B-Vehicle> A380 <E-Vehicle> to Peru <S-City> to pick up my Porsche <B-Vehicle> Cayenne <E-Vehicle>

Game <B-TV Show> of <I-TV Show> Thrones <E-TV Show> is my favorite series
```


따라서 이 예제에서는 모델이 이에 대해 명시적으로 훈련된 적이 없음에도 불구하고 "TV show" (_왕좌의 게임_), "vehicle" (_Airbus A380_ and _Porsche Cayenne_),
"soccer team" (_Bayern Munich_ and _Real Madrid_) 및 "river" (_Spree_) 와 같은 엔터티 클래스를 찾고 있습니다.
이는 진행중인 연구이며 예제는 약간 cherry-picked 된 것입니다. 제로샷 모델은 다음 릴리스까지 상당히 개선될 것으로 기대합니다.

## 사용 사례 #3: TARS 모델 학습 

또한 처음부터 또는 제공된 TARS 모델을 시작점으로 사용하여 고유한 TARS 모델을 훈련할 수 있습니다. 후자를 선택한 경우 새 작업을 훈련하는 데 필요한 훈련 데이터가 거의 없을 수 있습니다.

### 하나의 데이터셋으로 학습하는 방법

하나의 데이터 세트로 훈련하는 것은 Flair에서 다른 모델을 훈련하는 것과 거의 동일합니다. 유일한 차이점은 레이블 이름을 자연어 설명으로 바꾸는 것이 때때로 의미가 있다는 것입니다.
예를 들어, TREC 데이터 세트는 "엔티티에 대한 질문"으로 바꿔 말하는 "ENTY"와 같은 레이블을 정의합니다. 더 나은 설명은 TARS가 배우는 데 도움이 됩니다.

전체 훈련 코드는 다음과 같습니다:

```python
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
# 1. 일부 데이터 세트에는 수수께끼의 레이블 세트가 제공되므로 자연어로 레이블 이름을 정의하십시오.
label_name_map = {'ENTY': 'question about entity',
                  'DESC': 'question about description',
                  'ABBR': 'question about abbreviation',
                  'HUM': 'question about person',
                  'NUM': 'question about number',
                  'LOC': 'question about location'
                  }
# 2. 말뭉치 가져오기
corpus: Corpus = TREC_6(label_name_map=label_name_map)
# 3. 어떤 레이블을 예측할 것인가요?
label_type = 'question_class'
# 4. 레이블 사전 만들기
label_dict = corpus.make_label_dictionary(label_type=label_type)
# 5. 영어용 기존 TARS 기본 모델에서 시작
tars = TARSClassifier.load("tars-base")
# 5a: 또는 이전 줄에 주석을 달고 다음 줄에 주석을 달아 새로운 TARS 모델을 처음부터 훈련시키세요.
# tars = TARSClassifier(embeddings="bert-base-uncased")
# 6. 새 작업으로 전환 (TAS는 여러 작업을 수행할 수 있으므로 하나를 정의해야 함)
tars.add_and_switch_to_new_task(task_name="question classification",
                                label_dictionary=label_dict,
                                label_type=label_type,
                                )
# 7. 텍스트 분류기 트레이너 초기화
trainer = ModelTrainer(tars, corpus)
# 8. 훈련 시작
trainer.train(base_path='resources/taggers/trec',  # path to store the model artifacts
              learning_rate=0.02,  # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
              max_epochs=1,  # terminate after 10 epochs
              )
```

이 스크립트는 TARS 기반 모델에서 시작하므로 몇 에포크면 충분합니다. 그러나 대신 처음부터 새로운 TARS 모델을 훈련하면
(위의 코드 스니펫의 5a단계 참조) 10 또는 20 Epoch 동안 훈련하고 싶을 것입니다.


### 여러 데이터셋으로 학습하는 방법

TARS는 하나 이상의 분류 작업에서 학습하면 퓨샷 및 제로샷 예측에서 더 좋아집니다.

예를 들어 GO_EMOTIONS 데이터 세트를 사용하여 TREC_6에 대해 훈련한 모델을 계속 훈련해 보겠습니다. 코드는 다시 매우 유사해 보입니다. 새 데이터 세트를 학습하기 직전에 `add_and_switch_to_new_task`를 호출해야 합니다.
이렇게 하면 모델이 이제 TREC_6 대신 GO_EMOTIONS를 훈련해야 함을 알 수 있습니다:

```python
from flair.datasets import GO_EMOTIONS
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
# 1. 훈련된 모델 로드
tars = TARSClassifier.load('resources/taggers/trec/best-model.pt')
# 2. GO_EMOTIONS, SENTIMENT_140 등의 새로운 flair 말뭉치 로드
new_corpus = GO_EMOTIONS()
# 3. 레이블 유형 정의
label_type = "emotion"
# 4. 레이블 사전 만들기
label_dict = new_corpus.make_label_dictionary(label_type=label_type)
# 5. 중요: 새 작업으로 전환
tars.add_and_switch_to_new_task("GO_EMOTIONS",
                                label_dictionary=label_dict,
                                label_type=label_type)
# 6. 텍스트 분류기 트레이너 초기화
trainer = ModelTrainer(tars, new_corpus)
# 7. 훈련 시작
trainer.train(base_path='resources/taggers/go_emotions', # path to store the model artifacts
              learning_rate=0.02, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=10, # terminate after 10 epochs
              )
```

이 튜토리얼이 끝나면 결과 모델은 TREC_6 및 GO_EMOTIONS 모두에 대해 고품질 예측을 수행할 수 있으며 이전보다 적은 수의 학습을 위한 더 나은 기반이 됩니다.



## 작업 간 전환

TARS는 레이블 이름과 기본 언어 모델의 텍스트 간의 관계를 캡슐화할 수 있습니다. 위와 같이 여러 말뭉치에 대해 단일 모델을 학습할 수 있습니다. 
이것은 편의를 위해 내부적으로 레이블 집합을 다른 작업으로 그룹화합니다. 사용자는 TARS 모델이 훈련된 기존 작업을 조회한 다음 필요에 따라 그 중 하나로 전환할 수 있습니다.

```python
# 1. 사전 훈련된 TARS 모델 로드
tars = TARSClassifier.load('tars-base')
# 2. 어떤 데이터 세트에서 학습되었는지 확인하십시오.
existing_tasks = tars.list_existing_tasks()
print(f"Existing tasks are: {existing_tasks}")
# 3. 위 목록에 있는 특정 작업으로 전환
tars.switch_to_task("GO_EMOTIONS")
# 4. 테스트 문장 준비하기
sentence = Sentence("I absolutely love this!")
tars.predict(sentence)
print(sentence)
```
출력은 다음과 같습니다: 
```
Existing tasks are: {'AGNews', 'DBPedia', 'IMDB', 'SST', 'TREC_6', 'NEWS_CATEGORY', 'Amazon', 'Yelp', 'GO_EMOTIONS'}
Sentence: "I absolutely love this !"   [− Tokens: 5  − Sentence-Labels: {'label': [LOVE (0.9708)]}]
```

## TARS 사용 시 다음 논문을 인용하세요:

```
@inproceedings{halder2020coling,
  title={Task Aware Representation of Sentences for Generic Text Classification},
  author={Halder, Kishaloy and Akbik, Alan and Krapac, Josip and Vollgraf, Roland},
  booktitle = {{COLING} 2020, 28th International Conference on Computational Linguistics},
  year      = {2020}
}
```
