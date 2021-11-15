# 튜토리얼 9: 여러분만의 Flair 임베딩을 훈련하세요!

Flair 임베딩은 Flair의 비밀 소스이며, 이를 통해 다양한 NLP 작업에서 최첨단 정확도를 달성할 수 있습니다.
이 튜토리얼에서는 자신만의 Flair 임베딩을 훈련하는 방법을 알려줄 것입니다. 이는 Flair를 새로운 언어나 도메인에 적용하려는 경우에 유용할 수 있습니다.


## 텍스트 말뭉치(Corpus) 준비

언어 모델은 일반 텍스트로 학습됩니다. 문자 LM의 경우 문자 시퀀스에서 다음 문자를 예측하도록 훈련합니다.
자신의 모델을 학습시키려면 먼저 적절하게 큰 말뭉치를 식별해야 합니다. 실험에서는 약 10억 개의 단어가 있는 말뭉치를 사용했습니다.

코퍼스를 학습, 검증 및 테스트 부분으로 분할해야 합니다.
우리의 트레이너 클래스는 테스트 및 검증 데이터가 있는 'test.txt'와 'valid.txt'가 있는 코퍼스용 폴더가 있다고 가정하고 있습니다.
중요한 것은 분할된 훈련 데이터를 포함하는 'train'이라는 폴더도 있다는 것입니다.
예를 들어, 10억 단어 코퍼스는 100개 부분으로 나뉩니다.
모든 데이터가 메모리에 맞지 않는 경우 분할이 필요합니다. 이 경우 트레이너는 모든 분할을 무작위로 반복합니다.

따라서 폴더 구조는 다음과 같아야 합니다:

```
corpus/
corpus/train/
corpus/train/train_split_1
corpus/train/train_split_2
corpus/train/...
corpus/train/train_split_X
corpus/test.txt
corpus/valid.txt
```

대부분의 경우 문서나 문장에 대한 명확한 구분 기호가 없는 구조화되지 않은 형식으로 말뭉치를 제공하는 것이 좋습니다. LM이 문서 경계를 더 쉽게 식별할 수 있도록 하려면 "[SEP]"와 같은 구분 토큰을 도입할 수 있습니다.

## 언어 모델 훈련

이 폴더 구조가 있으면 `LanguageModelTrainer` 클래스를 이 폴더 구조로 지정하여 모델 학습을 시작하세요.

```python
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
# 앞으로 혹은 뒤로 LM을 훈련하고 있나요?
is_forward_lm = True
# 기본 문자 사전 로드
dictionary: Dictionary = Dictionary.load('chars')
# 당신의 말뭉치를 가져와서, 앞으로 그리고 문자 레벨에서 진행하세요.
corpus = TextCorpus('/path/to/your/corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)
# 언어 모델 인스턴스화, 숨겨진 크기 및 레이어 사이즈 설정
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=128,
                               nlayers=1)
# 언어 모델 훈련
trainer = LanguageModelTrainer(language_model, corpus)
trainer.train('resources/taggers/language_model',
              sequence_length=10,
              mini_batch_size=10,
              max_epochs=10)
```

이 스크립트의 매개변수는 매우 작습니다. 숨겨진 사이즈는 1024 또는 2048, 시퀀스 길이는 250, 미니 배치 크기는 100으로 좋은 결과를 얻었습니다.
리소스에 따라 대규모 모델을 훈련할 수 있지만 모델을 훈련하는 데 매우 강력한 GPU와 많은 시간이 필요하다는 점에 유의하십시오. (1주 이상 훈련)



## LM을 임베딩으로 사용

LM을 학습하면 임베딩으로 사용하기 쉽습니다. 모델을 `FlairEmbeddings` 클래스에 로드하고 Flair의 다른 임베딩처럼 사용하세요:

```python
sentence = Sentence('I love Berlin')
# 훈련된 LM에서 임베딩 초기화
char_lm_embeddings = FlairEmbeddings('resources/taggers/language_model/best-lm.pt')
# 임베딩된 문장
char_lm_embeddings.embed(sentence)
```

끝입니다!


## 라틴어가 아닌 알파벳

아랍어나 일본어와 같은 비라틴어 알파벳을 사용하는 언어에 대한 임베딩을 훈련하는 경우 먼저 고유한 문자 사전을 만들어야 합니다. 다음 코드로 이 작업을 수행할 수 있습니다:

```python
# 빈 문자 사전 만들기
from flair.data import Dictionary
char_dictionary: Dictionary = Dictionary()
# counter 오브젝트
import collections
counter = collections.Counter()
processed = 0
import glob
files = glob.glob('/path/to/your/corpus/files/*.*')
print(files)
for file in files:
    print(file)
    with open(file, 'r', encoding='utf-8') as f:
        tokens = 0
        for line in f:
            processed += 1            
            chars = list(line)
            tokens += len(chars)
            # 사전에 문자 추가
            counter.update(chars)
            # 속도를 높이려면 이 줄을 주석 처리하세요. (말뭉치가 너무 큰 경우)
            # if tokens > 50000000: break
    # break
total_count = 0
for letter, count in counter.most_common():
    total_count += count
print(total_count)
print(processed)
sum = 0
idx = 0
for letter, count in counter.most_common():
    sum += count
    percentile = (sum / total_count)
    # 문자의 상위 X 백분위수만 사용하려면 이 줄에 주석을 달고, 그렇지 않으면 나중에 필터링하세요.
    # if percentile < 0.00001: break
    char_dictionary.add_item(letter)
    idx += 1
    print('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))
print(char_dictionary.item2idx)
import pickle
with open('/path/to/your_char_mappings', 'wb') as f:
    mappings = {
        'idx2item': char_dictionary.idx2item,
        'item2idx': char_dictionary.item2idx
    }
    pickle.dump(mappings, f)
```

그런 다음 언어 모델 학습을 위해 코드의 기본 사전 대신 이 사전을 사용할 수 있습니다.

```python
import pickle
dictionary = Dictionary.load_from_file('/path/to/your_char_mappings')
```

## 파라미터

우리는 `LanguageModelTrainer`의 일부 학습 매개변수를 가지고 놀 수 있습니다.
예를 들어, 우리는 일반적으로 초기 학습률이 20이고 annealing 계수 4가 대부분의 말뭉치에 대해 꽤 좋은 것을 알 수 있습니다.
학습률 스케줄러의 '인내' 값을 수정할 수도 있습니다. 현재 25개로 설정되어 있습니다. 즉, 25개의 분할에 대해 훈련 손실이 개선되지 않으면 학습률이 감소합니다.


## 기존 LM 미세 조정

때로는 처음부터 훈련하는 대신 기존 언어 모델을 미세 조정하는 것이 합리적입니다. 예를 들어, 영어에 대한 일반 LM이 있고 특정 도메인에 대해 미세 조정하려는 경우입니다. 

`LanguageModel`을 미세 조정하려면 새 인스턴스를 생성하는 대신 기존 `LanguageModel`을 로드하기만 하면 됩니다. 나머지 훈련 코드는 위와 동일하게 유지됩니다.


```python
from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
# FlairEmbeddings와 같은 기존 LM 인스턴스화
language_model = FlairEmbeddings('news-forward').lm
# 앞으로 또는 뒤로 LM을 미세 조정하고 있나요?
is_forward_lm = language_model.is_forward_lm
# 기존 언어 모델에서 사전 가져오기
dictionary: Dictionary = language_model.dictionary
# 당신의 말뭉치를 가져와서, 앞으로 그리고 문자 레벨에서 진행하세요.
corpus = TextCorpus('path/to/your/corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)
# 모델 트레이너를 사용하여 코퍼스에서 이 모델을 미세 조정하세요.
trainer = LanguageModelTrainer(language_model, corpus)
trainer.train('resources/taggers/language_model',
              sequence_length=100,
              mini_batch_size=100,
              learning_rate=20,
              patience=10,
              checkpoint=True)
```              
              
미세조정 시에는 전과 동일한 문자사전을 사용해야 하며 방향(앞/뒤)을 복사해야 합니다.


## LM에 기여해보세요!

아직 Flair에 없는 언어나 도메인에 대해 우수한 LM을 훈련하고 있다면 저희에게 연락해주세요. 다른 사람들이 사용할 수 있도록 더 많은 LM을 라이브러리에 통합하게 되어 기쁩니다!
