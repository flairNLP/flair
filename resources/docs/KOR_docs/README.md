# Korean documentation for flairNLP
flairNLP를 한국어로 이해하기 쉽게 번역한 튜토리얼 번역본입니다.    


![alt text](https://github.com/flairNLP/flair/blob/master/resources/docs/flair_logo_2020.png?raw=true)

[![PyPI version](https://badge.fury.io/py/flair.svg)](https://badge.fury.io/py/flair)
[![GitHub Issues](https://img.shields.io/github/issues/flairNLP/flair.svg)](https://github.com/flairNLP/flair/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

**최첨단 NLP**를 위한 매우 간단한 프레임워크입니다. 
[Humboldt University of Berlin](https://www.informatik.hu-berlin.de/en/forschung-en/gebiete/ml-en/) 및 친구들에 의해 개발되었습니다.

---

Flair는:

* **강력한 NLP 라이브러리입니다.**    
 Flair를 사용하면 명명된 개체 인식(NER), 품사 태깅(PoS), [생체 의학 데이터](https://github.com/flairNLP/flair/blob/94393aa82444f28c5a1da6057b8ff57b3cb390e6/resources/docs/HUNFLAIR.md)에 대한 특별 지원과 같은 최첨단 자연어 처리(NLP) 모델을 텍스트에 적용할 수 있습니다.
 또한 빠르게 증가하는 언어를 지원하여 명확화 및 분류를 감지합니다.

* **텍스트 임베딩 라이브러리입니다.**    
Flair에는 제안된 **[Flair embeddings](https://www.aclweb.org/anthology/C18-1139/)**, BERT 임베딩 및 ELMo 임베딩을 포함하여 다양한 단어 및 문서 임베딩을 사용하고 결합할 수 있는 간단한 인터페이스가 있습니다.

* **파이토치 NLP 프레임워크입니다.**    
 우리의 프레임워크는 [PyTorch](https://pytorch.org/)를 기반으로 직접 구축되어 쉽게 자신의 모델을 훈련하고 Flair 임베딩 및 클래스를 사용하여 새로운 접근 방식을 실험할 수 있습니다.

 이제 [version 0.9](https://github.com/flairNLP/flair/releases)입니다!


## Join Us: HU-Berlin에서 채용 공고!

박사 학위를 추구하고 오픈 소스를 사랑하기 위해 NLP/ML 연구를 수행하는 데 관심이 있다면 연구원 및 박사 후보자를 위해 [open positions](https://github.com/flairNLP/flair/issues/2446)에 지원하는 것을 고려하십시오. 베를린 훔볼트 대학교에서! 현재 **3개의 공석**이 있으며 곧 지원 마감일입니다!

## 최첨단 모델

Flair는 다양한 NLP 작업을 위한 최신 모델과 함께 제공됩니다. 예를 들어 최신 NER 모델을 확인해보세요:

| Language | Dataset | Flair | Best published | Model card & demo
|  ---  | ----------- | ---------------- | ------------- | ------------- |
| English | Conll-03 (4-class)   |  **94.09**  | *94.3 [(Yamada et al., 2018)](https://doi.org/10.18653/v1/2020.emnlp-main.523)* | [Flair English 4-class NER demo](https://huggingface.co/flair/ner-english-large)  |
| English | Ontonotes (18-class)  |  **90.93**  | *91.3 [(Yu et al., 2016)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair English 18-class NER demo](https://huggingface.co/flair/ner-english-ontonotes-large) |
| German  | Conll-03 (4-class)   |  **92.31**  | *90.3 [(Yu et al., 2016)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair German 4-class NER demo](https://huggingface.co/flair/ner-german-large)  |
| Dutch  | Conll-03  (4-class)  |  **95.25**  | *93.7 [(Yu et al., 2016)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair Dutch 4-class NER demo](https://huggingface.co/flair/ner-dutch-large)  |
| Spanish  | Conll-03 (4-class)   |  **90.54** | *90.3 [(Yu et al., 2016)](https://www.aclweb.org/anthology/2020.acl-main.577.pdf)* | [Flair Spanish 18-class NER demo](https://huggingface.co/flair/ner-spanish-large)  |

**New:** 
대부분의 Flair 시퀀스 태깅 모델(명명된 엔티티 인식, 품사 태깅 등)이 이제  [__🤗 HuggingFace model hub__](https://huggingface.co/models?library=flair&sort=downloads)에서 호스팅됩니다! 모델을 검색하고 학습 방법에 대한 자세한 정보를 확인하고 각 모델을 온라인으로 시험해 볼 수도 있습니다!

## Quick Start

### 요구사항 및 설치

이 프로젝트는 PyTorch 1.5+ 및 Python 3.6+를 기반으로 합니다. 메소드 시그니처와 타입 힌트가 아름답기 때문입니다.
Python 3.6이 없으면 먼저 설치하십시오. [Ubuntu 16.04의 경우](https://vsupalov.com/developing-with-python3-6-on-ubuntu-16-04/).
그런 다음 선호하는 가상 환경에서 다음을 수행하십시오:

```
pip install flair
```

### 사용 예시

예제 문장에 대해 NER(Named Entity Recognition)을 실행해 보겠습니다. 'Sentence'를 만들고 사전 훈련된 모델을 로드하고 이를 사용하여 문장의 태그를 예측하기만 하면 됩니다.

```python
from flair.data import Sentence
from flair.models import SequenceTagger
# 문장 만들기
sentence = Sentence('I love Berlin .')
# NER tagger 로드하기
tagger = SequenceTagger.load('ner')
# 문장에 대해 NER 실행
tagger.predict(sentence)
```

완료입니다! 이제 'Sentence'에 엔티티 주석이 있습니다. 태그가 무엇을 찾았는지 보려면 문장을 출력하세요.

```python
print(sentence)
print('The following NER tags are found:')
# 엔티티를 반복하고 출력하기
for entity in sentence.get_spans('ner'):
    print(entity)
```

출력은 다음과 같습니다:

```console
Sentence: "I love Berlin ." - 4 Tokens

The following NER tags are found:

Span [3]: "Berlin"   [− Labels: LOC (0.9992)]
```

## Tutorials

라이브러리를 시작하는 데 도움이 되는 빠른 튜토리얼 세트를 제공합니다.

* [Tutorial 1: Basics](/resources/docs/KOR_docs/TUTORIAL_1_BASICS.md)
* [Tutorial 2: Tagging your Text](/resources/docs/KOR_docs/TUTORIAL_2_TAGGING.md)
* [Tutorial 3: Embedding Words](/resources/docs/KOR_docs/TUTORIAL_3_WORD_EMBEDDING.md)
* [Tutorial 4: List of All Word Embeddings](/resources/docs/KOR_docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)
* [Tutorial 5: Embedding Documents](/resources/docs/KOR_docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md)
* [Tutorial 6: Loading a Dataset](/resources/docs/KOR_docs/TUTORIAL_6_CORPUS.md)
* [Tutorial 7: Training a Model](/resources/docs/KOR_docs/TUTORIAL_7_TRAINING_A_MODEL.md)
* [Tutorial 8: Training your own Flair Embeddings](/resources/docs/KOR_docs/TUTORIAL_8_MODEL_OPTIMIZATION.md)
* [Tutorial 9: Training a Zero Shot Text Classifier (TARS)](/resources/docs/KOR_docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md)
* [Tutorial 10: Few-Shot and Zero-Shot Classification (TARS)](/resources/docs/KOR_docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md)

튜토리얼에서는 기본 NLP 클래스가 작동하는 방법, 사전 훈련된 모델을 로드하여 텍스트에 태그를 지정하는 방법, 다른 단어 또는 문서 임베딩으로 텍스트를 포함하는 방법, 고유한 언어 모델, 시퀀스 레이블링 모델 및 텍스트 분류 모델에 대해 설명하고있습니다. 불분명한 것이 있으면 알려주세요.

설치 지침 및 자습서가 포함된 **[biomedical NER and datasets](https://github.com/flairNLP/flair/blob/94393aa82444f28c5a1da6057b8ff57b3cb390e6/resources/docs/HUNFLAIR.md)** 전용 랜딩 페이지도 있습니다.

Flair를 사용하는 방법을 보여주는 훌륭한 타사 기사 및 게시물도 있습니다:
* [How to build a text classifier with Flair](https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f)
* [How to build a microservice with Flair and Flask](https://shekhargulati.com/2019/01/04/building-a-sentiment-analysis-python-microservice-with-flair-and-flask/)
* [A docker image for Flair](https://towardsdatascience.com/docker-image-for-nlp-5402c9a9069e)
* [Great overview of Flair functionality and how to use in Colab](https://www.analyticsvidhya.com/blog/2019/02/flair-nlp-library-python/)
* [Visualisation tool for highlighting the extracted entities](https://github.com/lunayach/visNER)
* [Practical approach of State-of-the-Art Flair in Named Entity Recognition](https://medium.com/analytics-vidhya/practical-approach-of-state-of-the-art-flair-in-named-entity-recognition-46a837e25e6b)
* [Benchmarking NER algorithms](https://towardsdatascience.com/benchmark-ner-algorithm-d4ab01b2d4c3)
* [Training a Flair text classifier on Google Cloud Platform (GCP) and serving predictions on GCP](https://github.com/robinvanschaik/flair-on-gcp)
* [Model Interpretability for transformer-based Flair models](https://github.com/robinvanschaik/interpret-flair)

## Flair 인용하기

Flair 임베딩을 사용할 때 [다음 논문](https://www.aclweb.org/anthology/C18-1139/)을 인용하세요.

```
@inproceedings{akbik2018coling,
  title={Contextual String Embeddings for Sequence Labeling},
  author={Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
  booktitle = {{COLING} 2018, 27th International Conference on Computational Linguistics},
  pages     = {1638--1649},
  year      = {2018}
}
```

실험에 Flair 프레임워크를 사용하는 경우 [이 문서](https://www.aclweb.org/anthology/papers/N/N19/N19-4010/)를 인용하세요:

```
@inproceedings{akbik2019flair,
  title={FLAIR: An easy-to-use framework for state-of-the-art NLP},
  author={Akbik, Alan and Bergmann, Tanja and Blythe, Duncan and Rasul, Kashif and Schweter, Stefan and Vollgraf, Roland},
  booktitle={{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)},
  pages={54--59},
  year={2019}
}
```

Flair 임베딩(PooledFlairEmbeddings)의 풀링 버전을 사용하는 경우 [이 문서](https://www.aclweb.org/anthology/papers/N/N19/N19-1078/)를 인용하세요:

```
@inproceedings{akbik2019naacl,
  title={Pooled Contextualized Embeddings for Named Entity Recognition},
  author={Akbik, Alan and Bergmann, Tanja and Vollgraf, Roland},
  booktitle = {{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  pages     = {724–728},
  year      = {2019}
}
```

새로운 "FLERT" 모델 또는 접근 방식을 사용하는 경우 [이 문서](https://arxiv.org/abs/2011.06993)를 인용하세요:

```
@misc{schweter2020flert,
    title={FLERT: Document-Level Features for Named Entity Recognition},
    author={Stefan Schweter and Alan Akbik},
    year={2020},
    eprint={2011.06993},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
```

## Contact

질문이나 의견은 [Alan Akbik](http://alanakbik.github.io/)로 이메일을 보내주세요.

한국어 번역에 대한 의견은    
김한결(hannn0414@naver.com), 박태현(abnc7800@gmail.com), 최완규(wanq123@gachon.ac.kr)로 이메일을 보내주세요.

## Contributing

contributing에 관심을 가져주셔서 감사합니다! 참여하는 방법에는 여러 가지가 있습니다.
[contributor guidelines](https://github.com/flairNLP/flair/blob/94393aa82444f28c5a1da6057b8ff57b3cb390e6/CONTRIBUTING.md)으로 시작한 다음
특정 작업에 대해서는 [open issues](https://github.com/flairNLP/flair/issues)를 확인하세요.

API에 대해 더 깊이 알고자 하는 기여자의 경우 레포지토리를 복제하고 메서드를 호출하는 방법에 대한 예제를 보려면 단위 테스트를 확인하는 것이 좋습니다. 
거의 모든 클래스와 메서드가 문서화되어 있으므로 코드를 찾는 것이 쉬울 것입니다.

### 로컬에서 단위 테스트 실행

이것을 위해 [Pipenv](https://pipenv.readthedocs.io/)가 필요합니다:

```bash
pipenv install --dev && pipenv shell
pytest tests/
```

통합 테스트를 실행하려면 다음을 실행하세요:
```bash
pytest --runintegration tests/
```
통합 테스트는 작은 모델을 훈련합니다.
