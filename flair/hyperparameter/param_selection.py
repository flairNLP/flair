import logging
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

from hyperopt import hp, fmin, tpe

import flair.nn
from flair.data import Corpus
from flair.embeddings import DocumentLSTMEmbeddings, DocumentPoolEmbeddings
from flair.hyperparameter import Parameter
from flair.hyperparameter.parameter import SEQUENCE_TAGGER_PARAMETERS, TRAINING_PARAMETERS, \
    DOCUMENT_EMBEDDING_PARAMETERS, MODEL_TRAINER_PARAMETERS
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric, log_line

log = logging.getLogger(__name__)


class SearchSpace(object):

    def __init__(self):
        self.search_space = {}

    def add(self, parameter: Parameter, func, **kwargs):
        self.search_space[parameter.value] = func(parameter.value, **kwargs)

    def get_search_space(self):
        return hp.choice('parameters', [ self.search_space ])


class ParamSelector(object):

    def __init__(self,
                 corpus: Corpus,
                 base_path: Path,
                 max_epochs=50,
                 evaluation_metric=EvaluationMetric.MICRO_F1_SCORE):
        self.corpus = corpus
        self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.run = 1

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    def _objective(self, params: dict):
        log_line()
        log.info(f'Evaluation run: {self.run}')
        log.info(f'Evaluating parameter combination:')
        for k, v in params.items():
            if isinstance(v, Tuple):
                v = ','.join([str(x) for x in v])
            log.info(f'\t{k}: {str(v)}')
        log_line()

        for sent in self.corpus.get_all_sentences():
            sent.clear_embeddings()

        model = self._set_up_model(params)

        training_params = {key: params[key] for key in params if key in TRAINING_PARAMETERS}
        model_trainer_parameters = {key: params[key] for key in params if key in MODEL_TRAINER_PARAMETERS}

        trainer: ModelTrainer = ModelTrainer(model, self.corpus, **model_trainer_parameters)

        result = trainer.train(self.base_path,
                               evaluation_metric=self.evaluation_metric,
                               max_epochs=self.max_epochs,
                               save_final_model=False,
                               test_mode=True,
                               **training_params)

        score = 1 - result['dev_score']

        log_line()
        log.info(f'Done evaluating parameter combination:')
        for k, v in params.items():
            if isinstance(v, Tuple):
                v = ','.join([str(x) for x in v])
            log.info(f'\t{k}: {v}')
        log.info(f'Score: {score}')
        log_line()

        self.run += 1

        return score

    def optimize(self, space: SearchSpace, max_evals=100):
        search_space = space.search_space
        best = fmin(self._objective, search_space, algo=tpe.suggest, max_evals=max_evals)

        log_line()
        log.info('Optimizing parameter configuration done.')
        log.info('Best parameter configuration found:')
        for k, v in best.items():
            log.info(f'\t{k}: {v}')
        log_line()


class SequenceTaggerParamSelector(ParamSelector):

    def __init__(self,
                 corpus: Corpus,
                 tag_type: str,
                 base_path: Path,
                 max_epochs=50,
                 evaluation_metric=EvaluationMetric.MICRO_F1_SCORE):
        super().__init__(corpus, base_path, max_epochs, evaluation_metric)

        self.tag_type = tag_type
        self.tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)

    def _set_up_model(self, params: dict):
        sequence_tagger_params = {key: params[key] for key in params if key in SEQUENCE_TAGGER_PARAMETERS}

        tagger: SequenceTagger = SequenceTagger(tag_dictionary=self.tag_dictionary,
                                                tag_type=self.tag_type,
                                                **sequence_tagger_params)
        return tagger


class TextClassifierParamSelector(ParamSelector):

    def __init__(self,
                 corpus: Corpus,
                 multi_label: bool,
                 base_path: Path,
                 document_embedding_type: str,
                 max_epochs=50,
                 evaluation_metric=EvaluationMetric.MICRO_F1_SCORE):
        super().__init__(corpus, base_path, max_epochs, evaluation_metric)

        self.multi_label = multi_label
        self.document_embedding_type = document_embedding_type

        self.label_dictionary = self.corpus.make_label_dictionary()

    def _set_up_model(self, params: dict):
        embdding_params = {key: params[key] for key in params if key in DOCUMENT_EMBEDDING_PARAMETERS}

        if self.document_embedding_type == 'lstm':
            document_embedding = DocumentLSTMEmbeddings(**embdding_params)
        else:
            document_embedding = DocumentPoolEmbeddings(**embdding_params)

        text_classifier: TextClassifier = TextClassifier(
            label_dictionary=self.label_dictionary,
            multi_label=self.multi_label,
            document_embeddings=document_embedding)

        return text_classifier
