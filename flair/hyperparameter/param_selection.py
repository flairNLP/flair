import logging
from abc import abstractmethod

from hyperopt import hp, fmin, tpe

import flair.nn
from flair.embeddings import DocumentLSTMEmbeddings, DocumentPoolEmbeddings
from flair.hyperparameter import Parameter
from flair.hyperparameter.parameter import SEQUENCE_TAGGER_PARAMETERS, TRAINING_PARAMETERS, \
    DOCUMENT_EMBEDDING_PARAMETERS
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

log = logging.getLogger(__name__)


class SearchSpace(object):

    def __init__(self):
        self.search_space = {}

    def add(self, parameter: Parameter, func, **kwargs):
        self.search_space[parameter.value] = func(parameter.value, **kwargs)

    def get_search_space(self):
        return hp.choice('parameters', [ self.search_space ])


class ParamSelector(object):

    def __init__(self, corpus, result_folder, max_epochs=50, evaluation_metric=EvaluationMetric.MICRO_F1_SCORE):
        self.corpus = corpus
        self.max_epochs = max_epochs
        self.result_folder = result_folder
        self.evaluation_metric = evaluation_metric

    @abstractmethod
    def _set_up_model(self, params) -> flair.nn.Model:
        pass

    def _objective(self, params):
        log.info('-' * 100)
        log.info(f'Evaluating parameter combination:')
        for k, v in params.items():
            log.info(f'\t{k}: {v}')
        log.info('-' * 100)

        for sent in self.corpus.get_all_sentences():
            sent.clear_embeddings()

        model = self._set_up_model(params)

        training_params = {key: params[key] for key in params if key in TRAINING_PARAMETERS}

        trainer: ModelTrainer = ModelTrainer(model, self.corpus)

        result = trainer.train(self.result_folder,
                               evaluation_metric=self.evaluation_metric,
                               max_epochs=self.max_epochs,
                               save_final_model=False,
                               test_mode=True,
                               **training_params)

        score = 1 - result['dev_score']

        log.info('-' * 100)
        log.info(f'Done evaluating parameter combination:')
        for k, v in params.items():
            log.info(f'\t{k}: {v}')
        log.info(f'Score: {score}')
        log.info('-' * 100)

        return score

    def optimize(self, space, max_evals=100):
        search_space = space.search_space
        best = fmin(self._objective, search_space, algo=tpe.suggest, max_evals=max_evals)

        log.info('-' * 100)
        log.info('Optimizing parameter configuration done.')
        log.info('Best parameter configuration found:')
        for k, v in best.items():
            log.info(f'\t{k}: {v}')
        log.info('-' * 100)


class SequenceTaggerParamSelector(ParamSelector):

    def __init__(self, corpus, tag_type, result_folder, max_epochs=50,
                 evaluation_metric=EvaluationMetric.MICRO_F1_SCORE):
        super().__init__(corpus, result_folder, max_epochs, evaluation_metric)

        self.tag_type = tag_type
        self.tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)

    def _set_up_model(self, params):
        sequence_tagger_params = {key: params[key] for key in params if key in SEQUENCE_TAGGER_PARAMETERS}

        tagger: SequenceTagger = SequenceTagger(tag_dictionary=self.tag_dictionary,
                                                tag_type=self.tag_type,
                                                **sequence_tagger_params)
        return tagger


class TextClassifierParamSelector(ParamSelector):

    def __init__(self, corpus, multi_label, result_folder, document_embedding_type, max_epochs=50,
                 evaluation_metric=EvaluationMetric.MICRO_F1_SCORE):
        super().__init__(corpus, result_folder, max_epochs, evaluation_metric)

        self.multi_label = multi_label
        self.document_embedding_type = document_embedding_type

        self.label_dictionary = self.corpus.make_label_dictionary()

    def _set_up_model(self, params):
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
