import logging
from enum import Enum

from hyperopt import hp, fmin, tpe

from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

log = logging.getLogger(__name__)

TRAINING_PARAMETERS = ['learning_rate', 'mini_batch_size', 'anneal_factor', 'patience', 'anneal_with_restarts']
SEQUENCE_TAGGER_PARAMETERS = [
    'embeddings', 'hidden_size', 'use_crf', 'use_rnn', 'rnn_layers', 'use_dropout', 'use_word_dropout',
    'use_locked_dropout'
]
TEXT_CLASSIFIER_PARAMETERS = ['embeddings']


class ParameterName(Enum):
    embeddings = 'embeddings'
    hidden_size = 'hidden_size',
    use_crf = 'use_crf',
    use_rnn = 'use_rnn',
    rnn_layers = 'rnn_layers',
    use_dropout = 'use_dropout',
    use_word_dropout = 'use_word_dropout',
    use_locked_dropout = 'use_locked_dropout',
    learning_rate = 'learning_rate',
    mini_batch_size = 'mini_batch_size',
    anneal_factor = 'anneal_factor',
    anneal_with_restarts = 'anneal_with_restarts'
    patience = 'patience'


class SearchSpace(object):

    def __init__(self):
        self.search_space = {}

    def add(self, parameter_name: ParameterName, func, **kwargs):
        self.search_space[parameter_name.name] = func(parameter_name.name, **kwargs)

    def get_search_space(self):
        return hp.choice('parameters', [ self.search_space ])


class SequenceTaggerParamSelector(object):

    def __init__(self, corpus, tag_type, result_folder, max_epochs=50,
                 evaluation_metric=EvaluationMetric.MICRO_F1_SCORE, anneal_with_restarts=False):
        self.corpus = corpus
        self.tag_type = tag_type
        self.max_epochs = max_epochs
        self.result_folder = result_folder
        self.evaluation_metric = evaluation_metric
        self.anneal_with_restarts = anneal_with_restarts

        self.tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)

    def _objective(self, params):
        log.info('-' * 100)
        log.info(f'Evaluating parameter combination:')
        for k, v in params.items():
            log.info(f'\t{k}: {v}')
        log.info('-' * 100)

        sequence_tagger_params = {key: params[key] for key in params if key in SEQUENCE_TAGGER_PARAMETERS}
        training_params = {key: params[key] for key in params if key in TRAINING_PARAMETERS}

        tagger: SequenceTagger = SequenceTagger(tag_dictionary=self.tag_dictionary,
                                                tag_type=self.tag_type,
                                                **sequence_tagger_params)

        trainer: ModelTrainer = ModelTrainer(tagger, self.corpus)

        score = 1 - trainer.train(self.result_folder,
                                  evaluation_metric=self.evaluation_metric,
                                  max_epochs=self.max_epochs,
                                  train_with_dev=False,
                                  monitor_train=False,
                                  embeddings_in_memory=True,
                                  checkpoint=False,
                                  save_final_model=False,
                                  test_mode=True,
                                  **training_params)

        log.info('-' * 100)
        log.info(f'Done evaluating parameter combination:')
        for k, v in params.items():
            log.info(f'\t{k}: {v}')
        log.info(f'Score: {score}')
        log.info('-' * 100)

        return score

    def optimize(self, space, max_evals=10):
        search_space = space.search_space
        best = fmin(self._objective, search_space, algo=tpe.suggest, max_evals=max_evals)

        log.info('-' * 100)
        log.info('Optimizing parameter configuration done.')
        log.info('Best parameter configuration found:')
        log.info(best)
        log.info('-' * 100)


class TextClassifierParamSelector(object):

    def __init__(self, corpus, multi_label, result_folder, max_epochs=50,
                 evaluation_metric=EvaluationMetric.MICRO_F1_SCORE, anneal_with_restarts=False):
        self.corpus = corpus
        self.max_epochs = max_epochs
        self.result_folder = result_folder
        self.evaluation_metric = evaluation_metric
        self.anneal_with_restarts = anneal_with_restarts
        self.multi_label = multi_label

        self.label_dictionary = self.corpus.make_label_dictionary()

    def _objective(self, params):
        log.info('-' * 100)
        log.info(f'Evaluating parameter combination:')
        for k, v in params.items():
            log.info(f'\t{k}: {v}')
        log.info('-' * 100)

        text_classifier_params = {key: params[key] for key in params if key in TEXT_CLASSIFIER_PARAMETERS}
        training_params = {key: params[key] for key in params if key in TRAINING_PARAMETERS}

        tagger: TextClassifier = TextClassifier(label_dictionary=self.label_dictionary,
                                                multi_label=self.multi_label,
                                                **text_classifier_params)

        trainer: ModelTrainer = ModelTrainer(tagger, self.corpus)

        score = 1 - trainer.train(self.result_folder,
                                  evaluation_metric=self.evaluation_metric,
                                  max_epochs=self.max_epochs,
                                  train_with_dev=False,
                                  monitor_train=False,
                                  embeddings_in_memory=True,
                                  checkpoint=False,
                                  save_final_model=False,
                                  anneal_with_restarts=self.anneal_with_restarts,
                                  test_mode=True,
                                  **training_params)

        log.info('-' * 100)
        log.info(f'Done evaluating parameter combination:')
        for k, v in params.items():
            log.info(f'\t{k}: {v}')
        log.info(f'Score: {score}')
        log.info('-' * 100)

        return score

    def optimize(self, space, max_evals=10):
        search_space = space.search_space
        best = fmin(self._objective, search_space, algo=tpe.suggest, max_evals=max_evals)

        log.info('-' * 100)
        log.info('Optimizing parameter configuration done.')
        log.info('Best parameter configuration found:')
        log.info(best)
        log.info('-' * 100)
