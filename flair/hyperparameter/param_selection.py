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
from flair.training_utils import EvaluationMetric, log_line, init_output_file

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
                 max_epochs: int = 50,
                 evaluation_metric:EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
                 training_runs: int = 1
                 ):
        self.corpus = corpus
        self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.run = 1
        self.training_runs = training_runs

        self.param_selection_file = init_output_file(base_path, 'param_selection.txt')

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

        scores = []

        for i in range(0, self.training_runs):
            log_line()
            log.info(f'Training run: {i + 1}')

            model = self._set_up_model(params)

            training_params = {key: params[key] for key in params if key in TRAINING_PARAMETERS}
            model_trainer_parameters = {key: params[key] for key in params if key in MODEL_TRAINER_PARAMETERS}

            trainer: ModelTrainer = ModelTrainer(model, self.corpus, **model_trainer_parameters)

            result = trainer.train(self.base_path,
                                   evaluation_metric=self.evaluation_metric,
                                   max_epochs=self.max_epochs,
                                   param_selection_mode=True,
                                   **training_params)

            # take the average over the last three scores of training
            l = result['score_history'][3:] if len(result['score_history']) > 3 else result['score_history']
            score = sum(l) / float(len(l))
            scores.append(score)

        # take average over the scroes from the different training runs
        final_score = sum(scores) / float(len(scores))

        log_line()
        log.info(f'Done evaluating parameter combination:')
        for k, v in params.items():
            if isinstance(v, Tuple):
                v = ','.join([str(x) for x in v])
            log.info(f'\t{k}: {v}')
        log.info(f'Score: {final_score}')
        log_line()

        with open(self.param_selection_file, 'a') as f:
            f.write(f'evaluation run {self.run}\n')
            for k, v in params.items():
                if isinstance(v, Tuple):
                    v = ','.join([str(x) for x in v])
                f.write(f'\t{k}: {str(v)}\n')
            f.write(f'score: {final_score}\n')
            f.write('-' * 100 + '\n')

        self.run += 1

        return final_score

    def optimize(self, space: SearchSpace, max_evals=100):
        search_space = space.search_space
        best = fmin(self._objective, search_space, algo=tpe.suggest, max_evals=max_evals)

        log_line()
        log.info('Optimizing parameter configuration done.')
        log.info('Best parameter configuration found:')
        for k, v in best.items():
            log.info(f'\t{k}: {v}')
        log_line()

        with open(self.param_selection_file, 'a') as f:
            f.write('best parameter combination\n')
            for k, v in best.items():
                if isinstance(v, Tuple):
                    v = ','.join([str(x) for x in v])
                f.write(f'\t{k}: {str(v)}\n')


class SequenceTaggerParamSelector(ParamSelector):

    def __init__(self,
                 corpus: Corpus,
                 tag_type: str,
                 base_path: Path,
                 max_epochs: int = 50,
                 evaluation_metric:EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
                 training_runs: int = 1):
        super().__init__(corpus, base_path, max_epochs, evaluation_metric, training_runs)

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
                 max_epochs: int = 50,
                 evaluation_metric:EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
                 training_runs: int = 1):
        super().__init__(corpus, base_path, max_epochs, evaluation_metric, training_runs)

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
