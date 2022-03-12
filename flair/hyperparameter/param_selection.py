import logging
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
from hyperopt import fmin, hp, tpe

import flair.nn
from flair.data import Corpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.hyperparameter.parameter import (
    SEQUENCE_TAGGER_PARAMETERS,
    TEXT_CLASSIFICATION_PARAMETERS,
    TRAINING_PARAMETERS,
    Parameter,
)
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric, init_output_file, log_line

log = logging.getLogger("flair")


class OptimizationValue(Enum):
    DEV_LOSS = "loss"
    DEV_SCORE = "score"


class SearchSpace(object):
    def __init__(self):
        self.search_space = {}

    def add(self, parameter: Parameter, func, **kwargs):
        self.search_space[parameter.value] = func(parameter.value, **kwargs)

    def get_search_space(self):
        return hp.choice("parameters", [self.search_space])


class ParamSelector(object):
    def __init__(
        self,
        corpus: Corpus,
        base_path: Union[str, Path],
        max_epochs: int,
        evaluation_metric: EvaluationMetric,
        training_runs: int,
        optimization_value: OptimizationValue,
    ):
        if type(base_path) is str:
            base_path = Path(base_path)

        self.corpus = corpus
        self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.run = 1
        self.training_runs = training_runs
        self.optimization_value = optimization_value

        self.param_selection_file = init_output_file(base_path, "param_selection.txt")

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    def _objective(self, params: dict):
        log_line(log)
        log.info(f"Evaluation run: {self.run}")
        log.info("Evaluating parameter combination:")
        for k, v in params.items():
            if isinstance(v, tuple):
                v = ",".join([str(x) for x in v])
            log.info(f"\t{k}: {str(v)}")
        log_line(log)

        scores = []
        vars = []

        for i in range(0, self.training_runs):
            log_line(log)
            log.info(f"Training run: {i + 1}")

            for sent in self.corpus.get_all_sentences():  # type: ignore
                sent.clear_embeddings()

            model = self._set_up_model(params)

            training_params = {key: params[key] for key in params if key in TRAINING_PARAMETERS}

            trainer: ModelTrainer = ModelTrainer(model, self.corpus)

            result = trainer.train(
                self.base_path,
                max_epochs=self.max_epochs,
                param_selection_mode=True,
                **training_params,
            )

            # take the average over the last three scores of training
            if self.optimization_value == OptimizationValue.DEV_LOSS:
                curr_scores = result["dev_loss_history"][-3:]
            else:
                curr_scores = list(map(lambda s: 1 - s, result["dev_score_history"][-3:]))

            score = sum(curr_scores) / float(len(curr_scores))
            var = np.var(curr_scores)
            scores.append(score)
            vars.append(var)

        # take average over the scores from the different training runs
        final_score = sum(scores) / float(len(scores))
        final_var = sum(vars) / float(len(vars))

        test_score = result["test_score"]
        log_line(log)
        log.info("Done evaluating parameter combination:")
        for k, v in params.items():
            if isinstance(v, tuple):
                v = ",".join([str(x) for x in v])
            log.info(f"\t{k}: {v}")
        log.info(f"{self.optimization_value.value}: {final_score}")
        log.info(f"variance: {final_var}")
        log.info(f"test_score: {test_score}\n")
        log_line(log)

        with open(self.param_selection_file, "a") as f:
            f.write(f"evaluation run {self.run}\n")
            for k, v in params.items():
                if isinstance(v, tuple):
                    v = ",".join([str(x) for x in v])
                f.write(f"\t{k}: {str(v)}\n")
            f.write(f"{self.optimization_value.value}: {final_score}\n")
            f.write(f"variance: {final_var}\n")
            f.write(f"test_score: {test_score}\n")
            f.write("-" * 100 + "\n")

        self.run += 1

        return {"status": "ok", "loss": final_score, "loss_variance": final_var}

    def optimize(self, space: SearchSpace, max_evals=100):
        search_space = space.search_space
        best = fmin(self._objective, search_space, algo=tpe.suggest, max_evals=max_evals)

        log_line(log)
        log.info("Optimizing parameter configuration done.")
        log.info("Best parameter configuration found:")
        for k, v in best.items():
            log.info(f"\t{k}: {v}")
        log_line(log)

        with open(self.param_selection_file, "a") as f:
            f.write("best parameter combination\n")
            for k, v in best.items():
                if isinstance(v, tuple):
                    v = ",".join([str(x) for x in v])
                f.write(f"\t{k}: {str(v)}\n")


class SequenceTaggerParamSelector(ParamSelector):
    def __init__(
        self,
        corpus: Corpus,
        tag_type: str,
        base_path: Union[str, Path],
        max_epochs: int = 50,
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
        training_runs: int = 1,
        optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS,
    ):
        """
        :param corpus: the corpus
        :param tag_type: tag type to use
        :param base_path: the path to the result folder (results will be written to that folder)
        :param max_epochs: number of epochs to perform on every evaluation run
        :param evaluation_metric: evaluation metric used during training
        :param training_runs: number of training runs per evaluation run
        :param optimization_value: value to optimize
        """
        super().__init__(
            corpus,
            base_path,
            max_epochs,
            evaluation_metric,
            training_runs,
            optimization_value,
        )

        self.tag_type = tag_type
        self.tag_dictionary = self.corpus.make_label_dictionary(self.tag_type)

    def _set_up_model(self, params: dict):
        sequence_tagger_params = {key: params[key] for key in params if key in SEQUENCE_TAGGER_PARAMETERS}

        tagger: SequenceTagger = SequenceTagger(
            tag_dictionary=self.tag_dictionary,
            tag_type=self.tag_type,
            **sequence_tagger_params,
        )
        return tagger


class TextClassifierParamSelector(ParamSelector):
    def __init__(
        self,
        corpus: Corpus,
        label_type: str,
        multi_label: bool,
        base_path: Union[str, Path],
        max_epochs: int = 50,
        fine_tune: bool = True,
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
        training_runs: int = 1,
        optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS,
    ):
        """
        Text classifier hyperparameter selector that leverages TransformerDocumentEmbeddings
        :param corpus: the corpus
        :param label_type: string to identify the label type ('question_class', 'sentiment', etc.)
        :param multi_label: true, if the dataset is multi label, false otherwise
        :param base_path: the path to the result folder (results will be written to that folder)
        :param max_epochs: number of epochs to perform on every evaluation run
        :param fine_tune: if True, allows transformers to be fine-tuned during training
        :param evaluation_metric: evaluation metric used during training
        :param training_runs: number of training runs per evaluation run
        :param optimization_value: value to optimize
        """
        super().__init__(
            corpus,
            base_path,
            max_epochs,
            evaluation_metric,
            training_runs,
            optimization_value,
        )

        self.multi_label = multi_label
        self.label_type = label_type
        self.fine_tune = fine_tune

        self.label_dictionary = self.corpus.make_label_dictionary(self.label_type)

    def _set_up_model(self, params: dict):
        text_classification_params = {key: params[key] for key in params if key in TEXT_CLASSIFICATION_PARAMETERS}

        document_embedding = TransformerDocumentEmbeddings(fine_tune=self.fine_tune, **text_classification_params)

        text_classifier: TextClassifier = TextClassifier(
            label_dictionary=self.label_dictionary,
            multi_label=self.multi_label,
            label_type=self.label_type,
            document_embeddings=document_embedding,
        )

        return text_classifier
