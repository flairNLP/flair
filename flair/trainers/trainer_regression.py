import flair
import torch
import torch.nn as nn

from typing import List, Union
from flair.training_utils import MetricRegression, EvaluationMetric, clear_embeddings, log_line
from flair.models.text_regression_model import TextRegressor
from flair.data import Sentence, Label
from pathlib import Path
import logging

log = logging.getLogger('flair')

class RegressorTrainer(flair.trainers.ModelTrainer):
  
    def train(self,
              base_path: Union[Path, str],
              evaluation_metric: EvaluationMetric = EvaluationMetric.MEAN_SQUARED_ERROR,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              eval_mini_batch_size: int = None,
              max_epochs: int = 100,
              anneal_factor: float = 0.5,
              patience: int = 3,
              anneal_against_train_loss: bool = True,
              train_with_dev: bool = False,
              monitor_train: bool = False,
              embeddings_in_memory: bool = True,
              checkpoint: bool = False,
              save_final_model: bool = True,
              anneal_with_restarts: bool = False,
              test_mode: bool = False,
              param_selection_mode: bool = False,
              **kwargs
              ) -> dict:

        return super(RegressorTrainer, self).train(
              base_path=base_path, 
              evaluation_metric=evaluation_metric, 
              learning_rate=learning_rate,
              mini_batch_size=mini_batch_size,
              eval_mini_batch_size=eval_mini_batch_size,
              max_epochs=max_epochs,
              anneal_factor=anneal_factor,
              patience=patience,
              anneal_against_train_loss=anneal_against_train_loss,
              train_with_dev=train_with_dev,
              monitor_train=monitor_train,
              embeddings_in_memory=embeddings_in_memory,
              checkpoint=checkpoint,
              save_final_model=save_final_model,
              anneal_with_restarts=anneal_with_restarts,
              test_mode=test_mode,
              param_selection_mode=param_selection_mode)

    @staticmethod
    def _evaluate_text_regressor(model: flair.nn.Model,
                                  sentences: List[Sentence],
                                  eval_mini_batch_size: int = 32,
                                  embeddings_in_memory: bool = False,
                                  out_path: Path = None) -> (dict, float):

        with torch.no_grad():
            eval_loss = 0

            batches = [sentences[x:x + eval_mini_batch_size] for x in
                       range(0, len(sentences), eval_mini_batch_size)]

            metric = MetricRegression('Evaluation')

            lines: List[str] = []
            for batch in batches:
              
                scores, loss = model.forward_labels_and_loss(batch)

                true_values = []
                for sentence in batch:
                  for label in sentence.labels:
                    true_values.append(float(label.value))

                results = []
                for score in scores:
                  if type(score[0]) is Label:
                    results.append(float(score[0].score))
                  else:
                    results.append(float(score[0]))

                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                eval_loss += loss

                metric.true.extend(true_values)
                metric.pred.extend(results)

            eval_loss /= len(sentences)

            ##TODO: not saving lines yet
            if out_path is not None:
                with open(out_path, "w", encoding='utf-8') as outfile:
                    outfile.write(''.join(lines))

            return metric, eval_loss

  
    def _calculate_evaluation_results_for(self,
                                          dataset_name: str,
                                          dataset: List[Sentence],
                                          evaluation_metric: EvaluationMetric,
                                          embeddings_in_memory: bool,
                                          eval_mini_batch_size: int,
                                          out_path: Path = None):

        metric, loss = RegressorTrainer._evaluate_text_regressor(self.model, dataset, eval_mini_batch_size=eval_mini_batch_size,
                                             embeddings_in_memory=embeddings_in_memory, out_path=out_path)

        mse = metric.mean_squared_error()
        mae = metric.mean_absolute_error()

        log.info(f'{dataset_name:<5}: loss {loss:.8f} - mse {mse:.4f} - mae {mae:.4f}')

        return metric, loss

    def final_test(self,
                   base_path: Path,
                   embeddings_in_memory: bool,
                   evaluation_metric: EvaluationMetric,
                   eval_mini_batch_size: int):

        log_line(log)
        log.info('Testing using best model ...')

        self.model.eval()

        if (base_path / 'best-model.pt').exists():
            self.model = TextRegressor.load_from_file(base_path / 'best-model.pt')

        test_metric, test_loss = self._evaluate_text_regressor(self.model, self.corpus.test, eval_mini_batch_size=eval_mini_batch_size,
                                               embeddings_in_memory=embeddings_in_memory)

        log.info(f'AVG: mse: {test_metric.mean_squared_error():.4f} - '
                 f'mae: {test_metric.mean_absolute_error():.4f} - '
                 f'pearson: {test_metric.pearsonr():.4f} - '
                 f'spearman: {test_metric.spearmanr():.4f}')

        log_line(log)

        return test_metric.mean_squared_error()