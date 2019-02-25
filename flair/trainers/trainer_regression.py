import flair
import torch
import torch.nn as nn

from typing import List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flair.training_utils import Metric, EvaluationMetric, clear_embeddings
from flair.data import Sentence, Label
from pathlib import Path
import logging

log = logging.getLogger('flair')

class RegressorTrainer(flair.trainers.ModelTrainer):
  
    @staticmethod
    def _evaluate_text_regressor(model: flair.nn.Model,
                                  sentences: List[Sentence],
                                  eval_mini_batch_size: int = 32,
                                  embeddings_in_memory: bool = False) -> (dict, float):

        with torch.no_grad():
            eval_loss = 0

            batches = [sentences[x:x + eval_mini_batch_size] for x in
                       range(0, len(sentences), eval_mini_batch_size)]

            metric = {}

            for batch in batches:
              
                scores, loss = model.forward_labels_and_loss(batch)

                indices = []
                for sentence in batch:
                  for label in sentence.labels:
                    indices.append(torch.FloatTensor([float(label.value)]))

                true_values = torch.cat(indices, 0)
                
                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                eval_loss += loss

                metric['mae'] = mean_absolute_error(scores.tolist(), true_values.tolist())
                metric['mse'] = mean_squared_error(scores.tolist(), true_values.tolist())

            eval_loss /= len(sentences)

            return metric, eval_loss

  
    def _calculate_evaluation_results_for(self,
                                          dataset_name: str,
                                          dataset: List[Sentence],
                                          evaluation_metric: EvaluationMetric,
                                          embeddings_in_memory: bool,
                                          eval_mini_batch_size: int,
                                          out_path: Path = None):

        metric, loss = RegressorTrainer._evaluate_text_regressor(self.model, dataset, eval_mini_batch_size=eval_mini_batch_size,
                                             embeddings_in_memory=embeddings_in_memory)

        mae = metric['mae']
        mse = metric['mse']

        log.info(f'{dataset_name:<5}: loss {loss:.8f} - mse {mse:.4f} - mae {mae:.4f}')

        return Metric('Evaluation'), loss