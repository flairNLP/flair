import logging
from pathlib import Path
from typing import Union, List, Dict
import random

import torch.nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

import flair.nn
from flair.data import Sentence
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Result, MultitaskResult, store_embeddings

log = logging.getLogger("flair")

class MultitaskModel(flair.nn.Model):
    """
    Basic multitask model.
    """

    def __init__(
        self,
        models: Dict,
    ):
        super(MultitaskModel, self).__init__()

        # Dynamically create task models from tag_spaces
        self.tasks = list()
        for task_id, model in models.items():
            self.__setattr__(task_id, model)
            self.tasks.append(task_id)
        self.to(flair.device)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Forward loss function from abstract base class in flair
        :param sentences: list of sentences
        """
        batch_split = self.split_batch_to_task_ids(sentences)
        loss = 0
        for model, split in batch_split.items():
            loss += self.__getattr__(model).forward_loss(sentences=[sentences[i] for i in split])

        return loss

    @staticmethod
    def split_batch_to_task_ids(sentences):
        sent_idx_to_model = {}
        for id, sentence in enumerate(sentences):
            task = random.choice(sentence.multitask_annotations.get("multitask_assignments"))
            if not task.task_id in sent_idx_to_model:
                sent_idx_to_model[task.task_id] = [id]
            elif task.task_id in sent_idx_to_model:
                sent_idx_to_model[task.task_id].append(id)

        return sent_idx_to_model

    def evaluate(
        self,
        sentences: Union[List[Sentence], Dataset],
        out_path: Path = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: int = 8
    ) -> (Result, float):

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        eval_loss = 0
        batch_no = 0

        for sentence_batch in data_loader:

            batch_split = self.split_batch_to_task_ids(sentence_batch)

            for task, split in batch_split.items():
                loss = self.__getattr__(task).evaluate(sentences=[sentence_batch[i] for i in split],
                                                       embedding_storage_mode=embedding_storage_mode)

                eval_loss += loss

            batch_no += 1

        eval_loss /= batch_no

        results = []
        for task in self.tasks:
            results.append(self.__getattr__(task).result)
            self.__getattr__(task)._reset_eval_metrics()

        result = MultitaskResult(results)

        return result, eval_loss