import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

import flair.nn
from flair.data import DT, Dictionary, Sentence
from flair.training_utils import Result

log = logging.getLogger("flair")


class MultitaskModel(flair.nn.Classifier):
    """
    Multitask Model class which acts as wrapper for creating custom multitask models.
    Takes different tasks as input, parameter sharing is done by objects in flair,
    i.e. creating a Embedding Layer and passing it to two different Models, will
    result in a hard parameter-shared embedding layer. The abstract class takes care
    of calling the correct forward propagation and loss function of the respective
    model.
    """

    def __init__(
        self,
        models: List[flair.nn.Classifier],
        task_ids: Optional[List[str]] = None,
        loss_factors: Optional[List[float]] = None,
    ):
        """
        :param models: Key (Task ID) - Value (flair.nn.Model) Pairs to stack model
        """
        super(MultitaskModel, self).__init__()

        task_ids_internal: List[str] = task_ids if task_ids else [f"Task_{i}" for i in range(len(models))]

        self.tasks: Dict[str, flair.nn.Classifier] = {}
        self.loss_factors: Dict[str, float] = {}

        if not loss_factors:
            loss_factors = [1.0] * len(models)

        label_types = dict()
        for task_id, model, loss_factor in zip(task_ids_internal, models, loss_factors):
            self.add_module(task_id, model)
            self.tasks[task_id] = model
            self.loss_factors[task_id] = loss_factor
            label_types[task_id] = model.label_type
        self._label_type = label_types
        self.to(flair.device)

    def forward(self, *args) -> torch.Tensor:
        raise NotImplementedError("`forward` is not used for multitask learning")

    def _prepare_tensors(self, data_points: List[DT]) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError("`_prepare_tensors` is not used for multitask learning")

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> Tuple[torch.Tensor, int]:
        """
        Abstract forward loss implementation of flair.nn.Model's interface.
        Calls the respective forward loss of each model.
        :param sentences: batch of sentences
        :return: loss
        """
        batch_split = self.split_batch_to_task_ids(sentences)
        loss = torch.tensor(0.0, device=flair.device)
        count = 0
        for task_id, split in batch_split.items():
            task_loss, task_count = self.tasks[task_id].forward_loss([sentences[i] for i in split])
            loss += self.loss_factors[task_id] * task_loss
            count += task_count
        return loss, count

    def predict(
        self,
        sentences,
        **predictargs,
    ):
        for task_id in self.tasks.keys():
            self.tasks[task_id].predict(sentences, **predictargs)

    @staticmethod
    def split_batch_to_task_ids(sentences: Union[List[Sentence], Sentence]) -> Dict:
        """
        Splits a batch of sentences to its respective model. If single sentence is assigned to several tasks
        (i.e. same corpus but different tasks), then the model assignment for this batch is randomly choosen.
        :param sentences: batch of sentences
        :return: Key-value pairs as (task_id, list of sentences ids in batch)
        """
        batch_to_task_mapping: Dict[str, List[int]] = {}
        for sentence_id, sentence in enumerate(sentences):
            multitask_id = random.choice(sentence.get_labels("multitask_id"))
            if multitask_id.value in batch_to_task_mapping:
                batch_to_task_mapping[multitask_id.value].append(sentence_id)
            elif multitask_id.value not in batch_to_task_mapping:
                batch_to_task_mapping[multitask_id.value] = [sentence_id]
        return batch_to_task_mapping

    def evaluate(
        self,
        data_points,
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **evalargs,
    ) -> Result:
        """
        :param sentences: batch of sentences
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
            'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param mini_batch_size: size of batches
        :param num_workers: number of workers for DataLoader class
        :return: Tuple of Result object and loss value (float)
        """

        batch_split = self.split_batch_to_task_ids(data_points)

        loss = torch.tensor(0.0, device=flair.device)
        main_score = 0.0
        all_detailed_results = ""
        all_classification_report: Dict[str, Dict[str, Any]] = dict()

        for task_id, split in batch_split.items():
            result = self.tasks[task_id].evaluate(
                data_points=[data_points[i] for i in split],
                gold_label_type=gold_label_type[task_id],
                out_path=f"{out_path}_{task_id}.txt",
                embedding_storage_mode=embedding_storage_mode,
                mini_batch_size=mini_batch_size,
                num_workers=mini_batch_size,
                main_evaluation_metric=main_evaluation_metric,
                exclude_labels=exclude_labels,
                gold_label_dictionary=gold_label_dictionary,
                return_loss=return_loss,
                **evalargs,
            )

            log.info(
                f"{task_id} - {self.tasks[task_id]._get_name()} - "
                f"loss: {result.loss} - {main_evaluation_metric[1]} "
                f"({main_evaluation_metric[0]})  {round(result.main_score, 4)}"
            )

            loss += result.loss
            main_score += result.main_score
            all_detailed_results += (
                50 * "-"
                + "\n\n"
                + task_id
                + " - "
                + "Label type: "
                + self.label_type.get(task_id)
                + "\n\n"
                + result.detailed_results
            )
            all_classification_report[task_id] = result.classification_report

        return Result(
            loss=loss.item() / len(batch_split),
            main_score=main_score / len(batch_split),
            detailed_results=all_detailed_results,
            log_header="",
            log_line="",
            classification_report=all_classification_report,
        )

    def _get_state_dict(self):
        """
        Returns the state dict of the multitask model which has multiple models underneath.
        :return model_state: model state for the multitask model
        """
        model_state = {}

        for task in self.tasks:
            model_state[task] = {
                "state_dict": self.__getattr__(task)._get_state_dict(),
                "class": self.__getattr__(task).__class__,
            }

        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        """
        Initializes the model based on given state dict.
        """
        models = []
        tasks = []

        for task, task_state in state.items():
            if task != "model_card":
                models.append(task_state["class"]._init_model_with_state_dict(task_state["state_dict"]))
                tasks.append(task)

        model = MultitaskModel(models=models, task_ids=tasks)
        return model

    @property
    def label_type(self):
        return self._label_type
