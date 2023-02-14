import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

import flair.nn
from flair.data import DT, Dictionary, Sentence
from flair.nn import Classifier
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
        use_all_tasks: bool = False,
    ):
        """
        :param models: Key (Task ID) - Value (flair.nn.Model) Pairs to stack model
        """
        super(MultitaskModel, self).__init__()

        task_ids_internal: List[str] = task_ids if task_ids else [f"Task_{i}" for i in range(len(models))]

        self.tasks: Dict[str, flair.nn.Classifier] = {}
        self.loss_factors: Dict[str, float] = {}
        self.use_all_tasks = use_all_tasks

        if not loss_factors:
            loss_factors = [1.0] * len(models)

        for task_id, model, loss_factor in zip(task_ids_internal, models, loss_factors):
            self.add_module(task_id, model)
            self.tasks[task_id] = model
            self.loss_factors[task_id] = loss_factor

            # the multi task model has several labels
            self._label_type = model.label_type
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
        batch_split = self.split_batch_to_task_ids(sentences, all_tasks=self.use_all_tasks)
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
        for task in self.tasks.values():
            task.predict(sentences, **predictargs)

    @staticmethod
    def split_batch_to_task_ids(sentences: Union[List[Sentence], Sentence], all_tasks: bool = False) -> Dict:
        """
        Splits a batch of sentences to its respective model. If single sentence is assigned to several tasks
        (i.e. same corpus but different tasks), then the model assignment for this batch is randomly choosen.
        :param sentences: batch of sentences
        :param all_tasks: use all tasks of each sentence. If deactivated, a random task will be sampled
        :return: Key-value pairs as (task_id, list of sentences ids in batch)
        """
        batch_to_task_mapping: Dict[str, List[int]] = {}
        for sentence_id, sentence in enumerate(sentences):
            if all_tasks:
                multitask_ids = sentence.get_labels("multitask_id")
            else:
                multitask_ids = [random.choice(sentence.get_labels("multitask_id"))]
            for multitask_id in multitask_ids:
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
        evaluate_all: bool = True,
        **evalargs,
    ) -> Result:
        """
        :param sentences: batch of sentences
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
            'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param mini_batch_size: size of batches
        :param num_workers: number of workers for DataLoader class
        :param evaluate_all: choose if all tasks should be evaluated, or a single one, depending on gold_label_type
        :return: Tuple of Result object and loss value (float)
        """

        if not evaluate_all:
            if gold_label_type not in self.tasks:
                raise ValueError(
                    "evaluating a single task on a multitask model requires 'gold_label_type' to be a valid task."
                )
            data = [
                dp
                for dp in data_points
                if any(label.value == gold_label_type for label in dp.get_labels("multitask_id"))
            ]
            return self.tasks[gold_label_type].evaluate(
                data,
                gold_label_type=self.tasks[gold_label_type].label_type,
                out_path=out_path,
                embedding_storage_mode=embedding_storage_mode,
                mini_batch_size=mini_batch_size,
                num_workers=num_workers,
                main_evaluation_metric=main_evaluation_metric,
                exclude_labels=exclude_labels,
                gold_label_dictionary=gold_label_dictionary,
                return_loss=return_loss,
                **evalargs,
            )

        batch_split = self.split_batch_to_task_ids(data_points, all_tasks=True)

        loss = torch.tensor(0.0, device=flair.device)
        main_score = 0.0
        all_detailed_results = ""
        all_classification_report: Dict[str, Dict[str, Any]] = dict()

        for task_id, split in batch_split.items():
            result = self.tasks[task_id].evaluate(
                data_points=[data_points[i] for i in split],
                gold_label_type=self.tasks[task_id].label_type,
                out_path=f"{out_path}_{task_id}.txt" if out_path is not None else None,
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
                + self.tasks[task_id].label_type
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
        initial_model_state = super()._get_state_dict()
        initial_model_state["state_dict"] = {}  # the model state is stored per model already.
        model_state = {
            **initial_model_state,
            "model_states": {task: model._get_state_dict() for task, model in self.tasks.items()},
            "loss_factors": [self.loss_factors[task] for task in self.tasks.keys()],
            "use_all_tasks": self.use_all_tasks,
        }

        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        """
        Initializes the model based on given state dict.
        """
        models = []
        tasks = []
        loss_factors = state["loss_factors"]

        for task, task_state in state["model_states"].items():
            models.append(Classifier.load(task_state))
            tasks.append(task)

        model = cls(
            models=models,
            task_ids=tasks,
            loss_factors=loss_factors,
            use_all_tasks=state.get("use_all_tasks", False),
            **kwargs,
        )
        return model

    @property
    def label_type(self):
        return self._label_type
