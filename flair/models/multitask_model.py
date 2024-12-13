import logging
import random
import typing
from pathlib import Path
from typing import Any, Optional, Union

import torch

import flair.nn
from flair.data import DT, Corpus, Sentence
from flair.file_utils import cached_path
from flair.nn import Classifier
from flair.training_utils import Result

log = logging.getLogger("flair")


class MultitaskModel(flair.nn.Classifier):
    """Multitask Model class which acts as wrapper for creating custom multitask models.

    Takes different tasks as input, parameter sharing is done by objects in flair,
    i.e. creating a Embedding Layer and passing it to two different Models, will
    result in a hard parameter-shared embedding layer. The abstract class takes care
    of calling the correct forward propagation and loss function of the respective
    model.
    """

    def __init__(
        self,
        models: list[flair.nn.Classifier],
        task_ids: Optional[list[str]] = None,
        loss_factors: Optional[list[float]] = None,
        use_all_tasks: bool = False,
    ) -> None:
        """Instantiates the MultiTaskModel.

        Args:
            models: The child models used during multitask training.
            task_ids: If given, add each corresponding model a specified task id. Otherwise, tasks get the ids 'Task_0', 'Task_1', ...
            loss_factors: If given, weight the losses of teh corresponding models during training.
            use_all_tasks: If True, each sentence will be trained on all tasks parallel, otherwise each epoch 1 task will be sampled to train the sentence on.
        """
        super().__init__()

        task_ids_internal: list[str] = task_ids if task_ids else [f"Task_{i}" for i in range(len(models))]

        self.tasks: dict[str, flair.nn.Classifier] = {}
        self.loss_factors: dict[str, float] = {}
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

    def _prepare_tensors(self, data_points: list[DT]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError("`_prepare_tensors` is not used for multitask learning")

    def forward_loss(self, sentences: Union[list[Sentence], Sentence]) -> tuple[torch.Tensor, int]:
        """Calls the respective forward loss of each model and sums them weighted by their loss factors.

        Args:
            sentences: batch of sentences

        Returns: loss and sample count
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
    def split_batch_to_task_ids(
        sentences: Union[list[Sentence], Sentence], all_tasks: bool = False
    ) -> dict[str, list[int]]:
        """Splits a batch of sentences to its respective model.

        If single sentence is assigned to several tasks (i.e. same corpus but different tasks), then the model
        assignment for this batch is randomly chosen.

        Args:
            sentences: batch of sentences
            all_tasks: use all tasks of each sentence. If deactivated, a random task will be sampled

        Returns: Key-value pairs as (task_id, list of sentences ids in batch)
        """
        batch_to_task_mapping: dict[str, list[int]] = {}
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

    def evaluate(  # type: ignore[override]
        self,
        data_points,
        gold_label_type: str,
        out_path: Optional[Union[str, Path]] = None,
        main_evaluation_metric: tuple[str, str] = ("micro avg", "f1-score"),
        evaluate_all: bool = True,
        **evalargs,
    ) -> Result:
        """Evaluates the model. Returns a Result object containing evaluation results and a loss value.

        Args:
            data_points: batch of sentences
            gold_label_type: if evaluate_all is False, specify the task to evaluate by the task_id.
            out_path: if not None, predictions will be created and saved at the respective file.
            main_evaluation_metric: Specify which metric to highlight as main_score
            evaluate_all: choose if all tasks should be evaluated, or a single one, depending on gold_label_type
            **evalargs: arguments propagated to :meth:`flair.nn.Model.evaluate`

        Returns: Tuple of Result object and loss value (float)
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
                main_evaluation_metric=main_evaluation_metric,
                **evalargs,
            )

        batch_split = self.split_batch_to_task_ids(data_points, all_tasks=True)

        loss = torch.tensor(0.0, device=flair.device)
        main_score = 0.0
        all_detailed_results = ""
        all_classification_report: dict[str, dict[str, Any]] = {}

        for task_id, split in batch_split.items():
            result = self.tasks[task_id].evaluate(
                data_points=[data_points[i] for i in split],
                gold_label_type=self.tasks[task_id].label_type,
                out_path=f"{out_path}_{task_id}.txt" if out_path is not None else None,
                main_evaluation_metric=main_evaluation_metric,
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

        scores = {"loss": loss.item() / len(batch_split)}

        return Result(
            main_score=main_score / len(batch_split),
            detailed_results=all_detailed_results,
            scores=scores,
            classification_report=all_classification_report,
        )

    def get_used_tokens(
        self, corpus: Corpus, context_length: int = 0, respect_document_boundaries: bool = True
    ) -> typing.Iterable[list[str]]:
        for model in self.tasks.values():
            yield from model.get_used_tokens(corpus, context_length, respect_document_boundaries)

    def _get_state_dict(self):
        """Returns the state dict of the multitask model which has multiple models underneath."""
        initial_model_state = super()._get_state_dict()
        initial_model_state["state_dict"] = {}  # the model state is stored per model already.
        model_state = {
            **initial_model_state,
            "model_states": {task: model._get_state_dict() for task, model in self.tasks.items()},
            "loss_factors": [self.loss_factors[task] for task in self.tasks],
            "use_all_tasks": self.use_all_tasks,
        }

        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        """Initializes the model based on given state dict."""
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

    @staticmethod
    def _fetch_model(model_name) -> str:
        model_map = {}
        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        # biomedical models
        model_map["hunflair"] = "/".join([hu_path, "bioner", "hunflair.pt"])
        model_map["hunflair-paper"] = "/".join([hu_path, "bioner", "hunflair-paper.pt"])

        # entity linker
        model_map["linker"] = "/".join([hu_path, "zelda", "v2", "zelda-v2.pt"])
        model_map["zelda"] = "/".join([hu_path, "zelda", "v2", "zelda-v2.pt"])

        cache_dir = Path("models")
        if model_name in model_map:
            if model_name in ["hunflair", "hunflair-paper", "bioner"]:
                log.warning(
                    "HunFlair (version 1) is deprecated. Consider using HunFlair2 for improved extraction performance: "
                    "Classifier.load('hunflair2')."
                    "See https://github.com/flairNLP/flair/blob/master/resources/docs/HUNFLAIR2.md for further "
                    "information."
                )

            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name

    @classmethod
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "MultitaskModel":
        from typing import cast

        return cast("MultitaskModel", super().load(model_path=model_path))
