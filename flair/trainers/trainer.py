import datetime
import inspect
import logging
import os
import random
import time
import warnings
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

import flair
import flair.nn
from flair.data import Corpus, Dictionary, _len_dataset
from flair.datasets import DataLoader
from flair.nn import Model
from flair.optim import ExpAnnealLR, LinearSchedulerWithWarmup
from flair.trainers.plugins import (
    MetricName,
    MetricRecord,
    Pluggable,
    TrainingInterrupt,
    CheckpointPlugin, LogFilePlugin, LossFilePlugin, WeightExtractorPlugin,
)
from flair.trainers.plugins.functional.anneal_on_plateau import AnnealingPlugin
from flair.trainers.plugins.functional.onecycle import OneCyclePlugin
from flair.training_utils import (
    AnnealOnPlateau,
    identify_dynamic_embeddings,
    init_output_file,
    log_line,
    store_embeddings,
)

log = logging.getLogger("flair")


class ModelTrainer(Pluggable):
    valid_events = {
        "before_training_setup",
        "after_data_setup",
        "after_optimizer_setup",
        "after_training_setup",
        "before_training_loop",
        "before_training_epoch",
        "before_training_batch",
        "before_training_batch_step",
        "before_training_batch_backward",
        "after_training_batch_step",
        "after_training_batch",
        "after_training_epoch",
        "evaluation",
        "after_evaluation",
        "after_training_loop",
        "training_interrupt",
        "_training_finally",
        "_training_exception",
        "collecting_train_return_values",
        "after_training",
        "metric_recorded",
    }

    def __init__(self, model: flair.nn.Model, corpus: Corpus):
        """
        Initialize a model trainer
        :param model: The model that you want to train. The model should inherit from flair.nn.Model  # noqa: E501
        :param corpus: The dataset used to train the model, should be of type Corpus
        """
        super().__init__()
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus

        self.reset_training_attributes()
        self.return_values: dict = {}

    def reset_training_attributes(self):
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            del self.optimizer

        self.optimizer = None
        self.mini_batch_size = None
        self.return_values: dict = {}

    @staticmethod
    def check_for_and_delete_previous_best_models(base_path):
        all_best_model_names = [filename for filename in os.listdir(base_path) if filename.startswith("best-model")]
        if len(all_best_model_names) != 0:
            warnings.warn(
                "There should be no best model saved at epoch 1 except there "
                "is a model from previous trainings"
                " in your training folder. All previous best models will be deleted."
            )
        for single_model in all_best_model_names:
            previous_best_path = os.path.join(base_path, single_model)
            if os.path.exists(previous_best_path):
                os.remove(previous_best_path)

    @staticmethod
    def get_batch_steps(batch, mini_batch_chunk_size):
        # if necessary, make batch_steps
        if mini_batch_chunk_size is not None and len(batch) > mini_batch_chunk_size:
            # break up the batch into slices of size
            # mini_batch_chunk_size
            return [batch[i: i + mini_batch_chunk_size] for i in range(0, len(batch), mini_batch_chunk_size)]
        else:
            return [batch]

    def get_train_data(self, train_with_dev, train_with_test):
        # if training also uses dev/train data, include in training set
        train_data = self.corpus.train

        if train_with_dev or train_with_test:
            parts = [self.corpus.train]
            if train_with_dev and self.corpus.dev:
                parts.append(self.corpus.dev)
            if train_with_test and self.corpus.test:
                parts.append(self.corpus.test)

            train_data = ConcatDataset(parts)

        return train_data

    def backward(self, loss):
        """Calls backward on the loss.

        This allows plugins to overwrite the backward call.
        """
        loss.backward()

    def train(self,
              base_path,
              scheduler=AnnealOnPlateau,
              anneal_factor: float = 0.5,
              patience: int = 3,
              min_learning_rate: Union[float, List[float]] = 0.0001,
              initial_extra_patience: int = 0,
              anneal_with_restarts: bool = False,
              anneal_against_dev_loss: bool = False,
              **kwargs):

        # activate annealing plugin
        AnnealingPlugin(base_path=base_path,
                        anneal_factor=anneal_factor,
                        patience=patience,
                        min_learning_rate=min_learning_rate,
                        initial_extra_patience=initial_extra_patience,
                        anneal_with_restarts=anneal_with_restarts,
                        anneal_against_dev_loss=anneal_against_dev_loss,
                        ).attach_to(self)

        return self.train_custom(base_path,
                                 **kwargs)

    def train_custom(
            self,
            base_path: Union[Path, str],
            learning_rate: float = 0.1,
            mini_batch_size: int = 32,
            eval_batch_size: int = 32,
            mini_batch_chunk_size: Optional[int] = None,
            max_epochs: int = 100,
            train_with_dev: bool = False,
            train_with_test: bool = False,
            monitor_test: bool = False,
            monitor_train_sample: float = 0.0,
            main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
            scheduler=AnnealOnPlateau,
            optimizer: Union[torch.optim.Optimizer, Type[torch.optim.Optimizer]] = SGD,
            embeddings_storage_mode: str = "cpu",
            checkpoint: bool = False,
            save_final_model: bool = True,
            use_final_model_for_eval: bool = False,
            shuffle: bool = True,
            param_selection_mode: bool = False,
            write_weights: bool = False,
            num_workers: Optional[int] = None,
            sampler=None,
            amp_opt_level: str = "O1",
            save_model_each_k_epochs: int = 0,
            gold_label_dictionary_for_eval: Optional[Dictionary] = None,
            exclude_labels: List[str] = [],
            create_file_logs: bool = True,
            create_loss_file: bool = True,
            epoch: int = 0,
            save_optimizer_state: bool = False,
            shuffle_first_epoch: bool = True,
            **kwargs,
    ) -> dict:
        """
        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved  # noqa: E501
        :param learning_rate: Initial learning rate (or max, if scheduler is OneCycleLR)  # noqa: E501
        :param mini_batch_size: Size of mini-batches during training  # noqa: E501
        :param eval_batch_size: Size of mini-batches during evaluation. Defaults to mini_batch_size.  # noqa: E501
        :param mini_batch_chunk_size: If mini-batches are larger than this number, they get broken down into chunks of this size for processing purposes  # noqa: E501
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.  # noqa: E501
        :param scheduler: The learning rate scheduler to use
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch  # noqa: E501
        :param train_with_dev:  If True, the data from dev split is added to the training data  # noqa: E501
        :param train_with_test: If True, the data from test split is added to the training data  # noqa: E501
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),  # noqa: E501
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param save_final_model: If True, final model is saved
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing  # noqa: E501
        parameter selection.
        :param write_weights: If True, write weights to weights.txt on each batch logging event.
        :param num_workers: Number of workers in your data loader.
        :param sampler: You can pass a data sampler here for special sampling of data.  # noqa: E501
        :param eval_on_train_fraction: the fraction of train data to do the evaluation on,  # noqa: E501
        if 0. the evaluation is not performed on fraction of training data,
        if 'dev' the size is determined from dev set size
        :param save_model_each_k_epochs: Each k epochs, a model state will be written out. If set to '5', a model will  # noqa: E501
        be saved each 5 epochs. Default is 0 which means no model saving.
        :param main_evaluation_metric: Type of metric to use for best model tracking and learning rate scheduling (if dev data is available, otherwise loss will be used), currently only applicable for text_classification_model  # noqa: E501
        :param optimizer: The optimizer to use (typically SGD or Adam)
        :param epoch: The starting epoch (normally 0 but could be higher if you continue training model)  # noqa: E501
        :param kwargs: Other arguments for the Optimizer
        :return:
        """

        # Create output folder
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True, parents=True)

        # === START BLOCK: ACTIVATE PLUGINS === #
        # We first activate all optional plugins. These take care of optional functionality such as various
        # logging techniques and checkpointing

        # log file plugin
        if create_file_logs:
            LogFilePlugin(base_path=base_path).attach_to(self)

        # loss file plugin
        if create_loss_file:
            LossFilePlugin(base_path=base_path, epoch=epoch).attach_to(self)

        # plugin for writing weights
        if write_weights:
            WeightExtractorPlugin(base_path=base_path).attach_to(self)

        # plugin for checkpointing
        if save_model_each_k_epochs > 0:
            CheckpointPlugin(save_model_each_k_epochs=save_model_each_k_epochs,
                             save_optimizer_state=save_optimizer_state,
                             base_path=base_path,
                             ).attach_to(self)
        # === END BLOCK: ACTIVATE PLUGINS === #

        # derive parameters the function was called with (or defaults)
        local_variables = locals()
        training_parameters = {parameter: local_variables[parameter] for parameter in
                               signature(self.train_custom).parameters}
        training_parameters.update(kwargs)

        # initialize model card with these parameters
        self.model.model_card = self._initialize_model_card(**training_parameters)

        # call first hook
        # -- AmpPlugin -> set opt level
        # -- SWAPlugin -> initializes SWA and stores learning rate
        self.dispatch("before_training_setup", **training_parameters)

        # Prepare training data and get dataset size
        train_data = self.get_train_data(train_with_dev=train_with_dev, train_with_test=train_with_test)
        dataset_size = _len_dataset(train_data)
        parameters = {"dataset_size": dataset_size, **training_parameters}

        # determine what splits (train, dev, test) to evaluate and log
        evaluate_dev_split = False if train_with_dev or not self.corpus.dev else True
        evaluate_test_split = True if (not param_selection_mode and self.corpus.test and monitor_test) else False
        evaluate_train_sample = True if monitor_train_sample > 0.0 else False

        if evaluate_train_sample:
            train_part_size = 0
            if evaluate_train_sample is float:
                train_part_size = int(_len_dataset(self.corpus.train) * evaluate_train_sample)
            if evaluate_train_sample is int:
                train_part_size = evaluate_train_sample

            assert train_part_size > 0

            # get a random sample of training sentences
            train_part_indices = list(range(_len_dataset(self.corpus.train)))
            random.shuffle(train_part_indices)
            train_part_indices = train_part_indices[:train_part_size]
            train_part = torch.utils.data.dataset.Subset(self.corpus.train, train_part_indices)

        # if optimizer class is passed, instantiate:
        if inspect.isclass(optimizer):
            kwargs["lr"] = learning_rate
            self.optimizer = optimizer = optimizer(self.model.parameters(), **kwargs)
        else:
            self.optimizer = optimizer

        # -- AmpPlugin -> wraps with AMP
        # -- AnnealingPlugin -> initialize schedulers (requires instantiated optimizer)
        # -- SWAPlugin -> wraps the optimizer with SWA
        self.dispatch("after_optimizer_setup", **parameters)

        # initialize sampler if provided
        if sampler is not None:
            # init with default values if only class is provided
            if inspect.isclass(sampler):
                sampler = sampler()
            # set dataset to sample from
            sampler.set_dataset(train_data)
            shuffle = False

        # this field stores the names of all dynamic embeddings in the model (determined after first forward pass)
        dynamic_embeddings = None

        # determine whether to save best model, minimize training loss if training with dev data, else maximize score
        save_best_model = not train_with_dev and not param_selection_mode and not use_final_model_for_eval
        best_model_value = None

        # Sanity checks
        assert len(train_data) > 0
        if epoch >= max_epochs:
            log.warning(f"Starting at epoch {epoch + 1}/{max_epochs}. No training will be done.")
        if epoch == 0:
            self.check_for_and_delete_previous_best_models(base_path)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            lr_info = ",".join([f"{group['lr']:.6f}" for group in self.optimizer.param_groups])

            log_line(log)
            log.info(f'Model: "{self.model}"')
            log_line(log)
            log.info(f'Corpus: "{self.corpus}"')
            log_line(log)
            log.info("Parameters:")
            log.info(f' - learning_rate: "{lr_info}"')
            log.info(f' - mini_batch_size: "{mini_batch_size}"')
            log.info(f' - max_epochs: "{max_epochs}"')
            log.info(f' - shuffle: "{shuffle}"')
            log.info(f' - train_with_dev: "{train_with_dev}"')
            log_line(log)
            log.info(f'Model training base path: "{base_path}"')
            log_line(log)
            log.info(f"Device: {flair.device}")
            log_line(log)
            log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

            total_train_samples = 0

            for epoch in range(epoch + 1, max_epochs + 1):
                # - SchedulerPlugin -> load state for anneal_with_restarts, batch_growth_annealing, logic for early stopping
                # - LossFilePlugin -> get the current epoch for loss file logging
                self.dispatch("before_training_epoch", epoch=epoch)
                self.model.model_card["training_parameters"]["epoch"] = epoch

                current_learning_rate = [group["lr"] for group in optimizer.param_groups]
                momentum = [group["momentum"] if "momentum" in group else 0 for group in optimizer.param_groups]

                lr_key = ("optimizer", "learning_rate")
                momentum_key = ("optimizer", "momentum")

                lr_info = " - lr: " + ",".join([f"{m:.4f}" for m in current_learning_rate])
                momentum_info = " - momentum: " + ",".join([f"{m:.4f}" for m in momentum])

                if len(current_learning_rate) == 1:
                    self.dispatch("metric_recorded", MetricRecord.scalar(lr_key, current_learning_rate[0], epoch))
                    self.dispatch("metric_recorded", MetricRecord.scalar(momentum_key, momentum[0], epoch))
                else:
                    self.dispatch("metric_recorded", MetricRecord.scalar_list(lr_key, current_learning_rate, epoch))
                    self.dispatch("metric_recorded", MetricRecord.scalar_list(momentum_key, momentum, epoch))

                # if shuffle_first_epoch==False, the first epoch is not shuffled
                shuffle_data_this_epoch = shuffle
                if not shuffle_first_epoch and epoch == 1:
                    shuffle_data_this_epoch = False

                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle_data_this_epoch,
                    num_workers=0 if num_workers is None else num_workers,
                    sampler=sampler,
                )

                self.model.train()

                epoch_train_loss: float = 0.0
                epoch_train_samples: int = 0

                epoch_start_time = time.time()

                # log infos on training progress every `log_modulo` batches
                log_modulo = max(1, int(len(batch_loader) / 10))

                # process mini-batches
                for batch_no, batch in enumerate(batch_loader):
                    batch_kw = {
                        "batch_no": batch_no,
                        "batch": batch,
                        "total_number_of_batches": len(batch_loader),
                        "epoch": epoch,
                    }

                    self.dispatch("before_training_batch", **batch_kw)

                    # zero the gradients on the model and optimizer
                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    batch_train_loss = 0.0
                    batch_train_samples = 0

                    batch_steps = self.get_batch_steps(batch, mini_batch_chunk_size=mini_batch_chunk_size)

                    # forward and backward for batch
                    for batch_step in batch_steps:
                        batch_step_kw = {"batch_step": batch_step, **batch_kw}

                        # forward pass
                        loss, datapoint_count = self.model.forward_loss(batch_step)

                        batch_train_samples += datapoint_count
                        batch_train_loss += loss.item()

                        self.backward(loss)

                        # identify dynamic embeddings (always deleted) on first sentence
                        if dynamic_embeddings is None:
                            dynamic_embeddings = identify_dynamic_embeddings(batch)

                        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                        store_embeddings(batch_step, embeddings_storage_mode, dynamic_embeddings)

                    # do the optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()

                    if batch_train_samples > 0:
                        train_loss = batch_train_loss / batch_train_samples
                        self.dispatch(
                            "metric_recorded",
                            MetricRecord.scalar(("train", "batch_loss"), train_loss, total_train_samples),
                        )

                        epoch_train_loss += batch_train_loss
                        epoch_train_samples += batch_train_samples

                    if (batch_no + 1) % log_modulo == 0:
                        intermittent_loss = (
                            epoch_train_loss / epoch_train_samples
                            if epoch_train_samples > 0
                            else epoch_train_samples / (batch_no + 1)
                        )

                        current_time = time.time()

                        log.info(
                            f"epoch {epoch}"
                            f" - iter {batch_no + 1}/{len(batch_loader)}"
                            f" - loss {intermittent_loss:.8f}"
                            f" - time (sec): {(current_time - epoch_start_time):.2f}"
                            f" - samples/sec: {epoch_train_samples / (current_time - epoch_start_time):.2f}"
                            f" - {lr_info}{momentum_info}"
                        )

                    # - SchedulerPlugin -> do the scheduler step if one-cycle or linear decay
                    # - WeightExtractorPlugin -> extracts weights
                    self.dispatch("after_training_batch", **batch_kw)

                if epoch_train_samples > 0:
                    train_loss = epoch_train_loss / epoch_train_samples
                    self.dispatch("metric_recorded", MetricRecord.scalar(("train", "loss"), train_loss, epoch))

                    total_train_samples += epoch_train_samples

                log_line(log)
                log.info(f"EPOCH {epoch} done: loss {epoch_train_loss:.4f}{lr_info}")

                # - CheckpointPlugin -> executes save_model_each_k_epochs
                # - SchedulerPlugin -> log bad epochs
                self.dispatch("after_training_epoch", epoch=epoch)

                self.model.eval()

                eval_kw = {
                    "mini_batch_size": eval_batch_size,
                    "num_workers": num_workers,
                    "exclude_labels": exclude_labels,
                    "main_evaluation_metric": main_evaluation_metric,
                    "gold_label_dictionary": gold_label_dictionary_for_eval,
                    "embedding_storage_mode": embeddings_storage_mode,
                    "gold_label_type": self.model.label_type,
                    "gold_label_dictionary_for_eval": gold_label_dictionary_for_eval,
                }

                # evaluate on train / dev / test split depending on training settings
                if evaluate_train_sample:

                    # evaluate on those
                    train_part_eval_result = self.model.evaluate(train_part, **eval_kw)

                    log.info(
                        f"TRAIN_SPLIT : loss {train_part_eval_result.loss}"
                        f' - {eval_kw["main_evaluation_metric"][1]}'
                        f' ({eval_kw["main_evaluation_metric"][0]})'
                        f" {round(train_part_eval_result.main_score, 4)}"
                    )

                    self.publish_eval_result(train_part_eval_result, "train_part", global_step=epoch)

                if evaluate_dev_split:
                    # evaluate on dev data
                    assert self.corpus.dev
                    dev_eval_result = self.model.evaluate(
                        self.corpus.dev,
                        out_path=base_path / "dev.tsv",
                        **eval_kw
                    )

                    # log results
                    log.info(
                        f"DEV : loss {dev_eval_result.loss}"
                        f' - {eval_kw["main_evaluation_metric"][1]}'
                        f' ({eval_kw["main_evaluation_metric"][0]})'
                        f"  {round(dev_eval_result.main_score, 4)}"
                    )

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.dev, embeddings_storage_mode)

                    self.publish_eval_result(dev_eval_result, "dev", global_step=epoch)

                if evaluate_test_split:
                    assert self.corpus.test
                    test_eval_result = self.model.evaluate(
                        self.corpus.test,
                        out_path=base_path / "test.tsv",
                        **eval_kw,
                    )

                    log.info(
                        f"TEST : loss {test_eval_result.loss} -"
                        f' {eval_kw["main_evaluation_metric"][1]}'
                        f' ({eval_kw["main_evaluation_metric"][0]}) '
                        f" {round(test_eval_result.main_score, 4)}"
                    )

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.test, embeddings_storage_mode)

                    self.publish_eval_result(test_eval_result, "test", global_step=epoch)

                # Determine if this is the best model or if we need to anneal
                current_epoch_has_best_model_so_far = False
                validation_scores: tuple

                if evaluate_dev_split:
                    validation_scores = dev_eval_result.main_score, dev_eval_result.loss

                    if best_model_value is None or dev_eval_result.main_score > best_model_value:
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = dev_eval_result.main_score

                else:
                    # there is no corpus.dev or it is used as part of the training data
                    validation_scores = (train_loss,)

                    if best_model_value is None or epoch_train_loss < best_model_value:
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = train_loss

                # - LossFilePlugin -> somehow prints all relevant metrics (TODO: I don't really understand how)
                # - AnnealPlugin -> scheduler step
                self.dispatch(
                    "after_evaluation",
                    epoch=epoch,
                    current_model_is_best=current_epoch_has_best_model_so_far,
                    validation_scores=validation_scores,
                )

                if save_best_model and current_epoch_has_best_model_so_far:
                    log.info("saving best model")
                    self.model.save(base_path / "best-model.pt", checkpoint=save_optimizer_state)

            # - SWAPlugin -> restores SGD weights from SWA
            self.dispatch("after_training_loop")

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt", checkpoint=save_optimizer_state)

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            self.dispatch("training_interrupt")  # TODO: no plugin calls this event

            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt", checkpoint=save_optimizer_state)
                log.info("Done.")

        except TrainingInterrupt as exc:
            log_line(log)
            log.info(str(exc))
            log_line(log)
            self.dispatch("training_interrupt")  # TODO: no plugin calls this event

            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt", checkpoint=save_optimizer_state)
                log.info("Done.")

        except Exception:
            self.dispatch("_training_exception")  # TODO: no plugin calls this event
            raise
        finally:
            # TensorboardLogger -> closes writer
            self.dispatch("_training_finally")

        # test best model if test data is present
        if self.corpus.test and not train_with_test:
            self.return_values["test_score"] = self.final_test(
                base_path=base_path,
                eval_mini_batch_size=eval_batch_size,
                num_workers=num_workers,
                main_evaluation_metric=main_evaluation_metric,
                gold_label_dictionary_for_eval=gold_label_dictionary_for_eval,
                exclude_labels=exclude_labels,
            )
        else:
            self.return_values["test_score"] = 0
            log.info("Test data not provided setting final score to 0")

        # MetricHistoryPlugin -> stores the loss history in return_values
        self.dispatch("after_training")

        # Store return values, as they will be erased by reset_training_attributes
        return_values = self.return_values

        self.reset_training_attributes()

        return return_values

    def flat_dict_items(self, d, composite_key=()):
        for key, value in d.items():
            if isinstance(key, str):
                key = composite_key + (key,)
            else:
                key = composite_key + tuple(key)

            if isinstance(value, dict):
                yield from self.flat_dict_items(value, composite_key=key)
            else:
                yield key, value

    def publish_eval_result(self, result, prefix=(), **kw):
        for key, value in self.flat_dict_items(result.scores, composite_key=MetricName(prefix)):
            try:
                self.dispatch("metric_recorded", MetricRecord.scalar(name=key, value=float(value), **kw))
            except TypeError:
                if isinstance(value, list):
                    self.dispatch("metric_recorded", MetricRecord.scalar_list(name=key, value=value, **kw))
                elif isinstance(value, torch.Tensor):
                    self.dispatch("metric_recorded", MetricRecord.histogram(name=key, value=value, **kw))
                else:
                    value = str(value)
                    self.dispatch("metric_recorded", MetricRecord.string(name=key, value=value, **kw))

        self.dispatch(
            "metric_recorded", MetricRecord.string(name=MetricName(prefix) + "score", value=result.main_score, **kw)
        )
        self.dispatch(
            "metric_recorded",
            MetricRecord.string(name=MetricName(prefix) + "detailed_result", value=result.detailed_results, **kw),
        )

    def resume(
            self,
            model: Model,
            additional_epochs: Optional[int] = None,
            **trainer_args,
    ):
        assert model.model_card is not None
        self.model = model
        # recover all arguments that were used to train this model
        args_used_to_train_model = model.model_card["training_parameters"]

        # you can overwrite params with your own
        for param in trainer_args:
            args_used_to_train_model[param] = trainer_args[param]
            if param == "optimizer" and "optimizer_state_dict" in args_used_to_train_model:
                del args_used_to_train_model["optimizer_state_dict"]
            if param == "scheduler" and "scheduler_state_dict" in args_used_to_train_model:
                del args_used_to_train_model["scheduler_state_dict"]

        # surface nested arguments
        kwargs = args_used_to_train_model["kwargs"]
        del args_used_to_train_model["kwargs"]

        if additional_epochs is not None:
            args_used_to_train_model["max_epochs"] = (
                    args_used_to_train_model.get("epoch", kwargs.get("epoch", 0)) + additional_epochs
            )

        # resume training with these parameters
        self.train(**args_used_to_train_model, **kwargs)

    def fine_tune(
            self,
            base_path: Union[Path, str],
            learning_rate: float = 5e-5,
            max_epochs: int = 10,
            optimizer=torch.optim.AdamW,
            scheduler=LinearSchedulerWithWarmup,
            warmup_fraction: float = 0.1,
            mini_batch_size: int = 4,
            embeddings_storage_mode: str = "none",
            use_final_model_for_eval: bool = True,
            decoder_lr_factor: float = 1.0,
            **trainer_args,
    ):

        # annealing logic
        OneCyclePlugin(warmup_fraction=warmup_fraction).attach_to(self)

        # If set, add a factor to the learning rate of all parameters with 'embeddings' not in name
        if decoder_lr_factor != 1.0:
            optimizer = optimizer(
                [
                    {
                        "params": [param for name, param in self.model.named_parameters() if "embeddings" not in name],
                        "lr": learning_rate * decoder_lr_factor,
                    },
                    {
                        "params": [param for name, param in self.model.named_parameters() if "embeddings" in name],
                        "lr": learning_rate,
                    },
                ]
            )
            log.info(
                f"Modifying learning rate to {learning_rate * decoder_lr_factor} for the following "
                f"parameters: {[name for name, param in self.model.named_parameters() if 'embeddings' not in name]}"
            )

        return self.train_custom(
            base_path=base_path,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            optimizer=optimizer,
            warmup_fraction=warmup_fraction,
            mini_batch_size=mini_batch_size,
            embeddings_storage_mode=embeddings_storage_mode,
            use_final_model_for_eval=use_final_model_for_eval,
            **trainer_args,
        )

    def final_test(
            self,
            base_path: Union[Path, str],
            eval_mini_batch_size: int,
            main_evaluation_metric: Tuple[str, str],
            num_workers: Optional[int] = 8,
            gold_label_dictionary_for_eval: Optional[Dictionary] = None,
            exclude_labels: List[str] = [],
    ):
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True, parents=True)

        log_line(log)

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model.load_state_dict(self.model.load(base_path / "best-model.pt").state_dict())
        else:
            log.info("Testing using last state of model ...")

        assert self.corpus.test
        test_results = self.model.evaluate(
            self.corpus.test,
            gold_label_type=self.model.label_type,
            mini_batch_size=eval_mini_batch_size,
            num_workers=num_workers,
            out_path=base_path / "test.tsv",
            embedding_storage_mode="none",
            main_evaluation_metric=main_evaluation_metric,
            gold_label_dictionary=gold_label_dictionary_for_eval,
            exclude_labels=exclude_labels,
            return_loss=False,
        )

        log.info(test_results.detailed_results)
        log_line(log)

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score

    def find_learning_rate(
            self,
            base_path: Union[Path, str],
            optimizer,
            file_name: str = "learning_rate.tsv",
            start_learning_rate: float = 1e-7,
            end_learning_rate: float = 10,
            iterations: int = 100,
            mini_batch_size: int = 32,
            stop_early: bool = True,
            smoothing_factor: float = 0.98,
            **kwargs,
    ) -> Path:
        best_loss = None
        moving_avg_loss = 0.0

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)
        learning_rate_tsv = init_output_file(base_path, file_name)

        with open(learning_rate_tsv, "a") as f:
            f.write("ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n")

        optimizer = optimizer(self.model.parameters(), lr=start_learning_rate, **kwargs)

        train_data = self.corpus.train

        scheduler = ExpAnnealLR(optimizer, end_learning_rate, iterations)

        model_state = self.model.state_dict()
        self.model.train()

        step = 0
        while step < iterations:
            batch_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
            for batch in batch_loader:
                step += 1

                # forward pass
                loss, datapoint_count = self.model.forward_loss(batch)

                # update optimizer and scheduler
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                learning_rate = scheduler.get_lr()[0]

                loss_item = loss.item()
                if step == 1:
                    best_loss = loss_item
                else:
                    if smoothing_factor > 0:
                        moving_avg_loss = smoothing_factor * moving_avg_loss + (1 - smoothing_factor) * loss_item
                        loss_item = moving_avg_loss / (1 - smoothing_factor ** (step + 1))
                    if loss_item < best_loss:  # type: ignore
                        best_loss = loss  # type: ignore

                if step > iterations:
                    break

                if stop_early and (loss_item > 4 * best_loss or torch.isnan(loss)):  # type: ignore
                    log_line(log)
                    log.info("loss diverged - stopping early!")
                    step = iterations
                    break

                with open(str(learning_rate_tsv), "a") as f:
                    f.write(f"{step}\t{datetime.datetime.now():%H:%M:%S}\t{learning_rate}\t{loss_item}\n")

            self.model.load_state_dict(model_state)
            self.model.to(flair.device)

        log_line(log)
        log.info(f"learning rate finder finished - plot {learning_rate_tsv}")
        log_line(log)

        return Path(learning_rate_tsv)


    def _initialize_model_card(self, **training_parameters):
        """
        initializes model card with library versions and parameters
        :param training_parameters:
        :return:
        """
        # create a model card for this model with Flair and PyTorch version
        model_card = {
            "flair_version": flair.__version__,
            "pytorch_version": torch.__version__,
        }

        # record Transformers version if library is loaded
        try:
            import transformers

            model_card["transformers_version"] = transformers.__version__
        except ImportError:
            pass

        # remember all parameters used in train() call
        model_card["training_parameters"] = {
            k: str(v) if isinstance(v, Path) else v for k, v in training_parameters.items()
        }

        return model_card

        # remember all activated plugins
        # model_card["plugins"] = [plugin.__class__ for plugin in trainer.plugins]

        # add model card to model
        # trainer.model.model_card = model_card
