import datetime
import inspect
import logging
import os
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
from flair.training_utils import (
    AnnealOnPlateau,
    identify_dynamic_embeddings,
    init_output_file,
    log_line,
    store_embeddings,
)

from .plugins import Pluggable, TrainingInterrupt, default_plugins

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
    }

    def __init__(self, model: flair.nn.Model, corpus: Corpus, plugins=default_plugins, **kw):
        """
        Initialize a model trainer
        :param model: The model that you want to train. The model should inherit from flair.nn.Model  # noqa: E501
        :param corpus: The dataset used to train the model, should be of type Corpus
        """
        super().__init__(plugins=plugins, **kw)
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus

        self.reset_training_attributes()

    def reset_training_attributes(self):
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            del self.optimizer

        self.optimizer = None
        self.base_path = None
        self.mini_batch_size = None

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
            return [batch[i : i + mini_batch_chunk_size] for i in range(0, len(batch), mini_batch_chunk_size)]
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

    def train(
        self,
        base_path: Union[Path, str],
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        eval_batch_size: int = None,
        mini_batch_chunk_size: Optional[int] = None,
        max_epochs: int = 100,
        train_with_dev: bool = False,
        train_with_test: bool = False,
        monitor_train: bool = False,
        monitor_test: bool = False,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        scheduler=AnnealOnPlateau,
        anneal_factor: float = 0.5,
        patience: int = 3,
        min_learning_rate: Union[float, List[float]] = 0.0001,
        initial_extra_patience: int = 0,
        optimizer: Union[torch.optim.Optimizer, Type[torch.optim.Optimizer]] = SGD,
        cycle_momentum: bool = False,
        warmup_fraction: float = 0.1,
        embeddings_storage_mode: str = "cpu",
        checkpoint: bool = False,
        save_final_model: bool = True,
        anneal_with_restarts: bool = False,
        anneal_with_prestarts: bool = False,
        anneal_against_dev_loss: bool = False,
        batch_growth_annealing: bool = False,
        shuffle: bool = True,
        param_selection_mode: bool = False,
        write_weights: bool = False,
        num_workers: Optional[int] = None,
        sampler=None,
        use_amp: bool = False,
        amp_opt_level: str = "O1",
        eval_on_train_fraction: Union[float, str] = 0.0,
        eval_on_train_shuffle: bool = False,
        save_model_each_k_epochs: int = 0,
        tensorboard_comment: str = "",
        use_swa: bool = False,
        use_final_model_for_eval: bool = False,
        gold_label_dictionary_for_eval: Optional[Dictionary] = None,
        exclude_labels: List[str] = [],
        create_file_logs: bool = True,
        create_loss_file: bool = True,
        epoch: int = 0,
        use_tensorboard: bool = False,
        tensorboard_log_dir=None,
        metrics_for_tensorboard=[],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        save_optimizer_state: bool = False,
        shuffle_first_epoch: bool = False,
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
        :param cycle_momentum: If scheduler is OneCycleLR, whether the scheduler should cycle also the momentum  # noqa: E501
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits  # noqa: E501
         until annealing the learning rate
        :param min_learning_rate: If the (in multi lr case: all) learning rate falls below this threshold, training terminates  # noqa: E501
        :param initial_extra_patience: Extra patience on top of the base patience value before the first learning rate annealment  # noqa: E501
        :param warmup_fraction: Fraction of warmup steps if the scheduler is LinearSchedulerWithWarmup  # noqa: E501
        :param train_with_dev:  If True, the data from dev split is added to the training data  # noqa: E501
        :param train_with_test: If True, the data from test split is added to the training data  # noqa: E501
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),  # noqa: E501
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param save_final_model: If True, final model is saved
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate  # noqa: E501
        :param anneal_with_prestarts: If True, the model preceding the last best model is restored when annealing the learning rate  # noqa: E501
        :param anneal_against_dev_loss: If True, the annealment is triggered when dev loss plateaus.  # noqa: E501
         If False (default), it is triggered when dev score plateaus.
        :param batch_growth_annealing: If True, mini_batch_size doubles every time learning_rate is annealed.  # noqa: E501
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing  # noqa: E501
        parameter selection.
        :param write_weights: If True, write weights to weights.txt on each batch logging event.
        :param num_workers: Number of workers in your data loader.
        :param sampler: You can pass a data sampler here for special sampling of data.  # noqa: E501
        :param eval_on_train_fraction: the fraction of train data to do the evaluation on,  # noqa: E501
        if 0. the evaluation is not performed on fraction of training data,
        if 'dev' the size is determined from dev set size
        :param eval_on_train_shuffle: if True the train data fraction is determined on the start of training  # noqa: E501
        and kept fixed during training, otherwise it's sampled at beginning of each epoch  # noqa: E501
        :param save_model_each_k_epochs: Each k epochs, a model state will be written out. If set to '5', a model will  # noqa: E501
        be saved each 5 epochs. Default is 0 which means no model saving.
        :param main_evaluation_metric: Type of metric to use for best model tracking and learning rate scheduling (if dev data is available, otherwise loss will be used), currently only applicable for text_classification_model  # noqa: E501
        :param tensorboard_comment: Comment to use for tensorboard logging
        :param create_file_logs: If True, the logs will also be stored in a file 'training.log' in the model folder  # noqa: E501
        :param create_loss_file: If True, the loss will be writen to a file 'loss.tsv' in the model folder  # noqa: E501
        :param optimizer: The optimizer to use (typically SGD or Adam)
        :param epoch: The starting epoch (normally 0 but could be higher if you continue training model)  # noqa: E501
        :param kwargs: Other arguments for the Optimizer
        :return:
        """

        # derive parameters the function was called with (or defaults)
        local_variables = locals()

        training_parameters = {parameter: local_variables[parameter] for parameter in signature(self.train).parameters}

        training_parameters.update(kwargs)

        # call first hook
        self.dispatch("before_training_setup", **training_parameters)

        assert self.corpus.train

        self.mini_batch_size = mini_batch_size

        if epoch >= max_epochs:
            log.warning(f"Starting at epoch {epoch + 1}/{max_epochs}. No training will be done.")

        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True, parents=True)

        if epoch == 0:
            self.check_for_and_delete_previous_best_models(base_path)

        # Prepare training data
        train_data = self.get_train_data(train_with_dev=train_with_dev, train_with_test=train_with_test)

        dataset_size = _len_dataset(train_data)

        parameters = {"dataset_size": dataset_size, **training_parameters}

        self.dispatch("after_data_setup", **parameters)

        # if optimizer class is passed, instantiate:
        if inspect.isclass(optimizer):
            kwargs["lr"] = learning_rate
            self.optimizer = optimizer(self.model.parameters(), **kwargs)
        else:
            self.optimizer = optimizer

        self.dispatch("after_optimizer_setup", **parameters)

        # load existing optimizer state dictionary if it exists
        if optimizer_state_dict:
            self.optimizer.load_state_dict(optimizer_state_dict)

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

        self.dispatch("after_training_setup", **parameters)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            self.dispatch("before_training_loop", **parameters)

            for epoch in range(epoch + 1, max_epochs + 1):
                self.dispatch("before_training_epoch", epoch=epoch)

                # if shuffle_first_epoch==False, the first epoch is not shuffled
                shuffle_data_this_epoch = shuffle
                if not shuffle_first_epoch and epoch == 1:
                    shuffle_data_this_epoch = False

                batch_loader = DataLoader(
                    train_data,
                    batch_size=self.mini_batch_size,
                    shuffle=shuffle_data_this_epoch,
                    num_workers=0 if num_workers is None else num_workers,
                    sampler=sampler,
                )

                self.model.train()

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

                    batch_steps = self.get_batch_steps(batch, mini_batch_chunk_size=mini_batch_chunk_size)

                    # forward and backward for batch
                    for batch_step in batch_steps:
                        batch_step_kw = {"batch_step": batch_step, **batch_kw}

                        self.dispatch("before_training_batch_step", **batch_step_kw)

                        # forward pass
                        loss, datapoint_count = self.model.forward_loss(batch_step)

                        self.dispatch(
                            "before_training_batch_backward",
                            loss=loss,
                            datapoint_count=datapoint_count,
                            **batch_step_kw,
                        )

                        self.backward(loss)

                        self.dispatch(
                            "after_training_batch_step", loss=loss, datapoint_count=datapoint_count, **batch_step_kw
                        )

                        # identify dynamic embeddings (always deleted) on first sentence

                        if dynamic_embeddings is None:
                            dynamic_embeddings = identify_dynamic_embeddings(batch)

                        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                        store_embeddings(batch, embeddings_storage_mode, dynamic_embeddings)

                    # do the optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()

                    self.dispatch("after_training_batch", **batch_kw)

                self.dispatch("after_training_epoch", epoch=epoch)

                self.model.eval()
                self.dispatch("evaluation", epoch=epoch)
                self.dispatch("after_evaluation", epoch=epoch)

            self.dispatch("after_training_loop")

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            self.dispatch("training_interrupt")
        except TrainingInterrupt as exc:
            log_line(log)
            log.info(str(exc))
            log_line(log)

            self.dispatch("training_interrupt")

        except Exception:
            self.dispatch("_training_exception")
            raise
        finally:
            self.dispatch("_training_finally")

        self.dispatch("after_training_loop")

        self.reset_training_attributes()

        return self.dispatch("collecting_train_return_values")

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

        return self.train(
            base_path=base_path,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
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
